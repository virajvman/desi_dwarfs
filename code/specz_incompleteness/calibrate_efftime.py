'''
Stage 3c (EARLY MILESTONE): Calibrate simExposure exptime against real depth,
and verify NNMF-reconstruction smoothness does not bias Dchi2.

Two known issues motivate this gate (slides + Julien Guy discussion):
  1. simExposure's nominal dark-sky `exptime` is not 1:1 with real
     EFFTIME_SPEC -- old 180s BGS sims systematically underpredicted Dchi2.
  2. A W_gen reconstruction is smoother than a real spectrum; if that
     changes redrock behavior, generation is biased at the source.

Procedure:
  prepare:  pick CALIB_N_SPECTRA real high-SNR reference spectra, and mock
            re-observe BOTH the real spectrum AND its W_gen reconstruction
            at each exptime scale factor in CALIB_SCALE_GRID (exptime =
            per-object EFFTIME * scale). Writes coadd files + redrock list.
  <run redrock via job_scripts/specz_incompleteness/run_redrock.sbatch on the
   list written by prepare>
  analyze:  compare re-observed Dchi2 against the original catalog DELTACHI2
            per scale factor; report the scale minimizing the median offset
            (-> set config.EXPTIME_CALIBRATION_FACTOR); compare real-vs-
            reconstruction Dchi2 at matched conditions (smoothness check).
            If no single scale factor reconciles the loci, that is the
            trigger for the desisim backend.

Usage:  python calibrate_efftime.py prepare
        python calibrate_efftime.py analyze
'''

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import h5py
import numpy as np
from astropy.table import Table

import config
import specz_utils


def calib_coadd_path(kind, scale):
    return f"{config.CALIB_DIR}/calib_{kind}_scale{scale:.2f}.fits"


def calib_sample_path():
    return f"{config.CALIB_DIR}/calib_sample.fits"


def prepare():
    specz_utils.check_dirs()
    rng = np.random.default_rng(config.NNMF_SEED)

    ref = Table.read(config.REFERENCE_CATALOG)
    with h5py.File(config.REFERENCE_SPECTRA_H5, "r") as f:
        tgids = f["TARGETID"][:]
        wave = f["WAVE"][:]
        flux = f["FLUX"][:]
        zreds = f["Z"][:]

    assert np.array_equal(tgids, np.asarray(ref["TARGETID"]))

    # very-high-SNR subset with per-object efftime available
    efftime = np.full(len(ref), np.nan)
    if "TSNR2_BGS" in ref.colnames:
        bgs_like = np.asarray(ref["REF_PROGRAM"]) == "BGS_BRIGHT"
        efftime[bgs_like] = config.TSNR2_BGS_TO_EFFTIME * np.asarray(ref["TSNR2_BGS"])[bgs_like]
    if "TSNR2_LRG" in ref.colnames:
        lowz_like = np.asarray(ref["REF_PROGRAM"]) == "LOWZ_DARK"
        efftime[lowz_like] = config.TSNR2_LRG_TO_EFFTIME * np.asarray(ref["TSNR2_LRG"])[lowz_like]

    good = (np.asarray(ref["DELTACHI2"]) > 1000) & np.isfinite(efftime) & (efftime > 0)
    idx = rng.choice(np.flatnonzero(good),
                     size=min(config.CALIB_N_SPECTRA, int(good.sum())),
                     replace=False)
    print(f"Calibration sample: {len(idx)} spectra")

    sample = ref[idx]
    sample["EFFTIME"] = efftime[idx]
    sample.write(calib_sample_path(), overwrite=True)

    # W_gen reconstructions of the same objects (features what generation makes)
    W_gen = np.load(config.GEN_BASIS_NPY)
    wave_rest = specz_utils.get_rest_wave()
    wave_obs = specz_utils.get_obs_wave()

    flux_real = np.empty((len(idx), len(wave_obs)))
    flux_recon = np.empty((len(idx), len(wave_obs)))
    with h5py.File(config.REFERENCE_SPECTRA_H5, "r") as f:
        ivar_all = f["FLUX_IVAR"]
        for row, i in enumerate(idx):
            fl, iv, z = flux[i], ivar_all[i], zreds[i]
            flux_real[row] = np.interp(wave_obs, wave, fl)
            _, _, _, coeffs = specz_utils.fit_feature_basis(
                W_gen, wave_rest, wave, fl, iv, z)
            recon_rest = W_gen @ coeffs
            flux_recon[row] = np.interp(wave_obs, wave_rest * (1 + z),
                                        recon_rest / (1 + z))

    from mock_observe import FeasiBGSBackend
    backend = FeasiBGSBackend()

    file_list = []
    for scale in config.CALIB_SCALE_GRID:
        # group by mean efftime: simExposure takes a scalar exptime, so use
        # the sample-median efftime per program times the scale factor
        eff = float(np.median(sample["EFFTIME"]))
        for kind, arr in (("real", flux_real), ("recon", flux_recon)):
            out = calib_coadd_path(kind, scale)
            backend.observe(wave_obs, arr, eff * scale, out)
            file_list.append(out)
            print(f"wrote {out}")

    list_path = f"{config.CALIB_DIR}/redrock_input_list_calib.ascii"
    with open(list_path, "w") as f:
        f.write("\n".join(file_list) + "\n")
    print(f"Redrock list -> {list_path}\nNow run run_redrock.sbatch on it.")


def analyze():
    sample = Table.read(calib_sample_path())
    dchi2_orig = np.asarray(sample["DELTACHI2"])
    z_orig = np.asarray(sample["Z"])

    print(f"{'scale':>6} {'kind':>6} {'med dlogDchi2':>14} {'catastrophic':>13}")
    results = {}
    for scale in config.CALIB_SCALE_GRID:
        for kind in ("real", "recon"):
            rr_file = f"{config.REDROCK_DIR}/{os.path.basename(calib_coadd_path(kind, scale))}"
            if not os.path.exists(rr_file):
                print(f"{scale:>6.2f} {kind:>6}   [missing {rr_file}]")
                continue
            try:
                rr = Table.read(rr_file, hdu="REDSHIFTS")
            except KeyError:
                rr = Table.read(rr_file)
            dlog = np.log10(np.asarray(rr["DELTACHI2"]).clip(1e-3)) - np.log10(dchi2_orig)
            catas = specz_utils.catastrophic_z(z_orig, np.asarray(rr["Z"]))
            results[(scale, kind)] = np.median(dlog)
            print(f"{scale:>6.2f} {kind:>6} {np.median(dlog):>14.3f} "
                  f"{np.mean(catas):>13.4f}")

    real_offsets = {s: v for (s, k), v in results.items() if k == "real"}
    if real_offsets:
        best = min(real_offsets, key=lambda s: abs(real_offsets[s]))
        print(f"\nRecommended EXPTIME_CALIBRATION_FACTOR = {best:.2f} "
              f"(median dlogDchi2 = {real_offsets[best]:+.3f})")
        print("Set this in config.py before the production mock run.")
        recon_off = results.get((best, "recon"))
        if recon_off is not None:
            delta = recon_off - real_offsets[best]
            verdict = "OK" if abs(delta) < 0.1 else "INVESTIGATE: reconstruction biases Dchi2"
            print(f"Smoothness check at that scale: recon - real median "
                  f"dlogDchi2 = {delta:+.3f} ({verdict})")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=["prepare", "analyze"])
    args = p.parse_args()
    prepare() if args.mode == "prepare" else analyze()

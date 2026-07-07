'''
Stage 6: Apply the trained models to the real dwarf catalog -> W_spec weights.

Computes the SAME measured features on the real data as on the mocks:
  - RFIB_ASINH from the real coadded spectrum through decamDR1noatm-r
    (the photometric FIBERMAG_R never enters the model -- avoids the
    fiber-loss / aperture covariate shift for extended dwarfs);
  - W_feat shape fractions + residual + overlap fit at the catalog z_RR;
  - LOG_DELTACHI2 and Z_RR from the catalog.

Outputs a per-galaxy table with:
  PURITY_WEIGHT   p = P(correct z | features, in catalog)   [down-weight]
  COMPLETENESS    c = P(correct z | features) over the full population
  W_SPEC          p / max(c, COMPLETENESS_FLOOR)            [science weight]

W_spec corrects redshift-measurement incompleteness ONLY. Targeting and
fiber-assignment incompleteness are separate multiplicative terms.

Run inside the DESI env on a CPU node.
'''

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from multiprocessing import Pool

import h5py
import numpy as np
from astropy.table import Table
from joblib import load

import config
import specz_utils

N_PROC = int(os.environ.get("SLURM_CPUS_PER_TASK", 64))
OUTPUT_FITS = f"{config.MODELS_DIR}/dwarf_catalog_specz_weights.fits"

# Floor on the completeness probability when forming 1/c, to keep single
# objects from carrying unbounded weight. Objects at the floor should be
# treated as "uncorrectable" parameter space (both slides' methods fail
# where completeness ~ 0) -- flag rather than trust.
COMPLETENESS_FLOOR = 0.02

_CTX = None


def _init_worker(ctx):
    global _CTX
    _CTX = ctx


def _measure_one(i):
    W_feat, wave_rest, wave, flux, ivar, z_rr = _CTX
    shape_frac, redchi2, overlap, _ = specz_utils.fit_feature_basis(
        W_feat, wave_rest, wave, flux[i], ivar[i], z_rr[i])
    return shape_frac, redchi2, overlap


if __name__ == "__main__":
    specz_utils.check_dirs()

    cat = Table.read(config.DWARF_CATALOG)
    print(f"Dwarf catalog: {len(cat)} objects")

    with h5py.File(config.DWARF_SPECTRA_H5, "r") as f:
        spec_tgids = f["TARGETID"][:]
        wave = f["WAVE"][:]
        flux = f["FLUX"][:]
        ivar = f["FLUX_IVAR"][:]
        z_h5 = f["Z"][:]

    # align catalog rows to the spectra file
    idx_of = {t: i for i, t in enumerate(spec_tgids)}
    has_spec = np.array([t in idx_of for t in cat["TARGETID"]])
    print(f"{int(has_spec.sum())} catalog objects have consolidated spectra "
          f"({int((~has_spec).sum())} missing -> excluded, no weight assigned)")
    cat = cat[has_spec]
    rows = np.array([idx_of[t] for t in cat["TARGETID"]])
    flux, ivar = flux[rows], ivar[rows]
    z_rr = np.asarray(cat["Z"], dtype=float)

    # ------------------------------------------------------------------
    # Features
    # ------------------------------------------------------------------
    W_feat = np.load(config.FEAT_BASIS_NPY)
    wave_rest = specz_utils.get_rest_wave()

    print("Measuring W_feat features on real spectra ...")
    with Pool(processes=N_PROC, initializer=_init_worker,
              initargs=((W_feat, wave_rest, wave, flux, ivar, z_rr),)) as pool:
        out = pool.map(_measure_one, range(len(cat)), chunksize=32)
    shape_frac = np.array([o[0] for o in out])
    redchi2 = np.array([o[1] for o in out])
    overlap = np.array([o[2] for o in out])

    feat = Table()
    feat["TARGETID"] = cat["TARGETID"]
    feat["RFIB_ASINH"] = specz_utils.synthetic_fiber_mag_asinh(wave, flux)
    feat["LOG_DELTACHI2"] = np.log10(np.clip(np.asarray(cat["DELTACHI2"], dtype=float),
                                             1e-3, None))
    feat["Z_RR"] = z_rr
    for i in range(config.N_TEMPLATES_FEAT - 1):
        feat[f"SHAPE_FRAC_{i}"] = shape_frac[:, i]
    feat["FEAT_REDCHI2"] = redchi2
    feat["FEAT_OVERLAP"] = overlap

    X = specz_utils.assemble_features(feat)

    # ------------------------------------------------------------------
    # Weights
    # ------------------------------------------------------------------
    purity = load(config.PURITY_MODEL_JOBLIB)
    completeness = load(config.COMPLETENESS_MODEL_JOBLIB)
    assert purity["feature_names"] == list(config.FEATURE_NAMES)
    assert completeness["feature_names"] == list(config.FEATURE_NAMES)

    p = specz_utils.predict_proba(purity["scaler"], purity["clf"], X)
    c = specz_utils.predict_proba(completeness["scaler"], completeness["clf"], X)

    feat["PURITY_WEIGHT"] = p
    feat["COMPLETENESS"] = c
    feat["AT_COMPLETENESS_FLOOR"] = (c < COMPLETENESS_FLOOR).astype(int)
    feat["W_SPEC"] = p / np.clip(c, COMPLETENESS_FLOOR, None)

    feat.write(OUTPUT_FITS, overwrite=True)
    print(f"Wrote {OUTPUT_FITS}")
    print(f"W_SPEC: median={np.median(feat['W_SPEC']):.3f}, "
          f"[16,84]=[{np.percentile(feat['W_SPEC'], 16):.3f}, "
          f"{np.percentile(feat['W_SPEC'], 84):.3f}], "
          f"at floor: {int(np.sum(feat['AT_COMPLETENESS_FLOOR']))} objects "
          f"(treat as uncorrectable parameter space)")

'''
Stage 1: Build the reference galaxy sample for NNMF training + bootstrapping.

Selects a BROAD (dwarfs AND non-dwarfs) high-SNR, robust-redshift galaxy
sample at z < 0.15 from the Iron BGS Bright + LOWZ Dark VAC catalogs,
downloads their coadded spectra, and consolidates everything into one HDF5.

Also:
  - saves the empirical effective-exposure-time distributions (from the FULL
    catalogs' TSNR2 columns, i.e. the target population, not just the high-SNR
    subset) used to sample realistic depths for the mocks;
  - flags the BRIGHT POOL subset (synthetic rfib < POOL_RFIB_MAX): the
    coefficient vectors bootstrapped for mock generation are drawn only from
    this subset, where even passive galaxies get robust redshifts, so the
    spectral-type mixture is nearly selection-free (quenched-dwarf coverage fix).

Run on a NERSC CPU node with the DESI environment loaded.
Output: config.REFERENCE_CATALOG, config.REFERENCE_SPECTRA_H5,
        config.REFERENCE_EFFTIME_NPZ
'''

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import h5py
import numpy as np
from astropy.table import Table, vstack

import config
import specz_utils


def select_reference(cat, program_label):
    """High-SNR robust-z galaxy selection for the NNMF training sample."""
    sel = (cat["Z"] > config.REF_Z_MIN) & (cat["Z"] < config.REF_Z_MAX)
    sel &= cat["DELTACHI2"] > config.REF_DELTACHI2_MIN
    sel &= np.isin(cat["ZWARN"], config.REF_ZWARN_ALLOWED)
    sel &= np.char.strip(np.asarray(cat["SPECTYPE"]).astype(str)) == config.REF_SPECTYPE
    out = cat[sel]
    out["REF_PROGRAM"] = program_label
    return out


def ensure_spectra_columns(cat, survey, program):
    """read_spectra_parallel needs TARGETID/HEALPIX/SURVEY/PROGRAM columns."""
    if "SURVEY" not in cat.colnames:
        cat["SURVEY"] = survey
    if "PROGRAM" not in cat.colnames:
        cat["PROGRAM"] = program
    assert "HEALPIX" in cat.colnames, "catalog must carry HEALPIX for spectra I/O"
    return cat


if __name__ == "__main__":
    specz_utils.check_dirs()

    # -------------------------------------------------------------------
    # Load catalogs, save efftime distributions from the FULL populations
    # -------------------------------------------------------------------
    print("Loading Iron VAC catalogs ...")
    bgs = Table.read(config.IRON_BGS_BRIGHT_VAC)
    lowz = Table.read(config.IRON_LOWZ_DARK_VAC)
    print(f"  BGS Bright: {len(bgs)}   LOWZ Dark: {len(lowz)}")

    specz_utils.build_efftime_distribution(
        bgs["TSNR2_BGS"], lowz["TSNR2_LRG"], config.REFERENCE_EFFTIME_NPZ)
    print(f"Saved efftime distributions -> {config.REFERENCE_EFFTIME_NPZ}")

    # -------------------------------------------------------------------
    # Reference selection + dedupe
    # -------------------------------------------------------------------
    bgs_ref = ensure_spectra_columns(select_reference(bgs, "BGS_BRIGHT"),
                                     survey="main", program="bright")
    lowz_ref = ensure_spectra_columns(select_reference(lowz, "LOWZ_DARK"),
                                      survey="main", program="dark")

    keep_cols = ["TARGETID", "HEALPIX", "SURVEY", "PROGRAM", "REF_PROGRAM",
                 "Z", "ZWARN", "DELTACHI2", "SPECTYPE", "FIBERMAG_R"]
    keep_cols = [c for c in keep_cols if c in bgs_ref.colnames]
    ref = vstack([bgs_ref[keep_cols], lowz_ref[[c for c in keep_cols
                                                if c in lowz_ref.colnames]]])

    _, unique_idx = np.unique(ref["TARGETID"], return_index=True)
    ref = ref[np.sort(unique_idx)]
    print(f"Reference sample after dedupe: {len(ref)}")

    # -------------------------------------------------------------------
    # Fetch + coadd spectra
    # -------------------------------------------------------------------
    import desispec.io
    from desispec import coaddition

    print("Reading coadded spectra (parallel) ...")
    spec = desispec.io.spectra.read_spectra_parallel(
        ref, nproc=32, prefix="coadd",
        rdspec_kwargs={"skip_hdus": ["EXP_FIBERMAP", "SCORES", "EXTRA_CATALOG",
                                     "MASK", "RESOLUTION"]},
        specprod=config.SPECPROD, match_order=True)
    spec = coaddition.coadd_cameras(spec)

    wave = spec.wave["brz"]
    flux = spec.flux["brz"]
    ivar = spec.ivar["brz"]

    # -------------------------------------------------------------------
    # Synthetic fiber mags (same definition the model uses) + pool flag
    # -------------------------------------------------------------------
    print("Computing synthetic fiber magnitudes ...")
    maggies = specz_utils.synthetic_fiber_maggies(wave, flux)
    rfib_synth = specz_utils.asinh_mag_from_maggies(maggies)
    is_pool = rfib_synth < config.POOL_RFIB_MAX
    print(f"Bright pool (rfib_synth < {config.POOL_RFIB_MAX}): {int(is_pool.sum())} "
          f"of {len(ref)}")

    ref["RFIB_SYNTH"] = rfib_synth
    ref["IS_POOL"] = is_pool.astype(int)
    ref.write(config.REFERENCE_CATALOG, overwrite=True)

    with h5py.File(config.REFERENCE_SPECTRA_H5, "w") as f:
        f.create_dataset("TARGETID", data=np.asarray(ref["TARGETID"]), dtype="i8")
        f.create_dataset("Z", data=np.asarray(ref["Z"]), dtype="f8")
        f.create_dataset("WAVE", data=wave, dtype="f4")
        f.create_dataset("FLUX", data=flux, dtype="f4")
        f.create_dataset("FLUX_IVAR", data=ivar, dtype="f4")
        f.create_dataset("RFIB_SYNTH", data=rfib_synth, dtype="f4")
        f.create_dataset("IS_POOL", data=is_pool.astype("i1"))

    print(f"Wrote {config.REFERENCE_CATALOG}")
    print(f"Wrote {config.REFERENCE_SPECTRA_H5}")

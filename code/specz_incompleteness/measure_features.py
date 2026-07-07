'''
Stage 4: Measure NN features on the noisy mocks and build the training table.

LOAD-BEARING CONDITION (see README): features are measured for EVERY mock
with z_true < 0.15 -- including redshift failures, ZWARN != 0, and wildly
wrong z_RR. The completeness model must be trained on this full population or
the weighted estimator breaks. De-redshifting at a bad z_RR leaves only
partial overlap with the W_feat rest grid; the fit uses the overlapping
pixels and records the overlap fraction as a feature.

Per mock, from the NOISY coadd (never the noise-free truth):
  RFIB_ASINH     asinh synthetic decamDR1noatm-r fiber mag of the noisy coadd
  LOG_DELTACHI2  log10(DELTACHI2) from redrock
  Z_RR           redrock redshift
  SHAPE_FRAC_i   W_feat NNLS coefficients at z_RR, rescaled to unit sum
  FEAT_REDCHI2   reduced chi2 of the W_feat fit
  FEAT_OVERLAP   fraction of the W_feat rest grid covered at z_RR
plus labels:
  IS_CORRECT     NOT catastrophic_z(Z_TRUE, Z_RR)   [threshold 0.0033]
  IN_CATALOG     mirrors the real dwarf-catalog selection (purity population)

Usage (inside DESI env, CPU node):
    python measure_features.py --start 0 --end 250      # per-chunk tables
    python measure_features.py --consolidate            # single training table
'''

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from multiprocessing import Pool

import numpy as np
from astropy.table import Table, vstack

import config
import specz_utils
from generate_mock_spectra import truth_path, coadd_path, redrock_path

N_PROC = int(os.environ.get("SLURM_CPUS_PER_TASK", 64))
TRAINING_TABLE = f"{config.TRAINING_DIR}/training_table.fits"


def features_path(chunk_id):
    return f"{config.TRAINING_DIR}/features_chunk_{chunk_id:05d}.fits"


# ----------------------------------------------------------------------------
# Per-spectrum worker
# ----------------------------------------------------------------------------
_CTX = None


def _init_worker(ctx):
    global _CTX
    _CTX = ctx


def _measure_one(i):
    W_feat, wave_rest, wave, flux, ivar, z_rr = _CTX
    shape_frac, redchi2, overlap, _ = specz_utils.fit_feature_basis(
        W_feat, wave_rest, wave, flux[i], ivar[i], z_rr[i])
    return shape_frac, redchi2, overlap


def read_redrock(chunk_id):
    rr_file = redrock_path(chunk_id)
    try:
        return Table.read(rr_file, hdu="REDSHIFTS")
    except KeyError:
        return Table.read(rr_file)


def process_chunk(chunk_id, W_feat, wave_rest):
    import desispec.io
    from desispec import coaddition

    truth = Table.read(truth_path(chunk_id))
    rr = read_redrock(chunk_id)
    assert len(rr) == len(truth), \
        f"chunk {chunk_id}: redrock rows ({len(rr)}) != truth rows ({len(truth)})"
    # simExposure assigns TARGETID = arange(nspec), row-aligned with input
    if "TARGETID" in rr.colnames:
        order = np.argsort(np.asarray(rr["TARGETID"]))
        rr = rr[order]

    spec = desispec.io.read_spectra(coadd_path(chunk_id))
    spec = coaddition.coadd_cameras(spec)
    wave = spec.wave["brz"]
    flux = spec.flux["brz"]
    ivar = spec.ivar["brz"]

    z_rr = np.asarray(rr["Z"], dtype=float)

    with Pool(processes=N_PROC, initializer=_init_worker,
              initargs=((W_feat, wave_rest, wave, flux, ivar, z_rr),)) as pool:
        out = pool.map(_measure_one, range(len(truth)), chunksize=32)

    shape_frac = np.array([o[0] for o in out])
    redchi2 = np.array([o[1] for o in out])
    overlap = np.array([o[2] for o in out])

    tab = truth.copy()
    tab["Z_RR"] = z_rr
    tab["DELTACHI2"] = np.asarray(rr["DELTACHI2"], dtype=float)
    tab["ZWARN"] = np.asarray(rr["ZWARN"], dtype=int)
    tab["SPECTYPE"] = np.asarray(rr["SPECTYPE"]).astype(str)

    # features (measured on the NOISY coadd)
    tab["RFIB_ASINH"] = specz_utils.synthetic_fiber_mag_asinh(wave, flux)
    tab["LOG_DELTACHI2"] = np.log10(np.clip(tab["DELTACHI2"], 1e-3, None))
    for i in range(config.N_TEMPLATES_FEAT - 1):
        tab[f"SHAPE_FRAC_{i}"] = shape_frac[:, i]
    tab["SHAPE_FRAC_LAST"] = shape_frac[:, -1]   # stored for reference, not a feature
    tab["FEAT_REDCHI2"] = redchi2
    tab["FEAT_OVERLAP"] = overlap

    # labels
    tab["IS_CORRECT"] = (~specz_utils.catastrophic_z(
        np.asarray(truth["Z_TRUE"]), z_rr)).astype(int)
    tab["IN_CATALOG"] = specz_utils.in_catalog_selection(
        z_rr, tab["ZWARN"], tab["SPECTYPE"]).astype(int)

    tab.write(features_path(chunk_id), overwrite=True)
    return tab


def consolidate():
    import glob
    files = sorted(glob.glob(f"{config.TRAINING_DIR}/features_chunk_*.fits"))
    print(f"Consolidating {len(files)} chunk tables ...")
    total = vstack([Table.read(f) for f in files], metadata_conflicts="silent")
    total.write(TRAINING_TABLE, overwrite=True)
    n_cat = int(np.sum(total["IN_CATALOG"]))
    n_cor = int(np.sum(total["IS_CORRECT"]))
    print(f"Wrote {TRAINING_TABLE}: {len(total)} mocks, "
          f"{n_cat} in-catalog (purity pop.), {n_cor} correct-z "
          f"({n_cor / len(total):.3f})")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--start", type=int)
    p.add_argument("--end", type=int)
    p.add_argument("--consolidate", action="store_true")
    args = p.parse_args()

    specz_utils.check_dirs()

    if args.consolidate:
        consolidate()
    else:
        assert args.start is not None and args.end is not None
        W_feat = np.load(config.FEAT_BASIS_NPY)
        wave_rest = specz_utils.get_rest_wave()
        for chunk_id in range(args.start, args.end):
            if os.path.exists(features_path(chunk_id)):
                print(f"chunk {chunk_id}: exists, skipping")
                continue
            process_chunk(chunk_id, W_feat, wave_rest)
            print(f"chunk {chunk_id}: features written")

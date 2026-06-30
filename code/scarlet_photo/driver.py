"""
driver.py -- STAGE 1: run the SCARLET fit over an input catalogue.

Mirrors tractor_model.py: argparse, a module-level `_worker` wrapped in
try/except (so one bad object never tears down the pool), mp.Pool.imap_unordered
by OBJECT, and incremental skip on per-object fragment existence. Each object
writes its VI bundle fragment to {FILE_PATH}/scarlet_vi_bundle.h5 (stage 2 packs
them); this driver also collects the initial-composite SCARLET magnitudes into
one output FITS catalogue (the benchmark deliverable).

Run as `python -m scarlet_photo.driver --input-catalog cat.fits` (scarlet env).
The input is a single pre-filtered FITS catalogue; selection is done upstream
(no --sample/--clean flags).
"""

import os
import sys
import argparse
import multiprocessing as mp

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from scarlet_photo.config import ScarletConfig
    from scarlet_photo import pipeline as _pipeline
else:
    from .config import ScarletConfig
    from . import pipeline as _pipeline


def _result_to_row(res):
    """Flatten a ScarletResult into scalar catalogue columns."""
    def g(arr, i):
        if arr is None:
            return np.nan
        try:
            return float(arr[i])
        except Exception:                                   # noqa: BLE001
            return np.nan
    out = {
        "TARGETID": int(res.targetid),
        "SCARLET_STATUS": str(res.status),
        "N_COMPONENTS": int(res.n_components),
        "N_EXTENDED": int(res.n_extended),
        "N_STARS": int(res.n_stars),
        "N_MEMBERS": int(res.n_members),
        "INVARIANT_MAX_FRAC": float(res.invariant_max_frac),
    }
    bands = ("G", "R", "Z")
    for i, b in enumerate(bands):
        out["SCARLET_MAG_{}_TOTAL".format(b)] = g(res.total_mag, i)
        out["SCARLET_MAGERR_{}_TOTAL".format(b)] = g(res.total_mag_err, i)
        out["SCARLET_MAG_{}_R4".format(b)] = g(res.r4_mag, i)
        out["SCARLET_MAGERR_{}_R4".format(b)] = g(res.r4_mag_err, i)
    return out


def _worker(task):
    row, cfg = task
    tgid = int(row["TARGETID"])
    try:
        res = _pipeline.run_scarlet_object(row, cfg)
    except Exception as exc:                                # noqa: BLE001
        # absolute backstop -- run_scarlet_object should already be total
        from scarlet_photo.pipeline import ScarletResult
        res = ScarletResult(targetid=tgid, status="error", error_msg=repr(exc))
    return _result_to_row(res)


def _rows_as_dicts(catalog):
    cols = catalog.colnames
    out = []
    for r in catalog:
        out.append({c: r[c] for c in cols})
    return out


def main(argv=None):
    from astropy.table import Table

    p = argparse.ArgumentParser(description="Stage-1 SCARLET fit over an input catalogue.")
    p.add_argument("--input-catalog", required=True,
                   help="FITS catalogue (pre-filtered *_w_aper_mags); selection done upstream.")
    p.add_argument("--output", default=None,
                   help="Output FITS of SCARLET mags (default: <input>_scarlet_mags.fits).")
    p.add_argument("--ncores", type=int, default=128)
    p.add_argument("--overwrite", action="store_true", help="Re-fit objects whose fragment exists.")
    p.add_argument("--tgids", type=int, nargs="*", default=None, help="Restrict to these TARGETIDs.")
    p.add_argument("--limit", type=int, default=None, help="Fit only the first N rows (testing).")
    p.add_argument("--save-plots", action="store_true", help="Save the per-object diagnostic panel.")
    p.add_argument("--cutouts-dir", default=None)
    p.add_argument("--psfs-dir", default=None)
    p.add_argument("--model-psf-sigma", type=float, default=None)
    p.add_argument("--detection-method", default=None, choices=["chi2", "ivar_weighted", "sum"])
    p.add_argument("--grouping-rule", default=None, choices=["or", "bluebox", "gmm"])
    args = p.parse_args(argv)

    cfg = ScarletConfig(save_plots=args.save_plots)
    if args.cutouts_dir:
        cfg.cutouts_dir = args.cutouts_dir
    if args.psfs_dir:
        cfg.psfs_dir = args.psfs_dir
    if args.model_psf_sigma is not None:
        cfg.model_psf_sigma = args.model_psf_sigma
    if args.detection_method:
        cfg.detection_method = args.detection_method
    if args.grouping_rule:
        cfg.grouping_rule = args.grouping_rule

    cat = Table.read(args.input_catalog)
    if args.tgids:
        cat = cat[np.isin(np.asarray(cat["TARGETID"]), np.asarray(args.tgids))]
    if args.limit:
        cat = cat[: args.limit]

    rows = _rows_as_dicts(cat)

    # incremental skip on fragment existence (unless overwrite)
    if not args.overwrite:
        rows = [r for r in rows if not _pipeline.fragment_exists(r, cfg)]
    if not rows:
        print("Nothing to do (all fragments present; use --overwrite to re-fit).")
        return

    print("Fitting {} objects with {} cores ...".format(len(rows), args.ncores))
    tasks = [(r, cfg) for r in rows]

    results = []
    try:
        from tqdm import tqdm
        progress = lambda it: tqdm(it, total=len(tasks))
    except Exception:                                       # noqa: BLE001
        progress = lambda it: it

    if args.ncores > 1 and len(tasks) > 1:
        with mp.Pool(args.ncores) as pool:
            for row_out in progress(pool.imap_unordered(_worker, tasks, chunksize=8)):
                results.append(row_out)
    else:
        for task in progress(tasks):
            results.append(_worker(task))

    # status summary
    from collections import Counter
    counts = Counter(r["SCARLET_STATUS"] for r in results)
    print("Status:", dict(counts))

    out_path = args.output or args.input_catalog.replace(".fits", "_scarlet_mags.fits")
    out_tbl = Table(rows=results) if results else Table()
    out_tbl.write(out_path, overwrite=True)
    print("Wrote SCARLET mags for {} objects -> {}".format(len(results), out_path))


if __name__ == "__main__":
    main()

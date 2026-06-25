"""build_bundle.py -- NERSC export step for the recon_vi GUI.

Packs a hand-picked VI catalog (a row-subset of the ``*_w_aper_mags*.fits``
catalogs) plus the per-object reconstruction artifacts into a single HDF5
bundle that is ``scp``-ed down and inspected locally.

What the GUI fine-tunes
-----------------------
The "reconstruction" the user edits is the REAL masked science cube
(``final_reconstruct_galaxy_subfunction.npy`` -- background+blend subtracted
data with the cog mask applied), exactly the image produced by the pipeline.
Fine-tuning = adding/subtracting individual Tractor *source models* on top of
that real image:

    fine_tuned = science_cube - sum(removed source models) + sum(added models)

so the per-source model ``.npy`` files (one ``(3, S, S)`` grz flux model per
parent-segment source, written only for z >= 0.005) are shipped as bbox-cropped
flux patches. Toggling a source on/off is then a pure numpy add/subtract in the
browser -- no Tractor/legacypipe at runtime.

z < 0.005 objects have no per-source models (``recon_variant == "no_isolate"``);
they are shipped view-only: input cutout + the stored no-isolate cube, with
toggling disabled.

Run (NERSC)::

    python build_bundle.py -vi_catalog PICKED.fits -out bundle.h5 [-ncores N]

Kept Python-3.8-safe: numpy / h5py / astropy / ``cutout_store`` only (no
legacypipe, no scipy). The galaxy-center pixel is computed from the cutout WCS
(astropy); the grid size S and recon variant are read from the files on disk,
so no ``IMAGE_SIZE_PIX`` column or ``make_custom_wcs`` is required.
"""

import os
import sys
import csv
import time
import argparse
import traceback
import multiprocessing as mp
from functools import partial

import numpy as np
import h5py

# code/ is the parent of code/recon_vi/ -- put it on the path so we can import
# the canonical cutout store (single source of truth for reading + cropping the
# input sky cutouts and their WCS).
_CODE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _CODE_DIR not in sys.path:
    sys.path.insert(0, _CODE_DIR)

import cutout_store  # noqa: E402


# Per-object cube filenames written by aperture_cogs.py.
ISOLATE_CUBE = "final_reconstruct_galaxy_subfunction.npy"
NO_ISOLATE_CUBE = "final_reconstruct_galaxy_subfunction_no_isolate.npy"
PARENT_SOURCES = "parent_galaxy_sources.fits"
ISOLATE_FINAL = "parent_galaxy_sources_isolate_FINAL.fits"
MODELS_SUBDIR = "tractor_models"
NOISE_RMS_FILE = "noise_per_band_rms.npy"  # [sig_g, sig_r, sig_z] from aperture_photo.py


def _model_filename(objid):
    return "tractor_parent_source_model_{:d}.npy".format(int(objid))


# Per-source fields copied straight out of parent_galaxy_sources.fits. These all
# live there (no join needed); missing optional ones are filled with NaN/"".
SOURCE_NUM_COLS = (
    "xpix", "ypix", "separations", "mag_g", "mag_r", "mag_z",
    "sersic", "shape_r", "shape_e1", "shape_e2", "ra", "dec",
)


def argument_parser():
    p = argparse.ArgumentParser(
        description="Pack a VI catalog + reconstruction artifacts into one HDF5 bundle.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("-vi_catalog", required=True,
                   help="FITS catalog (subset of *_w_aper_mags*.fits) with "
                        "TARGETID, BRICKNAME, FILE_PATH, "
                        "APER_RADEC_CEN_ISOLATE, APER_RADEC_CEN_NO_ISOLATE.")
    p.add_argument("-out", required=True, help="output HDF5 bundle path.")
    p.add_argument("-ncores", type=int, default=1,
                   help="worker processes reading per-object data.")
    p.add_argument("-rel_thresh", type=float, default=1e-4,
                   help="patch bbox keeps pixels where band-summed |model| > "
                        "rel_thresh * peak (display-only flux loss).")
    p.add_argument("-float16", action="store_true",
                   help="store the two display cubes as float16 to ~halve the "
                        "bundle (model patches stay float32 for subtraction).")
    p.add_argument("-limit", type=int, default=0,
                   help="debug: only process the first N catalog rows (0=all).")
    return p


# ----------------------------------------------------------------------
# Per-object readers (run in worker processes; read-only)
# ----------------------------------------------------------------------

def _crop_patch(model, rel_thresh):
    """Crop a ``(3, S, S)`` flux model to the bbox of its non-negligible
    pixels. Returns ``(patch (3,h,w) float32, y0, x0)``; for an all-zero model
    returns a 1x1 zero patch at (0, 0)."""
    band_sum = np.abs(model).sum(axis=0)
    peak = float(band_sum.max()) if band_sum.size else 0.0
    if peak <= 0.0:
        return np.zeros((3, 1, 1), dtype=np.float32), 0, 0
    keep = band_sum > (rel_thresh * peak)
    ys, xs = np.where(keep)
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    return np.ascontiguousarray(model[:, y0:y1, x0:x1], dtype=np.float32), y0, x0


def _read_table(path):
    from astropy.table import Table
    return Table.read(path)


def _col_or_nan(tbl, name, n):
    if name in tbl.colnames:
        return np.asarray(tbl[name], dtype=np.float64)
    return np.full(n, np.nan, dtype=np.float64)


def _finite_pair(val):
    """Return (ra, dec) floats if ``val`` is a finite 2-vector, else None."""
    try:
        arr = np.asarray(val, dtype=np.float64).ravel()
    except (TypeError, ValueError):
        return None
    if arr.size >= 2 and np.isfinite(arr[0]) and np.isfinite(arr[1]):
        return float(arr[0]), float(arr[1])
    return None


def _gal_pixel(wcs, gal_radec, fallback_xy):
    """world->pixel (0-indexed) for the galaxy center, with a pixel fallback.

    The stored cutout WCS is 3-D (the image is a ``(3, S, S)`` grz cube), so we
    reduce it to its celestial 2-D subset before projecting -- otherwise astropy
    rejects the 2-coordinate ``all_world2pix(ra, dec, 0)`` call.
    """
    if gal_radec is not None:
        cel = wcs.celestial
        x, y = cel.all_world2pix(gal_radec[0], gal_radec[1], 0)
        x, y = float(x), float(y)
        if np.isfinite(x) and np.isfinite(y):
            return x, y
    return float(fallback_xy[0]), float(fallback_xy[1])


def _bkg_rms_fallback(cube):
    """Per-band background RMS (g, r, z) when ``noise_per_band_rms.npy`` is
    absent: MAD-std of 3-sigma-clipped pixels, an astropy-only approximation of
    aperture_photo.py's ``MADStdBackgroundRMS`` + ``SigmaClip(sigma=3, maxiters=5)``.
    Returns float32[3], NaN for any degenerate band."""
    from astropy.stats import sigma_clip, mad_std
    out = np.full(3, np.nan, dtype=np.float32)
    for b in range(3):
        band = np.asarray(cube[b], dtype=np.float64).ravel()
        band = band[np.isfinite(band)]
        if band.size == 0:
            continue
        clipped = sigma_clip(band, sigma=3.0, maxiters=5, masked=True)
        val = mad_std(clipped, ignore_nan=True)
        if np.isfinite(val) and val > 0:
            out[b] = np.float32(val)
    return out


def _read_noise_rms(file_path, cutout):
    """Per-band bkg RMS for the masked-hole infill. Prefer the pipeline's own
    saved value (``noise_per_band_rms.npy`` = ``[sig_g, sig_r, sig_z]``, exactly
    the ``MADStdBackgroundRMS`` used in aperture_photo.py); else approximate it
    from the input cutout. float32[3], NaN where unavailable -> infill stays off
    for that object in the GUI."""
    path = os.path.join(file_path, NOISE_RMS_FILE)
    if os.path.exists(path):
        try:
            arr = np.asarray(np.load(path), dtype=np.float32).ravel()
            if arr.size >= 3 and np.all(np.isfinite(arr[:3])) and np.all(arr[:3] > 0):
                return np.ascontiguousarray(arr[:3], dtype=np.float32)
        except Exception:  # noqa: BLE001 -- corrupt/odd file -> recompute below
            pass
    try:
        return _bkg_rms_fallback(cutout)
    except Exception:  # noqa: BLE001 -- never abort a build over infill metadata
        return np.full(3, np.nan, dtype=np.float32)


def _build_object(row, rel_thresh):
    """Read everything the GUI needs for one catalog row.

    Returns a dict with ``status='ok'`` (+ payload) or ``status='skip'``
    (+ reason). Any unexpected error is caught and turned into a skip so one
    bad object never aborts the whole run.
    """
    tgid = int(row["TARGETID"])
    brick = str(row["BRICKNAME"])
    try:
        file_path = str(row["FILE_PATH"])
        if not os.path.isdir(file_path):
            return {"status": "skip", "targetid": tgid, "brickname": brick,
                    "reason": "FILE_PATH missing: {}".format(file_path)}

        models_dir = os.path.join(file_path, MODELS_SUBDIR)
        has_models = (os.path.isdir(models_dir)
                      and any(f.endswith(".npy") for f in os.listdir(models_dir)))

        iso_cen = _finite_pair(row["APER_RADEC_CEN_ISOLATE"]
                               if "APER_RADEC_CEN_ISOLATE" in row.colnames else None)
        noiso_cen = _finite_pair(row["APER_RADEC_CEN_NO_ISOLATE"]
                                 if "APER_RADEC_CEN_NO_ISOLATE" in row.colnames else None)

        # ---- view-only (no per-source models) -------------------------------
        if not has_models:
            cube_path = os.path.join(file_path, NO_ISOLATE_CUBE)
            if not os.path.exists(cube_path):
                return {"status": "skip", "targetid": tgid, "brickname": brick,
                        "reason": "no tractor_models/ and no no-isolate cube"}
            recon_cube = np.load(cube_path)
            return _finish_view_only(tgid, brick, file_path, recon_cube,
                                     noiso_cen or iso_cen)

        # ---- toggleable (z >= 0.005) ----------------------------------------
        cube_path = os.path.join(file_path, ISOLATE_CUBE)
        if not os.path.exists(cube_path):
            return {"status": "skip", "targetid": tgid, "brickname": brick,
                    "reason": "missing {}".format(ISOLATE_CUBE)}
        science_cube = np.load(cube_path)
        if science_cube.ndim != 3 or science_cube.shape[0] != 3:
            return {"status": "skip", "targetid": tgid, "brickname": brick,
                    "reason": "bad science cube shape {}".format(science_cube.shape)}
        S = int(science_cube.shape[-1])

        parent_path = os.path.join(file_path, PARENT_SOURCES)
        final_path = os.path.join(file_path, ISOLATE_FINAL)
        if not os.path.exists(parent_path):
            return {"status": "skip", "targetid": tgid, "brickname": brick,
                    "reason": "missing {}".format(PARENT_SOURCES)}
        if not os.path.exists(final_path):
            return {"status": "skip", "targetid": tgid, "brickname": brick,
                    "reason": "missing {}".format(ISOLATE_FINAL)}

        parent = _read_table(parent_path)
        final = _read_table(final_path)
        if "source_objid_new" not in parent.colnames:
            return {"status": "skip", "targetid": tgid, "brickname": brick,
                    "reason": "parent sources missing source_objid_new"}
        on_set = set(int(o) for o in np.asarray(final["source_objid_new"]))

        n = len(parent)
        objids = np.asarray(parent["source_objid_new"]).astype(np.int64)
        types = (np.asarray(parent["type"]).astype(str) if "type" in parent.colnames
                 else np.array(["?"] * n))
        cols = {c: _col_or_nan(parent, c, n) for c in SOURCE_NUM_COLS}

        sources = []
        for k in range(n):
            objid = int(objids[k])
            mpath = os.path.join(models_dir, _model_filename(objid))
            if not os.path.exists(mpath):
                return {"status": "skip", "targetid": tgid, "brickname": brick,
                        "reason": "missing model for objid {}".format(objid)}
            model = np.load(mpath)
            if model.shape != (3, S, S):
                return {"status": "skip", "targetid": tgid, "brickname": brick,
                        "reason": "model {} shape {} != (3,{},{})".format(
                            objid, model.shape, S, S)}
            patch, y0, x0 = _crop_patch(model, rel_thresh)
            sources.append({
                "objid": objid,
                "patch": patch,
                "bbox": (y0, x0),
                "initial_membership": objid in on_set,
                "type": str(types[k]),
                "xpix": float(cols["xpix"][k]),
                "ypix": float(cols["ypix"][k]),
                "separation": float(cols["separations"][k]),
                "mag_g": float(cols["mag_g"][k]),
                "mag_r": float(cols["mag_r"][k]),
                "mag_z": float(cols["mag_z"][k]),
                "sersic": float(cols["sersic"][k]),
                "shape_r": float(cols["shape_r"][k]),
                "shape_e1": float(cols["shape_e1"][k]),
                "shape_e2": float(cols["shape_e2"][k]),
                "ra": float(cols["ra"][k]),
                "dec": float(cols["dec"][k]),
            })

        # Target source = the (min-)separation source within 1". Used for the
        # cyan "protected" highlight and as a center-of-last-resort.
        target_objid = -1
        seps = cols["separations"]
        if np.any(np.isfinite(seps)):
            kmin = int(np.nanargmin(seps))
            if np.isfinite(seps[kmin]) and seps[kmin] < 1.0:
                target_objid = int(objids[kmin])
        if np.isfinite(cols["xpix"]).any():
            kmin = int(np.nanargmin(seps)) if np.any(np.isfinite(seps)) else 0
            fallback_xy = (cols["xpix"][kmin], cols["ypix"][kmin])
        else:
            fallback_xy = ((S - 1) / 2.0, (S - 1) / 2.0)

        cut = cutout_store.read_cutout(cutout_store.get_store_dir(), brick, tgid, size=S)
        wcs = cutout_store.get_wcs(cut["header"])
        gal_xpix, gal_ypix = _gal_pixel(wcs, iso_cen or noiso_cen, fallback_xy)
        gal_radec = iso_cen or noiso_cen or (float("nan"), float("nan"))
        noise_rms = _read_noise_rms(file_path, cut["image"])

        return {
            "status": "ok",
            "targetid": tgid,
            "brickname": brick,
            "toggle_disabled": False,
            "recon_variant": "isolate",
            "box_size": S,
            "input_cutout": np.ascontiguousarray(cut["image"], dtype=np.float32),
            "science_cube": np.ascontiguousarray(science_cube, dtype=np.float32),
            "noise_rms_grz": noise_rms,
            "sources": sources,
            "gal_xpix": gal_xpix,
            "gal_ypix": gal_ypix,
            "gal_ra": gal_radec[0],
            "gal_dec": gal_radec[1],
            "target_objid": target_objid,
        }
    except Exception as exc:  # noqa: BLE001 -- never abort the whole run
        return {"status": "skip", "targetid": tgid, "brickname": brick,
                "reason": "exception: {}".format(exc),
                "traceback": traceback.format_exc()}


def _finish_view_only(tgid, brick, file_path, recon_cube, gal_radec):
    if recon_cube.ndim != 3 or recon_cube.shape[0] != 3:
        return {"status": "skip", "targetid": tgid, "brickname": brick,
                "reason": "bad no-isolate cube shape {}".format(recon_cube.shape)}
    S = int(recon_cube.shape[-1])
    try:
        cut = cutout_store.read_cutout(cutout_store.get_store_dir(), brick, tgid, size=S)
    except Exception as exc:  # noqa: BLE001
        return {"status": "skip", "targetid": tgid, "brickname": brick,
                "reason": "no-isolate: cutout read failed: {}".format(exc)}
    wcs = cutout_store.get_wcs(cut["header"])
    gx, gy = _gal_pixel(wcs, gal_radec, ((S - 1) / 2.0, (S - 1) / 2.0))
    noise_rms = _read_noise_rms(file_path, cut["image"])
    return {
        "status": "ok",
        "targetid": tgid,
        "brickname": brick,
        "toggle_disabled": True,
        "recon_variant": "no_isolate",
        "box_size": S,
        "input_cutout": np.ascontiguousarray(cut["image"], dtype=np.float32),
        "recon_cube": np.ascontiguousarray(recon_cube, dtype=np.float32),
        "noise_rms_grz": noise_rms,
        "sources": [],
        "gal_xpix": gx, "gal_ypix": gy,
        "gal_ra": (gal_radec[0] if gal_radec else float("nan")),
        "gal_dec": (gal_radec[1] if gal_radec else float("nan")),
        "target_objid": -1,
    }


# ----------------------------------------------------------------------
# Writing the bundle (single writer, main process)
# ----------------------------------------------------------------------

def _write_object(h5, rec, cube_dtype):
    g = h5.create_group(str(rec["targetid"]))
    g.attrs["BRICKNAME"] = rec["brickname"]
    g.attrs["recon_variant"] = rec["recon_variant"]
    g.attrs["box_size"] = int(rec["box_size"])
    g.attrs["toggle_disabled"] = bool(rec["toggle_disabled"])
    g.attrs["gal_xpix"] = float(rec["gal_xpix"])
    g.attrs["gal_ypix"] = float(rec["gal_ypix"])
    g.attrs["gal_ra"] = float(rec["gal_ra"])
    g.attrs["gal_dec"] = float(rec["gal_dec"])
    g.attrs["target_objid"] = int(rec["target_objid"])
    g.attrs["n_sources"] = len(rec["sources"])
    # per-band bkg RMS [sig_g, sig_r, sig_z] for masked-hole infill (NaN -> off)
    g.attrs["noise_rms_grz"] = np.ascontiguousarray(rec["noise_rms_grz"], dtype=np.float32)

    kw = dict(compression="gzip", compression_opts=4)
    g.create_dataset("input_cutout",
                     data=rec["input_cutout"].astype(cube_dtype), **kw)
    if rec["toggle_disabled"]:
        g.create_dataset("recon_cube",
                         data=rec["recon_cube"].astype(cube_dtype), **kw)
        return

    g.create_dataset("science_cube",
                     data=rec["science_cube"].astype(cube_dtype), **kw)
    sgrp = g.create_group("sources")
    for s in rec["sources"]:
        d = sgrp.create_dataset(str(s["objid"]), data=s["patch"], **kw)
        d.attrs["source_objid_new"] = int(s["objid"])
        d.attrs["bbox"] = np.asarray(s["bbox"], dtype=np.int32)  # (y0, x0)
        d.attrs["initial_membership"] = bool(s["initial_membership"])
        d.attrs["type"] = s["type"]
        for key in ("xpix", "ypix", "separation", "mag_g", "mag_r", "mag_z",
                    "sersic", "shape_r", "shape_e1", "shape_e2", "ra", "dec"):
            d.attrs[key] = float(s[key])


def _write_index(h5, index_rows):
    """Top-level /index compound table for fast list/sidebar loading."""
    dt = np.dtype([
        ("targetid", "i8"),
        ("brickname", h5py.string_dtype()),
        ("recon_variant", h5py.string_dtype()),
        ("toggle_disabled", "?"),
        ("n_sources", "i4"),
    ])
    arr = np.empty(len(index_rows), dtype=dt)
    for i, r in enumerate(index_rows):
        arr[i] = (r["targetid"], r["brickname"], r["recon_variant"],
                  r["toggle_disabled"], r["n_sources"])
    if "index" in h5:
        del h5["index"]
    h5.create_dataset("index", data=arr)


def _write_skipped(path, skipped):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["TARGETID", "BRICKNAME", "reason"])
        for s in skipped:
            w.writerow([s["targetid"], s["brickname"], s["reason"]])


# ----------------------------------------------------------------------
# Driver
# ----------------------------------------------------------------------

def main(argv=None):
    args = argument_parser().parse_args(argv)
    cat = _read_table(args.vi_catalog)
    for col in ("TARGETID", "BRICKNAME", "FILE_PATH"):
        if col not in cat.colnames:
            raise KeyError("VI catalog missing required column {}".format(col))
    rows = list(cat)
    if args.limit:
        rows = rows[:args.limit]
    print("Building bundle for {} objects -> {}".format(len(rows), args.out),
          flush=True)

    cube_dtype = np.float16 if args.float16 else np.float32
    worker = partial(_build_object, rel_thresh=args.rel_thresh)

    out_tmp = args.out + ".tmp"
    if os.path.exists(out_tmp):
        os.remove(out_tmp)

    index_rows = []
    skipped = []
    n_ok = 0
    t0 = time.time()

    if args.ncores and args.ncores > 1:
        ctx = mp.get_context("spawn") if "NERSC_HOST" in os.environ else mp.get_context()
        pool = ctx.Pool(processes=args.ncores)
        results = pool.imap_unordered(worker, rows, chunksize=1)
    else:
        pool = None
        results = (worker(r) for r in rows)

    with h5py.File(out_tmp, "w") as h5:
        for i, rec in enumerate(results):
            if rec["status"] != "ok":
                skipped.append(rec)
                if rec.get("traceback"):
                    print("SKIP {} ({}): {}".format(rec["targetid"], rec["brickname"],
                                                     rec["reason"]), flush=True)
                continue
            _write_object(h5, rec, cube_dtype)
            index_rows.append({k: rec[k] for k in
                               ("targetid", "brickname", "recon_variant",
                                "toggle_disabled")})
            index_rows[-1]["n_sources"] = len(rec["sources"])
            n_ok += 1
            if (n_ok % 100) == 0 or (i + 1) == len(rows):
                print("  {}/{} packed ({} skipped, {:.0f}s)".format(
                    n_ok, len(rows), len(skipped), time.time() - t0), flush=True)
        _write_index(h5, index_rows)
        h5.attrs["n_objects"] = n_ok
        h5.attrs["created"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    if pool is not None:
        pool.close()
        pool.join()

    os.replace(out_tmp, args.out)

    skip_csv = os.path.splitext(args.out)[0] + "_skipped.csv"
    if skipped:
        _write_skipped(skip_csv, skipped)

    print("Done: {} packed, {} skipped ({:.0f}s).".format(
        n_ok, len(skipped), time.time() - t0))
    if skipped:
        print("  skipped objects logged to {}".format(skip_csv))


if __name__ == "__main__":
    main()

"""apply_decisions.py -- STAGE 4 (local): decisions CSV -> final galaxy cubes.

Replays the VI decisions against a bundle.h5 (from build_bundle.py) and writes
the PRIMARY product of the reconstruction project: the final finetuned galaxy
MODEL cube per object -- grz, PSF-convolved (observed frame, NOT deconvolved),
background-free.

The cube is built as the flux-space SUM of the VI-selected component patches::

    galaxy_cube = sum( patch_i  for every component i whose final membership is True )

where final membership matches the VI tool / server exactly::

    member = (initial_membership and cid not in removed) or (cid in added)

The component patches are stored float32 in the bundle (only the *display* cubes
are float16), so summing the members directly gives a clean float32 model cube
independent of the float16 baseline -- provably the same composite the VI tool
showed, up to float16 display roundoff. Pure numpy/h5py (+ astropy for the WCS);
no scarlet, no NERSC.

By default only VI-ACCEPTED objects (verdict=='accept') are written; widen with
--include-unsure / --include-undecided (the latter emits un-VI'd objects using
their automatic initial membership, marked verdict='' in the output).

Each cube also carries a reconstructed TAN WCS. The bundle did NOT propagate the
real cutout header, but it stores an exact anchor pair -- (gal_ra, gal_dec) at
0-based pixel (gal_xpix, gal_ypix), from wcs.all_world2pix in inputs.load_object
-- so a TAN WCS through that anchor at the Legacy Surveys 0.262"/px, N-up/E-left
scale is accurate to well under a pixel across the ~1.5' field.

Usage::

    python -m recon_vi_scarlet.apply_decisions \
        --bundle bundle_partial.h5 \
        --decisions scarlet_decisions.csv \
        --out scarlet_final_cubes.h5 [--include-unsure] [--include-undecided] \
        [--float16]

Output layout (one group per object, keyed by TARGETID)::

    /{TARGETID}/galaxy_cube   (3, S, S) float32  -- the final model cube (grz)
    attrs: TARGETID, BRICKNAME, box_size, gal_ra/dec, gal_xpix/ypix, pixscale,
           wcs_header (FITS header string, TAN),
           verdict, bad_fit, lsb_in_galaxy, comment,
           n_members_final, member_comp_ids (int64 array),
           removed_comp_ids / added_comp_ids (the deltas applied),
           mag_g/r/z (post-VI, = 22.5 - 2.5 log10 sum(member model-frame
           fluxes), NaN-skipped, NOT MW-corrected -- matches the CSV),
           inspector, decision_timestamp, created.
    /index    compound table: targetid, brickname, box_size, n_members_final,
              mag_g, mag_r, mag_z, verdict  (fast catalog-style access).
    top-level attrs: n_objects, source_bundle, source_decisions,
                     n_by_verdict, created.
"""

import os
import csv
import math
import time
import argparse

import numpy as np

# Legacy Surveys grz coadd pixel scale (see code/scarlet_photo/config.py::PIXSCALE).
PIXSCALE = 0.262  # arcsec / pixel


def _to_str(v):
    if isinstance(v, bytes):
        return v.decode("utf-8", "replace")
    return str(v)


def _read_decisions(path):
    """decisions.csv -> {targetid(int): row dict}. Same parsing conventions as
    server.DecisionStore._load (flags may be '1'/'true'/'yes')."""
    rows = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            try:
                tgid = int(row["TARGETID"])
            except (KeyError, ValueError):
                continue
            for flag in ("inspected", "lsb_in_galaxy", "bad_fit"):
                row[flag] = str(row.get(flag, "")).strip().lower() in ("1", "true", "yes")
            rows[tgid] = row
    return rows


def _split_ids(s):
    return [int(x) for x in str(s or "").split(";") if x]


def _apply_patch(cube, patch, y0, x0, sign):
    """cube (3,S,S) += sign * patch (3,h,w) at (y0,x0), clipped to bounds --
    same arithmetic as the VI tool's client-side compositor."""
    S = cube.shape[-1]
    h, w = patch.shape[1], patch.shape[2]
    ys, xs = max(y0, 0), max(x0, 0)
    ye, xe = min(y0 + h, S), min(x0 + w, S)
    if ye <= ys or xe <= xs:
        return
    cube[:, ys:ye, xs:xe] += sign * patch[:, ys - y0:ye - y0, xs - x0:xe - x0]


def _compose_galaxy_cube(g, removed, added):
    """Sum the final-member component patches into one (3,S,S) galaxy model cube.

    Returns (cube float64, sorted member comp_ids, [mag_g, mag_r, mag_z],
    lsb_in_galaxy bool). Membership matches server.post_vi_photometry exactly:
        member = (initial and cid not in removed) or cid in added
    Mags are pure sums of the model-frame per-component fluxes (NaN skipped)."""
    S = int(g.attrs["box_size"])
    comps = g["components"]
    removed, added = set(removed), set(added)
    cube = np.zeros((3, S, S), dtype=np.float64)   # accumulate in f64; cast on save
    members = []
    flux = np.zeros(3, dtype=np.float64)
    lsb_in = False
    for key in comps:
        d = comps[key]
        cid = int(d.attrs["comp_id"])
        initial = bool(d.attrs["initial_membership"])
        member = (initial and cid not in removed) or cid in added
        if not member:
            continue
        members.append(cid)
        if _to_str(d.attrs.get("type", "")) == "starlet_lsb":
            lsb_in = True
        y0, x0 = [int(v) for v in d.attrs["bbox"]]
        _apply_patch(cube, np.asarray(d[:], dtype=np.float64), y0, x0, +1)
        for b, name in enumerate(("flux_g", "flux_r", "flux_z")):
            v = float(d.attrs[name])
            if math.isfinite(v):
                flux[b] += v
    mags = [(22.5 - 2.5 * math.log10(flux[b])) if flux[b] > 0 else float("nan")
            for b in range(3)]
    return cube, sorted(members), mags, lsb_in


def _reconstruct_wcs_header(gal_ra, gal_dec, gal_xpix, gal_ypix, pixscale=PIXSCALE):
    """TAN WCS through the (gal_ra,gal_dec) <-> 0-based (gal_xpix,gal_ypix) anchor.

    Legacy Surveys coadd convention: N-up, E-left, uniform pixscale. Returns the
    FITS header as a string, or '' if the anchor is not finite."""
    from astropy.wcs import WCS
    vals = (gal_ra, gal_dec, gal_xpix, gal_ypix)
    if not all(v is not None and math.isfinite(float(v)) for v in vals):
        return ""
    w = WCS(naxis=2)
    # bundle stores 0-based pixels (wcs.all_world2pix(..., 0)); FITS CRPIX is 1-based
    w.wcs.crpix = [float(gal_xpix) + 1.0, float(gal_ypix) + 1.0]
    w.wcs.crval = [float(gal_ra), float(gal_dec)]
    w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    s = float(pixscale) / 3600.0
    w.wcs.cd = [[-s, 0.0], [0.0, s]]   # RA decreases to the right (E-left), Dec up
    return w.to_header().tostring()


def _index_dtype():
    import h5py
    return np.dtype([
        ("targetid", "i8"),
        ("brickname", h5py.string_dtype()),
        ("box_size", "i4"),
        ("n_members_final", "i4"),
        ("mag_g", "f4"),
        ("mag_r", "f4"),
        ("mag_z", "f4"),
        ("verdict", h5py.string_dtype()),
    ])


def main(argv=None):
    import h5py

    p = argparse.ArgumentParser(
        description="Apply VI decisions to a bundle -> final galaxy model cubes.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--bundle", required=True, help="bundle.h5 from build_bundle.py")
    p.add_argument("--decisions", required=True, help="decisions.csv from the VI server")
    p.add_argument("--out", required=True, help="output final-cubes .h5")
    p.add_argument("--include-unsure", action="store_true",
                   help="also write verdict=='unsure' objects")
    p.add_argument("--include-undecided", action="store_true",
                   help="also write objects with no verdict (their edits, if any, "
                        "are applied; otherwise the automatic initial membership)")
    p.add_argument("--float16", action="store_true",
                   help="store cubes as float16 (halves the file; default float32)")
    args = p.parse_args(argv)

    decisions = _read_decisions(args.decisions)

    out_parent = os.path.dirname(os.path.abspath(args.out))
    os.makedirs(out_parent, exist_ok=True)
    out_tmp = args.out + ".tmp"
    if os.path.exists(out_tmp):
        os.remove(out_tmp)

    dtype = np.float16 if args.float16 else np.float32
    kw = dict(compression="gzip", compression_opts=4)
    index_rows = []
    n_by_verdict = {}
    n_empty = 0
    written_tgids = set()
    t0 = time.time()

    with h5py.File(args.bundle, "r") as src, h5py.File(out_tmp, "w") as out:
        index = src["index"][:]
        for rec in index:
            tgid = int(rec["targetid"])
            dec = decisions.get(tgid)
            verdict = (dec.get("verdict", "") or "") if dec else ""

            if verdict == "accept":
                pass
            elif verdict == "unsure" and args.include_unsure:
                pass
            elif verdict in ("", None) and args.include_undecided:
                pass
            else:
                continue                       # rejected / not requested / not VI'd

            g = src[str(tgid)]
            removed = _split_ids(dec.get("removed_comp_ids", "")) if dec else []
            added = _split_ids(dec.get("added_comp_ids", "")) if dec else []
            cube, members, mags, lsb_in = _compose_galaxy_cube(g, removed, added)
            if not members:
                n_empty += 1

            brickname = _to_str(g.attrs.get("BRICKNAME", ""))
            box_size = int(g.attrs.get("box_size", cube.shape[-1]))
            gal_ra = float(g.attrs.get("gal_ra", np.nan))
            gal_dec = float(g.attrs.get("gal_dec", np.nan))
            gal_xpix = float(g.attrs.get("gal_xpix", np.nan))
            gal_ypix = float(g.attrs.get("gal_ypix", np.nan))

            og = out.create_group(str(tgid))
            og.create_dataset("galaxy_cube", data=cube.astype(dtype), **kw)
            og.attrs["TARGETID"] = tgid
            og.attrs["BRICKNAME"] = brickname
            og.attrs["box_size"] = box_size
            og.attrs["gal_ra"] = gal_ra
            og.attrs["gal_dec"] = gal_dec
            og.attrs["gal_xpix"] = gal_xpix
            og.attrs["gal_ypix"] = gal_ypix
            og.attrs["pixscale"] = PIXSCALE
            og.attrs["wcs_header"] = _reconstruct_wcs_header(
                gal_ra, gal_dec, gal_xpix, gal_ypix)
            og.attrs["verdict"] = verdict
            og.attrs["bad_fit"] = bool(dec.get("bad_fit", False)) if dec else False
            og.attrs["lsb_in_galaxy"] = bool(lsb_in)
            og.attrs["comment"] = str(dec.get("comment", "") or "") if dec else ""
            og.attrs["n_members_final"] = len(members)
            og.attrs["member_comp_ids"] = np.asarray(members, dtype=np.int64)
            og.attrs["removed_comp_ids"] = np.asarray(removed, dtype=np.int64)
            og.attrs["added_comp_ids"] = np.asarray(added, dtype=np.int64)
            og.attrs["mag_g"], og.attrs["mag_r"], og.attrs["mag_z"] = \
                [float(m) for m in mags]
            og.attrs["inspector"] = str(dec.get("inspector", "") or "") if dec else ""
            og.attrs["decision_timestamp"] = (str(dec.get("timestamp", "") or "")
                                              if dec else "")
            og.attrs["created"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

            index_rows.append((tgid, brickname, box_size, len(members),
                               mags[0], mags[1], mags[2], verdict))
            written_tgids.add(tgid)
            n_by_verdict[verdict or "undecided"] = \
                n_by_verdict.get(verdict or "undecided", 0) + 1
            n_written = len(index_rows)
            if (n_written % 100) == 0:
                print("  {} written ({:.0f}s)".format(n_written, time.time() - t0),
                      flush=True)

        arr = np.array(index_rows, dtype=_index_dtype())
        out.create_dataset("index", data=arr)
        out.attrs["n_objects"] = len(index_rows)
        out.attrs["source_bundle"] = os.path.abspath(args.bundle)
        out.attrs["source_decisions"] = os.path.abspath(args.decisions)
        out.attrs["n_by_verdict"] = str(n_by_verdict)
        out.attrs["created"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    os.replace(out_tmp, args.out)

    # Report decisions that never matched a bundle object (would be silent loss).
    requested = {t for t, d in decisions.items()
                 if (d.get("verdict", "") or "") == "accept"
                 or (args.include_unsure and (d.get("verdict", "") or "") == "unsure")
                 or (args.include_undecided and not (d.get("verdict", "") or ""))}
    missing = sorted(requested - written_tgids)

    print("Done: {} objects -> {} ({:.0f}s).".format(
        len(index_rows), args.out, time.time() - t0))
    print("  by verdict: {}".format(n_by_verdict))
    if n_empty:
        print("  WARNING: {} written object(s) had an EMPTY final member set "
              "(zero cube, NaN mags)".format(n_empty))
    if missing:
        print("  WARNING: {} requested decision(s) had no bundle group and were "
              "skipped: {}{}".format(
                  len(missing), missing[:10],
                  " ..." if len(missing) > 10 else ""))


if __name__ == "__main__":
    main()

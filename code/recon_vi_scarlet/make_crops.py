"""make_crops.py -- light-centered training crops from the final science cubes.

Reads the STAGE-4 final galaxy cubes (apply_decisions.py output) and, per object,
recenters on the light-weighted centroid and crops a fixed box (default 128x128).
One crop per object -- NO augmentation. Input file is never modified; READ only.

For each object the light center is the flux-weighted centroid of the total (grz-
summed) model cube, with negative model pixels clipped to 0. The crop origin is
snapped to the nearest integer and CLAMPED so the box stays fully on the frame
(recorded via `clamped`); no zero-padding. Optionally use a windowed/iterative
centroid (`--window R`) to be robust to off-center member flux.

The light center is also recorded as sky coordinates and as pixel coordinates in
the ORIGINAL cutout frame (the full box_size frame of the source cube, i.e. the
~350x350 cutout), so you can re-download a matching cutout from the real data at
exactly the light-weighted center:

    light_center_xpix/ypix   0-based pixels in the original cutout frame (== the
                             centroid the crop is centered on)
    light_center_ra/dec      those pixels pushed through the object's stored TAN
                             wcs_header (NaN if the object had no usable WCS)

Output is a stacked tensor, one row per object, keyed for TARGETID matching::

    /images      (N, 3, size, size) float32   -- N = n_objects (one crop each)
    /targetid    (N,)  int64                  -- source TARGETID per crop (join key)
    /index       (N,)  compound               -- per-crop provenance:
                 targetid,
                 light_center_ra, light_center_dec       (deg; original frame),
                 light_center_xpix, light_center_ypix    (0-based px; original frame),
                 crop_origin_x, crop_origin_y            (box top-left, original frame),
                 clamped,
                 mag_g/r/z                                (carried from the source object)
    top-level attrs: n_crops, n_objects, crop_size, recenter,
                     source_file, created.

Needs numpy/h5py (+ astropy for the pixel->sky conversion) -- no scarlet, no NERSC.

Usage::

    python -m recon_vi_scarlet.make_crops \
        --in  scarlet_final_cubes.h5 \
        --out scarlet_crops128.h5 [--crop-size 128] [--window 0]
"""

import os
import time
import argparse

import numpy as np


def _to_str(v):
    if isinstance(v, bytes):
        return v.decode("utf-8", "replace")
    return str(v)


def light_center(cube, tx=None, ty=None, window=0, iters=3):
    """Flux-weighted centroid of the grz-summed cube (negatives clipped to 0).

    window==0 -> global centroid over the whole frame. window>0 -> iterative
    centroid within radius `window` px, seeded at (tx,ty) (falls back to frame
    centre if no target given). Returns (cx, cy) in pixel coords; frame centre if
    there is no positive flux."""
    S = cube.shape[-1]
    w = np.clip(cube.sum(axis=0), 0.0, None)
    ys, xs = np.mgrid[0:S, 0:S]
    if w.sum() <= 0:
        return (S - 1) / 2.0, (S - 1) / 2.0
    if window and window > 0:
        cx = float(tx) if tx is not None else (S - 1) / 2.0
        cy = float(ty) if ty is not None else (S - 1) / 2.0
        for _ in range(iters):
            m = ((xs - cx) ** 2 + (ys - cy) ** 2) <= float(window) ** 2
            wv = w * m
            tot = wv.sum()
            if tot <= 0:
                break
            cx = float((wv * xs).sum() / tot)
            cy = float((wv * ys).sum() / tot)
        return cx, cy
    tot = w.sum()
    return float((w * xs).sum() / tot), float((w * ys).sum() / tot)


def crop_box(cube, cx, cy, size):
    """Crop `size`x`size` centred on (cx,cy), origin snapped to nearest int and
    clamped so the box stays on the frame. Returns (crop (3,size,size) contiguous
    float32, x0, y0, clamped_bool)."""
    S = cube.shape[-1]
    half = size // 2
    x_want, y_want = int(round(cx)) - half, int(round(cy)) - half
    x0 = min(max(x_want, 0), S - size)
    y0 = min(max(y_want, 0), S - size)
    clamped = (x0 != x_want) or (y0 != y_want)
    crop = np.ascontiguousarray(cube[:, y0:y0 + size, x0:x0 + size], dtype=np.float32)
    return crop, x0, y0, clamped


def light_center_world(wcs_header, cx, cy):
    """Push the 0-based pixel (cx,cy) through the object's stored TAN wcs_header to
    (ra,dec) in deg. Returns (nan,nan) if there is no usable header. The header is
    the FITS string apply_decisions wrote from the (gal_ra,gal_dec)<->(gal_xpix,
    gal_ypix) anchor, so the same 0-based convention applies here."""
    hdr_str = _to_str(wcs_header or "")
    if not hdr_str.strip():
        return float("nan"), float("nan")
    try:
        from astropy.io import fits
        from astropy.wcs import WCS
        w = WCS(fits.Header.fromstring(hdr_str))
        ra, dec = w.all_pix2world(float(cx), float(cy), 0)
        return float(ra), float(dec)
    except Exception:
        return float("nan"), float("nan")


def _index_dtype():
    return np.dtype([
        ("targetid", "i8"),
        ("light_center_ra", "f8"),
        ("light_center_dec", "f8"),
        ("light_center_xpix", "f4"),
        ("light_center_ypix", "f4"),
        ("crop_origin_x", "i4"),
        ("crop_origin_y", "i4"),
        ("clamped", "i1"),
        ("mag_g", "f4"),
        ("mag_r", "f4"),
        ("mag_z", "f4"),
    ])


def _f(attrs, key):
    try:
        return float(attrs[key])
    except (KeyError, TypeError, ValueError):
        return float("nan")


def main(argv=None):
    import h5py

    p = argparse.ArgumentParser(
        description="Light-centered crops (one per object) from the final science cubes.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--in", dest="inp", required=True,
                   help="scarlet_final_cubes.h5 from apply_decisions.py (read-only)")
    p.add_argument("--out", required=True, help="output crops .h5")
    p.add_argument("--crop-size", type=int, default=128, help="crop box side length")
    p.add_argument("--window", type=int, default=0,
                   help="centroid window radius in px (0 = global centroid)")
    args = p.parse_args(argv)

    size = int(args.crop_size)

    out_parent = os.path.dirname(os.path.abspath(args.out))
    os.makedirs(out_parent, exist_ok=True)
    out_tmp = args.out + ".tmp"
    if os.path.exists(out_tmp):
        os.remove(out_tmp)

    t0 = time.time()
    n_clamped = 0

    with h5py.File(args.inp, "r") as src, h5py.File(out_tmp, "w") as out:
        # deterministic object order = the source /index order
        targetids = [int(t) for t in src["index"]["targetid"]]
        N = len(targetids)

        images = out.create_dataset(
            "images", shape=(N, 3, size, size), dtype="f4",
            chunks=(1, 3, size, size), compression="gzip", compression_opts=4)
        tid_arr = np.empty(N, dtype=np.int64)
        index = np.empty(N, dtype=_index_dtype())

        for row, tgid in enumerate(targetids):
            g = src[str(tgid)]
            cube = np.asarray(g["galaxy_cube"][:], dtype=np.float64)
            S = cube.shape[-1]
            if S < size:
                raise ValueError("object {} cube {}px smaller than crop {}px"
                                 .format(tgid, S, size))
            tx, ty = _f(g.attrs, "gal_xpix"), _f(g.attrs, "gal_ypix")
            cx, cy = light_center(cube, tx, ty, window=args.window)
            ra, dec = light_center_world(g.attrs.get("wcs_header", ""), cx, cy)
            crop, x0, y0, clamped = crop_box(cube, cx, cy, size)
            if clamped:
                n_clamped += 1
            mg, mr, mz = _f(g.attrs, "mag_g"), _f(g.attrs, "mag_r"), _f(g.attrs, "mag_z")

            images[row] = crop
            tid_arr[row] = tgid
            index[row] = (tgid, ra, dec, cx, cy, x0, y0,
                          1 if clamped else 0, mg, mr, mz)

            if ((row + 1) % 200) == 0:
                print("  {}/{} objects ({:.0f}s)".format(
                    row + 1, N, time.time() - t0), flush=True)

        out.create_dataset("targetid", data=tid_arr)
        out.create_dataset("index", data=index)
        out.attrs["n_crops"] = N
        out.attrs["n_objects"] = N
        out.attrs["crop_size"] = size
        out.attrs["recenter"] = ("global_centroid" if not args.window
                                 else "windowed_centroid_R{}".format(args.window))
        out.attrs["source_file"] = os.path.abspath(args.inp)
        out.attrs["created"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    os.replace(out_tmp, args.out)
    print("Done: {} objects -> {} crops of {}x{} -> {} ({:.0f}s).".format(
        N, N, size, size, args.out, time.time() - t0))
    print("  edge-clamped crops (off-center to stay on frame): {}".format(n_clamped))


if __name__ == "__main__":
    main()

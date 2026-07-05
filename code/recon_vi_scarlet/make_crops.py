"""make_crops.py -- augmented 128x128 training crops from the final science cubes.

Reads the STAGE-4 final galaxy cubes (apply_decisions.py output) and, per object,
recenters on the light-weighted centroid, crops a fixed box (default 128x128), and
emits the flip augmentations. Input file is never modified; this only READS it.

For each object the light center is the flux-weighted centroid of the total (grz-
summed) model cube, with negative model pixels clipped to 0. The crop origin is
snapped to the nearest integer and CLAMPED so the box stays fully on the frame
(recorded via `clamped`); no zero-padding. Optionally use a windowed/iterative
centroid (`--window R`) to be robust to off-center member flux.

Augmentations per object (all pure per-band spatial reflections, no resampling):
    orig      the crop as-is
    flip_h    left-right mirror   (reverse x / columns)
    flip_v    up-down    mirror   (reverse y / rows)
    flip_hv   both                (== 180 deg rotation)

Output is a FLAT stacked tensor keyed for easy TARGETID matching::

    /images      (N, 3, size, size) float32   -- N = n_objects * n_variants
    /targetid    (N,)  int64                  -- source TARGETID per crop (join key)
    /index       (N,)  compound               -- full per-crop provenance:
                 targetid, variant, crop_center_x/y (light centroid, orig frame),
                 crop_origin_x/y (box top-left, orig frame, post-clamp), clamped,
                 mag_g/r/z (carried from the source object)
    top-level attrs: n_crops, n_objects, n_variants, crop_size, recenter,
                     variant_order, flip_definitions, source_file, created.

Row order is grouped by object with variants in a fixed order, so images can be
reshaped to (n_objects, n_variants, 3, size, size) if desired.

Pure numpy/h5py -- no scarlet, no astropy, no NERSC.

Usage::

    python -m recon_vi_scarlet.make_crops \
        --in  scarlet_final_cubes.h5 \
        --out scarlet_crops128_aug.h5 [--crop-size 128] [--window 0]
"""

import os
import time
import argparse

import numpy as np

# Fixed augmentation order (row layout depends on this).
VARIANTS = ("orig", "flip_h", "flip_v", "flip_hv")
FLIP_DEFS = "flip_h=reverse-x(LR mirror); flip_v=reverse-y(UD mirror); flip_hv=both(180deg)"


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


def apply_variant(crop, variant):
    """Pure per-band spatial reflection. Returns a contiguous float32 array."""
    if variant == "orig":
        out = crop
    elif variant == "flip_h":
        out = crop[:, :, ::-1]
    elif variant == "flip_v":
        out = crop[:, ::-1, :]
    elif variant == "flip_hv":
        out = crop[:, ::-1, ::-1]
    else:
        raise ValueError("unknown variant {!r}".format(variant))
    return np.ascontiguousarray(out, dtype=np.float32)


def _index_dtype():
    import h5py
    sd = h5py.string_dtype()
    return np.dtype([
        ("targetid", "i8"),
        ("variant", sd),
        ("crop_center_x", "f4"),
        ("crop_center_y", "f4"),
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
        description="Augmented 128x128 crops from the final science cubes.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--in", dest="inp", required=True,
                   help="scarlet_final_cubes.h5 from apply_decisions.py (read-only)")
    p.add_argument("--out", required=True, help="output augmented-crops .h5")
    p.add_argument("--crop-size", type=int, default=128, help="crop box side length")
    p.add_argument("--window", type=int, default=0,
                   help="centroid window radius in px (0 = global centroid)")
    args = p.parse_args(argv)

    size = int(args.crop_size)
    n_var = len(VARIANTS)

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
        n_obj = len(targetids)
        N = n_obj * n_var

        images = out.create_dataset(
            "images", shape=(N, 3, size, size), dtype="f4",
            chunks=(n_var, 3, size, size), compression="gzip", compression_opts=4)
        tid_arr = np.empty(N, dtype=np.int64)
        index = np.empty(N, dtype=_index_dtype())

        row = 0
        for j, tgid in enumerate(targetids):
            g = src[str(tgid)]
            cube = np.asarray(g["galaxy_cube"][:], dtype=np.float64)
            S = cube.shape[-1]
            if S < size:
                raise ValueError("object {} cube {}px smaller than crop {}px"
                                 .format(tgid, S, size))
            tx, ty = _f(g.attrs, "gal_xpix"), _f(g.attrs, "gal_ypix")
            cx, cy = light_center(cube, tx, ty, window=args.window)
            crop, x0, y0, clamped = crop_box(cube, cx, cy, size)
            if clamped:
                n_clamped += 1
            mg, mr, mz = _f(g.attrs, "mag_g"), _f(g.attrs, "mag_r"), _f(g.attrs, "mag_z")
            for v in VARIANTS:
                images[row] = apply_variant(crop, v)
                tid_arr[row] = tgid
                index[row] = (tgid, v, cx, cy, x0, y0, 1 if clamped else 0, mg, mr, mz)
                row += 1
            if ((j + 1) % 200) == 0:
                print("  {}/{} objects ({} crops, {:.0f}s)".format(
                    j + 1, n_obj, row, time.time() - t0), flush=True)

        out.create_dataset("targetid", data=tid_arr)
        out.create_dataset("index", data=index)
        out.attrs["n_crops"] = N
        out.attrs["n_objects"] = n_obj
        out.attrs["n_variants"] = n_var
        out.attrs["crop_size"] = size
        out.attrs["recenter"] = ("global_centroid" if not args.window
                                 else "windowed_centroid_R{}".format(args.window))
        out.attrs["variant_order"] = ";".join(VARIANTS)
        out.attrs["flip_definitions"] = FLIP_DEFS
        out.attrs["source_file"] = os.path.abspath(args.inp)
        out.attrs["created"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    os.replace(out_tmp, args.out)
    print("Done: {} objects x {} variants = {} crops of {}x{} -> {} ({:.0f}s).".format(
        n_obj, n_var, N, size, size, args.out, time.time() - t0))
    print("  edge-clamped crops (off-center to stay on frame): {}".format(n_clamped))


if __name__ == "__main__":
    main()

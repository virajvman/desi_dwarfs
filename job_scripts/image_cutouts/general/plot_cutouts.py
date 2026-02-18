#!/usr/bin/env python3

"""
plot_cutouts.py

Validation utility for HDF5 cutout datasets produced by consolidate_to_h5.py.
Shows a multi-panel grid of random cutouts with metadata in each panel title,
so you can visually verify that images match their catalog properties.

Usage:
  python3 plot_cutouts.py \
      --h5-path /path/to/galaxy_images.h5 \
      --n-sample 8 --nrows 2 --ncols 4 \
      --title-cols "Z,MAG_R,IS_DWARF" \
      --output validation_grid.png

Can also be imported and used programmatically:

  from plot_cutouts import read_h5_subset, sdss_rgb, plot_cutout_grid
  data = read_h5_subset("file.h5", n=8, cols=["Z","MAG_R"])
  fig = plot_cutout_grid(data)
  fig.savefig("grid.png")
"""

import os
import argparse
import numpy as np


# -----------------------------------------------------------------------
# Color mapping
# -----------------------------------------------------------------------

def sdss_rgb(imgs, bands=("g", "r", "z"),
             scales=None, m=0.03, Q=20):
    """Convert a (C, H, W) multi-band image to an (H, W, 3) RGB array.

    Uses an asinh stretch following the SDSS imaging convention.

    Parameters
    ----------
    imgs : array-like, shape (C, H, W)
        Multi-band image (one 2-D plane per band).
    bands : sequence of str
        Band names corresponding to each channel of *imgs*.
    scales : dict or None
        Mapping of band name -> (RGB plane index, intensity scale).
        Defaults to  g:(2, 6.0), r:(1, 3.4), z:(0, 2.2).
    m : float
        Softening parameter added before the stretch.
    Q : float
        Asinh stretch strength.

    Returns
    -------
    rgb : ndarray, shape (H, W, 3), float32 in [0, 1]
    """
    default_scales = {"g": (2, 6.0), "r": (1, 3.4), "z": (0, 2.2)}
    if scales is not None:
        default_scales.update(scales)
    rgbscales = default_scales

    I = np.zeros_like(imgs[0], dtype=np.float64)
    for img, band in zip(imgs, bands):
        _, scale = rgbscales[band]
        I += np.maximum(0, img * scale + m)
    I /= len(bands)

    fI = np.arcsinh(Q * I) / np.sqrt(Q)
    I += (I == 0.0) * 1e-6

    H, W = I.shape
    rgb = np.zeros((H, W, 3), dtype=np.float32)
    for img, band in zip(imgs, bands):
        plane, scale = rgbscales[band]
        rgb[:, :, plane] = (img * scale + m) * fI / I

    return np.clip(rgb, 0, 1)


# -----------------------------------------------------------------------
# HDF5 reader
# -----------------------------------------------------------------------

def read_h5_subset(h5_path, n=8, seed=42, cols=None, indices=None):
    """Read a random (or specified) subset from an HDF5 cutout dataset.

    Parameters
    ----------
    h5_path : str
        Path to the HDF5 file produced by consolidate_to_h5.py.
    n : int
        Number of objects to sample (ignored if *indices* is given).
    seed : int
        Random seed for reproducibility.
    cols : list of str or None
        Extra dataset names to read (e.g. ["Z", "MAG_R", "IS_DWARF"]).
        TARGETID, RA, DEC are always included.
    indices : array-like of int or None
        If provided, read exactly these row indices instead of random sampling.

    Returns
    -------
    dict with keys:
        "images"   : ndarray (n, C, H, W)
        "TARGETID" : ndarray (n,)
        "RA"       : ndarray (n,)
        "DEC"      : ndarray (n,)
        "indices"  : ndarray (n,)  -- row indices that were sampled
        plus any extra columns requested via *cols*.
    """
    import h5py

    if cols is None:
        cols = []

    with h5py.File(h5_path, "r") as f:
        N = f["images"].shape[0]

        if indices is not None:
            inds = np.sort(np.asarray(indices, dtype=int))
        else:
            rng = np.random.default_rng(seed)
            if n > N:
                raise ValueError(f"n={n} exceeds dataset size N={N}")
            inds = np.sort(rng.choice(N, size=n, replace=False))

        result = {
            "images": f["images"][inds],
            "TARGETID": f["TARGETID"][inds],
            "RA": f["RA"][inds],
            "DEC": f["DEC"][inds],
            "indices": inds,
        }

        for col in cols:
            if col in ("images", "TARGETID", "RA", "DEC", "indices"):
                continue
            if col in f:
                result[col] = f[col][inds]
            else:
                print(f"WARNING: column '{col}' not found in {h5_path}, skipping")

    return result


# -----------------------------------------------------------------------
# Plotting
# -----------------------------------------------------------------------

def plot_cutout_grid(data, nrows=2, ncols=4, title_cols=None,
                     bands=("g", "r", "z"), figsize=None,
                     show_mask=False, mask_alpha=0.35):
    """Plot a grid of RGB cutouts with metadata titles.

    Parameters
    ----------
    data : dict
        Output of :func:`read_h5_subset`.
    nrows, ncols : int
        Grid dimensions.  nrows * ncols should equal len(data["images"]).
    title_cols : list of str or None
        Extra column names to display in each panel title.
        RA and DEC are always shown.
    bands : tuple of str
        Band names for the RGB conversion.
    figsize : tuple or None
        Figure size in inches.  Defaults to (3.5 * ncols, 4 * nrows).
    show_mask : bool
        If True and ``data`` contains a ``"binary_mask"`` key, overlay
        masked pixels (value 0) with a semi-transparent red tint.
    mask_alpha : float
        Opacity of the mask overlay (0 = invisible, 1 = opaque).

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    import matplotlib
    if os.environ.get("DISPLAY") is None and matplotlib.get_backend() != "agg":
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if title_cols is None:
        title_cols = []

    images = data["images"]
    n_panels = nrows * ncols
    n_available = len(images)
    if n_panels > n_available:
        raise ValueError(
            f"Grid has {n_panels} panels but only {n_available} images in data"
        )

    has_mask = show_mask and "binary_mask" in data

    if figsize is None:
        figsize = (3.5 * ncols, 4.0 * nrows)

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    axes = np.atleast_2d(axes)

    for idx in range(n_panels):
        row, col = divmod(idx, ncols)
        ax = axes[row, col]

        rgb = sdss_rgb(images[idx], bands=bands)
        ax.imshow(rgb, origin="lower")

        if has_mask:
            bmask = data["binary_mask"][idx]
            overlay = np.zeros((*bmask.shape, 4), dtype=np.float32)
            masked_pixels = bmask == 0
            overlay[masked_pixels] = [1.0, 0.0, 0.0, mask_alpha]
            ax.imshow(overlay, origin="lower")

        ax.set_xticks([])
        ax.set_yticks([])

        ra_val = data["RA"][idx]
        dec_val = data["DEC"][idx]
        tid = data["TARGETID"][idx]

        title_parts = [f"RA={ra_val:.4f}  DEC={dec_val:.4f}"]

        extra_parts = []
        for cname in title_cols:
            if cname in data:
                val = data[cname][idx]
                if isinstance(val, (np.floating, float)):
                    extra_parts.append(f"{cname}={val:.3f}")
                elif isinstance(val, (np.bool_, bool)):
                    extra_parts.append(f"{cname}={bool(val)}")
                else:
                    extra_parts.append(f"{cname}={val}")
        if extra_parts:
            title_parts.append("  ".join(extra_parts))

        title_parts.append(f"TARGETID={tid}")
        title_parts.append(f"idx={data['indices'][idx]}")

        ax.set_title("\n".join(title_parts), fontsize=7, ha="center")

    fig.tight_layout()
    return fig


# -----------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Plot a grid of random cutouts from an HDF5 dataset for visual validation.",
    )
    parser.add_argument("--h5-path", type=str, required=True,
                        help="Path to the HDF5 cutout dataset.")
    parser.add_argument("--n-sample", type=int, default=8,
                        help="Number of cutouts to plot.")
    parser.add_argument("--nrows", type=int, default=2,
                        help="Number of grid rows.")
    parser.add_argument("--ncols", type=int, default=4,
                        help="Number of grid columns.")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for sampling.")
    parser.add_argument("--title-cols", type=str, default="",
                        help="Comma-separated column names to show in panel titles "
                             "(e.g. 'Z,MAG_R,IS_DWARF'). RA, DEC, TARGETID are always shown.")
    parser.add_argument("--output", type=str, default="cutout_grid.png",
                        help="Output image file path (png, pdf, jpg).")
    parser.add_argument("--dpi", type=int, default=200,
                        help="Output resolution.")
    parser.add_argument("--bands", type=str, default="g,r,z",
                        help="Comma-separated band names for RGB mapping.")
    parser.add_argument("--show-mask", action="store_true",
                        help="Overlay masked pixels (binary_mask==0) in red. "
                             "Requires the HDF5 to contain a 'binary_mask' dataset.")

    args = parser.parse_args()

    title_cols = [c.strip() for c in args.title_cols.split(",") if c.strip()]
    bands = tuple(b.strip() for b in args.bands.split(","))

    n_sample = args.n_sample
    expected = args.nrows * args.ncols
    if n_sample != expected:
        print(f"NOTE: adjusting n_sample from {n_sample} to {args.nrows}x{args.ncols}={expected}")
        n_sample = expected

    all_cols = list(set(title_cols))
    if args.show_mask and "binary_mask" not in all_cols:
        all_cols.append("binary_mask")

    data = read_h5_subset(
        args.h5_path, n=n_sample, seed=args.seed, cols=all_cols,
    )

    fig = plot_cutout_grid(
        data, nrows=args.nrows, ncols=args.ncols,
        title_cols=title_cols, bands=bands,
        show_mask=args.show_mask,
    )

    fig.savefig(args.output, dpi=args.dpi, bbox_inches="tight")
    print(f"Saved {args.output}  ({args.nrows}x{args.ncols} grid, seed={args.seed})")


if __name__ == "__main__":
    main()

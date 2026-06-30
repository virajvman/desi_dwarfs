"""
stack_mstar_haew_viz.py
=======================

Lightweight mean coadds for paper visualization: custom log M* windows crossed
with the three fixed H-alpha EW bins used in stack_mstar_haew_5pct.py.

Mass bins (half-open, log Msun):
  - (7.75, 8.25]
  - (7.50, 8.50]

EW bins (Angstroms, half-open):
  - ew_lt30:   EW <= 30
  - ew_30_100: 30 < EW <= 100
  - ew_gt100:  EW > 100

Sample: BGS_BRIGHT | BGS_FAINT | LOWZ
Catalog: load_catalog + H-alpha detection cuts (same as Product A)
Stack: Halpha-flux normalized mean coadd, full rest wavelength grid (0.2 A,
  3600-9800 A from desi_y1_dwarf_combine_deredshift_hires_noinvvar.h5)
Gate: skip cell if catalog N < 50

Output (FITS only, no FastSpecFit):
  .../stack_files/mstar_viz_haew/stack_ALL_mstar_{mlo}_{mhi}_{ewtoken}.fits

Usage:
    python stack_mstar_haew_viz.py
"""

import os
import sys

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_CODE_DIR = os.path.dirname(_THIS_DIR)
_STACK_DIR = os.path.join(_CODE_DIR, "stacking_analysis")
for _p in (_CODE_DIR, _STACK_DIR, _THIS_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np

from stack_explore import load_catalog, load_spectra
from stack_mstar_haew_5pct import (
    SAMPLES,
    Z_MIN_GLOBAL,
    Z_MAX_GLOBAL,
    EW_EDGES,
    EW_TOKENS,
    EW_LABELS,
    EW_STACK_NLIM,
    WAVE_MAX_VIZ,
    apply_halpha_detection_cuts,
    select_sample_ew_bin,
    bin_label,
    stack_one_mass_bin,
    write_single_row_fits,
)

SPECTRA_PATH = (
    "/pscratch/sd/v/virajvm/catalog_dr1_dwarfs/iron_spectra/spectra_files/"
    "desi_y1_dwarf_combine_deredshift_hires_noinvvar.h5"
)
STACK_PATH = (
    "/pscratch/sd/v/virajvm/catalog_dr1_dwarfs/iron_spectra/"
    "stack_files/mstar_viz_haew/"
)

# Custom mass bins for visualization (log M*, half-open (lo, hi]).
MSTAR_VIZ_BINS = [
    (7.75, 8.25),
    (7.50, 8.50),
]


def _wave_grid(spectra_data):
    """Return spectra_data trimmed to lambda < WAVE_MAX_VIZ if needed."""
    viz_mask = spectra_data["wave_rest"] < WAVE_MAX_VIZ
    if viz_mask.all():
        return spectra_data, spectra_data["wave_rest"]
    return {
        "targetid": spectra_data["targetid"],
        "wave_rest": spectra_data["wave_rest"][viz_mask],
        "flux": spectra_data["flux"][:, viz_mask],
        "flux_ivar": spectra_data["flux_ivar"][:, viz_mask],
    }, spectra_data["wave_rest"][viz_mask]


def main():
    os.makedirs(STACK_PATH, exist_ok=True)

    print("[1] Loading catalog ...")
    tot_cat = load_catalog()
    print(f"    Total galaxies after load_catalog cuts: {len(tot_cat)}")

    tot_cat_ew = apply_halpha_detection_cuts(tot_cat)
    print(f"    After H-alpha detection cuts: {len(tot_cat_ew)}")

    print("\n[2] Loading de-redshifted spectra ...")
    spectra_data = load_spectra(SPECTRA_PATH)
    print(f"    Total spectra loaded: {len(spectra_data['targetid'])}")
    spectra_data, wave = _wave_grid(spectra_data)
    print(f"    Wavelength grid: [{wave.min():.1f}, {wave.max():.1f}] A "
          f"({len(wave)} pixels, step={np.median(np.diff(wave)):.2f} A)")

    print(f"\n[3] Stacking: {len(MSTAR_VIZ_BINS)} mass bins x {len(EW_TOKENS)} EW bins")
    print(f"    Mass bins : {MSTAR_VIZ_BINS}")
    print(f"    EW bins   : {list(zip(EW_TOKENS, EW_LABELS))}")
    print(f"    N gate    : >= {EW_STACK_NLIM}")
    print(f"    Output dir: {STACK_PATH}")

    n_written = 0
    for mstar_min, mstar_max in MSTAR_VIZ_BINS:
        print(f"\n  --- log M* in ({mstar_min:.2f}, {mstar_max:.2f}] ---")

        ew_counts = []
        for j in range(len(EW_TOKENS)):
            sub = select_sample_ew_bin(
                tot_cat_ew, SAMPLES,
                Z_MIN_GLOBAL, Z_MAX_GLOBAL,
                mstar_min, mstar_max,
                EW_EDGES[j], EW_EDGES[j + 1],
            )
            ew_counts.append(len(sub))

        counts_str = ", ".join(
            f"{EW_TOKENS[j]}={ew_counts[j]}" for j in range(len(EW_TOKENS))
        )
        print(f"      EW counts: {counts_str}")

        for j, token in enumerate(EW_TOKENS):
            n_cat = ew_counts[j]
            if n_cat < EW_STACK_NLIM:
                print(f"      {token}: N={n_cat} < {EW_STACK_NLIM}; skipping")
                continue

            ew_min = EW_EDGES[j]
            ew_max = EW_EDGES[j + 1]
            sub_cat = select_sample_ew_bin(
                tot_cat_ew, SAMPLES,
                Z_MIN_GLOBAL, Z_MAX_GLOBAL,
                mstar_min, mstar_max,
                ew_min, ew_max,
            )
            label = bin_label(mstar_min, mstar_max, token)
            ew_hi_str = f"{ew_max:.1f}" if np.isfinite(ew_max) else "inf"
            print(f"\n    >> {token} | EW in ({ew_min:.1f}, {ew_hi_str}] | N={n_cat}")

            saved = stack_one_mass_bin(
                sub_cat, spectra_data, wave,
                mstar_min, mstar_max, label,
                out_dir=STACK_PATH, write_pkl=False,
            )
            if saved is None:
                continue

            write_single_row_fits(saved, wave, label, out_dir=STACK_PATH)
            n_written += 1

    print(f"\n[4] Done. Wrote {n_written} FITS files to {STACK_PATH}")


if __name__ == "__main__":
    main()

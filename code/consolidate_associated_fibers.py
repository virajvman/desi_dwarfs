"""
Consolidate associated fiber properties for the DESI dwarf galaxy catalog.

Two main functions:

1. symmetrize_and_group_associated_tgids(catalog)
   - Fixes asymmetric ASSOCIATED_TARGETIDS by computing connected components
     (Union-Find) among in-catalog TARGETIDs.
   - Pools non-catalog zpix TARGETIDs across each group.
   - Includes self in every row's ASSOCIATED_TARGETIDS.
   - Called inside finalize_main_hdu(), between find_associated_tgids() and
     get_dwarf_primary().

2. consolidate_associated_fiber_properties(cat_path)
   - Creates a FIBER_BASED extension preserving per-fiber values.
   - Propagates galaxy-level properties from the brightest MAG_R member
     (PROPERTY_SOURCE_TARGETID) to all group members in MAIN.
   - Consolidates photometry-dependent DWARF_MASKBIT bits while preserving
     fiber-specific bit 16 (wrong Redrock). Low continuum SNR for the mass
     pipeline is in MSTAR_MASKBIT and is copied with LOG_MSTAR_M24 from the
     property source.
   - Called as the very last pipeline step in consolidate_photometry.py.
"""

import os
import tempfile
import numpy as np
import astropy.io.fits as fits
from astropy.table import Table
from io import BytesIO
from tqdm import tqdm

from mass_and_photo_corrections import safe_read_table, make_catalog_unmasked


# ---------------------------------------------------------------------------
# Union-Find data structure
# ---------------------------------------------------------------------------

class UnionFind:
    """Weighted quick-union with path compression."""

    def __init__(self, elements):
        self.parent = {e: e for e in elements}
        self.rank = {e: 0 for e in elements}

    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1


# ---------------------------------------------------------------------------
# 1. Symmetrize and group ASSOCIATED_TARGETIDS
# ---------------------------------------------------------------------------

def symmetrize_and_group_associated_tgids(catalog):
    """
    Fix asymmetric associations and compute connected components.

    After this function:
    - ASSOCIATED_TARGETIDS includes self and all transitive in-catalog
      associates, plus the union of non-catalog zpix TGIDs from every
      group member.
    - Every member of a connected component has the same set of in-catalog
      TGIDs in ASSOCIATED_TARGETIDS.

    Parameters
    ----------
    catalog : astropy.table.Table
        Must already contain ASSOCIATED_TARGETIDS from find_associated_tgids().

    Returns
    -------
    catalog : astropy.table.Table
        Modified in-place with updated ASSOCIATED_TARGETIDS.
    """
    print("=" * 60)
    print("Symmetrizing and grouping ASSOCIATED_TARGETIDS")
    print("=" * 60)

    catalog_tids = set(int(t) for t in catalog["TARGETID"])
    tid_to_idx = {int(t): i for i, t in enumerate(catalog["TARGETID"])}
    n = len(catalog)

    # Partition each row's associations into in-catalog and non-catalog
    in_catalog_assoc = [set() for _ in range(n)]
    non_catalog_assoc = [set() for _ in range(n)]

    for i in range(n):
        for t in catalog["ASSOCIATED_TARGETIDS"][i]:
            t_int = int(t)
            if t_int in catalog_tids:
                in_catalog_assoc[i].add(t_int)
            else:
                non_catalog_assoc[i].add(t_int)

    # Build Union-Find over catalog TARGETIDs
    uf = UnionFind(catalog_tids)

    for i in range(n):
        my_tid = int(catalog["TARGETID"][i])
        for assoc_tid in in_catalog_assoc[i]:
            uf.union(my_tid, assoc_tid)

    # Group catalog TGIDs by their connected component root
    from collections import defaultdict
    component_members = defaultdict(set)
    for tid in catalog_tids:
        root = uf.find(tid)
        component_members[root].add(tid)

    # Pool non-catalog TGIDs within each component
    component_noncatalog = defaultdict(set)
    for i in range(n):
        my_tid = int(catalog["TARGETID"][i])
        root = uf.find(my_tid)
        component_noncatalog[root].update(non_catalog_assoc[i])

    # Track diagnostics
    n_multi_member_components = 0
    n_rows_changed = 0
    asymmetric_examples = []

    for root, members in component_members.items():
        if len(members) > 1:
            n_multi_member_components += 1

    # Update ASSOCIATED_TARGETIDS for each row
    new_assoc = [None] * n
    for i in range(n):
        my_tid = int(catalog["TARGETID"][i])
        root = uf.find(my_tid)

        full_group = component_members[root] | component_noncatalog[root]
        new_list = np.array(sorted(full_group), dtype=np.int64)

        old_set = set(int(t) for t in catalog["ASSOCIATED_TARGETIDS"][i])
        new_set = set(int(t) for t in new_list)

        if new_set != old_set:
            n_rows_changed += 1
            added = new_set - old_set - {my_tid}
            if added and len(asymmetric_examples) < 10:
                asymmetric_examples.append((my_tid, added))

        new_assoc[i] = new_list

    catalog["ASSOCIATED_TARGETIDS"] = new_assoc

    # Print diagnostics
    print(f"  Total catalog rows: {n}")
    print(f"  Connected components with >1 catalog member: {n_multi_member_components}")
    print(f"  Rows whose ASSOCIATED_TARGETIDS changed: {n_rows_changed}")

    if asymmetric_examples:
        print(f"\n  Examples of fixed asymmetric associations (TARGETID -> added TGIDs):")
        for my_tid, added in asymmetric_examples:
            added_str = ", ".join(str(t) for t in sorted(added))
            print(f"    TARGETID {my_tid}: gained [{added_str}]")
    else:
        print("  No asymmetric associations found.")

    print("=" * 60)
    return catalog


# ---------------------------------------------------------------------------
# 2. Consolidate associated fiber properties (final pipeline step)
# ---------------------------------------------------------------------------

COLUMNS_TO_CONSOLIDATE = [
    "MAG_G", "MAG_R", "MAG_Z",
    "R50_R",
    "RA", "DEC",
    "LOG_MSTAR_M24", "LOG_MSTAR_M24_ERR",
    "MSTAR_MASKBIT",
    "MAG_TYPE",
    "PHOTOMETRY_UPDATED",
    "SHAPE_PARAMS",
    "LUMI_DIST_MPC",
    "DIST_SOURCE",
]

# Bits 0-11, 13-15, and 17 depend on photometry and are inherited from the
# property source. Bits 12 (near SGA outskirts), 16 (wrong Redrock),
# 19 (junk spectrum: SNR<0 in >=2 arms), and 20 (suspect spectrum: negative
# smoothed continuum) are fiber-specific (depend on per-fiber RA/DEC and the
# per-fiber spectrum) and are kept from the row itself.
FIBER_BITS_MASK = np.int64((1 << 12) | (1 << 16) | (1 << 19) | (1 << 20))
PHOTO_BITS_MASK = np.int64((((1 << 16) - 1) | (1 << 17)) & ~FIBER_BITS_MASK)


def _find_property_source_per_group(catalog):
    """
    For each row, identify the property source: the in-catalog TARGETID
    with the brightest (minimum) MAG_R among all in-catalog members of
    ASSOCIATED_TARGETIDS.

    Returns
    -------
    property_source_tids : np.ndarray of int64
        One PROPERTY_SOURCE_TARGETID per row.
    """
    catalog_tids = set(int(t) for t in catalog["TARGETID"])
    tid_to_idx = {int(t): i for i, t in enumerate(catalog["TARGETID"])}
    mag_r = np.array(catalog["MAG_R"].data, dtype=float)

    property_source_tids = np.full(len(catalog), -1, dtype=np.int64)

    for i in tqdm(range(len(catalog)), desc="Identifying property sources"):
        my_tid = int(catalog["TARGETID"][i])

        in_catalog_members = []
        for t in catalog["ASSOCIATED_TARGETIDS"][i]:
            t_int = int(t)
            if t_int in tid_to_idx:
                in_catalog_members.append(t_int)

        if len(in_catalog_members) == 0:
            property_source_tids[i] = my_tid
            continue

        member_indices = [tid_to_idx[t] for t in in_catalog_members]
        member_mags = mag_r[member_indices]
        best_local_idx = np.argmin(member_mags)
        property_source_tids[i] = in_catalog_members[best_local_idx]

    return property_source_tids


def consolidate_associated_fiber_properties(cat_path):
    """
    Final pipeline step: create FIBER_BASED extension and consolidate
    galaxy-level properties in MAIN from the brightest MAG_R member of
    each associated group.

    Parameters
    ----------
    cat_path : str
        Path to the multi-extension FITS catalog.
    """
    print("=" * 60)
    print("Consolidating associated fiber properties")
    print("=" * 60)

    main_cat = safe_read_table(cat_path, hdu="MAIN")
    n = len(main_cat)

    # ── 1. Create FIBER_BASED as a snapshot before consolidation ──────
    print("Creating FIBER_BASED extension (pre-consolidation snapshot)...")
    fiber_based = main_cat.copy()
    cols_to_remove_from_fiber = []
    if "ASSOCIATED_TARGETIDS" in fiber_based.colnames:
        cols_to_remove_from_fiber.append("ASSOCIATED_TARGETIDS")
    if cols_to_remove_from_fiber:
        fiber_based.remove_columns(cols_to_remove_from_fiber)
    fiber_based = make_catalog_unmasked(fiber_based)

    # ── 2. Identify property source per group ─────────────────────────
    print("Identifying property source (brightest MAG_R) per group...")
    property_source_tids = _find_property_source_per_group(main_cat)
    main_cat["PROPERTY_SOURCE_TARGETID"] = property_source_tids

    tid_to_idx = {int(t): i for i, t in enumerate(main_cat["TARGETID"])}

    # ── 3. Propagate galaxy-level columns ─────────────────────────────
    cols_present = [c for c in COLUMNS_TO_CONSOLIDATE if c in main_cat.colnames]
    cols_missing = [c for c in COLUMNS_TO_CONSOLIDATE if c not in main_cat.colnames]
    if cols_missing:
        print(f"  Skipping columns not yet in catalog: {cols_missing}")
    print(f"  Consolidating columns: {cols_present}")

    n_rows_updated = 0
    n_maskbit_changed = 0
    n_source_differs_primary = 0
    mag_r_diffs = []

    dwarf_maskbits = np.array(main_cat["DWARF_MASKBIT"].data, dtype=np.int64)

    for i in tqdm(range(n), desc="Propagating properties"):
        source_tid = int(property_source_tids[i])
        my_tid = int(main_cat["TARGETID"][i])

        if source_tid == my_tid:
            if "DWARF_PRIMARY_TARGETID" in main_cat.colnames:
                primary_tid = int(main_cat["DWARF_PRIMARY_TARGETID"][i])
                if source_tid != primary_tid:
                    n_source_differs_primary += 1
            continue

        if source_tid not in tid_to_idx:
            continue

        source_idx = tid_to_idx[source_tid]

        if "DWARF_PRIMARY_TARGETID" in main_cat.colnames:
            primary_tid = int(main_cat["DWARF_PRIMARY_TARGETID"][i])
            if source_tid != primary_tid:
                n_source_differs_primary += 1

        old_mag_r = float(main_cat["MAG_R"][i])
        new_mag_r = float(main_cat["MAG_R"][source_idx])
        if np.isfinite(old_mag_r) and np.isfinite(new_mag_r):
            mag_r_diffs.append(abs(old_mag_r - new_mag_r))

        for col in cols_present:
            main_cat[col][i] = main_cat[col][source_idx]

        # Consolidate DWARF_MASKBIT: photometry bits from source, fiber bits from self
        source_bits = np.int64(dwarf_maskbits[source_idx])
        row_bits = np.int64(dwarf_maskbits[i])
        new_bits = (source_bits & PHOTO_BITS_MASK) | (row_bits & FIBER_BITS_MASK)
        if new_bits != dwarf_maskbits[i]:
            n_maskbit_changed += 1
        dwarf_maskbits[i] = new_bits

        n_rows_updated += 1

    main_cat["DWARF_MASKBIT"] = dwarf_maskbits

    # ── 4. Diagnostics ────────────────────────────────────────────────
    n_groups = len(set(int(property_source_tids[i]) for i in range(n)))
    print(f"\n  Consolidation summary:")
    print(f"    Total rows: {n}")
    print(f"    Rows updated (non-self source): {n_rows_updated}")
    print(f"    Distinct property sources: {n_groups}")
    print(f"    Groups where PROPERTY_SOURCE != DWARF_PRIMARY: {n_source_differs_primary}")
    if mag_r_diffs:
        print(f"    MAG_R change for updated rows: "
              f"median={np.median(mag_r_diffs):.3f}, max={np.max(mag_r_diffs):.3f} mag")
    print(f"    Rows with DWARF_MASKBIT changed: {n_maskbit_changed}")

    # ── 5. Write updated MAIN + new FIBER_BASED extension ────────────
    print("\nWriting updated catalog...")

    main_cat = make_catalog_unmasked(main_cat)

    cat_abs = os.path.abspath(cat_path)
    cat_dir = os.path.dirname(cat_abs) or "."
    fd, tmp_path = tempfile.mkstemp(suffix=".fits", prefix="assoc_consol_", dir=cat_dir)
    os.close(fd)

    try:
        with fits.open(cat_abs, memmap=False) as hdul:
            hdu_names = [hdu.name for hdu in hdul]

            # Rebuild MAIN HDU (preserves VLA columns like ASSOCIATED_TARGETIDS)
            main_idx = hdu_names.index("MAIN")
            main_hdu_new = fits.table_to_hdu(main_cat)
            main_hdu_new.name = "MAIN"
            main_hdu_new.add_checksum()

            # Build FIBER_BASED HDU
            fiber_hdu = fits.table_to_hdu(fiber_based)
            fiber_hdu.name = "FIBER_BASED"
            fiber_hdu.add_checksum()

            new_hdus = []
            for i, hdu in enumerate(hdul):
                if i == main_idx:
                    new_hdus.append(main_hdu_new)
                else:
                    new_hdus.append(hdu.copy())

            new_hdus.append(fiber_hdu)

            new_hdul = fits.HDUList(new_hdus)
            new_hdul[0].add_checksum()
            new_hdul.writeto(tmp_path, overwrite=True)

        os.replace(tmp_path, cat_abs)

    except BaseException:
        if os.path.isfile(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass
        raise

    print(f"  Updated MAIN HDU and added FIBER_BASED extension to {cat_path}")
    print("=" * 60)

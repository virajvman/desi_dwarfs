"""
SSL chunked HDF5 sync with the dwarf catalog.

**Suggested order of operations (catalog update):**
(1) Run ``make_ssl_datasets`` with ``prune_h5_against_catalog`` and/or
``write_need_image_download_fits`` to prune chunks and write ``ssl_need_image_download.fits``.
(2) Download cutouts for those rows until every ``IMAGE_PATH`` exists.
(3) Set ``append_new_h5_chunks = True`` (and turn off full ``create_*``) to add only
missing TARGETIDs in new ``data_chunk_N.h5`` files.
"""
from __future__ import annotations

import glob
import os
import re
from typing import Any, Dict, Set

import h5py
import numpy as np
from astropy.table import Table, join, unique

H5_KEYS_1D = (
    "mag_g",
    "mag_r",
    "mag_z",
    "mstar",
    "star_dist",
    "redshift",
    "targetid",
)
H5_KEY_IMAGES = "images"


def matching_image_paths(target_cat: Table, img_table: Table) -> Tuple[Table, Table]:
    """
    Matched / unmatched rows between target_cat and img_table on TARGETID
    (img_table is de-duplicated on TARGETID first).
    """
    print(f"Initial catalog size = {len(target_cat)}")

    img_table_unique = unique(img_table, keys="TARGETID")

    matched_cat = join(
        target_cat,
        img_table_unique,
        keys="TARGETID",
        join_type="inner",
    )

    print(f"Objects with images = {len(matched_cat)}")

    unmatched_mask = ~np.isin(target_cat["TARGETID"], matched_cat["TARGETID"])
    unmatched_cat = target_cat[unmatched_mask]
    print(f"Objects without images = {len(unmatched_cat)}")

    return matched_cat, unmatched_cat


def scan_h5_targetid_union(h5_glob: str) -> Set[int]:
    """Union of all targetid values across chunk files (no cross-file duplicate checks)."""
    paths = sorted(_glob_h5_chunk_paths(h5_glob))
    out: Set[int] = set()
    for p in paths:
        with h5py.File(p, "r") as f:
            t = f["targetid"][:]
        out.update(int(x) for x in t.astype(np.int64))
    return out


def _glob_h5_chunk_paths(h5_glob: str) -> List[str]:
    return glob.glob(h5_glob)


def next_data_chunk_index(h5_glob: str) -> int:
    """
    Next index to use for data_chunk_{n}.h5: max existing n plus one; 0 if none.
    """
    paths = _glob_h5_chunk_paths(h5_glob)
    if not paths:
        return 0
    nums: List[int] = []
    for p in paths:
        m = re.search(r"data_chunk_(\d+)\.h5$", os.path.basename(p))
        if m:
            nums.append(int(m.group(1)))
    if not nums:
        return 0
    return max(nums) + 1


def _read_h5_for_prune(
    path: str,
) -> Optional[Dict[str, np.ndarray]]:
    """Read all top-level array datasets; return None on missing keys."""
    with h5py.File(path, "r") as f:
        keys = set(f.keys())
        if H5_KEY_IMAGES not in keys or "targetid" not in keys:
            print(f"WARNING: {path} missing required datasets, skipping: {keys}")
            return None
        for k in H5_KEYS_1D:
            if k not in f:
                print(f"WARNING: {path} missing {k}, skipping")
                return None
        d: Dict[str, np.ndarray] = {}
        d[H5_KEY_IMAGES] = f[H5_KEY_IMAGES][...]
        for k in H5_KEYS_1D:
            d[k] = f[k][...]
    return d


def _write_h5_pruned(
    out_path: str, data: Dict[str, np.ndarray]
) -> None:
    n = data["targetid"].shape[0]
    pixel = 152
    if data[H5_KEY_IMAGES].shape[0] != n:
        raise ValueError("images length does not match targetid length")
    batch_size = 500
    with h5py.File(out_path, "w") as f:
        images = f.create_dataset(
            H5_KEY_IMAGES,
            shape=(n, 3, pixel, pixel),
            dtype=np.float32,
            chunks=(1, 3, pixel, pixel),
            compression="gzip",
            compression_opts=4,
        )
        for k in H5_KEYS_1D:
            if k == "targetid":
                f.create_dataset(k, data=data[k].astype(np.int64))
            else:
                f.create_dataset(k, data=data[k].astype(np.float32))
        for i in range(0, n, batch_size):
            j = min(i + batch_size, n)
            images[i:j] = data[H5_KEY_IMAGES][i:j].astype(np.float32, copy=False)


def prune_h5_against_allowed_ids(
    h5_glob: str,
    allowed_targetids: np.ndarray,
    dry_run: bool = False,
) -> Dict[str, Any]:
    """
    In each chunk file, keep only rows whose targetid is in allowed_targetids.
    Empty files are deleted. Writes use a temp file + os.replace.
    """
    allowed = np.unique(np.asarray(allowed_targetids, dtype=np.int64))
    paths = sorted(_glob_h5_chunk_paths(h5_glob))
    stats = {
        "n_files": len(paths),
        "n_deleted": 0,
        "n_rewritten": 0,
        "n_unchanged": 0,
        "n_rows_removed": 0,
    }
    for path in paths:
        data = _read_h5_for_prune(path)
        if data is None:
            continue
        tgid = data["targetid"].astype(np.int64)
        mask = np.isin(tgid, allowed)
        n_keep = int(mask.sum())
        n_tot = len(tgid)
        if n_keep == 0:
            print(f"Prune: delete empty (all stale): {path}")
            stats["n_rows_removed"] += n_tot
            if not dry_run:
                os.remove(path)
            stats["n_deleted"] += 1
            continue
        if n_keep == n_tot:
            stats["n_unchanged"] += 1
            continue
        stats["n_rows_removed"] += n_tot - n_keep
        print(f"Prune: {path}  keep {n_keep} / {n_tot}")
        new_data: Dict[str, np.ndarray] = {
            H5_KEY_IMAGES: data[H5_KEY_IMAGES][mask],
        }
        for k in H5_KEYS_1D:
            new_data[k] = data[k][mask]
        tmp = path + ".tmp"
        if dry_run:
            stats["n_rewritten"] += 1
            continue
        _write_h5_pruned(tmp, new_data)
        os.replace(tmp, path)
        stats["n_rewritten"] += 1
    return stats


def build_table_need_image_download(
    data_cat: Table,
    in_h5: Set[int],
    tot_temp: Table,
) -> Tuple[Table, Dict[str, int]]:
    """
    Catalog rows in data_cat not in in_h5, with IMAGE_PATH from tot_temp, where
    the image file is missing on disk.
    """
    if not len(data_cat):
        return Table(), {"n_in_catalog": 0, "n_missing_from_h5": 0, "n_with_file_on_disk": 0, "n_need_download": 0, "n_no_path_row": 0}

    tids = data_cat["TARGETID"]
    in_arr = np.fromiter(in_h5, dtype=np.int64) if in_h5 else np.array([], dtype=np.int64)
    miss_mask = ~np.isin(tids, in_arr)
    missing = data_cat[miss_mask]
    n_missing = len(missing)
    matched, _ = matching_image_paths(missing, tot_temp)

    n_no_path = int(n_missing - len(matched))
    exist_flags = [os.path.isfile(p) for p in matched["IMAGE_PATH"]]
    exist_arr = np.array(exist_flags, dtype=bool)
    n_with_file = int(exist_arr.sum())
    to_dl = matched[np.logical_not(exist_arr)]
    n_need = len(to_dl)
    print(
        f"Need-image-download summary: in_catalog={len(data_cat)}, missing_from_h5={n_missing}, "
        f"rows_with_path_match={len(matched)}, with_file_on_disk={n_with_file}, need_download={n_need}, no_path_in_join={n_no_path}"
    )
    meta = {
        "n_in_catalog": len(data_cat),
        "n_missing_from_h5": n_missing,
        "n_with_file_on_disk": n_with_file,
        "n_need_download": n_need,
        "n_no_path_row": n_no_path,
    }
    return to_dl, meta


def run_prune_and_need_image_report(
    h5_glob: str,
    data_cat: Table,
    tot_temp: Table,
    need_image_fits_path: str,
    do_prune: bool = True,
    write_need_fits: bool = True,
    dry_run: bool = False,
) -> Dict[str, Any]:
    """
    Optional prune, then re-scan in_h5, then build & write need-image FITS and return stats.
    """
    out: Dict[str, Any] = {"prune": None, "need_image": None, "in_h5_size": 0}
    allowed = np.asarray(data_cat["TARGETID"], dtype=np.int64)

    if do_prune:
        out["prune"] = prune_h5_against_allowed_ids(
            h5_glob, allowed, dry_run=dry_run
        )
        print(f"Prune stats: {out['prune']}")

    in_h5 = scan_h5_targetid_union(h5_glob)
    out["in_h5_size"] = len(in_h5)

    if write_need_fits and need_image_fits_path:
        t_need, need_meta = build_table_need_image_download(data_cat, in_h5, tot_temp)
        out["need_image"] = need_meta
        if len(t_need):
            t_need.write(need_image_fits_path, format="fits", overwrite=True)
            print(f"Wrote {len(t_need)} need-download rows to {need_image_fits_path}")
        else:
            if os.path.isfile(need_image_fits_path):
                os.remove(need_image_fits_path)
            print("No need-download rows; removed or skipped empty FITS.")
    return out

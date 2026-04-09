"""
Photometric correction functions for the dwarf galaxy catalog pipeline.

Loads NEBCORR delta-magnitude tables, matches by TARGETID, and applies the
full correction chain (BASS->DECam, nebular emission removal, DECam->SDSS
filter conversion, k-correction) to produce SDSS z=0 continuum-only
photometry used for stellar mass estimation.
"""

import numpy as np
import os
import tempfile
import multiprocessing as mp
import astropy.io.fits as fits
from astropy.table import Table, vstack
import h5py
from tqdm import tqdm
from desispec.interpolation import resample_flux
from fastspec_funcs import measure_photo_batch, get_fastspecfit_path
from desi_lowz_funcs import get_stellar_mass_mia

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

NEBCORR_DEFAULT_FOLDER = "/pscratch/sd/v/virajvm/catalog_dr1_dwarfs"

NEBCORR_INT_V2_BASENAMES = (
    "iron_lowz_filter_zsucc_zrr03_INT_V2_NEBCORR.fits",
    "iron_bgs_bright_filter_zsucc_zrr02_allfracflux_INT_V2_NEBCORR.fits",
    "iron_bgs_faint_filter_zsucc_zrr03_allfracflux_INT_V2_NEBCORR.fits",
    "iron_elg_filter_zsucc_zrr05_allfracflux_INT_V2_NEBCORR.fits",
)

FASTSPEC_DELTA_MAG_COLS = (
    "DELTA_MAG_G_BASS2DECAM",
    "DELTA_MAG_R_BASS2DECAM",
    "DELTA_MAG_G_NEB",
    "DELTA_MAG_R_NEB",
    "DELTA_MAG_G_DECAM2SDSS",
    "DELTA_MAG_R_DECAM2SDSS",
    "DELTA_MAG_G_KCORR",
    "DELTA_MAG_R_KCORR",
)

# ---------------------------------------------------------------------------
# Utility helpers (needed to avoid circular imports with consolidate_photometry)
# ---------------------------------------------------------------------------

def make_catalog_unmasked(cat):
    """
    Return a new Table where all MaskedColumns are replaced by regular ndarray columns.
    Masked entries are filled with appropriate default values.
    """
    new_cat = cat.copy()
    for col in new_cat.colnames:
        c = new_cat[col]
        if hasattr(c, "mask"):
            if np.issubdtype(c.dtype, np.floating):
                fill_val = np.nan
            elif np.issubdtype(c.dtype, np.integer):
                fill_val = -99
            elif np.issubdtype(c.dtype, np.bool_):
                fill_val = False
            elif c.dtype.kind in ('U', 'S', 'O'):
                fill_val = ""
            else:
                fill_val = 0
            new_cat[col] = np.asarray(c.filled(fill_val))
        else:
            new_cat[col] = np.asarray(c)

    return new_cat


def safe_read_table(*args, **kwargs):
    """Table.read wrapper that immediately strips all MaskedColumns."""
    return make_catalog_unmasked(Table.read(*args, **kwargs))


def safe_vstack(tables, **kwargs):
    """vstack wrapper that strips MaskedColumns introduced by stacking."""
    return make_catalog_unmasked(vstack(tables, **kwargs))


# ---------------------------------------------------------------------------
# NEBCORR loading helpers
# ---------------------------------------------------------------------------

def _load_nebcorr_is_south_lookup(
    save_folder=NEBCORR_DEFAULT_FOLDER,
    verbose=False,
):
    """Build TARGETID -> is_south from stacked INT_V2_NEBCORR tables (int keys)."""
    chunks = []
    for basename in NEBCORR_INT_V2_BASENAMES:
        path = os.path.join(save_folder, basename)
        if not os.path.exists(path):
            if verbose:
                raise ValueError(f"  WARNING: {path} not found, skipping")
            continue
        tab = safe_read_table(path)
        if "is_south" not in tab.colnames:
            if verbose:
                raise ValueError(f"  WARNING: {basename} has no is_south column, skipping")
            continue
        chunks.append(tab[["TARGETID", "is_south"]])
        if verbose:
            print(f"  Loaded is_south from {len(tab)} rows in {basename}")

    if len(chunks) == 0:
        return {}

    stacked = safe_vstack(chunks)
    _, unique_idx = np.unique(np.asarray(stacked["TARGETID"]), return_index=True)
    stacked = stacked[np.sort(unique_idx)]
    tids = np.asarray(stacked["TARGETID"])
    flags = np.asarray(stacked["is_south"], dtype=np.int64)
    return {int(t): int(s) for t, s in zip(tids, flags)}


def _bass2decam_apply_mask(catalog, nebcorr_folder=NEBCORR_DEFAULT_FOLDER):
    """float64 array: 1.0 where BASS->DECam applies (north, is_south==0), else 0.0."""
    n = len(catalog)

    lookup = _load_nebcorr_is_south_lookup(save_folder=nebcorr_folder, verbose=False)
    if len(lookup) == 0:
        print(
            "WARNING: _bass2decam_apply_mask: no NEBCORR is_south lookup; "
            "BASS2DECAM deltas skipped for all rows."
        )
        return np.zeros(n, dtype=np.float64)

    tids = np.asarray(catalog["TARGETID"].data)
    mask = np.zeros(n, dtype=np.float64)
    n_unmatched = 0
    for j in range(n):
        s = lookup.get(int(tids[j]))
        if s is None:
            n_unmatched += 1
        elif s == 0:
            mask[j] = 1.0
    if n_unmatched > 0:
        print(
            f"_bass2decam_apply_mask: {n_unmatched}/{n} TARGETIDs not in "
            f"NEBCORR is_south lookup (BASS2DECAM skipped for those rows)"
        )

    return mask


def _load_nebcorr_delta_mag_table(
    save_folder=NEBCORR_DEFAULT_FOLDER,
    verbose=True,
):
    """Stack TARGETID, is_south, and DELTA_MAG_* from INT_V2_NEBCORR (one row per TARGETID)."""
    chunks = []
    required = ("TARGETID", "is_south") + FASTSPEC_DELTA_MAG_COLS
    for basename in NEBCORR_INT_V2_BASENAMES:
        path = os.path.join(save_folder, basename)
        if not os.path.exists(path):
            if verbose:
                print(f"  WARNING: {path} not found, skipping")
            continue
        tab = safe_read_table(path)
        missing = [c for c in required if c not in tab.colnames]
        if missing:
            if verbose:
                print(f"  WARNING: {basename} missing columns {missing}, skipping")
            continue
        sub = tab[list(required)]
        chunks.append(sub)
        if verbose:
            print(f"  Loaded DELTA_MAG + is_south from {len(tab)} rows in {basename}")

    if len(chunks) == 0:
        return None

    stacked = safe_vstack(chunks)
    _, unique_idx = np.unique(np.asarray(stacked["TARGETID"]), return_index=True)
    stacked = stacked[np.sort(unique_idx)]
    if verbose:
        print(f"  Combined NEBCORR DELTA_MAG table: {len(stacked)} unique TARGETIDs")
    return stacked


# ---------------------------------------------------------------------------
# Core correction function
# ---------------------------------------------------------------------------

def _apply_delta_mag_corrections(
    catalog,
    mag_g_col="MAG_G",
    mag_r_col="MAG_R",
    nebcorr_folder=NEBCORR_DEFAULT_FOLDER,
):
    """Apply DELTA_MAG corrections to get SDSS z=0 continuum-only magnitudes.

    Loads the NEBCORR delta-mag table, matches every input catalog row by
    TARGETID, and applies the full correction chain:

        1. BASS -> DECam  (north only, is_south == 0)
        2. Nebular emission removal
        3. DECam -> SDSS filter conversion
        4. K-correction to z = 0

    Raises ValueError if any catalog TARGETID is missing from the NEBCORR
    table -- every object must have pre-computed correction terms.
    """
    mag_g = np.array(catalog[mag_g_col].data, dtype=float)
    mag_r = np.array(catalog[mag_r_col].data, dtype=float)
    n = len(catalog)

    delta_tab = _load_nebcorr_delta_mag_table(save_folder=nebcorr_folder, verbose=True)
    if delta_tab is None:
        raise ValueError("No NEBCORR tables found -- cannot apply corrections.")

    cat_tids = np.asarray(catalog["TARGETID"])
    neb_tids = np.asarray(delta_tab["TARGETID"])
    tid_to_row = {int(t): i for i, t in enumerate(neb_tids)}

    matched_rows = np.array([tid_to_row.get(int(t), -1) for t in cat_tids])
    n_unmatched = int(np.sum(matched_rows < 0))
    print(f"_apply_delta_mag_corrections: matched {n - n_unmatched}/{n} TARGETIDs to NEBCORR table")

    if n_unmatched > 0:
        unmatched_tids = cat_tids[matched_rows < 0]
        print(f"  First 10 unmatched TARGETIDs: {unmatched_tids[:10]}")
        raise ValueError(
            f"{n_unmatched}/{n} catalog TARGETIDs not found in NEBCORR table. "
            "All TARGETIDs must have correction terms."
        )

    matched_neb_tids = neb_tids[matched_rows]
    max_diff = np.max(np.abs(cat_tids - matched_neb_tids))
    print(f"  Sanity check: max |TARGETID_catalog - TARGETID_nebcorr| = {max_diff}")
    assert max_diff == 0, "TARGETID mismatch after matching!"

    is_south = np.asarray(delta_tab["is_south"], dtype=np.int64)[matched_rows]
    north = (is_south == 0).astype(np.float64)

    mag_g += np.asarray(delta_tab["DELTA_MAG_G_BASS2DECAM"], dtype=np.float64)[matched_rows] * north
    mag_r += np.asarray(delta_tab["DELTA_MAG_R_BASS2DECAM"], dtype=np.float64)[matched_rows] * north

    for dg, dr in [
        ("DELTA_MAG_G_NEB",        "DELTA_MAG_R_NEB"),
        ("DELTA_MAG_G_DECAM2SDSS", "DELTA_MAG_R_DECAM2SDSS"),
        ("DELTA_MAG_G_KCORR",      "DELTA_MAG_R_KCORR"),
    ]:
        mag_g += np.asarray(delta_tab[dg], dtype=np.float64)[matched_rows]
        mag_r += np.asarray(delta_tab[dr], dtype=np.float64)[matched_rows]

    return mag_g, mag_r


# ---------------------------------------------------------------------------
# FASTSPEC HDU delta-mag writer
# ---------------------------------------------------------------------------

def add_delta_mag_to_fastspec(
    cat_path,
    nebcorr_dir=NEBCORR_DEFAULT_FOLDER,
    verbose=True,
):
    """
    Copy tractor photometry correction deltas from INT_V2_NEBCORR tables into
    the FASTSPEC HDU (matched by TARGETID).

    BASS2DECAM columns are zero for south (is_south == 1); other deltas are
    copied unchanged.  is_south is read from the same NEBCORR rows.

    New FASTSPEC columns:
        DELTA_MAG_{G,R}_BASS2DECAM, _NEB, _DECAM2SDSS, _KCORR
    """
    print("=" * 60)
    print("Adding DELTA_MAG photometric correction columns to FASTSPEC HDU")
    print("=" * 60)

    delta_tab = _load_nebcorr_delta_mag_table(save_folder=nebcorr_dir, verbose=verbose)
    if delta_tab is None:
        print("  ERROR: No NEBCORR tables with DELTA_MAG columns found. Aborting.")
        print("=" * 60)
        return

    fspec_cat = safe_read_table(cat_path, hdu="FASTSPEC")
    n_objects = len(fspec_cat)
    cat_tids = np.asarray(fspec_cat["TARGETID"])

    if verbose:
        print(f"  FASTSPEC HDU has {n_objects} rows")

    neb_tids = np.asarray(delta_tab["TARGETID"])
    tid_to_row = {int(t): i for i, t in enumerate(neb_tids)}
    is_south_rows = np.asarray(delta_tab["is_south"], dtype=np.int64)
    north_row = (is_south_rows == 0).astype(np.float64)

    matched_rows = np.array([tid_to_row.get(int(t), -1) for t in cat_tids])
    n_matched = int(np.sum(matched_rows >= 0))
    n_unmatched = n_objects - n_matched
    print(f"  add_delta_mag_to_fastspec: matched {n_matched}/{n_objects} TARGETIDs to NEBCORR table")

    if n_matched > 0:
        valid = matched_rows >= 0
        matched_neb_tids = neb_tids[matched_rows[valid]]
        matched_cat_tids = cat_tids[valid]
        max_diff = np.max(np.abs(matched_cat_tids - matched_neb_tids))
        print(f"  Sanity check: max |TARGETID_catalog - TARGETID_nebcorr| = {max_diff}")

    for col in FASTSPEC_DELTA_MAG_COLS:
        arr = np.full(n_objects, np.nan, dtype=np.float64)
        src = np.asarray(delta_tab[col], dtype=np.float64)
        for j, tid in enumerate(cat_tids):
            row = tid_to_row.get(int(tid))
            if row is not None:
                v = src[row]
                if col in (
                    "DELTA_MAG_G_BASS2DECAM",
                    "DELTA_MAG_R_BASS2DECAM",
                ):
                    v = v * north_row[row]
                arr[j] = v
        fspec_cat[col] = arr

    n_matched = int(np.sum(np.isfinite(fspec_cat[FASTSPEC_DELTA_MAG_COLS[0]])))
    if verbose:
        print(f"  Matched {n_matched}/{n_objects} objects to NEBCORR DELTA_MAG columns")

    fspec_hdu_new = fits.table_to_hdu(fspec_cat)
    fspec_hdu_new.name = "FASTSPEC"
    fspec_hdu_new.add_checksum()

    cat_abs = os.path.abspath(cat_path)
    cat_dir = os.path.dirname(cat_abs) or "."
    fd, tmp_path = tempfile.mkstemp(
        suffix=".fits", prefix="delta_mag_", dir=cat_dir
    )
    os.close(fd)
    try:
        with fits.open(cat_abs, memmap=False) as hdul:
            hdu_names = [hdu.name for hdu in hdul]
            main_idx = hdu_names.index("MAIN")
            fspec_idx = hdu_names.index("FASTSPEC")
            main_tab = safe_read_table(cat_abs, hdu="MAIN")
            main_hdu_preserved = fits.table_to_hdu(main_tab)
            main_hdu_preserved.name = "MAIN"
            main_hdu_preserved.add_checksum()
            new_hdus = []
            for i, hdu in enumerate(hdul):
                if i == main_idx:
                    new_hdus.append(main_hdu_preserved)
                elif i == fspec_idx:
                    new_hdus.append(fspec_hdu_new)
                else:
                    new_hdus.append(hdu.copy())
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

    print(f"Updated {cat_path}:")
    print(f"  FASTSPEC HDU: added {', '.join(FASTSPEC_DELTA_MAG_COLS)}")
    print("=" * 60)



_SHARED_EMI_DATA = {}


def _init_emi_worker(wave, flux, ivar, tgid_to_row, targetids):
    """Pool initializer: set shared arrays in each worker."""
    _SHARED_EMI_DATA['wave'] = wave
    _SHARED_EMI_DATA['flux'] = flux
    _SHARED_EMI_DATA['ivar'] = ivar
    _SHARED_EMI_DATA['tgid_to_row'] = tgid_to_row
    _SHARED_EMI_DATA['targetids'] = targetids


def _process_one_emi_file(args):
    """Worker: process one fastspecfit healpix file for emission-subtracted photometry.

    Reads (or loads cached) emission-line models, subtracts them from observed
    spectra, and measures DECam g/r photometry with errors on the residual.
    Accesses the large H5 arrays via the module-level _SHARED_EMI_DATA dict
    (shared across fork-based workers via copy-on-write).
    """
    upath, cat_indices, batch_size, emi_cache_dir, overwrite = args

    h5_wave = _SHARED_EMI_DATA['wave']
    h5_flux = _SHARED_EMI_DATA['flux']
    h5_ivar = _SHARED_EMI_DATA['ivar']
    h5_tgid_to_row = _SHARED_EMI_DATA['tgid_to_row']
    targetids = _SHARED_EMI_DATA['targetids']

    cache_name = os.path.basename(upath).replace('.fits.gz', '.npz').replace('.fits', '.npz')
    cache_path = os.path.join(emi_cache_dir, cache_name) if emi_cache_dir else None
    use_cache = (cache_path is not None and not overwrite
                 and os.path.exists(cache_path))

    if use_cache:
        cached = np.load(cache_path)
        model_wave = cached['model_wave']
        all_emission = cached['emission']
        tgids_file = cached['targetids']
    else:
        try:
            iron_vac = fits.open(upath, memmap=True)
        except Exception:
            return None
        try:
            header = iron_vac["MODELS"].header
            model_wave = (header["CRVAL1"]
                          + (np.arange(header["NAXIS1"]) - header["CRPIX1"])
                          * header["CDELT1"])
            all_emission = iron_vac["MODELS"].data[:, 2, :].copy()
            tgids_file = np.array(iron_vac["FASTSPEC"].data["TARGETID"])
        finally:
            iron_vac.close()

        if cache_path is not None:
            os.makedirs(emi_cache_dir, exist_ok=True)
            np.savez(cache_path, model_wave=model_wave,
                     emission=all_emission, targetids=tgids_file)

    tgid_to_fits_row = {int(t): i for i, t in enumerate(tgids_file)}

    valid_cat, valid_fits_rows, valid_h5_rows = [], [], []
    for ci in cat_indices:
        fits_row = tgid_to_fits_row.get(int(targetids[ci]))
        h5_row = h5_tgid_to_row.get(int(targetids[ci]), -1)
        if fits_row is not None and h5_row >= 0:
            valid_cat.append(ci)
            valid_fits_rows.append(fits_row)
            valid_h5_rows.append(h5_row)

    if len(valid_cat) == 0:
        return None

    valid_cat = np.array(valid_cat)
    valid_fits_rows = np.array(valid_fits_rows)
    valid_h5_rows = np.array(valid_h5_rows)

    emission = all_emission[valid_fits_rows]
    obs_flux = h5_flux[valid_h5_rows]
    obs_ivar = h5_ivar[valid_h5_rows]

    need_resample = (len(model_wave) != len(h5_wave)
                     or not np.allclose(model_wave, h5_wave, atol=0.01))
    if need_resample:
        emission_resampled = np.zeros_like(obs_flux)
        for j in range(len(valid_cat)):
            emission_resampled[j] = resample_flux(h5_wave, model_wave, emission[j])
        emission = emission_resampled

    flux_no_emi = obs_flux - emission

    g_vals = np.full(len(valid_cat), np.nan)
    r_vals = np.full(len(valid_cat), np.nan)
    g_err_vals = np.full(len(valid_cat), np.nan)
    r_err_vals = np.full(len(valid_cat), np.nan)

    for start in range(0, len(valid_cat), batch_size):
        end = min(start + batch_size, len(valid_cat))
        try:
            phot = measure_photo_batch(
                h5_wave,
                flux_no_emi[start:end],
                ivar_2d=obs_ivar[start:end],
            )
            g_vals[start:end] = phot['g_decam']
            r_vals[start:end] = phot['r_decam']
            g_err_vals[start:end] = phot['g_decam_err']
            r_err_vals[start:end] = phot['r_decam_err']
        except Exception:
            pass

    return (valid_cat, g_vals, r_vals, g_err_vals, r_err_vals)


def compute_emission_subtracted_photo_errors(
    cat_path,
    spectra_h5_path="/pscratch/sd/v/virajvm/catalog_dr1_dwarfs/iron_spectra/spectra_files/data/desi_dr1_dwarf_catalog_spectra.h5",
    fastspec_base_dir="/global/cfs/cdirs/desi/public/dr1/vac/dr1/fastspecfit/iron/v2.1/healpix",
    batch_size=500,
    overwrite_model_files=False,
    emi_cache_dir="/pscratch/sd/v/virajvm/desi_dwarf_catalogs/dr1/v1.0/emi_model_cache/",
    ncores=8,
    verbose=True,
    rerun_nans=True, 
):
    """
    Subtract fastspec emission-line models from observed spectra, measure
    DECam g/r photometry (with errors) on the residual, and propagate
    those magnitude errors into a stellar-mass uncertainty.

    Parameters
    ----------
    cat_path : str
        Path to the multi-extension FITS catalog.
    spectra_h5_path : str
        Path to the HDF5 file with observed spectra (WAVE, FLUX, FLUX_IVAR, TARGETID).
    fastspec_base_dir : str
        Base directory for fastspecfit healpix FITS files.
    batch_size : int
        Number of spectra per photometry measurement batch.
    overwrite_model_files : bool
        If True (or cache files do not exist), read MODELS extensions from the
        fastspecfit FITS files and save them as .npz caches under *emi_cache_dir*.
        If False and caches exist, load from .npz instead of reading FITS.
    emi_cache_dir : str or None
        Directory for cached emission model .npz files. Set to None to disable caching.
        Final results in ``emi_subtracted_results.npz`` are merged by TARGETID: rows
        already in the cache are reused and only missing targets are processed.
    ncores : int
        Number of parallel workers for the healpix file loop. Uses fork-based
        multiprocessing so the large H5 arrays are shared via copy-on-write.
        Set to 1 for serial execution.
    verbose : bool
        Print progress information.
    rerun_nans : bool
        If True, cached rows whose g_noemi_err is NaN are treated as missing
        and reprocessed. Useful after re-downloading the spectra HDF5 file.

    Updates the multi-extension FITS catalog at *cat_path*:
      - MAIN HDU: adds LOG_MSTAR_M24_ERR, updates DWARF_MASKBIT (bit 17)
      - FASTSPEC HDU: adds MAG_G_FIBER_NOEMI, MAG_R_FIBER_NOEMI,
        MAG_G_FIBER_NOEMI_ERR, MAG_R_FIBER_NOEMI_ERR
    """
    print("=" * 60)
    print("Computing emission-subtracted photometry and stellar mass errors")
    print("=" * 60)

    # ── 1. Read catalog ──────────────────────────────────────────────
    main_cat = safe_read_table(cat_path, hdu="MAIN")
    fspec_cat = safe_read_table(cat_path, hdu="FASTSPEC")

    n_objects = len(main_cat)
    targetids = np.array(main_cat["TARGETID"])

    if verbose:
        print(f"Catalog has {n_objects} objects")

    # ── 2. Load partial/full cached results by TARGETID ──────────────
    results_cache_path = None
    if emi_cache_dir is not None:
        results_cache_path = os.path.join(emi_cache_dir,
                                          "emi_subtracted_results.npz")

    can_use_results_cache = (results_cache_path is not None
                             and not overwrite_model_files
                             and os.path.exists(results_cache_path))

    g_noemi = np.full(n_objects, np.nan)
    r_noemi = np.full(n_objects, np.nan)
    g_noemi_err = np.full(n_objects, np.nan)
    r_noemi_err = np.full(n_objects, np.nan)

    if can_use_results_cache:
        if verbose:
            print(f"  Loading cached results from {results_cache_path} ...")
        cached = np.load(results_cache_path)
        cached_tids = cached['targetids']
        tid_to_j = {int(t): j for j, t in enumerate(cached_tids)}
        idx_map = np.full(n_objects, -1, dtype=np.int64)
        for i in range(n_objects):
            tid = int(targetids[i])
            if tid in tid_to_j:
                idx_map[i] = tid_to_j[tid]
        hit = idx_map >= 0
        g_noemi[hit] = cached['g_noemi'][idx_map[hit]]
        r_noemi[hit] = cached['r_noemi'][idx_map[hit]]
        g_noemi_err[hit] = cached['g_noemi_err'][idx_map[hit]]
        r_noemi_err[hit] = cached['r_noemi_err'][idx_map[hit]]
        missing_idx = np.flatnonzero(~hit)

        if rerun_nans:
            nan_idx = np.flatnonzero(hit & ~np.isfinite(g_noemi_err))
            if verbose and len(nan_idx) > 0:
                print(f"  rerun_nans: {len(nan_idx)} cached rows have NaN errors; "
                      f"adding to reprocess queue")
            missing_idx = np.union1d(missing_idx, nan_idx)

        if verbose:
            print(f"  Matched {int(hit.sum())}/{n_objects} TARGETIDs from cache; "
                  f"{len(missing_idx)} need emission-subtracted photometry")
    else:
        missing_idx = np.arange(n_objects, dtype=int)

    # ── 3–5. Emission subtraction only for rows not in cache ────────
    if len(missing_idx) > 0:
        if verbose:
            print(f"Loading observed spectra from {spectra_h5_path} ...")
        with h5py.File(spectra_h5_path, "r") as f:
            h5_wave = f["WAVE"][:]
            h5_flux = f["FLUX"][:]
            h5_ivar = f["FLUX_IVAR"][:]
            h5_tgids = f["TARGETID"][:]

        h5_tgid_to_row = {int(t): i for i, t in enumerate(h5_tgids)}
        if verbose:
            print(f"  Loaded {len(h5_tgids)} spectra, wave shape {h5_wave.shape}")

        surveys = np.array(main_cat["SURVEY"].data).astype(str)
        programs = np.array(main_cat["PROGRAM"].data).astype(str)
        healpixes = np.array(main_cat["HEALPIX"].data, dtype=int)

        paths = np.array([
            get_fastspecfit_path(surveys[i], programs[i], healpixes[i],
                                fastspec_base_dir)
            for i in range(n_objects)
        ])
        paths_miss = paths[missing_idx]
        unique_paths = np.unique(paths_miss)
        n_files = len(unique_paths)

        if verbose:
            print(f"Unique fastspecfit FITS files (for missing rows): {n_files}")

        job_args = []
        for upath in unique_paths:
            cat_indices = np.intersect1d(
                np.where(paths == upath)[0], missing_idx, assume_unique=True
            )
            if len(cat_indices) == 0:
                continue
            job_args.append((upath, cat_indices, batch_size, emi_cache_dir,
                             overwrite_model_files))

        if ncores > 1:
            with mp.Pool(processes=ncores,
                         initializer=_init_emi_worker,
                         initargs=(h5_wave, h5_flux, h5_ivar,
                                   h5_tgid_to_row, targetids)) as pool:
                results = list(tqdm(
                    pool.imap(_process_one_emi_file, job_args),
                    total=len(job_args), desc="Emission subtraction"
                ))
        else:
            _init_emi_worker(h5_wave, h5_flux, h5_ivar,
                             h5_tgid_to_row, targetids)
            results = []
            for i, args in enumerate(job_args):
                results.append(_process_one_emi_file(args))
                if verbose and (i + 1) % 50 == 0:
                    print(f"  Processed {i+1}/{n_files} files")
            _SHARED_EMI_DATA.clear()

        for result in results:
            if result is None:
                continue
            valid_cat, g_vals, r_vals, g_err_vals, r_err_vals = result
            g_noemi[valid_cat] = g_vals
            r_noemi[valid_cat] = r_vals
            g_noemi_err[valid_cat] = g_err_vals
            r_noemi_err[valid_cat] = r_err_vals

        if verbose:
            n_good = int(np.sum(np.isfinite(g_noemi)))
            print(f"  Emission-subtracted photometry measured for "
                  f"{n_good}/{n_objects} objects")

    # ── 6. Propagate magnitude errors into stellar mass error ───────
    gr_colors = np.array(main_cat["MAG_G"]) - np.array(main_cat["MAG_R"])
    mag_g_arr = np.array(main_cat["MAG_G"])
    zcmb_arr = np.array(main_cat["Z_CMB"])
    dist_arr = np.array(main_cat["LUMI_DIST_MPC"])

    _, log_mstar_err = get_stellar_mass_mia(
        gr_colors, mag_g_arr, zcmb_arr,
        d_in_mpc=dist_arr, input_zred=False,
        mag_g_err=g_noemi_err, mag_r_err=r_noemi_err,
    )

    finite_err = np.isfinite(log_mstar_err)
    print(f"LOG_MSTAR_M24_ERR: {np.sum(finite_err)} finite values, median = {np.nanmedian(log_mstar_err):.3f} dex")

    snr_threshold = 5.0
    mag_err_limit = 1.0857 / snr_threshold
    low_cont_snr_mask = (
        ~np.isfinite(g_noemi_err) | ~np.isfinite(r_noemi_err)
        | (g_noemi_err >= mag_err_limit)
        | (r_noemi_err >= mag_err_limit)
    )

    if results_cache_path is not None:
        os.makedirs(emi_cache_dir, exist_ok=True)
        np.savez(results_cache_path,
                 targetids=targetids,
                 g_noemi=g_noemi, r_noemi=r_noemi,
                 g_noemi_err=g_noemi_err, r_noemi_err=r_noemi_err,
                 log_mstar_err=log_mstar_err,
                 low_cont_snr_mask=low_cont_snr_mask)
        if verbose:
            print(f"  Saved results cache to {results_cache_path}")

    # ── 7. Update MAIN HDU with LOG_MSTAR_M24_ERR ────────────────────
    main_cat["LOG_MSTAR_M24_ERR"] = log_mstar_err.astype(np.float64)
    if "LOG_MSTAR_M24" in main_cat.colnames:
        cols = list(main_cat.colnames)
        cols.remove("LOG_MSTAR_M24_ERR")
        ins_at = cols.index("LOG_MSTAR_M24") + 1
        cols.insert(ins_at, "LOG_MSTAR_M24_ERR")
        main_cat = main_cat[cols]

    # ── 7b. Flag objects without SNR>5 in g AND r continuum photometry (bit 17) ──
    dwarf_maskbits = np.asarray(main_cat["DWARF_MASKBIT"], dtype=np.int64)
    dwarf_maskbits[low_cont_snr_mask] |= np.int64(1) << 17
    main_cat["DWARF_MASKBIT"] = dwarf_maskbits

    if verbose:
        n_flagged = int(low_cont_snr_mask.sum())
        print(f"  DWARF_MASKBIT bit 17 (low continuum SNR): flagged "
              f"{n_flagged}/{n_objects} objects")

    # table_to_hdu avoids BytesIO→open→.copy() on VLAs (e.g. ASSOCIATED_TARGETIDS),
    # which can raise "Could not find heap data" for variable-length columns.
    main_hdu_new = fits.table_to_hdu(main_cat)
    main_hdu_new.name = "MAIN"
    main_hdu_new.add_checksum()

    # ── 8. Update FASTSPEC HDU with emission-subtracted fiber photometry ──
    fspec_cat["MAG_G_FIBER_NOEMI"] = g_noemi.astype(np.float64)
    fspec_cat["MAG_R_FIBER_NOEMI"] = r_noemi.astype(np.float64)
    fspec_cat["MAG_G_FIBER_NOEMI_ERR"] = g_noemi_err.astype(np.float64)
    fspec_cat["MAG_R_FIBER_NOEMI_ERR"] = r_noemi_err.astype(np.float64)

    fspec_hdu_new = fits.table_to_hdu(fspec_cat)
    fspec_hdu_new.name = "FASTSPEC"
    fspec_hdu_new.add_checksum()

    # ── 9. Rewrite catalog (MAIN + FASTSPEC) ───────────────────────────
    # mode="update" + replace one HDU can leave later HDUs with stale
    # file offsets; verify then fails ("element 2 is not an extension HDU").
    # Build a fresh HDUList (unchanged HDUs via .copy()) and atomic writeto.
    cat_abs = os.path.abspath(cat_path)
    cat_dir = os.path.dirname(cat_abs) or "."
    fd, tmp_path = tempfile.mkstemp(
        suffix=".fits", prefix="emi_subtract_", dir=cat_dir
    )
    os.close(fd)
    try:
        with fits.open(cat_abs, memmap=False) as hdul:
            hdu_names = [hdu.name for hdu in hdul]
            main_idx = hdu_names.index("MAIN")
            fspec_idx = hdu_names.index("FASTSPEC")
            new_hdus = []
            for i, hdu in enumerate(hdul):
                if i == main_idx:
                    new_hdus.append(main_hdu_new)
                elif i == fspec_idx:
                    new_hdus.append(fspec_hdu_new)
                else:
                    new_hdus.append(hdu.copy())
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

    print(f"Updated {cat_path}:")
    print(f"  MAIN HDU: added LOG_MSTAR_M24_ERR, updated DWARF_MASKBIT (bit 17: low continuum SNR)")
    print(f"  FASTSPEC HDU: added MAG_G_FIBER_NOEMI, MAG_R_FIBER_NOEMI, MAG_G_FIBER_NOEMI_ERR, MAG_R_FIBER_NOEMI_ERR")
    print("=" * 60)

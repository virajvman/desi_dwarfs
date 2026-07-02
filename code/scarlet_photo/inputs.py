"""
inputs.py -- load everything one SCARLET fit needs for a single object, from the
canonical stores and the per-object FILE_PATH directory.

The fitter CONSUMES the aperture pipeline's on-disk products rather than
recomputing them (segmentation, star mask, noise, target pixel), which both
saves work and guarantees the SCARLET freeze uses the SAME main-blob definition
as the aperture pipeline.

Sources of truth:
  * grz cube + invvar + WCS : cutout_store.read_cutout (sharded HDF5)
  * empirical PSF (3,63,63)  : psf_store.read_psf (sharded HDF5; NaN plane = band
                               with no CCD coverage)
  * per-object LS catalogue  : {FILE_PATH}/source_cat_f_more.fits
                               (fallback source_cat_f.fits)
  * on-disk masks            : {FILE_PATH}/{segment_map_v2,star_mask,
                               noise_per_band_rms,fiber_pix_pos}.npy
  * aperture geometry/colours: the input-catalogue row (ISOLATE variant)
"""

import os
import sys
from dataclasses import dataclass, field

import numpy as np

# Make sibling top-level modules in code/ importable regardless of CWD.
_CODE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _CODE_DIR not in sys.path:
    sys.path.insert(0, _CODE_DIR)

from .config import ScarletConfig, BANDS


class InputError(Exception):
    """Raised when an object cannot be loaded (missing cube/PSF/etc.)."""


@dataclass
class ObjectInputs:
    # identity / catalogue
    targetid: int
    ra: float
    dec: float
    z: float
    brickname: str
    file_path: str
    image_size_pix: int
    maskbits: int
    is_south: int
    logm: float

    # ISOLATE aperture geometry + comparison colours (from the catalogue row)
    aper_semi_major: float            # semimajor_sigma (px)
    aper_ba: float                    # b/a axis ratio
    aper_theta: float                 # radians
    aper_cen_x: float                 # aperture centroid x (px)
    aper_cen_y: float                 # aperture centroid y (px)
    cog_mag_grz: np.ndarray           # (3,) ISOLATE COG mag g,r,z (comparison)

    # imaging
    data: np.ndarray                  # (3, S, S) grz, nanomaggies, sky-subtracted
    invvar: np.ndarray                # (3, S, S) inverse variance
    mask: np.ndarray                  # (S, S) int or None
    header: str                       # FITS header string (WCS)
    wcs: object                       # astropy.wcs.WCS

    # PSF
    psf: np.ndarray                   # (3, 63, 63) standardized, unit-normalized
    psf_bands_covered: np.ndarray     # (3,) bool; False where the PSF plane is all-NaN

    # per-object source catalogue (LS DR9 Tractor, brick_primary only)
    source_cat: object               # astropy Table

    # on-disk masks (consumed, not recomputed)
    segment_map_v2: np.ndarray        # (S, S): 2=main island, 1=other segment, 0=bkg
    star_mask: np.ndarray             # (S, S): pixels set to 0 inside star circles, else !=0
    noise_per_band_rms: np.ndarray    # (3,) per-band sky RMS (nanomaggies) or None
    fiber_x: float                    # DESI target pixel position
    fiber_y: float

    # bookkeeping
    box_size: int = 0                 # S (== data.shape[-1])
    warnings: list = field(default_factory=list)


# ----------------------------------------------------------------------
# small helpers
# ----------------------------------------------------------------------

def _row_get(row, name, default=np.nan):
    try:
        return row[name]
    except (KeyError, IndexError):
        return default


def _load_npy(file_path, name):
    p = os.path.join(file_path, name)
    if os.path.exists(p):
        return np.load(p, allow_pickle=True)
    return None


def _read_source_cat(file_path):
    """Read {FILE_PATH}/source_cat_f_more.fits (fallback source_cat_f.fits),
    keeping brick_primary rows only -- same as aperture_photo/aperture_cogs."""
    from astropy.table import Table

    for name in ("source_cat_f_more.fits", "source_cat_f.fits"):
        p = os.path.join(file_path, name)
        if os.path.exists(p):
            tbl = Table.read(p)
            if "brick_primary" in tbl.colnames:
                tbl = tbl[np.asarray(tbl["brick_primary"], dtype=bool)]
            return tbl
    raise InputError(
        "no source_cat_f_more.fits / source_cat_f.fits in {}".format(file_path)
    )


# ----------------------------------------------------------------------
# main entry point
# ----------------------------------------------------------------------

def load_object(row, cfg=None):
    """Load all inputs for one object. `row` is an astropy Table Row (or any
    mapping) from the input catalogue. Raises InputError on a hard failure."""
    import cutout_store
    import psf_store

    if cfg is None:
        cfg = ScarletConfig()

    targetid = int(row["TARGETID"])
    ra = float(row["RA"])
    dec = float(row["DEC"])
    brickname = str(row["BRICKNAME"])
    file_path = str(row["FILE_PATH"])
    image_size_pix = int(row["IMAGE_SIZE_PIX"])          # native/catalog size

    # Fixed working box size override (e.g. NERSC production at 350px when the
    # catalog's per-row IMAGE_SIZE_PIX varies and is always >= it). Safe
    # because cutout_store only ever central-crops down (raises if native <
    # requested). When unset, fit_size==image_size_pix and start==0, so every
    # `- start` adjustment below is a no-op -- one code path, not two.
    fit_size = int(cfg.fixed_box_size) if cfg.fixed_box_size else image_size_pix
    start = (image_size_pix - fit_size) // 2

    cutouts_dir = cfg.cutouts_dir or cutout_store.get_store_dir()
    psfs_dir = cfg.psfs_dir or psf_store.get_store_dir()

    warnings = []

    # ---- imaging cube + invvar + WCS --------------------------------------
    try:
        cut = cutout_store.read_cutout(cutouts_dir, brickname, targetid,
                                       size=fit_size)
    except Exception as exc:                              # noqa: BLE001
        raise InputError("read_cutout failed for {}: {}".format(targetid, exc))
    data = np.asarray(cut["image"], dtype=np.float32)
    if data.ndim != 3 or data.shape[0] != 3:
        raise InputError("cube shape {} unexpected for {}".format(data.shape, targetid))
    S = data.shape[-1]

    invvar = cut.get("invvar", None)
    if invvar is None:
        raise InputError("no invvar for {} (needed for SCARLET weights)".format(targetid))
    invvar = np.asarray(invvar, dtype=np.float32)
    mask = cut.get("mask", None)
    header = cut["header"]
    wcs = cutout_store.get_wcs(header)

    # ---- empirical PSF -----------------------------------------------------
    try:
        psf_rec = psf_store.read_psf(psfs_dir, brickname, targetid)
    except Exception as exc:                              # noqa: BLE001
        raise InputError("read_psf failed for {}: {}".format(targetid, exc))
    psf = np.asarray(psf_rec["psf"], dtype=np.float32)
    # An all-NaN plane marks a band with no CCD coverage (psf_store convention).
    psf_bands_covered = np.array(
        [bool(np.all(np.isfinite(psf[b]))) for b in range(psf.shape[0])]
    )
    if not psf_bands_covered.any():
        raise InputError("all PSF bands uncovered (all-NaN) for {}".format(targetid))

    # ---- per-object source catalogue --------------------------------------
    source_cat = _read_source_cat(file_path)

    # ---- on-disk masks (consume, fall back gracefully) --------------------
    seg = _load_npy(file_path, "segment_map_v2.npy")
    if seg is not None:
        seg = np.asarray(seg)
        if seg.shape == (image_size_pix, image_size_pix):
            # Crop to match the fitted frame (no-op when fixed_box_size is unset).
            seg = seg[start:start + fit_size, start:start + fit_size]
        else:
            warnings.append(
                "segment_map_v2 shape {} != catalog IMAGE_SIZE_PIX ({},{}); "
                "using all-main-blob".format(seg.shape, image_size_pix, image_size_pix))
            seg = None
    else:
        warnings.append("segment_map_v2.npy missing; using all-main-blob")
    if seg is None:
        # Fallback: treat whole image as the main blob (no off-blob freeze).
        seg = np.full((S, S), 2, dtype=np.int32)
    seg = np.asarray(seg, dtype=np.int32)

    star_mask = _load_npy(file_path, "star_mask.npy")
    if star_mask is None or np.asarray(star_mask).shape[-2:] != (S, S):
        star_mask = np.ones((S, S), dtype=np.float32)     # no masking
        warnings.append("star_mask.npy missing/mismatched; no star pixels masked")
    star_mask = np.asarray(star_mask)

    nrms = _load_npy(file_path, "noise_per_band_rms.npy")
    noise_per_band_rms = (np.asarray(nrms, dtype=np.float64).ravel()[:3]
                          if nrms is not None else None)

    fiber = _load_npy(file_path, "fiber_pix_pos.npy")
    if fiber is not None:
        # Native-frame position on disk; shift into the fitted frame (no-op
        # when fixed_box_size is unset, since start==0).
        fiber_x, fiber_y = float(fiber[0]) - start, float(fiber[1]) - start
    else:
        # WCS is already fixed for the (possibly cropped) frame by
        # cutout_store, so this fallback needs no shift either way.
        fx, fy = wcs.celestial.all_world2pix(ra, dec, 0)
        fiber_x, fiber_y = float(fx), float(fy)

    # ---- ISOLATE aperture geometry + colours (from the row) ---------------
    aper_params = np.asarray(_row_get(row, "APER_PARAMS_ISOLATE",
                                      [np.nan, np.nan, np.nan]), dtype=np.float64).ravel()
    aper_cen = np.asarray(_row_get(row, "APER_CEN_XY_PIX_ISOLATE",
                                   [np.nan, np.nan]), dtype=np.float64).ravel()
    if aper_cen.size >= 2:
        # Native-frame position from the catalog; shift into the fitted frame
        # (no-op when fixed_box_size is unset, since start==0).
        aper_cen = aper_cen.copy()
        aper_cen[:2] -= start
    cog_mag_grz = np.array([
        float(_row_get(row, "COG_MAG_G_ISOLATE")),
        float(_row_get(row, "COG_MAG_R_ISOLATE")),
        float(_row_get(row, "COG_MAG_Z_ISOLATE")),
    ], dtype=np.float64)

    return ObjectInputs(
        targetid=targetid, ra=ra, dec=dec, z=float(_row_get(row, "Z")),
        brickname=brickname, file_path=file_path, image_size_pix=image_size_pix,
        maskbits=int(_row_get(row, "MASKBITS", 0)),
        is_south=int(_row_get(row, "is_south", 1)),
        logm=float(_row_get(row, "LOGM_M24_FIDU_CORR")),
        aper_semi_major=float(aper_params[0]) if aper_params.size > 0 else np.nan,
        aper_ba=float(aper_params[1]) if aper_params.size > 1 else np.nan,
        aper_theta=float(aper_params[2]) if aper_params.size > 2 else np.nan,
        aper_cen_x=float(aper_cen[0]) if aper_cen.size > 0 else np.nan,
        aper_cen_y=float(aper_cen[1]) if aper_cen.size > 1 else np.nan,
        cog_mag_grz=cog_mag_grz,
        data=data, invvar=invvar, mask=mask, header=header, wcs=wcs,
        psf=psf, psf_bands_covered=psf_bands_covered,
        source_cat=source_cat,
        segment_map_v2=seg, star_mask=star_mask,
        noise_per_band_rms=noise_per_band_rms,
        fiber_x=fiber_x, fiber_y=fiber_y,
        box_size=S, warnings=warnings,
    )

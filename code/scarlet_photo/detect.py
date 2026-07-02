"""
detect.py -- detection coadd, wavelet peak seeding, and Gaia star selection.

Seeding policy (LOCKED 2026-06-30):
  * Extended sources come ONLY from wavelet peaks found on a chi-squared
    detection coadd (colour-agnostic, so a clump significant in any single band
    is detected). No guaranteed target seed, no DR9 catalogue fallback.
  * Gaia stars are seeded directly at their catalogue pixel positions as forced
    PointSources (added in stage 3); no peak<->star matching.
  * Wavelet peaks within `dedup_radius_px` of a star are dropped so a star is
    not double-modelled as an extended source.
"""

from dataclasses import dataclass, field

import numpy as np

from .config import ScarletConfig, PIXSCALE


# ----------------------------------------------------------------------
# detection coadd
# ----------------------------------------------------------------------

def detection_coadd(data, invvar, method="chi2"):
    """Collapse a (3, S, S) grz cube + invvar into a single 2D detection image.

    'chi2' (default): sqrt(sum_b max(SNR_b, 0)^2) -- a positive, colour-agnostic
        significance image (detects a source significant in any band). Noise is
        chi-distributed; the wavelet detector keys off the band-pass DETAIL
        coefficients, which are insensitive to the positive DC offset.
    'ivar_weighted': sum_b(data_b*iv_b) / sqrt(sum_b iv_b) -- matched filter for
        a flat-spectrum source (SNR units).
    'sum': plain g+r+z (the prototype's choice; kept for parity).
    """
    d = np.nan_to_num(np.asarray(data, dtype=np.float64), nan=0.0)
    iv = np.clip(np.nan_to_num(np.asarray(invvar, dtype=np.float64), nan=0.0), 0.0, None)

    if method == "sum":
        return d.sum(axis=0)

    if method == "ivar_weighted":
        num = (d * iv).sum(axis=0)
        den = iv.sum(axis=0)
        out = np.zeros_like(num)
        good = den > 0
        out[good] = num[good] / np.sqrt(den[good])
        return out

    # default: chi-squared (positive) coadd
    snr = d * np.sqrt(iv)
    return np.sqrt(np.sum(np.clip(snr, 0.0, None) ** 2, axis=0))


def _robust_sigma(img):
    """MAD-based noise estimate, falling back to a clipped std."""
    finite = img[np.isfinite(img)]
    if finite.size == 0:
        return 1.0
    med = np.median(finite)
    mad = np.median(np.abs(finite - med))
    s = 1.4826 * mad
    if s > 0:
        return float(s)
    s = float(np.std(finite))
    return s if s > 0 else 1.0


def wavelet_peaks(detect_img, cfg):
    """Run the starlet wavelet detector on a 2D detection image and return a list
    of ``(y, x)`` peak pixel coordinates. May raise on a degenerate image -- the
    caller flags the object (no silent DR9 fallback)."""
    import scarlet
    from scarlet.detect_pybind11 import get_footprints

    sigma = _robust_sigma(detect_img)
    coeffs = scarlet.wavelet.starlet_transform(detect_img, scales=cfg.wavelet_scales)
    M = scarlet.wavelet.get_multiresolution_support(
        detect_img, coeffs, sigma, K=cfg.wavelet_K, epsilon=1e-1, max_iter=20)
    detect = M * coeffs
    detect[detect < 0] = 0

    scale = int(min(cfg.detect_scale, detect.shape[0] - 1))
    footprints = get_footprints(
        detect[scale],
        min_separation=int(cfg.min_separation),
        min_area=int(cfg.min_area),
        thresh=0,
    )
    peaks = []
    for fp in footprints:
        for pk in fp.peaks:
            peaks.append((float(pk.y), float(pk.x)))
    return peaks


# ----------------------------------------------------------------------
# Gaia stars
# ----------------------------------------------------------------------

def _as_str(arr):
    """Decode a FITS string column to clean str. np.asarray on an astropy
    Column bypasses its on-access bytes->str decoding and yields raw 'S'-dtype
    bytes, so str(x) would give "b'G2'" and every comparison silently fails
    (this un-seeded ALL Gaia stars); decode bytes explicitly."""
    out = []
    for x in np.asarray(arr):
        if isinstance(x, bytes):
            x = x.decode("utf-8", "replace")
        out.append(str(x).strip())
    return np.array(out)


def select_gaia_stars(source_cat, pm_sigma=2.0):
    """Boolean mask over `source_cat`: a Gaia source (ref_cat=='G2') that is
    Tractor-typed PSF AND has proper motion significant at >pm_sigma in either
    coordinate.

    This matches aperture_photo.py's `is_star` (G2 & PSF & signi_pm): PM
    significance is REQUIRED, not an alternative -- faint Gaia entries with no
    detected PM can be compact galaxies (and bright nearby HII regions can carry
    spurious PM), so kinematic confirmation gates the star treatment. (An
    earlier OR-form here wrongly admitted every faint G2 PSF; changed 2026-07-02.)

    NOT ported from aperture_photo: the G2+DUP association path
    (`select_dup_stars_g2` -- DUP placeholders confirmed by nearby non-G2 PSF
    rows). None of the local test catalogs contain DUP rows; revisit for the
    NERSC run if DUP stars matter for seeding."""
    n = len(source_cat)
    if n == 0:
        return np.zeros(0, dtype=bool)
    ref_cat = _as_str(source_cat["ref_cat"])
    typ = _as_str(source_cat["type"])
    pmra = np.asarray(source_cat["pmra"], dtype=np.float64)
    pmdec = np.asarray(source_cat["pmdec"], dtype=np.float64)
    pmra_iv = np.clip(np.asarray(source_cat["pmra_ivar"], dtype=np.float64), 0.0, None)
    pmdec_iv = np.clip(np.asarray(source_cat["pmdec_ivar"], dtype=np.float64), 0.0, None)

    signi_pm = (
        (np.abs(pmra) * np.sqrt(pmra_iv) > pm_sigma)
        | (np.abs(pmdec) * np.sqrt(pmdec_iv) > pm_sigma)
    )
    is_gaia = ref_cat == "G2"
    is_psf = typ == "PSF"
    return is_gaia & is_psf & signi_pm


def _gaia_max_mag(source_cat):
    """Brightest (min) of the three Gaia mags per row; NaN-safe."""
    mags = np.vstack([
        np.asarray(source_cat["gaia_phot_bp_mean_mag"], dtype=np.float64),
        np.asarray(source_cat["gaia_phot_rp_mean_mag"], dtype=np.float64),
        np.asarray(source_cat["gaia_phot_g_mean_mag"], dtype=np.float64),
    ])
    return np.nanmin(mags, axis=0)


def mask_radius_pix_for_mag(mag):
    """Rongpu bright-star mask radius (px) at the LS pixel scale, matching the
    prototype: radius_arcsec = 1630 * 1.396**(-mag)."""
    if not np.isfinite(mag):
        mag = 20.0
    return 1630.0 * (1.396 ** (-float(mag))) / PIXSCALE


# ----------------------------------------------------------------------
# seeds
# ----------------------------------------------------------------------

@dataclass
class Seeds:
    extended_centers: list           # list of (y, x) wavelet peaks (deduped vs stars)
    star_centers: np.ndarray         # (M, 2) (y, x) Gaia star pixel positions in frame
    star_radii: np.ndarray           # (M,) star weight-mask radii (px)
    star_ref_ids: np.ndarray         # (M,) Gaia ref_id (for deterministic component IDs)
    n_peaks_raw: int = 0
    n_peaks_kept: int = 0
    n_stars: int = 0


def build_seeds(inp, cfg=None):
    """Build extended-source (wavelet) and star (Gaia) seeds for one object."""
    if cfg is None:
        cfg = ScarletConfig()

    S = inp.box_size
    det = detection_coadd(inp.data, inp.invvar, cfg.detection_method)
    peaks = wavelet_peaks(det, cfg)                      # may raise -> caller flags object

    # Gaia stars -> pixel positions, mags -> radii, ref_ids for stable IDs
    is_star = select_gaia_stars(inp.source_cat, cfg.pm_sigma)
    if is_star.any():
        sc = inp.source_cat[is_star]
        ras = np.asarray(sc["ra"], dtype=np.float64)
        decs = np.asarray(sc["dec"], dtype=np.float64)
        xs, ys = inp.wcs.celestial.all_world2pix(ras, decs, 0)
        star_yx = np.column_stack([np.asarray(ys, float), np.asarray(xs, float)])
        # keep only stars inside the frame
        inb = ((star_yx[:, 0] >= 0) & (star_yx[:, 0] < S)
               & (star_yx[:, 1] >= 0) & (star_yx[:, 1] < S))
        star_yx = star_yx[inb]
        mags = _gaia_max_mag(sc)[inb]
        radii = np.array([mask_radius_pix_for_mag(m) for m in mags])
        ref_ids = (np.asarray(sc["ref_id"])[inb]
                   if "ref_id" in sc.colnames else np.arange(np.sum(inb)))
    else:
        star_yx = np.zeros((0, 2))
        radii = np.zeros(0)
        ref_ids = np.zeros(0, dtype=np.int64)

    # dedup wavelet peaks within dedup_radius of any star
    kept = []
    r2 = cfg.dedup_radius_px ** 2
    for (py, px) in peaks:
        if star_yx.shape[0] and np.any(
                (star_yx[:, 0] - py) ** 2 + (star_yx[:, 1] - px) ** 2 <= r2):
            continue
        kept.append((py, px))

    return Seeds(
        extended_centers=kept,
        star_centers=star_yx,
        star_radii=radii,
        star_ref_ids=ref_ids,
        n_peaks_raw=len(peaks),
        n_peaks_kept=len(kept),
        n_stars=int(star_yx.shape[0]),
    )


def apply_circular_mask(weights, cy, cx, radius):
    """Zero a circular region (centre (cy,cx) px, given radius) across all bands
    of a (3, S, S) weight cube, in place. Returns the cube."""
    H, W = weights.shape[-2], weights.shape[-1]
    yy = np.arange(H)[:, None]
    xx = np.arange(W)[None, :]
    circle = (yy - cy) ** 2 + (xx - cx) ** 2 <= radius ** 2
    weights[:, circle] = 0.0
    return weights

"""
photometry.py -- the initial-composite SCARLET magnitudes (the benchmark
deliverable to compare against the aperture-photometry pipeline).

Two mags, computed from DIFFERENT spaces (this distinction matters):
  * TOTAL  : sum of the MODEL-FRAME fluxes (scarlet.measure.flux) over the
             dwarf-member components + LSB -- the deconvolved total, free of
             neighbour PSF wings.
  * R4     : integral of the OBSERVED-FRAME rendered dwarf model within the
             R4.25 ISOLATE aperture ellipse -- "flux in an aperture" is an
             observed-frame spatial quantity, so it uses the rendered cube.

Both Milky-Way-extinction corrected via mw_transmission_grz from the matched
source-catalogue row. Uncertainties = a photon-noise FLOOR (propagate invvar
over the relevant footprint); this is a lower bound -- it ignores the
deblending/grouping degeneracy, which the VI step pins down. (Option b, locked.)
"""

from dataclasses import dataclass

import numpy as np

from .config import ScarletConfig

_LN10 = np.log(10.0)


@dataclass
class PhotometryResult:
    total_flux: np.ndarray           # (3,) model-frame, MW-corrected nanomaggies
    total_mag: np.ndarray            # (3,) g,r,z
    total_mag_err: np.ndarray        # (3,) photon-noise floor
    r4_flux: np.ndarray              # (3,) observed-frame aperture, MW-corrected
    r4_mag: np.ndarray
    r4_mag_err: np.ndarray
    mw_transmission: np.ndarray      # (3,)
    r4_measured: bool                # False if the aperture geometry was NaN


def _mag(flux):
    flux = np.asarray(flux, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        m = 22.5 - 2.5 * np.log10(flux)
    m[~np.isfinite(m)] = np.nan
    return m


def _mag_err(flux, flux_err):
    flux = np.asarray(flux, dtype=np.float64)
    flux_err = np.asarray(flux_err, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        e = (2.5 / _LN10) * np.abs(flux_err / flux)
    e[~np.isfinite(e)] = np.nan
    return e


def _mw_transmission(inp):
    """mw_transmission_grz from the source-catalogue row nearest (ra, dec).
    Returns ones (no correction) if unavailable."""
    sc = inp.source_cat
    cols = ("mw_transmission_g", "mw_transmission_r", "mw_transmission_z")
    if len(sc) == 0 or not all(c in sc.colnames for c in cols):
        return np.ones(3)
    ra = np.asarray(sc["ra"], dtype=np.float64)
    dec = np.asarray(sc["dec"], dtype=np.float64)
    # small-angle separation is fine here (we only need the nearest row)
    cosd = np.cos(np.radians(inp.dec))
    d2 = ((ra - inp.ra) * cosd) ** 2 + (dec - inp.dec) ** 2
    j = int(np.argmin(d2))
    mw = np.array([float(sc[c][j]) for c in cols], dtype=np.float64)
    mw[~np.isfinite(mw) | (mw <= 0)] = 1.0
    return mw


def _ellipse_mask(inp, cfg):
    """Boolean R4.25 aperture mask from the ISOLATE aperture geometry, or None
    if the geometry is undefined."""
    a0 = inp.aper_semi_major
    ba = inp.aper_ba
    th = inp.aper_theta
    cx = inp.aper_cen_x
    cy = inp.aper_cen_y
    if not all(np.isfinite([a0, ba, th, cx, cy])) or a0 <= 0 or ba <= 0:
        return None
    a = a0 * cfg.r4_aperture_scale
    b = a * ba
    S = inp.box_size
    yy, xx = np.mgrid[0:S, 0:S]
    dx = xx - cx
    dy = yy - cy
    ct, st = np.cos(th), np.sin(th)
    xp = dx * ct + dy * st
    yp = -dx * st + dy * ct
    return (xp / a) ** 2 + (yp / b) ** 2 <= 1.0


def _footprint_var(invvar, footprint):
    """Photon-noise variance of a flux summed over `footprint`, per band:
    sum of 1/invvar over footprint pixels with invvar>0."""
    iv = np.asarray(invvar, dtype=np.float64)
    var = np.zeros(iv.shape[0])
    for b in range(iv.shape[0]):
        sel = footprint & (iv[b] > 0)
        if np.any(sel):
            var[b] = np.sum(1.0 / iv[b][sel])
        else:
            var[b] = np.nan
    return var


def measure_photometry(inp, comp_flux, membership, dwarf_obs, cfg=None):
    """Compute total + R4 mags (MW-corrected) and photon-noise errors.

    comp_flux : (N,3) model-frame fluxes; membership : (N,) bool;
    dwarf_obs : (3,S,S) observed-frame dwarf model (== science_cube).
    """
    if cfg is None:
        cfg = ScarletConfig()

    members = np.where(np.asarray(membership, dtype=bool))[0]
    comp_flux = np.asarray(comp_flux, dtype=np.float64)
    mw = _mw_transmission(inp)
    mw_corr = 2.5 * np.log10(mw)                            # add to mag to de-extinct

    # --- TOTAL (model-frame) ---
    total_flux = (comp_flux[members].sum(axis=0) if members.size
                  else np.zeros(3))
    total_mag = _mag(total_flux) + mw_corr

    # dwarf footprint for the photon-noise floor on the total
    band_sum = np.abs(dwarf_obs).sum(axis=0)
    peak = float(band_sum.max()) if band_sum.size else 0.0
    foot = (band_sum > cfg.rel_thresh_patch * peak) if peak > 0 else np.zeros_like(band_sum, bool)
    total_var = _footprint_var(inp.invvar, foot)
    total_mag_err = _mag_err(total_flux, np.sqrt(total_var))

    # --- R4 (observed-frame aperture integral) ---
    aper = _ellipse_mask(inp, cfg)
    if aper is None:
        r4_flux = np.full(3, np.nan)
        r4_mag = np.full(3, np.nan)
        r4_mag_err = np.full(3, np.nan)
        r4_measured = False
    else:
        r4_flux = np.array([dwarf_obs[b][aper].sum() for b in range(3)], dtype=np.float64)
        r4_mag = _mag(r4_flux) + mw_corr
        r4_var = _footprint_var(inp.invvar, aper)
        r4_mag_err = _mag_err(r4_flux, np.sqrt(r4_var))
        r4_measured = True

    return PhotometryResult(
        total_flux=total_flux, total_mag=total_mag, total_mag_err=total_mag_err,
        r4_flux=r4_flux, r4_mag=r4_mag, r4_mag_err=r4_mag_err,
        mw_transmission=mw, r4_measured=r4_measured,
    )

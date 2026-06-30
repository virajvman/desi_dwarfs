"""
grouping.py -- the INITIAL automatic dwarf grouping (the VI default).

This reproduces the prototype's per-component `dwarf_mask` (the "colour/GMM
classifier"), which decides which fitted components are the dwarf BEFORE any
human VI. It is only a starting point -- the future VI tool adds/removes
components -- so it does not need to be perfect.

Default rule (LOCKED 2026-06-30) = the OR:

    member  =  ( (g-r < gr_cut) AND (r-z < rz_cut) )            # LSB-anchored blue box
               OR ( GMM_prob >= conf_levels[gmm_contour] )      # per-z-bin GMM contour

evaluated only over main-blob (segment 2), non-star components. The global LSB
StarletSource is ALWAYS a member. The blue-box cuts are anchored on the LSB's
own colour (the dwarf's diffuse colour); the `lsb != np.nan` bug of the
prototype is fixed here with `np.isnan`.

Inputs are model-frame per-component fluxes (scarlet.measure.flux), passed in so
this module needs no scarlet import. It does import the pre-trained per-z-bin
GMMs via aperture_photo._load_gmm (NERSC).
"""

import os
import sys
from dataclasses import dataclass

import numpy as np

_CODE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _CODE_DIR not in sys.path:
    sys.path.insert(0, _CODE_DIR)

from .config import ScarletConfig

# clevs -> dict key, matching the prototype's conf_levels keys
_CONF_KEYS = {0.38: "38", 0.68: "68", 0.86: "86", 0.954: "95.4", 0.987: "98.7"}

# colour-space grid used to turn the GMM into density-contour thresholds
_GR_RANGE = (-1.0, 2.0)
_RZ_RANGE = (-1.0, 1.5)
_GRID_N = 200


@dataclass
class GroupingResult:
    membership: np.ndarray           # (N,) bool, initial_membership per component
    gr: np.ndarray                   # (N,) g-r colour (NaN if flux<=0)
    rz: np.ndarray                   # (N,) r-z colour
    gmm_prob: np.ndarray             # (N,) exp(score) at the component colour (NaN if unavailable)
    gr_cut: float
    rz_cut: float
    gmm_contour_value: float         # the density threshold the GMM term compares to


def conf_interval(x, pdf, conf_level):
    """Prototype's contour-level root function: integral of pdf above x, minus
    the target confidence level."""
    return np.sum(pdf[pdf > x]) - conf_level


def _safe_mag(flux):
    flux = np.asarray(flux, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        mag = 22.5 - 2.5 * np.log10(flux)
    mag[~np.isfinite(mag)] = np.nan
    return mag


def _colors_from_flux(comp_flux):
    mags = _safe_mag(np.asarray(comp_flux, dtype=np.float64))      # (N,3)
    gr = mags[:, 0] - mags[:, 1]
    rz = mags[:, 1] - mags[:, 2]
    return gr, rz


def _load_gmm_for_z(z):
    """Per-z-bin GMM (sklearn GaussianMixture) via aperture_photo._load_gmm.
    Bin grid matches the prototype: np.arange(0.001, 0.525, 0.025)."""
    from aperture_photo import _load_gmm
    zgrid = np.arange(0.001, 0.525, 0.025)
    idx = np.where(z > zgrid)[0]
    file_index = int(idx[-1]) if idx.size else 0
    file_index = min(max(file_index, 0), len(zgrid) - 1)
    return _load_gmm(file_index)


def _gmm_contour_thresholds(gmm, conf_levels):
    """Map the GMM to per-confidence-level density thresholds in the SAME
    (un-normalised exp(score)) units as the per-component probabilities, exactly
    as the prototype does (grid-normalise, brentq, rescale by Znorm)."""
    from scipy.optimize import brentq
    gx = np.linspace(_GR_RANGE[0], _GR_RANGE[1], _GRID_N)
    gy = np.linspace(_RZ_RANGE[0], _RZ_RANGE[1], _GRID_N)
    xx, yy = np.meshgrid(gx, gy)
    pos = np.column_stack([xx.ravel(), yy.ravel()])
    pdf = np.exp(gmm.score_samples(pos))
    znorm = float(pdf.sum())
    if znorm <= 0:
        return {}
    pdf_n = pdf / znorm
    out = {}
    for cl, key in _CONF_KEYS.items():
        try:
            thr_n = brentq(conf_interval, 0.0, 1.0, args=(pdf_n, cl))
            out[key] = thr_n * znorm
        except (ValueError, RuntimeError):
            out[key] = np.inf
    return out


def group_components(fit_result, comp_flux, redshift, cfg=None):
    """Compute the initial dwarf membership + per-component colour/GMM diagnostics.

    fit_result : FitResult (for comp_meta + lsb_index)
    comp_flux  : (N, 3) model-frame fluxes (scarlet.measure.flux per component)
    redshift   : object redshift (selects the GMM z-bin)
    """
    if cfg is None:
        cfg = ScarletConfig()

    meta = fit_result.comp_meta
    N = len(meta)
    gr, rz = _colors_from_flux(comp_flux)

    # --- blue-box cuts anchored on the LSB colour ---
    lsb_i = fit_result.lsb_index
    lsb_gr, lsb_rz = gr[lsb_i], rz[lsb_i]
    if np.isfinite(lsb_gr) and np.isfinite(lsb_rz):
        gr_cut = max(lsb_gr, cfg.color_floor) + cfg.col_lenient
        rz_cut = max(lsb_rz, cfg.color_floor) + cfg.col_lenient
    else:
        gr_cut = cfg.gr_cut_fallback
        rz_cut = cfg.rz_cut_fallback

    # --- GMM contour term ---
    gmm_prob = np.full(N, np.nan)
    contour_value = np.inf
    if cfg.grouping_rule in ("or", "gmm"):
        try:
            gmm = _load_gmm_for_z(redshift)
            thresholds = _gmm_contour_thresholds(gmm, cfg.gmm_conf_levels)
            contour_value = thresholds.get(cfg.gmm_contour, np.inf)
            finite = np.isfinite(gr) & np.isfinite(rz)
            if finite.any():
                gmm_prob[finite] = np.exp(
                    gmm.score_samples(np.column_stack([gr[finite], rz[finite]])))
        except Exception:                                  # noqa: BLE001
            # GMM unavailable (e.g. local dev) -> GMM term contributes nothing
            contour_value = np.inf

    # --- per-component membership ---
    membership = np.zeros(N, dtype=bool)
    for i, m in enumerate(meta):
        if m["type"] == "starlet_lsb":
            membership[i] = True                            # LSB always included
            continue
        if m["is_star"] or not m["on_main_blob"]:
            continue                                        # stars / off-blob never default-in
        bluebox = bool((gr[i] < gr_cut) and (rz[i] < rz_cut))
        gmm_in = bool(np.isfinite(gmm_prob[i]) and gmm_prob[i] >= contour_value)
        if cfg.grouping_rule == "bluebox":
            membership[i] = bluebox
        elif cfg.grouping_rule == "gmm":
            membership[i] = gmm_in
        else:                                               # 'or'
            membership[i] = bluebox or gmm_in

    return GroupingResult(
        membership=membership, gr=gr, rz=rz, gmm_prob=gmm_prob,
        gr_cut=float(gr_cut), rz_cut=float(rz_cut),
        gmm_contour_value=float(contour_value),
    )

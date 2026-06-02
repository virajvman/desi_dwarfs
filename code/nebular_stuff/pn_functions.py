"""
Functions based on PyNeb which build the model and log_likelihood functions.
Code adapted from Dirk Scholte's desimetals.

Changes from the original:
  * Hgamma/Hbeta added to the model so A_V is jointly constrained by
    both Balmer decrements (matches Scholte+2026, Table 3).
  * [Ne III] 3869/Hbeta and the Ne++ abundance parameter removed -- they
    are decoupled from the oxygen-only fit.
  * Typo in r_O2_Hb fixed (second term now uses transmission at 3729 A,
    not 3726 A).
  * Catalog driver accepts an astropy.table.Table and returns one.
  * Inference uses UltraNest nested sampling (matches Scholte+2026).
  * Interpolation grids now extrapolate silently for tiny out-of-bound
    queries from the sampler's grid lookups near the prior edges.
"""
import numpy as np
from astropy.table import Table
from scipy.interpolate import RegularGridInterpolator
import pyneb as pn

from cardelli_attenuation import *

# ---------------------------------------------------------------------------
# Atomic data
# ---------------------------------------------------------------------------
pn.atomicData.addDataFilePath('/global/homes/v/virajvm/DESI2_LOWZ/desi_dwarfs/data/data_metal/atomic_data/')

DataFileDict = {
    'H1': {'rec': 'h_i_rec_extrapolated_SH95.fits'},
    'O2': {'atom': 'o_ii_atom_FFT04.dat', 'coll': 'o_ii_coll_Kal09.dat'},
    'O3': {'atom': 'o_iii_atom_FFT04.dat', 'coll': 'o_iii_coll_SSB14.dat'},
    'N2': {'atom': 'n_ii_atom_FFT04.dat', 'coll': 'n_ii_coll_T11.dat'},
}
pn.atomicData.setDataFileDict(DataFileDict)

# ---------------------------------------------------------------------------
# Interpolation grid bounds
# ---------------------------------------------------------------------------
den_min, den_max, n_den = 1e1, 5e3, 50
tem_min, tem_max, n_tem = 1e3, 35e3, 50


def getInterpEmisGrid(atom, level, wave,
                      tem_min=tem_min, tem_max=tem_max, n_tem=n_tem,
                      den_min=den_min, den_max=den_max, n_den=n_den):
    tem = 10 ** np.linspace(np.log10(tem_min), np.log10(tem_max), n_tem)
    den = 10 ** np.linspace(np.log10(den_min), np.log10(den_max), n_den)
    atomobj = pn.Atom(atom, level)
    tems, dens = np.meshgrid(tem, den, indexing='ij')
    grid = RegularGridInterpolator(
        [tem, den],
        atomobj.getEmissivity(tem=tems, den=dens, wave=wave, product=False),
        bounds_error=False, fill_value=None,   # extrapolate at the edges
    )
    return lambda te, ne: grid((te, ne))


def getInterpRecEmisGrid(atom, level, wave,
                         tem_min=tem_min, tem_max=tem_max, n_tem=n_tem,
                         den_min=den_min, den_max=den_max, n_den=n_den):
    tem = 10 ** np.linspace(np.log10(tem_min), np.log10(tem_max), n_tem)
    den = 10 ** np.linspace(np.log10(den_min), np.log10(den_max), n_den)
    atomobj = pn.RecAtom(atom, level)
    tems, dens = np.meshgrid(tem, den, indexing='ij')
    grid = RegularGridInterpolator(
        [tem, den],
        atomobj.getEmissivity(tem=tems, den=dens, wave=wave, product=False),
        bounds_error=False, fill_value=None,   # extrapolate at the edges
    )
    return lambda te, ne: grid((te, ne))


# Emissivity grids
O2_3726 = getInterpEmisGrid('O', 2, 3726)
O2_3729 = getInterpEmisGrid('O', 2, 3729)
O3_4363 = getInterpEmisGrid('O', 3, 4363)
O3_4959 = getInterpEmisGrid('O', 3, 4959)
O3_5007 = getInterpEmisGrid('O', 3, 5007)
N2_5755 = getInterpEmisGrid('N', 2, 5755)
N2_6584 = getInterpEmisGrid('N', 2, 6584)

H_gamma = getInterpRecEmisGrid('H', 1, 4340)
H_beta  = getInterpRecEmisGrid('H', 1, 4861)
H_alpha = getInterpRecEmisGrid('H', 1, 6563)


# ---------------------------------------------------------------------------
# Temperature relations
# ---------------------------------------------------------------------------
def tlow_thi(thi):
    # Campbell 1986
    return 0.7 * thi + 3000


def thi_tlow(tlow):
    # Campbell 1986
    return (tlow - 3000) / 0.7


def calc_Av_Ha_Hb(Ha_Hb, Ha_Hb_intrinsic):
    return np.clip(
        -2.5 * np.log10(Ha_Hb / Ha_Hb_intrinsic) / (k_ccm89(6563) - k_ccm89(4861)),
        0, np.inf,
    )


# ---------------------------------------------------------------------------
# Forward model and likelihood
# ---------------------------------------------------------------------------
# Parameters: (ne_oii, thi_oiii, Av, log_O2_abund, log_O3_abund)
# Ratios returned, in order:
#   0. [O II] 3726 / [O II] 3729           (n_e)
#   1. Hbeta / Halpha                      (A_V)
#   2. Hgamma / Hbeta                      (A_V)
#   3. [O III] 4363 / [O III] 5007         (T_high)
#   4. ([O II] 3726 + 3729) / Hbeta        (O+/H+)
#   5. [O III] 5007 / Hbeta                (O++/H+)
def r_model(theta):
    if theta.ndim == 2:
        theta = theta.T

    ne_oii, thi_oiii, Av, O2_abund, O3_abund = theta

    ne = ne_oii
    tlow = tlow_thi(thi_oiii)
    tlow = np.max([tlow, np.ones_like(tlow) * tem_min], axis=0)
    thi = thi_oiii

    r_O2_ratio = (
        (O2_3726(tlow, ne) * transmission_Av(3726., Av))
        / (O2_3729(tlow, ne) * transmission_Av(3729., Av))
    )
    r_Hb_Ha = (
        (H_beta(thi, ne) * transmission_Av(4861., Av))
        / (H_alpha(thi, ne) * transmission_Av(6563., Av))
    )
    r_Hg_Hb = (
        (H_gamma(thi, ne) * transmission_Av(4340., Av))
        / (H_beta(thi, ne) * transmission_Av(4861., Av))
    )
    r_O3_4363_5007 = (
        (O3_4363(thi, ne) * transmission_Av(4363., Av))
        / (O3_5007(thi, ne) * transmission_Av(5007., Av))
    )
    r_O2_Hb = 10**O2_abund * (
        O2_3726(tlow, ne) * transmission_Av(3726., Av)
        + O2_3729(tlow, ne) * transmission_Av(3729., Av)
    ) / (H_beta(thi, ne) * transmission_Av(4861., Av))
    r_O3_Hb = 10**O3_abund * (
        (O3_5007(thi, ne) * transmission_Av(5007., Av))
        / (H_beta(thi, ne) * transmission_Av(4861., Av))
    )

    rs = np.array([r_O2_ratio, r_Hb_Ha, r_Hg_Hb,
                   r_O3_4363_5007, r_O2_Hb, r_O3_Hb])
    if theta.ndim == 1:
        return rs[:, 0]
    return rs


def log_likelihood(theta, r, r_err):
    model = r_model(theta)
    if model.ndim == 1:
        return -0.5 * np.sum((r - model) ** 2 / r_err ** 2)
    r = np.expand_dims(r, axis=-1)
    r_err = np.expand_dims(r_err, axis=-1)
    return -0.5 * np.sum((r - model) ** 2 / r_err ** 2, axis=0)


def ratio_err(a, b, a_err, b_err):
    """Ratio and its 1-sigma uncertainty (independent Gaussian errors)."""
    ratio = a / b
    error = np.abs(ratio) * np.sqrt((a_err / a) ** 2 + (b_err / b) ** 2)
    return ratio, error



# ===========================================================================
# Catalog-level driver
# ===========================================================================
# Order MUST match the order of ratios returned by r_model.
RATIO_SPECS = [
    ('OII_3726',  'OII_3729'),               # density
    ('HBETA',     'HALPHA'),                 # A_V
    ('HGAMMA',    'HBETA'),                  # A_V
    ('OIII_4363', 'OIII_5007'),              # T_high
    (('OII_3726', 'OII_3729'), 'HBETA'),     # O+/H+
    ('OIII_5007', 'HBETA'),                  # O++/H+
]

PARAM_NAMES = ['ne_oii', 'te_oiii', 'Av', 'log_O2_abund', 'log_O3_abund']

# Uniform prior bounds, matching Scholte+2026 Table 3 priors closely.
# (slight insets from grid edges so finite-diff gradients don't escape)
PRIOR_LOWS = np.array([den_min * 1.001, tem_min * 1.001, 0.0, -6.0, -6.0])
PRIOR_HIGHS = np.array([den_max * 0.999, tem_max * 0.999, 5.0, -2.0, -2.0])


def _flux_and_err(row, line):
    """Return (flux, sigma) for a line key like 'OIII_4363'.
    Treats missing columns, non-finite values, or ivar <= 0 as missing."""
    flux_col = f'{line}_FLUX'
    ivar_col = f'{line}_FLUX_IVAR'
    colnames = getattr(row, 'colnames', None)
    if colnames is None:
        colnames = row.columns if hasattr(row, 'columns') else []
    if flux_col not in colnames or ivar_col not in colnames:
        return np.nan, np.nan
    f = row[flux_col]
    ivar = row[ivar_col]
    if not np.isfinite(f) or not np.isfinite(ivar) or ivar <= 0:
        return np.nan, np.nan
    return float(f), float(1.0 / np.sqrt(ivar))


def _build_ratios(row):
    """Build the ratio vector, its error vector, and a mask of usable ratios."""
    n = len(RATIO_SPECS)
    r = np.full(n, np.nan)
    r_err = np.full(n, np.nan)

    for i, (num, den) in enumerate(RATIO_SPECS):
        if isinstance(num, tuple):
            pairs = [_flux_and_err(row, ln) for ln in num]
            fs = [p[0] for p in pairs]
            es = [p[1] for p in pairs]
            if any(not np.isfinite(x) for x in fs + es):
                continue
            f_num = sum(fs)
            e_num = np.sqrt(sum(e ** 2 for e in es))
        else:
            f_num, e_num = _flux_and_err(row, num)
            if not np.isfinite(f_num):
                continue

        f_den, e_den = _flux_and_err(row, den)
        if not np.isfinite(f_den) or f_den <= 0:
            continue

        ratio, err = ratio_err(f_num, f_den, e_num, e_den)
        r[i] = ratio
        r_err[i] = err

    mask = np.isfinite(r) & np.isfinite(r_err) & (r_err > 0)
    return r, r_err, mask


def _nan_fit_result(n_par, n_used, success=False):
    """Standard result dict for rows that can't be fit."""
    nan_arr = np.full(n_par, np.nan)
    return {
        'theta': nan_arr,
        'theta_lo': nan_arr.copy(),
        'theta_hi': nan_arr.copy(),
        'theta_err': nan_arr.copy(),
        'twelve_log_OH': np.nan,
        'twelve_log_OH_lo': np.nan,
        'twelve_log_OH_hi': np.nan,
        'twelve_log_OH_err': np.nan,
        'success': success,
        'n_ratios': n_used,
    }


# ---------------------------------------------------------------------------
# Per-row UltraNest fit (matches Scholte+2026)
# ---------------------------------------------------------------------------
def _make_prior_transform(lows, highs):
    """Uniform priors via unit-cube transform."""
    spans = highs - lows
    def transform(cube):
        return lows + cube * spans
    return transform


def ptform_1d_from_samples(samples, lo, hi):
    """Build a 1D informative prior transform from posterior samples.

    Returns a vectorized callable mapping unit-cube value(s) u in [0, 1] to
    parameter value(s) via the empirical inverse CDF of *samples*, clipped to
    [lo, hi]. Used for Stage-2 of the two-stage fit so the Stage-1 posterior
    on ne/Te/Av becomes the Stage-2 prior (matches the collaborator's
    ptform_1d_from_samples usage)."""
    s = np.sort(np.asarray(samples, dtype=float))
    n = len(s)
    # Midpoint quantile grid: q_i = (i + 0.5) / n.
    q = (np.arange(n) + 0.5) / n

    def transform(u):
        v = np.interp(u, q, s)
        return np.clip(v, lo, hi)

    return transform


def _run_sampler(param_names, loglike, transform,
                 min_num_live_points=400,
                 verbose_sampler=False,
                 sampler_kwargs=None):
    """Run one UltraNest fit and return equal-weight resampled posterior
    points, or ``None`` on failure.

    Shared by both stages of the two-stage fit. Applies the same run_kwargs +
    sampler_kwargs termination-guard logic and the same seed-42 weighted
    resampling as ``_fit_row_ultranest`` so the two code paths behave
    consistently."""
    try:
        import ultranest
    except ImportError as e:
        raise ImportError(
            "method='ultranest' requires the ultranest package "
            "(pip install ultranest)"
        ) from e

    import logging
    if not verbose_sampler:
        logging.getLogger('ultranest').setLevel(logging.WARNING)

    sampler = ultranest.ReactiveNestedSampler(
        list(param_names), loglike, transform,
        vectorized=True,
    )
    run_kwargs = {'min_num_live_points': min_num_live_points,
                  'show_status': verbose_sampler,
                  'viz_callback': False}
    if sampler_kwargs:
        run_kwargs.update(sampler_kwargs)

    try:
        result = sampler.run(**run_kwargs)
    except Exception:
        return None

    ws = result['weighted_samples']
    pts = np.asarray(ws['points'])      # (N_samples, n_par)
    wts = np.asarray(ws['weights'])     # (N_samples,)
    wts_norm = wts / wts.sum()
    n_resample = min(10_000, max(2_000, len(pts)))
    rng = np.random.default_rng(42)
    idx = rng.choice(len(pts), size=n_resample, replace=True, p=wts_norm)
    return pts[idx]


def _fit_row_ultranest(r, r_err, mask,
                       min_num_live_points=400,
                       verbose_sampler=False,
                       sampler_kwargs=None):
    """Nested sampling with UltraNest. Returns posterior 16/50/84 percentiles.
    Matches the inference approach of Scholte+2026."""
    try:
        import ultranest
    except ImportError as e:
        raise ImportError(
            "method='ultranest' requires the ultranest package "
            "(pip install ultranest)"
        ) from e

    n_used = int(mask.sum())
    n_par = len(PARAM_NAMES)
    if n_used < 3:
        return _nan_fit_result(n_par, n_used)

    # Pre-slice the data for speed inside the likelihood.
    r_m = r[mask]
    e_m = r_err[mask]
    mask_idx = np.where(mask)[0]

    # Vectorized likelihood: accepts (N, n_par) batches and returns (N,).
    # r_model returns shape (n_ratios, N) when theta is 2D.
    r_m_col = r_m[:, None]
    e_m_col = e_m[:, None]

    def loglike(thetas):
        thetas = np.asarray(thetas)
        if thetas.ndim == 1:
            # UltraNest sometimes passes a single point even in vectorized mode.
            model = r_model(thetas)
            return -0.5 * np.sum(((r_m - model[mask_idx]) / e_m) ** 2)
        models = r_model(thetas)                  # (n_ratios, N)
        resid = (r_m_col - models[mask_idx]) / e_m_col
        return -0.5 * np.sum(resid ** 2, axis=0)  # (N,)

    # Vectorized prior transform.
    spans = PRIOR_HIGHS - PRIOR_LOWS

    def transform(cubes):
        cubes = np.asarray(cubes)
        return PRIOR_LOWS + cubes * spans  # broadcasts for both 1D and 2D

    # Silence ultranest's stderr output unless explicitly requested.
    import logging
    if not verbose_sampler:
        logging.getLogger('ultranest').setLevel(logging.WARNING)

    sampler = ultranest.ReactiveNestedSampler(
        list(PARAM_NAMES), loglike, transform,
        vectorized=True,
    )
    run_kwargs = {'min_num_live_points': min_num_live_points,
                  'show_status': verbose_sampler,
                  'viz_callback': False}
    if sampler_kwargs:
        run_kwargs.update(sampler_kwargs)

    try:
        result = sampler.run(**run_kwargs)
    except Exception:
        return _nan_fit_result(n_par, n_used, success=False)

    # Weighted posterior samples
    ws = result['weighted_samples']
    pts = np.asarray(ws['points'])      # (N_samples, n_par)
    wts = np.asarray(ws['weights'])     # (N_samples,)

    # Resample with weights to equal-weight samples for convenient percentile
    # + derived-quantity calculation.
    wts_norm = wts / wts.sum()
    n_resample = min(10_000, max(2_000, len(pts)))
    rng = np.random.default_rng(42)
    idx = rng.choice(len(pts), size=n_resample, replace=True, p=wts_norm)
    samples = pts[idx]

    theta_lo = np.percentile(samples, 16, axis=0)
    theta_med = np.percentile(samples, 50, axis=0)
    theta_hi = np.percentile(samples, 84, axis=0)
    theta_err = 0.5 * (theta_hi - theta_lo)

    # Derived: 12 + log10(O+/H+ + O++/H+), computed per posterior sample so
    # the O+/O++ anti-correlation is captured correctly.
    i2 = PARAM_NAMES.index('log_O2_abund')
    i3 = PARAM_NAMES.index('log_O3_abund')
    O_over_H_samples = 10 ** samples[:, i2] + 10 ** samples[:, i3]
    twelve_logOH_samples = 12 + np.log10(O_over_H_samples)
    twelve_logOH_lo = np.percentile(twelve_logOH_samples, 16)
    twelve_logOH_med = np.percentile(twelve_logOH_samples, 50)
    twelve_logOH_hi = np.percentile(twelve_logOH_samples, 84)
    twelve_logOH_err = 0.5 * (twelve_logOH_hi - twelve_logOH_lo)

    return {
        'theta': theta_med,
        'theta_lo': theta_lo,
        'theta_hi': theta_hi,
        'theta_err': theta_err,
        'twelve_log_OH': twelve_logOH_med,
        'twelve_log_OH_lo': twelve_logOH_lo,
        'twelve_log_OH_hi': twelve_logOH_hi,
        'twelve_log_OH_err': twelve_logOH_err,
        'success': True,
        'n_ratios': n_used,
    }


# ---------------------------------------------------------------------------
# Two-stage informative-prior fit (Plan B)
# ---------------------------------------------------------------------------
# Stage 1 fits {ne_oii, te_oiii, Av} from the density / Balmer / OIII-auroral
# ratios (indices 0-3); Stage 2 then fits {log_O2_abund, log_O3_abund} from
# the abundance ratios (indices 4-5) while marginalizing over ne/Te/Av with
# informative priors built from the Stage-1 posterior. Splitting the joint
# fit this way collapses the degeneracy that makes the single-stage 5D fit
# pathological for low-SNR objects. Mirrors the collaborator's abundance stage.
_STAGE1_RATIO_IDX = (0, 1, 2, 3)   # n_e, A_V (x2), T_high
_STAGE2_RATIO_IDX = (4, 5)         # O+/H+, O++/H+
_I_O2 = PARAM_NAMES.index('log_O2_abund')
_I_O3 = PARAM_NAMES.index('log_O3_abund')


def _twelve_logOH_from_samples(samples):
    """Compute 12 + log10(O+/H+ + O++/H+) percentiles from posterior samples
    (columns in PARAM_NAMES order). Per-sample so the O+/O++ anti-correlation
    is captured. Returns (med, lo, hi, err)."""
    O_over_H_samples = 10 ** samples[:, _I_O2] + 10 ** samples[:, _I_O3]
    twelve_logOH_samples = 12 + np.log10(O_over_H_samples)
    lo = np.percentile(twelve_logOH_samples, 16)
    med = np.percentile(twelve_logOH_samples, 50)
    hi = np.percentile(twelve_logOH_samples, 84)
    return med, lo, hi, 0.5 * (hi - lo)


def _fit_row_ultranest_twostage(r, r_err, mask,
                                min_num_live_points=400,
                                verbose_sampler=False,
                                sampler_kwargs=None):
    """Two-stage informative-prior nested-sampling fit (Plan B).

    Returns the same result-dict contract as ``_fit_row_ultranest`` so the
    catalog driver and cache are agnostic to which method produced a row."""
    n_used = int(mask.sum())
    n_par = len(PARAM_NAMES)
    if n_used < 3:
        return _nan_fit_result(n_par, n_used)

    idx1 = [i for i in _STAGE1_RATIO_IDX if mask[i]]
    idx2 = [i for i in _STAGE2_RATIO_IDX if mask[i]]
    if len(idx1) < 3 or len(idx2) < 1:
        return _nan_fit_result(n_par, n_used, success=False)

    r1 = r[idx1]
    e1 = r_err[idx1]
    r2 = r[idx2]
    e2 = r_err[idx2]
    r1_col = r1[:, None]
    e1_col = e1[:, None]
    r2_col = r2[:, None]
    e2_col = e2[:, None]

    # --- Stage 1: {ne_oii, te_oiii, Av} with uniform priors --------------
    lows1 = PRIOR_LOWS[:3]
    highs1 = PRIOR_HIGHS[:3]
    spans1 = highs1 - lows1

    def transform1(cubes):
        return lows1 + np.asarray(cubes) * spans1

    def loglike1(thetas):
        thetas = np.asarray(thetas)
        if thetas.ndim == 1:
            # Pad with dummy abundances; ratios 0-3 don't depend on them.
            theta5 = np.concatenate([thetas, [-4.0, -4.0]])
            model = r_model(theta5)
            return -0.5 * np.sum(((r1 - model[idx1]) / e1) ** 2)
        pad = np.full((thetas.shape[0], 2), -4.0)
        theta5 = np.hstack([thetas, pad])
        models = r_model(theta5)                   # (n_ratios, N)
        resid = (r1_col - models[idx1]) / e1_col
        return -0.5 * np.sum(resid ** 2, axis=0)

    samples1 = _run_sampler(
        PARAM_NAMES[:3], loglike1, transform1,
        min_num_live_points=min_num_live_points,
        verbose_sampler=verbose_sampler,
        sampler_kwargs=sampler_kwargs,
    )
    if samples1 is None:
        return _nan_fit_result(n_par, n_used, success=False)

    # --- Stage 2: full 5-param fit; ne/Te/Av get informative priors ------
    ne_tf = ptform_1d_from_samples(samples1[:, 0], PRIOR_LOWS[0], PRIOR_HIGHS[0])
    te_tf = ptform_1d_from_samples(samples1[:, 1], PRIOR_LOWS[1], PRIOR_HIGHS[1])
    av_tf = ptform_1d_from_samples(samples1[:, 2], PRIOR_LOWS[2], PRIOR_HIGHS[2])
    o2_lo, o2_hi = PRIOR_LOWS[_I_O2], PRIOR_HIGHS[_I_O2]
    o3_lo, o3_hi = PRIOR_LOWS[_I_O3], PRIOR_HIGHS[_I_O3]
    o2_span = o2_hi - o2_lo
    o3_span = o3_hi - o3_lo

    def transform2(cubes):
        cubes = np.asarray(cubes)
        if cubes.ndim == 1:
            out = np.empty(5)
            out[0] = ne_tf(cubes[0])
            out[1] = te_tf(cubes[1])
            out[2] = av_tf(cubes[2])
            out[3] = o2_lo + cubes[3] * o2_span
            out[4] = o3_lo + cubes[4] * o3_span
            return out
        out = np.empty_like(cubes, dtype=float)
        out[:, 0] = ne_tf(cubes[:, 0])
        out[:, 1] = te_tf(cubes[:, 1])
        out[:, 2] = av_tf(cubes[:, 2])
        out[:, 3] = o2_lo + cubes[:, 3] * o2_span
        out[:, 4] = o3_lo + cubes[:, 4] * o3_span
        return out

    def loglike2(thetas):
        thetas = np.asarray(thetas)
        if thetas.ndim == 1:
            model = r_model(thetas)
            return -0.5 * np.sum(((r2 - model[idx2]) / e2) ** 2)
        models = r_model(thetas)                   # (n_ratios, N)
        resid = (r2_col - models[idx2]) / e2_col
        return -0.5 * np.sum(resid ** 2, axis=0)

    samples2 = _run_sampler(
        PARAM_NAMES, loglike2, transform2,
        min_num_live_points=min_num_live_points,
        verbose_sampler=verbose_sampler,
        sampler_kwargs=sampler_kwargs,
    )
    if samples2 is None:
        return _nan_fit_result(n_par, n_used, success=False)

    theta_lo = np.percentile(samples2, 16, axis=0)
    theta_med = np.percentile(samples2, 50, axis=0)
    theta_hi = np.percentile(samples2, 84, axis=0)
    theta_err = 0.5 * (theta_hi - theta_lo)

    oh_med, oh_lo, oh_hi, oh_err = _twelve_logOH_from_samples(samples2)

    return {
        'theta': theta_med,
        'theta_lo': theta_lo,
        'theta_hi': theta_hi,
        'theta_err': theta_err,
        'twelve_log_OH': oh_med,
        'twelve_log_OH_lo': oh_lo,
        'twelve_log_OH_hi': oh_hi,
        'twelve_log_OH_err': oh_err,
        'success': True,
        'n_ratios': n_used,
    }


# ---------------------------------------------------------------------------
# Public driver
# ---------------------------------------------------------------------------
def _fit_one_row(row, min_num_live_points, verbose_sampler, sampler_kwargs):
    """Top-level per-row worker. Defined at module level (not nested) so it
    can be pickled for joblib's loky workers."""
    r, r_err, mask = _build_ratios(row)
    return _fit_row_ultranest(
        r, r_err, mask,
        min_num_live_points=min_num_live_points,
        verbose_sampler=verbose_sampler,
        sampler_kwargs=sampler_kwargs,
    )


def _fit_one_row_twostage(row, min_num_live_points, verbose_sampler,
                          sampler_kwargs):
    """Top-level per-row worker for the two-stage fit (Plan B). Module-level
    so it can be pickled for joblib's loky workers."""
    r, r_err, mask = _build_ratios(row)
    return _fit_row_ultranest_twostage(
        r, r_err, mask,
        min_num_live_points=min_num_live_points,
        verbose_sampler=verbose_sampler,
        sampler_kwargs=sampler_kwargs,
    )


def compute_direct_metallicities(catalog,
                                 n_jobs=1,
                                 min_num_live_points=400,
                                 verbose=False,
                                 verbose_sampler=False,
                                 sampler_kwargs=None,
                                 use_informative_priors=False):
    """
    Compute direct-Te oxygen abundances row-by-row from a line-flux catalog
    using UltraNest nested sampling (matches Scholte+2026).

    Parameters
    ----------
    catalog : astropy.table.Table
        Must contain {LINE}_FLUX and {LINE}_FLUX_IVAR columns. Required lines:
        OIII_4363, OIII_5007, OII_3726, OII_3729, HBETA.
        At least one of HALPHA or HGAMMA must be present to constrain A_V.
    n_jobs : int
        Number of parallel worker processes for the per-row fits.
        1 (default) runs serially in the main process.
        -1 uses all available CPU cores.
        Requires joblib (pip install joblib).
        Recommended on NERSC compute nodes: set to the number of cores you
        allocated (e.g., 64 or 128 on Perlmutter CPU nodes). Do NOT use n_jobs
        > 1 on login nodes.
    min_num_live_points : passed to UltraNest's run().
    verbose : print per-row metallicity status. With n_jobs > 1 the per-row
        prints arrive out of order and are buffered; for clean progress
        tracking with parallel jobs use a joblib-aware progress bar instead.
    verbose_sampler : print UltraNest's own status output (default False).
        Strongly recommended to leave False when n_jobs > 1 -- otherwise
        many workers will interleave their UltraNest progress to stderr.
    sampler_kwargs : extra kwargs passed to sampler.run().
    use_informative_priors : bool
        If False (default) use the single-stage joint 5-parameter fit
        (``_fit_row_ultranest``, Plan A). If True use the two-stage
        informative-prior fit (``_fit_row_ultranest_twostage``, Plan B):
        fit ne/Te/Av first, then the abundances using the Stage-1 posteriors
        as priors. Both methods return the identical result-table schema.

    Returns
    -------
    astropy.table.Table with one row per input row, containing for each
    parameter (ne_oii, te_oiii, Av, log_O2_abund, log_O3_abund) the columns:
        {name}        : posterior median
        {name}_lo     : 16th percentile
        {name}_hi     : 84th percentile
        {name}_err    : 0.5 * (hi - lo)
    plus:
        twelve_log_OH, twelve_log_OH_lo, twelve_log_OH_hi, twelve_log_OH_err,
        n_ratios, fit_success.
    """
    if not isinstance(catalog, Table):
        raise TypeError('catalog must be an astropy.table.Table')

    n_rows = len(catalog)

    # Cache TARGETIDs if available so the verbose printer can show them.
    targetids = catalog['TARGETID'] if 'TARGETID' in catalog.colnames else None

    # Select the per-row fitter: single-stage (Plan A) or two-stage (Plan B).
    worker = _fit_one_row_twostage if use_informative_priors else _fit_one_row

    # Run the per-row fits (parallel or serial)
    if n_jobs == 1 or n_rows == 1:
        # Serial path -- avoids joblib overhead and preserves print ordering.
        fits = []
        for idx, row in enumerate(catalog):
            fit = worker(
                row, min_num_live_points, verbose_sampler, sampler_kwargs,
            )
            fits.append(fit)
            if verbose:
                tid = targetids[idx] if targetids is not None else None
                _print_row_status(idx, fit, targetid=tid)
    else:
        try:
            from joblib import Parallel, delayed
        except ImportError as e:
            raise ImportError(
                "n_jobs > 1 requires joblib (pip install joblib)"
            ) from e

        # joblib with 'loky' backend uses subprocesses -- each worker imports
        # this module fresh and rebuilds its own PyNeb interpolation grids.
        # That import cost (~few seconds per worker) happens once per worker
        # for the whole catalog, not per row, so it amortizes well.
        if verbose_sampler and n_jobs != 1:
            import warnings
            warnings.warn(
                "verbose_sampler=True with n_jobs>1 will produce interleaved "
                "output from many workers. Consider verbose_sampler=False."
            )

        fits = Parallel(n_jobs=n_jobs, backend='loky',
                        verbose=10 if verbose else 0)(
            delayed(worker)(
                row, min_num_live_points, verbose_sampler, sampler_kwargs,
            )
            for row in catalog
        )

        if verbose:
            # joblib's verbose=10 already prints progress; also print per-row
            # results in order now that all fits are complete.
            for idx, fit in enumerate(fits):
                tid = targetids[idx] if targetids is not None else None
                _print_row_status(idx, fit, targetid=tid)

    # Assemble output table
    out_cols = {}
    # Include TARGETID first if it was present in the input catalog.
    if targetids is not None:
        out_cols['TARGETID'] = np.asarray(targetids)
    for name in PARAM_NAMES:
        out_cols[name] = np.full(n_rows, np.nan)
        out_cols[f'{name}_lo'] = np.full(n_rows, np.nan)
        out_cols[f'{name}_hi'] = np.full(n_rows, np.nan)
        out_cols[f'{name}_err'] = np.full(n_rows, np.nan)
    out_cols['twelve_log_OH'] = np.full(n_rows, np.nan)
    out_cols['twelve_log_OH_lo'] = np.full(n_rows, np.nan)
    out_cols['twelve_log_OH_hi'] = np.full(n_rows, np.nan)
    out_cols['twelve_log_OH_err'] = np.full(n_rows, np.nan)
    out_cols['n_ratios'] = np.zeros(n_rows, dtype=int)
    out_cols['fit_success'] = np.zeros(n_rows, dtype=bool)

    for idx, fit in enumerate(fits):
        for i, name in enumerate(PARAM_NAMES):
            out_cols[name][idx] = fit['theta'][i]
            out_cols[f'{name}_lo'][idx] = fit['theta_lo'][i]
            out_cols[f'{name}_hi'][idx] = fit['theta_hi'][i]
            out_cols[f'{name}_err'][idx] = fit['theta_err'][i]
        out_cols['twelve_log_OH'][idx] = fit['twelve_log_OH']
        out_cols['twelve_log_OH_lo'][idx] = fit['twelve_log_OH_lo']
        out_cols['twelve_log_OH_hi'][idx] = fit['twelve_log_OH_hi']
        out_cols['twelve_log_OH_err'][idx] = fit['twelve_log_OH_err']
        out_cols['n_ratios'][idx] = fit['n_ratios']
        out_cols['fit_success'][idx] = fit['success']

    return Table(out_cols)


def _print_row_status(idx, fit, targetid=None):
    """Pretty-print a single row's fit result. Includes TARGETID if given."""
    prefix = f"row {idx}"
    if targetid is not None:
        prefix += f" (TARGETID={targetid})"
    if np.isfinite(fit['twelve_log_OH']):
        hi = fit['twelve_log_OH_hi'] - fit['twelve_log_OH']
        lo = fit['twelve_log_OH'] - fit['twelve_log_OH_lo']
        print(
            f"{prefix}: 12+log(O/H) = {fit['twelve_log_OH']:.3f} "
            f"(+{hi:.3f} / -{lo:.3f}) [n_ratios={fit['n_ratios']}]"
        )
    else:
        print(
            f"{prefix}: fit failed or insufficient data "
            f"[n_ratios={fit['n_ratios']}]"
        )
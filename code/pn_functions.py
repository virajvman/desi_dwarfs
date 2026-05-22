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
"""
import numpy as np
from astropy.table import Table
from scipy.optimize import minimize
from scipy.interpolate import RegularGridInterpolator
import pyneb as pn

from cardelli_attenuation import *

# ---------------------------------------------------------------------------
# Atomic data
# ---------------------------------------------------------------------------
pn.atomicData.addDataFilePath('./data/atomic_data/')

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

THETA0 = np.array([1e2, 1.2e4, 0.3, -4.0, -3.5])
BOUNDS = [
    (den_min, den_max),     # n_e [cm^-3]
    (tem_min, tem_max),     # T_high [K]
    (0.0, 5.0),             # A_V [mag]
    (-6.0, -2.0),           # log(O+/H+)
    (-6.0, -2.0),           # log(O++/H+)
]


def _flux_and_err(row, line):
    """Return (flux, sigma) for a line key like 'OIII_4363'.
    Treats missing columns, non-finite values, or ivar <= 0 as missing."""
    flux_col = f'{line}_FLUX'
    ivar_col = f'{line}_FLUX_IVAR'
    # astropy Row supports .colnames; fall back to membership tests otherwise
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


def _neg_log_like_masked(theta, r, r_err, mask):
    model = r_model(np.asarray(theta))
    return 0.5 * np.sum(((r[mask] - model[mask]) / r_err[mask]) ** 2)


def _fit_row(r, r_err, mask, theta0=THETA0, bounds=BOUNDS):
    n_used = int(mask.sum())
    if n_used < 3:
        nan_arr = np.full(len(theta0), np.nan)
        return nan_arr, nan_arr.copy(), False, n_used

    res = minimize(
        _neg_log_like_masked,
        theta0,
        args=(r, r_err, mask),
        method='L-BFGS-B',
        bounds=bounds,
    )

    try:
        cov = (res.hess_inv.todense()
               if hasattr(res.hess_inv, 'todense')
               else np.asarray(res.hess_inv))
        theta_err = np.sqrt(np.clip(np.diag(cov), 0, np.inf))
    except Exception:
        theta_err = np.full(len(theta0), np.nan)

    return res.x, theta_err, bool(res.success), n_used


def compute_direct_metallicities(catalog, theta0=THETA0, bounds=BOUNDS,
                                 verbose=False):
    """
    Compute direct-Te oxygen abundances row-by-row from a line-flux catalog.

    Parameters
    ----------
    catalog : astropy.table.Table
        Must contain {LINE}_FLUX and {LINE}_FLUX_IVAR columns. Required lines:
        OIII_4363, OIII_5007, OII_3726, OII_3729, HBETA.
        At least one of HALPHA or HGAMMA must be present to constrain A_V.
    theta0, bounds : starting point and bounds for scipy.optimize.minimize.
    verbose : print per-row status.

    Returns
    -------
    astropy.table.Table with one row per input row, containing:
        ne_oii, te_oiii, Av, log_O2_abund, log_O3_abund (and *_err for each),
        twelve_log_OH, twelve_log_OH_err, n_ratios, fit_success.
    """
    if not isinstance(catalog, Table):
        raise TypeError('catalog must be an astropy.table.Table')

    n_rows = len(catalog)
    out_cols = {name: np.full(n_rows, np.nan) for name in PARAM_NAMES}
    out_cols.update({f'{name}_err': np.full(n_rows, np.nan) for name in PARAM_NAMES})
    out_cols['twelve_log_OH'] = np.full(n_rows, np.nan)
    out_cols['twelve_log_OH_err'] = np.full(n_rows, np.nan)
    out_cols['n_ratios'] = np.zeros(n_rows, dtype=int)
    out_cols['fit_success'] = np.zeros(n_rows, dtype=bool)

    for idx, row in enumerate(catalog):
        r, r_err, mask = _build_ratios(row)
        theta, theta_err, success, n_used = _fit_row(r, r_err, mask, theta0, bounds)

        for i, name in enumerate(PARAM_NAMES):
            out_cols[name][idx] = theta[i]
            out_cols[f'{name}_err'][idx] = theta_err[i]

        log_O2 = theta[PARAM_NAMES.index('log_O2_abund')]
        log_O3 = theta[PARAM_NAMES.index('log_O3_abund')]
        if np.isfinite(log_O2) and np.isfinite(log_O3):
            O_over_H = 10**log_O2 + 10**log_O3
            twelve_logOH = 12 + np.log10(O_over_H)
            w2 = 10**log_O2 / O_over_H
            w3 = 10**log_O3 / O_over_H
            e_O2 = out_cols['log_O2_abund_err'][idx]
            e_O3 = out_cols['log_O3_abund_err'][idx]
            twelve_logOH_err = np.sqrt((w2 * e_O2) ** 2 + (w3 * e_O3) ** 2)
        else:
            twelve_logOH = np.nan
            twelve_logOH_err = np.nan

        out_cols['twelve_log_OH'][idx] = twelve_logOH
        out_cols['twelve_log_OH_err'][idx] = twelve_logOH_err
        out_cols['n_ratios'][idx] = n_used
        out_cols['fit_success'][idx] = success

        if verbose:
            print(f'row {idx}: 12+log(O/H) = {twelve_logOH:.3f} +/- '
                  f'{twelve_logOH_err:.3f} (n_ratios={n_used}, success={success})')

    return Table(out_cols)
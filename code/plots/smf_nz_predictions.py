"""
Stellar-mass-function (SMF) -> n(z) prediction helpers for the catalog paper.

This module centralizes the SMF definitions and the n(z) prediction machinery
that used to live inline in ``notebooks/catalog_paper/survey_plot.ipynb`` so the
same code can be reused by ``elg_property_explore.ipynb`` (and any other plot).

Two SMFs are supported:

* GAMA field SMF (Driver+2022, z<0.1 double Schechter)            -> ``gama_smf``
* SAGAbg SMF (Kado-Fong+2025, "SAGAbg III", Table 5), supplied as the published
  CDS/MRT table with pointwise percentiles (5/16/50/84/95) for several
  environment classes (all / satellite / field / isolated)       -> ``sagabg_smf``

All SMF callables take ``logM`` (log10 stellar mass, dex) and return the SMF
``dn/dlog10(M)`` in ``Mpc^-3 dex^-1`` so they can be integrated directly with
``scipy.integrate.quad`` over a stellar-mass bin.

Two n(z) prediction paths are provided (both kept on purpose):

* Analytic: integrate the SMF over the observable mass range at each redshift
  (``predict_nz_in_mass_bin``) -- used for the horizontal total-density band and
  any quick magnitude-limited prediction.
* Mock-catalog: sample an intrinsic (logM, g-r, M_r) catalog from the SMF and
  apply a distance modulus + magnitude cut (``build_intrinsic_catalog`` +
  ``nz_from_intrinsic``) -- used for the magnitude-limited prediction lines.

The grey shaded "expected total density" band is built by integrating the SMF
over a stellar-mass bin (volume-limited), so no mock catalog is needed for it.
A single string ``mode`` flag selects between:

* ``"saga_posterior"`` : nested 68% (16-84) / outer 90% (5-95) band from the
  SAGAbg percentiles (5-95 is the widest interval in the published table).
* ``"saga_vs_gama"``    : single band spanning the SAGAbg median curve and the
  GAMA (Driver+22) best-fit curve.
"""
import os

import numpy as np
from scipy.integrate import quad, cumulative_trapezoid
from scipy.interpolate import UnivariateSpline, interp1d
from scipy.optimize import brentq
from astropy.cosmology import Planck18

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
SAGABG_POSTERIOR_PATH = os.path.join(REPO_ROOT, "data", "SAGAbg_smf_posterior.txt")

# Default SAGAbg environment class to use (one of: all, satellite, field, isolated).
SAGABG_ENV = "all"

# Recognized environment labels in the SAGAbg MRT table.
_SAGABG_ENVS = ("all", "satellite", "field", "isolated")

cosmo = Planck18

# --------------------------------------------------------------------------
# Sampling / color-relation constants (shared by the mock-catalog path)
# --------------------------------------------------------------------------
LOGM_SAMPLE_MIN = 6.0
LOGM_SAMPLE_MAX = 12.0
GR_SCATTER = 0.1


# --------------------------------------------------------------------------
# SMF definitions
# --------------------------------------------------------------------------
def schechter_logm(logM, phi_star, logMc, alpha):
    """Single Schechter component as ``dn/dlog10(M)`` [Mpc^-3 dex^-1].

    ``logM`` and ``logMc`` are log10 stellar masses (dex); ``phi_star`` is the
    normalization in ``Mpc^-3``.
    """
    x = 10.0 ** (logM - logMc)
    return np.log(10.0) * phi_star * x ** (alpha + 1.0) * np.exp(-x)


# Driver+2022, z<0.1 GAMA double Schechter (shared M* between components).
_GAMA_LOGMC = 10.745
_GAMA_COMPONENTS = (
    (10.0 ** -2.437, -0.466),
    (10.0 ** -3.201, -1.530),
)


def gama_smf(logM):
    """GAMA field SMF (Driver+2022) as ``dn/dlog10(M)`` [Mpc^-3 dex^-1]."""
    total = 0.0
    for phi_star, alpha in _GAMA_COMPONENTS:
        total = total + schechter_logm(logM, phi_star, _GAMA_LOGMC, alpha)
    return total


def load_sagabg_posterior(path=SAGABG_POSTERIOR_PATH, env=SAGABG_ENV):
    """Load the SAGAbg SMF percentile table (Kado-Fong+2025, Table 5).

    The published CDS/MRT file has a header block followed by whitespace-delimited
    rows with columns::

        log(M*min)  log(M*max)  SMF05  SMF16  SMF50  SMF84  SMF95  Env

    The SMF columns are *linear* ``dn/dlog10(M*)`` [Mpc^-3 dex^-1] at the 5/16/50/84/95
    percentiles, and ``Env`` is one of ``all``/``satellite``/``field``/``isolated``.
    Only rows matching ``env`` are kept. The stellar-mass coordinate is the bin
    center, ``0.5*(log M*min + log M*max)``.

    Returns a dict with the ``logM`` grid plus callables for each percentile
    (``central`` = SMF50, ``p5``, ``p16``, ``p50``, ``p84``, ``p95``) that map
    ``logM`` to the SMF in linear units. Interpolation is done in log10(SMF) vs
    logM, with constant edge extrapolation just outside the tabulated bins.
    """
    if env not in _SAGABG_ENVS:
        raise ValueError(f"Unknown env {env!r}; expected one of {_SAGABG_ENVS}.")

    rows = []
    with open(path) as fh:
        for line in fh:
            parts = line.split()
            if len(parts) == 8 and parts[-1] == env:
                try:
                    vals = [float(x) for x in parts[:7]]
                except ValueError:
                    continue
                rows.append(vals)
    if not rows:
        raise ValueError(f"No rows found for env={env!r} in {path}.")

    arr = np.array(rows)
    logM = 0.5 * (arr[:, 0] + arr[:, 1])           # bin center
    order = np.argsort(logM)
    logM = logM[order]
    # linear SMF percentile columns
    smf = {
        "p5": arr[order, 2],
        "p16": arr[order, 3],
        "p50": arr[order, 4],
        "p84": arr[order, 5],
        "p95": arr[order, 6],
    }

    def _make_callable(smf_col):
        log_smf_col = np.log10(smf_col)
        spline = interp1d(
            logM, log_smf_col, kind="linear",
            bounds_error=False, fill_value=(log_smf_col[0], log_smf_col[-1]),
        )
        return lambda lm: 10.0 ** spline(lm)

    posterior = {"logM": logM, "env": env}
    for key, col in smf.items():
        posterior[key] = _make_callable(col)
    posterior["central"] = posterior["p50"]
    return posterior


def sagabg_smf(logM, posterior, quantile="central"):
    """Evaluate the SAGAbg SMF at ``logM`` for the requested percentile.

    ``quantile`` is one of ``central`` (=``p50``), ``p5``, ``p16``, ``p50``,
    ``p84``, ``p95``.
    """
    return posterior[quantile](logM)


# --------------------------------------------------------------------------
# Color / observability relations
# --------------------------------------------------------------------------
def get_mstar_gr(log_mstar):
    """Mean g-r color as a function of stellar mass (capped at 0.8)."""
    ave_gr = 0.09 * log_mstar - 0.3857
    return np.minimum(0.8, ave_gr)


def Mr_from_logM_gr(logM, gr):
    """Inverse of  logM = 1.986 + 0.950*gr - 0.365*M_r."""
    return (1.986 + 0.950 * gr - logM) / 0.365


def get_logM_min_observable(zred, rmag_limit, logM_floor=3.0, logM_ceil=12.0):
    """
    Find the minimum observable log M* at redshift `zred` for an r-band
    magnitude limit `rmag_limit`, using a mass-dependent g-r color.

    Stellar mass relation (Mg-based, converted to r):
        logM = 1.986 + 1.315 * gr - 0.365 * M_g
             = 1.986 + 0.950 * gr - 0.365 * M_r       (since M_g = M_r + gr)

    Color relation:
        gr = get_mstar_gr(logM)   (mass-dependent, capped at 0.8)

    These are coupled. We solve self-consistently by finding the root of
        f(logM) = logM - [1.986 + 0.950 * gr - 0.365 * M_r] = 0
    """
    # Apparent r -> absolute M_r at this redshift
    lumi_dist = cosmo.luminosity_distance(zred).value      # Mpc
    mu = 5.0 * np.log10(lumi_dist * 1e6) - 5.0
    M_r = rmag_limit - mu

    # Closed-form logM that the relation would give for a *specified* gr
    def logM_from_gr(gr):
        return 1.986 + 0.950 * gr - 0.365 * M_r

    # f(logM) = predicted logM (using gr=gr(logM)) minus logM itself.
    # Root of f = self-consistent solution.
    def f(logM):
        return logM_from_gr(get_mstar_gr(logM)) - logM

    # Check that a root exists in the bracket. If not, the function is
    # monotonically positive or negative -> handle edge cases.
    f_lo, f_hi = f(logM_floor), f(logM_ceil)
    if f_lo * f_hi > 0:
        # No sign change: either the limit is so deep that even logM_floor
        # is observable (f_lo < 0 everywhere) or so shallow that nothing is
        # (f_hi > 0 everywhere).
        return logM_floor if f_lo < 0 else logM_ceil

    return brentq(f, logM_floor, logM_ceil)


# --------------------------------------------------------------------------
# n(z) prediction -- analytic (integrate SMF over observable mass range)
# --------------------------------------------------------------------------
def predict_nz_in_mass_bin(smf_func, z_edges, logM_lo, logM_hi, rmag_limit=None):
    """
    Predict n(z) [Mpc^-3] in stellar-mass bin [logM_lo, logM_hi] under an
    optional r-band magnitude limit. Uses the self-consistent color-mass
    relation to determine the observable mass floor at each z.
    """
    z_edges = np.asarray(z_edges)
    nz = np.zeros(len(z_edges) - 1)
    for i in range(len(z_edges) - 1):
        z_mid = 0.5 * (z_edges[i] + z_edges[i + 1])

        if rmag_limit is None:
            lo_eff = logM_lo
        else:
            logM_min_obs = get_logM_min_observable(z_mid, rmag_limit)
            lo_eff = max(logM_lo, logM_min_obs)

        if lo_eff >= logM_hi:
            nz[i] = 0.0
            continue

        # Integrate SMF (dn/dlogM) over the observable mass range
        nz[i], _ = quad(smf_func, lo_eff, logM_hi)
    return nz


# --------------------------------------------------------------------------
# n(z) prediction -- mock intrinsic catalog (for magnitude-limited panels)
# --------------------------------------------------------------------------
def build_inverse_cdf(smf_func, logM_lo=LOGM_SAMPLE_MIN, logM_hi=LOGM_SAMPLE_MAX,
                      n_grid=1000):
    """Inverse-CDF spline (for sampling) + total density of ``smf_func``.

    The CDF is built by cumulative trapezoidal integration on a dense logM grid
    rather than ``scipy.integrate.quad`` called per grid point. ``quad`` assumes
    a smooth integrand and silently loses accuracy on the SAGAbg interpolant
    (piecewise-linear in log10(SMF), ~40 kinks): it exhausts its subdivision
    limit and can return a slightly *non-monotonic* cumulative curve, which then
    makes ``UnivariateSpline(cdf, ..., s=0)`` raise "x must be strictly
    increasing". Evaluating the SMF on a grid far finer than the tabulated bins
    and integrating with the trapezoid rule is accurate to <1% on the bin
    densities and guarantees a strictly increasing CDF for both the GAMA
    (smooth) and SAGAbg (tabulated) SMFs. The returned ``(inv_cdf, norm)``
    signature is unchanged, so the downstream mock-catalog path is identical.
    """
    grid = np.linspace(logM_lo, logM_hi, n_grid)
    smf_vals = np.asarray([smf_func(m) for m in grid], dtype=float).ravel()
    cdf = np.concatenate(([0.0], cumulative_trapezoid(smf_vals, grid)))
    norm = cdf[-1]
    cdf /= norm
    return UnivariateSpline(cdf, grid, s=0, k=3), norm


def build_intrinsic_catalog(inv_cdf, n_samples=1_000_000, gr_scatter=GR_SCATTER,
                            seed=42):
    """
    Sample (logM, gr, M_r) from the SMF + color relation.
    No redshift assigned -- we apply distance modulus at evaluation time.
    """
    rng = np.random.default_rng(seed)
    u_rnd = rng.uniform(0, 1, size=n_samples)
    logM = inv_cdf(u_rnd)
    gr = get_mstar_gr(logM) + rng.normal(0.0, gr_scatter, size=n_samples)
    M_r = Mr_from_logM_gr(logM, gr)
    return {"logM": logM, "gr": gr, "M_r": M_r}


def nz_from_intrinsic(cat, mean_density, z_centers, logM_lo, logM_hi,
                      rmag_limit=None):
    """
    n(z) [Mpc^-3] from an intrinsic mock catalog.

    Logic: the intrinsic catalog represents the population in some fixed
    comoving volume. The fraction of galaxies in [logM_lo, logM_hi] that
    are observable at redshift z (m_r < rmag_limit) gives the *fraction* of
    the SMF visible. Multiply by the SMF density in that bin to get n(z).
    """
    z_centers = np.asarray(z_centers)
    in_mass_bin = (cat["logM"] >= logM_lo) & (cat["logM"] < logM_hi)
    N_in_bin = in_mass_bin.sum()
    if N_in_bin == 0:
        return np.zeros(len(z_centers))

    # Density in this mass bin (Mpc^-3) -- fraction of total times mean_density
    density_in_bin = mean_density * N_in_bin / len(cat["logM"])

    if rmag_limit is None:
        # No cut: density is z-independent
        return np.full(len(z_centers), density_in_bin)

    M_r_bin = cat["M_r"][in_mass_bin]
    nz = np.empty(len(z_centers))
    for i, z in enumerate(z_centers):
        mu = 5.0 * np.log10(cosmo.luminosity_distance(z).value * 1e6) - 5.0
        frac_visible = (M_r_bin + mu < rmag_limit).mean()
        nz[i] = density_in_bin * frac_visible
    return nz


# --------------------------------------------------------------------------
# Total-density band (volume-limited): integrate SMF over a mass bin
# --------------------------------------------------------------------------
def total_density_in_bin(smf_func, logM_lo, logM_hi):
    """Volume-limited total number density [Mpc^-3] in [logM_lo, logM_hi]."""
    return quad(smf_func, logM_lo, logM_hi)[0]


def total_density_band(logM_lo, logM_hi, mode, posterior, gama_smf_func=gama_smf):
    """
    Expected total number density in a stellar-mass bin and its band edges.

    mode == "saga_posterior":
        Integrate the SAGAbg percentile curves over the mass bin.
        Returns {"central", "lo68", "hi68", "lo_out", "hi_out"} where the inner
        band is the 16-84 percentile (68%) and the outer band is the 5-95
        percentile (90%, the widest interval in the published table).
        NOTE: the table stores *pointwise* percentiles, so integrating the
        16/84 (and 5/95) curves over the bin is an approximation to the true
        band on the integrated density (it ignores cross-bin covariance), but is
        the natural estimate given the published summary.

    mode == "saga_vs_gama":
        Band spans the SAGAbg median curve and the GAMA (Driver+22) best-fit.
        Returns {"central", "lo", "hi"} (central = SAGAbg).
    """
    if mode == "saga_posterior":
        return {
            "central": total_density_in_bin(posterior["central"], logM_lo, logM_hi),
            "lo68": total_density_in_bin(posterior["p16"], logM_lo, logM_hi),
            "hi68": total_density_in_bin(posterior["p84"], logM_lo, logM_hi),
            "lo_out": total_density_in_bin(posterior["p5"], logM_lo, logM_hi),
            "hi_out": total_density_in_bin(posterior["p95"], logM_lo, logM_hi),
        }
    if mode == "saga_vs_gama":
        saga_c = total_density_in_bin(posterior["central"], logM_lo, logM_hi)
        gama_c = total_density_in_bin(gama_smf_func, logM_lo, logM_hi)
        return {
            "central": saga_c,
            "lo": min(saga_c, gama_c),
            "hi": max(saga_c, gama_c),
            "saga": saga_c,
            "gama": gama_c,
        }
    raise ValueError(
        f"Unknown mode {mode!r}; expected 'saga_posterior' or 'saga_vs_gama'."
    )


def _fuzzy_fill(ax, x, lo, hi, color, alpha_core=0.15, n_layers=15,
                spread_dex=0.1, alpha_each=0.025, label=None):
    """Soft-edged horizontal fill between ``lo`` and ``hi`` (log-space fuzz)."""
    log_hi = np.log10(np.full_like(x, hi, dtype=float))
    log_lo = np.log10(np.full_like(x, lo, dtype=float))
    for j in range(n_layers):
        frac = (j + 1) / n_layers
        offset = spread_dex * frac
        ax.fill_between(
            x, 10.0 ** (log_hi + offset), 10.0 ** (log_lo - offset),
            facecolor=color, edgecolor="none", alpha=alpha_each,
            label=label if j == 0 else None,
        )
    ax.fill_between(
        x, 10.0 ** log_hi, 10.0 ** log_lo,
        facecolor=color, edgecolor="none", alpha=alpha_core,
    )


def plot_total_density_band(ax, z_range, logM_lo, logM_hi, mode, posterior,
                            gama_smf_func=gama_smf, color="k", label=None,
                            alpha_out=0.12, alpha68=0.22):
    """
    Draw the expected total-density band as a horizontal region across
    ``z_range = (z_min, z_max)``.

    * "saga_posterior" : nested outer 90% (5-95, lighter) + inner 68% (16-84,
      darker) grey bands.
    * "saga_vs_gama"   : single soft-edged band spanning SAGAbg & GAMA.

    Returns the band dict from ``total_density_band``.
    """
    band = total_density_band(logM_lo, logM_hi, mode, posterior,
                              gama_smf_func=gama_smf_func)
    x = np.asarray(z_range, dtype=float)

    if mode == "saga_posterior":
        # Outer 5-95 band, then inner 16-84 band on top.
        ax.fill_between(x, band["lo_out"], band["hi_out"],
                        facecolor=color, edgecolor="none", alpha=alpha_out,
                        label=label)
        ax.fill_between(x, band["lo68"], band["hi68"],
                        facecolor=color, edgecolor="none", alpha=alpha68)
    else:  # "saga_vs_gama"
        _fuzzy_fill(ax, x, band["lo"], band["hi"], color, label=label)

    return band

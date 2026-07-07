'''
Shared machinery for the spectroscopic incompleteness pipeline.

Deliberately CPU-only (no cupy import) so it can be used on both GPU nodes
(NNMF training) and CPU nodes (generation, feature measurement).
'''

import os
import numpy as np
from astropy.io import fits
from scipy.optimize import nnls

import config


# =============================================================================
# Wavelength grids
# =============================================================================
def get_rest_wave():
    """Common rest-frame log-lambda grid for both NNMF bases."""
    n = np.log10(config.REST_WAVEMAX / config.REST_WAVEMIN) / config.REST_DLOGLAM
    return 10 ** (np.log10(config.REST_WAVEMIN) + config.REST_DLOGLAM * np.arange(n))


def get_obs_wave():
    """Observed-frame grid handed to the forward-model backend."""
    return np.arange(config.OBS_WAVEMIN, config.OBS_WAVEMAX, config.OBS_DWAVE)


# =============================================================================
# De-redshifting / resampling (flux-conserving)
# =============================================================================
def deredshift_resample_one(wave_obs, flux_obs, ivar_obs, zred, wave_out):
    """De-redshift one spectrum onto `wave_out` with a flux-conserving rebin.

    f_lambda conventions: lambda_rest = lambda_obs/(1+z),
    f_rest = f_obs*(1+z), ivar_rest = ivar_obs/(1+z)^2.

    Pixels of `wave_out` outside the de-redshifted coverage get ivar=0.
    Uses the pure area-preserving rebin (NOT ivar-weighted) so emission-line
    cores are not suppressed -- important for both NNMF training and feature
    fits (see plan: flux-conserving resampling decision).
    """
    from desispec.interpolation import resample_flux

    rest_wave = wave_obs / (1.0 + zred)
    flux_rest = flux_obs * (1.0 + zred)
    ivar_rest = ivar_obs / (1.0 + zred) ** 2

    inside = (wave_out >= rest_wave[0]) & (wave_out <= rest_wave[-1])
    flux_out = np.zeros_like(wave_out)
    ivar_out = np.zeros_like(wave_out)
    if inside.sum() > 1:
        flux_out[inside] = resample_flux(wave_out[inside], rest_wave, flux_rest, ivar=None)
        _, ivar_out[inside] = resample_flux(wave_out[inside], rest_wave, flux_rest,
                                            ivar=ivar_rest)
    return flux_out, ivar_out


# =============================================================================
# Labels + catalog selection (single source of truth)
# =============================================================================
def catastrophic_z(z_true, z_meas, threshold=config.CATASTROPHIC_THRESHOLD):
    """True where the measured redshift is catastrophically wrong (~1000 km/s)."""
    return np.abs(np.asarray(z_meas) - np.asarray(z_true)) / (1.0 + np.asarray(z_true)) > threshold


def in_catalog_selection(z_rr, zwarn, spectype):
    """Would this (mock or real) redrock result enter the dwarf catalog?

    Mirrors the basic_cleaning + ZWARN cut in construct_dwarf_galaxy_catalogs.py.
    Used to define the PURITY model training population.
    """
    z_rr = np.asarray(z_rr)
    zwarn = np.asarray(zwarn)
    spectype = np.char.strip(np.asarray(spectype).astype(str))
    sel = (z_rr > config.CATALOG_Z_MIN) & (z_rr < config.CATALOG_Z_MAX)
    sel &= np.isin(zwarn, config.CATALOG_ZWARN_ALLOWED)
    sel &= (spectype == config.CATALOG_SPECTYPE)
    return sel


# =============================================================================
# Synthetic fiber magnitude (identical for mocks and real data)
# =============================================================================
_FILTER_CACHE = {}


def _get_filter():
    if config.FIBER_FILTER not in _FILTER_CACHE:
        import speclite.filters
        _FILTER_CACHE[config.FIBER_FILTER] = speclite.filters.load_filters(config.FIBER_FILTER)
    return _FILTER_CACHE[config.FIBER_FILTER]


def synthetic_fiber_maggies(wave, flux_1e17):
    """Filter-integrated AB maggies of a (fiber) spectrum.

    Parameters
    ----------
    wave : observed-frame wavelength [A]
    flux_1e17 : f_lambda in DESI units (1e-17 erg/s/cm^2/A); may be a 2D
        array (n_spec, n_pix). Noisy spectra can integrate to <= 0 maggies;
        that is fine, downstream uses asinh magnitudes.
    """
    import astropy.units as u
    filt = _get_filter()
    flux_cgs = np.atleast_2d(flux_1e17) * 1e-17 * u.erg / (u.cm ** 2 * u.s * u.Angstrom)
    # pad spectra that do not fully cover the filter response
    maggies_tab = filt.get_ab_maggies(flux_cgs, wave * u.Angstrom,
                                      axis=-1, mask_invalid=True)
    return np.asarray(maggies_tab[config.FIBER_FILTER])


def asinh_mag_from_maggies(maggies, softening_nmgy=config.ASINH_SOFTENING_NMGY):
    """Asinh (luptitude) magnitude; well-defined for maggies <= 0."""
    b = softening_nmgy * 1e-9  # softening in maggies
    return -2.5 / np.log(10) * (np.arcsinh(np.asarray(maggies) / (2 * b)) + np.log(b))


def synthetic_fiber_mag_asinh(wave, flux_1e17):
    """RFIB_ASINH feature: asinh fiber mag of the coadded (fiber) spectrum."""
    return asinh_mag_from_maggies(synthetic_fiber_maggies(wave, flux_1e17))


def pogson_mag_to_maggies(mag):
    return 10 ** (-0.4 * np.asarray(mag))


# =============================================================================
# Feature-basis (W_feat) fit with partial-wavelength-overlap handling
# =============================================================================
def fit_feature_basis(W_feat, wave_rest_grid, wave_obs, flux_obs, ivar_obs, z_rr):
    """Fit the coarse feature basis to one observed spectrum at z_rr.

    De-redshifts the observed spectrum at z_rr (which may be catastrophically
    wrong -- that is by design: features are measured quantities), fits W_feat
    by non-negative least squares over the OVERLAPPING pixels only, and
    returns measured-feature summaries.

    Returns
    -------
    shape_frac : (n_templates,) coefficients rescaled to unit sum
        (all zero if the fit has no support)
    redchi2 : reduced chi2 of the fit over good pixels
    overlap_frac : fraction of the rest grid with usable data after
        de-redshifting at z_rr (small => z_rr likely very wrong)
    coeffs : raw NNLS coefficients (amplitude information, for diagnostics)
    """
    n_templates = W_feat.shape[1]
    flux_r, ivar_r = deredshift_resample_one(wave_obs, flux_obs, ivar_obs, z_rr,
                                             wave_rest_grid)
    good = ivar_r > 0
    n_good = int(good.sum())
    overlap_frac = n_good / len(wave_rest_grid)

    if n_good <= n_templates:
        return np.zeros(n_templates), 0.0, overlap_frac, np.zeros(n_templates)

    sqrt_ivar = np.sqrt(ivar_r[good])
    A = sqrt_ivar[:, None] * W_feat[good, :]
    b = sqrt_ivar * flux_r[good]
    coeffs, rnorm = nnls(A, b)

    chi2 = rnorm * rnorm
    redchi2 = chi2 / max(n_good - n_templates, 1)

    total = coeffs.sum()
    shape_frac = coeffs / total if total > 0 else np.zeros(n_templates)
    return shape_frac, redchi2, overlap_frac, coeffs


# =============================================================================
# Emission-line strength score + stratification (for oversampling the pool)
# =============================================================================
def emission_line_score(wave_rest, flux_rest):
    """Crude summed pseudo-EW over the standard dwarf line set.

    For each line: (mean flux in +/-LINE_HALFWIDTH window) minus (mean flux in
    two continuum sidebands), times the window width, divided by the sideband
    level. Positive = emission-dominated; ~0 or negative = passive.
    Operates on (n_pix,) or (n_pix, n_obj) arrays.
    """
    flux_rest = np.atleast_2d(flux_rest.T).T  # (n_pix, n_obj)
    score = np.zeros(flux_rest.shape[1])
    for center in config.LINE_CENTERS:
        line = np.abs(wave_rest - center) <= config.LINE_HALFWIDTH
        side = ((np.abs(wave_rest - (center - config.LINE_SIDEBAND_OFFSET))
                 <= config.LINE_SIDEBAND_HALFWIDTH)
                | (np.abs(wave_rest - (center + config.LINE_SIDEBAND_OFFSET))
                   <= config.LINE_SIDEBAND_HALFWIDTH))
        if line.sum() == 0 or side.sum() == 0:
            continue
        line_mean = flux_rest[line].mean(axis=0)
        cont_mean = flux_rest[side].mean(axis=0)
        width = 2.0 * config.LINE_HALFWIDTH
        denom = np.where(cont_mean > 0, cont_mean, np.nan)
        score += np.nan_to_num((line_mean - cont_mean) * width / denom)
    return np.squeeze(score)


def assign_strata(scores, n_strata=config.N_STRATA):
    """Assign each pool galaxy to an emission-strength stratum (0 = most passive).

    Stratum 0 collects everything with score <= a low threshold (passive /
    weak-emission); the remaining objects are split into n_strata-1 quantile
    bins of their scores. Returns (strata array, bin edges of the >threshold part).
    """
    scores = np.asarray(scores)
    passive_thresh = 2.0  # summed pseudo-EW [A]; below this = "passive" stratum
    strata = np.zeros(len(scores), dtype=int)
    active = scores > passive_thresh
    if active.sum() > 0:
        edges = np.quantile(scores[active], np.linspace(0, 1, n_strata))
        edges[0], edges[-1] = -np.inf, np.inf
        strata[active] = 1 + np.clip(np.digitize(scores[active], edges) - 1, 0,
                                     n_strata - 2)
    return strata, None if active.sum() == 0 else edges


def stratified_bootstrap_indices(strata, n_draws, rng):
    """Draw pool indices uniformly across strata (equal expected counts/stratum)."""
    unique = np.unique(strata)
    per = np.full(len(unique), n_draws // len(unique))
    per[: n_draws % len(unique)] += 1
    out = []
    for s, n in zip(unique, per):
        members = np.flatnonzero(strata == s)
        out.append(rng.choice(members, size=n, replace=True))
    out = np.concatenate(out)
    rng.shuffle(out)
    return out


# =============================================================================
# Effective exposure time sampling
# =============================================================================
def build_efftime_distribution(bgs_tsnr2_bgs, lowz_tsnr2_lrg, out_npz):
    """Convert real TSNR2 columns to EFFTIME and save empirical distributions."""
    eff_bgs = config.TSNR2_BGS_TO_EFFTIME * np.asarray(bgs_tsnr2_bgs)
    eff_lowz = config.TSNR2_LRG_TO_EFFTIME * np.asarray(lowz_tsnr2_lrg)
    eff_bgs = eff_bgs[np.isfinite(eff_bgs) & (eff_bgs > 0)]
    eff_lowz = eff_lowz[np.isfinite(eff_lowz) & (eff_lowz > 0)]
    np.savez(out_npz, efftime_bgs=eff_bgs, efftime_lowz=eff_lowz)
    return eff_bgs, eff_lowz


def sample_efftime(n, rng, npz_path=None):
    """Sample effective exposure times from the saved empirical distributions.

    A realistic *spread* of depths is simulated (mix of BGS-Bright-like and
    LOWZ-Dark-like), but efftime is NOT an NN feature -- Dchi2 is the SNR
    proxy (flagged assumption, see config).
    """
    npz_path = npz_path or config.REFERENCE_EFFTIME_NPZ
    data = np.load(npz_path)
    take_bgs = rng.random(n) < config.EFFTIME_MIX_BGS
    out = np.empty(n)
    out[take_bgs] = rng.choice(data["efftime_bgs"], size=int(take_bgs.sum()))
    out[~take_bgs] = rng.choice(data["efftime_lowz"], size=int((~take_bgs).sum()))
    return out


# =============================================================================
# Redrock-input SCORES hack (feasibgs omits the HDU redrock expects)
# =============================================================================
def add_scores_hdu(file_path, n_rows,
                   template_fits=config.SCORES_TEMPLATE_FITS):
    """Append a SCORES HDU (copied+truncated from a real coadd) to a mock file."""
    with fits.open(file_path, mode="update") as hdulist:
        table_hdu = fits.open(template_fits)["SCORES"]
        new_data = np.resize(table_hdu.data, n_rows)
        hdulist.append(fits.BinTableHDU(data=new_data, header=table_hdu.header,
                                        name="SCORES"))
        hdulist.flush()


# =============================================================================
# NN feature assembly + MLP training recipe
# =============================================================================
def assemble_features(table):
    """Build the (n, n_features) design matrix from a training/apply table.

    Column order defined by config.FEATURE_NAMES. Note SHAPE_FRAC drops the
    last fraction (unit-sum makes it redundant).
    """
    cols = []
    for name in config.FEATURE_NAMES:
        cols.append(np.asarray(table[name], dtype=float))
    X = np.column_stack(cols)
    return X


def train_mlp_classifier(X_train, y_train, seed=config.MLP_SEED, verbose=True):
    """Warm-start MLP training recipe (Harris & Speagle 2023 style, as in
    old nn_tutorial.ipynb): progressively larger data subsets with a
    decaying learning-rate schedule. Returns (scaler, clf)."""
    from sklearn.neural_network import MLPClassifier
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler().fit(X_train)
    Xs = scaler.transform(X_train)

    lrs = config.MLP_LEARNING_RATES
    clf = MLPClassifier(solver="adam", activation="relu",
                        hidden_layer_sizes=config.MLP_HIDDEN_LAYERS,
                        alpha=config.MLP_ALPHA,
                        learning_rate="constant", learning_rate_init=lrs[0],
                        batch_size=config.MLP_BATCH_SIZE,
                        max_iter=50, n_iter_no_change=np.inf,
                        random_state=seed,
                        warm_start=True, verbose=verbose)

    for subset in (20, 10, 5):
        schedule = lrs if subset == 20 else lrs[5:]
        for lr in schedule:
            clf.learning_rate_init = lr
            clf.fit(Xs[::subset], y_train[::subset])
    for lr in lrs[5:]:
        clf.learning_rate_init = lr
        clf.fit(Xs, y_train)
    clf.learning_rate_init = lrs[-1]
    clf.max_iter = 100
    clf.fit(Xs, y_train)
    return scaler, clf


def predict_proba(scaler, clf, X):
    """P(correct redshift) for the feature matrix X."""
    return clf.predict_proba(scaler.transform(X))[:, 1]


def check_dirs():
    for d in (config.REFERENCE_DIR, config.BASES_DIR, config.MOCK_DIR,
              config.REDROCK_DIR, config.TRAINING_DIR, config.MODELS_DIR,
              config.VALIDATION_DIR, config.CALIB_DIR):
        os.makedirs(d, exist_ok=True)

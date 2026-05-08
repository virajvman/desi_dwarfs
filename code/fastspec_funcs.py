from astropy.cosmology import Planck18
from desispec.interpolation import resample_flux
###FUNCTIONS BELOW TO HELP WITH NEBULAR EMISSION PHOTO TESTS

import numpy as np
from astropy.io import fits
from astropy.table import Table
import speclite.filters
from astropy import units as u
import h5py
import multiprocessing as mp

# Load filters once at module level
DECAM_G = speclite.filters.load_filters('decamDR1noatm-g')
DECAM_R = speclite.filters.load_filters('decamDR1noatm-r')

BASS_G = speclite.filters.load_filters('BASS-g')
BASS_R = speclite.filters.load_filters('BASS-r')

SDSS_G = speclite.filters.load_filters('sdss2010noatm-g')
SDSS_R = speclite.filters.load_filters('sdss2010noatm-r')


def get_fastspecfit_path(survey, program, healpix,
                         base_dir="/global/cfs/cdirs/desi/public/dr1/vac/dr1/fastspecfit/iron/v2.1/healpix"):
    healpix_parent = healpix // 100
    filename = f"fastspec-{survey}-{program}-{healpix}.fits.gz"
    return f"{base_dir}/{survey}/{program}/{healpix_parent}/{healpix}/{filename}"


def get_filter_weights(filt, wave_arr):
    """
    Interpolate a speclite FilterResponse onto an arbitrary wavelength
    grid and return per-pixel weights for the AB maggies integral.

    The weights w_i satisfy:
        maggies = sum(f_lambda_i * w_i)   [dimensionless]
    so that:
        mag = -2.5 * log10(maggies)
    and:
        var_maggies = sum(w_i^2 * var_flambda_i)

    Parameters
    ----------
    filt : speclite FilterResponse
    wave_arr : 1D array, shape (Nwave,), in Angstrom

    Returns
    -------
    weights : 1D array, shape (Nwave,)
    """
    # Handle both Quantity and plain ndarray wavelengths
    filt_wave = filt.wavelength
    if hasattr(filt_wave, 'to'):
        filt_wave = filt_wave.to("Angstrom").value

    R = np.interp(wave_arr, filt_wave, filt.response, left=0.0, right=0.0)

    c_ang = 2.998e18
    f_nu_ref = 3.631e-20

    dlam = np.gradient(wave_arr)
    num_weights = R * wave_arr * dlam
    denom = f_nu_ref * c_ang * np.sum(R / wave_arr * dlam)

    return num_weights / denom


def measure_photo_batch(wave_arr, flux_2d, ivar_2d=None, zred=None,
                        measure_bass=False,
                        measure_sdss=False,
                        measure_sdss_z0=False,
                        measure_sdss_z01=False):
    """
    Measure AB magnitudes (and optionally magnitude errors) for a batch
    of spectra in requested filter systems.

    Always measures DECam g,r. Optionally measures BASS, SDSS (observed frame),
    SDSS at z=0 (rest frame), and SDSS at z=0.1 (band-shifted, for k-correction
    validation). Each band-shift uses a per-object resample_flux.

    Magnitudes are computed using speclite's get_ab_magnitudes.
    Errors (if ivar_2d provided) are computed via analytic propagation
    using get_filter_weights.

    Parameters
    ----------
    wave_arr : 1D array, shape (Nwave,)
    flux_2d : 2D array, shape (N_spectra, Nwave)
        Flux in 1e-17 erg/s/cm2/Ang.
    ivar_2d : 2D array, shape (N_spectra, Nwave), or None
        Inverse variance in (1e-17 erg/s/cm2/Ang)^{-2} units.
        If provided, magnitude errors are returned. If None, no errors.
    zred : 1D array, shape (N_spectra,), optional
        Redshift of objects. Required when measure_sdss_z0=True or
        measure_sdss_z01=True.
    measure_bass : bool
    measure_sdss : bool
    measure_sdss_z0 : bool
    measure_sdss_z01 : bool

    Returns
    -------
    result : dict
        Keys: 'g_decam', 'r_decam', and optionally 'g_bass', 'r_bass',
        'g_sdss', 'r_sdss', 'g_sdss_z0', 'r_sdss_z0', 'g_sdss_z01',
        'r_sdss_z01'.
        If ivar_2d is provided, also contains '*_err' variants of each key.
        Each value is a 1D array of shape (N_spectra,).
    """
    flux_2d = np.atleast_2d(flux_2d)
    n_spec = flux_2d.shape[0]
    do_errors = ivar_2d is not None

    if do_errors:
        ivar_2d = np.atleast_2d(ivar_2d)

    # Speclite units
    wlen_f = wave_arr * u.Angstrom
    flux_f = flux_2d * 1e-17 * u.erg / (u.cm**2 * u.s * u.Angstrom)

    result = {}

    # --- Helper: magnitudes via speclite, errors via filter weights ---
    def _measure_filters(filters_dict, wave, wlen, flux_with_units,
                         flux_raw, ivar_raw=None):
        """
        filters_dict : dict of {result_key: (FilterSequence, column_name)}
        wave         : 1D array in Angstrom (for get_filter_weights)
        wlen         : wave * u.Angstrom (for speclite)
        flux_with_units : flux with astropy units (for speclite)
        flux_raw     : flux in 1e-17 units, no astropy units (for error calc)
        ivar_raw     : ivar in (1e-17)^{-2} units, or None
        """
        for key, (filt_seq, col_name) in filters_dict.items():
            # Magnitudes via speclite
            result[key] = filt_seq.get_ab_magnitudes(flux_with_units, wlen)[col_name].data

            # Errors via analytic propagation
            if ivar_raw is not None:
                w = get_filter_weights(filt_seq[0], wave)
                flux_phys = flux_raw * 1e-17
                var_phys = np.where(ivar_raw > 0, 1.0 / ivar_raw, 0.0) * (1e-17)**2
                maggies = flux_phys @ w
                var_maggies = var_phys @ (w**2)
                result[key + '_err'] = np.where(
                    maggies > 0,
                    (2.5 / np.log(10)) * np.sqrt(var_maggies) / np.abs(maggies),
                    np.nan
                )

    ivar_for_errors = ivar_2d if do_errors else None

    # --- DECam g, r ---
    _measure_filters({
        'g_decam': (DECAM_G, 'decamDR1noatm-g'),
        'r_decam': (DECAM_R, 'decamDR1noatm-r'),
    }, wave_arr, wlen_f, flux_f, flux_2d, ivar_for_errors)

    # --- BASS g, r ---
    if measure_bass:
        _measure_filters({
            'g_bass': (BASS_G, 'BASS-g'),
            'r_bass': (BASS_R, 'BASS-r'),
        }, wave_arr, wlen_f, flux_f, flux_2d, ivar_for_errors)

    # --- SDSS g, r (observed frame) ---
    if measure_sdss:
        _measure_filters({
            'g_sdss': (SDSS_G, 'sdss2010noatm-g'),
            'r_sdss': (SDSS_R, 'sdss2010noatm-r'),
        }, wave_arr, wlen_f, flux_f, flux_2d, ivar_for_errors)

    def _measure_sdss_at_target_z(target_z, key_g, key_r):
        """
        Per-object band-shift to ``target_z``: transform observed
        ``(wave, flux)`` to ``(wave * (1+target_z)/(1+z_obs),
        flux * (1+z_obs)/(1+target_z))``, resample onto ``wave_arr``, then
        integrate against the SDSS g/r filters. Reduces to the rest-frame
        case when ``target_z=0``.
        """
        if zred is None:
            raise ValueError("zred is required when measuring SDSS at a target z")

        wave_out = wave_arr
        factor_num = (1.0 + target_z)

        flux_shifted_resampled = np.full_like(flux_2d, np.nan)
        ivar_shifted_resampled = np.full_like(flux_2d, 0.0) if do_errors else None

        valid = np.isfinite(zred) & (zred > 0)

        for j in np.where(valid)[0]:
            z_j = zred[j]
            scale = factor_num / (1.0 + z_j)
            wave_target = wave_arr * scale
            flux_target_j = flux_2d[j] / scale

            if do_errors:
                ivar_target_j = ivar_2d[j] * (scale ** 2)
                f_out, iv_out = resample_flux(wave_out, wave_target, flux_target_j,
                                              ivar=ivar_target_j)
                flux_shifted_resampled[j] = f_out
                ivar_shifted_resampled[j] = iv_out
            else:
                f_out = resample_flux(wave_out, wave_target, flux_target_j)
                flux_shifted_resampled[j] = f_out

        g_out = np.full(n_spec, np.nan)
        r_out = np.full(n_spec, np.nan)
        g_err_out = np.full(n_spec, np.nan) if do_errors else None
        r_err_out = np.full(n_spec, np.nan) if do_errors else None

        if np.any(valid):
            sub_flux = flux_shifted_resampled[valid]
            sub_wlen = wave_out * u.Angstrom
            sub_flux_f = sub_flux * 1e-17 * u.erg / (u.cm**2 * u.s * u.Angstrom)

            g_out[valid] = SDSS_G.get_ab_magnitudes(sub_flux_f, sub_wlen)["sdss2010noatm-g"].data
            r_out[valid] = SDSS_R.get_ab_magnitudes(sub_flux_f, sub_wlen)["sdss2010noatm-r"].data

            if do_errors:
                sub_ivar = ivar_shifted_resampled[valid]
                sub_flux_phys = sub_flux * 1e-17
                sub_var_phys = np.where(sub_ivar > 0, 1.0 / sub_ivar, 0.0) * (1e-17)**2

                w_g = get_filter_weights(SDSS_G[0], wave_out)
                w_r = get_filter_weights(SDSS_R[0], wave_out)

                maggies_g = sub_flux_phys @ w_g
                maggies_r = sub_flux_phys @ w_r
                var_maggies_g = sub_var_phys @ (w_g**2)
                var_maggies_r = sub_var_phys @ (w_r**2)

                g_err_out[valid] = np.where(
                    maggies_g > 0,
                    (2.5 / np.log(10)) * np.sqrt(var_maggies_g) / np.abs(maggies_g),
                    np.nan)
                r_err_out[valid] = np.where(
                    maggies_r > 0,
                    (2.5 / np.log(10)) * np.sqrt(var_maggies_r) / np.abs(maggies_r),
                    np.nan)

        result[key_g] = g_out
        result[key_r] = r_out
        if do_errors:
            result[key_g + '_err'] = g_err_out
            result[key_r + '_err'] = r_err_out

    # --- SDSS g, r at z=0 (rest frame) ---
    if measure_sdss_z0:
        _measure_sdss_at_target_z(0.0, 'g_sdss_z0', 'r_sdss_z0')

    # --- SDSS g, r band-shifted to z=0.1 (k-correction validation) ---
    if measure_sdss_z01:
        _measure_sdss_at_target_z(0.1, 'g_sdss_z01', 'r_sdss_z01')

    return result
    

def _process_single_file(args):
    """
    Process one fastspecfit FITS file for model-only photometry.
    Used as a multiprocessing worker by compute_photometry_catalog.

    Parameters
    ----------
    args : tuple
        (upath, cat_indices, targetids_for_file, redshifts_for_file, batch_size,
         compute_kcorr_z01_validation)

    Returns
    -------
    dict or None
        Dict with 'cat_indices' and photometry/absmag arrays, or None on failure.
        When ``compute_kcorr_z01_validation`` is True, the returned dict also
        contains the four ``*_w_emi_no_smooth`` keys (observed-frame and
        z=0.1 SDSS g/r on continuum+emission).
    """
    (upath, cat_indices, targetids_for_file, redshifts_for_file, batch_size,
     compute_kcorr_z01_validation) = args

    try:
        iron_vac = fits.open(upath, memmap=True)
    except Exception:
        print("ERROR: FALL NOT FOUND!!")
        return None

    header = iron_vac["MODELS"].header
    wavelength = (header["CRVAL1"]
                  + (np.arange(header["NAXIS1"]) - header["CRPIX1"]) * header["CDELT1"])
    model_data = iron_vac["MODELS"].data

    fastspec_data = iron_vac["FASTSPEC"].data
    specphot_data = iron_vac["SPECPHOT"].data 

    print("TODO: NEED TO FIX THIS! SPECPHOT IS NOT VALID FOR V2.1?")
    
    tgids_file = fastspec_data["TARGETID"]
    tgid_to_fits_row = {t: i for i, t in enumerate(tgids_file)}

    valid_cat = []
    valid_fits_rows = []
    valid_local_indices = []
    for ci_local, ci in enumerate(cat_indices):
        row = tgid_to_fits_row.get(targetids_for_file[ci_local])
        if row is not None:
            valid_cat.append(ci)
            valid_fits_rows.append(row)
            valid_local_indices.append(ci_local)

    if len(valid_cat) == 0:
        return None

    valid_cat = np.array(valid_cat)
    valid_fits_rows = np.array(valid_fits_rows)
    valid_local_indices = np.array(valid_local_indices)
    valid_redshifts = redshifts_for_file[valid_local_indices]
    n_valid = len(valid_cat)

    # SPECPHOT and FASTSPEC HDUs are aligned row-by-row in the iron VAC,
    # so we use the same valid_fits_rows index into both.
    result = {
        "cat_indices":   valid_cat,
        "halpha_ew":      np.array(fastspec_data["HALPHA_EW"][valid_fits_rows], dtype=float),
        "halpha_ew_ivar": np.array(fastspec_data["HALPHA_EW_IVAR"][valid_fits_rows], dtype=float),
        "kcorr01_sdss_g": np.array(specphot_data["KCORR01_SDSS_G"][valid_fits_rows], dtype=float),
        "kcorr01_sdss_r": np.array(specphot_data["KCORR01_SDSS_R"][valid_fits_rows], dtype=float),
    }

    continuum = model_data[valid_fits_rows, 0, :]
    smooth_continuum = model_data[valid_fits_rows, 1, :]
    emission  = model_data[valid_fits_rows, 2, :]

    # -- DECam photometry for both model variants (existing) --
    g_no_emi = np.full(n_valid, np.nan)
    r_no_emi = np.full(n_valid, np.nan)
    g_w_emi  = np.full(n_valid, np.nan)
    r_w_emi  = np.full(n_valid, np.nan)

    # -- BASS photometry on continuum+emission (for BASS->DECam conversion) --
    g_bass_w_emi = np.full(n_valid, np.nan)
    r_bass_w_emi = np.full(n_valid, np.nan)

    #we will have decam continuum only photmetry, and so will be applying sdss conversion to that!
    
    # -- SDSS photometry on continuum only (for DECam->SDSS conversion) --
    g_sdss_no_emi = np.full(n_valid, np.nan)
    r_sdss_no_emi = np.full(n_valid, np.nan)

    # -- SDSS z=0 photometry on continuum only (for k-correction) --
    g_sdss_z0_no_emi = np.full(n_valid, np.nan)
    r_sdss_z0_no_emi = np.full(n_valid, np.nan)

    # -- ONLY_CONT diagnostic: same set of photometry, but flux variants
    #    drop the smooth_continuum component so we can later test its effect.
    g_no_emi_only_cont = np.full(n_valid, np.nan)
    r_no_emi_only_cont = np.full(n_valid, np.nan)
    g_w_emi_only_cont  = np.full(n_valid, np.nan)
    r_w_emi_only_cont  = np.full(n_valid, np.nan)
    g_bass_w_emi_only_cont = np.full(n_valid, np.nan)
    r_bass_w_emi_only_cont = np.full(n_valid, np.nan)
    g_sdss_no_emi_only_cont = np.full(n_valid, np.nan)
    r_sdss_no_emi_only_cont = np.full(n_valid, np.nan)
    g_sdss_z0_no_emi_only_cont = np.full(n_valid, np.nan)
    r_sdss_z0_no_emi_only_cont = np.full(n_valid, np.nan)

    # -- KCORR_Z01 validation diagnostic: SDSS g/r on continuum+emission
    #    (no smooth_continuum), in observed frame and band-shifted to z=0.1.
    if compute_kcorr_z01_validation:
        g_sdss_w_emi_no_smooth     = np.full(n_valid, np.nan)
        r_sdss_w_emi_no_smooth     = np.full(n_valid, np.nan)
        g_sdss_z01_w_emi_no_smooth = np.full(n_valid, np.nan)
        r_sdss_z01_w_emi_no_smooth = np.full(n_valid, np.nan)

    # --- Continuum + emission: measure DECam and BASS ---
    flux_w_emi = continuum + smooth_continuum + emission
    for start in range(0, n_valid, batch_size):
        end = min(start + batch_size, n_valid)
     
        phot = measure_photo_batch(wavelength, flux_w_emi[start:end],
                                   measure_bass=True)
        g_w_emi[start:end] = phot['g_decam']
        r_w_emi[start:end] = phot['r_decam']
        g_bass_w_emi[start:end] = phot['g_bass']
        r_bass_w_emi[start:end] = phot['r_bass']
 

    # --- Continuum only: measure DECam, SDSS, and SDSS at z=0 ---
    flux_cont_only = continuum + smooth_continuum
    for start in range(0, n_valid, batch_size):
        end = min(start + batch_size, n_valid)
        
        phot = measure_photo_batch(
            wavelength, flux_cont_only[start:end],
            zred=valid_redshifts[start:end],
            measure_sdss=True,
            measure_sdss_z0=True,
        )
        g_no_emi[start:end] = phot['g_decam']
        r_no_emi[start:end] = phot['r_decam']
        g_sdss_no_emi[start:end] = phot['g_sdss']
        r_sdss_no_emi[start:end] = phot['r_sdss']
        g_sdss_z0_no_emi[start:end] = phot['g_sdss_z0']
        r_sdss_z0_no_emi[start:end] = phot['r_sdss_z0']

    # --- ONLY_CONT: continuum + emission (no smooth) -> DECam and BASS ---
    flux_w_emi_only_cont = continuum + emission
    for start in range(0, n_valid, batch_size):
        end = min(start + batch_size, n_valid)

        phot = measure_photo_batch(wavelength, flux_w_emi_only_cont[start:end],
                                   measure_bass=True)
        g_w_emi_only_cont[start:end]      = phot['g_decam']
        r_w_emi_only_cont[start:end]      = phot['r_decam']
        g_bass_w_emi_only_cont[start:end] = phot['g_bass']
        r_bass_w_emi_only_cont[start:end] = phot['r_bass']

    # --- ONLY_CONT: continuum only (no smooth) -> DECam, SDSS, SDSS@z=0 ---
    flux_no_emi_only_cont = continuum
    for start in range(0, n_valid, batch_size):
        end = min(start + batch_size, n_valid)

        phot = measure_photo_batch(
            wavelength, flux_no_emi_only_cont[start:end],
            zred=valid_redshifts[start:end],
            measure_sdss=True,
            measure_sdss_z0=True,
        )
        g_no_emi_only_cont[start:end]       = phot['g_decam']
        r_no_emi_only_cont[start:end]       = phot['r_decam']
        g_sdss_no_emi_only_cont[start:end]  = phot['g_sdss']
        r_sdss_no_emi_only_cont[start:end]  = phot['r_sdss']
        g_sdss_z0_no_emi_only_cont[start:end] = phot['g_sdss_z0']
        r_sdss_z0_no_emi_only_cont[start:end] = phot['r_sdss_z0']

    # --- KCORR_Z01 validation: continuum+emission (no smooth) -> SDSS@z_obs and SDSS@z=0.1 ---
    if compute_kcorr_z01_validation:
        # Reuse flux_w_emi_only_cont (= continuum + emission) from above.
        for start in range(0, n_valid, batch_size):
            end = min(start + batch_size, n_valid)

            phot = measure_photo_batch(
                wavelength, flux_w_emi_only_cont[start:end],
                zred=valid_redshifts[start:end],
                measure_sdss=True,
                measure_sdss_z01=True,
            )
            g_sdss_w_emi_no_smooth[start:end]     = phot['g_sdss']
            r_sdss_w_emi_no_smooth[start:end]     = phot['r_sdss']
            g_sdss_z01_w_emi_no_smooth[start:end] = phot['g_sdss_z01']
            r_sdss_z01_w_emi_no_smooth[start:end] = phot['r_sdss_z01']

    result["g_model_no_emi"] = g_no_emi
    result["r_model_no_emi"] = r_no_emi
    result["g_model_w_emi"]  = g_w_emi
    result["r_model_w_emi"]  = r_w_emi
    result["g_bass_w_emi"]   = g_bass_w_emi
    result["r_bass_w_emi"]   = r_bass_w_emi
    result["g_sdss_no_emi"]  = g_sdss_no_emi
    result["r_sdss_no_emi"]  = r_sdss_no_emi
    result["g_sdss_z0_no_emi"] = g_sdss_z0_no_emi
    result["r_sdss_z0_no_emi"] = r_sdss_z0_no_emi

    result["g_model_no_emi_ONLY_CONT"] = g_no_emi_only_cont
    result["r_model_no_emi_ONLY_CONT"] = r_no_emi_only_cont
    result["g_model_w_emi_ONLY_CONT"]  = g_w_emi_only_cont
    result["r_model_w_emi_ONLY_CONT"]  = r_w_emi_only_cont
    result["g_bass_w_emi_ONLY_CONT"]   = g_bass_w_emi_only_cont
    result["r_bass_w_emi_ONLY_CONT"]   = r_bass_w_emi_only_cont
    result["g_sdss_no_emi_ONLY_CONT"]  = g_sdss_no_emi_only_cont
    result["r_sdss_no_emi_ONLY_CONT"]  = r_sdss_no_emi_only_cont
    result["g_sdss_z0_no_emi_ONLY_CONT"] = g_sdss_z0_no_emi_only_cont
    result["r_sdss_z0_no_emi_ONLY_CONT"] = r_sdss_z0_no_emi_only_cont

    if compute_kcorr_z01_validation:
        result["g_sdss_w_emi_no_smooth"]     = g_sdss_w_emi_no_smooth
        result["r_sdss_w_emi_no_smooth"]     = r_sdss_w_emi_no_smooth
        result["g_sdss_z01_w_emi_no_smooth"] = g_sdss_z01_w_emi_no_smooth
        result["r_sdss_z01_w_emi_no_smooth"] = r_sdss_z01_w_emi_no_smooth

    iron_vac.close()

    return result


def compute_photometry_catalog(catalog,
                               spectra_h5_path=None,
                               compute_data_photometry=True,
                               base_dir="/global/cfs/cdirs/desi/public/dr1/vac/dr1/fastspecfit/iron/v2.1/healpix",
                               save_path=None,
                               batch_size=500,
                               ncore=8,
                               compute_kcorr_z01_validation=False,
                               verbose=True):
    """
    Loop over all objects in a DESI catalog, extract fastspecfit models,
    extract absolute magnitudes from SPECPHOT, and measure DECam g/r
    photometry for model (and optionally data) flux variants.

    Parameters
    ----------
    catalog : astropy Table or structured array
        Must contain columns: TARGETID, HEALPIX, SURVEY, PROGRAM.
    spectra_h5_path : str or None
        Path to the HDF5 file with WAVE, FLUX, TARGETID datasets.
        Required if compute_data_photometry=True.
    compute_data_photometry : bool
        If True, also compute photometry on the observed data (with/without
        emission). Requires spectra_h5_path. If False, only model photometry
        and absolute magnitude columns are returned.
    base_dir : str
        Root directory of the fastspecfit healpix VAC.
    save_path : str or None
        If provided, write the output table to this path (FITS format).
    batch_size : int
        Number of spectra per batch for speclite.
    ncore : int
        Number of parallel workers. Only used when compute_data_photometry=False.
        Falls back to serial if compute_data_photometry=True.
    compute_kcorr_z01_validation : bool
        If True, also compute SDSS g/r on (continuum + emission) -- with
        smooth_continuum dropped -- in observed frame and band-shifted to
        z=0.1, and emit four extra columns (g/r_sdss_w_emi_no_smooth and
        g/r_sdss_z01_w_emi_no_smooth). Off by default.
    verbose : bool
        Print progress updates.

    Returns
    -------
    result : astropy Table
    """

    use_parallel = (ncore > 1) and (not compute_data_photometry)
    if ncore > 1 and compute_data_photometry:
        if verbose:
            print("WARNING: Parallel mode not supported with compute_data_photometry=True. "
                  "Falling back to serial.")

    n_objects = len(catalog)
    targetids = np.array(catalog["TARGETID"])
    redshifts = np.array(catalog["Z"], dtype=float)

    # ---- Load observed spectra if needed ----
    h5_tgid_to_row = {}
    h5_flux = None
    h5_wave = None

    if compute_data_photometry:
        if spectra_h5_path is None:
            raise ValueError("spectra_h5_path is required when compute_data_photometry=True")
        if verbose:
            print(f"Loading observed spectra from {spectra_h5_path} ...")
        with h5py.File(spectra_h5_path, "r") as f:
            h5_wave = f["WAVE"][:]
            h5_flux = f["FLUX"][:]
            h5_targetid = f["TARGETID"][:]
        h5_tgid_to_row = {t: i for i, t in enumerate(h5_targetid)}
        if verbose:
            print(f"  Loaded {len(h5_targetid)} spectra, wave shape {h5_wave.shape}")

    # ---- Output arrays ----
    g_model_no_emi = np.full(n_objects, np.nan)
    r_model_no_emi = np.full(n_objects, np.nan)
    g_model_w_emi  = np.full(n_objects, np.nan)
    r_model_w_emi  = np.full(n_objects, np.nan)

    # BASS on continuum+emission (for BASS->DECam conversion)
    g_bass_w_emi = np.full(n_objects, np.nan)
    r_bass_w_emi = np.full(n_objects, np.nan)

    # SDSS on continuum only (for DECam->SDSS conversion)
    g_sdss_no_emi = np.full(n_objects, np.nan)
    r_sdss_no_emi = np.full(n_objects, np.nan)

    # SDSS at z=0 on continuum only (for k-correction)
    g_sdss_z0_no_emi = np.full(n_objects, np.nan)
    r_sdss_z0_no_emi = np.full(n_objects, np.nan)

    # ONLY_CONT diagnostic: same five photometry pairs but the model flux
    # variants drop the smooth_continuum component.
    g_model_no_emi_ONLY_CONT = np.full(n_objects, np.nan)
    r_model_no_emi_ONLY_CONT = np.full(n_objects, np.nan)
    g_model_w_emi_ONLY_CONT  = np.full(n_objects, np.nan)
    r_model_w_emi_ONLY_CONT  = np.full(n_objects, np.nan)
    g_bass_w_emi_ONLY_CONT   = np.full(n_objects, np.nan)
    r_bass_w_emi_ONLY_CONT   = np.full(n_objects, np.nan)
    g_sdss_no_emi_ONLY_CONT  = np.full(n_objects, np.nan)
    r_sdss_no_emi_ONLY_CONT  = np.full(n_objects, np.nan)
    g_sdss_z0_no_emi_ONLY_CONT = np.full(n_objects, np.nan)
    r_sdss_z0_no_emi_ONLY_CONT = np.full(n_objects, np.nan)

    # KCORR_Z01 validation: only allocated when requested.
    if compute_kcorr_z01_validation:
        g_sdss_w_emi_no_smooth     = np.full(n_objects, np.nan)
        r_sdss_w_emi_no_smooth     = np.full(n_objects, np.nan)
        g_sdss_z01_w_emi_no_smooth = np.full(n_objects, np.nan)
        r_sdss_z01_w_emi_no_smooth = np.full(n_objects, np.nan)

    halpha_ew      = np.full(n_objects, np.nan)
    halpha_ew_ivar = np.full(n_objects, np.nan)

    # FastSpecFit SPECPHOT-derived k-corrections (band-shifted to z=0.1).
    kcorr01_sdss_g = np.full(n_objects, np.nan)
    kcorr01_sdss_r = np.full(n_objects, np.nan)

    if compute_data_photometry:
        g_data_no_emi = np.full(n_objects, np.nan)
        r_data_no_emi = np.full(n_objects, np.nan)
        g_data_w_emi  = np.full(n_objects, np.nan)
        r_data_w_emi  = np.full(n_objects, np.nan)

    # ---- Group by unique FITS file ----
    paths = np.array([
        get_fastspecfit_path(catalog["SURVEY"][i], catalog["PROGRAM"][i],
                            catalog["HEALPIX"][i], base_dir)
        for i in range(n_objects)
    ])
    unique_paths = np.unique(paths)
    n_files = len(unique_paths)

    if verbose:
        print(f"Total objects: {n_objects}, unique FITS files: {n_files}")

    # ==================================================================
    # PARALLEL PATH: model-only photometry across multiple cores
    # ==================================================================
    if use_parallel:
        work_items = []
        for upath in unique_paths:
            ci = np.where(paths == upath)[0]
            work_items.append((upath, ci, targetids[ci], redshifts[ci], batch_size,
                               compute_kcorr_z01_validation))

        if verbose:
            print(f"Processing {n_files} files with {ncore} cores ...")

        files_done = 0
        with mp.Pool(ncore) as pool:
            for file_result in pool.imap_unordered(_process_single_file, work_items):
                files_done += 1
                if file_result is None:
                    continue
                idx = file_result["cat_indices"]
                g_model_no_emi[idx] = file_result["g_model_no_emi"]
                r_model_no_emi[idx] = file_result["r_model_no_emi"]
                g_model_w_emi[idx]  = file_result["g_model_w_emi"]
                r_model_w_emi[idx]  = file_result["r_model_w_emi"]
                g_bass_w_emi[idx]   = file_result["g_bass_w_emi"]
                r_bass_w_emi[idx]   = file_result["r_bass_w_emi"]
                g_sdss_no_emi[idx]  = file_result["g_sdss_no_emi"]
                r_sdss_no_emi[idx]  = file_result["r_sdss_no_emi"]
                g_sdss_z0_no_emi[idx] = file_result["g_sdss_z0_no_emi"]
                r_sdss_z0_no_emi[idx] = file_result["r_sdss_z0_no_emi"]

                g_model_no_emi_ONLY_CONT[idx] = file_result["g_model_no_emi_ONLY_CONT"]
                r_model_no_emi_ONLY_CONT[idx] = file_result["r_model_no_emi_ONLY_CONT"]
                g_model_w_emi_ONLY_CONT[idx]  = file_result["g_model_w_emi_ONLY_CONT"]
                r_model_w_emi_ONLY_CONT[idx]  = file_result["r_model_w_emi_ONLY_CONT"]
                g_bass_w_emi_ONLY_CONT[idx]   = file_result["g_bass_w_emi_ONLY_CONT"]
                r_bass_w_emi_ONLY_CONT[idx]   = file_result["r_bass_w_emi_ONLY_CONT"]
                g_sdss_no_emi_ONLY_CONT[idx]  = file_result["g_sdss_no_emi_ONLY_CONT"]
                r_sdss_no_emi_ONLY_CONT[idx]  = file_result["r_sdss_no_emi_ONLY_CONT"]
                g_sdss_z0_no_emi_ONLY_CONT[idx] = file_result["g_sdss_z0_no_emi_ONLY_CONT"]
                r_sdss_z0_no_emi_ONLY_CONT[idx] = file_result["r_sdss_z0_no_emi_ONLY_CONT"]

                if compute_kcorr_z01_validation:
                    g_sdss_w_emi_no_smooth[idx]     = file_result["g_sdss_w_emi_no_smooth"]
                    r_sdss_w_emi_no_smooth[idx]     = file_result["r_sdss_w_emi_no_smooth"]
                    g_sdss_z01_w_emi_no_smooth[idx] = file_result["g_sdss_z01_w_emi_no_smooth"]
                    r_sdss_z01_w_emi_no_smooth[idx] = file_result["r_sdss_z01_w_emi_no_smooth"]

                halpha_ew[idx]      = file_result["halpha_ew"]
                halpha_ew_ivar[idx] = file_result["halpha_ew_ivar"]
                kcorr01_sdss_g[idx] = file_result["kcorr01_sdss_g"]
                kcorr01_sdss_r[idx] = file_result["kcorr01_sdss_r"]

                if verbose and files_done % 50 == 0:
                    print(f"  Processed {files_done}/{n_files} files")

        if verbose:
            print(f"  Processed {files_done}/{n_files} files (done)")

    # ==================================================================
    # SERIAL PATH: original loop (supports data photometry)
    # ==================================================================
    else:
        for file_idx, upath in enumerate(unique_paths):
            cat_indices = np.where(paths == upath)[0]

            try:
                iron_vac = fits.open(upath, memmap=True)
            except Exception as e:
                if verbose:
                    print(f"  SKIP file {upath}: {e}")
                continue

            try:
                header = iron_vac["MODELS"].header
                wavelength = (header["CRVAL1"]
                              + (np.arange(header["NAXIS1"]) - header["CRPIX1"]) * header["CDELT1"])
                model_data = iron_vac["MODELS"].data
                
                fastspec_data = iron_vac["FASTSPEC"].data
                specphot_data = iron_vac["SPECPHOT"].data
                tgids_file = fastspec_data["TARGETID"]
                tgid_to_fits_row = {t: i for i, t in enumerate(tgids_file)}

                valid_cat = []
                valid_fits_rows = []
                for ci in cat_indices:
                    row = tgid_to_fits_row.get(targetids[ci])
                    if row is not None:
                        valid_cat.append(ci)
                        valid_fits_rows.append(row)
                    elif verbose:
                        print(f"  WARNING: TARGETID {targetids[ci]} not in {upath}")

                if len(valid_cat) == 0:
                    continue

                valid_cat = np.array(valid_cat)
                valid_fits_rows = np.array(valid_fits_rows)
                
                halpha_ew[valid_cat]      = fastspec_data["HALPHA_EW"][valid_fits_rows]
                halpha_ew_ivar[valid_cat] = fastspec_data["HALPHA_EW_IVAR"][valid_fits_rows]
                # SPECPHOT and FASTSPEC are aligned row-by-row in the iron VAC.
                kcorr01_sdss_g[valid_cat] = specphot_data["KCORR01_SDSS_G"][valid_fits_rows]
                kcorr01_sdss_r[valid_cat] = specphot_data["KCORR01_SDSS_R"][valid_fits_rows]

                continuum = model_data[valid_fits_rows, 0, :]
                smooth_continuum = model_data[valid_fits_rows, 1, :]
                emission  = model_data[valid_fits_rows, 2, :]
                valid_zred = redshifts[valid_cat]

                # --- Continuum + emission: DECam and BASS ---
                flux_w_emi = continuum + smooth_continuum + emission
                for start in range(0, len(valid_cat), batch_size):
                    end = min(start + batch_size, len(valid_cat))
                    try:
                        phot = measure_photo_batch(wavelength, flux_w_emi[start:end],
                                                   measure_bass=True)
                        g_model_w_emi[valid_cat[start:end]]  = phot['g_decam']
                        r_model_w_emi[valid_cat[start:end]]  = phot['r_decam']
                        g_bass_w_emi[valid_cat[start:end]]   = phot['g_bass']
                        r_bass_w_emi[valid_cat[start:end]]   = phot['r_bass']
                    except Exception as e:
                        if verbose:
                            print(f"  Photometry error (w_emi, batch {start}-{end}): {e}")

                # --- Continuum only: DECam, SDSS, and SDSS at z=0 ---
                flux_only_cont = continuum + smooth_continuum
                for start in range(0, len(valid_cat), batch_size):
                    end = min(start + batch_size, len(valid_cat))
                    try:
                        phot = measure_photo_batch(
                            wavelength, flux_only_cont[start:end],
                            zred=valid_zred[start:end],
                            measure_sdss=True,
                            measure_sdss_z0=True,
                        )
                        g_model_no_emi[valid_cat[start:end]]   = phot['g_decam']
                        r_model_no_emi[valid_cat[start:end]]   = phot['r_decam']
                        g_sdss_no_emi[valid_cat[start:end]]    = phot['g_sdss']
                        r_sdss_no_emi[valid_cat[start:end]]    = phot['r_sdss']
                        g_sdss_z0_no_emi[valid_cat[start:end]] = phot['g_sdss_z0']
                        r_sdss_z0_no_emi[valid_cat[start:end]] = phot['r_sdss_z0']
                    except Exception as e:
                        if verbose:
                            print(f"  Photometry error (no_emi, batch {start}-{end}): {e}")

                # --- ONLY_CONT: continuum + emission (no smooth) -> DECam and BASS ---
                flux_w_emi_only_cont = continuum + emission
                for start in range(0, len(valid_cat), batch_size):
                    end = min(start + batch_size, len(valid_cat))
                    try:
                        phot = measure_photo_batch(wavelength, flux_w_emi_only_cont[start:end],
                                                   measure_bass=True)
                        g_model_w_emi_ONLY_CONT[valid_cat[start:end]] = phot['g_decam']
                        r_model_w_emi_ONLY_CONT[valid_cat[start:end]] = phot['r_decam']
                        g_bass_w_emi_ONLY_CONT[valid_cat[start:end]]  = phot['g_bass']
                        r_bass_w_emi_ONLY_CONT[valid_cat[start:end]]  = phot['r_bass']
                    except Exception as e:
                        if verbose:
                            print(f"  Photometry error (w_emi ONLY_CONT, batch {start}-{end}): {e}")

                # --- ONLY_CONT: continuum only (no smooth) -> DECam, SDSS, SDSS@z=0 ---
                flux_no_emi_only_cont = continuum
                for start in range(0, len(valid_cat), batch_size):
                    end = min(start + batch_size, len(valid_cat))
                    try:
                        phot = measure_photo_batch(
                            wavelength, flux_no_emi_only_cont[start:end],
                            zred=valid_zred[start:end],
                            measure_sdss=True,
                            measure_sdss_z0=True,
                        )
                        g_model_no_emi_ONLY_CONT[valid_cat[start:end]]   = phot['g_decam']
                        r_model_no_emi_ONLY_CONT[valid_cat[start:end]]   = phot['r_decam']
                        g_sdss_no_emi_ONLY_CONT[valid_cat[start:end]]    = phot['g_sdss']
                        r_sdss_no_emi_ONLY_CONT[valid_cat[start:end]]    = phot['r_sdss']
                        g_sdss_z0_no_emi_ONLY_CONT[valid_cat[start:end]] = phot['g_sdss_z0']
                        r_sdss_z0_no_emi_ONLY_CONT[valid_cat[start:end]] = phot['r_sdss_z0']
                    except Exception as e:
                        if verbose:
                            print(f"  Photometry error (no_emi ONLY_CONT, batch {start}-{end}): {e}")

                # --- KCORR_Z01 validation: continuum+emission (no smooth)
                #     -> SDSS@z_obs and SDSS@z=0.1 ---
                if compute_kcorr_z01_validation:
                    for start in range(0, len(valid_cat), batch_size):
                        end = min(start + batch_size, len(valid_cat))
                        try:
                            phot = measure_photo_batch(
                                wavelength, flux_w_emi_only_cont[start:end],
                                zred=valid_zred[start:end],
                                measure_sdss=True,
                                measure_sdss_z01=True,
                            )
                            g_sdss_w_emi_no_smooth[valid_cat[start:end]]     = phot['g_sdss']
                            r_sdss_w_emi_no_smooth[valid_cat[start:end]]     = phot['r_sdss']
                            g_sdss_z01_w_emi_no_smooth[valid_cat[start:end]] = phot['g_sdss_z01']
                            r_sdss_z01_w_emi_no_smooth[valid_cat[start:end]] = phot['r_sdss_z01']
                        except Exception as e:
                            if verbose:
                                print(f"  Photometry error (kcorr_z01 validation, batch {start}-{end}): {e}")

                if compute_data_photometry:
                    h5_rows = np.array([h5_tgid_to_row.get(targetids[ci], -1) for ci in valid_cat])
                    has_h5 = h5_rows >= 0

                    if np.any(has_h5):
                        sub_cat = valid_cat[has_h5]
                        sub_h5 = h5_rows[has_h5]
                        sub_emission = emission[has_h5]

                        obs_flux = h5_flux[sub_h5]

                        data_variants = {
                            "data_no_emi": (obs_flux - sub_emission, g_data_no_emi, r_data_no_emi),
                            "data_w_emi":  (obs_flux,                g_data_w_emi,  r_data_w_emi),
                        }
                        for vname, (flux_2d_dv, g_out, r_out) in data_variants.items():
                            for start in range(0, len(sub_cat), batch_size):
                                end = min(start + batch_size, len(sub_cat))
                                try:
                                    phot = measure_photo_batch(h5_wave, flux_2d_dv[start:end])
                                    g_out[sub_cat[start:end]] = phot['g_decam']
                                    r_out[sub_cat[start:end]] = phot['r_decam']
                                except Exception as e:
                                    if verbose:
                                        print(f"  Photometry error ({vname}, batch {start}-{end}): {e}")

            finally:
                iron_vac.close()

            if verbose and (file_idx + 1) % 50 == 0:
                print(f"  Processed {file_idx + 1}/{n_files} files")

    # ---- Build output table ----
    columns = {
        "TARGETID":                    targetids,
        "g_model_no_emi":             g_model_no_emi,
        "r_model_no_emi":             r_model_no_emi,
        "g_model_w_emi":              g_model_w_emi,
        "r_model_w_emi":              r_model_w_emi,
        "g_bass_w_emi":               g_bass_w_emi,
        "r_bass_w_emi":               r_bass_w_emi,
        "g_sdss_no_emi":              g_sdss_no_emi,
        "r_sdss_no_emi":              r_sdss_no_emi,
        "g_sdss_z0_no_emi":           g_sdss_z0_no_emi,
        "r_sdss_z0_no_emi":           r_sdss_z0_no_emi,
        "g_model_no_emi_ONLY_CONT":   g_model_no_emi_ONLY_CONT,
        "r_model_no_emi_ONLY_CONT":   r_model_no_emi_ONLY_CONT,
        "g_model_w_emi_ONLY_CONT":    g_model_w_emi_ONLY_CONT,
        "r_model_w_emi_ONLY_CONT":    r_model_w_emi_ONLY_CONT,
        "g_bass_w_emi_ONLY_CONT":     g_bass_w_emi_ONLY_CONT,
        "r_bass_w_emi_ONLY_CONT":     r_bass_w_emi_ONLY_CONT,
        "g_sdss_no_emi_ONLY_CONT":    g_sdss_no_emi_ONLY_CONT,
        "r_sdss_no_emi_ONLY_CONT":    r_sdss_no_emi_ONLY_CONT,
        "g_sdss_z0_no_emi_ONLY_CONT": g_sdss_z0_no_emi_ONLY_CONT,
        "r_sdss_z0_no_emi_ONLY_CONT": r_sdss_z0_no_emi_ONLY_CONT,
        "HALPHA_EW":                   halpha_ew,
        "HALPHA_EW_IVAR":             halpha_ew_ivar,
        "KCORR01_SDSS_G":             kcorr01_sdss_g,
        "KCORR01_SDSS_R":             kcorr01_sdss_r,
    }
    
    if compute_data_photometry:
        columns["g_data_no_emi"] = g_data_no_emi
        columns["r_data_no_emi"] = r_data_no_emi
        columns["g_data_w_emi"]  = g_data_w_emi
        columns["r_data_w_emi"]  = r_data_w_emi

    if compute_kcorr_z01_validation:
        columns["g_sdss_w_emi_no_smooth"]     = g_sdss_w_emi_no_smooth
        columns["r_sdss_w_emi_no_smooth"]     = r_sdss_w_emi_no_smooth
        columns["g_sdss_z01_w_emi_no_smooth"] = g_sdss_z01_w_emi_no_smooth
        columns["r_sdss_z01_w_emi_no_smooth"] = r_sdss_z01_w_emi_no_smooth

    result = Table(columns)

    if save_path is not None:
        result.write(save_path, overwrite=True)
        if verbose:
            print(f"Saved to {save_path}")

    if verbose:
        n_good_model = np.sum(np.isfinite(g_model_w_emi))
        msg = (f"Done. {n_good_model}/{n_objects} with valid model photometry, ")
        if compute_data_photometry:
            n_good_data = np.sum(np.isfinite(g_data_w_emi))
            msg += f" {n_good_data}/{n_objects} with valid data photometry."
        print(msg)

    return result


# ======================================================================
# Full photometric correction pipeline
# ======================================================================

def _run_correction_chain(
    mag_g_in, mag_r_in, north_mask,
    g_model_no_emi, r_model_no_emi,
    g_model_w_emi,  r_model_w_emi,
    g_bass_w_emi,   r_bass_w_emi,
    g_sdss_no_emi,  r_sdss_no_emi,
    g_sdss_z0_no_emi, r_sdss_z0_no_emi,
    label="",
):
    """Run the four-step photometric correction chain on one set of model magnitudes.

    Used by :func:`apply_photometric_corrections` to produce both the default
    (continuum + smooth_continuum) and the ONLY_CONT (continuum-only) chains
    from the same arithmetic, so the two cannot drift.

    Parameters
    ----------
    mag_g_in, mag_r_in : 1D arrays
        Tractor input magnitudes (DECam where ``is_south==1``, BASS otherwise).
        These are *copied* before being modified.
    north_mask : 1D bool array
        True where Step 1 (BASS -> DECam) must be applied.
    g_model_*, r_model_*, g_bass_*, r_bass_*, g_sdss_*, r_sdss_*, g_sdss_z0_*, r_sdss_z0_* : 1D arrays
        Model photometry (one of the two flavors).
    label : str
        Tag prepended to the diagnostic prints (e.g. ``""`` or ``"[ONLY_CONT] "``).

    Returns
    -------
    dict with keys: delta_bass2decam_g/r, delta_neb_g/r, delta_decam2sdss_g/r,
    delta_kcorr_g/r, mag_g_sdss_z0, mag_r_sdss_z0.
    """
    n = len(mag_g_in)

    delta_bass2decam_g = g_model_w_emi - g_bass_w_emi
    delta_bass2decam_r = r_model_w_emi - r_bass_w_emi

    mag_g_working = np.array(mag_g_in, dtype=float, copy=True)
    mag_r_working = np.array(mag_r_in, dtype=float, copy=True)

    mag_g_working[north_mask] += delta_bass2decam_g[north_mask]
    mag_r_working[north_mask] += delta_bass2decam_r[north_mask]

    delta_neb_g_raw = g_model_no_emi - g_model_w_emi
    delta_neb_r_raw = r_model_no_emi - r_model_w_emi

    delta_neb_g = np.where(np.isfinite(delta_neb_g_raw),
                           np.maximum(delta_neb_g_raw, 0.0), 0.0)
    delta_neb_r = np.where(np.isfinite(delta_neb_r_raw),
                           np.maximum(delta_neb_r_raw, 0.0), 0.0)

    n_neb_applied = int(np.sum(np.isfinite(delta_neb_g_raw)))
    n_neb_nan = n - n_neb_applied
    print(f"  {label}Nebular correction: {n_neb_applied} objects corrected (direct model delta), "
          f"{n_neb_nan} with NaN model photometry (delta=0)")
    print(f"  {label}Median delta_mag_g = {np.nanmedian(delta_neb_g):.4f}, "
          f"Median delta_mag_r = {np.nanmedian(delta_neb_r):.4f}")

    mag_g_working += delta_neb_g
    mag_r_working += delta_neb_r

    delta_decam2sdss_g = g_sdss_no_emi - g_model_no_emi
    delta_decam2sdss_r = r_sdss_no_emi - r_model_no_emi

    mag_g_working += delta_decam2sdss_g
    mag_r_working += delta_decam2sdss_r

    delta_kcorr_g = g_sdss_z0_no_emi - g_sdss_no_emi
    delta_kcorr_r = r_sdss_z0_no_emi - r_sdss_no_emi

    mag_g_working += delta_kcorr_g
    mag_r_working += delta_kcorr_r

    for band, d_b2d, d_neb, d_d2s, d_kc in [
        ("g", delta_bass2decam_g, delta_neb_g, delta_decam2sdss_g, delta_kcorr_g),
        ("r", delta_bass2decam_r, delta_neb_r, delta_decam2sdss_r, delta_kcorr_r),
    ]:
        print(f"  {label}{band}-band median deltas: "
              f"bass2decam={np.nanmedian(d_b2d):.4f}, "
              f"neb={np.nanmedian(d_neb):.4f}, "
              f"decam2sdss={np.nanmedian(d_d2s):.4f}, "
              f"kcorr={np.nanmedian(d_kc):.4f}")

    return {
        "delta_bass2decam_g": delta_bass2decam_g,
        "delta_bass2decam_r": delta_bass2decam_r,
        "delta_neb_g": delta_neb_g,
        "delta_neb_r": delta_neb_r,
        "delta_decam2sdss_g": delta_decam2sdss_g,
        "delta_decam2sdss_r": delta_decam2sdss_r,
        "delta_kcorr_g": delta_kcorr_g,
        "delta_kcorr_r": delta_kcorr_r,
        "mag_g_sdss_z0": mag_g_working,
        "mag_r_sdss_z0": mag_r_working,
    }


def apply_photometric_corrections(cat, model_phot_table):
    """Apply the full photometric correction chain for both g and r bands.

    Converts tractor apparent magnitudes (DECam or BASS) to SDSS z=0
    continuum-only apparent magnitudes suitable for stellar mass estimation.

    Correction steps (applied in this order):
        1. BASS -> DECam  (only where is_south=0; measured on w_emi model)
        2. Nebular emission removal in DECam (always applied from model
           template difference: continuum+smooth vs continuum+smooth+emission)
        3. DECam -> SDSS  (measured on continuum-only model)
        4. k-correction: SDSS z_obs -> SDSS z=0 (measured on continuum-only model)

    The same chain is run a second time on the ONLY_CONT model variants (where
    smooth_continuum is dropped from the flux) and returned alongside under
    keys with the ``_ONLY_CONT`` suffix. Error columns are not duplicated:
    in the current pipeline none of the deltas carry uncertainty (delta_neb_*_err
    is hard-coded to zero, and the other deltas add no error contribution), so
    ``mag_g_sdss_z0_err`` is identical between the two chains by construction.

    Parameters
    ----------
    cat : astropy Table
        Galaxy catalog with columns MAG_G, MAG_R, MAG_G_ERR, MAG_R_ERR,
        is_south (1=DECam, 0=BASS).
    model_phot_table : astropy Table
        Output of compute_photometry_catalog, with columns for DECam/BASS/SDSS
        model photometry (default and ``*_ONLY_CONT`` variants), HALPHA_EW /
        HALPHA_EW_IVAR, and the SPECPHOT-derived KCORR01_SDSS_G / KCORR01_SDSS_R.

    Returns
    -------
    corrections : dict
        Keys include all intermediate deltas and final corrected magnitudes
        for both chains:
        - delta_bass2decam_g/r, delta_neb_g/r, delta_neb_g/r_err,
          delta_decam2sdss_g/r, delta_kcorr_g/r
        - mag_g_sdss_z0, mag_r_sdss_z0  (final corrected apparent mags)
        - mag_g_sdss_z0_err, mag_r_sdss_z0_err
        - <same set with ``_ONLY_CONT`` suffix, except no _err keys>
        - halpha_ew, halpha_ew_ivar
        - kcorr01_sdss_g, kcorr01_sdss_r  (passthrough from FASTSPEC SPECPHOT)
    """
    n = len(cat)
    is_south = np.asarray(cat["is_south"].data, dtype=int)
    north_mask = (is_south == 0)

    halpha_ew = np.asarray(model_phot_table["HALPHA_EW"].data, dtype=float)
    halpha_ew_ivar = np.asarray(model_phot_table["HALPHA_EW_IVAR"].data, dtype=float)
    kcorr01_sdss_g = np.asarray(model_phot_table["KCORR01_SDSS_G"].data, dtype=float)
    kcorr01_sdss_r = np.asarray(model_phot_table["KCORR01_SDSS_R"].data, dtype=float)

    mag_g_in = np.array(cat["MAG_G"].data, dtype=float)
    mag_r_in = np.array(cat["MAG_R"].data, dtype=float)

    n_north = int(np.sum(north_mask))
    n_south = int(np.sum(~north_mask))
    print(f"  Photometric corrections applied: {n_south} south (DECam), {n_north} north (BASS->DECam)")

    # ------------------------------------------------------------------
    # Default chain: continuum + smooth_continuum (+ emission for w_emi)
    # ------------------------------------------------------------------
    chain_default = _run_correction_chain(
        mag_g_in, mag_r_in, north_mask,
        g_model_no_emi=model_phot_table["g_model_no_emi"].data,
        r_model_no_emi=model_phot_table["r_model_no_emi"].data,
        g_model_w_emi=model_phot_table["g_model_w_emi"].data,
        r_model_w_emi=model_phot_table["r_model_w_emi"].data,
        g_bass_w_emi=model_phot_table["g_bass_w_emi"].data,
        r_bass_w_emi=model_phot_table["r_bass_w_emi"].data,
        g_sdss_no_emi=model_phot_table["g_sdss_no_emi"].data,
        r_sdss_no_emi=model_phot_table["r_sdss_no_emi"].data,
        g_sdss_z0_no_emi=model_phot_table["g_sdss_z0_no_emi"].data,
        r_sdss_z0_no_emi=model_phot_table["r_sdss_z0_no_emi"].data,
        label="",
    )

    # ------------------------------------------------------------------
    # ONLY_CONT chain: continuum (+ emission for w_emi), no smooth_continuum
    # Same arithmetic, only the model photometry inputs differ.
    # ------------------------------------------------------------------
    chain_only_cont = _run_correction_chain(
        mag_g_in, mag_r_in, north_mask,
        g_model_no_emi=model_phot_table["g_model_no_emi_ONLY_CONT"].data,
        r_model_no_emi=model_phot_table["r_model_no_emi_ONLY_CONT"].data,
        g_model_w_emi=model_phot_table["g_model_w_emi_ONLY_CONT"].data,
        r_model_w_emi=model_phot_table["r_model_w_emi_ONLY_CONT"].data,
        g_bass_w_emi=model_phot_table["g_bass_w_emi_ONLY_CONT"].data,
        r_bass_w_emi=model_phot_table["r_bass_w_emi_ONLY_CONT"].data,
        g_sdss_no_emi=model_phot_table["g_sdss_no_emi_ONLY_CONT"].data,
        r_sdss_no_emi=model_phot_table["r_sdss_no_emi_ONLY_CONT"].data,
        g_sdss_z0_no_emi=model_phot_table["g_sdss_z0_no_emi_ONLY_CONT"].data,
        r_sdss_z0_no_emi=model_phot_table["r_sdss_z0_no_emi_ONLY_CONT"].data,
        label="[ONLY_CONT] ",
    )

    # ------------------------------------------------------------------
    # Error propagation (default chain only; ONLY_CONT errors are identical
    # because all deltas have zero error contribution in the current pipeline).
    # ------------------------------------------------------------------
    delta_neb_g_err = np.zeros(n)
    delta_neb_r_err = np.zeros(n)

    mag_g_err_base = np.zeros(n)
    mag_r_err_base = np.zeros(n)
    if "MAG_G_ERR" in cat.colnames:
        mag_g_err_base = np.array(cat["MAG_G_ERR"].data, dtype=float)
    if "MAG_R_ERR" in cat.colnames:
        mag_r_err_base = np.array(cat["MAG_R_ERR"].data, dtype=float)

    mag_g_sdss_z0_err = np.sqrt(mag_g_err_base**2 + delta_neb_g_err**2)
    mag_r_sdss_z0_err = np.sqrt(mag_r_err_base**2 + delta_neb_r_err**2)

    out = {
        "delta_bass2decam_g": chain_default["delta_bass2decam_g"],
        "delta_bass2decam_r": chain_default["delta_bass2decam_r"],
        "delta_neb_g": chain_default["delta_neb_g"],
        "delta_neb_r": chain_default["delta_neb_r"],
        "delta_neb_g_err": delta_neb_g_err,
        "delta_neb_r_err": delta_neb_r_err,
        "delta_decam2sdss_g": chain_default["delta_decam2sdss_g"],
        "delta_decam2sdss_r": chain_default["delta_decam2sdss_r"],
        "delta_kcorr_g": chain_default["delta_kcorr_g"],
        "delta_kcorr_r": chain_default["delta_kcorr_r"],
        "mag_g_sdss_z0": chain_default["mag_g_sdss_z0"],
        "mag_r_sdss_z0": chain_default["mag_r_sdss_z0"],
        "mag_g_sdss_z0_err": mag_g_sdss_z0_err,
        "mag_r_sdss_z0_err": mag_r_sdss_z0_err,
        "halpha_ew": halpha_ew,
        "halpha_ew_ivar": halpha_ew_ivar,
        "kcorr01_sdss_g": kcorr01_sdss_g,
        "kcorr01_sdss_r": kcorr01_sdss_r,
    }

    for k in (
        "delta_bass2decam_g", "delta_bass2decam_r",
        "delta_neb_g", "delta_neb_r",
        "delta_decam2sdss_g", "delta_decam2sdss_r",
        "delta_kcorr_g", "delta_kcorr_r",
        "mag_g_sdss_z0", "mag_r_sdss_z0",
    ):
        out[k + "_ONLY_CONT"] = chain_only_cont[k]

    return out


def plot_neb_correction_diagnostic(
    halpha_ew, delta_mag_g, delta_mag_r,
    save_path=None, gal_type="",
):
    """Three-panel diagnostic plot for the nebular emission correction.

    Shows the EW vs delta_mag distribution for all objects with EW > 0
    and finite model photometry.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from desi_lowz_funcs import make_subplots, get_contours

    ew  = np.asarray(halpha_ew, dtype=float)
    dmg = np.asarray(delta_mag_g, dtype=float)
    dmr = np.asarray(delta_mag_r, dtype=float)

    valid = (ew > 0) & np.isfinite(dmg) & np.isfinite(dmr)

    fig, ax = make_subplots(ncol=3, nrow=1, return_fig=True)
    x_bins = np.logspace(np.log10(7), np.log10(1.5e3), 25)

    # ---------- Panel 0: delta_mag_g ----------
    x0, y0 = ew[valid], dmg[valid]
    c0 = get_contours(x0, y0, x_bins, sigs=True)
    col0 = "cadetblue"
    ax[0].plot(c0["bin_cents"], c0["median"], color="k", lw=2)
    ax[0].fill_between(c0["bin_cents"], c0["sig1_low"], c0["sig1_high"],
                       alpha=0.5, color=col0)
    ax[0].fill_between(c0["bin_cents"], c0["sig2_low"], c0["sig2_high"],
                       alpha=0.25, color=col0)

    # ---------- Panel 1: delta_mag_r ----------
    x1, y1 = ew[valid], dmr[valid]
    c1 = get_contours(x1, y1, x_bins, sigs=True)
    col1 = "firebrick"
    ax[1].plot(c1["bin_cents"], c1["median"], color="k", lw=2)
    ax[1].fill_between(c1["bin_cents"], c1["sig1_low"], c1["sig1_high"],
                       alpha=0.5, color=col1)
    ax[1].fill_between(c1["bin_cents"], c1["sig2_low"], c1["sig2_high"],
                       alpha=0.25, color=col1)

    # ---------- Panel 2: delta(g-r) ----------
    x2 = ew[valid]
    y2 = dmg[valid] - dmr[valid]
    c2 = get_contours(x2, y2, x_bins, sigs=True)
    col2 = "grey"
    ax[2].plot(c2["bin_cents"], c2["median"], color="k", lw=2)
    ax[2].fill_between(c2["bin_cents"], c2["sig1_low"], c2["sig1_high"],
                       alpha=0.5, color=col2)
    ax[2].fill_between(c2["bin_cents"], c2["sig2_low"], c2["sig2_high"],
                       alpha=0.25, color=col2)

    for axi in ax:
        axi.set_xlim([10, 1e3])
        axi.set_xscale("log")
        axi.set_xlabel(r"H$\alpha$ EW ($\AA$)", fontsize=15)

    ax[0].set_ylim([0, 1])
    ax[1].set_ylim([0, 1])
    ax[2].set_ylim([-0.5, 0.5])

    ax[0].set_ylabel(r"$g_{\rm wo/neb}$ - $g_{\rm w/neb}$", fontsize=17.5)
    ax[1].set_ylabel(r"$r_{\rm wo/neb}$ - $r_{\rm w/neb}$", fontsize=17.5)
    ax[2].set_ylabel(
        r"$(g-r)_{\rm wo/neb}$ - $(g-r)_{\rm w/neb}$", fontsize=17.5)

    if gal_type:
        fig.suptitle(gal_type, fontsize=16, y=1.02)

    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved diagnostic plot: {save_path}")
    plt.close(fig)

import numpy as np
from astropy.cosmology import Planck18


def line_snr_mask(fastspec_cat, line_names=["HALPHA"], snr_val=3):
    """
    Returns a boolean mask selecting objects with line flux SNR > snr_val
    for the specified emission lines.
    """
    mask = np.ones(len(fastspec_cat), dtype=bool)

    for li in line_names:
        flux = fastspec_cat[f"{li}_FLUX"]
        ivar = fastspec_cat[f"{li}_FLUX_IVAR"]
        
        snr = flux * np.sqrt(ivar)
        mask &= (snr > snr_val) & (flux > 1) 

    return mask


def compute_o32(fastspec):
    '''
    Function that computes the O32 = OIII 5007 / OII 3726 index
    '''
    o32 = np.array(fastspec["OIII_5007_FLUX"]) / np.array(fastspec["OII_3726_FLUX"])
    return o32 


def compute_r32(fastspec):
    '''
    Function that computes the R32 = (OIII 4959,5007 + OI 3726) / Hbeta index
    '''
    r32 =  ( fastspec["OIII_5007_FLUX"] + fastspec["OIII_4959_FLUX"] + fastspec["OII_3726_FLUX"] ) / fastspec["HBETA_FLUX"]
    return np.array(r32)



def calc_SFR_Halpha(EW_Halpha, EW_Halpha_ivar, spec_z, spec_z_err, Mr, r_err, EWc=0, BD=3.25, BD_err=0.1,_IMF_FACTOR = 0.66):
    """
    Calculate Halpha-based EW SFR
    Bauer+ (2013) https://ui.adsabs.harvard.edu/abs/2013MNRAS.434..209B/abstract

    This function does an apeture correction through the Mr term

    we will set EWc = 0, because fastspecfit already accounts for stellar absorption
    """

    EW_Halpha_err = 1/np.sqrt(EW_Halpha_ivar)

    # Bauer, EQ 2, term1
    term1 = (EW_Halpha + EWc) * 10 ** (-0.4 * (Mr - 34.1))

    # Bauer Eq 2, term2
    term2 = 3e18 / (6564.6 * (1.0 + spec_z)) ** 2

    # Balmer Decrement
    term3 = (BD / 2.86) ** 2.36

    L_Halpha = term1 * term2 * term3

    # EQ 3, Bauer et al above, also account for Salpeter -> Koupa IMF
    # in SAGA, they assume some IMF_FACTOR = 0.66. See equation 2 of SAGA IV paper
    #https://github.com/sagasurvey/saga/blob/master/SAGA/objects/calc_sfr.py

    SFR = (L_Halpha * _IMF_FACTOR) / 1.27e34
    log_Ha_SFR = np.log10(SFR)

    # PROPAGATE ERRORS: EW_err, Mr_err and AV_err
    term1_EW_frac_err = EW_Halpha_err / (EW_Halpha + EWc)
    term1_Mr_frac_err = 0.4 * np.log(10) * r_err
    term1_frac_err = np.hypot(term1_EW_frac_err, term1_Mr_frac_err)
    
    term2_frac_err = 2.0 * spec_z_err / (1.0 + spec_z)
    
    term3_frac_err = 2.36 * (BD_err / BD)
    
    L_Halpha_frac_err = np.sqrt(term1_frac_err ** 2 + term2_frac_err ** 2 + term3_frac_err ** 2)
    #the above is the fractional error
    
    log_Ha_SFR_err  = L_Halpha_frac_err / np.log(10)

    return log_Ha_SFR, log_Ha_SFR_err


def get_halpha_sfrs(cat, halpha_ew, halpha_ew_ivar):
    '''
    Get approximate halpha based sfrs. Approximate because the aperture corrections for lowest redshift galaxies is difficult
    '''

    absm_r = cat["MAG_R"] + 5 - 5*np.log10(1e6*cat["LUMI_DIST_MPC"] )

    log_halpha_sfr, _  = calc_SFR_Halpha(halpha_ew, halpha_ew_ivar, cat["Z"], 0*cat["Z"].data, absm_r, 0 * cat["Z"].data)

    return log_halpha_sfr

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


def get_fastspecfit_path(survey, program, healpix,
                         base_dir="/global/cfs/cdirs/desi/public/dr1/vac/dr1/fastspecfit/iron/v3.0/healpix"):
    healpix_parent = healpix // 100
    filename = f"fastspec-{survey}-{program}-{healpix}.fits.gz"
    return f"{base_dir}/{survey}/{program}/{healpix_parent}/{healpix}/{filename}"


def measure_photo_batch(wave_arr, flux_2d):
    """
    Measure g and r AB magnitudes for a batch of spectra.

    Parameters
    ----------
    wave_arr : 1D array, shape (Nwave,)
    flux_2d : 2D array, shape (N_spectra, Nwave)
        Flux in 1e-17 erg/s/cm2/Ang.

    Returns
    -------
    mag_g, mag_r : 1D arrays, shape (N_spectra,)
    """
    wlen_f = wave_arr * u.Angstrom
    flux_f = flux_2d * 1e-17 * u.erg / (u.cm**2 * u.s * u.Angstrom)
    mag_g = DECAM_G.get_ab_magnitudes(flux_f, wlen_f)["decamDR1noatm-g"].data
    mag_r = DECAM_R.get_ab_magnitudes(flux_f, wlen_f)["decamDR1noatm-r"].data
    return mag_g, mag_r


def _process_single_file(args):
    """
    Process one fastspecfit FITS file for model-only photometry.
    Used as a multiprocessing worker by compute_photometry_catalog.

    Parameters
    ----------
    args : tuple
        (upath, cat_indices, targetids_for_file, batch_size)

    Returns
    -------
    dict or None
        Dict with 'cat_indices' and photometry/absmag arrays, or None on failure.
    """
    upath, cat_indices, targetids_for_file, batch_size = args

    try:
        iron_vac = fits.open(upath, memmap=True)
    except Exception:
        return None

    try:
        header = iron_vac["MODELS"].header
        wavelength = (header["CRVAL1"]
                      + (np.arange(header["NAXIS1"]) - header["CRPIX1"]) * header["CDELT1"])
        model_data = iron_vac["MODELS"].data

        specphot_data = iron_vac["SPECPHOT"].data
        fastspec_data = iron_vac["FASTSPEC"].data
        tgids_file = specphot_data["TARGETID"]
        tgid_to_fits_row = {t: i for i, t in enumerate(tgids_file)}

        valid_cat = []
        valid_fits_rows = []
        for ci_local, ci in enumerate(cat_indices):
            row = tgid_to_fits_row.get(targetids_for_file[ci_local])
            if row is not None:
                valid_cat.append(ci)
                valid_fits_rows.append(row)

        if len(valid_cat) == 0:
            return None

        valid_cat = np.array(valid_cat)
        valid_fits_rows = np.array(valid_fits_rows)
        n_valid = len(valid_cat)

        result = {
            "cat_indices":   valid_cat,
            "absmag_g":      np.array(specphot_data["ABSMAG01_SYNTH_SDSS_G"][valid_fits_rows], dtype=float),
            "absmag_r":      np.array(specphot_data["ABSMAG01_SYNTH_SDSS_R"][valid_fits_rows], dtype=float),
            "absmag_ivar_g": np.array(specphot_data["ABSMAG01_SYNTH_IVAR_SDSS_G"][valid_fits_rows], dtype=float),
            "absmag_ivar_r": np.array(specphot_data["ABSMAG01_SYNTH_IVAR_SDSS_R"][valid_fits_rows], dtype=float),
            "halpha_ew":      np.array(fastspec_data["HALPHA_EW"][valid_fits_rows], dtype=float),
            "halpha_ew_ivar": np.array(fastspec_data["HALPHA_EW_IVAR"][valid_fits_rows], dtype=float),
        }

        continuum = model_data[valid_fits_rows, 0, :]
        emission  = model_data[valid_fits_rows, 2, :]

        g_no_emi = np.full(n_valid, np.nan)
        r_no_emi = np.full(n_valid, np.nan)
        g_w_emi  = np.full(n_valid, np.nan)
        r_w_emi  = np.full(n_valid, np.nan)

        for flux_2d, g_arr, r_arr in [
            (continuum,            g_no_emi, r_no_emi),
            (continuum + emission, g_w_emi,  r_w_emi),
        ]:
            for start in range(0, n_valid, batch_size):
                end = min(start + batch_size, n_valid)
                try:
                    mg, mr = measure_photo_batch(wavelength, flux_2d[start:end])
                    g_arr[start:end] = mg
                    r_arr[start:end] = mr
                except Exception:
                    pass

        result["g_model_no_emi"] = g_no_emi
        result["r_model_no_emi"] = r_no_emi
        result["g_model_w_emi"]  = g_w_emi
        result["r_model_w_emi"]  = r_w_emi

    finally:
        iron_vac.close()

    return result


def compute_photometry_catalog(catalog,
                               spectra_h5_path=None,
                               compute_data_photometry=True,
                               base_dir="/global/cfs/cdirs/desi/public/dr1/vac/dr1/fastspecfit/iron/v3.0/healpix",
                               save_path=None,
                               batch_size=500,
                               ncore=1,
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

    absmag_g       = np.full(n_objects, np.nan)
    absmag_r       = np.full(n_objects, np.nan)
    absmag_ivar_g  = np.full(n_objects, np.nan)
    absmag_ivar_r  = np.full(n_objects, np.nan)

    halpha_ew      = np.full(n_objects, np.nan)
    halpha_ew_ivar = np.full(n_objects, np.nan)

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
            work_items.append((upath, ci, targetids[ci], batch_size))

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
                absmag_g[idx]       = file_result["absmag_g"]
                absmag_r[idx]       = file_result["absmag_r"]
                absmag_ivar_g[idx]  = file_result["absmag_ivar_g"]
                absmag_ivar_r[idx]  = file_result["absmag_ivar_r"]
                halpha_ew[idx]      = file_result["halpha_ew"]
                halpha_ew_ivar[idx] = file_result["halpha_ew_ivar"]
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

                specphot_data = iron_vac["SPECPHOT"].data
                fastspec_data = iron_vac["FASTSPEC"].data
                tgids_file = specphot_data["TARGETID"]
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

                absmag_g[valid_cat]      = specphot_data["ABSMAG01_SYNTH_SDSS_G"][valid_fits_rows]
                absmag_r[valid_cat]      = specphot_data["ABSMAG01_SYNTH_SDSS_R"][valid_fits_rows]
                absmag_ivar_g[valid_cat] = specphot_data["ABSMAG01_SYNTH_IVAR_SDSS_G"][valid_fits_rows]
                absmag_ivar_r[valid_cat] = specphot_data["ABSMAG01_SYNTH_IVAR_SDSS_R"][valid_fits_rows]

                halpha_ew[valid_cat]      = fastspec_data["HALPHA_EW"][valid_fits_rows]
                halpha_ew_ivar[valid_cat] = fastspec_data["HALPHA_EW_IVAR"][valid_fits_rows]

                continuum = model_data[valid_fits_rows, 0, :]
                emission  = model_data[valid_fits_rows, 2, :]

                model_variants = {
                    "model_no_emi": (continuum,            g_model_no_emi, r_model_no_emi),
                    "model_w_emi":  (continuum + emission, g_model_w_emi,  r_model_w_emi),
                }
                for vname, (flux_2d, g_out, r_out) in model_variants.items():
                    for start in range(0, len(valid_cat), batch_size):
                        end = min(start + batch_size, len(valid_cat))
                        try:
                            mg, mr = measure_photo_batch(wavelength, flux_2d[start:end])
                            g_out[valid_cat[start:end]] = mg
                            r_out[valid_cat[start:end]] = mr
                        except Exception as e:
                            if verbose:
                                print(f"  Photometry error ({vname}, batch {start}-{end}): {e}")

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
                        for vname, (flux_2d, g_out, r_out) in data_variants.items():
                            for start in range(0, len(sub_cat), batch_size):
                                end = min(start + batch_size, len(sub_cat))
                                try:
                                    mg, mr = measure_photo_batch(h5_wave, flux_2d[start:end])
                                    g_out[sub_cat[start:end]] = mg
                                    r_out[sub_cat[start:end]] = mr
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
        "ABSMAG01_SYNTH_SDSS_G":      absmag_g,
        "ABSMAG01_SYNTH_SDSS_R":      absmag_r,
        "ABSMAG01_SYNTH_IVAR_SDSS_G": absmag_ivar_g,
        "ABSMAG01_SYNTH_IVAR_SDSS_R": absmag_ivar_r,
        "HALPHA_EW":                   halpha_ew,
        "HALPHA_EW_IVAR":             halpha_ew_ivar,
    }
    if compute_data_photometry:
        columns["g_data_no_emi"] = g_data_no_emi
        columns["r_data_no_emi"] = r_data_no_emi
        columns["g_data_w_emi"]  = g_data_w_emi
        columns["r_data_w_emi"]  = r_data_w_emi

    result = Table(columns)

    if save_path is not None:
        result.write(save_path, overwrite=True)
        if verbose:
            print(f"Saved to {save_path}")

    if verbose:
        n_good_model = np.sum(np.isfinite(g_model_w_emi))
        n_good_absmag = np.sum(np.isfinite(absmag_g))
        msg = (f"Done. {n_good_model}/{n_objects} with valid model photometry, "
               f"{n_good_absmag}/{n_objects} with valid absolute mags.")
        if compute_data_photometry:
            n_good_data = np.sum(np.isfinite(g_data_w_emi))
            msg += f" {n_good_data}/{n_objects} with valid data photometry."
        print(msg)

    return result


# ======================================================================
# Nebular emission correction helpers
# ======================================================================

def _build_ew_relation(ew, delta_mag, n_bins=25, ew_lo=7.0, ew_hi=1500.0,
                       min_count=5):
    """Build a piecewise-linear lookup between HALPHA_EW and delta_mag.

    Parameters
    ----------
    ew, delta_mag : 1-D arrays (same length, high-SNR subsample only)
    n_bins : int
        Number of log-spaced bin edges.
    ew_lo, ew_hi : float
        EW range for the bin edges.
    min_count : int
        Minimum number of sources in a bin to keep it.

    Returns
    -------
    dict with keys 'bin_cents', 'log10_bin_cents', 'median', 'scatter_1sig'.
    Bins with fewer than *min_count* sources are dropped.
    """
    x_bins = np.logspace(np.log10(ew_lo), np.log10(ew_hi), n_bins)
    bin_cents = 0.5 * (x_bins[1:] + x_bins[:-1])

    medians = np.full(len(bin_cents), np.nan)
    scatter = np.full(len(bin_cents), np.nan)

    for i in range(len(x_bins) - 1):
        in_bin = (ew > x_bins[i]) & (ew <= x_bins[i + 1]) & np.isfinite(delta_mag)
        if np.sum(in_bin) >= min_count:
            vals = delta_mag[in_bin]
            medians[i] = np.nanmedian(vals)
            scatter[i] = (np.nanpercentile(vals, 84)
                          - np.nanpercentile(vals, 16)) / 2.0

    good = np.isfinite(medians) & np.isfinite(scatter)
    return {
        "bin_cents":       bin_cents[good],
        "log10_bin_cents": np.log10(bin_cents[good]),
        "median":          medians[good],
        "scatter_1sig":    scatter[good],
    }


def apply_neb_correction_with_ew_relation(
    halpha_ew, halpha_ew_ivar,
    delta_mag_g_direct, delta_mag_r_direct,
    delta_mag_g_err_direct, delta_mag_r_err_direct,
    snr_threshold=5.0, ew_max_extrap=1000.0,
    n_bins=25, ew_bin_lo=7.0, ew_bin_hi=1500.0,
):
    """Apply three-tier nebular emission correction to broadband photometry.

    Tier 1 -- high SNR  (EW_SNR > *snr_threshold* **and** EW > 0):
        Use the directly measured delta_mag from fastspec model photometry.
    Tier 2 -- low SNR   (EW_SNR <= *snr_threshold* **and** EW > 0):
        Piecewise-linear interpolation of the median relation built from
        the high-SNR subsample.  The 1-sigma scatter of that relation is
        stored as the error.  EW is clamped to [min_bin_center, *ew_max_extrap*].
    Tier 3 -- negative EW  (EW <= 0):
        No correction (delta_mag = 0, error = 0).

    All delta_mag values are capped at >= 0 (corrections can only make
    sources fainter).

    Returns
    -------
    delta_mag_g, delta_mag_r, delta_mag_g_err, delta_mag_r_err : 1-D arrays
    relation_info : dict
        Contains 'g' and 'r' sub-dicts from ``_build_ew_relation`` plus
        'train_mask' boolean array for the high-SNR training set.
    """
    ew   = np.asarray(halpha_ew, dtype=float)
    ivar = np.asarray(halpha_ew_ivar, dtype=float)

    with np.errstate(divide="ignore", invalid="ignore"):
        snr = ew * np.sqrt(np.maximum(ivar, 0.0))

    # Cap direct delta_mags at zero
    dmg = np.where(np.isfinite(delta_mag_g_direct),
                   np.maximum(np.asarray(delta_mag_g_direct, dtype=float), 0.0),
                   np.nan)
    dmr = np.where(np.isfinite(delta_mag_r_direct),
                   np.maximum(np.asarray(delta_mag_r_direct, dtype=float), 0.0),
                   np.nan)

    # ---- masks ----
    high_snr = (snr > snr_threshold) & (ew > 0)
    low_snr  = ~high_snr & (ew > 0)

    # ---- build median relations from high-SNR subsample ----
    train_mask = high_snr & np.isfinite(dmg) & np.isfinite(dmr)
    rel_g = _build_ew_relation(ew[train_mask], dmg[train_mask],
                               n_bins, ew_bin_lo, ew_bin_hi)
    rel_r = _build_ew_relation(ew[train_mask], dmr[train_mask],
                               n_bins, ew_bin_lo, ew_bin_hi)

    # ---- populate output (initialised to zero = Tier 3 default) ----
    n = len(ew)
    out_g     = np.zeros(n)
    out_r     = np.zeros(n)
    out_g_err = np.zeros(n)
    out_r_err = np.zeros(n)

    # Tier 1: high SNR -- direct values
    out_g[high_snr]     = dmg[high_snr]
    out_r[high_snr]     = dmr[high_snr]
    out_g_err[high_snr] = np.asarray(delta_mag_g_err_direct, dtype=float)[high_snr]
    out_r_err[high_snr] = np.asarray(delta_mag_r_err_direct, dtype=float)[high_snr]

    # Tier 2: low SNR -- piecewise-linear interpolation of the relation
    if np.any(low_snr) and len(rel_g["bin_cents"]) > 1:
        ew_clamped = np.clip(ew[low_snr], 0, ew_max_extrap)
        log_ew = np.log10(np.maximum(ew_clamped, 1e-10))

        # left=0  -> EW below lowest bin center gets no correction
        out_g[low_snr] = np.maximum(
            np.interp(log_ew, rel_g["log10_bin_cents"], rel_g["median"],
                      left=0.0), 0.0)
        out_r[low_snr] = np.maximum(
            np.interp(log_ew, rel_r["log10_bin_cents"], rel_r["median"],
                      left=0.0), 0.0)
        out_g_err[low_snr] = np.maximum(
            np.interp(log_ew, rel_g["log10_bin_cents"],
                      rel_g["scatter_1sig"], left=0.0), 0.0)
        out_r_err[low_snr] = np.maximum(
            np.interp(log_ew, rel_r["log10_bin_cents"],
                      rel_r["scatter_1sig"], left=0.0), 0.0)

    # High-SNR sources with NaN model photometry -> propagate NaN
    nan_mask = high_snr & (~np.isfinite(dmg) | ~np.isfinite(dmr))
    out_g[nan_mask]     = np.nan
    out_r[nan_mask]     = np.nan
    out_g_err[nan_mask] = np.nan
    out_r_err[nan_mask] = np.nan

    # ---- diagnostics ----
    n_high = int(np.sum(high_snr))
    n_low  = int(np.sum(low_snr))
    n_neg  = int(np.sum(ew <= 0))
    n_nan  = int(np.sum(nan_mask))
    print(f"  Nebular correction tiers: {n_high} high-SNR (direct), "
          f"{n_low} low-SNR (interpolated), {n_neg} EW<=0 (no correction)")
    if n_nan > 0:
        print(f"  WARNING: {n_nan} high-SNR sources have NaN model photometry")
    if len(rel_g["bin_cents"]) > 0:
        print(f"  Relation EW range: "
              f"[{rel_g['bin_cents'][0]:.1f}, {rel_g['bin_cents'][-1]:.1f}] "
              f"Angstrom ({len(rel_g['bin_cents'])} bins)")
    print(f"  Median delta_mag_g = {np.nanmedian(out_g):.4f}, "
          f"Median delta_mag_r = {np.nanmedian(out_r):.4f}")

    relation_info = {"g": rel_g, "r": rel_r, "train_mask": train_mask}
    return out_g, out_r, out_g_err, out_r_err, relation_info


def plot_neb_correction_diagnostic(
    halpha_ew, delta_mag_g, delta_mag_r,
    halpha_ew_snr, snr_threshold, relation_info,
    save_path=None, gal_type="",
):
    """Three-panel diagnostic plot for the nebular emission correction.

    Shows the high-SNR data distribution (``get_contours`` shaded bands)
    and overlays the piecewise-linear interpolation curve +/- 1-sigma
    scatter used for the low-SNR tier.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from desi_lowz_funcs import make_subplots, get_contours

    ew  = np.asarray(halpha_ew, dtype=float)
    snr = np.asarray(halpha_ew_snr, dtype=float)
    dmg = np.asarray(delta_mag_g, dtype=float)
    dmr = np.asarray(delta_mag_r, dtype=float)

    high_snr = (snr > snr_threshold) & (ew > 0) & np.isfinite(dmg) & np.isfinite(dmr)

    fig, ax = make_subplots(ncol=3, nrow=1, return_fig=True)
    x_bins = np.logspace(np.log10(7), np.log10(1.5e3), 25)

    # ---------- Panel 0: delta_mag_g ----------
    x0, y0 = ew[high_snr], dmg[high_snr]
    c0 = get_contours(x0, y0, x_bins, sigs=True)
    col0 = "cadetblue"
    ax[0].plot(c0["bin_cents"], c0["median"], color="k", lw=2)
    ax[0].fill_between(c0["bin_cents"], c0["sig1_low"], c0["sig1_high"],
                       alpha=0.5, color=col0)
    ax[0].fill_between(c0["bin_cents"], c0["sig2_low"], c0["sig2_high"],
                       alpha=0.25, color=col0)

    rel_g = relation_info["g"]
    if len(rel_g["bin_cents"]) > 0:
        ax[0].plot(rel_g["bin_cents"], rel_g["median"],
                   color="orangered", lw=2, ls="--", label="interp median")
        ax[0].fill_between(
            rel_g["bin_cents"],
            np.maximum(rel_g["median"] - rel_g["scatter_1sig"], 0),
            rel_g["median"] + rel_g["scatter_1sig"],
            alpha=0.3, color="orangered", label=r"interp $\pm1\sigma$")
        ax[0].legend(fontsize=8, loc="upper left")

    # ---------- Panel 1: delta_mag_r ----------
    x1, y1 = ew[high_snr], dmr[high_snr]
    c1 = get_contours(x1, y1, x_bins, sigs=True)
    col1 = "firebrick"
    ax[1].plot(c1["bin_cents"], c1["median"], color="k", lw=2)
    ax[1].fill_between(c1["bin_cents"], c1["sig1_low"], c1["sig1_high"],
                       alpha=0.5, color=col1)
    ax[1].fill_between(c1["bin_cents"], c1["sig2_low"], c1["sig2_high"],
                       alpha=0.25, color=col1)

    rel_r = relation_info["r"]
    if len(rel_r["bin_cents"]) > 0:
        ax[1].plot(rel_r["bin_cents"], rel_r["median"],
                   color="orangered", lw=2, ls="--")
        ax[1].fill_between(
            rel_r["bin_cents"],
            np.maximum(rel_r["median"] - rel_r["scatter_1sig"], 0),
            rel_r["median"] + rel_r["scatter_1sig"],
            alpha=0.3, color="orangered")

    # ---------- Panel 2: delta(g-r) ----------
    x2 = ew[high_snr]
    y2 = dmg[high_snr] - dmr[high_snr]
    c2 = get_contours(x2, y2, x_bins, sigs=True)
    col2 = "grey"
    ax[2].plot(c2["bin_cents"], c2["median"], color="k", lw=2)
    ax[2].fill_between(c2["bin_cents"], c2["sig1_low"], c2["sig1_high"],
                       alpha=0.5, color=col2)
    ax[2].fill_between(c2["bin_cents"], c2["sig2_low"], c2["sig2_high"],
                       alpha=0.25, color=col2)

    if len(rel_g["bin_cents"]) > 0 and len(rel_r["bin_cents"]) > 0:
        common_lo = max(rel_g["log10_bin_cents"][0],
                        rel_r["log10_bin_cents"][0])
        common_hi = min(rel_g["log10_bin_cents"][-1],
                        rel_r["log10_bin_cents"][-1])
        common_log_ew = np.linspace(common_lo, common_hi, 50)
        common_ew = 10**common_log_ew
        ig = np.interp(common_log_ew, rel_g["log10_bin_cents"],
                       rel_g["median"])
        ir = np.interp(common_log_ew, rel_r["log10_bin_cents"],
                       rel_r["median"])
        ax[2].plot(common_ew, ig - ir, color="orangered", lw=2, ls="--")

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

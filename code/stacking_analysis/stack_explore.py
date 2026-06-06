import os
import sys

# This module lives in code/stacking_analysis/, but its helpers (and the
# helpers' own transitive imports) reach into modules that live one level up
# in code/: desi_lowz_funcs.py, mass_and_photo_corrections.py, and the
# nnmf_pca_analysis package. Make code/ importable so the script works
# regardless of the caller's cwd or PYTHONPATH.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_CODE_DIR = os.path.dirname(_THIS_DIR)
if _CODE_DIR not in sys.path:
    sys.path.insert(0, _CODE_DIR)

from astropy.table import Table, hstack
import h5py
import numpy as np
from nnmf_pca_analysis.nnmf_analysis import deredshift_resample_desi_spectra
from desi_lowz_funcs import print_stage
from mass_and_photo_corrections import DWARF_CATALOG_SPEC_HDU, DWARF_CATALOG_DERIVED_HDU

##deredshifting functions

def deredshift_for_stacking(use_invvar=True, delta_wave=None):
    spectra_dir = "/pscratch/sd/v/virajvm/catalog_dr1_dwarfs/iron_spectra/spectra_files"

    if delta_wave is None:
        grid_suffix = "_native"
    elif delta_wave == 0.2:
        grid_suffix = "_hires"
    else:
        grid_suffix = f"_d{delta_wave}"

    save_dered = os.path.join(
        spectra_dir,
        f"desi_y1_dwarf_combine_deredshift{grid_suffix}.h5",
    )

    # When using the flux-conserving rebin (use_invvar=False), write to a
    # distinct file so the existing inverse-variance-weighted output is not
    # overwritten.
    if not use_invvar:
        base, ext = os.path.splitext(save_dered)
        save_dered = f"{base}_noinvvar{ext}"

    print(f"Making deredshited spectra file! (use_invvar={use_invvar}, delta_wave={delta_wave})")

    #read the entire consolidated file!
    with h5py.File("/pscratch/sd/v/virajvm/catalog_dr1_dwarfs/iron_spectra/spectra_files/data/desi_dr1_dwarf_catalog_spectra.h5", "r") as f:
        wave = f["WAVE"][:]
        all_flux = f["FLUX"][:] 
        all_flux_ivar = f["FLUX_IVAR"][:] 
        all_zreds = f["Z"][:]  
        all_tgids = f["TARGETID"][:] 
        
    print("wave shape", wave.shape)
    print("flux shape", all_flux.shape)
    print("flux_ivar shape", all_flux_ivar.shape)
    print("zreds shape", all_zreds.shape)
    
    # ################

    ##I should de-redshift this once and then save it!

    print_stage("De-redshifting the spectra and clipping to relevant wavelength range")

    if delta_wave is None:
        wave_out = wave
        step = np.median(np.diff(wave_out)) if len(wave_out) > 1 else np.nan
        print(f"wave_out: native DESI grid, n={len(wave_out)}, median step={step:.4f} A")
    else:
        wave_out = np.arange(3600, 9800, delta_wave)
        print(f"wave_out: linear grid 3600-9800 A, n={len(wave_out)}, step={delta_wave} A")

    print(f"save path: {save_dered}")
    print(wave_out[:10])
    print(wave_out[-10:])
    
    wave_rest, all_fluxs_out, all_flux_ivars_out = deredshift_resample_desi_spectra(wave, all_flux, all_flux_ivar, all_zreds,
                                     wave_out=wave_out, ncores=128,verbose=True,
                                     use_invvar=use_invvar)

    with h5py.File(save_dered, "w") as f:
        f.create_dataset("TARGETID", data=all_tgids, dtype='i8')
        f.create_dataset("Z", data=all_zreds, dtype='f4')
        f.create_dataset("WAVE_REST", data=wave_rest, dtype='f4')
        f.create_dataset("FLUX", data=all_fluxs_out, dtype='f4')
        f.create_dataset("FLUX_IVAR", data=all_flux_ivars_out, dtype='f4')

    print(f"Saved {save_dered}")
    return 

##data loading functions

def load_catalog(filename="/pscratch/sd/v/virajvm/desi_dwarf_catalogs/dr1/v1.0/desi_dr1_dwarf_catalog.fits"):
    """Load and filter the dwarf galaxy catalog."""
    
    tot_cat = Table.read(filename, hdu="MAIN")

    tractor_cat = Table.read(filename, hdu="TRACTOR")
    
    fspec_cat = Table.read(filename, hdu=DWARF_CATALOG_SPEC_HDU)

    # DELTA_MAG_* (and MAG_*_MODEL_*) columns now live in the SPEC_DERIVED HDU,
    # appended by add_nebular_props.py. Row order matches MAIN/FASTSPEC.
    derived_cat = Table.read(filename, hdu=DWARF_CATALOG_DERIVED_HDU)

    print(f"Total catalog size = {len(tot_cat)}")

    halpha_snr = ( np.array(fspec_cat["HALPHA_FLUX"]) * np.sqrt(np.array(fspec_cat["HALPHA_FLUX_IVAR"])) )

    mask = (tot_cat["DWARF_MASKBIT"] == 0) & (halpha_snr > 3) & ( np.array(fspec_cat["HALPHA_FLUX"]) > 1) #& (tot_cat["MAG_TYPE"] == "TRACTOR_OG")
    
    tot_cat_f = tot_cat[ mask ]

    fspec_cat_f = fspec_cat[ mask ]

    tractor_cat_f = tractor_cat[ mask ]

    derived_cat_f = derived_cat[ mask ]
    
    tot_cat_new =  hstack( [tot_cat_f, fspec_cat_f["HALPHA_FLUX", "HALPHA_FLUX_IVAR","HBETA_FLUX", "HBETA_FLUX_IVAR", "OIII_5007_FLUX", "OIII_5007_FLUX_IVAR", "HALPHA_EW", "HALPHA_EW_IVAR", "OII_3726_FLUX", "OII_3726_FLUX_IVAR", "OII_3729_FLUX", "OII_3729_FLUX_IVAR", "HALPHA_BOXFLUX", "HALPHA_BOXFLUX_IVAR"], derived_cat_f["DELTA_MAG_G_KCORR", "DELTA_MAG_R_KCORR"], tractor_cat_f["FLUX_G","FLUX_R","FLUX_Z","FLUX_IVAR_G", "FLUX_IVAR_R", "FLUX_IVAR_Z"] ] )
    
    print(f"Cleaned catalog size = {len(tot_cat_new)}")
    
    return tot_cat_new


def load_spectra(h5_file="/pscratch/sd/v/virajvm/catalog_dr1_dwarfs/iron_spectra/spectra_files/desi_y1_dwarf_combine_deredshift_hires.h5"):
    """Load de-redshifted spectra from HDF5 file."""
    print(f"Reading file = {h5_file}")
    with h5py.File(h5_file, "r") as f:
        data = {
            "targetid": f["TARGETID"][:],
            "z": f["Z"][:],
            "wave_rest": f["WAVE_REST"][:],
            "flux": f["FLUX"][:],
            "flux_ivar": f["FLUX_IVAR"][:]
        }
    return data


def select_sample(catalog, sample_name, z_min=0.05, z_max=0.1, 
                  logmstar_min=6, logmstar_max=9.25):
    """Select galaxies matching sample criteria."""

    if sample_name == "ELG":
        samp_mask = (catalog["SAMPLE"] == "ELG")
    elif sample_name == "NO_ELG":
        samp_mask = (catalog["SAMPLE"] != "ELG")
    else:
        samp_mask = (catalog["SAMPLE"] == sample_name)
    
    # Half-open mass interval (low-exclusive, high-inclusive) so a galaxy
    # sitting exactly on a bin boundary lands in the lower-edge bin and is
    # not silently dropped from both neighbors. Matches select_sample_2d in
    # stack_mstar_haew.py.
    mask = (
         samp_mask &
        (catalog["Z"] > z_min) &
        (catalog["Z"] < z_max) &
        (catalog["LOG_MSTAR_M24"] > logmstar_min) &
        (catalog["LOG_MSTAR_M24"] <= logmstar_max)
    )

    print(f"Total number selected = {np.sum(mask)}")
    return catalog[mask]


def get_sample_spectra_with_linenorm(catalog_subset, spectra_data, line_norm="HALPHA", norm_col=None):
    """
    Extract spectra for galaxies in the catalog subset, along with their Halpha fluxes.

    Parameters
    ----------
    catalog_subset : Table
    spectra_data : dict
    line_norm : str
        Line stem used to build the default normalization column
        (``f"{line_norm}_FLUX"``).
    norm_col : str or None
        If given, use this catalog column directly for the normalization flux
        (overrides ``f"{line_norm}_FLUX"``). e.g. ``"HALPHA_BOXFLUX"`` to
        normalize by the boxcar Halpha flux instead of the Gaussian fit.

    Returns
    -------
    fluxes : 2D array (n_spectra, n_wavelengths)
    ivars : 2D array (n_spectra, n_wavelengths)
    halpha_fluxes : 1D array (n_spectra,) - line flux from catalog
    n_matched : int
    """
    cat_targetids = catalog_subset["TARGETID"]

    col = norm_col if norm_col is not None else f"{line_norm}_FLUX"
    cat_line = catalog_subset[col]
    
    spec_targetids = spectra_data["targetid"]
    
    # Fast lookup
    spec_id_to_idx = {tid: idx for idx, tid in enumerate(spec_targetids)}
    
    matched_spec_indices = []
    matched_cat_indices = []
    
    for i, tid in enumerate(cat_targetids):
        if tid in spec_id_to_idx:
            matched_spec_indices.append(spec_id_to_idx[tid])
            matched_cat_indices.append(i)
    
    matched_spec_indices = np.array(matched_spec_indices)
    matched_cat_indices = np.array(matched_cat_indices)
    
    if len(matched_spec_indices) == 0:
        return None, None, None, None
    
    fluxes = spectra_data["flux"][matched_spec_indices]
    ivars = spectra_data["flux_ivar"][matched_spec_indices]
    line_fluxes = np.array(cat_line[matched_cat_indices])

    return fluxes, ivars, line_fluxes, cat_targetids[matched_cat_indices]


#### spectra manipulations

def normalize_by_boxcar_line(fluxes, ivars, wave, line_window, cont_width=5.0):
    """
    Normalize each spectrum by its own boxcar-measured line flux.
    This ensures self-consistency: the same method used to normalize
    is the same method used to measure lines on the stack.

    Parameters
    ----------
    fluxes : 2D array (n_spectra, n_wavelengths)
    ivars : 2D array (n_spectra, n_wavelengths)
    wave : 1D array (n_wavelengths,)
    line_window : tuple (lam_lo, lam_hi)
        Same window used in boxcar_line_flux for this line.
    cont_width : float
        Continuum sideband width (same as used in LINE_WINDOWS).

    Returns
    -------
    norm_fluxes : 2D array
    norm_ivars : 2D array
    valid_mask : 1D boolean array
    line_fluxes : 1D array - the measured line fluxes used for normalization
    """
    n_spec = fluxes.shape[0]
    line_fluxes = np.zeros(n_spec)

    for i in range(n_spec):
        lf, _ = boxcar_line_flux(wave, fluxes[i], line_window, cont_width=cont_width)
        line_fluxes[i] = lf

    valid_mask = np.isfinite(line_fluxes) & (line_fluxes > 0)

    norm_fluxes = np.zeros_like(fluxes)
    norm_ivars = np.zeros_like(ivars)

    for i in range(n_spec):
        if valid_mask[i]:
            norm_fluxes[i] = fluxes[i] / line_fluxes[i]
            norm_ivars[i] = ivars[i] * line_fluxes[i] ** 2

    n_valid = np.sum(valid_mask)
    print(f"    Line normalization: {n_valid}/{n_spec} spectra with valid line flux")

    return norm_fluxes, norm_ivars, valid_mask, line_fluxes


def normalize_by_line_catalog(fluxes, ivars, line_fluxes):
    """
    Normalize each spectrum by its line flux from the catalog.
    NOTE: This will NOT be self-consistent with boxcar measurements on the stack.
    Use normalize_by_boxcar_line instead for self-consistency.

    Parameters
    ----------
    fluxes : 2D array (n_spectra, n_wavelengths)
    ivars : 2D array (n_spectra, n_wavelengths)
    line_fluxes : 1D array (n_spectra,) - line fluxes from catalog

    Returns
    -------
    norm_fluxes : 2D array
    norm_ivars : 2D array
    valid_mask : 1D boolean array
    """
    n_spec = fluxes.shape[0]
    valid_mask = np.isfinite(line_fluxes) & (line_fluxes > 0)

    norm_fluxes = np.zeros_like(fluxes)
    norm_ivars = np.zeros_like(ivars)

    norm_fluxes[valid_mask] = fluxes[valid_mask] / line_fluxes[valid_mask, None]
    norm_ivars[valid_mask] = ivars[valid_mask] * line_fluxes[valid_mask, None] ** 2

    n_valid = np.sum(valid_mask)
    print(f"    Catalog normalization: {n_valid}/{n_spec} spectra with valid line flux")

    return norm_fluxes, norm_ivars, valid_mask


def coadd_mean_with_propagated_ivar(norm_flux, norm_ivar):
    """Unweighted mean coadd + propagated measurement ivar.

    norm_flux, norm_ivar: (N_gal, N_wave), already Hα-boxflux-normalized.
    Masked/uncovered pixels are returned as flux=0.0, ivar=0.0 (NOT NaN).
    """
    valid = np.isfinite(norm_flux) & (norm_ivar > 0)
    n_contrib = valid.sum(axis=0)

    f = np.where(valid, norm_flux, np.nan)
    mean = np.nanmean(f, axis=0)
    stack_flux = np.where(n_contrib > 0, mean, 0.0)

    var_i = np.where(valid, 1.0 / norm_ivar, np.nan)
    sum_var = np.nansum(var_i, axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        stack_ivar = np.where(
            (n_contrib > 0) & (sum_var > 0), n_contrib ** 2 / sum_var, 0.0,
        )

    return stack_flux.astype(np.float32), stack_ivar.astype(np.float32)


def bootstrap_stack(fluxes, ivars, wave, n_bootstrap=200, n_draw=5000,
                    random_seed=42,
                    norm_method="boxcar_line",
                    line_window=None, cont_width=5.0,
                    flux_window=None,
                    catalog_line_fluxes=None,
                    min_n_valid=25):
    """
    Bootstrap stack with flexible normalization methods.

    Parameters
    ----------
    fluxes : 2D array (n_spectra, n_wavelengths)
    ivars : 2D array (n_spectra, n_wavelengths)
    wave : 1D array
    n_bootstrap : int
    n_draw : int
    random_seed : int
    norm_method : str
        "boxcar_line" - normalize by self-measured boxcar line flux (recommended)
        "flux_window" - normalize by integrated flux in a window
        "catalog" - normalize by catalog line flux (not self-consistent)
    line_window : tuple (lam_lo, lam_hi)
        Required for norm_method="boxcar_line".
    cont_width : float
        Continuum width for boxcar. Default 5.0.
    flux_window : tuple (w1, w2)
        Required for norm_method="flux_window".
    catalog_line_fluxes : 1D array
        Required for norm_method="catalog".
    min_n_valid : int
        Minimum number of normalized spectra required to stack. Default 25
        preserves legacy behavior; pass 1 to allow single-spectrum bins.

    Returns
    -------
    central_flux : 1D array
        Mean flux over bootstrap realizations.
    boot_std : 1D array
        Per-pixel bootstrap standard deviation (diagnostic only).
    real_flux : 2D array (n_bootstrap, n_wavelengths) or None
        Per-realization coadded flux.
    real_ivar : 2D array (n_bootstrap, n_wavelengths) or None
        Per-realization propagated measurement ivar.
    central_ivar : 1D array
        Mean propagated measurement ivar over realizations (Scholte step v).
    """
    rng = np.random.default_rng(random_seed)

    # Normalize
    if norm_method == "boxcar_line":
        if line_window is None:
            raise ValueError("line_window required for boxcar_line normalization")
        norm_fluxes, norm_ivars, valid, _ = normalize_by_boxcar_line(
            fluxes, ivars, wave, line_window, cont_width=cont_width
        )
    elif norm_method == "flux_window":
        if flux_window is None:
            raise ValueError("flux_window required for flux_window normalization")
        norm_fluxes, norm_ivars, valid, _ = normalize_by_flux_window(
            fluxes, ivars, wave, flux_window
        )
    elif norm_method == "catalog":
        if catalog_line_fluxes is None:
            raise ValueError("catalog_line_fluxes required for catalog normalization")
        norm_fluxes, norm_ivars, valid = normalize_by_line_catalog(
            fluxes, ivars, catalog_line_fluxes
        )
    else:
        raise ValueError(f"Unknown norm_method: {norm_method}")

    use_fluxes = norm_fluxes[valid].astype(np.float32, copy=False)
    use_ivars = norm_ivars[valid].astype(np.float32, copy=False)
    n_valid, n_wave = use_fluxes.shape

    nan_flux = np.full(n_wave, np.nan, dtype=np.float32)
    nan_ivar = np.full(n_wave, np.nan, dtype=np.float32)

    if n_valid < min_n_valid:
        print(f"    Warning: only {n_valid} valid spectra "
              f"(< min_n_valid={min_n_valid}), returning NaN")
        return nan_flux, nan_flux, None, None, nan_ivar

    print(f"    Bootstrap: n_valid={n_valid}, n_bootstrap={n_bootstrap}")

    real_flux = np.empty((n_bootstrap, n_wave), dtype=np.float32)
    real_ivar = np.empty((n_bootstrap, n_wave), dtype=np.float32)

    if n_valid == 1:
        flux0, ivar0 = coadd_mean_with_propagated_ivar(use_fluxes, use_ivars)
        real_flux[:] = flux0
        real_ivar[:] = ivar0
        boot_std = np.zeros(n_wave, dtype=np.float32)
        return flux0, boot_std, real_flux, real_ivar, ivar0

    for b in range(n_bootstrap):
        idx = rng.integers(0, n_valid, size=n_valid)
        real_flux[b], real_ivar[b] = coadd_mean_with_propagated_ivar(
            use_fluxes[idx], use_ivars[idx],
        )

    central_flux = np.nanmean(real_flux, axis=0).astype(np.float32)
    central_ivar = np.nanmean(real_ivar, axis=0).astype(np.float32)
    boot_std = np.nanstd(real_flux, axis=0).astype(np.float32)

    return central_flux, boot_std, real_flux, real_ivar, central_ivar



#### line flux measurement functions

def shift_line_windows(line_windows, redshift):
    """
    Shift rest-frame line windows to observed frame at given redshift.
    
    Parameters
    ----------
    line_windows : dict
        Rest-frame line windows, e.g. {"Halpha": (6557.0, 6572.0, 5)}
        Format: (lam_lo, lam_hi, cont_width)
    redshift : float
        Redshift to shift to. Use 0 for rest-frame.
    
    Returns
    -------
    shifted : dict
        Same structure with wavelengths scaled by (1+z).
    """
    shifted = {}
    for name, (lam_lo, lam_hi, cont_w) in line_windows.items():
        shifted[name] = (
            lam_lo * (1 + redshift),
            lam_hi * (1 + redshift),
            cont_w * (1 + redshift),
        )
    return shifted


def boxcar_line_flux(wave, flux, line_window, cont_width=5.0):
    """
    Measure emission line flux via boxcar integration with local flat continuum.

    Parameters
    ----------
    wave : 1D array
        Wavelength grid.
    flux : 1D array
        Flux density (per Angstrom).
    line_window : tuple (lam_lo, lam_hi)
        Wavelength bounds for the line integration.
    cont_width : float
        Width (in Angstroms) of continuum sidebands on each side.

    Returns
    -------
    line_flux : float
        Continuum-subtracted integrated line flux.
    cont_level : float
        Estimated continuum level.
    """
    lam_lo, lam_hi = line_window

    line_mask = (wave >= lam_lo) & (wave <= lam_hi)
    cont_left = (wave >= lam_lo - cont_width) & (wave < lam_lo)
    cont_right = (wave > lam_hi) & (wave <= lam_hi + cont_width)
    cont_mask = cont_left | cont_right

    if not np.any(line_mask) or not np.any(cont_mask):
        return np.nan, np.nan

    cont_level = np.median(flux[cont_mask])
    dlam = np.gradient(wave)
    line_flux = np.sum((flux[line_mask] - cont_level) * dlam[line_mask])

    return line_flux, cont_level


def measure_all_lines_boxcar(wave, flux, line_windows, redshift=0.0, plot=False):
    """Measure multiple emission lines using boxcar integration."""
    windows = shift_line_windows(line_windows, redshift)
    results = {}
    for name, window in windows.items():
        lf, cont = boxcar_line_flux(
            wave, flux, (window[0], window[1]),
            cont_width=window[2])
        results[name + "_flux"] = lf
        results[name + "_cont"] = cont
    return results
    

def measure_lines_bootstrap(wave, all_stacks, line_windows):
    """
    Measure line fluxes on each bootstrap realization and return
    median + 16/84 percentiles for asymmetric errors.
    
    Parameters
    ----------
    wave : array
        Wavelength array
    all_stacks : array, shape (n_bootstrap, n_wave)
        Bootstrap realizations of stacked spectrum
    line_windows : dict
        Line window definitions
        
    Returns
    -------
    results : dict
        For each line: _flux, _flux_err, _flux_lo, _flux_hi, _cont, _cont_err
    """
    n_boot = all_stacks.shape[0]
    
    # Measure lines on each bootstrap
    all_measurements = {name: {"flux": [], "cont": []} for name in line_windows.keys()}
    
    for i in range(n_boot):
        meas = measure_all_lines_boxcar(wave, all_stacks[i], line_windows, plot=False)
        for name in line_windows.keys():
            all_measurements[name]["flux"].append(meas[name + "_flux"])
            all_measurements[name]["cont"].append(meas[name + "_cont"])
    
    # Compute percentiles
    results = {}
    for name in line_windows.keys():
        flux_arr = np.array(all_measurements[name]["flux"])
        cont_arr = np.array(all_measurements[name]["cont"])
        
        p16, p50, p84 = np.nanpercentile(flux_arr, [16, 50, 84])
        results[name + "_flux"] = p50
        results[name + "_flux_err"] = 0.5 * (p84 - p16)  # symmetric approx
        results[name + "_flux_lo"] = p50 - p16  # asymmetric lower
        results[name + "_flux_hi"] = p84 - p50  # asymmetric upper
        
        p16, p50, p84 = np.nanpercentile(cont_arr, [16, 50, 84])
        results[name + "_cont"] = p50
        results[name + "_cont_err"] = 0.5 * (p84 - p16)
    
    return results


def compute_line_ratios_bootstrap(wave, all_stacks, line_windows):
    """
    Compute line ratios directly on each bootstrap realization.
    This properly captures covariances between lines.
    
    Returns
    -------
    results : dict
        Each ratio with median, symmetric error, and asymmetric bounds
    """
    n_boot = all_stacks.shape[0]
    
    # Arrays to store ratios for each bootstrap
    halpha_ew = []
    halpha_flux = []
    hbeta_flux = []
    ha_hb = []
    log_oiii_oii = []
    log_oiii_hb = []
    log_sii_halpha = []
    
    for i in range(n_boot):
        meas = measure_all_lines_boxcar(wave, all_stacks[i], line_windows, plot=False)
        
        # Compute ratios for this realization
        halpha_ew.append(meas["Halpha_flux"] / meas["Halpha_cont"])
        halpha_flux.append(meas["Halpha_flux"])

        hbeta_flux.append(meas["Hbeta_flux"])
        
        
        ha_hb.append(meas["Halpha_flux"] / meas["Hbeta_flux"])
        
        oiii_oii = meas["OIII_5007_flux"] / meas["OII_flux"]
        log_oiii_oii.append(np.log10(oiii_oii) if oiii_oii > 0 else np.nan)
        
        oiii_hb = meas["OIII_5007_flux"] / meas["Hbeta_flux"]
        log_oiii_hb.append(np.log10(oiii_hb) if oiii_hb > 0 else np.nan)
        
        sii_ha = meas["SII_flux"] / meas["Halpha_flux"]
        log_sii_halpha.append(np.log10(sii_ha) if sii_ha > 0 else np.nan)
    
    # Compute percentiles for each ratio
    def get_stats(arr):
        arr = np.array(arr)
        p16, p50, p84 = np.nanpercentile(arr, [16, 50, 84])
        return {
            "val": p50,
            "err": 0.5 * (p84 - p16),
            "err_lo": p50 - p16,
            "err_hi": p84 - p50
        }
    
    return {
        "halpha_flux": get_stats(halpha_flux),
        "hbeta_flux": get_stats(hbeta_flux),
        "halpha_ew": get_stats(halpha_ew),
        "ha_hb": get_stats(ha_hb),
        "log_oiii_oii": get_stats(log_oiii_oii),
        "log_oiii_hb": get_stats(log_oiii_hb),
        "log_sii_halpha": get_stats(log_sii_halpha),
    }



from astropy.io import fits

def write_stacked_spectra(
    outfile,
    wave,
    flux,
    ivar,
    resolution=None,
    stackids=None,
    stack_redshift=None,
    table_column_dict={},
    table_format_dict={},
):
    """
    Save stacked spectra to a FITS file compatible with FastSpecFit stackfit.
    Follows the same format as desigal.specutils.stack.write_binned_stacks.
    """
    from astropy.io import fits

    flux = np.atleast_2d(flux)
    ivar = np.atleast_2d(ivar)
    nobj, _ = flux.shape

    if stackids is None:
        stackids = np.arange(nobj)
    if stack_redshift is None:
        stack_redshift = np.zeros(nobj)
    if np.isscalar(stack_redshift):
        stack_redshift = np.full(nobj, stack_redshift)

    hdulist = []

    hdr = fits.Header()
    hdr["COMMENT"] = "Stacked spectra for FastSpecFit stackfit"
    empty_primary = fits.PrimaryHDU(header=hdr)
    hdulist.append(empty_primary)

    hduflux = fits.ImageHDU(flux.astype("f4"))
    hduflux.header["EXTNAME"] = "FLUX"
    hdulist.append(hduflux)

    hduivar = fits.ImageHDU(ivar.astype("f4"))
    hduivar.header["EXTNAME"] = "IVAR"
    hdulist.append(hduivar)

    hduwave = fits.ImageHDU(wave.astype("f8"))
    hduwave.header["EXTNAME"] = "WAVE"
    hduwave.header["BUNIT"] = "Angstrom"
    hduwave.header["AIRORVAC"] = ("vac", "vacuum wavelengths")
    hdulist.append(hduwave)

    if resolution is not None and not np.all(resolution == None):
        hdures = fits.ImageHDU(resolution.astype("f4"))
        hdures.header["EXTNAME"] = "RES"
        hdulist.append(hdures)

    c1 = fits.Column(name="STACKID", array=stackids, format="K")
    c2 = fits.Column(name="Z", array=stack_redshift, format="D")
    columns = [c1, c2]
    for key in table_column_dict.keys():
        if table_format_dict[key][0] == "P":
            columns.append(
                fits.Column(
                    name=key,
                    array=np.array(table_column_dict[key], dtype="object"),
                    format=table_format_dict[key],
                )
            )
        else:
            columns.append(
                fits.Column(
                    name=key,
                    array=table_column_dict[key],
                    format=table_format_dict[key],
                )
            )

    hdutable = fits.BinTableHDU.from_columns(columns)
    hdutable.header["EXTNAME"] = "STACKINFO"
    hdulist.append(hdutable)

    hx = fits.HDUList(hdulist)
    hx.writeto(outfile, overwrite=True, checksum=True)
    print(f"Saved {nobj} stacked spectra to {outfile}")
    


# def bootstrap_stack_fast_capped(fluxes, ivars, line_fluxes, wave,
#                                 n_bootstrap=200,
#                                 n_draw=5000,
#                                 random_seed=42):

#     rng = np.random.default_rng(random_seed)

#     # Normalize once
#     norm_fluxes, norm_ivars, valid = normalize_by_line_catalog(
#         fluxes, ivars, line_fluxes
#     )

#     use_fluxes = norm_fluxes[valid].astype(np.float32, copy=False)
#     n_valid, n_wave = use_fluxes.shape

#     if n_valid < 25:
#         return np.full(n_wave, np.nan), np.full(n_wave, np.nan), None

#     n_draw = min(n_draw, n_valid)

#     # Bootstrap indices: (n_bootstrap, n_draw)
#     indices = rng.integers(0, n_valid, size=(n_bootstrap, n_draw))

#     # Resample: (n_bootstrap, n_draw, n_wave)
#     boot_fluxes = use_fluxes[indices]

#     # Stack: mean over spectra axis
#     all_stacks = np.nanmean(boot_fluxes, axis=1, dtype=np.float32)

#     stacked_flux = np.nanmean(all_stacks, axis=0)
#     stacked_error = np.nanstd(all_stacks, axis=0)

#     return stacked_flux, stacked_error, all_stacks

    

'''
Script contains functions to construct the SHRED_MASKBITS for the photometry outputs
'''

import os
import sys

import numpy as np
from scipy.ndimage import uniform_filter1d, median_filter
from desi_lowz_funcs import save_table, get_useful_cat_colms, _n_or_more_gt, _n_or_more_lt, get_remove_flag
from easyquery import Query, QueryMaker

# ``sfr_and_metallicity`` lives in the flat ``code/nebular_stuff/`` folder (no
# __init__.py). Make it importable so this module works regardless of which
# script imports it (e.g. ``consolidate_photometry.py``).
_NEBULAR_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "nebular_stuff")
if _NEBULAR_DIR not in sys.path:
    sys.path.insert(0, _NEBULAR_DIR)

#####
#####
# THE BITMASK BOOL FUNCTIONS
#####
#####




def cog_nan_mask(cat,verbose=True):
    '''
    Function that contructs mask for objects where the fiducial COG mags are nan. MASKBIT = 0
    '''
    nan_mask = np.isnan(cat["COG_MAG_G_FINAL"].data) | np.isnan(cat["COG_MAG_R_FINAL"].data) | np.isnan(cat["COG_MAG_Z_FINAL"].data)
    
    # nan_mask_1 = np.isnan(cat["COG_MAG_G_ISOLATE"].data) | np.isnan(cat["COG_MAG_R_ISOLATE"].data) | np.isnan(cat["COG_MAG_Z_ISOLATE"].data)
    # nan_mask_2 = np.isnan(cat["COG_MAG_G_NO_ISOLATE"].data) | np.isnan(cat["COG_MAG_R_NO_ISOLATE"].data) | np.isnan(cat["COG_MAG_Z_NO_ISOLATE"].data)

    # nan_mask = nan_mask_1 | nan_mask_2

    if verbose:
        frac = np.sum(nan_mask)/len(nan_mask)
        print(f"MASKBIT=2^0, cog nan mask, {frac:.4f}",  )
    
    return nan_mask


def cog_mag_converge(catalog, mag_cut=0.5, verbose=True):
    """
    Check if fiducial COG magnitudes converge relative to R425 aperture mags.
    Flags objects where (APER - COG) > mag_cut in any band.

    These are the mags for which we will rever to the tractor only based mags. After we apply all the cleaning cuts!
    However, there are few cases where this is robust. We will just flag these as suspicious objects. But include them in the catalog

    MASKBIT = 1
    
    Parameters
    ----------
    catalog : astropy.table.Table
        Input catalog with COG_MAG_*_FINAL and APER_R425_MAG_*_FINAL columns.
    mag_cut : float, optional
        Threshold for suspicious magnitude difference.
    verbose : bool, optional
        If True, print summary statistics.

    Returns
    -------
    bad_mask : np.ndarray (bool)
        Boolean mask for suspicious objects.
    band_diffs : dict
        Per-band magnitude differences.
    """
    bands = ["G", "R", "Z"]
    band_diffs = {}

    for b in bands:
        cog = catalog[f"COG_MAG_{b}_FINAL"].data
        aper = catalog[f"APER_R4_MAG_{b}_FINAL"].data
        band_diffs[b] = aper - cog

    # stack differences into (N, nbands) array
    diffs = np.column_stack(list(band_diffs.values()))
    bad_mask = np.any(diffs > mag_cut, axis=1)

    if verbose:
        frac = bad_mask.sum() / len(bad_mask)
        print(f"MASKBIT=2^1, cog not converge, fraction: {frac:4f}")
        
    return bad_mask


def bad_cog_resid(cat,chi2_cut = 0.5, verbose=True):
    '''
    Function where the empirical fit to the COG curve is not good. MASKBIT = 2

    Note: some of these bad resid values are objects where te tractor model being subtracted elsewhere is over subtracting .. 
    
    '''
    all_chi2 = cat["COG_CHI2_FINAL"].data

    max_chi2 = np.max(all_chi2, axis = 1)

    chi2_mask = max_chi2 > chi2_cut
   
    if verbose:        
        print(f"MASKBIT=2^2, bad resid, fraction : {np.sum(chi2_mask)/len(chi2_mask):.4f}")
    
    return chi2_mask


def cog_curve_decrease(cat, mag_lim=0.2, len_lim=4, verbose=True):
    """
    Identify objects whose curve-of-growth decreases significantly.
    Flags cases where decrease in magnitude exceeds `mag_lim`
    and the length of the decrease is >= `len_lim`.

    MASKBIT = 3
    
    Returns
    -------
    tot_bad_mask : np.ndarray (bool)
        Boolean mask (True = suspicious object).
    band_bad_masks : dict
        Per-band bad masks (e.g., {"g": mask_g, "r": mask_r, "z": mask_z}).
    """

    all_decrease_mag = cat["COG_DECREASE_MAX_MAG_FINAL"].data   # shape (N, 3)
    all_decrease_len = cat["COG_DECREASE_MAX_LEN_FINAL"].data   # shape (N, 3)

    bands = ["g", "r", "z"]
    band_bad_masks = {}

    for i, band in enumerate(bands):
        mag = all_decrease_mag[:, i]
        length = all_decrease_len[:, i]
        mask = (mag > mag_lim) & (length >= len_lim)
        band_bad_masks[band] = mask
        # if verbose:
        # print(f"{band}-band suspicious objects: {mask.sum()} / {len(mask)}")

    tot_bad_mask = np.any(np.column_stack(list(band_bad_masks.values())), axis=1)

    # all_bad_mask = []
    # for flag in ["_ISOLATE", "_NO_ISOLATE"]:
    #     all_decrease_mag = cat[f"COG_DECREASE_MAX_MAG{flag}"].data   # shape (N, 3)
    #     all_decrease_len = cat[f"COG_DECREASE_MAX_LEN{flag}"].data   # shape (N, 3)
    #     bands = ["g", "r", "z"]
    #     band_bad_masks = {}
    #     for i, band in enumerate(bands):
    #         mag = all_decrease_mag[:, i]
    #         length = all_decrease_len[:, i]
    #         mask = (mag > mag_lim) & (length >= len_lim)
    #         band_bad_masks[band] = mask
    #         # if verbose:
    #         # print(f"{band}-band suspicious objects: {mask.sum()} / {len(mask)}")
    #     # combine across bands
    #     tot_bad_mask_i = np.any(np.column_stack(list(band_bad_masks.values())), axis=1)
    #     all_bad_mask.append(tot_bad_mask_i)
    # tot_bad_mask = all_bad_mask[0] | all_bad_mask[1]

    if verbose:
        print(f"MASKBIT=2^3, cog curve decrease, fraction: {tot_bad_mask.sum() / len(tot_bad_mask):.4f}")

    return tot_bad_mask


def cog_fracin_image(catalog,frac_cut = 0.75,verbose=True):
    '''
    This identifies sources where the final parent aperture extends significantly beyond the image cutout! MASKBIT = 4
    '''
    fracin_image = catalog["APERFRAC_R4_IN_IMG_FINAL"].data
    bad_mask = fracin_image < frac_cut

    bad_frac = np.sum(bad_mask)/len(bad_mask)

    if verbose:
        print(f"MASKBIT=2^4, aperfrac-in image, fraction: {bad_frac:4f}")

    return bad_mask



def cog_frac_mask_image(catalog,frac_cut = 1/3,verbose=True):
    '''
    This identifies sources where the final parent aperture has significant fraction of pixels masked! MASKBIT = 5
    '''
    fracin_image = catalog["APER_R4_MASK_FRAC_FINAL"].data
    bad_mask = fracin_image > frac_cut

    bad_frac = np.sum(bad_mask)/len(bad_mask)
    
    if verbose:
        print(f"MASKBIT=2^5, aperfrac-mask image, fraction: {bad_frac:4f}")

    return bad_mask


def image_mask_frac(catalog, frac_cut = 1/3,verbose=True):
    '''
    Fraction of pixels masked in image cutout. MASKBIT = 6
    '''

    img_frac_mask = catalog["IMAGE_MASK_PIX_FRAC"].data
    bad_mask = (img_frac_mask > frac_cut)
    
    bad_frac = np.sum(bad_mask)/len(bad_mask)
    if verbose:
        print(f"MASKBIT=2^6, image-frac mask, fraction: {bad_frac:4f}")

    return bad_mask


def bad_colors(catalog, col_cut = 2,verbose=True, what_mag = "_BEST"):
    '''
    With the best photometry (e.g., tractor, simple) we get extreme colors
    '''
    gr_colors = np.abs(catalog[f"MAG_G{what_mag}"].data - catalog[f"MAG_R{what_mag}"].data)
    rz_colors = np.abs(catalog[f"MAG_R{what_mag}"].data - catalog[f"MAG_Z{what_mag}"].data)
    bad_mask = (gr_colors > 2) | (rz_colors > 2)

    bad_frac = np.sum(bad_mask)/len(bad_mask)

    if verbose:
        print(f"MASKBIT=2^7, bad colors, fraction: {bad_frac:4f}")
    
    return bad_mask


def source_not_on_segment_mask(cat, verbose=True):
    """
    Check if a DESI source lies on the original segmented blob.

    Parameters
    ----------
    cat : Table or dict-like
        Catalog containing the column 'APER_SOURCE_ON_ORG_BLOB'.
    verbose : bool, optional
        Whether to print diagnostic information (default: True).

    Returns
    -------
    not_on_seg : np.ndarray (bool)
        Boolean mask where True indicates sources *not* on the original blob.
    """
    on_seg = np.asarray(cat["APER_SOURCE_ON_ORG_BLOB"].data, dtype=bool)
    not_on_seg = ~on_seg

    if verbose:
        frac = np.mean(not_on_seg)
        print(f"MASKBIT=2^8, source not on segment, fraction: {frac:.4f}")

    return not_on_seg


def very_near_bstar(catalog, radius_cut = 1,verbose=True):
    '''
    Sources that are very close to a bright star. Like within 0.5 times the star masking radius
    We use both the kinds of normalizd distance we do
    '''

    near_star = (catalog["STARFDIST"].data < radius_cut) #| (catalog["NEAREST_STAR_NORM_DIST"].data < radius_cut)

    #we do not want to just remove all sources that are close to stars, only sources that are likely shreds and quite close to stars as
    #their tractor models get iffy. so the below criterion is aimed at finding shredded sources close to stars
    # likely_not_just_blend = (catalog["NUM_TRACTOR_SOURCES_FINAL"] > 1) | (catalog["PCNN_FRAGMENT"] > 0.5)

    bad_mask = near_star #& likely_not_just_blend
    
    bad_frac = np.sum(bad_mask)/len(bad_mask)

    if verbose:
        print(f"MASKBIT=2^9, within star mask radius and not just a simple blend, fraction: {bad_frac:4f}")

    return bad_mask


def aper_cen_masked(cat,verbose=True):
    '''
    Sources where the aperture center is on a masked pixel is masked!! 
    What will happen to do this when the we do the light-weighted mask and no geometrical mask?
    '''

    col = cat["APER_CEN_MASKED_FINAL"]
    
    if hasattr(col, "mask"):
        raise ValueError("This column is a masked column type. Be careful of bugs!!!")

    bad_mask = np.asarray(col, dtype=bool)

    bad_frac = np.sum(bad_mask)/len(bad_mask)
    if verbose:
        print(f"MASKBIT=2^10, aper-cen masked, fraction: {bad_frac:4f}")
    
    return bad_mask


def no_seg_found(cat, verbose=True):
    '''
    This is if a source is likely shred, and no smooth segment is found. It could be suspcisious ... it could also be fain
    '''

    bad_mask = (cat["COG_NUM_SEG_SMOOTH"]==0) & (cat["COG_NUM_SEG"]==0)

    bad_frac = np.sum(bad_mask)/len(bad_mask)
    if verbose:
        print(f"MASKBIT=2^11, no seg found, fraction: {bad_frac:4f}")
    
    return bad_mask


def near_sga_outskirts(cat, norm_dist=2, verbose=True):
    """
    Flag sources that are near the outskirts of an SGA galaxy (1 < norm_dist < 2)
    but are NOT MASKBITS bit 13, that is, known association with SGA
    """
    # 1. Identify sources in the outskirts. The lower bound is to remove the small number of sources that are on SGA galaxy, but just for some reason MASKBIT not flagged
    in_outskirts = (cat["SGA_D26_NORM_DIST"] > 1) & (cat["SGA_D26_NORM_DIST"] < norm_dist)

    # 2. Exclude sources that have bit 12 set in MASKBITS
    # bit 12 corresponds to value 2**12 = 4096
    maskbit_12_flagged = (cat["MASKBITS"] & (1 << 12)) != 0

    # 3. Combine conditions
    bad_mask = in_outskirts & (~maskbit_12_flagged)

    if verbose:
        bad_frac = np.sum(bad_mask)/len(bad_mask) 
        print(f"MASKBIT=2^12, near sga outskirts, fraction: {bad_frac:4f}")

    return bad_mask
    
def iffy_tractor_model(cat, rchi_cut = 10, verbose=True):
    '''
    If the SNR on photometry is < 5 in all bands, or rchisq bad or something ... 
    '''

    bad_mask = (cat["RCHISQ_G"] > rchi_cut) | (cat["RCHISQ_R"] > rchi_cut) | (cat["RCHISQ_Z"] > rchi_cut)

    if verbose:
        bad_frac = np.sum(bad_mask)/len(bad_mask) 
        print(f"MASKBIT=2^13, bad rchisq, fraction: {bad_frac:4f}")

    return bad_mask



def low_SNR(cat, sigma_cut=5, nbands=2, verbose=True):
    """
    Flag sources that have low SNR. Require 5 sigma detection in at least two bands! Some of these low snr events tend to be faint emission in outskirts of massive galaxies
    """
    sigma_grz = [f"SIGMA_GOOD_{b}" for b in "GRZ"]
    sigma_queries = [Query(_n_or_more_gt(sigma_grz, nbands, sigma_cut)) ]
    # note that the this is n_or_more_LT!! so be careful about that!
    #these are masks for objects that did not satisfy the above condition!
    bad_snr_mask = get_remove_flag(cat, sigma_queries) == 0

    if verbose:
        bad_frac = np.sum(bad_snr_mask)/len(bad_snr_mask) 
        print(f"MASKBIT=2^14, low snr, fraction: {bad_frac:4f}")

    return bad_snr_mask


def other_tractor_maskbits(cat,verbose=True):
    '''
    Note that we already remove sources with 1,5,6,7,13 from the original catalog.
    Flag additional sources where Tractor MASKBITS has bit 8 and/or 9 set.
    '''
    
    maskbits_to_flag = [8, 9]  # bits to check
    maskbits_values = [2**b for b in maskbits_to_flag]

    tractor_maskbits = cat["MASKBITS"].data

    # Create boolean mask where any of those bits are set
    tractor_flag_mask = np.zeros(len(tractor_maskbits), dtype=bool)
    for val in maskbits_values:
        tractor_flag_mask |= (tractor_maskbits & val) != 0

    if verbose:
        bad_frac = np.mean(tractor_flag_mask)
        print(f"MASKBIT=2^15, Tractor maskbit flagged: {bad_frac:.4f}")

    return tractor_flag_mask

    
### 
#GENERAL MASKBIT FUNCTIONS
###


bitmask_dict = {
    0: {"value": 1 << 0, "description": "cog nan", "func": cog_nan_mask },
    1: {"value": 1 << 1, "description": "cog not converge", "func": cog_mag_converge },
    2: {"value": 1 << 2, "description": "cog bad residual", "func":bad_cog_resid },
    3: {"value": 1 << 3, "description": "cog curve decrease", "func": cog_curve_decrease },
    4: {"value": 1 << 4, "description": "cog aperfrac in image", "func": cog_fracin_image},
    5: {"value": 1 << 5, "description": "cog aperfrac mask", "func": cog_frac_mask_image},
    6: {"value": 1 << 6, "description": "image frac mask", "func": image_mask_frac } ,
    7: {"value": 1 << 7, "description": "bad gr/rz color", "func": bad_colors },
    8: {"value": 1 << 8, "description": "source not on segment", "func": source_not_on_segment_mask },
    9: {"value": 1 << 9, "description": "shredded and near bstar", "func": very_near_bstar },
    10: {"value": 1 << 10, "description": "cop aper center masked", "func": aper_cen_masked },
    11: {"value": 1 << 11, "description": "no seg found", "func": no_seg_found },
    12: {"value": 1 << 12, "description": "near SGA outskirts", "func": near_sga_outskirts},
    13: {"value": 1 << 13, "description": "org tractor, bad rchisq", "func": iffy_tractor_model},
    14: {"value": 1 << 14, "description": "low sigma detection", "func": low_SNR},
    15: {"value": 1 << 15, "description": "tractor maskbits", "func": other_tractor_maskbits}   
}

#note that 13,14,15 maskbits are only if source has mag_type = tractor_og

def create_shred_maskbits_from_dict(cat, bitmasks_to_apply = [0,1,2,3,4,5,6,7,8,9,10,12], verbose=False, mag_type = "_BEST"):
    """
    Create maskbit values using bitmask_dict entries that include 'func'.
    """
    import numpy as np

    n = len(cat)
    maskbits = np.zeros(n, dtype=np.int32)


    for bit_num in bitmasks_to_apply:
        info = bitmask_dict[bit_num]
        func = info.get("func", None)
        if func is None:
            print(f"Skipping bit {bit_num}: no function assigned ({info['description']})")
            continue

        # Call the function to get a boolean mask
        if bit_num == 7:
            cond = func(cat, what_mag = mag_type, verbose=verbose)
        else:
            cond = func(cat, verbose=verbose)
            
        if not isinstance(cond, (np.ndarray, list)):
            raise ValueError(f"Function for bit {bit_num} did not return a boolean array!")

        maskbits[cond] |= info["value"]  # bitwise OR to set bits

    return maskbits



def print_maskbit_statistics(maskbit_col, bitmasks_to_use = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]):
    """
    Print statistics on what fraction of sources have each maskbit (0..n_bits-1) set.

    Parameters
    ----------
    maskbit_col : array-like (e.g., np.ndarray or astropy column)
        Integer maskbit values for all sources.
    n_bits : int, optional
        Number of bits to check (default = 11 → bits 0 through 10).

    Returns
    -------
    None
        Prints formatted summary of fraction and count for each bit.
    """


    maskbit_col = np.asarray(maskbit_col, dtype=np.int64)
    n_total = len(maskbit_col)

    print(f"\n--- Maskbit Statistics (n = {n_total}) ---")

    print(f"Fraction with no maskbit on = { np.sum(maskbit_col == 0)/len(maskbit_col) }")

    for bit in bitmasks_to_use:
        bit_value = 1 << bit
        bit_on = (maskbit_col & bit_value) != 0
        n_on = np.count_nonzero(bit_on)
        frac_on = n_on / n_total if n_total > 0 else 0

        print(f"Bit {bit:2d} (2^{bit:<2d} = {bit_value:4d})  ->  {frac_on:.2%} fraction")

    return


    
##### OTHER MASKBITS

def inspect_anomalies(spec_cat, umap_x_cen, umap_y_cen, radius_cut = 0.1):
    rads = np.sqrt( (spec_cat["SPEC_UMAP_0"].data - umap_x_cen)**2 + (spec_cat["SPEC_UMAP_1"].data - umap_y_cen)**2 )

    mask = (rads < radius_cut)

    print(f"Objects found = {np.sum(mask)}")
    
    return mask

from sfr_and_metallicity import line_snr_mask

def flag_weird_spectra(spec_cat, main_cat, fspec_cat):
    '''
    Function that constructs maskbits for weird spectra, likely wrong redrock redshifts
    '''

    mask_1 = inspect_anomalies(spec_cat, 9,7, radius_cut = 0.5)

    mask_2 = inspect_anomalies(spec_cat, 8.5,8.5, radius_cut = 0.2)
    
    mask_3 = inspect_anomalies(spec_cat,8.125,8.75, radius_cut = 0.1)
    
    mask_4 = inspect_anomalies(spec_cat,6.5,7.5, radius_cut = 0.5)

    mask_6 = inspect_anomalies(spec_cat,3.5,13.5, radius_cut = 0.5)
    
    #get a mask of objects where we have fairly confident emission lines! At least three well detected lines!
    good_line_mask = line_snr_mask(fspec_cat,line_names=["HALPHA","HBETA","OIII_5007","OIII_4959", "OII_3726", "OII_3729","SII_6716"], min_lines=3)
    
    weird_mask = (mask_1 | mask_2 | mask_3 | mask_4 | mask_6) & (~good_line_mask)

    print(f"Total weird identified = {np.sum(weird_mask)}")

    return weird_mask


def smooth_spectrum(wave, flux, ivar=None, mask=None, method="median", boxcar_A=80.0):
    """
    Smooth a 1D spectrum on a (possibly masked) pixel grid.

    Parameters
    ----------
    wave : array, shape (n,)
        Wavelength in Angstrom, increasing.
    flux : array, shape (n,)
        Flux in coadd units (typically 1e-17 erg/s/cm2/A).
    ivar : array, optional
        Inverse variance. Pixels with ivar <= 0 are ignored.
    mask : array, optional
        Bad-pixel mask. Nonzero = bad (DESI SPECMASK convention).
    method : {"median", "boxcar", "ivar_boxcar"}
    boxcar_A : float
        Smoothing scale in Angstrom.

    Returns
    -------
    flux_smooth : array
        Smoothed flux, same shape as input. Bad/input NaN pixels -> NaN.
    good : bool array
        Pixels used as valid input.
    """
    wave = np.asarray(wave, dtype=float)
    flux = np.asarray(flux, dtype=float)

    good = np.isfinite(flux)
    if ivar is not None:
        good &= np.asarray(ivar) > 0
    if mask is not None:
        good &= np.asarray(mask) == 0

    flux_out = np.full_like(flux, np.nan, dtype=float)
    if good.sum() < 5:
        return flux_out, good

    # Convert smoothing width in Angstrom to pixels (median spacing)
    dw = np.median(np.diff(wave[good]))
    if not np.isfinite(dw) or dw <= 0:
        return flux_out, good
    halfwin = max(1, int(round(0.5 * boxcar_A / dw)))

    if method == "median":
        # Fill bad pixels with NaN, median_filter ignores NaN in recent scipy
        tmp = flux.astype(float).copy()
        tmp[~good] = np.nan
        flux_smooth = median_filter(tmp, size=2 * halfwin + 1, mode="nearest")
        flux_out[good] = flux_smooth[good]

    elif method in ("boxcar", "ivar_boxcar"):
        tmp = flux.astype(float).copy()
        tmp[~good] = 0.0
        num = uniform_filter1d(tmp, size=2 * halfwin + 1, mode="nearest")
        den = uniform_filter1d(good.astype(float), size=2 * halfwin + 1, mode="nearest")

        if method == "ivar_boxcar" and ivar is not None:
            w = np.zeros_like(flux)
            w[good] = np.asarray(ivar)[good]
            num = uniform_filter1d(flux * w, size=2 * halfwin + 1, mode="nearest")
            den = uniform_filter1d(w, size=2 * halfwin + 1, mode="nearest")

        with np.errstate(invalid="ignore", divide="ignore"):
            flux_smooth = num / den
        flux_out[good] = flux_smooth[good]

    else:
        raise ValueError(f"Unknown method={method}")

    return flux_out, good


def spectrum_negative_continuum_stats(wave, flux, ivar=None, boxcar_A=200.0,
                                      neg_thresh=-5.0, wave_max=4000.0):
    """
    Median-smooth one spectrum on a ``boxcar_A`` Angstrom scale and summarize how
    negative the smoothed *blue* continuum is. These statistics feed
    DWARF_MASKBIT bit 20 (suspect spectrum): an overwhelmingly negative blue
    continuum is a signature of a sky-subtraction / normalization failure.

    Only finite smoothed pixels with observed wavelength < ``wave_max`` are
    considered. ``ivar`` is passed through to :func:`smooth_spectrum`, so pixels
    with ivar <= 0 are gated out of the smoothing; because a median over the
    ``boxcar_A``-wide window is robust to a minority of gated pixels, sparse
    sky-line masks leave the continuum intact while wide contiguous chip gaps go
    NaN locally and are simply excluded from the statistics below.

    The flagging decision (e.g. ``frac_neg_blue >= 0.75`` with
    ``n_finite_blue >= 20``) is applied by the caller, so these raw statistics can
    be cached once and re-thresholded later without re-smoothing.

    Parameters
    ----------
    wave, flux : array, shape (n,)
        Observed wavelength (Angstrom, increasing) and flux (1e-17 cgs).
    ivar : array, optional
        Inverse variance; pixels with ivar <= 0 are gated out.
    boxcar_A : float
        Median smoothing scale in Angstrom.
    neg_thresh : float
        A smoothed pixel counts as "negative" if it is below this value.
    wave_max : float
        Only pixels with observed wave < wave_max contribute (blue end).

    Returns
    -------
    n_finite_blue : int
        Number of finite smoothed pixels with wave < wave_max.
    frac_neg_blue : float
        Fraction of those finite blue pixels whose smoothed flux is < neg_thresh.
        NaN if n_finite_blue == 0.
    """
    smooth_spec, _ = smooth_spectrum(
        wave, flux, ivar=ivar, mask=None, method="median", boxcar_A=boxcar_A
    )
    blue = np.asarray(wave, dtype=float) < wave_max
    vals = smooth_spec[blue]
    finite = np.isfinite(vals)
    n_finite_blue = int(np.sum(finite))
    if n_finite_blue == 0:
        return 0, float("nan")
    frac_neg_blue = float(np.sum(vals[finite] < neg_thresh) / n_finite_blue)
    return n_finite_blue, frac_neg_blue


    
    

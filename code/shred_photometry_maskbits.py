'''
Script contains functions to construct the SHRED_MASKBITS for the photometry outputs
'''

import numpy as np
from desi_lowz_funcs import save_table, get_useful_cat_colms, _n_or_more_gt, _n_or_more_lt, get_remove_flag
from easyquery import Query, QueryMaker

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

    if verbose:
        frac = np.sum(nan_mask.data)/len(nan_mask)
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

    # combine across bands
    tot_bad_mask = np.any(np.column_stack(list(band_bad_masks.values())), axis=1)

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

    bad_mask = cat["APER_CEN_MASKED_FINAL"].data

    bad_frac = np.sum(bad_mask)/len(bad_mask)
    if verbose:
        print(f"MASKBIT=2^10, aper-cen masked, fraction: {bad_frac:4f}")
    
    return bad_mask


def iffy_tractor_model(cat, rchi_cut = 10, verbose=True):
    '''
    If the SNR on photometry is < 5 in all bands, or rchisq bad or something ... 
    '''

    bad_mask = (cat["RCHISQ_G"] > rchi_cut) | (cat["RCHISQ_R"] > rchi_cut) | (cat["RCHISQ_Z"] > rchi_cut)

    if verbose:
        bad_frac = np.sum(bad_mask)/len(bad_mask) 
        print(f"MASKBIT=2^11, bad rchisq, fraction: {bad_frac:4f}")

    return bad_mask


def near_sga_outskirts(cat, norm_dist=2, verbose=True):
    """
    Flag sources that are near the outskirts of an SGA galaxy (1 < norm_dist < 2)
    but are NOT MASKBITS bit 12, that is, known association with SGA
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
        print(f"MASKBIT=2^13, low snr, fraction: {bad_frac:4f}")

    return bad_snr_mask


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
    11: {"value": 1 << 11, "description": "org tractor, bad rchisq", "func": iffy_tractor_model},
    12: {"value": 1 << 12, "description": "near SGA outskirts", "func": near_sga_outskirts},
    13: {"value": 1 << 13, "description": "low sigma detection", "func": low_SNR}
}

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



def print_maskbit_statistics(maskbit_col, bitmasks_to_use = [0,1,2,3,4,5,6,7,8,9,10,11,12,13]):
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


    
# # #mask sources where the nearest smooth blob final is not on the DESI fiber?: 39627462455334432



    
    

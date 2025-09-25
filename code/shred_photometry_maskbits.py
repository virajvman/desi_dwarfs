'''
Script contains functions to construct the SHRED_MASKBITS for the photometry outputs
'''

import numpy as np


def create_shred_maskbits(cat):
    '''
    Adds a 'shred_maskbits' column to the table `cat`, where each bit corresponds to a flag.
    
    Here are the different MASKBITS/flags we are going to construct

    0: Curve of growth fit failed (One of the COG params are nans)
    1: Poor fit( large residuals) in any one of the COGs
    2: Nearest normalized distance to a star < 0.75
    3: If r35 is brighter than rCOG by 0.25 mags
    4: Is there a continuous decrease in cog mag for 4 or more with a drop in 0.25 mags
    5: More than 25% of the R_4.25 aperture lies outisde the image
    6: More than 25% of the R_4.25 aperture is masked by bad pixels 
    7: More than 50% of image cutout is masked by bad pixels
    8: Extreme colors
    9: source does not lie on any segmentation island
    '''

    n = len(cat)
    maskbits = np.zeros(n, dtype=np.int32)

    # Each flag number maps to its bit value (power of 2)
    bit_value = {
        0: 1,        # 2^0
        1: 2,        # 2^1
        2: 4,        # 2^2
        3: 8,        # 2^3
        4: 16,       # 2^4
        5: 32,       # 2^5
        6: 64,       # 2^6
        7: 128,      # 2^7
        8: 256,      # 2^8
    }

    # Boolean arrays from your flag functions
    conditions = [
        cog_nan_mask(cat),
        cog_mag_converge(cat),
        bad_cog_resid(cat),
        cog_curve_decrease(cat),
        cog_fracin_image(cat),
        image_mask_frac(cat),
        bad_colors(cat),
        source_not_on_segment_mask(cat),
        very_near_bstar(cat)
    ]

        # large_frac_cog_aper_out(cat),
        # large_frac_aper_mask(cat),
        # large_frac_image_mask(cat),
        # bad_gr_colors(cat),
        # source_not_on_segment_mask(cat),

    # Add powers of 2 for each flag that is True
    for flag_num, cond in enumerate(conditions):
        maskbits[cond] += bit_value[flag_num]

    cat["PHOTO_MASKBIT"] = maskbits
    return cat



##here we finalize some of the cog_maskbits

def cog_nan_mask(cat,verbose=False):
    '''
    Function that contructs mask for objects where the fiducial COG mags are nan. MASKBIT = 0
    '''
    nan_mask = np.isnan(cat["COG_MAG_G_FINAL"].data) | np.isnan(cat["COG_MAG_G_FINAL"].data) | np.isnan(cat["COG_MAG_G_FINAL"].data)

    if verbose:
        print( np.sum(nan_mask.data)/len(nan_mask) )
    
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
        print(f"MASKBIT=1 fraction: {frac:4f}")
        
    return bad_mask

def bad_cog_resid(cat,chi2_cut = 0.5, verbose=False):
    '''
    Function where the empirical fit to the COG curve is not good. MASKBIT = 2

    Note: some of these bad resid values are objects where te tractor model being subtracted elsewhere is over subtracting .. 
    
    '''
    all_chi2 = cat["COG_CHI2_FINAL"].data

    max_chi2 = np.max(all_chi2, axis = 1)

    chi2_mask = max_chi2 > chi2_cut
   
    if verbose:        
        print(f"MASKBIT=2 fraction : {np.sum(chi2_mask)/len(chi2_mask):.4f}")
    
    return chi2_mask

def cog_curve_decrease(cat, mag_lim=0.2, len_lim=4, verbose=False):
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

    print(f"MASKBIT=3 fraction: {tot_bad_mask.sum() / len(tot_bad_mask):.4f}")

    return tot_bad_mask



def cog_fracin_image(catalog,frac_cut = 2/3):
    '''
    This identifies sources where the final parent aperture extends significantly beyond the image cutout! MASKBIT = 4
    '''
    fracin_image = catalog["APER_R4_FRAC_IN_IMG_FINAL"].data
    bad_mask = fracin_image < frac_cut

    bad_frac = np.sum(bad_mask)/len(bad_mask)
    print(f"MASKBIT=4 fraction: {bad_frac:4f}")

    return bad_mask


def cog_frac_mask_image(catalog,frac_cut = 1/3):
    '''
    This identifies sources where the final parent aperture has significant fraction of pixels masked! MASKBIT = 5
    '''
    fracin_image = catalog["APER_R4_MASK_FRAC_FINAL"].data
    bad_mask = fracin_image > frac_cut

    bad_frac = np.sum(bad_mask)/len(bad_mask)
    print(f"MASKBIT=5 fraction: {bad_frac:4f}")

    return bad_mask


def image_mask_frac(catalog, frac_cut = 1/3):
    '''
    Fraction of pixels masked in image cutout. MASKBIT = 6
    '''

    img_frac_mask = catalog["IMAGE_MASK_PIX_FRAC"].data
    bad_mask = (img_frac_mask > frac_cut)
    
    bad_frac = np.sum(bad_mask)/len(bad_mask)
    print(f"MASKBIT=6 fraction: {bad_frac:4f}")

    return bad_mask


def bad_colors(catalog, col_cut = 2):
    '''
    With the best photometry (e.g., tractor, simple) we get extreme colors
    '''
    gr_colors = np.abs(catalog["MAG_G_BEST"].data - catalog["MAG_R_BEST"].data)
    rz_colors = np.abs(catalog["MAG_R_BEST"].data - catalog["MAG_Z_BEST"].data)
    bad_mask = (gr_colors > 2) | (rz_colors > 2)

    bad_frac = np.sum(bad_mask)/len(bad_mask)
    print(f"MASKBIT=7 fraction: {bad_frac:4f}")
    
    return bad_mask


def source_not_on_segment_mask(cat, verbose=False):
    '''
    Was the DESI source on the original segmented blob? If not, it is still could be a real source, but we flag it as suspicious
    '''

    on_seg = cat["APER_SOURCE_ON_ORG_BLOB"].data
    not_on_seg = ~on_seg
    
    if verbose:
        print(f"MASKBIT=8 fraction: {np.sum(not_on_seg.data) / len(not_on_seg):4f}")
        
    return not_on_seg


def very_near_bstar(catalog, radius_cut = 0.5):
    '''
    Sources that are very close to a bright star. Like within 0.5 times the star masking radius
    We use both the kinds of normalizd distance we do
    '''

    bad_mask = (catalog["STARFDIST"].data < radius_cut) | (catalog["NEAREST_STAR_NORM_DIST"].data < radius_cut)

    bad_frac = np.sum(bad_mask)/len(bad_mask)
    print(f"MASKBIT=9 fraction: {bad_frac:4f}")

    return bad_mask


# #mask sources where the nearest smooth blob final is not on the DESI fiber?: 39627462455334432



    
    

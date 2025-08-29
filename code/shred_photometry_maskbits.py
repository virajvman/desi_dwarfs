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
        9: 512,      # 2^9
    }

    # Boolean arrays from your flag functions
    conditions = [
        cog_nan_mask(cat),
        bad_cog_chi2(cat),
        bstar_mask(cat),
        larger_r35(cat),
        cog_curve_decrease(cat),
        large_frac_cog_aper_out(cat),
        large_frac_aper_mask(cat),
        large_frac_image_mask(cat),
        bad_gr_colors(cat),
        source_not_on_segment_mask(cat),
    ]

    # Add powers of 2 for each flag that is True
    for flag_num, cond in enumerate(conditions):
        maskbits[cond] += bit_value[flag_num]

    cat["SHRED_MASKBIT"] = maskbits
    return cat


def cog_nan_mask(cat,verbose=False):
    '''
    Function that contructs mask for objects where the COG mags are nan
    '''
    nan_mask = np.isnan(cat["MAG_G_APERTURE_COG_ERR"].data) | np.isnan(cat["MAG_R_APERTURE_COG_ERR"].data) | np.isnan(cat["MAG_Z_APERTURE_COG_ERR"].data)

    if verbose:
        print( np.sum(nan_mask.data)/len(nan_mask) )
    
    return nan_mask.data
    
def bad_cog_chi2(cat,chi2_cut = 0.5, verbose=False):
    g_chi2 = cat["G_APERTURE_COG_CHI2"].data
    r_chi2 = cat["R_APERTURE_COG_CHI2"].data
    z_chi2 = cat["Z_APERTURE_COG_CHI2"].data

    chi2_mask = (g_chi2 > chi2_cut) |  (r_chi2 > chi2_cut) | (z_chi2 > chi2_cut)
   
    if verbose:
        # plt.figure(figsize = (4,4))
        # plt.hist(g_chi2,range=(0,5),bins=50,density=True)
        # plt.hist(r_chi2,range=(0,5),bins=50,density=True)
        # plt.hist(z_chi2,range=(0,5),bins=50,density=True)
        # plt.yscale("log")
        # plt.xlim([0,5])
        # plt.show()
        
        print( np.sum(chi2_mask.data)/len(chi2_mask) )
    
    return chi2_mask.data


def get_single_band_r35_mask(cat, band, mag_lim):
    cog_mag = cat[f"MAG_{band}_APERTURE_COG"].data
    r35_mag = cat[f"MAG_{band}_APERTURE_R375"].data
    nan_mask = np.isnan(cat[f"MAG_{band}_APERTURE_COG_ERR"].data)    
    #if the cog fit was not a nan, and r35 is brighter, somethign sus!
    mag_diff = cog_mag - r35_mag
    
    bad_mask = (nan_mask.data == False) & (mag_diff > mag_lim)
    
    return bad_mask

def larger_r35(cat,mag_lim = 0.25, verbose=False):
    gbad_mask = get_single_band_r35_mask(cat, "G", mag_lim)
    rbad_mask = get_single_band_r35_mask(cat, "R", mag_lim)
    zbad_mask = get_single_band_r35_mask(cat, "Z", mag_lim)

    tot_bad_mask = gbad_mask | rbad_mask | zbad_mask
    
    if verbose:        
        print( np.sum(tot_bad_mask.data)/len(tot_bad_mask) )
    
    return tot_bad_mask



def bstar_mask(cat, star_lim = 1, verbose=False):
    bstar_mask = (cat["STARFDIST"].data < star_lim) | (cat["NEAREST_STAR_NORM_DIST"] < star_lim) 

    if verbose:
        print( np.sum(bstar_mask.data)/len(bstar_mask) )
    
    return bstar_mask.data


def cog_curve_decrease(cat,mag_lim = 0.25, len_lim = 4, verbose=False):
    g_decrease_mag = cat["COG_DECREASE_MAX_MAG"][:,0].data
    r_decrease_mag = cat["COG_DECREASE_MAX_MAG"][:,1].data
    z_decrease_mag = cat["COG_DECREASE_MAX_MAG"][:,2].data

    g_decrease_len = cat["COG_DECREASE_MAX_LEN"][:,0].data
    r_decrease_len = cat["COG_DECREASE_MAX_LEN"][:,1].data
    z_decrease_len = cat["COG_DECREASE_MAX_LEN"][:,2].data

    g_bad_mask = (g_decrease_mag > mag_lim) & (g_decrease_len >= 4)
    r_bad_mask = (r_decrease_mag > mag_lim) & (r_decrease_len >= 4)
    z_bad_mask = (z_decrease_mag > mag_lim) & (z_decrease_len >= 4)

    tot_bad_mask = g_bad_mask | r_bad_mask | z_bad_mask


    if verbose:
        print( np.sum(tot_bad_mask.data)/len(tot_bad_mask) )
    
    return tot_bad_mask 

def bad_gr_colors(cat, max_col = 2,verbose=False):
    gmags = cat["MAG_G_APERTURE_COG"].data
    rmags = cat["MAG_R_APERTURE_COG"].data

    gr_abs = np.abs(gmags - rmags)

    bad_col = (gr_abs > max_col)

    if verbose:
        # plt.figure(figsize = (4,4))
        # plt.hist(gmags - rmags,range=(-4,4),bins=50,density=True,histtype = "step")
        # plt.yscale("log")
        # plt.xlim([-4,4])
        # plt.show()

        
        print( np.sum(bad_col.data) / len(bad_col) )
    
    return bad_col
    

def large_frac_image_mask(cat, max_frac= 0.5, verbose=False):
    '''
    Is a large fraction of the image masked by bright stars and saturated pixels?
    '''

    mask_frac = cat["IMAGE_MASK_PIX_FRAC"].data

    bad_mask = (mask_frac > max_frac)

    if verbose:
        # plt.figure(figsize = (4,4))
        # plt.hist(mask_frac,range=(0,1),bins=10,density=True,histtype = "step")
        # plt.yscale("log")
        # plt.xlim([0,1])
        # plt.show()

        print( np.sum(bad_mask.data) / len(bad_mask) )  

    return bad_mask


def large_frac_aper_mask(cat, max_frac= 0.5, verbose=False):
    '''
    Is a large fraction of the image masked by bright stars and saturated pixels?
    '''

    mask_frac = cat["APER_R35_MASK_PIX_FRAC"].data

    bad_mask = (mask_frac > max_frac)

    if verbose:
        # plt.figure(figsize = (4,4))
        # plt.hist(mask_frac,range=(0,1),bins=10,density=True,histtype = "step")
        # plt.yscale("log")
        # plt.xlim([0,1])
        # plt.show()

        print( np.sum(bad_mask.data) / len(bad_mask) )  

    return bad_mask


def large_frac_cog_aper_out(cat, min_frac= 0.25, verbose=False):
    '''
    Is a large fraction of the largest cog aperture outside the image?
    '''

    out_frac = cat["COG_MAXAPER_FRAC_IN"].data

    bad_mask = (out_frac < min_frac)

    if verbose:
        # plt.figure(figsize = (4,4))
        # plt.hist(out_frac,range=(0,1),bins=10,density=True,histtype = "step")
        # plt.yscale("log")
        # plt.xlim([0,1])
        # plt.show()

        print( np.sum(bad_mask.data) / len(bad_mask) )  

    return bad_mask

    
def source_not_on_segment_mask(cat, verbose=False):

    on_seg = cat["APER_SOURCE_ON_SEGMENT"].data
    not_on_seg = ~on_seg
    
    if verbose:
        print( np.sum(not_on_seg.data) / len(not_on_seg) )  
    
    return not_on_seg


def more_than_one_segment(segment_img):
    '''
    Does the final segmented image used to estimate the aperture of the galaxy have one segment or more than one segment in the smoothed image
    If there are more than one segment, then there is some potential issue ... and we should flag this.

    TODO: How does smoothing affect this?
    '''





    
    

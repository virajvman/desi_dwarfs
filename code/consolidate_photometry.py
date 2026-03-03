'''
Functions where preferred, best photometry column is created along with photometry maskbits to identify reliable measurements. This is also the script where we produce the final, multi-extension fits files as the final catalog output.
'''
import numpy as np
import astropy.io.fits as fits
import astropy.units as u
from astropy.table import Table, vstack, join, hstack
from shred_photometry_maskbits import cog_mag_converge, cog_nan_mask, cog_curve_decrease, bad_colors, iffy_tractor_model
from io import BytesIO
from shred_photometry_maskbits import create_shred_maskbits_from_dict, print_maskbit_statistics, flag_weird_spectra
import os
import shutil
import glob
from tqdm import trange
import h5py
from desi_lowz_funcs import get_stellar_mass, match_c_to_catalog, match_fastspec_catalog, get_stellar_mass_mia
from desi_lowz_funcs import add_sweeps_column
from desi_lowz_funcs import get_sga_norm_dists_FAST
from construct_dwarf_galaxy_catalogs import bright_star_filter

from get_associated_fibers import find_associated_tgids, get_dwarf_primary
from independent_distances import update_distance_catalog

def combine_arrays(no_iso, w_iso, mask):
    '''
    For each element: if mask == True, take the value from no_iso; otherwise, take the value from w_iso.
    '''
    if no_iso.ndim == 1:  # 1D case
        return np.where(mask, no_iso, w_iso)
    else:  # 2D or higher
        # Expand mask along all extra dims so it broadcasts
        # expanded_mask = mask[(...,) + (None,) * (no_iso.ndim - 1)]
        expanded_mask = np.expand_dims(mask, axis=tuple(range(1, no_iso.ndim)))
        return np.where(expanded_mask, no_iso, w_iso)

# def make_catalog_unmasked(cat):
#     """
#     Return a new Table where all MaskedColumns are replaced by regular ndarray columns.
#     Masked entries are filled with np.nan.
#     """
#     new_cat = cat.copy()
#     for col in new_cat.colnames:
#         c = new_cat[col]
#         if hasattr(c, "mask"):   # MaskedColumn
#             new_cat[col] = np.asarray(c.filled(np.nan))
#         else:
#             new_cat[col] = np.asarray(c)
#     return new_cat

def make_catalog_unmasked(cat):
    """
    Return a new Table where all MaskedColumns are replaced by regular ndarray columns.
    Masked entries are filled with appropriate default values.
    """
    new_cat = cat.copy()
    for col in new_cat.colnames:
        c = new_cat[col]
        if hasattr(c, "mask"):
            if np.issubdtype(c.dtype, np.floating):
                fill_val = np.nan
            elif np.issubdtype(c.dtype, np.integer):
                fill_val = -99  # or 0, or whatever sentinel makes sense for your catalog
            elif np.issubdtype(c.dtype, np.bool_):
                fill_val = False
            elif c.dtype.kind in ('U', 'S', 'O'):  # string/object
                fill_val = ""
            else:
                fill_val = 0
            new_cat[col] = np.asarray(c.filled(fill_val))
        else:
            new_cat[col] = np.asarray(c)
            
    return new_cat
    

def num_deblend_blob_boundary(zred):
    '''
    Function that defines the linear boundary in zred vs. r2_mur space for considering objects that are likely over-deblended
    '''

    # ax[0].plot([0.005,0.1],[22.5,28], color = "k",ls = "--")
    slope = (28 - 22.5) / (0.1 - 0.005)
    y_intp = 28 - slope*0.1

    bound_value = slope * zred + y_intp

    return bound_value


def likely_over_deblended(zred, r2_mur):
    """
    Returns boolean array (or scalar if inputs are scalars) indicating
    whether the source is in the over-deblended regime.

    For sources that satisfy this mask, we use the 
    """
    
    bound_value = num_deblend_blob_boundary(zred)
    
    # core condition
    likely = r2_mur > bound_value
    
    # apply the zred < 0.005 override
    likely = np.where(zred < 0.005, True, likely)
    
    return likely


def org_tractor_is_likely_good(cat,use_pcnn=True):
    '''
    Function that identifies the subset of sources tha are likely just pure blends where the original tractor model is all good!
    Oh these will also be sources where nothing is detected because num_tractor_sources_final gets only if the cog part is run

    In our fiducial run, we will not be using PCNN
    '''

    ntractor = np.array(cat["NUM_TRACTOR_SOURCES_FINAL"])
    
    #this cirterion will be good for the sources that have good significance
    #however, in addition to just being a single source, we also want to make sure the COG mag is not better as bad photometry

    if use_pcnn:
        likely_pure_blend = (np.array(cat["PCNN_FRAGMENT"]) < 0.25) | ( (ntractor <= 1) | np.isnan(ntractor) )
    else:
        likely_pure_blend = ( (ntractor <= 1) | np.isnan(ntractor) )
        
    #sometimes no sources are listed if no smooth component for parent galaxy isolate is found.
    #this we do <= 1 or np.nan
    
    return likely_pure_blend


def revert_back_to_org_tractor(cat,use_pcnn=True):
    '''
    Function that identifies the subset of sources tha are likely just pure blends where the original tractor model is all good!
    Oh these will also be sources where nothing is detected because num_tractor_sources_final gets only if the cog part is run
    '''

    likely_pure_blend = org_tractor_is_likely_good(cat,use_pcnn=use_pcnn)

    #if the source was soo faint that it was not on the original blob!
    cog_was_not_run = (cat["APER_SOURCE_ON_ORG_BLOB"] == 0)

    #for the very faint sources, we need to check too. if no cog segment was detected
    cog_seg_not_detected = (cat["COG_NUM_SEG_SMOOTH"] == 0) | (cat["COG_NUM_SEG"] == 0)

    return likely_pure_blend | cog_was_not_run | cog_seg_not_detected


def add_best_mags(catalog, bands=("G", "R", "Z"), use_pcnn=True):
    """
    Add MAG_[band]_BEST columns to the catalog by combining
    tractor, simple, and cog-based magnitudes according to preference masks.
    This is only for the sources whose photometry has been remeasured
    """

    #consolidate this into the best photometry!
    #some criterion for consolidation:
    #1) if COG_MAG_FINAL in any band is 0.5 mag or larger than its R4_FINAL mag, we revert to the tractor only based reconsutrction
    #2) if over-subtraction like either nans or consecutive decrease, we use the simplest photo    
    #3) NOT SURE: if the tractor based mag is much brighter than the COG based based mag, and there is no decrease, we will revert to the tractor based mag?

    prefer_tractor_based_mag = cog_mag_converge(catalog, verbose=False) 
    prefer_simple_mag = cog_nan_mask(catalog, verbose=False) | cog_curve_decrease(catalog, verbose=False)
    
    prefer_org_tractor_mag = revert_back_to_org_tractor(catalog, use_pcnn=use_pcnn)

    print("FRACTION REVERT BACK TO TRACTOR:", np.sum(prefer_org_tractor_mag)/len(prefer_org_tractor_mag))
    
    for b in bands:
        trac = np.array(catalog[f"TRACTOR_ONLY_MAG_{b}_FINAL"])
        simp = np.array(catalog[f"SIMPLE_PHOTO_MAG_{b}"])
        cog  = np.array(catalog[f"COG_MAG_{b}_FINAL"])
        
        org_trac = np.array(catalog[f"MAG_{b}"])

        best = np.select(
            [prefer_tractor_based_mag, prefer_simple_mag],
            [trac, simp],
            default=cog
        )

        catalog[f"MAG_{b}_BEST"] = best

        #then we overwrite the above if the prefer_org_tractor condition is satisfied
        #so just updating the ones where that is satisfied
        best_new = best.copy()
        best_new[prefer_org_tractor_mag] = org_trac[prefer_org_tractor_mag]

        catalog[f"MAG_{b}_BEST"] = best_new

    
    # Source label as string
    source = np.full(len(catalog), "COG", dtype=object)  # default COG
    source[prefer_simple_mag] = "SIMPLE"
    source[prefer_tractor_based_mag] = "TRACTOR_BASED"
    source[prefer_org_tractor_mag] = "TRACTOR_OG"

    print("Need to include the VI column that only does the aperture no subtract photo")
    
    catalog[f"MAG_TYPE"] = source

    return catalog


def consolidate_cog_photo(catalog,sample=None, add_pcnn=True):
    '''
    Function where we add PCNN column and consolidate the ISOLATE and no ISOLATE cog photometry using the over de-deblending criterion. 
    '''

     # this was due to a bug I had in my code
    if "APERFRAC_R4_IN_IMG_ISOLATE" in catalog.colnames:
        pass
    else:
        print("NEED TO REMOVE THIS IN THE NEXT RUN ITERATION!")
        catalog["APERFRAC_R4_IN_IMG_ISOLATE"] = np.array(catalog["APERFRAC_R4_IN_IMG_NO_ISOLATE"]).copy()


    catalog = make_catalog_unmasked(catalog)

    #these are the columns we want to make that consolidate based on whether to use the isolate or no isolate mask
    org_keys_to_combine = ["COG_MAG_G", "COG_MAG_R", "COG_MAG_Z", "TRACTOR_ONLY_MAG_G", "TRACTOR_ONLY_MAG_R", "TRACTOR_ONLY_MAG_Z", 
                           "APER_R4_MAG_G", "APER_R4_MAG_R","APER_R4_MAG_Z", "APERFRAC_R4_IN_IMG", "COG_CHI2", "COG_DOF", "COG_MAG_ERR",
                           "FIBER_MAG", "COG_PARAMS_G","COG_PARAMS_R", "COG_PARAMS_Z", "COG_PARAMS_G_ERR", "COG_PARAMS_R_ERR",
                           "COG_PARAMS_Z_ERR", "COG_DECREASE_MAX_LEN", "COG_DECREASE_MAX_MAG", "APER_CEN_RADEC", "APER_CEN_XY_PIX",
                           "APER_R4_MASK_FRAC", "APER_CEN_MASKED", "APER_PARAMS", "APER_MU_R_SIZES"]

    #to the above we also add the tractor columns!
    tractor_keys_to_combine = ["TRACTOR_ONLY_COG_MAG", "TRACTOR_ONLY_FIBER_MAG", "TRACTOR_ONLY_COG_MAG_ERR", "TRACTOR_ONLY_COG_PARAMS_G", 
                               "TRACTOR_ONLY_COG_PARAMS_G_ERR", "TRACTOR_ONLY_COG_PARAMS_R", "TRACTOR_ONLY_COG_PARAMS_R_ERR", 
                               "TRACTOR_ONLY_COG_PARAMS_Z", "TRACTOR_ONLY_COG_PARAMS_Z_ERR", "TRACTOR_ONLY_COG_CHI2",
                               "TRACTOR_ONLY_APER_CEN_RADEC", "TRACTOR_ONLY_APER_PARAMS","TRACTOR_APER_CEN_MASKED","NUM_TRACTOR_SOURCES", "TRACTOR_MU_R_SIZES",
                              "TRACTOR_BRIGHTEST_SOURCE_MAGS"]

    all_keys_to_combine = org_keys_to_combine + tractor_keys_to_combine

    apply_no_isolate_mask = likely_over_deblended(catalog["Z"].data, catalog["APER_R2_MU_R_ISLAND_TRACTOR"].data)
    #when this is true, we use no_isolate photometry. Otherwise, we use the isolate photometry
    
    pairs = {}

    for ki in all_keys_to_combine:
        pairs[ki] = ( catalog[ki + "_ISOLATE" ].data , catalog[ki + "_NO_ISOLATE"].data )
        
    for newcol, (w_iso, no_iso) in pairs.items():
        catalog[newcol + "_FINAL"] = combine_arrays(no_iso, w_iso, apply_no_isolate_mask)

    #add the pcnn column, load the appropriate sample
    if add_pcnn:
        if sample == "SGA":
            flag="sga"
        else:
            flag = "shreds"
        pcnn_cat = Table.read(f"/pscratch/sd/v/virajvm/catalog_dr1_dwarfs/iron_photometry/iron_{sample}_{flag}_catalog_w_aper_mags_pcnn_vals.fits")
    
        if len(pcnn_cat) != len(catalog):
            raise ValueError(f"Pcnn cat and catalog do not have the same lengths = { len(pcnn_cat), len(catalog) }")
        
        #then we match them
        idx,d2d, _ = match_c_to_catalog(c_cat=catalog, catalog_cat=pcnn_cat)
        if np.max(d2d.arcsec) > 1e-3:
            raise ValueError(f"Angular separation is non-zero = { np.max(d2d.arcsec) }")
    
        pcnn_cat = pcnn_cat[idx]
        
        tgid_max_diff = np.abs(np.max( catalog["TARGETID"].data - pcnn_cat["TARGETID"].data)) 
        if tgid_max_diff != 0:
            raise ValueError(f"TARGETIDs do not match")
    
        catalog["PCNN_FRAGMENT"] = pcnn_cat["PCNN_FRAGMENT"].data
        print("Added PCNN values!")
    else:
        print("Not adding the PCNN values! Adding all 1.")
        catalog["PCNN_FRAGMENT"] = np.ones(len(catalog))
        
    
    #need to make a column indicating whether the final photo was with isolate or no isolate
    catalog["ISOLATE_MASK_LIKELY_SHREDDING"] = apply_no_isolate_mask

    return catalog


def consolidate_new_photo(catalog,plot=False,sample=None, add_pcnn=True, use_pcnn=False, flag_cog_nan_always=True):
    '''

    Note that the PHOTO_MASKBIT reflects only on the final type of photometry used. 
    '''

    catalog = consolidate_cog_photo(catalog,sample=sample, add_pcnn=add_pcnn)
  
    catalog = add_best_mags(catalog,use_pcnn=use_pcnn)

    #add the photomaskbit column
    if sample == "SGA":
        #not to apply maskbit=12 as these are objects already in SGA!
       bitmasks_list = [0,1,2,3,4,5,6,7,8,9,10,11]
    else:
       bitmasks_list = [0,1,2,3,4,5,6,7,8,9,10,11,12]
    
    print("Adding the photo maskbits")
    photo_maskbits =  create_shred_maskbits_from_dict(catalog, bitmasks_to_apply = bitmasks_list, verbose=True)
    catalog["PHOTO_MASKBIT"] = photo_maskbits

    #now for the subset that we think has robust photometry, we want to ignore majority of their maskbits above and just start again!
    #However, we only want to do this for sources that have a failed COG photometry?
    using_org_tractor = revert_back_to_org_tractor(catalog,use_pcnn=use_pcnn)
    
    print(f"Fraction of sources where org trac is likely good = {np.sum(using_org_tractor)/len(catalog)}")
    #then we need to update some of the maskbits accordingly: bad color, iffy tractor model, we now do not care about the star!!

    #if we want to always flag objects where cog value is not measured
    #the bitmask = 0 is for objects that had a nan cog value. So something suspicious could be happening.
    if flag_cog_nan_always:
        print("Adding bit=0 for tractor_og type!")
        extra_bit = [0]
    else:
        print("Not adding bit=0 for tractor_og type!")
        extra_bit = []
    
    if sample == "SGA":
        #not to apply maskbit=13 as these are objects already in SGA!
       bitmasks_list = extra_bit + [7,11,13,14,15]
    else:
       bitmasks_list = extra_bit + [7,11,12,13,14,15]
        
    print("Updating the maskbits to reflect some objects reverted to original Tractor photometry")
    # only_trac_maskbits = create_shred_maskbits_from_dict(catalog, bitmasks_to_apply = bitmasks_list, verbose=True)
    current_maskbits = np.array(catalog["PHOTO_MASKBIT"], copy=True)
    
    new_bits = create_shred_maskbits_from_dict(
        catalog[using_org_tractor],
        bitmasks_to_apply=bitmasks_list,
        verbose=True
    )

    assert new_bits.shape[0] == np.sum(using_org_tractor)

    current_maskbits[using_org_tractor] = new_bits

    catalog["PHOTO_MASKBIT"] = current_maskbits

    #now print the summary statistics of the consolidated photometry!!
    print_maskbit_statistics(current_maskbits, bitmasks_to_use = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15])

    if sample == "SGA":
        #rename the SAMPLE columns
        catalog["IN_SGA_2020"] = np.ones(len(catalog)).astype(bool)
        #overwriting the SAMPLE column (which is just SGA now)
        catalog["SAMPLE"] = catalog["SAMPLE_DESI"].copy()
    else:
        catalog["IN_SGA_2020"] = np.zeros(len(catalog)).astype(bool)
        catalog.remove_columns("SGA_ID_MATCH")
        catalog.remove_columns("SGA_D26_NORM_DIST")
        catalog.remove_columns("SGA_DIST_DEG")
        
    #ONE CONSIDERATION TO KEEP IN MIND IS THAT TRACTOR TENDS TO NOT DO WELL, NEXT TO VERY BRIGHT STARS, AND THIS CAN AFFECT STUFF : 39627633918479081

    #39628088539091024 -> i am confused about this source and why is it close to a bstar??

    #39633183850892629 this is a very interesting offset merger object .. 

    ##this is a nice example to show working!: 39627642730709361, 39627643640878867
    #example of how merging systems can be hard: 39627643015922148
    #we have no optimally combined all the columns together!
    
    return catalog


#######################
### THE BELOW FUNCTIONS ARE ADDING THE DATA MODEL AND UNITS TO THE CATALOG. 
#######################

from data_model import tractor_datamodel, fastspec_hdu_datamodel, main_datamodel, zcat_datamodel, photo_datamodel

def half_light_radius_analytic(mtot, m0, alpha_1, r0, alpha_2):
    """
    Returns the half-light radius from the analytic expression of the 
    curve-of-growth model, given the best-fit parameters. 

    See equation 2 in SGA-2020 catalog paper. This expression is equivalent to that
    
    Parameters
    ----------
    mtot : float
        Total magnitude (not used directly, but included for completeness).
    m0 : float
        The m0 parameter of the empirical model.
    alpha_1, r0, alpha_2 : float
        Other model parameters.
    """
    exponent = 0.7525 / m0
    numerator = alpha_1
    denominator = np.exp(exponent) - 1.0
    return r0 * (numerator / denominator)**(1/alpha_2)

def measure_half_light_radius(cog_params=None, aper_params=None):
    '''
    Helper function that takes in the astropy table and adds half light radius columns
    '''

    new_shape_r = []

    for i in trange(len(cog_params)):

        mtot, m0, alpha_1, r0, alpha_2 = cog_params[i]
        aper_size = aper_params[i][0]

        #r12 in units of aperture size 
        r12_scaled = half_light_radius_analytic(mtot, m0, alpha_1, r0, alpha_2)

        r12_pix = r12_scaled * aper_size
        r12_arcsec = r12_pix * 0.262

        new_shape_r.append(r12_arcsec)
        
    return np.array(new_shape_r)
    

def consolidate_positions_and_shapes(catalog):
    """
    Consolidate RA, DEC, and shape parameters based on MAG_TYPE.

    We will have a separate column on size in arcsec. And the shape_params will be BA and PHI in an array.
    """

    print("Adding the consolidated RA,DEC, and SHAPE columns")

    catalog["RA_TARGET"] = catalog["RA"].copy()
    catalog["DEC_TARGET"] = catalog["DEC"].copy()
    
    # Extract MAG_TYPE as string array
    mag_type = np.array(catalog["MAG_TYPE"].data).astype(str)

    # Get input coordinate sets
    ra_aper_cen, dec_aper_cen = catalog["APER_CEN_RADEC_FINAL"].data[:, 0], catalog["APER_CEN_RADEC_FINAL"].data[:, 1]
    ra_trac_cen, dec_trac_cen = catalog["TRACTOR_ONLY_APER_CEN_RADEC_FINAL"].data[:, 0], catalog["TRACTOR_ONLY_APER_CEN_RADEC_FINAL"].data[:, 1]
    ra_org, dec_org = catalog["RA"].data, catalog["DEC"].data

    aper_params = catalog["APER_PARAMS_FINAL"].data                   # shape (N, 3)
    cog_aper_params = catalog["COG_PARAMS_R_FINAL"].data # Get the cog parameters for the r-band. we are only computing sizes for that!
    
    trac_aper_params = catalog["TRACTOR_ONLY_APER_PARAMS_FINAL"].data # shape (N, 3)
    cog_trac_aper_params = catalog["TRACTOR_ONLY_COG_PARAMS_R_FINAL"].data # shape (N, 3)

    #BELOW ARE COG CURVE BASED HALF-LIGHT RADIUS IN ARCSECONDS!
    cog_based_rhalf = measure_half_light_radius(cog_params = cog_aper_params, aper_params = aper_params  )
    trac_based_rhalf = measure_half_light_radius(cog_params = cog_trac_aper_params, aper_params = trac_aper_params  )

    org_rhalf = catalog["SHAPE_R"].data
    
    aper_shape_params = aper_params[:,1:].copy()
    trac_aper_shape_params = trac_aper_params[:,1:].copy()

    #convert the angles into standard astro convention. convert to degrees first
    aper_shape_params[:,1] = 90 + np.degrees(aper_shape_params[:,1])
    trac_aper_shape_params[:,1] = 90 + np.degrees(trac_aper_shape_params[:,1])

    
    # #converting the semi-major axis in pixels to arcseconds!
    # aper_params[:, 0] *= 0.262
    # trac_aper_params[:, 0] *= 0.262

    #NOTE THAT THE PHI (computed from tractor catalog) is in the standard astronomical convection 

    org_aper_shape_params = np.vstack([
        catalog["BA"].data,
        catalog["PHI"].data
    ]).T.astype(np.float32)                                           # shape (N, 2)

    # print("TODO: check that the SHAPE_R, BA, PHI columns are consistent with the aperture ones, especially PHI.")
    # print("TODO: add the VI + aper r3 based stuff here too")
        
    # Prepare output arrays
    n = len(mag_type)
    ra_final = np.full(n, np.nan, dtype=np.float64)
    dec_final = np.full(n, np.nan, dtype=np.float64)
    shape_final = np.full((n, 2), np.nan, dtype=np.float32)
    size_final = np.full(n, np.nan, dtype=np.float64)
    
    phot_update_final = np.ones(len(catalog))

    # Masks for each type
    mask_cog_simple = np.isin(mag_type, ["COG", "SIMPLE"])
    mask_only_simple = (mag_type == "SIMPLE") #this will be used for setting the sizes to be zero
    mask_trac_based = (mag_type == "TRACTOR_BASED")
    mask_trac_org   = (mag_type == "TRACTOR_OG")

    # Assign values
    ra_final[mask_cog_simple]  = ra_aper_cen[mask_cog_simple]
    dec_final[mask_cog_simple] = dec_aper_cen[mask_cog_simple]
    size_final[mask_cog_simple] = cog_based_rhalf[mask_cog_simple]
    shape_final[mask_cog_simple] = aper_shape_params[mask_cog_simple]

    #making just the simple ones back nan's again
    size_final[mask_only_simple] = np.nan
    
    ra_final[mask_trac_based]  = ra_trac_cen[mask_trac_based]
    dec_final[mask_trac_based] = dec_trac_cen[mask_trac_based]
    size_final[mask_trac_based] = trac_based_rhalf[mask_trac_based]
    shape_final[mask_trac_based] = trac_aper_shape_params[mask_trac_based]

    ra_final[mask_trac_org]  = ra_org[mask_trac_org]
    dec_final[mask_trac_org] = dec_org[mask_trac_org]
    size_final[mask_trac_org] = org_rhalf[mask_trac_org]
    shape_final[mask_trac_org] = org_aper_shape_params[mask_trac_org]

    phot_update_final[mask_trac_org] = 0

    # Add to catalog, and over-writing the original RA,DEC columns. The original RA,DEC columns are stored in RA_TARGET, DEC_TARGET
    catalog["RA"] = ra_final
    catalog["DEC"] = dec_final
    catalog["R50_R"] = size_final   #the half light radius in arcseconds in the r-band
    catalog["SHAPE_PARAMS"] = shape_final
    catalog["PHOTOMETRY_UPDATED"] =  phot_update_final.astype(bool)
    
    print("Consolidated RA, DEC, and SHAPE_PARAMS columns added. Added PHOTOMETRY_UPDATED column")
    
    return catalog


def create_main_data_model(catalog, save_name, clean_cat=False):
    '''
    Function that creates the data model for the main hdu. Containing the most important information.

    Note that the stuff passed here is before the shred and clean photo are combined. Here we are just selecting the relevant columns and prepping them
    '''

    #let us duplicate the RA,DEC to RA_TARGET,DEC_TARGET
    #for the shredded sources, the RA,DEC columns will be updated!
    
    catalog["RA_TARGET"] = catalog["RA"].copy()
    catalog["DEC_TARGET"] = catalog["DEC"].copy()
    catalog.rename_column("DIST_MPC_FIDU", "LUMI_DIST_MPC")

    catalog["MAG_G_TARGET"]  = catalog["MAG_G"].copy()
    catalog["MAG_R_TARGET"]  = catalog["MAG_R"].copy()
    catalog["MAG_Z_TARGET"]  = catalog["MAG_Z"].copy()

    #make sure none of the columns are masked columns to avoid subtle, unknown bugs!!

    if clean_cat:

        if "SGA_RA_MOMENT" in catalog.keys():
            #we need to remove some of the SGA columns as they are masked and it makes some issues! 
            #there are SGA columns here because there are 47 objects in SGA catalog that have robust tractor photometry, so we just put them in here
            in_sga_2020 = np.zeros(len(catalog))
            print(f"{np.sum(~catalog['SGA_RA_MOMENT'].data.mask)} objects in clean that are in SGA-2020")
            in_sga_2020[~catalog["SGA_RA_MOMENT"].data.mask]  = 1
            catalog["IN_SGA_2020"] = in_sga_2020.astype(bool)
        else:
            #we need to add the IN_SGA_2020 column as this is where we deal with the QSO scn catalog
            #this is only if the tractor maskbit = 12 is on
            maskbit_12_flagged = (catalog["MASKBITS"] & (1 << 12)) != 0
            catalog["IN_SGA_2020"] = maskbit_12_flagged
            #note: there will be objects in qso/scnd catalog that are shreds, but low fracflux values, and in SGA-2020 catalog
            #these will likely be in outskirts and so will be flagged by clean_maskbits later in dwarf_maskbit code below (dwarf_maskbit=12)
            #but that source is explicity also not considering sources with tractor maskbits = 12.
            #so make a note in the catalog that the OTHER objects are not processsed through our pipeline!
            
        #then remove the not necessary columns
        columns_to_remove = [
            "SGA_RA_MOMENT", "SGA_DEC_MOMENT", "SGA_SMA_SB26", "SGA_SMA_SB25",
            "SGA_BA", "SGA_PA", "SGA_R_COG_MAG", "SGA_G_COG_MAG", "SGA_Z_COG_MAG",
            "SGA_ZRED_LEDA", "SGA_ID", "SGA_MAG_LEDA", "SGA_ID_MATCH", "SGA_DIST_DEG"]

        # Only remove columns that exist in the catalog
        cols_in_cat = [col for col in columns_to_remove if col in catalog.colnames]
        catalog.remove_columns(cols_in_cat)

        catalog = make_catalog_unmasked(catalog)
        
    else:
        catalog = make_catalog_unmasked(catalog)

    if clean_cat:
        print("Processing clean catalog!")

        catalog["MAG_TYPE"] = np.full(len(catalog), "TRACTOR_OG", dtype=object)


        if "LOGM_SAGA_FIDU" in catalog.keys():
            catalog.rename_column("LOGM_SAGA_FIDU", "LOG_MSTAR_SAGA")
        else:
            #compute the fiducial stellar mass
            gr_colors = catalog["MAG_G"].data - catalog["MAG_R"].data
            mstars_SAGA_new = get_stellar_mass(gr_colors, catalog["MAG_R"].data, catalog["Z_CMB"].data ,d_in_mpc = catalog["LUMI_DIST_MPC"].data, input_zred=False )
            catalog["LOG_MSTAR_SAGA"] = mstars_SAGA_new
            
        catalog["PHOTOMETRY_UPDATED"] = np.zeros(len(catalog)).astype(bool)  

        #updating the stellar mass from M24, uses g band magnitude
        gr_colors = catalog["MAG_G"].data - catalog["MAG_R"].data
        log_mstars_M24 = get_stellar_mass_mia(gr_colors, catalog["MAG_G"].data , catalog["Z_CMB"].data, d_in_mpc = catalog["LUMI_DIST_MPC"].data, input_zred =  False)

        catalog["LOG_MSTAR_M24"] = log_mstars_M24

        print("Adding DWARF MASKBIT columns to clean catalog")
        clean_maskbits = create_shred_maskbits_from_dict(catalog, bitmasks_to_apply = [7,12,13,14,15], verbose=True, mag_type = "")

        catalog["DWARF_MASKBIT"] = clean_maskbits

        #add the SHAPE_PARAMS column
        org_aper_params = np.vstack([catalog["BA"].data,catalog["PHI"].data]).T.astype(np.float32)
        catalog["SHAPE_PARAMS"] = org_aper_params

        catalog["R50_R"] = catalog["SHAPE_R"].data

    else:
        print("Processing shred catalog!")
        #then overwriting the original magnitude with the new magnitudes
        catalog["MAG_G"] = catalog["MAG_G_BEST"].copy()
        catalog["MAG_R"] = catalog["MAG_R_BEST"].copy()
        catalog["MAG_Z"] = catalog["MAG_Z_BEST"].copy()
    
        catalog = consolidate_positions_and_shapes(catalog)

        catalog.rename_column("PHOTO_MASKBIT", "DWARF_MASKBIT")

        #with the finalized photometry, get the stellar masses!!
        gr_colors = catalog["MAG_G"].data - catalog["MAG_R"].data
        
        mstars_SAGA_new = get_stellar_mass(gr_colors, catalog["MAG_R"].data, catalog["Z_CMB"].data ,d_in_mpc = catalog["LUMI_DIST_MPC"].data, input_zred=False )
        
        mstars_M24_new = get_stellar_mass_mia(gr_colors, catalog["MAG_G"].data , catalog["Z_CMB"].data, d_in_mpc = catalog["LUMI_DIST_MPC"].data, input_zred =  False)

        catalog["LOG_MSTAR_SAGA"] = mstars_SAGA_new
        catalog["LOG_MSTAR_M24"] = mstars_M24_new


    print("Applying the dwarf galaxy cut!")
    # print("Need to update to the M24 mass cut")
    print(f"Number before dwarf mass cut = {len(catalog)}")
    catalog = catalog[catalog["LOG_MSTAR_M24"].data < 9.25]
    print(f"Number after dwarf mass cut = {len(catalog)}")
    
    #then we loop over the columns to get the final subset of columns
    # Keep only columns present in main_datamodel
    print("Selecting the subset of columns for MAIN extension")
    catalog_main = catalog[ [col for col in main_datamodel.keys() if col not in ["ASSOCIATED_TARGETIDS", "DWARF_PRIMARY_TARGETID", "DWARF_PRIMARY", "DIST_SOURCE"] ] ]
    
    for col in main_datamodel.keys():
        if col not in ["ASSOCIATED_TARGETIDS", "DWARF_PRIMARY_TARGETID", "DWARF_PRIMARY", "DIST_SOURCE"]:
            print(f"Column : {col}")
            meta = main_datamodel[col]
    
            # Set dtype if it doesn’t match (optional, only if you want strict consistency)
            desired_dtype = np.dtype(meta["dtype"])
            if catalog_main[col].dtype != desired_dtype:
                catalog_main[col] = catalog_main[col].astype(desired_dtype)
    
            # Add description and unit
            if meta.get("description"):
                catalog_main[col].description = meta["description"]
            if meta.get("unit") is not None:
                catalog_main[col].unit = meta["unit"]
    
            # Handle blank values if desired, we will only explicity provide a blank value in the datamodel if it is a nan type
    
            blank_val = meta.get("blank_value", None)
            if blank_val is not None:
                # replace masked or invalid entries
                mask = np.isnan(catalog_main[col])
                catalog_main[col][mask] = blank_val

    #save to fits file
    catalog_main.write(save_name, overwrite=True)

    return catalog_main, catalog


def finalize_main_hdu(catalog_main):
    '''
    We add the associated targetid columns!
    '''
    ##add the associated targetid columns!!

    print("Finding the associated TARGETIDs!")

    catalog_main = find_associated_tgids(catalog_main)
    catalog_main = get_dwarf_primary(catalog_main)

    print("Need to think a bit more about the blank value stuff")
    for col in main_datamodel.keys():
        print(f"Column : {col}")
        meta = main_datamodel[col]

        # Set dtype if it doesn’t match (optional, only if you want strict consistency)
        desired_dtype = np.dtype(meta["dtype"])
        if catalog_main[col].dtype != desired_dtype:
            catalog_main[col] = catalog_main[col].astype(desired_dtype)

        # Add description and unit
        if meta.get("description"):
            catalog_main[col].description = meta["description"]
        if meta.get("unit") is not None:
            catalog_main[col].unit = meta["unit"]

        # Handle blank values if desired, we will only explicity provide a blank value in the datamodel if it is a nan type

        blank_val = meta.get("blank_value", None)
        if blank_val is not None:
            # replace masked or invalid entries
            mask = np.isnan(catalog_main[col])
            catalog_main[col][mask] = blank_val

    return catalog_main


def create_tractor_data_model(catalog,save_name):
    '''
    Function that creates the data model for the tractor hdu
    '''

    print("Selecting the subset of columns for TRACTOR extension")
    tractor_tab = catalog[[col for col in tractor_datamodel.keys()]]

    ##add the fiber fluxes for the LOWZ subset
    tractor_tab = add_lowz_fiberflux(tractor_tab, catalog)

    # 2. Add metadata from tractor_datamodel
    for col in tractor_tab.colnames:
        print(f"Column : {col}")
        meta = tractor_datamodel[col]

        # Set dtype if it doesn’t match (optional, only if you want strict consistency)
        desired_dtype = np.dtype(meta["dtype"])
        if tractor_tab[col].dtype != desired_dtype:
            tractor_tab[col] = tractor_tab[col].astype(desired_dtype)

        # Add description and unit
        if meta.get("description"):
            tractor_tab[col].description = meta["description"]
        if meta.get("unit") is not None:
            tractor_tab[col].unit = meta["unit"]

        # Handle blank values if desired

        blank_val = meta.get("blank_value", None)
        if blank_val is not None:
            # replace masked or invalid entries
            print(col)
            mask = np.isnan(tractor_tab[col])
            tractor_tab[col][mask] = blank_val

    # 3. Save to FITS
    tractor_tab.write(save_name, overwrite=True)

    return tractor_tab



def create_zcat_data_model(catalog, save_name):

    print("Selecting the subset of columns for ZCAT extension")
    zcat_tab = catalog[[col for col in zcat_datamodel.keys()]]

    # 2. Add metadata from tractor_datamodel
    for col in zcat_tab.colnames:
        print(f"Column : {col}")
        meta = zcat_datamodel[col]

        # Set dtype if it doesn’t match (optional, only if you want strict consistency)
        desired_dtype = np.dtype(meta["dtype"])
        if zcat_tab[col].dtype != desired_dtype:
            zcat_tab[col] = zcat_tab[col].astype(desired_dtype)

        # Add description and unit
        if meta.get("description"):
            zcat_tab[col].description = meta["description"]
        if meta.get("unit") is not None:
            zcat_tab[col].unit = meta["unit"]

        # Handle blank values if desired

        blank_val = meta.get("blank_value", None)
        if blank_val is not None:
            # replace masked or invalid entries
            mask = np.isnan(zcat_tab[col])
            zcat_tab[col][mask] = blank_val

    # 3. Save to FITS
    zcat_tab.write(save_name, overwrite=True)

    return zcat_tab


def create_fastspec_data_model(fastspec_cat,save_name):
    '''
    Function that creates the data model for the tractor hdu
    '''
    
    fastspec_cat.rename_column("RA","RA_TARGET")
    fastspec_cat.rename_column("DEC","DEC_TARGET")

    # 2. Add metadata from tractor_datamodel
    for col in fastspec_cat.colnames:
        print(f"Column : {col}")
        meta = fastspec_hdu_datamodel[col]

        # Set dtype if it doesn’t match (optional, only if you want strict consistency)
        desired_dtype = np.dtype(meta["dtype"])
        if fastspec_cat[col].dtype != desired_dtype:
            fastspec_cat[col] = fastspec_cat[col].astype(desired_dtype)

        # Add description and unit
        if meta.get("description"):
            fastspec_cat[col].description = meta["description"]
        if meta.get("unit") is not None:
            fastspec_cat[col].unit = meta["unit"]

        # Handle blank values if desired
        blank_val = meta.get("blank_value", None)
        if blank_val is not None:
            # replace masked or invalid entries
            mask = fastspec_cat[col].mask if hasattr(fastspec_cat[col], 'mask') else np.isnan(fastspec_cat[col])
            fastspec_cat[col][mask] = blank_val

    # 3. Save to FITS
    # fastspec_cat.write(save_name, overwrite=True)

    return fastspec_cat


def create_new_photo_data_model(catalog, save_name):
    '''
    We will only included galaxies here whose photometry we did update!! So the number of rows will not be consistent
    '''
    
    #updating the names of some columns to be consistent with the data model dict
    catalog.rename_column("COG_CHI2_NO_ISOLATE","COG_FIT_RESID_NO_ISOLATE")
    catalog.rename_column("COG_CHI2_ISOLATE","COG_FIT_RESID_ISOLATE")
    catalog.rename_column("APER_R2_MU_R_ISLAND_TRACTOR","APER_R2_MU_R_BLOB_TRACTOR")

    for bi in "GRZ":
        for ii in ["ISOLATE","NO_ISOLATE"]:
            catalog.rename_column(f"TRACTOR_ONLY_MAG_{bi}_{ii}",f"TRACTOR_BASED_MAG_{bi}_{ii}")
            
    #then we loop over the columns to get the final subset of columns
    print("Selecting the subset of columns for REPROCESS_PHOTO_CAT extension")
    catalog = catalog[[col for col in photo_datamodel.keys()]]
    
    print("Need to think a bit more about the blank value stuff")
    for col in photo_datamodel.keys():
        print(f"Column : {col}")
        meta = photo_datamodel[col]

        # Set dtype if it doesn’t match (optional, only if you want strict consistency)
        desired_dtype = np.dtype(meta["dtype"])
        if catalog[col].dtype != desired_dtype:
            catalog[col] = catalog[col].astype(desired_dtype)

        # Add description and unit
        if meta.get("description"):
            catalog[col].description = meta["description"]
        if meta.get("unit") is not None:
            catalog[col].unit = meta["unit"]

        # Handle blank values if desired, we will only explicity provide a blank value in the datamodel if it is a nan type

        blank_val = meta.get("blank_value", None)
        if blank_val is not None:
            # replace masked or invalid entries
            mask = np.isnan(catalog[col])
            catalog[col][mask] = blank_val

    #save to fits file
    catalog.write(save_name, overwrite=True)

    return




def compute_emission_line_snr(flux, flux_ivar):
    """
    Compute emission-line signal-to-noise ratio as flux * sqrt(flux_ivar).
    Returns 0 for entries with non-finite or negative ivar.
    """
    flux = np.asarray(flux, dtype=np.float64)
    flux_ivar = np.asarray(flux_ivar, dtype=np.float64)
    valid = np.isfinite(flux_ivar) & (flux_ivar > 0) & np.isfinite(flux)
    snr = np.zeros_like(flux)
    snr[valid] = flux[valid] * np.sqrt(flux_ivar[valid])
    return snr


def apply_emission_line_snr_cuts(fastspec_cat, snr_threshold=3.0):
    """
    Return a boolean mask selecting rows where ALL three emission lines
    (Halpha, Hbeta, OIII 5007) have SNR >= snr_threshold.
    """
    halpha_snr = compute_emission_line_snr(fastspec_cat["HALPHA_FLUX"], fastspec_cat["HALPHA_FLUX_IVAR"])
    hbeta_snr = compute_emission_line_snr(fastspec_cat["HBETA_FLUX"], fastspec_cat["HBETA_FLUX_IVAR"])
    oiii_snr = compute_emission_line_snr(fastspec_cat["OIII_5007_FLUX"], fastspec_cat["OIII_5007_FLUX_IVAR"])

    passing = (halpha_snr >= snr_threshold) & (hbeta_snr >= snr_threshold) & (oiii_snr >= snr_threshold)

    print(f"Emission line SNR >= {snr_threshold} cut summary:")
    print(f"  Halpha passing: {np.sum(halpha_snr >= snr_threshold)}")
    print(f"  Hbeta  passing: {np.sum(hbeta_snr >= snr_threshold)}")
    print(f"  OIII   passing: {np.sum(oiii_snr >= snr_threshold)}")
    print(f"  All three passing: {np.sum(passing)} / {len(passing)}")

    return passing


def load_and_filter_qso_scnd_candidates(input_path = "/pscratch/sd/v/virajvm/catalog_dr1_dwarfs/hidden_dwarf_candidates_qso_mws_scnd.fits", snr_threshold=3.0):
    """
    Load hidden dwarf candidates from QSO/SCND target selections, deduplicate,
    remove MWS objects, match to FastSpecFit, and apply emission-line SNR cuts
    to ensure robust redshifts.

    Parameters
    ----------
    input_path : str
        Path to the hidden_dwarf_candidates_qso_mws_scnd.fits file.
    snr_threshold : float
        Minimum SNR required on Halpha, Hbeta, and OIII 5007 (all three must pass).

    Returns
    -------
    catalog : astropy.table.Table
        Filtered catalog with SAMPLE set to "OTHER".
    """
    print("=" * 60)
    print("Loading QSO/SCND hidden dwarf candidates")
    print("=" * 60)

    cats = Table.read(input_path)
    print(f"Total rows loaded: {len(cats)}")

    # Deduplicate by TARGETID
    _, uni_idx = np.unique(cats["TARGETID"].data, return_index=True)
    cats = cats[uni_idx]
    print(f"After TARGETID deduplication: {len(cats)}")

    # Remove MWS objects
    cats = cats[cats["ORIGIN_SAMPLE"] != "MWS"]
    print(f"After removing MWS: {len(cats)}")
    print(f"  Unique ORIGIN_SAMPLE values: {np.unique(cats['ORIGIN_SAMPLE'])}")
    for sample in np.unique(cats["ORIGIN_SAMPLE"]):
        print(f"    {sample}: {np.sum(cats['ORIGIN_SAMPLE'] == sample)}")

    # Match to FastSpecFit by TARGETID to get emission-line measurements
    print("Matching to FastSpecFit catalog by TARGETID...")
    fastspec_matched = match_fastspec_catalog(cats, coord_name="", match_method="TARGETID")

    # Apply emission-line SNR cuts
    snr_mask = apply_emission_line_snr_cuts(fastspec_matched, snr_threshold=snr_threshold)

    cats = cats[snr_mask]
    print(f"After emission-line SNR >= {snr_threshold} cuts: {len(cats)}")
    for sample in np.unique(cats["ORIGIN_SAMPLE"]):
        print(f"    {sample}: {np.sum(cats['ORIGIN_SAMPLE'] == sample)}")

    # Set SAMPLE column to "OTHER"
    cats["SAMPLE"] = np.full(len(cats), "OTHER", dtype=object)

    ##add in SGA and other information!

    cats = get_sga_norm_dists_FAST(cats, siena_path="/global/cfs/cdirs/cosmo/data/sga/2020/SGA-2020.fits")

    #keys to be added to catalog regarding bright star information
    bstar_keys = [ "STARFDIST", "STARDIST_DEG","STARMAG", "STAR_RADIUS_ARCSEC", "STAR_RA","STAR_DEC"]

    # Check if all bright star keys exist
    if all(key in cats.colnames for key in bstar_keys):
        print("Bright star information already exists!")
    else:
        # Recompute if missing
        print("Bright star information did not exist and will be computed.")
        cats = bright_star_filter(cats)

    cats = add_sweeps_column(cats)

    print(f"Final QSO/SCND catalog: {len(cats)} objects (SAMPLE='OTHER')")
    print("=" * 60)

    return cats


def get_fastspec_matched_catalog(gal_cat, save_name, match_method = "TARGETID"):
    '''
    Get the RA,DEC matched fastspec catalog and save it   
    '''

    #make sure the catalog being matched to us v2
    fastspec_cat = match_fastspec_catalog(gal_cat,coord_name = "",match_method = match_method)

    #make sure this is not a masked column!
    fastspec_cat = make_catalog_unmasked(fastspec_cat)

    #save this 
    fastspec_cat.write(f"{save_name}",overwrite=True)

    #see what fraction of the catalog has np.nans in catalog
    mask = np.isnan(fastspec_cat["RA"])
    print(f"{np.sum(mask)}/{len(mask)} objects have no match in Fastspecfit catalog!")
    return


##THESE ARE ALL THE FASTSPEC COLUMNS WE WISH TO READ!
fastspec_metadata_cols = ["TARGETID","RA","DEC"]

fastspec_specphot_cols = [
    "DN4000", "DN4000_OBS", "DN4000_IVAR", "DN4000_MODEL", "DN4000_MODEL_IVAR",
    "VDISP", "VDISP_IVAR",
    "FOII_3727_CONT", "FOII_3727_CONT_IVAR",
    "FHBETA_CONT", "FHBETA_CONT_IVAR",
    "FOIII_5007_CONT", "FOIII_5007_CONT_IVAR",
    "FHALPHA_CONT", "FHALPHA_CONT_IVAR",
    "FLUX_SYNTH_G", "FLUX_SYNTH_R", "FLUX_SYNTH_Z",
    "FLUX_SYNTH_SPECMODEL_G", "FLUX_SYNTH_SPECMODEL_R", "FLUX_SYNTH_SPECMODEL_Z",
    "FLUX_SYNTH_PHOTMODEL_G", "FLUX_SYNTH_PHOTMODEL_R", "FLUX_SYNTH_PHOTMODEL_Z",
]

fastspec_cols = ["SNR_B", "SNR_R", "SNR_Z", "APERCORR", "APERCORR_G", "APERCORR_R", "APERCORR_Z"] 

fastspec_mag_cols = ["ABSMAG01_SDSS_G", "ABSMAG01_SDSS_R", "ABSMAG01_SDSS_I", "ABSMAG01_SDSS_Z", "ABSMAG01_IVAR_SDSS_G", "ABSMAG01_IVAR_SDSS_R", "ABSMAG01_IVAR_SDSS_I", "ABSMAG01_IVAR_SDSS_Z" ]

fastspec_emlines_cols = ["OII_3726_FLUX", "OII_3726_FLUX_IVAR", "OII_3729_FLUX", "OII_3729_FLUX_IVAR", "OIII_4363_FLUX", "OIII_4363_FLUX_IVAR", "HEII_4686_FLUX", "HEII_4686_FLUX_IVAR", "HBETA_FLUX", "HBETA_FLUX_IVAR", "OIII_4959_FLUX", "OIII_4959_FLUX_IVAR", "OIII_5007_FLUX", "OIII_5007_FLUX_IVAR", "HEI_5876_FLUX", "HEI_5876_FLUX_IVAR", "NII_6548_FLUX", "NII_6548_FLUX_IVAR", "HALPHA_FLUX", "HALPHA_FLUX_IVAR", "HALPHA_BROAD_FLUX", "HALPHA_BROAD_FLUX_IVAR", "NII_6584_FLUX", "NII_6584_FLUX_IVAR", "SII_6716_FLUX", "SII_6716_FLUX_IVAR", "SII_6731_FLUX", "SII_6731_FLUX_IVAR", "SIII_9069_FLUX", "SIII_9069_FLUX_IVAR", "SIII_9532_FLUX", "SIII_9532_FLUX_IVAR", "HALPHA_BOXFLUX", "HALPHA_BOXFLUX_IVAR", "HALPHA_EW", "HALPHA_EW_IVAR", "HALPHA_SIGMA", "HALPHA_SIGMA_IVAR"]

fastspec_tot_cols = fastspec_cols +  fastspec_emlines_cols + fastspec_mag_cols

def get_fastspec_fit_catalog_V3():
    '''
    In this function, we combine the relevant columns and healpix fastspec files (VERSION 3 CATALOG). 

    NOTE: WE WILL BE USING FASTSPEC V2.1, not V3.0
    '''

    # Path pattern to your FITS files
    files_bright = glob.glob("/global/cfs/cdirs/desi/public/dr1/vac/dr1/fastspecfit/iron/v3.0/catalogs/fastspec-iron*bright*.fits")
    files_dark = glob.glob("/global/cfs/cdirs/desi/public/dr1/vac/dr1/fastspecfit/iron/v3.0/catalogs/fastspec-iron*dark*.fits")
    files_backup = glob.glob("/global/cfs/cdirs/desi/public/dr1/vac/dr1/fastspecfit/iron/v3.0/catalogs/fastspec-iron*backup*.fits")
    files_other = glob.glob("/global/cfs/cdirs/desi/public/dr1/vac/dr1/fastspecfit/iron/v3.0/catalogs/fastspec-iron*other*.fits")
    
    files = files_bright + files_dark + files_backup + files_other

    print(f"Total number of files to read = {len(files)}")
    
    #goal is to create our own main fastspec!!
    
    tables = []
    for ind,f in enumerate(files):
        print(ind,f)
        with fits.open(f) as hdul:
            # usually the table is in HDU 1; adjust if needed
            tab_meta_zred = hdul["METADATA"].data["Z"]
            tab_meta_spectype = hdul["METADATA"].data["SPECTYPE"]
            

            #select for redshift and spectype
            zmask = (tab_meta_zred < 0.5) & (tab_meta_spectype == "GALAXY")
            print(f"Selecting {np.sum(zmask)/len(zmask):.3f} fraction of objects")
    
            tab_specphot = Table(hdul["SPECPHOT"].data[zmask])[fastspec_specphot_cols]
            tab_fastspec = Table(hdul["FASTSPEC"].data[zmask])[fastspec_tot_cols]
            tab_meta = Table(hdul["METADATA"].data[zmask])[fastspec_metadata_cols]
    
            #hstack these!!
            tab_i = hstack([tab_meta, tab_fastspec, tab_specphot])
    
            #let us only keep objects that are galaxies a
            print(len(tab_i))
            tables.append(tab_i)
        print("---")
    

    #now we stack this all and save it!
    tables = vstack(tables)

    tables.write("/pscratch/sd/v/virajvm/catalog_dr1_dwarfs/iron_fastspec_catalog/iron_fastspec_v3.fits",overwrite=True)
        
    return


def get_fastspec_fit_catalog_V2(chunk_size = 250000):
    '''
    In this function, we combine the relevant columns and healpix fastspec files (VERSION 2 CATALOG)
    This function is run before the dwarf catalog is being constructed as it prepares the intermediate file we will be matching to
    ''' 

    main_cat_path = "/global/cfs/cdirs/desi/public/dr1/vac/dr1/fastspecfit/iron/v2.1/catalogs/fastspec-iron.fits"

    #as we are dealing with the v2 here, not all columns are available
    to_remove_v2 = ["HALPHA_SIGMA","HALPHA_SIGMA_IVAR", "DN4000_MODEL_IVAR","FOII_3727_CONT", "FOII_3727_CONT_IVAR", "FHBETA_CONT", "FHBETA_CONT_IVAR", "FOIII_5007_CONT", "FOIII_5007_CONT_IVAR","FHALPHA_CONT", "FHALPHA_CONT_IVAR"]
    fastspec_tot_cols_v2 = [s for s in fastspec_tot_cols if s not in to_remove_v2]
    fastspec_specphot_cols_v2 = [s for s in fastspec_specphot_cols if s not in to_remove_v2]
    
    with fits.open(main_cat_path, memmap=True) as hdul:
        meta = hdul["METADATA"].data
        fastspec = hdul["FASTSPEC"].data
        nrows = len(meta)

        # Prepare an output list
        out_chunks = []
        for start in range(0, nrows, chunk_size):
            stop = min(start + chunk_size, nrows)
            print(f"Reading {start}:{stop} out of {nrows}")
        
            zmask = (meta["Z"][start:stop] < 0.5) & (meta["SPECTYPE"][start:stop] == "GALAXY")
            if not np.any(zmask):
                print("No galaxies in this chunk satisfy the cut")
                continue

            tab_meta = Table(meta[start:stop][zmask])[fastspec_metadata_cols]
            tab_fastspec = Table(fastspec[start:stop][zmask])[fastspec_tot_cols_v2 + fastspec_specphot_cols_v2]
            out_chunks.append(hstack([tab_meta, tab_fastspec]))

        result = vstack(out_chunks)
        print(len(result))
        result.write("/pscratch/sd/v/virajvm/catalog_dr1_dwarfs/iron_fastspec_catalog/iron_fastspec_v21.fits",overwrite=True)

        # # usually the table is in HDU 1; adjust if needed
        # tab_meta_zred = hdul["METADATA"].data["Z"]
        # tab_meta_spectype = hdul["METADATA"].data["SPECTYPE"]
        
        # #select for redshift and spectype
        # zmask = (tab_meta_zred < 0.5) & (tab_meta_spectype == "GALAXY")
        # print(f"Selecting {np.sum(zmask)/len(zmask):.3f} fraction of objects")

        # tab_fastspec = Table(hdul["FASTSPEC"].data[zmask], columns = fastspec_tot_cols + fastspec_specphot_cols)
        # tab_meta = Table(hdul["METADATA"].data[zmask], columns = fastspec_metadata_cols)

        # #hstack these!!
        # tables = hstack([tab_meta, tab_fastspec])

        # #let us only keep objects that are galaxies a
        # print(len(tables))
    return


def add_lowz_fiberflux(trac_cat,tot_cat):
    '''
    Function where we cross-match the lowz sources and add the fiber info there! 
    '''

    if len(trac_cat) != len(tot_cat):
        raise ValueError("Incorrect lengths for the trac_cat and tot_cat tables!")
    
    lowz_mask = (tot_cat["SAMPLE"] == "LOWZ")
    print(f"{np.sum(lowz_mask)} number of objects in the LOWZ catalog!")

    lowz_tot_cat = tot_cat[lowz_mask]

    print(f"BEFORE: Example FIBERFLUX_R: {trac_cat[lowz_mask]['FIBERFLUX_R'].data[:5]}")
    
    #loading the updated lowz tractor catalog
    lowz_trac_cat_f = Table.read("/pscratch/sd/v/virajvm/catalog_dr1_dwarfs/iron_lowz_filter_zsucc_zrr03_INT.fits")

    ##we cross_match this trac_cat with that
    idx,d2d,_ = match_c_to_catalog(c_cat = lowz_tot_cat, catalog_cat = lowz_trac_cat_f, c_ra="RA_TARGET",c_dec="DEC_TARGET",catalog_ra="RA",catalog_dec="DEC")

    #get the matching fiberflux_r and fibermag_r
    lowz_rfib_flux = lowz_trac_cat_f["FIBERFLUX_R"].data[idx]
    lowz_rfib_mag = lowz_trac_cat_f["FIBERMAG_R"].data[idx]

    #update the trac_cat accordingly!
    trac_cat["FIBERFLUX_R"][lowz_mask] = lowz_rfib_flux
    trac_cat["FIBERMAG_R"][lowz_mask] = lowz_rfib_mag

    print(f"AFTER: Example FIBERFLUX_R: {trac_cat[lowz_mask]['FIBERFLUX_R'].data[:5]}")
    
    return trac_cat


def combine_hdus(hdu_list, base_path="/pscratch/sd/v/virajvm/desi_dwarf_catalogs/dr1/v1.0/temp_cats",
                 output_file="/pscratch/sd/v/virajvm/desi_dwarf_catalogs/dr1/v1.0/desi_dwarfs_combined.fits",
                 extra_prefixes=None):
    """
    Combine multiple HDUs (Astropy tables) into a single multi-extension FITS file.

    Parameters
    ----------
    hdu_list : list of str
        List of HDU identifiers to combine, e.g., ["main", "fspec"].
    base_path : str
        Directory containing the HDU FITS files.
    output_file : str
        Path for the combined FITS file.
    extra_prefixes : list of str, optional
        Additional catalog prefixes to vstack alongside clean/shreds
        (e.g., ["qso_scnd"]). Files are included only if they exist on disk.
    """
    if extra_prefixes is None:
        extra_prefixes = []

    hdu_tables = []
    for hdu_name in hdu_list:
        if hdu_name in ["REPROCESS_PHOTO"]:
            shred_fname = os.path.join(base_path, f"shreds_{hdu_name}_hdu.fits")
            print(f"Reading {shred_fname}...")
            shred_tab = Table.read(shred_fname)
            
            hdu_tables.append(shred_tab)
        
        else:
            shred_fname = os.path.join(base_path, f"shreds_{hdu_name}_hdu.fits")
            clean_fname = os.path.join(base_path, f"clean_{hdu_name}_hdu.fits")
            
            print(f"Reading {shred_fname}...")
            print(f"Reading {clean_fname}...")
            
            clean_tab = Table.read(clean_fname)
            shred_tab = Table.read(shred_fname)
    
            tables_to_stack = [clean_tab, shred_tab]

            for prefix in extra_prefixes:
                extra_fname = os.path.join(base_path, f"{prefix}_{hdu_name}_hdu.fits")
                if os.path.exists(extra_fname):
                    print(f"Reading {extra_fname}...")
                    tables_to_stack.append(Table.read(extra_fname))
                else:
                    print(f"Skipping {extra_fname} (not found)")

            tab = vstack(tables_to_stack)

            if hdu_name in ["MAIN"]:
                ##if main, we will add three new columns
                tab = finalize_main_hdu(tab)
            
            hdu_tables.append(tab)


    #we ignore the reprocess_cat from comparison as that as a different number of rows by construction

    # Sanity check: number of rows
    nrows = [len(tab) for tab in hdu_tables[:-1]]
    if len(set(nrows)) != 1:
        raise ValueError(f"Row count mismatch across HDUs: {dict(zip(hdu_list, nrows))}")

    # Sanity check: TARGETID alignment
    target_ids = [tab["TARGETID"] for tab in hdu_tables[:-1]]
    for i in range(1, len(target_ids)):
        diff = target_ids[i] - target_ids[0]
        if not (diff == 0).all():
            raise ValueError(f"TARGETID mismatch between {hdu_list[0]} and {hdu_list[i]}")

    print(f"Total number of dwarf galaxies = {len(target_ids[0])}")

    # Create primary HDU
    primary_hdu = fits.PrimaryHDU()
    hdul = [primary_hdu]

    # Convert each Table to BinTableHDU (preserves units, descriptions)
    for tab, hdu_name in zip(hdu_tables, hdu_list):
        buf = BytesIO()
        tab.write(buf, format="fits")
        buf.seek(0)
        bintable_hdu = fits.open(buf)[1]
        bintable_hdu.name = hdu_name.upper()
        bintable_hdu.add_checksum()
        hdul.append(bintable_hdu)

    hdulist = fits.HDUList(hdul)

    # Add checksum to primary HDU
    hdulist[0].add_checksum()

    # Write out to FITS
    hdulist.writeto(output_file, overwrite=True)
    print(f"Combined FITS written to {output_file}")


def create_spectra_hdu(file_path):
    """
    Create a new Astropy Table (HDU) for the NMF+PCA coefficients, normalization factors, 
    and UMAP 2D coordinates. The table has the same TARGETIDs/order as MAIN HDU in file_path,
    with missing TARGETIDs filled with zeros (-99 for UMAP).
    """

    # Load main catalog
    main_cat = Table.read(file_path, hdu="MAIN")
    main_tgids = main_cat["TARGETID"].data
    n_objects = len(main_tgids)

    # Load HDF5 coefficients
    h5_path = "/pscratch/sd/v/virajvm/catalog_dr1_dwarfs/iron_spectra/spectra_files/desi_dr1_dwarf_catalog_nnmf_pca.h5"
    with h5py.File(h5_path, "r") as f:
        pca_coeffs = f["PCA_COEFFS"][:]       # shape (N, 20)
        nnmf_coeffs = f["NNMF_COEFFS"][:]     # shape (N, 10)
        tgids = f["TARGETID"][:]
        norm_facs = f["NORM_FACTOR"][:]
        nnmf_rnorm = f["NNMF_RNORM"][:]

    print(f"PCA COEFFS SHAPE: {pca_coeffs.shape}")

    print(f"PCA COEFFS Example: {pca_coeffs[5]}")

    print(f"PCA COEFFS 3 median and std: {np.median(pca_coeffs[3]), np.std(pca_coeffs[3]) }")

    print(f"NNMF COEFFS SHAPE: {nnmf_coeffs.shape}")

    print(f"NNMF COEFFS Example: {nnmf_coeffs[10]}")

    print(f"NNMF COEFFS 5 median and std: {np.median(nnmf_coeffs[5]), np.std(nnmf_coeffs[5])}")
    

    #we ensure that norm_facs do not have any zeroes
    scale_facs = 1/norm_facs
    scale_facs = scale_facs[scale_facs != 0]
    norm_facs = 1/scale_facs
    
    # Load UMAP
    spec_umap_2d = np.load("/pscratch/sd/v/virajvm/catalog_dr1_dwarfs/iron_spectra/spectra_files/desi_dwarfs_umap_nnmf_and_pca_v2.npy")

    # Consistency checks
    n = pca_coeffs.shape[0]
    assert nnmf_coeffs.shape[0] == n, "NMF/PCA length mismatch"
    assert len(tgids) == n, "TGID length mismatch"
    assert len(norm_facs) == n, "Norm factor length mismatch"
    assert spec_umap_2d.shape[0] == n, "UMAP length mismatch"
    assert len(nnmf_rnorm) == n, "NNMF RNORM mismatch"
    
    # Create TGID -> index mapping
    tgid_to_idx = {tgid: i for i, tgid in enumerate(tgids)}

    # Prepare columns
    new_table = Table()
    new_table["TARGETID"] = main_tgids

    # NNMF columns
    print(f"NNMF params = {nnmf_coeffs.shape[1]}")
    for j in range(nnmf_coeffs.shape[1]):
        col = np.full(n_objects, -99.0)
        for i, tgid in enumerate(main_tgids):
            if tgid in tgid_to_idx:
                col[i] = nnmf_coeffs[tgid_to_idx[tgid], j]
        new_table[f"NNMF_{j}"] = col

    # PCA columns
    print(f"PCA params = {pca_coeffs.shape[1]}")
    for j in range(pca_coeffs.shape[1]):
        col = np.full(n_objects, -99.0)
        for i, tgid in enumerate(main_tgids):
            if tgid in tgid_to_idx:
                col[i] = pca_coeffs[tgid_to_idx[tgid], j]
        new_table[f"PCA_{j}"] = col

    # Normalization factor
    norm_col = np.full(n_objects, -99.0)
    for i, tgid in enumerate(main_tgids):
        if tgid in tgid_to_idx:
            norm_col[i] = norm_facs[tgid_to_idx[tgid]]
    new_table["NNMF_NORM_FACTOR"] = norm_col

    #NNMF residuals
    nnmf_resid_col = np.full(n_objects, -99.0)
    for i, tgid in enumerate(main_tgids):
        if tgid in tgid_to_idx:
            nnmf_resid_col[i] = nnmf_rnorm[tgid_to_idx[tgid]]
    new_table["NNMF_RESID"] = nnmf_resid_col
    
    # UMAP 2D coordinates
    umap0 = np.full(n_objects, -99.0)
    umap1 = np.full(n_objects, -99.0)
    for i, tgid in enumerate(main_tgids):
        if tgid in tgid_to_idx:
            idx = tgid_to_idx[tgid]
            umap0[i] = spec_umap_2d[idx, 0]
            umap1[i] = spec_umap_2d[idx, 1]
    new_table["SPEC_UMAP_0"] = umap0
    new_table["SPEC_UMAP_1"] = umap1

    # Convert Table to BinTableHDU
    new_hdu = fits.BinTableHDU(new_table, name="SPECTRA_TEMPLATE")

    # Open original FITS, append new HDU, and overwrite file
    with fits.open(file_path, mode="update") as hdul:
        hdul.append(new_hdu)
        hdul.flush()  # write changes to disk

    print(f"Added SPECTRA_TEMPLATE extension to {file_path} (length = {n_objects})")

    return


def create_image_ssl_hdu(file_path):
    """
    Create and append an IMG_SSL HDU containing image-based SSL UMAP 2D
    coordinates and the top-10 most similar TARGETIDs (with cosine similarity
    scores) for every object in the MAIN HDU.

    Rows are aligned to MAIN by TARGETID.  Objects without SSL data are
    filled with -99 (int64 columns) or -99.0 (float64 columns).
    """

    # Load main catalog to get authoritative TARGETID ordering
    main_cat = Table.read(file_path, hdu="MAIN")
    main_tgids = main_cat["TARGETID"].data
    n_objects = len(main_tgids)

    # Load similarity data
    sim_scores = np.load("/pscratch/sd/v/virajvm/catalog_dr1_dwarfs/ssl_shred_data/similarity_search_magb/all_similarity_scores_total.npy")
    sim_tgids = np.load("/pscratch/sd/v/virajvm/catalog_dr1_dwarfs/ssl_shred_data/similarity_search_magb/all_similarity_targetids_total.npy")

    # Load UMAP 2D coordinates and their associated TARGETIDs
    umaps_dwarfs = np.load("/pscratch/sd/v/virajvm/catalog_dr1_dwarfs/ssl_shred_data/umap/total_umap_embedding_2d.npy")
    tgid_vals = np.load("/pscratch/sd/v/virajvm/catalog_dr1_dwarfs/ssl_shred_data/dwarf_dr1/total_targetids_arr.npy")

    print(f"Similarity arrays shape: scores={sim_scores.shape}, tgids={sim_tgids.shape}")
    print(f"UMAP array shape: {umaps_dwarfs.shape}, tgid_vals length: {len(tgid_vals)}")

    assert umaps_dwarfs.shape[0] == len(tgid_vals), "UMAP rows and tgid_vals length mismatch"
    assert sim_scores.shape == sim_tgids.shape, "sim_scores and sim_tgids shape mismatch"

    n_sim = 10

    # Build lookup: query TARGETID -> row index in sim arrays
    # sim_tgids[:, 0] holds the query TARGETID for each row
    sim_tgid_to_row = {int(sim_tgids[i, 0]): i for i in range(sim_tgids.shape[0])}

    # Build lookup: TARGETID -> row index in UMAP arrays
    umap_tgid_to_idx = {int(tgid_vals[i]): i for i in range(len(tgid_vals))}

    # Prepare output columns
    img_umap_0 = np.full(n_objects, -99.0, dtype=np.float64)
    img_umap_1 = np.full(n_objects, -99.0, dtype=np.float64)

    sim_targetid_cols = np.full((n_objects, n_sim), -99, dtype=np.int64)
    sim_score_cols = np.full((n_objects, n_sim), -99.0, dtype=np.float64)

    for i, tgid in enumerate(main_tgids):
        tgid_int = int(tgid)

        # UMAP coordinates
        if tgid_int in umap_tgid_to_idx:
            uidx = umap_tgid_to_idx[tgid_int]
            img_umap_0[i] = umaps_dwarfs[uidx, 0]
            img_umap_1[i] = umaps_dwarfs[uidx, 1]

        # Similarity: skip self (column 0), take next 10
        if tgid_int in sim_tgid_to_row:
            row = sim_tgid_to_row[tgid_int]
            avail = min(n_sim, sim_tgids.shape[1] - 1)
            sim_targetid_cols[i, :avail] = sim_tgids[row, 1:1 + avail]
            sim_score_cols[i, :avail] = sim_scores[row, 1:1 + avail]

    # Build astropy Table
    new_table = Table()
    new_table["TARGETID"] = main_tgids

    new_table["IMG_UMAP_0"] = img_umap_0
    new_table["IMG_UMAP_1"] = img_umap_1

    for j in range(n_sim):
        new_table[f"SIM_TARGETID_{j}"] = sim_targetid_cols[:, j]
        new_table[f"SIM_SCORE_{j}"] = sim_score_cols[:, j]

    # Append as new HDU
    new_hdu = fits.BinTableHDU(new_table, name="IMG_SSL")

    with fits.open(file_path, mode="update") as hdul:
        hdul.append(new_hdu)
        hdul.flush()

    n_matched_umap = np.sum(img_umap_0 != -99.0)
    n_matched_sim = np.sum(sim_targetid_cols[:, 0] != -99)
    print(f"Added IMG_SSL extension to {file_path} (length = {n_objects})")
    print(f"  UMAP matched: {n_matched_umap}/{n_objects}")
    print(f"  Similarity matched: {n_matched_sim}/{n_objects}")

    return


def add_wrong_redrock_maskbit(cat_path, main_datamodel, bit=16):
    """
    Identify objects with wrong Redrock redshifts and update DWARF_MASKBIT.
    Updates the MAIN HDU safely, preserving variable-length columns and all other HDUs.
    
    Parameters
    ----------
    cat_path : str
        Path to the multi-extension FITS catalog.
    main_datamodel : dict
        Dictionary describing column metadata (dtype, description, unit, blank_value).
    bit : int, optional
        Bit index to set for weird/redshift-flagged objects (default 16).
    """

    # --- Read relevant tables ---
    main_cat = Table.read(cat_path, hdu="MAIN")
    fspec_cat = Table.read(cat_path, hdu="FASTSPEC")
    spec_cat = Table.read(cat_path, hdu="SPECTRA_TEMPLATE")

    # --- Identify weird/redshift-mismatch objects ---
    weird_mask = flag_weird_spectra(spec_cat, main_cat, fspec_cat)
    weird_mask = np.asarray(weird_mask, dtype=bool)
    if len(weird_mask) != len(main_cat):
        raise ValueError("weird_mask length does not match MAIN table")

    # --- Update DWARF_MASKBIT ---
    dwarf_maskbits = np.asarray(main_cat["DWARF_MASKBIT"], dtype=np.int64)
    dwarf_maskbits[weird_mask] |= np.int64(1) << bit
    main_cat["DWARF_MASKBIT"] = dwarf_maskbits

    # --- Apply main_datamodel metadata ---
    for col in main_datamodel.keys():
        if col not in main_cat.colnames:
            print(f"Skipping column: {col}")
            continue  # skip if column missing
        meta = main_datamodel[col]

        # Ensure dtype matches datamodel
        desired_dtype = np.dtype(meta["dtype"])
        if main_cat[col].dtype != desired_dtype:
            main_cat[col] = main_cat[col].astype(desired_dtype)

        # Set description
        if meta.get("description"):
            main_cat[col].description = meta["description"]

        # Set unit
        if meta.get("unit") is not None:
            main_cat[col].unit = meta["unit"]

        # Set blank values if provided
        blank_val = meta.get("blank_value", None)
        if blank_val is not None:
            mask = np.isnan(main_cat[col])
            main_cat[col][mask] = blank_val

    # --- Write MAIN HDU to a temporary HDU ---
    buf = BytesIO()
    main_cat.write(buf, format="fits")
    buf.seek(0)
    main_hdu = fits.open(buf)[1]
    main_hdu.name = "MAIN"
    main_hdu.add_checksum()

    # --- Open original file and replace MAIN HDU ---
    with fits.open(cat_path, mode="update") as hdul:
        hdul[1] = main_hdu  # HDU[1] is MAIN in your multi-extension file
        hdul.flush()

    print(f"Set DWARF_MASKBIT bit {bit} for {weird_mask.sum()} objects (MAIN HDU updated).")



def incorporate_updated_distances(main_cat_path):

    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm

    main_hdu, _, _, _ = update_distance_catalog(main_cat_path, keep_lumi_dist_orig=False)

    # With the updated distance in LUMI_DIST_MPC, remeasure the stellar mass
    gr_arr = np.array(main_hdu["MAG_G"]) - np.array(main_hdu["MAG_R"])
    mag_r_arr = np.array(main_hdu["MAG_R"])
    mag_g_arr = np.array(main_hdu["MAG_G"])
    zcmb_arr = np.array(main_hdu["Z_CMB"])
    dist_arr = np.array(main_hdu["LUMI_DIST_MPC"])

    logm_saga = get_stellar_mass(gr_arr, mag_r_arr, zcmb_arr, d_in_mpc=dist_arr, input_zred=False)
    logm_m24 = get_stellar_mass_mia(gr_arr, mag_g_arr, zcmb_arr, d_in_mpc=dist_arr, input_zred=False)

    # Quick comparison plot
    plt.figure(figsize=(5, 5))
    plt.hist2d(main_hdu["LOG_MSTAR_M24"], logm_m24, range=((6, 9.25), (6, 9.25)), bins=50, norm=LogNorm())
    plt.xlabel("MSTAR M24 OLD", fontsize=12)
    plt.ylabel("MSTAR M24 NEW", fontsize=12)
    plt.xlim([6, 9.25])
    plt.ylim([6, 9.25])
    plt.savefig("global/homes/v/virajvm/DESI2_LOWZ/quenched_fracs_nbs/paper_plots/mstar_dist_update_comp.png")
    plt.close()

    # Statistics on stellar mass shifts
    old_logm_m24 = np.array(main_hdu["LOG_MSTAR_M24"])
    delta_mstar = np.abs(logm_m24 - old_logm_m24)
    n_shifted = np.sum(delta_mstar > 0.25)
    print(f"Objects with > 0.25 dex shift in LOG_MSTAR_M24: {n_shifted}/{len(logm_m24)}")
    print(f"  Median |delta|: {np.median(delta_mstar):.4f} dex")
    print(f"  Max    |delta|: {np.max(delta_mstar):.4f} dex")

    # Update the stellar mass columns
    main_hdu["LOG_MSTAR_SAGA"] = logm_saga
    main_hdu["LOG_MSTAR_M24"] = logm_m24

    # Mask for objects that are no longer below the dwarf mass cut
    not_dwarf_mask = (logm_m24 > 9.25)
    dwarf_mask = ~not_dwarf_mask

    n_removed = np.sum(not_dwarf_mask)
    print(f"Removing {n_removed} objects with LOG_MSTAR_M24 > 9.25 after distance update")

    # Save the TARGETIDs of removed objects
    removed_tgids = np.array(main_hdu["TARGETID"])[not_dwarf_mask]
    removed_path = "/pscratch/sd/v/virajvm/catalog_dr1_dwarfs/temp/removed_targetids.fits"
    Table({"TARGETID": removed_tgids}).write(removed_path, overwrite=True)
    print(f"Saved {len(removed_tgids)} removed TARGETIDs to {removed_path}")

    # Filter the updated MAIN table
    main_hdu_filtered = main_hdu[dwarf_mask]
    kept_tgids = set(np.array(main_hdu["TARGETID"])[dwarf_mask])

    # REPROCESS_PHOTO has only shreds (not clean+shreds+qso_scnd),
    # so it has a different row count and must be filtered by TARGETID.
    diff_row_extensions = {"REPROCESS_PHOTO"}

    # Read the original file to get the exact HDU names and ordering
    with fits.open(main_cat_path) as orig_hdul:
        orig_hdu_names = [hdu.name for hdu in orig_hdul]
        orig_primary_header = orig_hdul[0].header.copy()

    print(f"Original HDU structure: {orig_hdu_names}")

    # Rebuild the HDUList preserving the exact extension order
    new_hdul = fits.HDUList()
    new_hdul.append(fits.PrimaryHDU(header=orig_primary_header))

    # Pre-read all extension Tables before we overwrite the file
    ext_tables = {}
    for hdu_name in orig_hdu_names:
        if hdu_name == "PRIMARY":
            continue
        ext_tables[hdu_name] = Table.read(main_cat_path, hdu=hdu_name)

    for hdu_name in orig_hdu_names:
        if hdu_name == "PRIMARY":
            continue

        if hdu_name == "MAIN":
            filtered_tab = main_hdu_filtered
        elif hdu_name in diff_row_extensions:
            ext_tab = ext_tables[hdu_name]
            keep = np.isin(np.array(ext_tab["TARGETID"]), np.array(main_hdu["TARGETID"])[dwarf_mask])
            filtered_tab = ext_tab[keep]
        else:
            filtered_tab = ext_tables[hdu_name][dwarf_mask]

        buf = BytesIO()
        filtered_tab.write(buf, format="fits")
        buf.seek(0)
        new_hdu = fits.open(buf)[1]
        new_hdu.name = hdu_name
        new_hdu.add_checksum()
        new_hdul.append(new_hdu)

        print(f"  {hdu_name}: {len(ext_tables[hdu_name])} -> {len(filtered_tab)} rows")

    new_hdul[0].add_checksum()
    new_hdul.writeto(main_cat_path, overwrite=True)
    print(f"Saved updated catalog to {main_cat_path} ({len(main_hdu_filtered)} objects)")

    return


def _prune_single_h5(h5_path, catalog_tgids, tgid_key, row_datasets, copy_datasets):
    """
    Backup an HDF5 file, then rewrite it keeping only rows whose TARGETID
    appears in *catalog_tgids*.

    Parameters
    ----------
    h5_path : str
        Path to the HDF5 file to prune.
    catalog_tgids : np.ndarray
        1-D array of TARGETIDs that should be kept.
    tgid_key : str
        Name of the TARGETID dataset inside the HDF5 (e.g. "TARGETID" or "targetid").
    row_datasets : list[str]
        Dataset names that are indexed along axis-0 by object and must be filtered.
    copy_datasets : list[str]
        Dataset names that are shared / not row-indexed and should be copied unchanged.
    """
    from desi_lowz_funcs import print_stage

    basename = os.path.basename(h5_path)
    print_stage(f"Pruning {basename}")

    # --- 1. Backup --------------------------------------------------------
    backup_path = h5_path.replace(".h5", "_OG.h5")
    if os.path.exists(backup_path):
        print(f"  Backup already exists at {backup_path} — skipping copy")
    else:
        shutil.copy2(h5_path, backup_path)
        print(f"  Backup saved to {backup_path}")

    # --- 2. Read TARGETIDs and compute keep mask --------------------------
    with h5py.File(h5_path, "r") as f:
        h5_tgids = f[tgid_key][:]
    
    keep_mask = np.isin(h5_tgids, catalog_tgids)
    n_before = len(h5_tgids)
    n_keep = int(keep_mask.sum())
    n_remove = n_before - n_keep
    print(f"  {basename}: {n_before} objects -> keeping {n_keep}, removing {n_remove}")

    if n_remove == 0:
        print(f"  Nothing to remove from {basename}")
        return

    # --- 3. Write pruned file to a temp path, then swap -------------------
    tmp_path = h5_path + ".tmp.h5"

    with h5py.File(h5_path, "r") as fin, h5py.File(tmp_path, "w") as fout:
        for ds_name in row_datasets:
            data = fin[ds_name][:]
            filtered = data[keep_mask]

            create_kwargs = {"dtype": filtered.dtype}

            orig_ds = fin[ds_name]
            if orig_ds.compression is not None:
                create_kwargs["compression"] = orig_ds.compression
                if orig_ds.compression_opts is not None:
                    create_kwargs["compression_opts"] = orig_ds.compression_opts
            if orig_ds.chunks is not None:
                orig_chunks = orig_ds.chunks
                new_chunks = (min(orig_chunks[0], len(filtered)),) + orig_chunks[1:]
                create_kwargs["chunks"] = new_chunks

            fout.create_dataset(ds_name, data=filtered, **create_kwargs)

        for ds_name in copy_datasets:
            data = fin[ds_name][:]
            fout.create_dataset(ds_name, data=data, dtype=fin[ds_name].dtype)

    os.replace(tmp_path, h5_path)
    print(f"  Pruned file written to {h5_path}")

    # --- 4. Validate ------------------------------------------------------
    with h5py.File(h5_path, "r") as f:
        final_count = f[tgid_key].shape[0]
    assert final_count == n_keep, (
        f"Validation failed for {basename}: expected {n_keep} rows, got {final_count}"
    )
    print(f"  Validated: {final_count} objects in pruned {basename}")


def prune_h5_files(main_cat_path, spectra_h5_path, images_h5_path):
    """
    After the FITS catalog has been pruned to remove non-dwarfs, propagate
    those removals into the spectra and images HDF5 files.

    Each H5 file is first backed up to ``<name>_OG.h5`` in the same directory,
    then rewritten to contain only the rows whose TARGETID still appears in
    the MAIN HDU of *main_cat_path*.
    """
    from desi_lowz_funcs import print_stage

    print_stage("Pruning H5 files to match updated catalog")

    catalog_tgids = np.array(Table.read(main_cat_path, hdu="MAIN")["TARGETID"])
    print(f"  Catalog contains {len(catalog_tgids)} surviving TARGETIDs")

    # --- Spectra H5 -------------------------------------------------------
    if os.path.exists(spectra_h5_path):
        _prune_single_h5(
            h5_path=spectra_h5_path,
            catalog_tgids=catalog_tgids,
            tgid_key="TARGETID",
            row_datasets=["TARGETID", "Z", "FLUX", "FLUX_IVAR"],
            copy_datasets=["WAVE"],
        )
    else:
        print(f"  WARNING: spectra H5 not found at {spectra_h5_path}, skipping")

    # --- Images H5 --------------------------------------------------------
    if os.path.exists(images_h5_path):
        _prune_single_h5(
            h5_path=images_h5_path,
            catalog_tgids=catalog_tgids,
            tgid_key="targetid",
            row_datasets=["targetid", "images"],
            copy_datasets=[],
        )
    else:
        print(f"  WARNING: images H5 not found at {images_h5_path}, skipping")

    print_stage("H5 pruning complete")


if __name__ == '__main__':

    save_path = "/pscratch/sd/v/virajvm/desi_dwarf_catalogs/dr1/v1.0/temp_cats"
    
    process_shreds = True
    process_clean = True
    process_qso_scnd = True
    
    process_fastspec=True

    main_cat_outpath = "/pscratch/sd/v/virajvm/desi_dwarf_catalogs/dr1/v1.0/desi_dr1_dwarf_catalog.fits"

    if process_shreds:
        #loading the shredded catalogs!
        print("Reading ELG shreds!")
        elg_shred = Table.read(f"/pscratch/sd/v/virajvm/catalog_dr1_dwarfs/iron_photometry/iron_ELG_shreds_catalog_w_aper_mags.fits")
        elg_shred = consolidate_new_photo(elg_shred,sample="ELG",flag_cog_nan_always=False)
        print("=="*10)
    
        print("Reading BGS Bright shreds!")
        bgsb_shred = Table.read(f"/pscratch/sd/v/virajvm/catalog_dr1_dwarfs/iron_photometry/iron_BGS_BRIGHT_shreds_catalog_w_aper_mags.fits")
        bgsb_shred = consolidate_new_photo(bgsb_shred,sample="BGS_BRIGHT",flag_cog_nan_always=True)
        print("=="*10)
        
        print("Reading BGS Faint shreds!")
        bgsf_shred = Table.read(f"/pscratch/sd/v/virajvm/catalog_dr1_dwarfs/iron_photometry/iron_BGS_FAINT_shreds_catalog_w_aper_mags.fits")
        bgsf_shred = consolidate_new_photo(bgsf_shred,sample="BGS_FAINT",flag_cog_nan_always=True)
        print("=="*10)
    
        print("Reading LOWZ shreds!")
        lowz_shred = Table.read(f"/pscratch/sd/v/virajvm/catalog_dr1_dwarfs/iron_photometry/iron_LOWZ_shreds_catalog_w_aper_mags.fits")
        lowz_shred = consolidate_new_photo(lowz_shred,sample="LOWZ",flag_cog_nan_always=True)
        print("=="*10)

        print("Reading SGA shreds!")
        sga_all = Table.read("/pscratch/sd/v/virajvm/catalog_dr1_dwarfs/iron_photometry/iron_SGA_sga_catalog_w_aper_mags.fits")
        sga_all = consolidate_new_photo(sga_all,sample="SGA",flag_cog_nan_always=True)
        print("=="*10)
    
        # --- remove extra columns from SGA before stacking ---
        extra_cols = set(sga_all.colnames) - set(lowz_shred.colnames)
        if extra_cols:
            print(f"Removing {len(extra_cols)} extra columns from SGA: {extra_cols}")
            sga_all.remove_columns(list(extra_cols))
        
        # optional: reorder columns to match LOWZ (keeps order consistent)
        sga_all = sga_all[lowz_shred.colnames]
    
        tot_shred = vstack([ bgsb_shred, bgsf_shred, lowz_shred, elg_shred, sga_all])
    
        ##get the main hdu
        print("Creating the shred main hdu")
        #the tot_shred catalog is the one with the subset of columns for main hdu
        tot_shred, tot_shred_entire = create_main_data_model(tot_shred, save_path + "/shreds_MAIN_hdu.fits", clean_cat=False)

        #get the tractor hdu
        print("Creating the shred tractor hdu")
        create_tractor_data_model(tot_shred_entire,save_path + "/shreds_TRACTOR_hdu.fits")

        #create the zcat hdu
        print("Creating the shred zcat hdu")
        create_zcat_data_model(tot_shred_entire, save_path + "/shreds_ZCAT_hdu.fits")

        #create the reprocess photo hdu
        print("Creating the shred reprocess photo hdu")
        create_new_photo_data_model(tot_shred_entire, save_path + "/shreds_REPROCESS_PHOTO_hdu.fits")
        
        ##get the fastspecfit hdu
        if process_fastspec:
            print("Creating the shred fastspecfit hdu")
            get_fastspec_matched_catalog(tot_shred, save_path + "/shreds_FASTSPEC_hdu.fits", match_method="TARGETID")
            
        ##get the other hdus

    if process_clean:
        ##get the clean catalog stuff now!!
        clean_cat = Table.read("/pscratch/sd/v/virajvm/catalog_dr1_dwarfs/desi_y1_dwarf_clean_catalog_v4.fits")

        print("Creating the clean main hdu")
        clean_cat, clean_cat_entire = create_main_data_model(clean_cat, save_path + "/clean_MAIN_hdu.fits", clean_cat=True)

        #get the tractor hdu
        print("Creating the clean tractor main hdu")
        create_tractor_data_model(clean_cat_entire,save_path  + "/clean_TRACTOR_hdu.fits")

        #create the zcat hdu
        print("Creating the clean zcat main hdu")
        create_zcat_data_model(clean_cat_entire, save_path + "/clean_ZCAT_hdu.fits")
        
        #will not make the reprocess photo hdu here!!

        ##get the fastspecfit hdu
        if process_fastspec:
            print("Creating the clean fastspecfit hdu")
            get_fastspec_matched_catalog(clean_cat, save_path + "/clean_FASTSPEC_hdu.fits", match_method="TARGETID")

    if process_qso_scnd:
        qso_scnd_input = "/pscratch/sd/v/virajvm/catalog_dr1_dwarfs/hidden_dwarf_candidates_qso_mws_scnd.fits"
        qso_scnd_cat = load_and_filter_qso_scnd_candidates(qso_scnd_input, snr_threshold=3.0)

        print("Creating the QSO/SCND main hdu")
        qso_scnd_cat, qso_scnd_entire = create_main_data_model(
            qso_scnd_cat, save_path + "/qso_scnd_MAIN_hdu.fits", clean_cat=True
        )

        create_tractor_data_model(qso_scnd_entire, save_path + "/qso_scnd_TRACTOR_hdu.fits")
        create_zcat_data_model(qso_scnd_entire, save_path + "/qso_scnd_ZCAT_hdu.fits")

        if process_fastspec:
            print("Creating the QSO/SCND fastspecfit hdu")
            get_fastspec_matched_catalog(qso_scnd_cat, save_path + "/qso_scnd_FASTSPEC_hdu.fits", match_method="TARGETID")

    #then we consolidate it all into a multi-ext file!
    #make sure the REPROCESS_PHOTO_CAT is also last in the below list!
    combine_hdus(["MAIN", "ZCAT", "TRACTOR", "FASTSPEC","REPROCESS_PHOTO"],
                 base_path="/pscratch/sd/v/virajvm/desi_dwarf_catalogs/dr1/v1.0/temp_cats",
                 output_file=main_cat_outpath,
                 extra_prefixes=["qso_scnd"] if process_qso_scnd else [])

    ##add the spectra NMF+PCA information!!
    create_spectra_hdu(main_cat_outpath)

    ##add image SSL UMAP + similarity information
    create_image_ssl_hdu(main_cat_outpath)

    print("TODO: remake the NNMF plots and the criterion for weird spectra!!")
        
    #update the dwarf_maskbit with some weird spectra masks
    add_wrong_redrock_maskbit(main_cat_outpath, main_datamodel)

    ##update the distances in the catalog and then remove objects that are not dwarf based on that!!
    incorporate_updated_distances(main_cat_outpath)

    ##propagate the TARGETID removals into the spectra and images H5 files
    # prune_h5_files(
    #     main_cat_path=main_cat_outpath,
    #     spectra_h5_path="/pscratch/sd/v/virajvm/catalog_dr1_dwarfs/iron_spectra/spectra_files/data/desi_dr1_dwarf_catalog_spectra.h5",
    #     images_h5_path="/pscratch/sd/v/virajvm/catalog_dr1_dwarfs/ssl_shred_data/desi_dr1_dwarf_catalog_images.h5",
    # )

    


























    

    

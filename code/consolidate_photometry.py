'''
Functions where preferred, best photometry column is created along with photometry maskbits to identify reliable measurements. This is also the script where we produce the final, multi-extension fits files as the final catalog output.
'''
import warnings
import numpy as np
import astropy.io.fits as fits
import astropy.units as u
from astropy.table import Table, vstack, join, hstack

warnings.filterwarnings(
    "ignore",
    message="The following header keyword is invalid",
    category=fits.verify.VerifyWarning,
)
warnings.filterwarnings(
    "ignore",
    message="The following header keyword is invalid",
    module=r"astropy\.io\.fits",
)
from shred_photometry_maskbits import cog_mag_converge, cog_nan_mask, cog_curve_decrease, bad_colors, iffy_tractor_model
from io import BytesIO
from shred_photometry_maskbits import create_shred_maskbits_from_dict, print_maskbit_statistics, flag_weird_spectra
import os
import glob
import tempfile
from tqdm import trange
import h5py
from desi_lowz_funcs import match_c_to_catalog, match_fastspec_catalog, get_stellar_mass_mia
from desi_lowz_funcs import add_sweeps_column
from desi_lowz_funcs import get_sga_norm_dists_FAST
from desitarget.targetmask import desi_mask, bgs_mask
from construct_dwarf_galaxy_catalogs import (
    bright_star_filter,
    get_sv_bgs_mask,
    get_sv_elg_mask,
)

from get_associated_fibers import find_associated_tgids, get_dwarf_primary
from consolidate_associated_fibers import symmetrize_and_group_associated_tgids, consolidate_associated_fiber_properties

from sfr_and_metallicity import add_sfr_halpha_to_spec_derived

from mass_and_photo_corrections import (
    make_catalog_unmasked,
    safe_read_table,
    safe_vstack,
    _load_nebcorr_is_south_lookup,
    _bass2decam_apply_mask,
    _load_nebcorr_delta_mag_table,
    _apply_delta_mag_corrections,
    add_delta_mag_to_fastspec,
    compute_emission_subtracted_photo_errors,
    NEBCORR_DEFAULT_FOLDER,
    NEBCORR_INT_V2_BASENAMES,
    INT_V2_BASENAMES,
    FASTSPEC_DELTA_MAG_COLS,
    DWARF_CATALOG_SPEC_HDU,
)


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


# make_catalog_unmasked, safe_read_table, safe_vstack are imported from mass_and_photo_corrections


def safe_hstack(tables, **kwargs):
    """hstack wrapper that strips MaskedColumns introduced by stacking."""
    return make_catalog_unmasked(hstack(tables, **kwargs))


def assert_no_masked_columns(table, label=""):
    """Debug helper: raises ValueError if any column is still a MaskedColumn."""
    masked_cols = [col for col in table.colnames if hasattr(table[col], "mask")]
    if masked_cols:
        raise ValueError(
            f"[{label}] Masked columns found: {masked_cols}. "
            "Call make_catalog_unmasked() first."
        )
    

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


def consolidate_cog_photo(catalog,sample=None, add_pcnn=False):
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
        pcnn_cat = safe_read_table(f"/pscratch/sd/v/virajvm/catalog_dr1_dwarfs/iron_photometry/iron_{sample}_{flag}_catalog_w_aper_mags_pcnn_vals.fits")
    
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


def consolidate_new_photo(catalog,plot=False,sample=None, add_pcnn=False, use_pcnn=False, flag_cog_nan_always=True):
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


def fix_zwarn(catalog_main):
    """
    Set ZWARN on MAIN to an integer column matching main_datamodel so FITS
    is written with an integer TFORM (e.g. J for int32), not logical (L) / bool.
    """
    if "ZWARN" not in catalog_main.colnames:
        return
    meta = main_datamodel["ZWARN"]
    target_dtype = np.dtype(meta["dtype"])
    a = np.asarray(catalog_main["ZWARN"])
    a = np.asarray(a, dtype=np.int64, order="C")
    a = a.astype(target_dtype, copy=False)
    catalog_main["ZWARN"] = a
    if meta.get("description"):
        catalog_main["ZWARN"].description = meta["description"]
    if meta.get("unit") is not None:
        catalog_main["ZWARN"].unit = meta["unit"]


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
            #there are SGA columns here because there are 47 objects in SGA catalog that have robust tractor photometry, so we just put them in here
            sga_col = catalog['SGA_RA_MOMENT']
            if hasattr(sga_col, 'mask'):
                valid = ~np.asarray(sga_col.mask)
            else:
                valid = ~np.isnan(np.asarray(sga_col))
            in_sga_2020 = np.zeros(len(catalog))
            print(f"{np.sum(valid)} objects in clean that are in SGA-2020")
            in_sga_2020[valid] = 1
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

        # Use the nebular+filter+k-corrected stellar mass from
        # construct_dwarf_galaxy_catalogs.py.  Fall back to applying the
        # DELTA_MAG correction chain if the pre-computed column is absent.
        if "LOGM_M24_FIDU_CORR" in catalog.keys():
            print("OLD CORRECTED MSTAR COLUMN FOUND!")
            catalog["LOG_MSTAR_M24"] = catalog["LOGM_M24_FIDU_CORR"].data.copy()
        else:
            mag_g_corr, mag_r_corr = _apply_delta_mag_corrections(catalog)
            gr_corr = mag_g_corr - mag_r_corr
            catalog["LOG_MSTAR_M24"] = get_stellar_mass_mia(
                gr_corr, mag_g_corr,
                zred=np.zeros(len(catalog)),
                d_in_mpc=catalog["LUMI_DIST_MPC"].data,
                input_zred=False,
            )

        catalog["PHOTOMETRY_UPDATED"] = np.zeros(len(catalog)).astype(bool)

        print("Adding DWARF MASKBIT columns to clean catalog")
        clean_maskbits = create_shred_maskbits_from_dict(catalog, bitmasks_to_apply = [7,12,13,14,15], verbose=True, mag_type = "")

        catalog["DWARF_MASKBIT"] = clean_maskbits

        #add the SHAPE_PARAMS column
        org_aper_params = np.vstack([catalog["BA"].data,catalog["PHI"].data]).T.astype(np.float32)
        catalog["SHAPE_PARAMS"] = org_aper_params

        catalog["R50_R"] = catalog["SHAPE_R"].data

    else:


        print("Processing shred catalog!")
        catalog["MAG_G"] = catalog["MAG_G_BEST"].copy()
        catalog["MAG_R"] = catalog["MAG_R_BEST"].copy()
        catalog["MAG_Z"] = catalog["MAG_Z_BEST"].copy()

        catalog = consolidate_positions_and_shapes(catalog)

        catalog.rename_column("PHOTO_MASKBIT", "DWARF_MASKBIT")

        # Apply DELTA_MAG corrections (nebular + filter + k) to the
        # reprocessed photometry, then compute stellar mass with zred=0
        # (model k-correction already folded into the deltas).
        mag_g_corr, mag_r_corr = _apply_delta_mag_corrections(catalog)
        gr_corr = mag_g_corr - mag_r_corr
        catalog["LOG_MSTAR_M24"] = get_stellar_mass_mia(
            gr_corr, mag_g_corr,
            zred=np.zeros(len(catalog)),
            d_in_mpc=catalog["LUMI_DIST_MPC"].data,
            input_zred=False,
        )


        
    print("Applying the dwarf galaxy cut!")
    
    print(f"Number before dwarf mass cut = {len(catalog)}")
    catalog = catalog[catalog["LOG_MSTAR_M24"].data < 9.25]
    print(f"Number after dwarf mass cut = {len(catalog)}")

    

    # Flag sources brighter than Mg = -18.5 (bit 18 of DWARF_MASKBIT)
    mag_g_corr_bright, _ = _apply_delta_mag_corrections(catalog)
    dist_pc = np.asarray(catalog["LUMI_DIST_MPC"], dtype=float) * 1.0e6
    valid_bright = np.isfinite(dist_pc) & (dist_pc > 0) & np.isfinite(mag_g_corr_bright)
    abs_mag_g = np.full(len(catalog), np.nan)
    abs_mag_g[valid_bright] = mag_g_corr_bright[valid_bright] - 5.0 * np.log10(dist_pc[valid_bright]) + 5.0
    too_bright = np.isfinite(abs_mag_g) & (abs_mag_g < -18.5)
    dwarf_maskbits = np.asarray(catalog["DWARF_MASKBIT"], dtype=np.int64)
    dwarf_maskbits[too_bright] |= np.int64(1) << 18
    catalog["DWARF_MASKBIT"] = dwarf_maskbits
    n_flagged = int(too_bright.sum())
    print(f"DWARF_MASKBIT bit 18 (Mg < -18.5): flagged {n_flagged}/{len(catalog)} "
          f"({100.0 * n_flagged / len(catalog):.1f}%)")

    mstar_maskbits = np.zeros(len(catalog), dtype=np.int64)
    mstar_maskbits[too_bright] |= np.int64(1) << 1
    catalog["MSTAR_MASKBIT"] = mstar_maskbits.astype(np.int32)

    #then we loop over the columns to get the final subset of columns
    # Keep only columns present in main_datamodel
    print("Selecting the subset of columns for MAIN extension")
    catalog_main = catalog[ [col for col in main_datamodel.keys() if col not in ["ASSOCIATED_TARGETIDS", "DWARF_PRIMARY_TARGETID", "DWARF_PRIMARY", "DIST_SOURCE","LOG_MSTAR_M24_ERR","PROPERTY_SOURCE_TARGETID", "DWARF_PRIMARY_TARGETID"] ] ]
    
    for col in main_datamodel.keys():
        if col not in ["ASSOCIATED_TARGETIDS", "DWARF_PRIMARY_TARGETID", "DWARF_PRIMARY", "DIST_SOURCE", "LOG_MSTAR_M24_ERR","PROPERTY_SOURCE_TARGETID", "DWARF_PRIMARY_TARGETID"]:
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
    
            blank_val = meta.get("blank_value", None)
            if blank_val is not None:
                col_data = np.asarray(catalog_main[col], dtype=float)
                bad = np.isnan(col_data)
                col_data[bad] = blank_val
                catalog_main[col] = col_data

    fix_zwarn(catalog_main)

    #save to fits file
    catalog_main.write(save_name, overwrite=True)

    return catalog_main, catalog


def _load_dist_source_lookup():
    """Build a TARGETID -> DIST_SOURCE lookup from the authoritative INT_V2_NEBCORR catalogs."""
    save_folder = "/pscratch/sd/v/virajvm/catalog_dr1_dwarfs"
    chunks = []
    for basename in NEBCORR_INT_V2_BASENAMES:
        path = os.path.join(save_folder, basename)
        if not os.path.exists(path):
            print(f"  WARNING: {path} not found, skipping")
            continue
        tab = Table.read(path)
        chunks.append(Table({"TARGETID": tab["TARGETID"], "DIST_SOURCE": tab["DIST_SOURCE"]}))
        print(f"  Loaded {len(tab)} rows from {basename}")

    if len(chunks) == 0:
        print("  ERROR: No INT_V2_NEBCORR files found for DIST_SOURCE lookup")
        return {}

    lookup_tab = vstack(chunks)
    _, unique_idx = np.unique(np.asarray(lookup_tab["TARGETID"]), return_index=True)
    lookup_tab = lookup_tab[np.sort(unique_idx)]
    return dict(zip(lookup_tab["TARGETID"], lookup_tab["DIST_SOURCE"]))


def finalize_main_hdu(catalog_main):
    '''
    We add the associated targetid columns!
    '''
    ##add the associated targetid columns!!

    print("Finding the associated TARGETIDs!")

    catalog_main = find_associated_tgids(catalog_main)
    catalog_main = symmetrize_and_group_associated_tgids(catalog_main)
    catalog_main = get_dwarf_primary(catalog_main)

    print("Populating DIST_SOURCE from INT_V2_NEBCORR catalogs...")
    dist_source_map = _load_dist_source_lookup()
    dist_source_arr = np.array([
        dist_source_map.get(tid, "") for tid in catalog_main["TARGETID"]
    ], dtype="U10")
    n_matched = np.sum(dist_source_arr != "")
    print(f"  DIST_SOURCE matched for {n_matched}/{len(catalog_main)} objects")
    catalog_main["DIST_SOURCE"] = dist_source_arr

    print("Need to think a bit more about the blank value stuff")
    for col in main_datamodel.keys():
        if col not in catalog_main.colnames:
            continue
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

        blank_val = meta.get("blank_value", None)
        if blank_val is not None:
            col_data = np.asarray(catalog_main[col], dtype=float)
            bad = np.isnan(col_data)
            col_data[bad] = blank_val
            catalog_main[col] = col_data

    fix_zwarn(catalog_main)

    return catalog_main


def create_tractor_data_model(catalog,save_name):
    '''
    Function that creates the data model for the tractor hdu
    '''

    print("Selecting the subset of columns for TRACTOR extension")
    _fibertot_cols = ("FIBERTOTFLUX_G", "FIBERTOTFLUX_R")
    _subset_cols = [c for c in tractor_datamodel.keys() if c not in _fibertot_cols]
    tractor_tab = catalog[_subset_cols]

    ##add the fiber fluxes for the LOWZ subset
    tractor_tab = add_lowz_fiberflux(tractor_tab, catalog)
    tractor_tab = add_fibertotflux_from_int_v2(tractor_tab, catalog)

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

        blank_val = meta.get("blank_value", None)
        if blank_val is not None:
            print(col)
            col_data = np.asarray(tractor_tab[col], dtype=float)
            bad = np.isnan(col_data)
            col_data[bad] = blank_val
            tractor_tab[col] = col_data

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

        blank_val = meta.get("blank_value", None)
        if blank_val is not None:
            col_data = np.asarray(zcat_tab[col], dtype=float)
            bad = np.isnan(col_data)
            col_data[bad] = blank_val
            zcat_tab[col] = col_data

    # 3. Save to FITS
    zcat_tab.write(save_name, overwrite=True)

    return zcat_tab


def create_fastspec_data_model(fastspec_cat,save_name):
    '''
    Function that creates the data model for the tractor hdu
    '''
    
    fastspec_cat = make_catalog_unmasked(fastspec_cat)

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
            col_data = np.asarray(fastspec_cat[col], dtype=float)
            bad = np.isnan(col_data)
            col_data[bad] = blank_val
            fastspec_cat[col] = col_data

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

        blank_val = meta.get("blank_value", None)
        if blank_val is not None:
            col_data = np.asarray(catalog[col], dtype=float)
            bad = np.isnan(col_data)
            col_data[bad] = blank_val
            catalog[col] = col_data

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


def load_and_filter_qso_scnd_candidates(
    input_path="/pscratch/sd/v/virajvm/catalog_dr1_dwarfs/iron_other_qso_scnd_candidates_INT_V2_NEBCORR.fits",
    snr_threshold=3.0,
):
    """
    Load QSO/SCND dwarf candidates after nebular correction and corrected stellar-mass
    filtering, then deduplicate, remove MWS objects, match to FastSpecFit, and apply
    emission-line SNR cuts for robust redshifts.

    The default ``input_path`` is ``iron_other_qso_scnd_candidates_INT_V2_NEBCORR.fits``,
    produced by ``construct_other_dwarf_catalog`` (INT_V2) followed by
    ``run_nebular_correction_int_v2`` / ``construct_dwarf_galaxy_catalogs`` nebular step.
    Rows include ``LOGM_M24_FIDU_CORR`` and ``DELTA_MAG_*`` so ``create_main_data_model``
    (``clean_cat=True``) uses corrected masses without a separate NEBCORR lookup.

    Parameters
    ----------
    input_path : str
        Path to the post-nebular candidate table (NEBCORR FITS). The legacy wider
        ``hidden_dwarf_candidates_qso_mws_scnd.fits`` may be passed explicitly if needed.
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

    cats = safe_read_table(input_path)
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
    "FHALPHA_CONT", "FHALPHA_CONT_IVAR","LOGMSTAR"
]

fastspec_cols = ["SNR_B", "SNR_R", "SNR_Z", "APERCORR", "APERCORR_G", "APERCORR_R", "APERCORR_Z"] 

fastspec_emlines_cols = ["OII_3726_FLUX", "OII_3726_FLUX_IVAR", "OII_3729_FLUX", "OII_3729_FLUX_IVAR", "OIII_4363_FLUX", "OIII_4363_FLUX_IVAR", "HEII_4686_FLUX", "HEII_4686_FLUX_IVAR", "HBETA_FLUX", "HBETA_FLUX_IVAR", "OIII_4959_FLUX", "OIII_4959_FLUX_IVAR", "OIII_5007_FLUX", "OIII_5007_FLUX_IVAR", "HEI_5876_FLUX", "HEI_5876_FLUX_IVAR", "NII_6548_FLUX", "NII_6548_FLUX_IVAR", "HALPHA_FLUX", "HALPHA_FLUX_IVAR", "HALPHA_BROAD_FLUX", "HALPHA_BROAD_FLUX_IVAR", "NII_6584_FLUX", "NII_6584_FLUX_IVAR", "SII_6716_FLUX", "SII_6716_FLUX_IVAR", "SII_6731_FLUX", "SII_6731_FLUX_IVAR", "SIII_9069_FLUX", "SIII_9069_FLUX_IVAR", "SIII_9532_FLUX", "SIII_9532_FLUX_IVAR", "HALPHA_BOXFLUX", "HALPHA_BOXFLUX_IVAR", "HALPHA_EW", "HALPHA_EW_IVAR", "HALPHA_SIGMA", "HALPHA_SIGMA_IVAR"]

fastspec_tot_cols = fastspec_cols +  fastspec_emlines_cols

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
            tab_i = safe_hstack([tab_meta, tab_fastspec, tab_specphot])
    
            #let us only keep objects that are galaxies a
            print(len(tab_i))
            tables.append(tab_i)
        print("---")
    

    #now we stack this all and save it!
    tables = safe_vstack(tables)

    tables.write("/pscratch/sd/v/virajvm/catalog_dr1_dwarfs/iron_fastspec_catalog/iron_fastspec_v3.fits",overwrite=True)
        
    return


def get_fastspec_fit_catalog_V2(chunk_size = 250000):
    '''
    In this function, we combine the relevant columns and healpix fastspec files (VERSION 2 CATALOG)
    IMPORTANT = This function is run before the dwarf catalog is being constructed as it prepares the intermediate file we will be matching to
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
            out_chunks.append(safe_hstack([tab_meta, tab_fastspec]))

        result = safe_vstack(out_chunks)
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
    lowz_trac_cat_f = safe_read_table("/pscratch/sd/v/virajvm/catalog_dr1_dwarfs/iron_lowz_filter_zsucc_zrr03_INT.fits")

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


def _assert_targetid_aligned(tab_ref, tab_other, name_ref, name_other):
    """Require identical length and row-wise TARGETID equality."""
    if len(tab_ref) != len(tab_other):
        raise ValueError(
            f"{name_ref} ({len(tab_ref)} rows) vs {name_other} ({len(tab_other)} rows): length mismatch"
        )
    if not (
        np.asarray(tab_ref["TARGETID"]) == np.asarray(tab_other["TARGETID"])
    ).all():
        raise ValueError(f"TARGETID mismatch between {name_ref} and {name_other}")


def _reassign_sample_from_zcat(main_tab, zcat_tab):
    """
    Set MAIN.SAMPLE from ZCAT targeting bits (main survey + SV), same logic as
    construct_dwarf_galaxy_catalogs.py. Priority: BGS_BRIGHT > BGS_FAINT > ELG.

    Only rows with SAMPLE in {BGS_BRIGHT, BGS_FAINT, ELG} are updated. LOWZ, OTHER,
    and other labels are left unchanged. Rows slated for update but with no matching
    bit keep their current SAMPLE.
    """
    _assert_targetid_aligned(main_tab, zcat_tab, "MAIN", "ZCAT")

    z = zcat_tab
    iron_bgs = z["BGS_TARGET"]
    bgsb_main = (iron_bgs & bgs_mask["BGS_BRIGHT"]) != 0
    bgsb_sv = get_sv_bgs_mask(z, bgs_class="BGS_BRIGHT")
    is_bgsb = bgsb_main | bgsb_sv

    bgsf_main = (iron_bgs & bgs_mask["BGS_FAINT"]) != 0
    bgsf_sv = get_sv_bgs_mask(z, bgs_class="BGS_FAINT")
    is_bgsf = bgsf_main | bgsf_sv

    desi_tgt = z["DESI_TARGET"]
    elg_main = (desi_tgt & desi_mask["ELG"]) != 0
    elg_sv = get_sv_elg_mask(z)
    is_elg = elg_main | elg_sv

    n = len(main_tab)
    reassigned = np.full(n, "", dtype=object)
    reassigned[is_bgsb] = "BGS_BRIGHT"
    reassigned[~is_bgsb & is_bgsf] = "BGS_FAINT"
    reassigned[~is_bgsb & ~is_bgsf & is_elg] = "ELG"

    samples = np.asarray(main_tab["SAMPLE"], dtype=str)
    update_mask = np.isin(samples, ["BGS_BRIGHT", "BGS_FAINT", "ELG"])
    apply_mask = update_mask & (reassigned != "")
    no_bit = update_mask & (reassigned == "")
    n_no_bit = int(np.sum(no_bit))
    if n_no_bit > 0:
        print(
            f"  WARNING: {n_no_bit} rows with SAMPLE in BGS_BRIGHT/BGS_FAINT/ELG have no "
            "BGS_BRIGHT/BGS_FAINT/ELG bit in ZCAT; leaving SAMPLE unchanged"
        )

    out = samples.copy()
    out[apply_mask] = reassigned[apply_mask]
    main_tab["SAMPLE"] = out


def _targetid_dedup_keep_indices(targetids):
    """First occurrence of each TARGETID in table order (stable, vstack order)."""
    tids = np.asarray(targetids)
    _, first_idx = np.unique(tids, return_index=True)
    return np.sort(first_idx)


FIBERTOT_MATCH_RADIUS_ARCSEC = 1.0
_int_v2_stack_for_fibertot_cache = {"key": None, "table": None}


def _int_v2_fibertot_cache_key(folder):
    paths = [os.path.join(folder, b) for b in INT_V2_BASENAMES]
    return tuple(
        (p, os.path.getmtime(p) if os.path.isfile(p) else None) for p in paths
    )


def load_stacked_int_v2_for_fibertot(folder=None):
    """Stack the four subsample _INT_V2.fits tables; dedupe TARGETID (first wins)."""
    if folder is None:
        folder = NEBCORR_DEFAULT_FOLDER
    chunks = []
    required_cols = ("TARGETID", "RA", "DEC", "FIBERTOTFLUX_G", "FIBERTOTFLUX_R")
    for basename in INT_V2_BASENAMES:
        path = os.path.join(folder, basename)
        if not os.path.isfile(path):
            raise FileNotFoundError(
                f"INT_V2 catalog not found (required for TRACTOR FIBERTOTFLUX): {path}"
            )
        tab = safe_read_table(path)
        missing = [c for c in required_cols if c not in tab.colnames]
        if missing:
            raise ValueError(f"{path} missing columns {missing}")
        chunks.append(tab[list(required_cols)])
    stacked = vstack(chunks)
    keep = _targetid_dedup_keep_indices(stacked["TARGETID"])
    stacked = stacked[keep]
    return stacked


def _get_cached_stacked_int_v2_for_fibertot(folder=None):
    if folder is None:
        folder = NEBCORR_DEFAULT_FOLDER
    key = _int_v2_fibertot_cache_key(folder)
    if _int_v2_stack_for_fibertot_cache["key"] == key:
        return _int_v2_stack_for_fibertot_cache["table"]
    tab = load_stacked_int_v2_for_fibertot(folder=folder)
    _int_v2_stack_for_fibertot_cache["key"] = key
    _int_v2_stack_for_fibertot_cache["table"] = tab
    return tab


def add_fibertotflux_from_int_v2(tractor_tab, tot_cat, int_v2_folder=None):
    """
    Fill FIBERTOTFLUX_G and FIBERTOTFLUX_R by matching to stacked _INT_V2.fits
    catalogs (sky match + TARGETID agreement within FIBERTOT_MATCH_RADIUS_ARCSEC).
    """
    if len(tractor_tab) != len(tot_cat):
        raise ValueError("Incorrect lengths for tractor_tab and tot_cat tables!")

    int_v2_ref = _get_cached_stacked_int_v2_for_fibertot(folder=int_v2_folder)
    idx, d2d, _ = match_c_to_catalog(
        c_cat=tot_cat,
        catalog_cat=int_v2_ref,
        c_ra="RA_TARGET",
        c_dec="DEC_TARGET",
        catalog_ra="RA",
        catalog_dec="DEC",
    )
    close = d2d.arcsec < FIBERTOT_MATCH_RADIUS_ARCSEC
    ref_tid = np.asarray(int_v2_ref["TARGETID"].data)[idx]
    cat_tid = np.asarray(tot_cat["TARGETID"].data)
    # mismatch = close & (ref_tid != cat_tid)

    #it is possible that there is not a perfect match as sometimes there are same sources in DESI with two different targetids.
    #matching by distance (within 1 arcsec is good enough for fiber tot flux purposes!
    # if np.any(mismatch):
    #     i = int(np.flatnonzero(mismatch)[0])
    #     raise ValueError(
    #         "FIBERTOT INT_V2 cross-match: TARGETID mismatch for a close sky match "
    #         f"(row {i}: catalog TARGETID {cat_tid[i]} vs INT_V2 TARGETID {ref_tid[i]}, "
    #         f"d2d={d2d[i].arcsec:.4f} arcsec)"
    #     )

    g = np.full(len(tractor_tab), np.nan, dtype=np.float32)
    r = np.full(len(tractor_tab), np.nan, dtype=np.float32)
    if np.any(close):
        g[close] = np.asarray(int_v2_ref["FIBERTOTFLUX_G"].data[idx[close]], dtype=np.float32)
        r[close] = np.asarray(int_v2_ref["FIBERTOTFLUX_R"].data[idx[close]], dtype=np.float32)

    tractor_tab["FIBERTOTFLUX_G"] = g
    tractor_tab["FIBERTOTFLUX_R"] = r
    n_close = int(np.sum(close))
    print(
        f"FIBERTOTFLUX from INT_V2: matched {n_close}/{len(tractor_tab)} within "
        f"{FIBERTOT_MATCH_RADIUS_ARCSEC} arcsec"
    )
    return tractor_tab


def combine_hdus(hdu_list, base_path="/pscratch/sd/v/virajvm/desi_dwarf_catalogs/dr1/v1.0/temp_cats",
                 output_file="/pscratch/sd/v/virajvm/desi_dwarf_catalogs/dr1/v1.0/desi_dwarfs_combined.fits",
                 extra_prefixes=None):
    """
    Combine multiple HDUs (Astropy tables) into a single multi-extension FITS file.

    After stacking clean + shreds (+ optional extras) with identical row order for
    MAIN, ZCAT, TRACTOR, and SPEC_DERIVED, this function (1) reassigns MAIN ``SAMPLE``
    from ZCAT targeting bits with priority BGS_BRIGHT > BGS_FAINT > ELG for rows
    whose SAMPLE is BGS_BRIGHT, BGS_FAINT, or ELG; (2) deduplicates on TARGETID (first row
    kept); (3) runs ``finalize_main_hdu`` on MAIN. LOWZ and OTHER are not relabeled.

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

    stacked = {}
    reprocess_tab = None

    # Phase A: stack clean + shreds (+ extras) per HDU into aligned tables
    for hdu_name in hdu_list:
        if hdu_name in ["REPROCESS_PHOTO"]:
            shred_fname = os.path.join(base_path, f"shreds_{hdu_name}_hdu.fits")
            print(f"Reading {shred_fname}...")
            reprocess_tab = safe_read_table(shred_fname)
        else:
            shred_fname = os.path.join(base_path, f"shreds_{hdu_name}_hdu.fits")
            clean_fname = os.path.join(base_path, f"clean_{hdu_name}_hdu.fits")

            print(f"Reading {shred_fname}...")
            print(f"Reading {clean_fname}...")

            clean_tab = safe_read_table(clean_fname)
            shred_tab = safe_read_table(shred_fname)

            tables_to_stack = [clean_tab, shred_tab]

            for prefix in extra_prefixes:
                extra_fname = os.path.join(base_path, f"{prefix}_{hdu_name}_hdu.fits")
                if os.path.exists(extra_fname):
                    print(f"Reading {extra_fname}...")
                    tables_to_stack.append(safe_read_table(extra_fname))
                else:
                    print(f"Skipping {extra_fname} (not found)")

            stacked[hdu_name] = safe_vstack(tables_to_stack)

    # Phase B: SAMPLE from ZCAT bits, dedupe TARGETID, finalize MAIN
    science_names = [n for n in hdu_list if n != "REPROCESS_PHOTO"]
    main_pre = stacked["MAIN"]
    _assert_targetid_aligned(main_pre, stacked["ZCAT"], "MAIN", "ZCAT")
    for sn in science_names:
        if sn != "MAIN":
            _assert_targetid_aligned(main_pre, stacked[sn], "MAIN", sn)

    n_before = len(main_pre)
    print("combine_hdus: Reassigning SAMPLE from ZCAT targeting bits (BGS_BRIGHT > BGS_FAINT > ELG)")
    _reassign_sample_from_zcat(stacked["MAIN"], stacked["ZCAT"])

    tids_pre = np.asarray(stacked["MAIN"]["TARGETID"])
    _, tid_counts = np.unique(tids_pre, return_counts=True)
    n_targetids_with_duplicates = int(np.sum(tid_counts > 1))

    keep_idx = _targetid_dedup_keep_indices(stacked["MAIN"]["TARGETID"])
    n_after = len(keep_idx)
    n_rows_dropped = n_before - n_after
    if n_rows_dropped > 0:
        print(
            "combine_hdus: TARGETID deduplication — "
            f"dropped {n_rows_dropped} row(s) ({n_before} -> {n_after} rows); "
            f"{n_targetids_with_duplicates} distinct TARGETID(s) appeared more than once "
            "(first occurrence in stack order kept)"
        )
    else:
        print(
            f"combine_hdus: TARGETID deduplication — no duplicates "
            f"({n_before} rows, {len(tid_counts)} unique TARGETIDs)"
        )

    for sn in science_names:
        stacked[sn] = stacked[sn][keep_idx]

    print("combine_hdus: finalize_main_hdu (associated fibers, DIST_SOURCE, …)")
    stacked["MAIN"] = finalize_main_hdu(stacked["MAIN"])

    hdu_tables = []
    for hdu_name in hdu_list:
        if hdu_name in ["REPROCESS_PHOTO"]:
            hdu_tables.append(reprocess_tab)
        else:
            hdu_tables.append(stacked[hdu_name])

    # we ignore the reprocess_cat from comparison as that as a different number of rows by construction

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
    main_cat = safe_read_table(file_path, hdu="MAIN")
    main_tgids = main_cat["TARGETID"].data
    n_objects = len(main_tgids)

    # Load HDF5 coefficients
    h5_path = "/pscratch/sd/v/virajvm/catalog_dr1_dwarfs/iron_spectra/spectra_files/desi_dr1_dwarf_catalog_nnmf_pca_NEW.h5"
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
    spec_umap_2d = np.load("/pscratch/sd/v/virajvm/catalog_dr1_dwarfs/iron_spectra/spectra_files/desi_dwarfs_umap_nnmf_and_pca_NEW.npy")

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
    main_cat = safe_read_table(file_path, hdu="MAIN")
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


def add_too_bright_maskbit(cat_path, bit=18, mag_cut=-18.5):
    """
    Flag sources whose corrected absolute g-band magnitude is brighter than
    *mag_cut* (default Mg < -18.5) by setting *bit* in DWARF_MASKBIT.

    The absolute magnitude is computed from the fully corrected z=0 SDSS
    continuum photometry (via ``_apply_delta_mag_corrections``) and the
    luminosity distance::

        d_pc  = LUMI_DIST_MPC * 1e6
        Mg    = mag_g_corrected - 5 * log10(d_pc) + 5

    Parameters
    ----------
    cat_path : str
        Path to the multi-extension FITS catalog.
    bit : int, optional
        Bit index to set (default 18).
    mag_cut : float, optional
        Absolute magnitude threshold (default -18.5).  Sources *brighter*
        (more negative) than this value are flagged.
    """
    main_cat = safe_read_table(cat_path, hdu="MAIN")

    mag_g_corr, _ = _apply_delta_mag_corrections(main_cat)

    dist_mpc = np.asarray(main_cat["LUMI_DIST_MPC"], dtype=float)
    dist_pc = dist_mpc * 1.0e6

    valid = np.isfinite(dist_pc) & (dist_pc > 0) & np.isfinite(mag_g_corr)
    abs_mag_g = np.full(len(main_cat), np.nan)
    abs_mag_g[valid] = mag_g_corr[valid] - 5.0 * np.log10(dist_pc[valid]) + 5.0

    too_bright = np.isfinite(abs_mag_g) & (abs_mag_g < mag_cut)

    dwarf_maskbits = np.asarray(main_cat["DWARF_MASKBIT"], dtype=np.int64)
    dwarf_maskbits[too_bright] |= np.int64(1) << bit
    main_cat["DWARF_MASKBIT"] = dwarf_maskbits

    if "MSTAR_MASKBIT" in main_cat.colnames and bit == 18:
        mstar = np.asarray(main_cat["MSTAR_MASKBIT"], dtype=np.int64)
        mstar[too_bright] |= np.int64(1) << 1
        main_cat["MSTAR_MASKBIT"] = mstar.astype(np.int32)

    n_total = len(main_cat)
    n_flagged = int(too_bright.sum())
    print(f"DWARF_MASKBIT bit {bit} (Mg < {mag_cut}): "
          f"flagged {n_flagged}/{n_total} ({100.0 * n_flagged / n_total:.1f}%) total")

    samples = np.asarray(main_cat["SAMPLE"], dtype=str)
    for sample in sorted(set(samples)):
        in_sample = samples == sample
        n_samp = int(in_sample.sum())
        n_flag = int((too_bright & in_sample).sum())
        pct = 100.0 * n_flag / n_samp if n_samp > 0 else 0.0
        print(f"  {sample}: {n_flag}/{n_samp} ({pct:.1f}%)")

    buf = BytesIO()
    main_cat.write(buf, format="fits")
    buf.seek(0)
    main_hdu = fits.open(buf)[1]
    main_hdu.name = "MAIN"
    main_hdu.add_checksum()

    with fits.open(cat_path, mode="update") as hdul:
        hdul[1] = main_hdu
        hdul.flush()


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
    main_cat = safe_read_table(cat_path, hdu="MAIN")
    fspec_cat = safe_read_table(cat_path, hdu=DWARF_CATALOG_SPEC_HDU)
    spec_cat = safe_read_table(cat_path, hdu="SPECTRA_TEMPLATE")

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

        blank_val = meta.get("blank_value", None)
        if blank_val is not None:
            col_data = np.asarray(main_cat[col], dtype=float)
            bad = np.isnan(col_data)
            col_data[bad] = blank_val
            main_cat[col] = col_data

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


def add_model_photometry_to_fastspec(
    cat_path,
    model_phot_dir="/pscratch/sd/v/virajvm/catalog_dr1_dwarfs",
    gal_types=("LOWZ", "BGS_FAINT", "BGS_BRIGHT", "ELG", "OTHER"),
    verbose=True,
):
    """
    Read pre-computed fastspec model photometry from
    model_photometry_diffs_{gal_type}.fits files, cross-match by TARGETID,
    and append 10 model-magnitude columns to the SPEC_DERIVED HDU of the
    multi-extension catalog at *cat_path*.

    New SPEC_DERIVED columns:
        MAG_{G,R}_DECAM_MODEL_NOEMI   - DECam model mags, continuum only
        MAG_{G,R}_DECAM_MODEL_WEMI    - DECam model mags, continuum + emission
        MAG_{G,R}_BASS_MODEL_WEMI     - BASS  model mags, continuum + emission
        MAG_{G,R}_SDSS_MODEL_NOEMI    - SDSS  model mags, continuum only
        MAG_{G,R}_SDSS_Z0_MODEL_NOEMI - SDSS  z=0 rest-frame model mags, continuum only
    """
    print("=" * 60)
    print("Adding fastspec model photometry columns to SPEC_DERIVED HDU")
    print("=" * 60)

    _COL_MAP = {
        "g_model_no_emi":   "MAG_G_DECAM_MODEL_NOEMI",
        "r_model_no_emi":   "MAG_R_DECAM_MODEL_NOEMI",
        "g_model_w_emi":    "MAG_G_DECAM_MODEL_WEMI",
        "r_model_w_emi":    "MAG_R_DECAM_MODEL_WEMI",
        "g_bass_w_emi":     "MAG_G_BASS_MODEL_WEMI",
        "r_bass_w_emi":     "MAG_R_BASS_MODEL_WEMI",
        "g_sdss_no_emi":    "MAG_G_SDSS_MODEL_NOEMI",
        "r_sdss_no_emi":    "MAG_R_SDSS_MODEL_NOEMI",
        "g_sdss_z0_no_emi": "MAG_G_SDSS_Z0_MODEL_NOEMI",
        "r_sdss_z0_no_emi": "MAG_R_SDSS_Z0_MODEL_NOEMI",
    }

    # ── 1. Read and combine model photometry tables ────────────────────
    tables = []
    for gal_type in gal_types:
        path = os.path.join(model_phot_dir, f"model_photometry_diffs_{gal_type}.fits")
        if not os.path.exists(path):
            print(f"  WARNING: {path} not found, skipping {gal_type}")
            continue
        tab = safe_read_table(path)
        if verbose:
            print(f"  Loaded {len(tab)} rows from {path}")
        tables.append(tab)

    if len(tables) == 0:
        print("  ERROR: No model photometry files found. Aborting.")
        return

    model_phot = safe_vstack(tables)

    # De-duplicate on TARGETID (keep first occurrence)
    _, unique_idx = np.unique(np.asarray(model_phot["TARGETID"]), return_index=True)
    model_phot = model_phot[np.sort(unique_idx)]
    if verbose:
        print(f"  Combined model photometry table: {len(model_phot)} unique TARGETIDs")

    # ── 2. Read SPEC_DERIVED HDU from the catalog ──────────────────────────
    fspec_cat = safe_read_table(cat_path, hdu=DWARF_CATALOG_SPEC_HDU)
    n_objects = len(fspec_cat)
    cat_tids = np.asarray(fspec_cat["TARGETID"])

    if verbose:
        print(f"  SPEC_DERIVED HDU has {n_objects} rows")

    # ── 3. Build TARGETID lookup and fill columns ──────────────────────
    model_tid_to_row = {int(t): i for i, t in enumerate(model_phot["TARGETID"])}

    for old_col, new_col in _COL_MAP.items():
        arr = np.full(n_objects, np.nan, dtype=np.float64)
        src = np.asarray(model_phot[old_col], dtype=np.float64)
        for j, tid in enumerate(cat_tids):
            row = model_tid_to_row.get(int(tid))
            if row is not None:
                arr[j] = src[row]
        fspec_cat[new_col] = arr

    n_matched = int(np.sum(np.isfinite(fspec_cat["MAG_G_DECAM_MODEL_NOEMI"])))
    if verbose:
        print(f"  Matched {n_matched}/{n_objects} objects to model photometry")

    # ── 4. Rewrite catalog (SPEC_DERIVED only) ─────────────────────────────
    # Same as compute_emission_subtracted_photo_errors: mode="update" after
    # resizing an extension breaks verify on the following HDU (here el. 5).
    # MAIN must use table_to_hdu (not hdu.copy()) so VLAs e.g. ASSOCIATED_TARGETIDS
    # do not hit "Could not find heap data" on copy.
    fspec_hdu_new = fits.table_to_hdu(fspec_cat)
    fspec_hdu_new.name = DWARF_CATALOG_SPEC_HDU
    fspec_hdu_new.add_checksum()

    cat_abs = os.path.abspath(cat_path)
    cat_dir = os.path.dirname(cat_abs) or "."
    fd, tmp_path = tempfile.mkstemp(
        suffix=".fits", prefix="model_phot_", dir=cat_dir
    )
    os.close(fd)
    try:
        with fits.open(cat_abs, memmap=False) as hdul:
            hdu_names = [hdu.name for hdu in hdul]
            main_idx = hdu_names.index("MAIN")
            fspec_idx = hdu_names.index(DWARF_CATALOG_SPEC_HDU)
            main_tab = safe_read_table(cat_abs, hdu="MAIN")
            main_hdu_preserved = fits.table_to_hdu(main_tab)
            main_hdu_preserved.name = "MAIN"
            main_hdu_preserved.add_checksum()
            new_hdus = []
            for i, hdu in enumerate(hdul):
                if i == main_idx:
                    new_hdus.append(main_hdu_preserved)
                elif i == fspec_idx:
                    new_hdus.append(fspec_hdu_new)
                else:
                    new_hdus.append(hdu.copy())
            new_hdul = fits.HDUList(new_hdus)
            new_hdul[0].add_checksum()
            new_hdul.writeto(tmp_path, overwrite=True)
        os.replace(tmp_path, cat_abs)
    except BaseException:
        if os.path.isfile(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass
        raise

    new_cols_str = ", ".join(_COL_MAP.values())
    print(f"Updated {cat_path}:")
    print(f"  SPEC_DERIVED HDU: added {new_cols_str}")
    print("=" * 60)


# _load_nebcorr_delta_mag_table and add_delta_mag_to_fastspec
# are imported from mass_and_photo_corrections

if __name__ == '__main__':

    save_path = "/pscratch/sd/v/virajvm/desi_dwarf_catalogs/dr1/v1.0/temp_cats"

    #need to add to mstar_maskbit ..     
    # iii) flagging large k correction or large difference between g and r template corrections, as that will point to something suspicious ...

    # TODO: update the image cutouts download function to selectively update the cutouts for dwarfs that do not exist in the catalog. 

    process_shreds = True
    process_clean = True
    compute_mstar_err = False
    add_model_phot = True
    process_qso_scnd = True
    process_post_hdu = True

    #make sure the get_fastspec_fit_catalog_V2 function is run before hand in case there are any new columns added
    process_fastspec=True

    add_sfrs_zmet = False

    main_cat_outpath = "/pscratch/sd/v/virajvm/desi_dwarf_catalogs/dr1/v1.0/desi_dr1_dwarf_catalog.fits"

    if process_shreds:
        #loading the shredded catalogs!
        print("Reading ELG shreds!")
        elg_shred = safe_read_table(f"/pscratch/sd/v/virajvm/catalog_dr1_dwarfs/iron_photometry/iron_ELG_shreds_catalog_w_aper_mags.fits")
        elg_shred = consolidate_new_photo(elg_shred,sample="ELG",flag_cog_nan_always=False)
        print("=="*10)
    
        print("Reading BGS Bright shreds!")
        bgsb_shred = safe_read_table(f"/pscratch/sd/v/virajvm/catalog_dr1_dwarfs/iron_photometry/iron_BGS_BRIGHT_shreds_catalog_w_aper_mags.fits")
        bgsb_shred = consolidate_new_photo(bgsb_shred,sample="BGS_BRIGHT",flag_cog_nan_always=True)
        print("=="*10)
        
        print("Reading BGS Faint shreds!")
        bgsf_shred = safe_read_table(f"/pscratch/sd/v/virajvm/catalog_dr1_dwarfs/iron_photometry/iron_BGS_FAINT_shreds_catalog_w_aper_mags.fits")
        bgsf_shred = consolidate_new_photo(bgsf_shred,sample="BGS_FAINT",flag_cog_nan_always=True)
        print("=="*10)
    
        print("Reading LOWZ shreds!")
        lowz_shred = safe_read_table(f"/pscratch/sd/v/virajvm/catalog_dr1_dwarfs/iron_photometry/iron_LOWZ_shreds_catalog_w_aper_mags.fits")
        lowz_shred = consolidate_new_photo(lowz_shred,sample="LOWZ",flag_cog_nan_always=True)
        print("=="*10)

        print("Reading SGA shreds!")

        temp = Table.read("/pscratch/sd/v/virajvm/catalog_dr1_dwarfs/iron_photometry/iron_SGA_sga_catalog_w_aper_mags.fits")

        sga_all = safe_read_table("/pscratch/sd/v/virajvm/catalog_dr1_dwarfs/iron_photometry/iron_SGA_sga_catalog_w_aper_mags.fits")
        sga_all = consolidate_new_photo(sga_all,sample="SGA",flag_cog_nan_always=True)
        print("=="*10)

    
        # --- remove extra columns from SGA before stacking ---
        extra_cols = set(sga_all.colnames) - set(lowz_shred.colnames)
        if extra_cols:
            print(f"Removing {len(extra_cols)} extra columns from SGA: {extra_cols}")
            sga_all.remove_columns(list(extra_cols))

        # --- add missing columns to SGA that exist in LOWZ ---
        missing_cols = set(lowz_shred.colnames) - set(sga_all.colnames)
        if missing_cols:
            print(f"Adding {len(missing_cols)} missing columns to SGA with defaults: {missing_cols}")
            for col in missing_cols:
                dtype = lowz_shred[col].dtype
                if np.issubdtype(dtype, np.floating):
                    sga_all[col] = np.full(len(sga_all), np.nan, dtype=dtype)
                elif np.issubdtype(dtype, np.integer):
                    sga_all[col] = np.full(len(sga_all), -99, dtype=dtype)
                elif dtype.kind in ('U', 'S', 'O'):
                    sga_all[col] = np.full(len(sga_all), "", dtype=dtype)
                else:
                    sga_all[col] = np.full(len(sga_all), 0, dtype=dtype)

        # reorder columns to match LOWZ
        sga_all = sga_all[lowz_shred.colnames]

        
    
        tot_shred = safe_vstack([ bgsb_shred, bgsf_shred, lowz_shred, elg_shred, sga_all])

        
    
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
            get_fastspec_matched_catalog(tot_shred, save_path + "/shreds_SPEC_DERIVED_hdu.fits", match_method="TARGETID")
            
        ##get the other hdus

    if process_clean:
        ##get the clean catalog stuff now!!
        clean_cat = safe_read_table("/pscratch/sd/v/virajvm/catalog_dr1_dwarfs/desi_y1_dwarf_clean_catalog_v5.fits")

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
            get_fastspec_matched_catalog(clean_cat, save_path + "/clean_SPEC_DERIVED_hdu.fits", match_method="TARGETID")

    if process_qso_scnd:
        qso_scnd_input = "/pscratch/sd/v/virajvm/catalog_dr1_dwarfs/iron_other_qso_scnd_candidates_INT_V2_NEBCORR.fits"
        qso_scnd_cat = load_and_filter_qso_scnd_candidates(qso_scnd_input, snr_threshold=3.0)

        print("Creating the QSO/SCND main hdu")
        qso_scnd_cat, qso_scnd_entire = create_main_data_model(
            qso_scnd_cat, save_path + "/qso_scnd_MAIN_hdu.fits", clean_cat=True
        )

        create_tractor_data_model(qso_scnd_entire, save_path + "/qso_scnd_TRACTOR_hdu.fits")
        create_zcat_data_model(qso_scnd_entire, save_path + "/qso_scnd_ZCAT_hdu.fits")

        if process_fastspec:
            print("Creating the QSO/SCND fastspecfit hdu")
            get_fastspec_matched_catalog(qso_scnd_cat, save_path + "/qso_scnd_SPEC_DERIVED_hdu.fits", match_method="TARGETID")

    #then we consolidate it all into a multi-ext file!
    #make sure the REPROCESS_PHOTO_CAT is also last in the below list!
    combine_hdus(["MAIN", "ZCAT", "TRACTOR", DWARF_CATALOG_SPEC_HDU, "REPROCESS_PHOTO"],
                 base_path="/pscratch/sd/v/virajvm/desi_dwarf_catalogs/dr1/v1.0/temp_cats",
                 output_file=main_cat_outpath,
                 extra_prefixes=["qso_scnd"] if process_qso_scnd else [])

    if compute_mstar_err:
        print("Computing emission-subtracted photometry and stellar mass errors")
        compute_emission_subtracted_photo_errors(main_cat_outpath)

    if add_model_phot:
        add_model_photometry_to_fastspec(main_cat_outpath)
        add_delta_mag_to_fastspec(main_cat_outpath)

    if process_post_hdu:
        ##add the spectra NMF+PCA information!!
        create_spectra_hdu(main_cat_outpath)

        ##add image SSL UMAP + similarity information
        # create_image_ssl_hdu(main_cat_outpath)

        #update the dwarf_maskbit with some weird spectra masks
        add_wrong_redrock_maskbit(main_cat_outpath, main_datamodel)

    # TODO: need to confirm if the pruning and appending in ssl datasets is working fine and the data exists!

    consolidate_associated_fiber_properties(main_cat_outpath)

    if add_sfrs_zmet:
        add_sfr_halpha_to_spec_derived(main_cat_outpath)





























    

    

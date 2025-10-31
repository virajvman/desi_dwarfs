'''
Functions where preferred, best photometry column is created along with photometry maskbits to identify reliable measurements. This is also the script where we produce the final, multi-extension fits files as the final catalog output.
'''
import numpy as np
import astropy.io.fits as fits
import astropy.units as u
from astropy.table import Table, vstack, join, hstack
from shred_photometry_maskbits import cog_mag_converge, cog_nan_mask, cog_curve_decrease, bad_colors, iffy_tractor_model
from io import BytesIO
from shred_photometry_maskbits import create_shred_maskbits_from_dict, print_maskbit_statistics
import os
import glob

from desi_lowz_funcs import get_stellar_mass, match_c_to_catalog, match_fastspec_catalog, get_stellar_mass_mia

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

def make_catalog_unmasked(cat):
    """
    Return a new Table where all MaskedColumns are replaced by regular ndarray columns.
    Masked entries are filled with np.nan.
    """
    new_cat = cat.copy()
    for col in new_cat.colnames:
        c = new_cat[col]
        if hasattr(c, "mask"):   # MaskedColumn
            new_cat[col] = np.asarray(c.filled(np.nan))
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


def org_tractor_is_likely_good(cat):
    '''
    Function that identifies the subset of sources tha are likely just pure blends where the original tractor model is all good!
    Oh these will also be sources where nothing is detected because num_tractor_sources_final gets only if the cog part is run
    '''

    ntractor = np.array(cat["NUM_TRACTOR_SOURCES_FINAL"])
    
    #this cirterion will be good for the sources that have good significance
    #however, in addition to just being a single source, we also want to make sure the COG mag is not better as bad photometry
    likely_pure_blend = (np.array(cat["PCNN_FRAGMENT"]) < 0.25) | ( (ntractor <= 1) | np.isnan(ntractor) )

    #sometimes no sources are listed if no smooth component for parent galaxy isolate is found.
    #this we do <= 1 or np.nan

    print(type(likely_pure_blend))
    print(type(cat["APER_SOURCE_ON_ORG_BLOB"]))
    print(type(cat["COG_NUM_SEG_SMOOTH"]))
    print(type(cat["COG_NUM_SEG"]))
    
    return likely_pure_blend


def revert_back_to_org_tractor(cat):
    '''
    Function that identifies the subset of sources tha are likely just pure blends where the original tractor model is all good!
    Oh these will also be sources where nothing is detected because num_tractor_sources_final gets only if the cog part is run
    '''

    likely_pure_blend = org_tractor_is_likely_good(cat)

    #if the source was soo faint that it was not on the original blob!
    cog_was_not_run = (cat["APER_SOURCE_ON_ORG_BLOB"] == 0)

    #for the very faint sources, we need to check too. if no cog segment was detected
    cog_seg_not_detected = (cat["COG_NUM_SEG_SMOOTH"] == 0) | (cat["COG_NUM_SEG"] == 0)

    return likely_pure_blend | cog_was_not_run | cog_seg_not_detected


def add_best_mags(catalog, bands=("G", "R", "Z")):
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
    
    prefer_org_tractor_mag = revert_back_to_org_tractor(catalog)

    print(prefer_org_tractor_mag[:5])

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
    source[prefer_org_tractor_mag] = "TRACTOR_ORIGINAL"

    print("Need to include the VI column that only does the aperture no subtract photo")
    
    catalog[f"MAG_TYPE"] = source

    return catalog


def consolidate_new_photo(catalog,plot=False,sample=None):
    '''
    In this function, we consolidate the different quantities using the over de-deblending criterion. 
    Note that the fraction of the aperture that is masked is the initial aperture and not the final COG aperture    
    '''

    catalog = make_catalog_unmasked(catalog)

    # this was due to a bug I had in my code
    # if "APERFRAC_R4_IN_IMG_ISOLATE" in catalog.colnames:
    #     pass
    # else:
    #     print("NEED TO REMOVE THIS IN THE NEXT RUN ITERATION!")
    #     catalog["APERFRAC_R4_IN_IMG_ISOLATE"] = np.array(catalog["APERFRAC_R4_IN_IMG_NO_ISOLATE"]).copy()

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
                               "TRACTOR_ONLY_APER_CEN_RADEC", "TRACTOR_ONLY_APER_PARAMS","TRACTOR_APER_CEN_MASKED","NUM_TRACTOR_SOURCES", "TRACTOR_MU_R_SIZES"]

    all_keys_to_combine = org_keys_to_combine + tractor_keys_to_combine

    apply_no_isolate_mask = likely_over_deblended(catalog["Z"].data, catalog["APER_R2_MU_R_ISLAND_TRACTOR"].data)
    #when this is true, we use no_isolate photometry. Otherwise, we use the isolate photometry
    
    pairs = {}

    for ki in all_keys_to_combine:
        pairs[ki] = ( catalog[ki + "_ISOLATE" ].data , catalog[ki + "_NO_ISOLATE"].data )
        
    for newcol, (w_iso, no_iso) in pairs.items():
        catalog[newcol + "_FINAL"] = combine_arrays(no_iso, w_iso, apply_no_isolate_mask)

    #add the pcnn column, load the appropriate sample
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

    #need to make a column indicating whether the final photo was with isolate or no isolate
    catalog["ISOLATE_MASK_LIKELY_SHREDDING"] = apply_no_isolate_mask

    catalog = add_best_mags(catalog)

    #add the photomaskbit column
    if sample == "SGA":
        #not to apply maskbit=12 as these are objects already in SGA!
       bitmasks_list = [0,1,2,3,4,5,6,7,8,9,10]
    else:
       bitmasks_list = [0,1,2,3,4,5,6,7,8,9,10,12]
    
    print("Adding the photo maskbits")
    photo_maskbits =  create_shred_maskbits_from_dict(catalog, bitmasks_to_apply = bitmasks_list, verbose=True)
    catalog["PHOTO_MASKBIT"] = photo_maskbits
    
    #now for the subset that we think has robust photometry, we want to ignore majority of their maskbits above and just start again!
    #However, we only want to do this for sources that have a failed COG photometry?
    using_org_tractor = revert_back_to_org_tractor(catalog)
    
    print(f"Fraction of sources where org trac is likely good = {np.sum(using_org_tractor)/len(catalog)}")
    #then we need to update some of the maskbits accordingly: bad color, iffy tractor model, we now do not care about the star!!

    if sample == "SGA":
        #not to apply maskbit=12 as these are objects already in SGA!
       bitmasks_list = [7,11,13]
    else:
       bitmasks_list = [7,11,12,13]
        
    print("Updating the maskbits to reflect some objects reverted to original Tractor photometry")
    only_trac_maskbits = create_shred_maskbits_from_dict(catalog, bitmasks_to_apply = bitmasks_list, verbose=True)

    current_maskbits = np.array(catalog["PHOTO_MASKBIT"])
    current_maskbits[using_org_tractor] = only_trac_maskbits[using_org_tractor]

    #updating this in the catalog
    catalog["PHOTO_MASKBIT"] = current_maskbits

    #now print the summary statistics of the consolidated photometry!!
    print_maskbit_statistics(current_maskbits)

    print("TODO: ADD WAYS TO COMBINE THE SIZE OF THE SYSTEM, AND FINAL RA,DEC BASED ON REVERTING BACK TO TRACTOR OR SIMPLE ETC. OR NOT")
    print("WILL BE USEFUL FOR VI'ing! E.G., not having any consolidated size and ra/dec info for the simple photo objects.")
    print("referring to the tractor based one")
    print("add sizes for the simple aperture based light one?")
    print("It seems like cog based sizes might not be best if not converging fast,so doing the light weighted ones on the aperture would be good!")

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

from data_model import tractor_datamodel, fastspec_hdu_datamodel, main_datamodel


def consolidate_positions_and_shapes(catalog):
    """
    Consolidate RA, DEC, and shape parameters based on MAG_TYPE.
    """

    print("Adding the consolidated RA,DEC, and SHAPE columns")
    
    # Extract MAG_TYPE as string array
    mag_type = np.array(catalog["MAG_TYPE"].data).astype(str)

    # Get input coordinate sets
    ra_aper_cen, dec_aper_cen = catalog["APER_CEN_RADEC_FINAL"].data[:, 0], catalog["APER_CEN_RADEC_FINAL"].data[:, 1]
    ra_trac_cen, dec_trac_cen = catalog["TRACTOR_ONLY_APER_CEN_RADEC_FINAL"].data[:, 0], catalog["TRACTOR_ONLY_APER_CEN_RADEC_FINAL"].data[:, 1]
    ra_org, dec_org = catalog["RA"].data, catalog["DEC"].data

    #NOTE: in the updated photometry, the semi-major axis is based on the g+r+z image.
    #this might be different from the tractor based shape_r which might be based on r band?

    # Get shape parameters
    aper_params = catalog["APER_PARAMS_FINAL"].data                   # shape (N, 3)
    trac_aper_params = catalog["TRACTOR_ONLY_APER_PARAMS_FINAL"].data # shape (N, 3)
    #converting the semi-major axis in pixels to arcseconds!
    aper_params[:, 0] *= 0.262
    trac_aper_params[:, 0] *= 0.262
    org_aper_params = np.vstack([
        catalog["SHAPE_R"].data,
        catalog["BA"].data,
        catalog["PHI"].data
    ]).T.astype(np.float32)                                           # shape (N, 3)

    print("TODO: check that the SHAPE_R, BA, PHI columns are consistent with the aperture ones, especially PHI.")
    print("TODO: add the VI + aper r3 based stuff here too")
        
    # Prepare output arrays
    n = len(mag_type)
    ra_final = np.full(n, np.nan, dtype=np.float64)
    dec_final = np.full(n, np.nan, dtype=np.float64)
    shape_final = np.full((n, 3), np.nan, dtype=np.float32)
    phot_update_final = np.ones(len(catalog))

    # Masks for each type
    mask_cog_simple = np.isin(mag_type, ["COG", "SIMPLE"])
    mask_trac_based = (mag_type == "TRACTOR_BASED")
    mask_trac_org   = (mag_type == "TRACTOR_ORIGINAL")

    # Assign values
    ra_final[mask_cog_simple]  = ra_aper_cen[mask_cog_simple]
    dec_final[mask_cog_simple] = dec_aper_cen[mask_cog_simple]
    shape_final[mask_cog_simple] = aper_params[mask_cog_simple]

    ra_final[mask_trac_based]  = ra_trac_cen[mask_trac_based]
    dec_final[mask_trac_based] = dec_trac_cen[mask_trac_based]
    shape_final[mask_trac_based] = trac_aper_params[mask_trac_based]

    ra_final[mask_trac_org]  = ra_org[mask_trac_org]
    dec_final[mask_trac_org] = dec_org[mask_trac_org]
    shape_final[mask_trac_org] = org_aper_params[mask_trac_org]

    phot_update_final[mask_trac_org] = 0

    # Add to catalog, and over-writing the original RA,DEC columns. The original RA,DEC columns are stored in RA_TARGET, DEC_TARGET
    catalog["RA"] = ra_final
    catalog["DEC"] = dec_final
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

    print("TODO: add Z_CMB to main hdu, also add MU_R and sizes.")
    
    
    catalog["RA_TARGET"] = catalog["RA"].copy()
    catalog["DEC_TARGET"] = catalog["DEC"].copy()
    catalog.rename_column("DIST_MPC_FIDU", "LUMI_DIST_MPC")

    catalog["MAG_G_TARGET"]  = catalog["MAG_G"].copy()
    catalog["MAG_R_TARGET"]  = catalog["MAG_R"].copy()
    catalog["MAG_Z_TARGET"]  = catalog["MAG_Z"].copy()

    #make sure none of the columns are masked columns to avoid subtle, unknown bugs!!

    if clean_cat:
        #we need to remove some of the SGA columns as they are masked and it makes some issues! 
        #there are SGA columns here because there are 47 objects in SGA catalog that have robust tractor photometry, so we just put them in here
        in_sga_2020 = np.zeros(len(catalog))
        print(f"{np.sum(~catalog['SGA_RA_MOMENT'].data.mask)} objects in clean that are in SGA-2020")
        in_sga_2020[~catalog["SGA_RA_MOMENT"].data.mask]  = 1
        catalog["IN_SGA_2020"] = in_sga_2020.astype(bool)

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

        catalog["MAG_TYPE"] = np.full(len(catalog), "TRACTOR_ORIGINAL", dtype=object)

        catalog.rename_column("LOGM_SAGA_FIDU", "LOG_MSTAR_SAGA")
        #note that even for the clean catalog, there will be some maskbits activated, within twice of D26 of SGA at same redshift, bad gr color, low SNR, close to star? Need to think about whether the TRACTOR MASKBITS should be ignored or not here.
        #TODO: create function that will add maskbits to clean catalog
        catalog["PHOTOMETRY_UPDATED"] = np.zeros(len(catalog)).astype(bool)  

        #updating the stellar mass from M24, uses g band magnitude
        gr_colors = catalog["MAG_G"].data - catalog["MAG_R"].data
        log_mstars_M24 = get_stellar_mass_mia(gr_colors, catalog["MAG_G"].data , catalog["Z_CMB"].data, d_in_mpc = catalog["LUMI_DIST_MPC"].data, input_zred =  False)

        catalog["LOG_MSTAR_M24"] = log_mstars_M24

        print("Adding DWARF MASKBIT columns to clean catalog")
        clean_maskbits = only_trac_maskbits = create_shred_maskbits_from_dict(catalog, bitmasks_to_apply = [7,11,12,13], verbose=True, mag_type = "")
        catalog["DWARF_MASKBIT"] = clean_maskbits

        #add the SHAPE_PARAMS column
        org_aper_params = np.vstack([catalog["SHAPE_R"].data,catalog["BA"].data,catalog["PHI"].data]).T.astype(np.float32)
        catalog["SHAPE_PARAMS"] = org_aper_params

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
    print("Need to update to the M24 mass cut")
    print(f"Number before dwarf mass cut = {len(catalog)}")
    catalog = catalog[catalog["LOG_MSTAR_SAGA"].data < 9.25]
    print(f"Number after dwarf mass cut = {len(catalog)}")
    
    #then we loop over the columns to get the final subset of columns
    # Keep only columns present in main_datamodel
    catalog = catalog[[col for col in main_datamodel.keys()]]
    
    print("Need to think a bit more about the blank value stuff")
    for col in main_datamodel.keys():
        print(f"Column : {col}")
        meta = main_datamodel[col]

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

    return catalog


def create_tractor_data_model(catalog,save_name):
    '''
    Function that creates the data model for the tractor hdu
    '''

    tractor_hdu_cols = [
    "RELEASE", "BRICKNAME", "BRICKID", "BRICK_OBJID", "EBV", "FIBERFLUX_R", "MASKBITS", "REF_ID", "REF_CAT",
    "FLUX_G", "FLUX_IVAR_G", "MAG_G", "MAG_G_ERR", "FLUX_R", "FLUX_IVAR_R", "MAG_R", "MAG_R_ERR",
    "FLUX_Z", "FLUX_IVAR_Z", "MAG_Z", "MAG_Z_ERR", "FIBERMAG_R", "OBJID", "SIGMA_G", "FRACFLUX_G",
    "RCHISQ_G", "SIGMA_R", "FRACFLUX_R", "RCHISQ_R", "SIGMA_Z", "FRACFLUX_Z", "RCHISQ_Z",
    "SHAPE_R", "SHAPE_R_ERR", "MU_R", "MU_R_ERR", "SERSIC", "SERSIC_IVAR", "BA", "TYPE", "PHI",
    "NOBS_G", "NOBS_R", "NOBS_Z", "MW_TRANSMISSION_G", "MW_TRANSMISSION_R", "MW_TRANSMISSION_Z", "SWEEP"]

    #subselect the columns and save it as a separate file after adding the units and stuff

    #RENAME MASKBITS TO TRACTOR_MASKBITS TO AVOID CONFUSION

    ##ADD THE UNIT STUFF
    tractor_tab = catalog[tractor_hdu_cols]

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
            mask = tractor_tab[col].mask if hasattr(tractor_tab[col], 'mask') else np.isnan(tractor_tab[col])
            tractor_tab[col][mask] = blank_val

    # 3. Save to FITS
    tractor_tab.write(save_name, overwrite=True)

    return tractor_tab


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




def create_new_photo_data_model():

    return


def get_fastspec_matched_catalog(gal_cat, save_name, match_method = "TARGETID"):
    '''
    Get the RA,DEC matched fastspec catalog and save it   
    '''
    fastspec_cat = match_fastspec_catalog(gal_cat,coord_name = "",match_method = match_method)

    #make sure this is not a masked column!
    fastspec_cat = make_catalog_unmasked(fastspec_cat)

    #save this 
    fastspec_cat.write(f"{save_name}",overwrite=True)

    #see what fraction of the catalog has np.nans in catalog
    mask = np.isnan(fastspec_cat["RA"])
    print(f"{np.sum(mask)}/{len(mask)} objects have no match in Fastspecfit catalog!")
    return fastspec_cat

def get_fastspec_fit_catalog():
    '''
    In this function, we combine the relevant columns and healpix fastspec files
    '''

    # Path pattern to your FITS files
    files_bright = glob.glob("/global/cfs/cdirs/desi/public/dr1/vac/dr1/fastspecfit/iron/v3.0/catalogs/fastspec-iron*bright*.fits")
    files_dark = glob.glob("/global/cfs/cdirs/desi/public/dr1/vac/dr1/fastspecfit/iron/v3.0/catalogs/fastspec-iron*dark*.fits")
    files_backup = glob.glob("/global/cfs/cdirs/desi/public/dr1/vac/dr1/fastspecfit/iron/v3.0/catalogs/fastspec-iron*backup*.fits")
    files_other = glob.glob("/global/cfs/cdirs/desi/public/dr1/vac/dr1/fastspecfit/iron/v3.0/catalogs/fastspec-iron*other*.fits")
    
    files = files_bright + files_dark + files_backup + files_other

    print(f"Total number of files to read = {len(files)}")
    
    fastspec_metadata_cols = ["TARGETID","RA","DEC"]
    
    fastspec_specphot_cols = ["DN4000", "DN4000_OBS", "DN4000_IVAR", "DN4000_MODEL", "DN4000_MODEL_IVAR", "VDISP", "VDISP_IVAR", "FOII_3727_CONT", "FOII_3727_CONT_IVAR", "FHBETA_CONT", "FHBETA_CONT_IVAR", "FOIII_5007_CONT", "FOIII_5007_CONT_IVAR","FHALPHA_CONT", "FHALPHA_CONT_IVAR" ]
    
    fastspec_cols = ["SNR_B", "SNR_R", "SNR_Z", "APERCORR", "APERCORR_G", "APERCORR_R", "APERCORR_Z"] 
    
    fastspec_emlines_cols = ["OII_3726_FLUX", "OII_3726_FLUX_IVAR", "OII_3729_FLUX", "OII_3729_FLUX_IVAR", "OIII_4363_FLUX", "OIII_4363_FLUX_IVAR", "HEII_4686_FLUX", "HEII_4686_FLUX_IVAR", "HBETA_FLUX", "HBETA_FLUX_IVAR", "OIII_4959_FLUX", "OIII_4959_FLUX_IVAR", "OIII_5007_FLUX", "OIII_5007_FLUX_IVAR", "HEI_5876_FLUX", "HEI_5876_FLUX_IVAR", "NII_6548_FLUX", "NII_6548_FLUX_IVAR", "HALPHA_FLUX", "HALPHA_FLUX_IVAR", "HALPHA_BROAD_FLUX", "HALPHA_BROAD_FLUX_IVAR", "NII_6584_FLUX", "NII_6584_FLUX_IVAR", "SII_6716_FLUX", "SII_6716_FLUX_IVAR", "SII_6731_FLUX", "SII_6731_FLUX_IVAR", "SIII_9069_FLUX", "SIII_9069_FLUX_IVAR", "SIII_9532_FLUX", "SIII_9532_FLUX_IVAR", "HALPHA_BOXFLUX", "HALPHA_BOXFLUX_IVAR", "HALPHA_EW", "HALPHA_EW_IVAR", "HALPHA_SIGMA", "HALPHA_SIGMA_IVAR"]
    
    fastspec_tot_cols = fastspec_cols +  fastspec_emlines_cols
    
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



def combine_hdus(hdu_list, base_path="/pscratch/sd/v/virajvm/desi_dwarf_catalogs/dr1/v1.0/temp_cats",
                 output_file="/pscratch/sd/v/virajvm/desi_dwarf_catalogs/dr1/v1.0/desi_dwarfs_combined.fits"):
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
    """

    hdu_tables = []
    for hdu_name in hdu_list:
        shred_fname = os.path.join(base_path, f"shreds_{hdu_name}_hdu.fits")
        clean_fname = os.path.join(base_path, f"clean_{hdu_name}_hdu.fits")
        
        print(f"Reading {shred_fname}...")
        print(f"Reading {clean_fname}...")
        
        clean_tab = Table.read(clean_fname)
        shred_tab = Table.read(shred_fname)

        tab = vstack([clean_tab, shred_tab])
        
        hdu_tables.append(tab)

    # Sanity check: number of rows
    nrows = [len(tab) for tab in hdu_tables]
    if len(set(nrows)) != 1:
        raise ValueError(f"Row count mismatch across HDUs: {dict(zip(hdu_list, nrows))}")

    # Sanity check: TARGETID alignment
    target_ids = [tab["TARGETID"] for tab in hdu_tables]
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
    
if __name__ == '__main__':

    save_path = "/pscratch/sd/v/virajvm/desi_dwarf_catalogs/dr1/v1.0/temp_cats"
    
    process_shreds = True
    process_clean = True

    if process_shreds:
        #loading the shredded catalogs!
        print("Reading ELG shreds!")
        elg_shred = Table.read(f"/pscratch/sd/v/virajvm/catalog_dr1_dwarfs/iron_photometry/iron_ELG_shreds_catalog_w_aper_mags.fits")
        elg_shred = consolidate_new_photo(elg_shred,sample="ELG")
        print("=="*10)
    
        print("Reading BGS Bright shreds!")
        bgsb_shred = Table.read(f"/pscratch/sd/v/virajvm/catalog_dr1_dwarfs/iron_photometry/iron_BGS_BRIGHT_shreds_catalog_w_aper_mags.fits")
        bgsb_shred = consolidate_new_photo(bgsb_shred,sample="BGS_BRIGHT")
        print("=="*10)
        
        print("Reading BGS Faint shreds!")
        bgsf_shred = Table.read(f"/pscratch/sd/v/virajvm/catalog_dr1_dwarfs/iron_photometry/iron_BGS_FAINT_shreds_catalog_w_aper_mags.fits")
        bgsf_shred = consolidate_new_photo(bgsf_shred,sample="BGS_FAINT")
        print("=="*10)
    
        print("Reading LOWZ shreds!")
        lowz_shred = Table.read(f"/pscratch/sd/v/virajvm/catalog_dr1_dwarfs/iron_photometry/iron_LOWZ_shreds_catalog_w_aper_mags.fits")
        lowz_shred = consolidate_new_photo(lowz_shred,sample="LOWZ")
        print("=="*10)
    
        print("Reading SGA shreds!")
        sga_all = Table.read("/pscratch/sd/v/virajvm/catalog_dr1_dwarfs/iron_photometry/iron_SGA_sga_catalog_w_aper_mags.fits")
        sga_all = consolidate_new_photo(sga_all,sample="SGA")
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
        tot_shred = create_main_data_model(tot_shred, save_path + "/shreds_MAIN_hdu.fits", clean_cat=False)
        
        ##get the fastspecfit hdu
        print("Creating the shred fastspecfit hdu")
        _ = get_fastspec_matched_catalog(tot_shred, save_path + "/shreds_FASTSPEC_hdu.fits", match_method="TARGETID")
        
        ##get the other hdus


    if process_clean:
        ##get the clean catalog stuff now!!
        clean_cat = Table.read("/pscratch/sd/v/virajvm/catalog_dr1_dwarfs/desi_y1_dwarf_clean_catalog_v4.fits")

        print("Creating the clean main hdu")
        clean_cat = create_main_data_model(clean_cat, save_path + "/clean_MAIN_hdu.fits", clean_cat=True)
        
        ##get the fastspecfit hdu
        print("Creating the clean fastspecfit hdu")
        _ = get_fastspec_matched_catalog(clean_cat, save_path + "/clean_FASTSPEC_hdu.fits", match_method="TARGETID")



    #then we consolidate it all into a multi-ext file!
    combine_hdus(["MAIN", "FASTSPEC"], base_path="/pscratch/sd/v/virajvm/desi_dwarf_catalogs/dr1/v1.0/temp_cats",
                 output_file="/pscratch/sd/v/virajvm/desi_dwarf_catalogs/dr1/v1.0/desi_dr1_dwarf_catalog.fits")
        

    


























    

    

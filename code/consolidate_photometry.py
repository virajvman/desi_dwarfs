'''
Functions where preferred, best photometry column is created along with photometry maskbits to identify reliable measurements. This is also the script where we produce the final, multi-extension fits files as the final catalog output.
'''


import numpy as np
import astropy.io.fits as fits
import astropy.units as u
from astropy.table import Table, vstack, join, hstack
from shred_photometry_maskbits import cog_mag_converge, cog_nan_mask, cog_curve_decrease, create_shred_maskbits
import glob

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


def likely_over_deblended(zred, r2_mur):
    """
    Returns boolean array (or scalar if inputs are scalars) indicating
    whether the source is in the over-deblended regime.

    For sources that satisfy this mask, we use the 
    """
    #this used to be 26
    slope = (26 - 22.75) / (0.1 - 0.005)
    y_intp = 26 - slope*0.1
    
    bound_value = slope * zred + y_intp
    
    # core condition
    likely = r2_mur > bound_value
    
    # apply the zred < 0.005 override
    likely = np.where(zred < 0.005, True, likely)
    
    return likely


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
    #3) if the tractor based mag is much brighter than the COG based based mag, and there is no decrease, we will revert to the tractor based mag?
    
    prefer_tractor_based_mag = cog_mag_converge(catalog) 
    prefer_simple_mag = cog_nan_mask(catalog) | cog_curve_decrease(catalog)
    
    for b in bands:
        trac = catalog[f"TRACTOR_ONLY_MAG_{b}_FINAL"].data
        simp = catalog[f"SIMPLE_PHOTO_MAG_{b}"].data
        cog  = catalog[f"COG_MAG_{b}_FINAL"].data

        best = np.select(
            [prefer_tractor_based_mag, prefer_simple_mag],
            [trac, simp],
            default=cog
        )

        catalog[f"MAG_{b}_BEST"] = best

    
    # Source label as string
    source = np.full(len(catalog), "COG", dtype=object)  # default COG
    source[prefer_simple_mag] = "SIMPLE"
    source[prefer_tractor_based_mag] = "TRACTOR_BASED"
    catalog[f"MAG_TYPE"] = source

    return catalog


def consolidate_photo(catalog,plot=False):
    '''
    In this function, we consolidate the different COG magnitude using the over de-deblending criterion.
    Note that the fraction of the aperture that is masked is the initial aperture and not the final COG aperture    
    '''

    #these are the columns we want to make that consolidate based on whether to use the isolate or no isolate mask
    org_keys_to_combine = ["COG_MAG_G", "COG_MAG_R", "COG_MAG_Z", "TRACTOR_ONLY_MAG_G", "TRACTOR_ONLY_MAG_R", "TRACTOR_ONLY_MAG_Z", "APER_R4_MAG_G", "APER_R4_MAG_R",
                           "APER_R4_MAG_Z", "APER_R4_FRAC_IN_IMG", "COG_CHI2", "COG_DOF", "COG_MAG_ERR", "FIBER_MAG", "COG_PARAMS_G", "COG_PARAMS_R", "COG_PARAMS_Z", 
                           "COG_PARAMS_G_ERR", "COG_PARAMS_R_ERR", "COG_PARAMS_Z_ERR", "COG_DECREASE_MAX_LEN", "COG_DECREASE_MAX_MAG",
                           "APER_CEN_RADEC", "APER_CEN_XY_PIX", "APER_R4_MASK_FRAC", "APER_CEN_MASKED", "APER_PARAMS"]

    #to the above we also add the tractor columns!
    tractor_keys_to_combine = ["TRACTOR_ONLY_COG_MAG", "TRACTOR_ONLY_FIBER_MAG", "TRACTOR_ONLY_COG_MAG_ERR", "TRACTOR_ONLY_COG_PARAMS_G", 
                               "TRACTOR_ONLY_COG_PARAMS_G_ERR", "TRACTOR_ONLY_COG_PARAMS_R", "TRACTOR_ONLY_COG_PARAMS_R_ERR", "TRACTOR_ONLY_COG_PARAMS_Z",
                               "TRACTOR_ONLY_COG_PARAMS_Z_ERR", "TRACTOR_ONLY_COG_CHI2", "TRACTOR_ONLY_APER_CEN_RADEC", "TRACTOR_ONLY_APER_PARAMS",
                               "TRACTOR_APER_CEN_MASKED"]

    all_keys_to_combine = org_keys_to_combine + tractor_keys_to_combine

    print("Once catalog is re-run, make sure the APER_R2 ELLIPSE is changed to ISLAND")
    apply_no_isolate_mask = likely_over_deblended(catalog["Z"].data, catalog["APER_R2_MU_R_ISLAND_TRACTOR"].data)
    #when this is true, we use no_isolate photometry. Otherwise, we use the isolate photometry
    
    pairs = {}

    for ki in all_keys_to_combine:
        pairs[ki] = ( catalog[ki + "_ISOLATE" ].data , catalog[ki + "_NO_ISOLATE"].data )
        
    for newcol, (w_iso, no_iso) in pairs.items():
        catalog[newcol + "_FINAL"] = combine_arrays(no_iso, w_iso, apply_no_isolate_mask)

    #need to make a column indicating whether the final photo was with isolate or no isolate
    catalog["ISOLATE_MASK_LIKELY_SHREDDED"] = apply_no_isolate_mask

    catalog = add_best_mags(catalog)

    #add the photomaskbit column
    catalog = create_shred_maskbits(catalog)

    # TODO: include a maskbit on how different the tractor based and cog based mags are: extreme negative values could make things weird .. example 0.5 or more? probably more important when tractor is larger?
    # hmm suspicious ... need to verify this ...

    ##TODO: add BRIGHT STAR MASKBUT 1?

    ##if close to a star, just use the tractor model photometry as very likely it will be under-estimated due to masking

    # TODO: add function to add PCNN values!!

    #ONE CONSIDERATION TO KEEP IN MIND IS THAT TRACTOR TENDS TO NOT DO WELL, NEXT TO VERY BRIGHT STARS, AND THIS CAN AFFECT STUFF : 39627633918479081

    ##this is a nice example to show working!: 39627642730709361, 39627643640878867
    #example of how merging systems can be hard: 39627643015922148
    #we have no optimally combined all the columns together!
    return catalog



### THE BELOW FUNCTIONS ARE ADDING THE DATA MODEL AND UNITS TO THE CATALOG

from data_model import tractor_datamodel, fastspec_hdu_datamodel

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


def create_main_data_model():


    return


def create_new_photo_data_model():

    return


    
    ##updating name of EXTNAME
    # #and adding checksum
    
    # filename = "/global/cfs/cdirs/desi/users/virajvm/desi_dwarfs_vac/dr1/v1.0/desi_dwarfs_y1_catalog.fits"
    
    # with fits.open(filename, mode="update") as hdul:
    #     # hdul[1].header["EXTNAME"] = "EG_DWARFS"  # Change name of 2nd HDU
    #     # hdul[0].header["EXTNAME"] = "PRIMARY"  # Change name of 2nd HDU
    
    #      # Add CHECKSUM and DATASUM to all HDUs
    #     hdul[0].add_checksum()
    #     hdul[1].add_checksum()
    
    #     hdul.flush()  # Save changes

    # with fits.open("/pscratch/sd/v/virajvm/catalog_dr1_dwarfs/desi_dwarfs_y1_catalog.fits", mode="update") as hdul:
    # hdul[1].header["EXTNAME"] = "DATA"  # Change name of 2nd HDU
    # hdul[0].header["EXTNAME"] = "PRIMARY"  # Change name of 2nd HDU
    # hdul.flush()  # Save changes

    # ogM_sun = u.def_unit('log(solMass)', format={'latex': r'\log(M_\odot)'})
    # mur = u.Unit(u.mag / u.arcsec**2, format={'latex': r'mag\,arcsec$^{-2}$'})

#     all_dwarfs["RA"].unit = u.deg
# all_dwarfs["DEC"].unit = u.deg



# def combine_with_clean():
#     '''
#     In this function, we combine the shredded catalog with the clean catalog!! 
#     '''


# def create_multi_ext(catalog, catalog_path):
#     '''
#     In this function, we translate the single large astropy table into a multi-extension catalog. The different extensions are 

#     ext1 -> MAIN (targetid, ra,dec,final photo, gr-based mstar, halpha flux, dwarf_primary, targetid_matchs, SAMPLE)
#     ext2 -> PHOTOMETRY (all the relevant photometry columns of the galaxies whose photometry was remeasured, + tractor columns)
#     ext3 -> Z_CATALOG (relevant redshift catalog quantities)
#     ext4 -> FASTSPECFIT (relevant line fluxes from fastspecfit)
#     ext5 -> SPECTRA_ANOMALY STUFF
#     ext6 -> IMAGE ANOMALY STUFF?

    # '''

#     # Convert each sub-table into a BinTableHDU
#     hdu1 = fits.BinTableHDU(tab1, name="PHOTOMETRY")
#     hdu2 = fits.BinTableHDU(tab2, name="SPECTRO")
#     hdu3 = fits.BinTableHDU(tab3, name="FLUXES")
    
#     # Primary HDU (empty, just header info)
#     primary_hdu = fits.PrimaryHDU()
    
#     # Combine all HDUs into an HDUList
#     hdul = fits.HDUList([primary_hdu, hdu1, hdu2, hdu3])
    
#     # Write out to new FITS file

##TODO: use the fastspec function in the desi_lowz_funcs to add all the relevant stuff. Make sure to port over their datamodel?
## to make it easy save these different extensions as different fits file and then later simply combine them into different extensions!


##TODO: add functions here for each extension so we do some additional renaming and stuff as needed.
## add different photo masks and total maskbits


def get_fastspec_matched_catalog(gal_cat, save_name):
    '''
    Get the RA,DEC matched fastspec catalog and save it   
    '''
    fastspec_cat = match_fastspec_catalog(gal_cat,coord_name = "")
    #save this 
    fastspec_cat.write(f"/pscratch/sd/v/virajvm/catalog_dr1_dwarfs/iron_fastspec_catalog/{save_name}")
    return fastspec_cat


def get_fastspec_fit_catalog():
    '''
    In this function, we combine the relevant columns and healpix fastspec files
    '''

    # Path pattern to your FITS files
    files_bright = glob.glob("/global/cfs/cdirs/desi/public/dr1/vac/dr1/fastspecfit/iron/v3.0/catalogs/fastspec-iron*bright*.fits")
    files_dark = glob.glob("/global/cfs/cdirs/desi/public/dr1/vac/dr1/fastspecfit/iron/v3.0/catalogs/fastspec-iron*dark*.fits")

    files = files_bright + files_dark

    print(f"Total number of files to read = {len(files)}")
    
    fastspec_metadata_cols = ["TARGETID","RA","DEC"]
    fastspec_specphot_cols = ["DN4000", "DN4000_OBS", "DN4000_IVAR", "DN4000_MODEL", "DN4000_MODEL_IVAR"]
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
        
            zmask = (tab_meta_zred < 0.5)
    
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






























    

    

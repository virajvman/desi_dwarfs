'''
In this catalog, we read in all the shred catalog and filter for sources that have robust photometry!!

We will choose objects that do not have nan photometry in g and r band. Will select objects that have reasonable colors. And also objects with 6 < LogM <9.25 

'''

from astropy.table import Table, vstack
import numpy as np

def read_in_shred_catalog(file_path):
    print(file_path)
    cat = Table.read(file_path)
    print(f"Cat total = {len(cat)}")
    #first filter by stellar mass

    nan_mask = (np.isnan(cat["MAG_G_APERTURE_COG"].data) ) | (np.isnan(cat["MAG_R_APERTURE_COG"].data) )
    
    print(f"Cat total with nan either g-band or r-band = {np.sum(nan_mask)}")

    cat = cat[ (cat["LOGM_SAGA_APERTURE_COG"] < 9.25) & (cat["LOGM_SAGA_APERTURE_COG"] > 6) ]
    print(f"Cat total after Mstar cut = {len(cat)}")

    #remove sources with SGA filtering !!
    cat = cat[cat["SGA_D26_NORM_DIST"] > 2] 
    print(f"Cat total after SGA cut = {len(cat)}")

    #remove sources very close to bright stars
    cat = cat[cat["NEAREST_STAR_NORM_DIST"] > 0.5]
    print(f"Cat total after star fdist cut = {len(cat)}")
    
    #filter by g-r color
    cat_grs = cat["MAG_G_APERTURE_COG"].data - cat["MAG_R_APERTURE_COG"].data 

    print(f"g-r min = {np.min(cat_grs)}, g-r max = {np.max(cat_grs)}")

    ## need to remove weird color objects!
    cat = cat[cat_grs < 1]

    print(f"Cat total after g-r < 1.0 cut = {len(cat)}")
    
    return cat


def combine_w_clean_catalog(shred_cat, clean_cat):
    '''
    In this function, we combine these two catalogs to produce a final catalog
    '''

    # Raw string from your example (copied here for reference and cleanup)
    raw_cols = "COADD_FIBERSTATUS CMX_TARGET DESI_TARGET BGS_TARGET MWS_TARGET SCND_TARGET SV1_DESI_TARGET SV1_BGS_TARGET SV1_MWS_TARGET SV2_DESI_TARGET SV2_BGS_TARGET SV2_MWS_TARGET SV3_DESI_TARGET SV3_BGS_TARGET SV3_MWS_TARGET SV1_SCND_TARGET SV2_SCND_TARGET SV3_SCND_TARGET TSNR2_BGS TSNR2_LRG TSNR2_ELG LOGM_CIGALE LOGM_ERR_CIGALE AGNFRAC_CIGALE LOGSFR_CIGALE LOGSFR_ERR_CIGALE FLAGINFRARED_CIGALE IMAGE_FITS_PATH MAG_G_APERTURE_R375 MAG_R_APERTURE_R375 MAG_Z_APERTURE_R375 NEAREST_STAR_DIST NEAREST_STAR_MAX_MAG LOGM_SAGA_APERTURE_R375 SAVE_PATH"
    # Convert to list
    remove_cols = raw_cols.split()

    for col in remove_cols:
        if col in shred_cat.colnames:
            shred_cat.remove_column(col)
        if col in clean_cat.colnames:
            clean_cat.remove_column(col)

    #we need to add some blank columns to the clean_cat table so we can stack these two tables

    # List of columns to add
    missing_cols = [
        "MAG_G_APERTURE_COG", "MAG_R_APERTURE_COG", "MAG_Z_APERTURE_COG",
        "MAG_G_APERTURE_COG_ERR", "MAG_R_APERTURE_COG_ERR", "MAG_Z_APERTURE_COG_ERR",
        "MAG_G_APERTURE_COG_PARAMS", "MAG_R_APERTURE_COG_PARAMS", "MAG_Z_APERTURE_COG_PARAMS",
        "MAG_G_APERTURE_COG_PARAMS_ERR", "MAG_R_APERTURE_COG_PARAMS_ERR", "MAG_Z_APERTURE_COG_PARAMS_ERR",
        "LOGM_SAGA_APERTURE_COG"]

    # Ensure t1 has all columns from t2 with correct dtype
    for col in missing_cols:
        if col not in clean_cat.colnames:
            if col in shred_cat.colnames:
                dtype = shred_cat[col].dtype
                shape = shred_cat[col].shape[1:]  # get shape per row

                print(col, dtype, shape)
                
                length = len(clean_cat)
                
                # Determine fill value based on dtype
                if np.issubdtype(dtype, np.floating):
                    fill_value = np.full(shape, np.nan)
                elif np.issubdtype(dtype, np.integer):
                    fill_value = np.full(shape, -1)
                else:
                    fill_value = np.full(shape, np.nan)
    
                # Create array of shape (length, *shape) with dtype
                clean_cat[col] = np.tile(fill_value, (length, 1) if shape else (length,)).astype(dtype)
            else:
                raise ValueError(f"Column {col} not found in shred_cat to copy dtype from.")

    tot_cat = vstack([clean_cat, shred_cat])

    #renaming the MAG_R, MAG_G columns
    for b in "GRZ":
        tot_cat.rename_column(f"MAG_{b}", f"MAG_{b}_DR9")
        tot_cat.rename_column(f"MAG_{b}_ERR", f"MAG_{b}_ERR_DR9")

    tot_cat.rename_column("LOGM_SAGA", "LOGM_SAGA_DR9")
    tot_cat.rename_column("LOGM_M24", "LOGM_M24_DR9")
    
    #producing the final summary columns!
    for b in "GRZ":
        tot_cat[f"MAG_{b}"] = np.where(
                                ~np.isnan(tot_cat[f"MAG_{b}_APERTURE_COG"]),  # if this is not nan
                                tot_cat[f"MAG_{b}_APERTURE_COG"],             # use this value
                                tot_cat[f"MAG_{b}_DR9"]                           # else fall back to MAG_R
                            )

        tot_cat[f"MAG_{b}_ERR"] = np.where(
                                ~np.isnan(tot_cat[f"MAG_{b}_APERTURE_COG_ERR"]),  # if this is not nan
                                tot_cat[f"MAG_{b}_APERTURE_COG_ERR"],             # use this value
                                tot_cat[f"MAG_{b}_ERR_DR9"]                           # else fall back to MAG_R
                            )

    
    tot_cat[f"LOGM_SAGA"] = np.where(
                                ~np.isnan(tot_cat[f"LOGM_SAGA_APERTURE_COG"]),  # if this is not nan
                                tot_cat[f"LOGM_SAGA_APERTURE_COG"],             # use this value
                                tot_cat[f"LOGM_SAGA_DR9"]                           # else fall back to MAG_R
                            )
    
    return tot_cat


if __name__ == '__main__':

    photo_path = "/pscratch/sd/v/virajvm/catalog_dr1_dwarfs/iron_photometry/" 
    
    # iron_ELG_shreds_catalog_w_aper_mags.fits

    all_samps = ["BGS_BRIGHT","BGS_FAINT","LOWZ","ELG"]

    all_cats_f = []
    
    for sampi in all_samps:
        cat_f = read_in_shred_catalog(photo_path + f"iron_{sampi}_shreds_catalog_w_aper_mags.fits")
        all_cats_f.append(cat_f)

    all_cats_f = vstack(all_cats_f)

    print(f"Combined shred catalog of dwarfs = {len(all_cats_f)}")

    #save this combined catalog
    all_cats_f.write("/pscratch/sd/v/virajvm/catalog_dr1_dwarfs/desi_y1_dwarf_shreds_catalog_filter.fits",overwrite=True)

    clean_cat = Table.read("/pscratch/sd/v/virajvm/catalog_dr1_dwarfs/desi_y1_dwarf_clean_catalog_v3.fits")
    clean_cat = clean_cat[ clean_cat["LOGM_SAGA"] < 9.25 ]

    ##DO SOME ADDITIONAL CLEANING HERE OR LATER LIKE G-R cut, and bright star cut, and mask bits?
    print(len(clean_cat))

    tot_cat = combine_w_clean_catalog(shred_cat = all_cats_f, clean_cat = clean_cat)

    print(f"Total number of dwarfs = {len(tot_cat)}")

    #we then select for unique targetids

    _, uni_idx = np.unique(tot_cat["TARGETID"].data, return_index=True)

    tot_cat = tot_cat[uni_idx]
    
    print(f"Unique TARGETIDS = {len(tot_cat)}")

    #make a final column of MAG_G, MAG_R, MAG_Z, AND ERRORS AND LOGM_SAGA
    tot_cat.write("/pscratch/sd/v/virajvm/catalog_dr1_dwarfs/desi_y1_dwarf_combine_catalog.fits",overwrite=True)

    #obtain some basic spectra properties like Halpha flux, halpha ew!
    fspec_col_names = ["TARGETID", "HALPHA_FLUX", "HALPHA_FLUX_IVAR", "SNR_B","SNR_R","SNR_Z","HALPHA_EW", "HALPHA_EW_IVAR"]
        
    from desi_lowz_funcs import get_tgids_fastspec

    tgids = tot_cat["TARGETID"].data

    fastspec_table, mask = get_tgids_fastspec(tgids, fspec_col_names)

    print(len(fastspec_table["TARGETID"]))
    print(len(np.unique(fastspec_table["TARGETID"])))
    print(len(tgids))

    _, uni_mask = np.unique(fastspec_table["TARGETID"],return_index=True)
    
    print(tgids[:5])
    print(fastspec_table["TARGETID"][:5])

    order_mask = np.argsort(fastspec_table["TARGETID"][uni_mask])

    print( np.max(np.abs(tgids - fastspec_table["TARGETID"][uni_mask][order_mask] )) )

    tot_cat["HALPHA_FLUX"] = fastspec_table["HALPHA_FLUX"][uni_mask][order_mask]
    tot_cat["HALPHA_FLUX_IVAR"] = fastspec_table["HALPHA_FLUX_IVAR"][uni_mask][order_mask]
    tot_cat["SNR_B"] = fastspec_table["SNR_B"][uni_mask][order_mask]
    tot_cat["SNR_R"] = fastspec_table["SNR_R"][uni_mask][order_mask]
    tot_cat["SNR_Z"] = fastspec_table["SNR_Z"][uni_mask][order_mask]
    tot_cat["HALPHA_EW"] = fastspec_table["HALPHA_EW"][uni_mask][order_mask]
    tot_cat["HALPHA_EW_IVAR"] = fastspec_table["HALPHA_EW_IVAR"][uni_mask][order_mask]

    #saving it all now!
    tot_cat.write("/pscratch/sd/v/virajvm/catalog_dr1_dwarfs/desi_y1_dwarf_combine_catalog.fits",overwrite=True)
    
    
    
    

    
    
    






    
    

    


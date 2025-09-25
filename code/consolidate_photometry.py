'''
Functions where preferred, best photometry column is created along with photometry maskbits to identify reliable measurements. This is also the script where we produce the final, multi-extension fits files as the final catalog output.
'''


import numpy as np
import astropy.io.fits as fits
import astropy.units as u
from astropy.table import Table, vstack, join
from shred_photometry_maskbits import cog_mag_converge, cog_nan_mask, cog_curve_decrease, create_shred_maskbits

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
    slope = (25 - 22.5) / (0.05 - 0.005)
    y_intp = 22.2222
    
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
    prefer_tractor_based_mag = cog_mag_converge(catalog)
    prefer_simple_mag = cog_nan_mask(catalog) | cog_curve_decrease(catalog)

    for b in bands:
        trac = catalog[f"TRACTOR_PARENT_MAG_{b}_FINAL"].data
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
    keys_to_combine = ["COG_MAG_G", "COG_MAG_R", "COG_MAG_Z", "TRACTOR_PARENT_MAG_G", "TRACTOR_PARENT_MAG_R", "TRACTOR_PARENT_MAG_Z",
                       "APER_R4_MAG_G", "APER_R4_MAG_R", "APER_R4_MAG_Z", "APER_R4_FRAC_IN_IMG", 
                       "COG_CHI2", "COG_DOF", "COG_MAG_ERR", "COG_PARAMS_G", "COG_PARAMS_R", "COG_PARAMS_Z", "COG_PARAMS_G_ERR", 
                       "COG_PARAMS_R_ERR", "COG_PARAMS_Z_ERR", "COG_DECREASE_MAX_LEN", "COG_DECREASE_MAX_MAG",   
                       "APER_CEN_RADEC", "APER_CEN_XY_PIX", "APER_PARAMS", "LOGM_SAGA_COG"] #,"APER_R4_MASK_FRAC"]

    print("MAKE SURE THE APER_R4_MASK_FRAC COLUMN IS ADDED BACK AFTER THE RERUN")

    apply_no_isolate_mask = likely_over_deblended(catalog["Z"].data, catalog["APER_R2_MU_R_TRACTOR"].data)
    #when this is true, we use no_isolate photometry. Otherwise, we use the isolate photometry
    
    pairs = {}

    for ki in keys_to_combine:
        pairs[ki] = ( catalog[ki + "_ISOLATE" ].data , catalog[ki + "_NO_ISOLATE"].data )
        
    for newcol, (w_iso, no_iso) in pairs.items():
        catalog[newcol + "_FINAL"] = combine_arrays(no_iso, w_iso, apply_no_isolate_mask)

    #need to make a column indicating whether the final photo was with isolate or no isolate
    catalog["ISOLATE_MASK_LIKELY_SHREDDED"] = apply_no_isolate_mask

    #further consolidate this into the best photometry!
    #some criterion for consolidation:
    #1) if COG_MAG_FINAL in any band is 0.5 mag or larger than its R4_FINAL mag, we revert to the tractor only based reconsutrction
    #2) if over-subtraction like either nans or consecutive decrease, we use the simplest photo

    catalog = add_best_mags(catalog)

    #add the photomaskbit column
    catalog = create_shred_maskbits(catalog)

    #we have no optimally combined all the columns together!
    return catalog


def create_photometry_data_model():
    '''
    Function that creates the data model for all the photometry columns
    '''

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





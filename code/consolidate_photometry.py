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

    print("Change the SIMPLE_PHOTO_G to include MAG next")
    for b in bands:
        trac = catalog[f"TRACTOR_ONLY_MAG_{b}_FINAL"].data
        simp = catalog[f"SIMPLE_PHOTO_{b}"].data
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

    print("Also add back in COG_PARAMS_G")
    #these are the columns we want to make that consolidate based on whether to use the isolate or no isolate mask
    org_keys_to_combine = ["COG_MAG_G", "COG_MAG_R", "COG_MAG_Z", "TRACTOR_ONLY_MAG_G", "TRACTOR_ONLY_MAG_R", "TRACTOR_ONLY_MAG_Z", "APER_R4_MAG_G", "APER_R4_MAG_R",
                           "APER_R4_MAG_Z", "APER_R4_FRAC_IN_IMG", "COG_CHI2", "COG_DOF", "COG_MAG_ERR", "FIBER_MAG", "COG_PARAMS_R", "COG_PARAMS_Z", 
                           "COG_PARAMS_G_ERR", "COG_PARAMS_R_ERR", "COG_PARAMS_Z_ERR", "COG_DECREASE_MAX_LEN", "COG_DECREASE_MAX_MAG",
                           "APER_CEN_RADEC", "APER_CEN_XY_PIX", "APER_R4_MASK_FRAC", "APER_CEN_MASKED", "APER_PARAMS"]

    #to the above we also add the tractor columns!
    tractor_keys_to_combine = ["TRACTOR_ONLY_COG_MAG", "TRACTOR_ONLY_FIBER_MAG", "TRACTOR_ONLY_COG_MAG_ERR", "TRACTOR_ONLY_COG_PARAMS_G", 
                               "TRACTOR_ONLY_COG_PARAMS_G_ERR", "TRACTOR_ONLY_COG_PARAMS_R", "TRACTOR_ONLY_COG_PARAMS_R_ERR", "TRACTOR_ONLY_COG_PARAMS_Z",
                               "TRACTOR_ONLY_COG_PARAMS_Z_ERR", "TRACTOR_ONLY_COG_CHI2", "TRACTOR_ONLY_APER_CEN_RADEC", "TRACTOR_ONLY_APER_PARAMS",
                               "TRACTOR_APER_CEN_MASKED"]

    all_keys_to_combine = org_keys_to_combine + tractor_keys_to_combine

    print("Once catalog is re-run, make sure the APER_R2 ELLIPSE is changed to ISLAND")
    apply_no_isolate_mask = likely_over_deblended(catalog["Z"].data, catalog["APER_R2_MU_R_ELLIPSE_TRACTOR"].data)
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

    # TODO: add function to add PCNN values!!
        
    #we have no optimally combined all the columns together!
    return catalog


    
def create_photometry_data_model():
    '''
    Function that creates the data model for all the photometry columns
    '''

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





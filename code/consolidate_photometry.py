'''
Functions where preferred, best photometry column is created along with photometry maskbits to identify reliable measurements
'''


def combine_arrays(no_iso, w_iso, mask):
    if no_iso.ndim == 1:  # 1D case
        return np.where(mask, no_iso, w_iso)
    else:  # 2D or higher
        # Expand mask along all extra dims so it broadcasts
        # expanded_mask = mask[(...,) + (None,) * (no_iso.ndim - 1)]
        expanded_mask = np.expand_dims(mask, axis=tuple(range(1, no_iso.ndim)))
        return np.where(expanded_mask, no_iso, w_iso)
        
        
def consolidate_photo(catalog,plot=False):
    '''
    In this function, we consolidate the different COG magnitude using the over de-deblending criterion.
    Note that the fraction of the aperture that is masked is the initial aperture and not the final COG aperture    
    '''

    #these are the columns we want to make that consolidate based on whether to use the isolate or no isolate mask
    keys_to_combine = ["COG_MAG_G", "COG_MAG_R", "COG_MAG_Z", "TRACTOR_PARENT_MAG_G", "TRACTOR_PARENT_MAG_R", "TRACTOR_PARENT_MAG_Z",
                       "APER_R425_MAG_G", "APER_R425_MAG_R", "APER_R425_MAG_Z", "APER_R425_FRAC_IN_IMG", 
                       "COG_CHI2", "COG_DOF", "COG_MAG_ERR", "COG_PARAMS_G", "COG_PARAMS_R", "COG_PARAMS_Z", "COG_PARAMS_G_ERR", 
                       "COG_PARAMS_R_ERR", "COG_PARAMS_Z_ERR", "COG_DECREASE_MAX_LEN", "COG_DECREASE_MAX_MAG",   
                       "APER_CEN_RADEC", "APER_CEN_XY_PIX", "APER_PARAMS"]
    

    NEED TO FINALIZE THIS MASK
    no_apply_isolate_mask = ( (catalog["Z"].data < 0.035) & (catalog["APER_R2_MU_R_SMOOTH"].data > 21.5) ) | (catalog["Z"].data < 0.01)

    pairs = {}

    for ki in keys_to_combine:
        pairs[ki] = ( catalog[ki + "_ISOLATE" ].data , catalog[ki + "_NO_ISOLATE"].data )
        
    for newcol, (no_iso, w_iso) in pairs.items():
        catalog[newcol + "_FINAL"] = combine_arrays(no_iso, w_iso, no_apply_isolate_mask)

    #need to make a column indicating whether the final photo was with isolate or no isolate
    catalog[""]

    #let us test that we are apply this mask correctyl!
    if plot:
        plt.figure(figsize = (3,3))
        plt.scatter(catalog["Z"].data , catalog["APER_R2_MU_R_SMOOTH"].data, s= 1)
        plt.scatter(catalog["Z"].data[no_apply_isolate_mask] , catalog["APER_R2_MU_R_SMOOTH"].data[no_apply_isolate_mask], s= 1)
        plt.xlim([0.001,0.1])
        plt.ylim([-1,24])
        plt.show()

    #ahh the reason we do not see any points for z<0.01 is because they have their APER_R2_MU_SMOOTH = 0

    #create a new fits file with different extensions

 
    return catalog




# def create_multi_ext(catalog, catalog_path):
#     '''
#     In this function, we translate the single large astropy table into a multi-extension catalog. The different extensions are 

#     ext1 -> MAIN
#     ext2 -> PHOTOMETRY
#     ext3 -> Z_CATALOG
#     ext4 -> FASTSPECFIT LINE FLUXES
#     ext5 -> 
#     '''

        

    
#     # Convert each sub-table into a BinTableHDU
#     hdu1 = fits.BinTableHDU(tab1, name="PHOTOMETRY")
#     hdu2 = fits.BinTableHDU(tab2, name="SPECTRO")
#     hdu3 = fits.BinTableHDU(tab3, name="FLUXES")
    
#     # Primary HDU (empty, just header info)
#     primary_hdu = fits.PrimaryHDU()
    
#     # Combine all HDUs into an HDUList
#     hdul = fits.HDUList([primary_hdu, hdu1, hdu2, hdu3])
    
#     # Write out to new FITS file





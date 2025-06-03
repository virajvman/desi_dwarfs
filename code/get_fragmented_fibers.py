'''
In this script, we will find other DESI DR1 fibers that are associated with the parent galaxy.
'''

def find_matches(source_ra, source_dec, source_zred, source_file_path, source_img_path, entire_skycoords, entire_catalog, verbose=False):
    '''
    source_ra/dec is the RA,DEC of the reference source of interest

    source_file_path is the folder path to where all the important info is stored

    entire_catalog is the minimal version (with RA,DEC,TARGETID) of the entire dwarf galaxy catalog.

    entire_skycoords is the skycoord object for all the entire catalog: skycoords = SkyCoord(ra=ra_catalog*u.deg, dec=dec_catalog*u.deg)
    '''
    
    #read the segment map
    segm = np.load(f"{source_file_path}/main_segment_map.npy")
    img_data = fits.open(source_img_path)[0].data
    wcs_cutout = WCS(fits.getheader(img_path))

    #filter the entire catalog for only nearby sources
    ref_coord = SkyCoord(ra=source_ra*u.deg, dec=source_dec*u.deg)
    sep = skycoords.separation(ref_coord)
    mask_nearby = sep < 120*u.arcsec

    
    relevant_catalog = entire_catalog[(mask_nearby)]

    if verbose:
        print(f"{len(relevant_catalog)} Relevant objects left after separation filtering.")

    if len(relevant_catalog) == 0:
        return np.array( [], dtype = np.int64)

    #hoping for redshift that is consistent to within 1000km/s (very lax here)
    delta_zred = 1000/300000
    redshift_mask = (np.abs(relevant_catalog["Z"] -  source_zred) < delta_zred )

    relevant_catalog = relevant_catalog[delta_zred]

    if verbose:
        print(f"{len(relevant_catalog)} Relevant objects left after redshift filtering.")

    if len(relevant_catalog) == 0:
        return np.array( [], dtype = np.int64)

    #find the relevant sources that lie on the main segment 
    xpix_all, ypix_all, _ = wcs_cutout.all_world2pix(relevant_catalog["RA"], relevant_catalog["DEC"], 0, 1)

    on_main = ~np.isnan(segm[ypix_all.astype(int), xpix_all.astype(int)])
    #we want whatever source that has remained to be on the main segment we identified!!
    relevant_catalog = relevant_catalog[on_main]

    if verbose:
        print(f"{len(relevant_catalog)} Relevant objects left after filtering for main segment association.")

    matching_tgids = relevant_catalog["TARGETID"].data
    
    return np.array( matching_tgids, dtype = np.int64)

    
if __name__ == '__main__':

    #the fibers are hoping to find will be at similar redshifts and thus will also be present in the dwarf galaxy catalog if they indeed are dwarf galaxies! So we do not have to search the entire DESI spectroscopic catalog for matches!
    dwarf_cat = Table.read(READ THE ENTIRE DWARF GALAYX CATALOG)

    dwarf_cat_minimal = dwarf_cat["RA","DEC","TARGETID"]

    #make a smaller version of the entire dwarf catalog where we just store the ra, dec and targetid.

    
    # Create an astropy Column with object dtype to hold variable-length arrays
    new_col = Column(match_targetids, name='TARGETIDS_MATCHING', dtype=object)
    
    # Add the column to the table
    t.add_column(new_col)

    
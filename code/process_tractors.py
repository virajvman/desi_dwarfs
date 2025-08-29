
import numpy as np
import astropy.io.fits as fits
import astropy.units as u
import os
import sys
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.wcs import WCS
from astropy.table import Table, vstack, join
url_prefix = 'https://www.legacysurvey.org/viewer/'
from desiutil import brick
import fitsio
from desi_lowz_funcs import print_stage, check_path_existence, is_target_in_south, match_c_to_catalog, get_sweep_filename


def get_nearby_source_catalog(ra_k, dec_k, objid_k, brickid_k, box_size, wcat, brick_i, save_path_k, source_cat, source_pzs=None,save=True,primary=True):
    '''
    Function that gets source catalog within 45 arcsec radius of object
    primary = True if ra_k,dec_k are the main source. If False, then this function is just being used to get the relevant files!
    '''

    #define the center.see the nearby dwarf catalog mISTY ipynb for a visual confirmation
    center = SkyCoord(ra=ra_k * u.deg, dec=dec_k * u.deg)
    # Define the source catalog coordinates
    source_coords = SkyCoord(ra=source_cat["ra"] * u.deg,
                             dec=source_cat["dec"] * u.deg)
    # Compute angular separation
    separation = source_coords.separation(center)
    # Select sources within 45 arcsec. This was originally only for the 1.5 armin area, but now we will larger areas!
    # the 1.05 is just an extra boost to make sure we are capturing the any bright things along the edges.
    # this is especially true for stars as they can bleed along the pixels in vertical and horizontal direction
    radius_search = int(box_size*0.262/2)
    within_45arcsec = separation < radius_search * u.arcsec
    source_cat_f = source_cat[within_45arcsec]

    if primary == False and len(source_cat_f) == 0:
        #we have to do this because is it not possible to join empty tables
        #if no sources are find, then we just return None. So we know that no sources are supposed to be added!
        return None

    #rename columns
    source_cat_f.rename_column("brickid","BRICKID" )
    source_cat_f.rename_column("objid","OBJID" )

    if source_pzs is None:
        pass
    else:
        ## unique identifier hash is RELEASE,BRICKID,OBJID.
        # Perform the join operation on 'brickid' and 'objid'
        source_cat_f = join(source_cat_f, source_pzs, keys=['BRICKID', 'OBJID'], join_type='inner')

    for BAND in ("g", "r", "z"):
            source_cat_f[f"sigma_{BAND}"] = source_cat_f[f"flux_{BAND}"] * np.sqrt(source_cat_f[f"flux_ivar_{BAND}"])
    for BAND in ("g", "r", "z"):
            ##to avoid the many runtime warning messages, let us set the negative or zero flux values to nan!
            fluxs = source_cat_f[f"flux_{BAND}"]
            good_fluxs = np.where(fluxs > 0, fluxs, np.nan)
            source_cat_f[f"mag_{BAND}"] = 22.5 - 2.5*np.log10(good_fluxs)

    #filtering to ensure the source is detected at 5 sigma atleast in one band
    # source_cat_f = source_cat_f[ (source_cat_f["sigma_r"] > 3) | (source_cat_f["sigma_g"] > 3) | (source_cat_f["sigma_z"] > 3) ]

    ##compute some color information and errors on source of interest
    source_cat_f["g-r"] = source_cat_f["mag_g"] - source_cat_f["mag_r"]
    source_cat_f["r-z"] = source_cat_f["mag_r"] - source_cat_f["mag_z"]

    for BAND in ("g","r","z"):
        ##compute the errors in the mag assuming they are small :) 
        fluxs = source_cat_f[f"flux_{BAND}"]
        good_fluxs = np.where(fluxs > 0, fluxs, np.nan)

        flux_ivars = source_cat_f[f"flux_ivar_{BAND}"]
        good_flux_ivars = np.where(flux_ivars > 0, flux_ivars, np.nan)
        #we are converting the inverse variance into sigma here
        source_cat_f[f"mag_{BAND}_err"] = 1.087*(np.sqrt(1/good_flux_ivars) / good_fluxs ) 
    
    source_cat_f["g-r_err"] = np.sqrt(  source_cat_f["mag_g_err"]**2 + source_cat_f["mag_r_err"]**2)
    source_cat_f["r-z_err"] = np.sqrt(  source_cat_f["mag_r_err"]**2 + source_cat_f["mag_z_err"]**2)

    #remove if there are any nans in the data!
    # source_cat_f = source_cat_f[ ~np.isnan(source_cat_f["g-r_err"]) &  ~np.isnan(source_cat_f["r-z_err"]) & ~np.isnan(source_cat_f["g-r"]) &  ~np.isnan(source_cat_f["r-z"])  ]

    #We will not be filtering for nan sources now. We will apply that cut later as easier to downsample later

    if primary:
        #if we are working in the brick where the primary source is!
        ##however, if the source object is not in this (for eg. ELGs) we add it back
        #find difference between source and all the other catalog objects
        ref_coord = SkyCoord(ra=ra_k * u.deg, dec=dec_k * u.deg)
        
        catalog_coords = SkyCoord(ra=source_cat_f["ra"].data * u.deg, dec=source_cat_f["dec"].data * u.deg)
        # Compute separations
        
        separations = ref_coord.separation(catalog_coords).arcsec
        
        source_cat_obs = source_cat_f[np.argmin(separations)]
        
        #sometimes there are significant separations ~1 arcsec, due to the differences between the north and south targeting catalogs
        if source_cat_obs["OBJID"] != objid_k and source_cat_obs["BRICKID"] !=  brickid_k:
            if np.min(separations) < 0.1:
                print(f"The target source was not matching, but is within 0.1 arcsec. RA = {ra_k}, DEC = {dec_k}.")                
            else:
                print(f"The target source was not matching. Separations is {np.min(separations)}. RA = {ra_k}, DEC = {dec_k}")     
        else:
            pass   
    else:
        pass
    
    #save this file
    if save:
        source_cat_f.write(save_path_k + "/source_cat_f.fits",overwrite=True)
        return
    else:
        return source_cat_f


#the legacy survey definition of brick
bricks = brick.Bricks(bricksize=0.25)

def bricks_overlapping_circle(ra_center, dec_center, radius_arcsec, n_points=16):
    '''
    This function figures out the bricks that are overlapping a given circle. It also returns the ra,decs that were used to compute this!
    '''
    center = SkyCoord(ra=ra_center*u.deg, dec=dec_center*u.deg)
    radius = radius_arcsec * u.arcsec

    # Sample positions around the circle
    theta = np.linspace(0, 2*np.pi, n_points, endpoint=False)
    perimeter_coords = center.directional_offset_by(position_angle=theta*u.rad, separation=radius)

    # Combine center + perimeter points into one SkyCoord array
    all_coords = SkyCoord(
        ra=np.concatenate(([center.ra.deg], perimeter_coords.ra.deg)) * u.deg,
        dec=np.concatenate(([center.dec.deg], perimeter_coords.dec.deg)) * u.deg
    )

    bricknames = bricks.brickname(all_coords.ra.deg, all_coords.dec.deg)

    #this will be of length 16 and has not been filtered to be unique!
    return bricknames, all_coords.ra.deg, all_coords.dec.deg
    

def are_more_bricks_needed(ra,dec,radius_arcsec = 45):

    brick_names, all_ras, all_decs = bricks_overlapping_circle(ra, dec, radius_arcsec, n_points=16 )

    #get the relevant wcats and sweeps for this ra,dec!
    is_souths = is_target_in_south(all_ras,all_decs).astype(int)
    all_wcats = np.array(["north","south"])
    neigh_wcats = all_wcats[is_souths]

    neigh_sweeps = []
    for i in range(len(all_ras)):
        neigh_sweeps.append( get_sweep_filename( all_ras[i], all_decs[i] )  )
    neigh_sweeps = np.array(neigh_sweeps)
        
    #this is the original brick that contains the center
    brick_org = bricks.brickname(ra, dec)

    #also return the indices that make the array unique
    brick_uniqs, uni_inds = np.unique(brick_names, return_index = True)
    #use those indices to get the corresponding wcat values
    wcat_uniqs = neigh_wcats[uni_inds]
    sweeps_uniqs = neigh_sweeps[uni_inds]

    #get the bricks that are not the center one!
    neigh_bricks = brick_uniqs[brick_uniqs != brick_org]
    neigh_wcats = wcat_uniqs[brick_uniqs != brick_org]  
    neigh_sweeps = sweeps_uniqs[brick_uniqs != brick_org] 
    
    return neigh_bricks, neigh_wcats, neigh_sweeps


def get_neighboring_bricks(ra, dec, objid, brickid, box_size, neigh_bricks ,neigh_wcats, neigh_sweeps,use_pz = False):
    '''
    In this function, we check if a given source and the region of interest around it is fully contained in one brick or not.
    size is the radius of the circular region of interest in arcseconds
    '''

    #we are sitting at the boundary of multiple bricks! We will load each brick and perform the same cut!
    #is there an issue if I am sitting at the edge of north and south? What about the edge of a sweep?
    ## TO DO: ADD THE EDGE CASE WHERE WE ARE AT THE EDGE OF A SWEEP AS WELL
    # print_stage("%f, %f, We are sitting at the edge of more than 1 brick! Reading %d neighboring bricks"%(ra,dec, len(brick_uniqs) - 1) )

    new_sources = []

    for i,nbi in enumerate(neigh_bricks):
        source_cat_i  = read_source_cat(neigh_wcats[i], nbi)

        #read the corresponding pzs of this object?
        if use_pz:
            source_pzs_i = read_source_pzs(neigh_wcats[i],neigh_sweeps[i])
        else:
            source_pzs_i = None

        source_cat_i = get_nearby_source_catalog(ra, dec, objid, brickid, box_size, neigh_wcats[i], nbi, None, source_cat_i, source_pzs_i, save=False,primary=False)
        # print("%d objects read from the one of the neighboring bricks"%len(source_cat_i))
        if source_cat_i is not None:
            new_sources.append(source_cat_i)

    #stack'em all!
    if new_sources == []:
        return None
    else:
        new_sources = vstack(new_sources)
        #this will be the list of additional sources that will be tacked onto the existing source file!
        return new_sources


def return_sources_wneigh_bricks(save_path, ra, dec, objid, brickid, box_size, more_bricks, more_wcats, more_sweeps,use_pz = False):
    '''
    Function that saves the final source cat file that consolidates sources from the other bricks
    '''
    source_cat_f = Table.read(save_path + "/source_cat_f.fits")
    #neighboring bricks are needed
    # if os.path.exists(save_path_k + "/source_cat_f_more.fits"):
        
    #     #the combined bricks file already exists
    #     source_cat_f = Table.read( save_path_k + "/source_cat_f_more.fits"  )
    # else:
    #the combined brick file is being made!
    source_cat_more = get_neighboring_bricks(ra, dec, objid, brickid, box_size, more_bricks,more_wcats, more_sweeps,use_pz=use_pz)

    if source_cat_more is not None:
        source_cat_f = vstack([source_cat_f, source_cat_more])
    else:
        pass

    source_cat_f.write(save_path + "/source_cat_f_more.fits",overwrite=True)

    return



def read_source_cat(wcat, brick_i):
    '''
    Given north/south and brick name, read the corresponding source catalog!
    '''
    #read the source catalog for this brick
    if "p" in brick_i:
        brick_folder = brick_i[:brick_i.find("p")-1]
    else:
        brick_folder = brick_i[:brick_i.find("m")-1]


    imp_cols = ["ra","dec","ref_cat","type","pmra","pmdec","pmra_ivar","pmdec_ivar", "brickid", "objid", "gaia_phot_bp_mean_mag", "gaia_phot_rp_mean_mag", "gaia_phot_g_mean_mag"]


    #these other cols are for the source model reconstruction
    other_cols = ['bx', 'by', 'ref_id',
        'sersic', 'shape_r', 'shape_e1', 'shape_e2',
        'nobs_g', 'nobs_r', 'nobs_z',
        'psfdepth_g', 'psfdepth_r', 'psfdepth_z',
        'psfsize_g', 'psfsize_r', 'psfsize_z']

    include_cols = imp_cols + other_cols
    
    for bi in "grz":
        include_cols.append( "flux_%s"%bi )
        include_cols.append( "flux_ivar_%s"%bi )
        include_cols.append( "mw_transmission_%s"%bi )
        include_cols.append( "fracflux_%s"%bi )
        
    ## only read the columns I need
    source_cat = Table(fitsio.read("/global/cfs/cdirs/cosmo/data/legacysurvey/dr9/%s/tractor/%s/tractor-%s.fits"%(wcat, brick_folder,brick_i), columns = include_cols  ))

    return source_cat


def read_source_pzs(wcat, sweep_i):
    '''
    Function that reads the source photo-zs sweeps. sweep_i is taken from the "SWEEP" columns in the catalog
    '''
    include_cols = ["Z_PHOT_L95", "Z_PHOT_U95","Z_PHOT_L68", "Z_PHOT_U68", "Z_PHOT_STD", "Z_PHOT_MEAN", "BRICKID", "OBJID"]
    source_pzs_i = Table(fitsio.read( "/global/cfs/cdirs/cosmo/data/legacysurvey/dr9/%s/sweep/9.1-photo-z/"%wcat + sweep_i, columns = include_cols))

    return source_pzs_i

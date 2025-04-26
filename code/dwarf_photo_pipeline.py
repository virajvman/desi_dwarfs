import numpy as np
import astropy.io.fits as fits
import matplotlib as mpl
import matplotlib.pyplot as plt
import astropy.units as u
import os
import sys
import scipy.optimize as opt
from astropy import units as u
from astropy.coordinates import SkyCoord
import argparse
from astropy.io import fits
from astropy.wcs import WCS
from astropy.table import Table, vstack, join
import multiprocessing as mp
from tqdm import tqdm, trange
url_prefix = 'https://www.legacysurvey.org/viewer/'
import requests
from io import BytesIO
import matplotlib.patches as patches
from scarlet_photo import run_scarlet_pipe
from aperture_photo import run_aperture_pipe
# from get_sga_distances import get_sga_info
from desi_lowz_funcs import save_subimage, fetch_psf, generate_random_string
from desiutil import brick
import fitsio
from easyquery import Query, QueryMaker
import shutil

def parse_tgids(value):
    if not value:
        return None
    return [int(x) for x in value.split(',')]

    
def argument_parser():
    '''
    Function that parses the arguments passed while running a script
    '''
    result = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # path to the config file with parameters and information about the run
    result.add_argument('-sample', dest='sample', type=str, default = "BGS_BRIGHT") 
    #the sample is like BGS_BRIGHT, BGS_FAINT, etc.
    result.add_argument('-save_img', dest='save_img', type=str, default = "") 
    result.add_argument('-min', dest='min', type=int,default = 0)
    result.add_argument('-max', dest='max', type=int,default = 100000) 
    result.add_argument('-ncores', dest='ncores', type=int,default = 64) 
    result.add_argument('-tgids',dest="tgids_list", type=parse_tgids) 
    result.add_argument('-run_parr', dest='parallel',  action='store_true') 
    result.add_argument('-use_clean',dest='use_clean',action='store_true')
    result.add_argument('-overwrite',dest='overwrite',action='store_true')
    result.add_argument('-make_cats',dest='make_cats', action='store_true')
    result.add_argument('-run_w_source',dest='run_w_source', action='store_true')
    result.add_argument('-run_aper',dest='run_aper', action='store_true')
    result.add_argument('-run_scarlet',dest='run_scarlet', action='store_true')
    result.add_argument('-nchunks',dest='nchunks', type=int,default = 1)
    result.add_argument('-no_pz_aper',dest='no_pz_aper', action = "store_true")
    result.add_argument('-no_save',dest='no_save', action = "store_true")
    result.add_argument('-make_main_cats',dest='make_main_cats', action = "store_true")
    result.add_argument('-end_name',dest='end_name', type = str, default = "")
    

    return result

def save_cutouts(ra,dec,img_path,session, size=350, timeout = 30):
    url = url_prefix + f'cutout.fits?ra={ra}&dec={dec}&size=%s&'%size

    url += 'layer=ls-dr9&pixscale=0.262&bands=grz'

    try:
        resp = session.get(url, timeout=timeout)
        resp.raise_for_status()  # Raise error for bad status codes
        # Save the FITS file
        with open(img_path, "wb") as f:
            f.write(resp.content)
    
        #load the wcs and image data
        # img_data = fits.open(img_path)
        # data_arr = img_data[0].data
        # wcs = WCS(fits.getheader( img_path ))
    except:
        print("getting coadd image data failed!")
        print(url)
    
    return


def get_nearby_source_catalog(ra_k, dec_k, wcat, brick_i, save_path_k, source_cat, source_pzs,save=True,primary=True):
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
    # Select sources within 45 arcsec
    within_45arcsec = separation < 45 * u.arcsec
    source_cat_f = source_cat[within_45arcsec]
    
    if primary == False and len(source_cat_f) == 0:
        #we have to do this because is it not possible to join empty tables
        #if no sources are find, then we just return None. So we know that no sources are supposed to be added!
        return None

    #rename columns
    source_cat_f.rename_column("brickid","BRICKID" )
    source_cat_f.rename_column("objid","OBJID" )

    ## unique identifier hash is RELEASE,BRICKID,OBJID.
    # Perform the join operation on 'brickid' and 'objid'
    source_cat_f = join(source_cat_f, source_pzs, keys=['BRICKID', 'OBJID'], join_type='inner')

    for BAND in ("g", "r", "z"):
            source_cat_f[f"sigma_{BAND}"] = source_cat_f[f"flux_{BAND}"] * np.sqrt(source_cat_f[f"flux_ivar_{BAND}"])
    for BAND in ("g", "r", "z"):
            source_cat_f[f"mag_{BAND}"] = 22.5 - 2.5*np.log10(source_cat_f[f"flux_{BAND}"])

    #filtering to ensure the source is detected at 5 sigma atleast in one band
    source_cat_f = source_cat_f[ (source_cat_f["sigma_r"] > 5) | (source_cat_f["sigma_g"] > 5) | (source_cat_f["sigma_z"] > 5) ]

    ##compute some color information and errors on source of interest
    source_cat_f["g-r"] = source_cat_f["mag_g"] - source_cat_f["mag_r"]
    source_cat_f["r-z"] = source_cat_f["mag_r"] - source_cat_f["mag_z"]

    for BAND in ("g","r","z"):
        ##compute the errors in the mag assuming they are small :) 
        source_cat_f[f"mag_{BAND}_err"] = 1.087*(np.sqrt(1/source_cat_f[f"flux_ivar_{BAND}"]) / source_cat_f[f"flux_{BAND}"]) 
    
    
    source_cat_f["g-r_err"] = np.sqrt(  source_cat_f["mag_g_err"]**2 + source_cat_f["mag_r_err"]**2)
    source_cat_f["r-z_err"] = np.sqrt(  source_cat_f["mag_r_err"]**2 + source_cat_f["mag_z_err"]**2)

    #remove if there are any nans in the data!
    source_cat_f = source_cat_f[ ~np.isnan(source_cat_f["g-r_err"]) &  ~np.isnan(source_cat_f["r-z_err"]) & ~np.isnan(source_cat_f["g-r"]) &  ~np.isnan(source_cat_f["r-z"])  ]

    if primary:
        #if we are working in the brick where the primary source is!
        ##however, if the source object is not in this (for eg. ELGs) we add it back
        #find difference between source and all the other catalog objects
        ref_coord = SkyCoord(ra=ra_k * u.deg, dec=dec_k * u.deg)
        
        catalog_coords = SkyCoord(ra=source_cat_f["ra"].data * u.deg, dec=source_cat_f["dec"].data * u.deg)
        # Compute separations
        
        separations = ref_coord.separation(catalog_coords).arcsec
        
        source_cat_obs = source_cat_f[np.argmin(separations)]
        
        if np.min(separations) != 0:
            #the source we pointed at has been removed
            tgt_source = source_cat[ (source_cat["ra"] == ra_k) & (source_cat["dec"] == dec_k) ]
            if len(tgt_source) == 0:
                #hmm no target source found?
                print(ra_k, dec_k)
                print("Hmm. The target source was not found.")
            source_cat_f = vstack( [source_cat_f,  tgt_source ] )
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


def get_neighboring_bricks(ra, dec, neigh_bricks ,neigh_wcats, neigh_sweeps):
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
        source_pzs_i = read_source_pzs(neigh_wcats[i],neigh_sweeps[i])

        source_cat_i = get_nearby_source_catalog(ra, dec, neigh_wcats[i], nbi, None, source_cat_i, source_pzs_i, save=False,primary=False)
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


def return_sources_wneigh_bricks(save_path, ra, dec, more_bricks, more_wcats, more_sweeps):
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
    source_cat_more = get_neighboring_bricks(ra, dec, more_bricks,more_wcats, more_sweeps)

    if source_cat_more is not None:
        source_cat_f = vstack([source_cat_f, source_cat_more])
    else:
        pass

    source_cat_f.write(save_path + "/source_cat_f_more.fits",overwrite=True)

    return

def get_relevant_files_scarlet(input_dict):
    '''
    Get the relevant files for a single object!
    '''

    tgid_k = input_dict["TARGETID"]
    samp_k = input_dict["SAMPLE"]
    ra_k = input_dict["RA"]
    dec_k = input_dict["DEC"]
    top_folder = input_dict["top_folder"]
    sweep_folder = input_dict["sweep_folder"]
    brick_i = input_dict["brick_i"]
    wcat = input_dict["wcat"]
    session = input_dict["session"]
    
    top_path_k = top_folder + "/%s/"%wcat + sweep_folder + "/" + brick_i + "/%s_tgid_%d"%(samp_k, tgid_k) 
        
    #check if psf already exists
    if os.path.exists(top_path_k + "/psf_data_z.npy"):
        pass
    else:
        psf_dict = fetch_psf(ra_k,dec_k, session)
        if psf_dict is not None:
            #as psf can be of different sizes and so saving each band separately!
            np.save(top_path_k  + "/psf_data_g.npy",psf_dict["g"])
            np.save(top_path_k  + "/psf_data_r.npy",psf_dict["r"])
            np.save(top_path_k  + "/psf_data_z.npy",psf_dict["z"])

    
    #similarly, get the subimage!
    sbimg_path = top_path_k + "/grz_subimage.fits"
    if os.path.exists(sbimg_path):
        pass
    else:
        save_subimage(ra_k, dec_k, sbimg_path, session, size = 350, timeout = 30)

    ##done saving all the files!
    return

def get_relevant_files_aper(input_dict):
    '''
    Get the relevant files for a single object!
    '''

    tgid_k = input_dict["TARGETID"]
    samp_k = input_dict["SAMPLE"]
    ra_k = input_dict["RA"]
    dec_k = input_dict["DEC"]
    redshift_k = input_dict["Z"]
    objid_k = input_dict["OBJID"]
    top_folder = input_dict["top_folder"]
    sweep_folder = input_dict["sweep_folder"]
    brick_i = input_dict["brick_i"]
    wcat = input_dict["wcat"]
    source_pzs_i = input_dict["source_pzs_i"]
    source_cat = input_dict["source_cat"]
    session = input_dict["session"]
    
    top_path_k = top_folder + "/%s/"%wcat + sweep_folder + "/" + brick_i + "/%s_tgid_%d"%(samp_k, tgid_k)

    print(top_path_k)

    check_path_existence(all_paths=[top_path_k])
    #inside this folder, we will save all the relevant files and info!!
    image_path =  top_folder + "_cutouts/image_tgid_%d_ra_%f_dec_%f.fits"%(tgid_k,ra_k,dec_k)

    ## save the source catalog 
    # if os.path.exists(top_path_k + "/source_cat_f.fits"):
    #     pass
    # else:

    #TODO: ADD FEATURE TO REMOVE ALL THE PREVIOUS SOURCE FILE IN CASES THAT CREATES CONFUSION
    
    get_nearby_source_catalog(ra_k, dec_k, wcat, brick_i, top_path_k, source_cat, source_pzs_i)
    
    ## check if the source is at the edge of the brick, if so we will need to combine stuff
    more_bricks, more_wcats, more_sweeps = are_more_bricks_needed(ra_k,dec_k,radius_arcsec = 45)

    if len(more_bricks) == 0:
        #there are no neighboring bricks needed
        pass
    else:
        return_sources_wneigh_bricks(top_path_k, ra_k, dec_k, more_bricks, more_wcats, more_sweeps)
        

    if os.path.exists(image_path):
        pass
    else:
        #if this image path does not exist, does it exist in the other folder?
        if "all_deshreds" in image_path:
            image_path_other = image_path.replace("all_deshreds","all_good")
        if "all_good" in image_path:
            image_path_other = image_path.replace("all_good","all_deshreds")
            
        if os.path.exists(image_path_other):
            print(image_path_other)
            print(image_path)
            #copy it
            shutil.copy(image_path_other, image_path)
        else:
            #we need to obtain the cutout data!
            save_cutouts(ra_k,dec_k,image_path,session,size=350)
        
    ##done saving all the files!
    return

def make_clean_shreds_catalogs():
    '''
    This function makes the primary clean and shred catalogs we work with!
    '''

    bgsb_list = Table.read("/pscratch/sd/v/virajvm/catalog_dr1_dwarfs/iron_bgs_bright_filter_zsucc_zrr02_allfracflux.fits")
    bgsf_list = Table.read("/pscratch/sd/v/virajvm/catalog_dr1_dwarfs/iron_bgs_faint_filter_zsucc_zrr03_allfracflux.fits")
    elg_list = Table.read("/pscratch/sd/v/virajvm/catalog_dr1_dwarfs/iron_elg_filter_zsucc_zrr05_allfracflux.fits")
    lowz_list = Table.read("/pscratch/sd/v/virajvm/catalog_dr1_dwarfs/iron_lowz_filter_zsucc_zrr03.fits")

    from desi_lowz_funcs import get_sweep_filename, save_table, is_target_in_south, get_sga_norm_dists, get_sga_norm_dists_FAST
    from construct_dwarf_galaxy_catalogs import bright_star_filter

    # ##add the sweep, catalog info
    # bgsb_list = add_sweeps_column(bgsb_list, "/pscratch/sd/v/virajvm/catalog_dr1_dwarfs/iron_bgs_bright_filter_zsucc_zrr02_allfracflux.fits")
    # bgsf_list = add_sweeps_column(bgsf_list, "/pscratch/sd/v/virajvm/catalog_dr1_dwarfs/iron_bgs_faint_filter_zsucc_zrr03_allfracflux.fits")
    # elg_list = add_sweeps_column(elg_list, "/pscratch/sd/v/virajvm/catalog_dr1_dwarfs/iron_elg_filter_zsucc_zrr05_allfracflux.fits")
    # lowz_list = add_sweeps_column(lowz_list, "/pscratch/sd/v/virajvm/catalog_dr1_dwarfs/iron_lowz_filter_zsucc_zrr03.fits")

    #identifying the objects that have the 
    fracflux_grz = [f"FRACFLUX_{b}" for b in "GRZ"]
    # rchisq_grz = [f"RCHISQ_{b}" for b in "GRZ"]

    remove_queries = [Query(_n_or_more_lt(fracflux_grz, 2, 0.2)) ]
    
    # note that the this is n_or_more_LT!! so be careful about that!
    #these are masks for objects that did not satisfy the above condition!
    bgsb_mask = get_remove_flag(bgsb_list, remove_queries) == 0
    bgsf_mask = get_remove_flag(bgsf_list, remove_queries) == 0
    elg_mask = get_remove_flag(elg_list, remove_queries) == 0
    lowz_mask = get_remove_flag(lowz_list, remove_queries) == 0

    #invert the above to get the clean mask!
    clean_mask_bgsb = (~bgsb_mask)
    clean_mask_bgsf = (~bgsf_mask)
    clean_mask_elg = (~elg_mask)
    clean_mask_lowz = (~lowz_mask)
    

    ## what is correct fracflux limit
    ## I want to use the fraflux limit for z<0.05, not really needed for z>0.05 objects
    # fracflux_limit = 0.15
    
    # clean_mask_bgsb = (bgsb_list["FRACFLUX_R"] < fracflux_limit) & (bgsb_list["FRACFLUX_G"] < fracflux_limit) & (bgsb_list["FRACFLUX_Z"] < fracflux_limit)
    # clean_mask_bgsf = (bgsf_list["FRACFLUX_R"] < fracflux_limit) & (bgsf_list["FRACFLUX_G"] < fracflux_limit) & (bgsf_list["FRACFLUX_Z"] < fracflux_limit)
    # clean_mask_elg = (elg_list["FRACFLUX_R"] < fracflux_limit) & (elg_list["FRACFLUX_G"] < fracflux_limit) & (elg_list["FRACFLUX_Z"] < fracflux_limit)
    # clean_mask_lowz = (lowz_list["FRACFLUX_R"] < fracflux_limit) & (lowz_list["FRACFLUX_G"] < fracflux_limit) & (lowz_list["FRACFLUX_Z"] < fracflux_limit)
    
    
    ##construct the clean catalog
    ## to make things easy in the cleaning stage, we will use the optical based colors

    dwarf_mask_bgsb = (bgsb_list["LOGM_SAGA"] < 9.5) 
    dwarf_mask_bgsf = (bgsf_list["LOGM_SAGA"] < 9.5) 
    dwarf_mask_lowz = (lowz_list["LOGM_SAGA"] < 9.5) 
    dwarf_mask_elg = (elg_list["LOGM_SAGA"] < 9.5)   

    bgsb_clean_dwarfs = bgsb_list[ clean_mask_bgsb  & (dwarf_mask_bgsb) ] #& (bgsb_list["LOGM_SAGA"] > 7)

    bgsf_clean_dwarfs = bgsf_list[ clean_mask_bgsf  & (dwarf_mask_bgsf) ] # & (bgsf_list["LOGM_SAGA"] > 7)

    lowz_clean_dwarfs = lowz_list[ clean_mask_lowz  & dwarf_mask_lowz ]

    elg_clean_dwarfs = elg_list[ clean_mask_elg & dwarf_mask_elg  ] # & (elg_list["LOGM_SAGA"] > 7)

    bgsb_clean_dwarfs["SAMPLE"] = np.array(len(bgsb_clean_dwarfs)*["BGS_BRIGHT"])
    bgsf_clean_dwarfs["SAMPLE"] = np.array(len(bgsf_clean_dwarfs)*["BGS_FAINT"])
    elg_clean_dwarfs["SAMPLE"] = np.array(len(elg_clean_dwarfs)*["ELG"])
    lowz_clean_dwarfs["SAMPLE"] = np.array(len(lowz_clean_dwarfs)*["LOWZ"])

    all_clean_dwarfs = vstack( [ bgsb_clean_dwarfs, bgsf_clean_dwarfs, lowz_clean_dwarfs, elg_clean_dwarfs] )
    
    all_clean_dwarfs.remove_column("OII_FLUX")
    all_clean_dwarfs.remove_column("OII_FLUX_IVAR")
    all_clean_dwarfs.remove_column("Z_HPX")
    all_clean_dwarfs.remove_column("TARGETID_2")
    all_clean_dwarfs.remove_column("OII_SNR")
    all_clean_dwarfs.remove_column("ZSUCC")

    print("Total number of galaxies in clean catalog=",len(all_clean_dwarfs))

    ## we add the SGA information to the objects now!
    all_clean_dwarfs = get_sga_norm_dists_FAST(all_clean_dwarfs, siena_path="/global/cfs/cdirs/cosmo/data/sga/2020/SGA-2020.fits")

    #load the existing file
    clean_all_exist = Table.read("/pscratch/sd/v/virajvm/catalog_dr1_dwarfs/desi_y1_dwarf_clean_catalog_v3.fits")
    
    # does existing file also already have bright star info?
    bstar_keys = [ "STARFDIST", "STARDIST_DEG","STARMAG", "STAR_RADIUS_ARCSEC", "STAR_RA","STAR_DEC"]
    
    try:
        for ki in bstar_keys:
            all_clean_dwarfs[ki] = clean_all_exist[ki].data
        print("Bright star information already existed. Copying values from there.")    
    except:
        #if not, then we have to recompute it!
        all_clean_dwarfs = bright_star_filter(all_clean_dwarfs)

    del clean_all_exist

    # save the clean dwarfs now!
    save_table(all_clean_dwarfs,"/pscratch/sd/v/virajvm/catalog_dr1_dwarfs/desi_y1_dwarf_clean_catalog_v3.fits",comment="This is a compilation of dwarf galaxy candidates in DESI Y1 data from the BGS Bright, BGS Faint, ELG and LOW-Z samples. Only galaxies with LogMstar < 9.5 (w/SAGA based stellar masses) and that have robust photometry are included.")
        
    ##applying the mask and then stacking them!!

    bgsb_shreds = bgsb_list[ ~clean_mask_bgsb & dwarf_mask_bgsb]
    bgsf_shreds = bgsf_list[  ~clean_mask_bgsf & dwarf_mask_bgsf]
    elg_shreds = elg_list[ ~clean_mask_elg & dwarf_mask_elg]
    lowz_shreds = lowz_list[ ~clean_mask_lowz & dwarf_mask_lowz]
    
    #many elgs are below this stellar mass cut and so we do not apply it
    #even including clean dwarfs that are below LogMstar < 7 to confirm their existence!
    # bgsb_shreds_p1 = bgsb_list[ clean_mask_bgsb & (bgsb_list["LOGM_SAGA"] <= 7) ]
    # bgsf_shreds_p1 = bgsf_list[ clean_mask_bgsf & (bgsf_list["LOGM_SAGA"] <= 7)  ]
    # lowz_shreds = lowz_list[lowz_list["LOGM_SAGA"] <= 7]
    # bgsb_shreds_p1["SAMPLE"] = np.array(["BGS_BRIGHT"]*len(bgsb_shreds_p1))
    # bgsf_shreds_p1["SAMPLE"] = np.array(["BGS_FAINT"]*len(bgsf_shreds_p1))
    # elg_shreds_p1["SAMPLE"] = np.array(["ELG"]*len(elg_shreds_p1))
    
    #adding the sample column
    bgsb_shreds["SAMPLE"] = np.array(["BGS_BRIGHT"]*len(bgsb_shreds))
    bgsf_shreds["SAMPLE"] = np.array(["BGS_FAINT"]*len(bgsf_shreds))
    elg_shreds["SAMPLE"] = np.array(["ELG"]*len(elg_shreds))
    lowz_shreds["SAMPLE"] = np.array(["LOWZ"]*len(lowz_shreds))

    #stacking all the shreds now 
    # shreds_all = vstack( [bgsb_shreds, bgsf_shreds, elg_shreds, lowz_shreds, bgsb_shreds_p1,bgsf_shreds_p1]) #, elg_shreds_p1 ] )
    shreds_all = vstack( [bgsb_shreds, bgsf_shreds, elg_shreds, lowz_shreds ])
    
    shreds_all.remove_column("OII_FLUX")
    shreds_all.remove_column("OII_FLUX_IVAR")
    shreds_all.remove_column("Z_HPX")
    shreds_all.remove_column("TARGETID_2")
    shreds_all.remove_column("OII_SNR")
    shreds_all.remove_column("ZSUCC")
    
    print("Total number of objects whose photometry needs to be redone = ", len(shreds_all))

    shreds_ra, shreds_dec, shreds_z = shreds_all["RA"].data, shreds_all["DEC"].data, shreds_all["Z"].data

    ##if the existing file already has SGA info, then we just add that
    shreds_all = get_sga_norm_dists_FAST(shreds_all, siena_path="/global/cfs/cdirs/cosmo/data/sga/2020/SGA-2020.fits")

    #load the existing file
    shreds_all_exist = Table.read("/pscratch/sd/v/virajvm/catalog_dr1_dwarfs/desi_y1_dwarf_shreds_catalog_v3.fits")
    
    try:
        for ki in bstar_keys:
            shreds_all[ki] = shreds_all_exist[ki].data
        print("Bright star information already existed. Copying values from there.")
            
    except:
        #if not, then we have to recompute it!
        shreds_all = bright_star_filter(shreds_all)

    del shreds_all_exist
    
    save_table(shreds_all,"/pscratch/sd/v/virajvm/catalog_dr1_dwarfs/desi_y1_dwarf_shreds_catalog_v3.fits")
    
    ## PLOTTING THE FRACTION OF GALAXIES THAT ARE LIKELY SHREDS/BAD PHOTOMETRY
    zgrid = np.arange(0.00,0.15,0.005)

    def get_shred_frac(tot_cat, sample,zlow, zhi, shred_mask):
        shreds_count = len(tot_cat[ (tot_cat["SAMPLE"] == sample) & (tot_cat["Z"] < zhi) & (tot_cat["Z"] > zlow) & shred_mask ] )
        tot_count = len(tot_cat[ (tot_cat["SAMPLE"] == sample) & (tot_cat["Z"] < zhi) & (tot_cat["Z"] > zlow) ] )
    
        if tot_count > 0:
            return shreds_count / tot_count
        else:
            return None

    shred_frac_all_bgsb = []
    shred_frac_all_bgsf = []
    shred_frac_all_elg = []
    shred_frac_all_lowz = []
    
    bgsb_list["SAMPLE"] = np.array(len(bgsb_list)*["BGS_BRIGHT"])
    bgsf_list["SAMPLE"] = np.array(len(bgsf_list)*["BGS_FAINT"])
    elg_list["SAMPLE"] = np.array(len(elg_list)*["ELG"])
    lowz_list["SAMPLE"] = np.array(len(lowz_list)*["LOWZ"])
    
    #also fitlering for dwarf candidates as we only care about those!
    all_list = vstack( [bgsb_list[dwarf_mask_bgsb], bgsf_list[dwarf_mask_bgsf], elg_list[dwarf_mask_elg], lowz_list[dwarf_mask_lowz] ] )

    # mask_shred = ~((all_list["FRACFLUX_G"] < fracflux_limit) & (all_list["FRACFLUX_R"] < fracflux_limit) & (all_list["FRACFLUX_Z"] < fracflux_limit))

    remove_queries = [Query(_n_or_more_lt(fracflux_grz, 2, 0.2)) ]
    # note that the this is n_or_more_LT!! so be careful about that!
    #these are masks for objects that did not satisfy the above condition!
    mask_shred = get_remove_flag(all_list, remove_queries) == 0

    
    for i in trange(len(zgrid)-1):
        zlow = zgrid[i]
        zhi = zgrid[i+1]

        shred_frac_all_bgsb.append(   get_shred_frac(all_list, "BGS_BRIGHT",zlow, zhi, mask_shred)  ) 
        shred_frac_all_bgsf.append(   get_shred_frac(all_list, "BGS_FAINT",zlow, zhi, mask_shred)  ) 
        shred_frac_all_elg.append(   get_shred_frac(all_list, "ELG",zlow, zhi, mask_shred)  ) 
        shred_frac_all_lowz.append(   get_shred_frac(all_list, "LOWZ",zlow, zhi, mask_shred)  ) 
        


    bgs_col = "#648FFF" #DC267F
    lowz_col = "#DC267F"
    elg_col = "#FFB000"
    
    zcens = 0.5*(zgrid[1:] + zgrid[:-1])
    
    plt.figure(figsize = (5,5))
    plt.plot(zcens, shred_frac_all_bgsb,label = "BGS Bright",lw = 3,color = bgs_col,ls = "-",alpha = 0.75)
    
    plt.plot(zcens, shred_frac_all_bgsf,label = "BGS Faint",lw = 3,color = "r",ls = "-",alpha = 0.75)

    plt.plot(zcens, shred_frac_all_lowz,label = "LOWZ",lw = 3,color = lowz_col,ls = "-",alpha = 0.75)
    
    plt.plot(zcens, shred_frac_all_elg,label = "ELG",lw = 3,color = elg_col,ls = "-",alpha = 0.75)
    
    plt.legend(fontsize = 12)
    plt.xlim([0,0.1])
    plt.ylim([0,1])
    plt.xlabel("z (Redshift)",fontsize = 15)
    plt.ylabel(r"Likely Shredded Source Fraction",fontsize = 15)
    plt.grid(ls=":",color = "lightgrey",alpha = 0.5)
    plt.savefig("/global/homes/v/virajvm/DESI2_LOWZ/quenched_fracs_nbs/paper_plots/frac_shreds.pdf", bbox_inches="tight")
    plt.show()
        
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


if __name__ == '__main__':

    import warnings
    from astropy.units import UnitsWarning
    from astropy.wcs import FITSFixedWarning
    import numpy as np
    np.seterr(invalid='ignore')
    warnings.simplefilter('ignore', category=UnitsWarning)
    warnings.simplefilter('ignore', FITSFixedWarning)
    
    rootdir = '/global/u1/v/virajvm/'
    sys.path.append(os.path.join(rootdir, 'DESI2_LOWZ'))
    from desi_lowz_funcs import print_stage, check_path_existence, get_remove_flag, _n_or_more_lt, is_target_in_south, match_c_to_catalog, calc_normalized_dist, get_sweep_filename, get_random_markers, save_table, make_subplots, _n_or_more_gt, _n_or_more_lt

    mpl.rcParams['font.family'] = 'serif'
    mpl.rcParams['axes.linewidth'] = 1.5
    mpl.rcParams['axes.xmargin'] = 1
    mpl.rcParams['xtick.labelsize'] = 'x-large'
    mpl.rcParams['xtick.major.size'] = 5
    mpl.rcParams['xtick.major.width'] = 1.5
    mpl.rcParams['ytick.labelsize'] = 'x-large'
    mpl.rcParams['ytick.major.size'] = 5
    mpl.rcParams['ytick.major.width'] = 1.5
    mpl.rcParams['legend.frameon'] = False
    # use a good colormap and don't interpolate the pixels
    mpl.rc('image', cmap='viridis', interpolation='none', origin='lower')

    # read in command line arguments
    args = argument_parser().parse_args()

    #sample_str could also be multiple samples together!
    sample_str = args.sample

    sample_list = False
    if "," in sample_str:
        sample_list = True
        all_samples  = sample_str.split(",")

    min_ind = args.min
    max_ind = args.max
    run_parr = args.parallel
    #use clean catalog or shred catalog
    use_clean = args.use_clean
    ncores = args.ncores
    #if we will be using user-input catalog or not
    #will we overwrite/redo the image segmentation part?
    overwrite_bool = args.overwrite
    make_cats = args.make_cats
    tgids_list = args.tgids_list
    #this will be the name of pdf file where we store summary images
    summary_scroll_file_name = args.save_img
    nchunks = args.nchunks
    #if no_save is true, a summary file will NOT be saved
    no_save = args.no_save
    #this is for whether we will run the make shreds and clean catalog function
    make_main_cats = args.make_main_cats
    end_name = args.end_name
    
    run_w_source = args.run_w_source
    no_pz_aper = args.no_pz_aper    
    
    run_aper = args.run_aper
    run_scarlet = args.run_scarlet


    run_own_detect = not run_w_source
    #whether to use photo-z in separating sources
    use_pz_aper = not no_pz_aper

    #this is the flag that is used in the file names 
    if use_pz_aper:
        pz_flag = "w_pz"
    else:
        pz_flag = "no_pz"
    
    ## can I come up with a robust way to choose box size?
    box_size = 350
        
    c_light = 299792 #km/s

    ##################
    ##PART 1: Make the clean and shredded catalogs!
    ##################

    if make_main_cats:
        make_clean_shreds_catalogs()

    #information on sga and bright stars is added in the above function

    ##################
    ##PART 2: Generate nested folder structure with relevant files for doing photometry
    ##################

    # creating a single session!
    session = requests.Session()

    ##load the relevant catalogs!
    if use_clean==False:
        shreds_all = Table.read("/pscratch/sd/v/virajvm/catalog_dr1_dwarfs/desi_y1_dwarf_shreds_catalog_v3.fits")
        # shreds_all = shreds_all[  shreds_all["SGA_D26_NORM_DIST"] > 1.5 ]
    else:
        shreds_all = Table.read("/pscratch/sd/v/virajvm/catalog_dr1_dwarfs/desi_y1_dwarf_clean_catalog_v3.fits")

    
    ## filtering for the sample now 
    if sample_list:
        #we have provided a list of samples
        data_samples = shreds_all['SAMPLE'].data.astype(str)
        all_samples = np.array(all_samples).astype(str)
        print(all_samples)
        
        sample_mask = np.isin(data_samples, all_samples) 
        
    else:
        #only a single sample is given!
        sample_mask = (shreds_all["SAMPLE"] ==  sample_str) # & (shreds_all["Z"] < 0.007)

    shreds_focus = shreds_all[sample_mask]

    print(len(shreds_focus))

    ##finally, if a list of targetids is provided, then we only select those
    if tgids_list is not None:
        print("List of targetids to process:",tgids_list)
        shreds_focus = shreds_focus[np.isin(shreds_focus['TARGETID'], np.array(tgids_list) )]
        
        print("Number of targetids to process =", len(shreds_focus))

    #apply the max_ind cut if relevant
    max_ind = np.minimum( max_ind, len(shreds_focus) )
    shreds_focus = shreds_focus[min_ind:max_ind]

    if make_cats == True:    
        print_stage("Generating relevant files for doing aperture photometry")
        print("Using cleaned catalogs =",use_clean==True)
    
        print("Number of objects whose photometry will be redone = ", len(shreds_focus) )
    
        if use_clean == False:
            top_folder = "/pscratch/sd/v/virajvm/redo_photometry_plots/all_deshreds"
        else:
            top_folder = "/pscratch/sd/v/virajvm/redo_photometry_plots/all_good"
            
        print(len(shreds_focus[shreds_focus["is_south"] == 1]))
        print(len(shreds_focus[shreds_focus["is_south"] == 0]))
    
        for wcat_ind, wcat in enumerate(["north","south"]):
            check_path_existence(all_paths=[top_folder + "/%s"%wcat])

            ##we can invert this if we want to make sure all cats are being made without long code times
            unique_sweeps = np.unique( shreds_focus[shreds_focus["is_south"] == wcat_ind]["SWEEP"])
            print(len(unique_sweeps), f"unique sweeps found in {wcat}")
    
            #this is just to make sure the very end sweeps have their files made as well
            # unique_sweeps = unique_sweeps[::-1]
    
            #counter for number of sweeps done
            sweeps_done=0
            
            for sweep_i in unique_sweeps:
                
                print(f"Sweeps done = {sweeps_done}/{len(unique_sweeps)}")
                sweeps_done += 1
                #make a sweep folder if it does not exist
                sweep_folder = sweep_i.replace("-pz.fits","")
                check_path_existence(all_paths=[top_folder + "/%s/"%wcat + sweep_folder ])
                
                ##load the relevant sweep photo-z file
                #only read the relevant columns of interest!
                source_pzs_i = read_source_pzs(wcat, sweep_i)
                
                #get all the galaxies that fall in this sweep
                shreds_focus_i = shreds_focus[(shreds_focus["SWEEP"] == sweep_i) & (shreds_focus["is_south"] == wcat_ind)]
        
                #get all the unique brick names for this source
                brick_names_i = np.unique( shreds_focus_i["BRICKNAME"] )
    
                ## for a given sweep can we parallelize this??
                print("Total number of bricks in this sweep = ", len(brick_names_i) )
                print("Total number of galaxies in this sweep = ", len(shreds_focus_i) )

                ##prepare all inputs now!
    
                all_source_cats = {}
    
                #looping over all the bricks and saving the source cat files!
                for brick_i in tqdm(brick_names_i):
                    #make a folder for each brick
                    check_path_existence(all_paths=[top_folder + "/%s/"%wcat + sweep_folder  + "/" + brick_i])
    
                    #inside this brick folder, we will have folders for each galaxy then
    
                    shreds_focus_ij = shreds_focus_i[ shreds_focus_i["BRICKNAME"] == brick_i]
    
                    #read the source catalog for this brick
                    source_cat = read_source_cat(wcat, brick_i)
                    
                    all_source_cats[brick_i] = source_cat


                ## so not how what is made parallel here is per galaxy
                ## what happens if I make the parallel the bricks themselves and the galaxies in each brick are serial
                ## As I have more galaxies than bricks? This does not make sense?
    
                #list of dictionaries
                all_input_dicts = []
    
                ##looping over all the galaxies in this sweep file to get get the input dict file
                for i in range(len(shreds_focus_i)):
                    temp = { "TARGETID": shreds_focus_i["TARGETID"][i], "SAMPLE": shreds_focus_i["SAMPLE"][i], "RA": shreds_focus_i["RA"][i], 
                            "DEC": shreds_focus_i["DEC"][i], "Z": shreds_focus_i["Z"][i], "OBJID": shreds_focus_i["OBJID"][i],
                            "wcat":wcat, "brick_i":shreds_focus_i["BRICKNAME"][i], "sweep_folder":sweep_folder, "top_folder":top_folder,
                            "source_cat" : all_source_cats[ shreds_focus_i["BRICKNAME"][i] ], "source_pzs_i": source_pzs_i, 
                           "session":session }
    
                    all_input_dicts.append(temp)
    
                ## get the relevant files for the the 
                # with mp.Pool(processes=ncores) as pool:
                #     results = list(tqdm(pool.imap(get_relevant_files_aper, all_input_dicts), total = len(all_input_dicts)  ))

                #as we are not returning anything, this is faster than pool.imap
                with mp.Pool(processes=ncores) as pool:
                    pool.map_async(get_relevant_files_aper, all_input_dicts)
                    pool.close()
                    pool.join()
                     
    ##################
    ##PART 3: Preparing inputs for the aperture and/or scarlet photometry functions
    ##################

    ## is photometry being run or no?
    if run_aper | run_scarlet:
        
        print_stage("Constructing input files for the photometry functions!")
    
        all_wcats = ["north","south"]
    
        ##prepare the catalogs on which photo pipeline will be run
        if use_clean == False:
            #we deal with the shredded catalogs
            
            #the path where we will also collect all the final summary figures
            top_folder = "/pscratch/sd/v/virajvm/redo_photometry_plots/all_deshreds"
            save_sample_path = "/pscratch/sd/v/virajvm/redo_photometry_plots/all_redo_figures/%s/"%sample_str
        
            check_path_existence(all_paths=[save_sample_path])
    
        else:
            #we deal with the nice clean catalogs
            #this is to check for robustness for our aperture photo pipeline
     
            top_folder = "/pscratch/sd/v/virajvm/redo_photometry_plots/all_good"
            
            #the path where we will also collect all the final summary figures
            save_sample_path = "/pscratch/sd/v/virajvm/redo_photometry_plots/all_redo_figures/%s_good/"%sample_str
        
            check_path_existence(all_paths=[save_sample_path])
    
                    
        def produce_input_dicts(k):
            '''
            Function that produces the input dictionaries that will be fed to both the scarlet and aperture photometry functions
            '''
    
            tgid_k = shreds_focus["TARGETID"][k]
            ra_k = shreds_focus["RA"][k]
            dec_k = shreds_focus["DEC"][k]
            redshift_k = shreds_focus["Z"][k]
            sweep_k = shreds_focus["SWEEP"][k]
            brick_k = shreds_focus["BRICKNAME"][k]
    
            #get the bright star info
            bstar_ra = shreds_focus["STAR_RA"][k]
            bstar_dec = shreds_focus["STAR_DEC"][k]
            bstar_radius = shreds_focus["STAR_RADIUS_ARCSEC"][k]
            bstar_fdist = shreds_focus["STARFDIST"][k]
    
            sample_str_i = shreds_focus["SAMPLE"][k]
    
            #first one is SGA distance in degrees
            sga_dist = shreds_focus["SGA_DIST_DEG"][k]
            sga_ndist = shreds_focus["SGA_D26_NORM_DIST"][k]
            
            wcat_k = all_wcats[int(shreds_focus["is_south"][k])]
    
            sweep_folder = sweep_k.replace("-pz.fits","")
    
    
            if use_clean == False:
                
                save_path_k = top_folder + "/%s/"%wcat_k + sweep_folder + "/" + brick_k + "/%s_tgid_%d"%(sample_str_i, tgid_k)
                
                img_path_k = "/pscratch/sd/v/virajvm/redo_photometry_plots/all_deshreds_cutouts/image_tgid_%d_ra_%f_dec_%f.fits"%(tgid_k,ra_k,dec_k) 
            else:
        
                save_path_k = top_folder + "/%s/"%wcat_k + sweep_folder + "/" + brick_k + "/%s_tgid_%d"%(sample_str_i, tgid_k)
                    
                img_path_k = "/pscratch/sd/v/virajvm/redo_photometry_plots/all_good_cutouts/image_tgid_%d_ra_%f_dec_%f.fits"%(tgid_k,ra_k,dec_k) 
    
    
            if tgids_list is not None:
                print(save_path_k)
    
            ## check if the source is at the edge of the brick, if so we will need to combine stuff
            more_bricks, more_wcats, more_sweeps = are_more_bricks_needed(ra_k,dec_k,radius_arcsec = 45)
            
            if len(more_bricks) == 0:
                #there are no neighboring bricks needed
                source_cat_f = Table.read(save_path_k + "/source_cat_f.fits")
            else:
                if os.path.exists(save_path_k + "/source_cat_f_more.fits"):
                    source_cat_f = Table.read(save_path_k + "/source_cat_f_more.fits")
                else:
                   ##this means that there this source was missed!
                    print_stage("Multiple bricks are intersecting and was not accounted for. Getting all the sources now!")
                    
                    return_sources_wneigh_bricks(save_path_k, ra_k, dec_k, more_bricks, more_wcats, more_sweeps)
                    source_cat_f = Table.read(save_path_k + "/source_cat_f_more.fits")
            
                
            # if os.path.exists(save_path_k + "/source_cat_f.fits"):
            #     pass
            # else:
            #     print("Source catalog is being downloaded as did not exist!")
                
            #     #in case, we have arrived at this point and stil some files are not made, we make them!
            #     source_pzs_i = Table.read( "/global/cfs/cdirs/cosmo/data/legacysurvey/dr9/%s/sweep/9.1-photo-z/"%wcat_k + sweep_k)
            #     source_cat_i = read_source_cat(wcat, brick_i)
                
            #     get_nearby_source_catalog(ra_k, dec_k, wcat_k, brick_k, save_path_k, source_cat_i, source_pzs_i)
    
            if os.path.exists(img_path_k):
                img_data = fits.open(img_path_k)
                data_arr = img_data[0].data
                wcs = WCS(fits.getheader( img_path_k ))
            else:
                #in case image is not downloaded, we download it!
                print("Image is being downloaded as did not exist!")
                save_cutouts(ra_k,dec_k,img_path_k,session,size=350)
                img_data = fits.open(img_path_k)
                data_arr = img_data[0].data
                wcs = WCS(fits.getheader( img_path_k ))
    
            if np.shape(data_arr[0])[0] != 350:
                raise ValueError("Issue with image size here=%s"%img_path_k)
        
                # import shutil
                # shutil.copy(img_path_k, save_path_k + "/")
    
            temp_dict = {"tgid":tgid_k, "ra":ra_k, "dec":dec_k, "redshift":redshift_k, "save_path":save_path_k, "img_path":img_path_k, "wcs": wcs , "image_data": data_arr, "source_cat": source_cat_f, "index":k , "org_mag_g": shreds_focus["MAG_G"][k], "overwrite": overwrite_bool, "run_own_detect":run_own_detect, "box_size" : box_size, "session":session, "use_photoz": use_pz_aper,
                        "bright_star_info": (bstar_ra, bstar_dec, bstar_radius, bstar_fdist), "sga_info": (sga_dist, sga_ndist) }
    
            return temp_dict
            
        print("Number of cores used is =",ncores)
    
        #to avoid memory issues, we will split up the dataset into nchunks 
        all_ks = np.arange(len(shreds_focus))
        print(len(all_ks))
        
        all_ks_chunks = np.array_split(all_ks, nchunks)
    
        all_aper_saveimgs = []
        #a boolean mask for objects that we want to save in a different pdf!
        special_plot_mask = []
        all_scarlet_saveimgs = []
            
        #if nchunks = 1, then it just returns the entire original list as 1 list
        #LOOPING OVER ALL THE CHUNKS!!
        for chunk_i in range(nchunks):
            
            print_stage("Started chunk %d/%d"%(chunk_i, nchunks) )
        
            all_ks_i =  all_ks_chunks[chunk_i]
            print_stage("Number of objects in this chunk = %d"%len(all_ks_i))
    
            #getting the table associated with this chunk!
            shreds_focus_i = shreds_focus[all_ks_i]
    
            if run_parr:
                with mp.Pool(processes = ncores ) as pool:
                    all_inputs = list(tqdm(pool.imap(produce_input_dicts, all_ks_i), total = len(all_ks_i)  ))
            else:
                all_inputs = []
                for i in trange(len(all_ks_i)):
                    all_inputs.append( produce_input_dicts(all_ks[i]) )
        
            all_inputs = np.array(all_inputs)
            print(len(all_inputs))
        
            ##This is the length of the list that contains info on all the objects on which aperture photometry will be run!
        
            ##################
            ##PART 4a: Run aperture photometry
            ##First step before scarlet photometry to weed out massive galaxies
            ##################
            
            if run_aper == True:
        
                if use_pz_aper:
                    print_stage("Photo-zs will be used in separating sources")
                else:
                    print_stage("Photo-zs will NOT be used in separating sources")
                
        
                if run_parr:
                    with mp.Pool(processes= ncores ) as pool:
                        results = list(tqdm(pool.imap(run_aperture_pipe, all_inputs), total = len(all_inputs)  ))
                else:
                    results = []
                    for i in trange(len(all_inputs)):
                        results.append( run_aperture_pipe(all_inputs[i]) )
        
                ### saving the results of the photometry pipeline
                results = np.array(results)
            
                print_stage("Done running aperture photometry!!")
            
                final_close_star_dists = results[:,0].astype(float)
                final_close_star_maxmags = results[:,1].astype(float)
                
                final_new_mags = np.vstack(results[:,2])
                final_org_mags = np.vstack(results[:,3])
                
                final_save_paths = results[:,4].astype(str)
                
        
                #these final image paths will be used to make a scrollable png file!
                #this is the summary aperture photo images
                final_image_paths = results[:,5].astype(str)
                
                final_image_paths = final_image_paths[final_image_paths != ""]
                all_aper_saveimgs += list(final_image_paths)
    
                bkg_estimated = results[:,6].astype(bool)
    
                #this is the fits data image
                all_data_paths = results[:,7].astype(str)
                
                
                ##what criterion to use here when showing the special objects?
                ##if within 1.5 of the stellar radius and the star is within 45 arcsecs of ths source
                special_plot_mask += list( (shreds_focus_i["STARFDIST"] < 1.5) & (shreds_focus_i["STARDIST_DEG"]*3600 < 45) )
                
        
                #check that the org mags make sense
                print("Maximum Abs difference between org mags = ",np.max( np.abs(final_org_mags[:,0] - shreds_focus_i["MAG_G"]) ))
                
                shreds_focus_i["NEAREST_STAR_DIST"] = final_close_star_dists
                shreds_focus_i["NEAREST_STAR_MAX_MAG"] = final_close_star_maxmags
                
                shreds_focus_i["MAG_G_APERTURE"] = final_new_mags[:,0]
                shreds_focus_i["MAG_R_APERTURE"] = final_new_mags[:,1]
                shreds_focus_i["MAG_Z_APERTURE"] = final_new_mags[:,2]
                
                shreds_focus_i["SAVE_PATH"] = final_save_paths 
                shreds_focus_i["IMAGE_PATH"] = all_data_paths
                
                shreds_focus_i["APER_BKG_ESTIMATED"] = bkg_estimated
            
                print("Compute aperture-photometry based stellar masses now!")
                    
                #compute the aperture photometry based stellar masses!
                rmag_aper = shreds_focus_i["MAG_R_APERTURE"].data
                gr_aper = shreds_focus_i["MAG_G_APERTURE"].data - shreds_focus_i["MAG_R_APERTURE"].data
        
                #are any of the aperture magnitudes nans?
                nan_mask = (np.isnan(shreds_focus_i["MAG_G_APERTURE"]) | np.isnan(shreds_focus_i["MAG_R_APERTURE"]) )
                #if any of the bands are nan, we automatically include it in scarlet method to try to do a better job at estimating photometry
                
                from desi_lowz_funcs import get_stellar_mass
        
                all_mstar_aper = np.ones(len(shreds_focus_i)) *np.nan
        
                mstar_aper_nonan = get_stellar_mass( gr_aper[~nan_mask],rmag_aper[~nan_mask], shreds_focus_i["Z"][~nan_mask]  )
        
                all_mstar_aper[~nan_mask] = mstar_aper_nonan
                
                shreds_focus_i["LOGM_SAGA_APERTURE"] = all_mstar_aper 
    
                #then save this file!
                if tgids_list is None:
                    if use_clean == False:
                        file_save = "/pscratch/sd/v/virajvm/catalog_dr1_dwarfs/iron_photometry/iron_%s_shreds_catalog_w_aper_mags_%s_chunk_%d.fits"%(sample_str, pz_flag, chunk_i)
                    else:
                        file_save = "/pscratch/sd/v/virajvm/catalog_dr1_dwarfs/iron_photometry/iron_%s_clean_catalog_w_aper_mags_%s_chunk_%d.fits"%(sample_str,pz_flag, chunk_i) 
    
                    save_table( shreds_focus_i, file_save)   
                    print_stage("Saved aperture summary files at %s!"%file_save)
                    
            ##################
            ##PART 4b: Run scarlet photometry for sources that remain as candidate dwarfs after aperture photometry
            ##################
        
            if run_scarlet == True:
    
                if run_aper:
                    #if run_aper was just run then we do not need to load in the datafiles
                    pass
                else:
                    #run aper was not run right before and so we have to load in the saved data files 
                    #note that if pipeline is being run on a single object, then we need to run_aper before to produce shreds_focus_i table as summary files are not saved in the tgids mode
                    if use_clean == False:
                        try:
                            file_save = "/pscratch/sd/v/virajvm/catalog_dr1_dwarfs/iron_photometry/iron_%s_shreds_catalog_w_aper_mags_%s_chunk_%d.fits"%(sample_str, pz_flag, chunk_i)
                        except:
                            file_save = "/pscratch/sd/v/virajvm/catalog_dr1_dwarfs/iron_photometry/iron_%s_shreds_catalog_w_aper_mags_%s.fits"%(sample_str, pz_flag)
                            
                    else:
                        try:
                            file_save = "/pscratch/sd/v/virajvm/catalog_dr1_dwarfs/iron_photometry/iron_%s_clean_catalog_w_aper_mags_%s_chunk_%d.fits"%(sample_str, pz_flag, chunk_i) 
    
                        except:
                            file_save = "/pscratch/sd/v/virajvm/catalog_dr1_dwarfs/iron_photometry/iron_%s_clean_catalog_w_aper_mags_%s.fits"%(sample_str, pz_flag) 
                            
                    shreds_focus_i = Table.read(file_save)
    
                ##we run scarlet only on objects were the updated aperture stellar mass is below Log Mstar 9.5 and if aperture photometry returned a nan value
                ##in case of the nan values, it is usually the case that a very bright star is nearby, and the original DR9 photometry is robust!
                ##need to apply a cut on that too
    
                print("Number of objects initially (in south only) -> %d"%len(shreds_focus_i[shreds_focus_i["is_south"] == 1]))
    
                ##furthermore, right now scarlet only works on south data  ....      
                
                temp_if = shreds_focus_i[ (shreds_focus_i["LOGM_SAGA_APERTURE"] <= 9.5) & (~np.isnan(shreds_focus_i["LOGM_SAGA_APERTURE"])) & (shreds_focus_i["is_south"] == 1)   ]
                
                temp_i_nans = shreds_focus_i[ (np.isnan(shreds_focus_i["LOGM_SAGA_APERTURE"])) & (shreds_focus_i["is_south"] == 1)  ]
    
                print("Number of objects with updated Mstar <= 9.5 -> %d"%len(temp_if))
                print("Number of objects with NaN photometry -> %d"%len(temp_i_nans))
    
                #the mask is that take objects from south, and if they have NaN stellar masses or stellar masses that are below 9.5 and not nans
                do_scarlet_mask = (shreds_focus_i["is_south"] == 1) & ( ( (shreds_focus_i["LOGM_SAGA_APERTURE"] <= 9.5) & (~np.isnan(shreds_focus_i["LOGM_SAGA_APERTURE"])) ) |  (np.isnan(shreds_focus_i["LOGM_SAGA_APERTURE"] ) ) ) & (shreds_focus_i["Z"] < 0.02)
                                                                                                          
                #these are the objects on which we will be doing scarlet photometry!                                                                                                          
                all_inputs_scarlet = all_inputs[  do_scarlet_mask ]
                shreds_focus_scarlet =shreds_focus_i[  do_scarlet_mask ]
    
                #for plotting purposes, we need to input in the dictionary 3 additional keys on the aperture mags
                all_inputs_scarlet_v2 = []
    
                for di, dicti in tqdm(enumerate(all_inputs_scarlet)):
                    dicti["MAG_G_APERTURE"] = shreds_focus_scarlet[di]["MAG_G_APERTURE"]
                    dicti["MAG_R_APERTURE"] = shreds_focus_scarlet[di]["MAG_R_APERTURE"]
                    dicti["MAG_Z_APERTURE"] = shreds_focus_scarlet[di]["MAG_Z_APERTURE"]
                    all_inputs_scarlet_v2.append(dicti)
    
                ##generate the relevant files for scarlet photometry
                if make_cats== True:
                    print_stage("Generating relevant files for doing scarlet photometry")
                    
                    print("Using cleaned catalogs =",use_clean==True)
            
                    print("Number of objects whose scarlet files will be obtained = ", len(shreds_focus_scarlet))
            
                    if use_clean == False:
                        top_folder = "/pscratch/sd/v/virajvm/redo_photometry_plots/all_deshreds"
                    else:
                        top_folder = "/pscratch/sd/v/virajvm/redo_photometry_plots/all_good"
                        
                    print(len(shreds_focus_scarlet[shreds_focus_scarlet["is_south"] == 1]))
                    print(len(shreds_focus_scarlet[shreds_focus_scarlet["is_south"] == 0]))
    
                    #list of dictionaries we will feed to get_relevant_files_scarlet function
                    
                    ##this can be sped up as we are not reading files or anyting!
    
                    wcats_array = np.array(["north","south"])[shreds_focus_scarlet["is_south"].data]
    
                    if len(wcats_array) != len(shreds_focus_scarlet):
                        raise ValueError("Error in lenghts of wcat and shreds_focus_scarlet array!")
                    
                    all_input_dicts = []
                    for i in range(len(shreds_focus_scarlet)):
                        
                        temp = { "TARGETID": shreds_focus_scarlet["TARGETID"][i], "SAMPLE": shreds_focus_scarlet["SAMPLE"][i], "RA": shreds_focus_scarlet["RA"][i], 
                                "DEC": shreds_focus_scarlet["DEC"][i],"wcat": wcats_array[i] , "brick_i":shreds_focus_scarlet["BRICKNAME"][i], "sweep_folder": shreds_focus_scarlet["SWEEP"][i].replace("-pz.fits","") , "top_folder":top_folder, "session":session }
    
                        all_input_dicts.append(temp)
    
                    
                    ## get the relevant files like psfs and subimages for doing scarlet photometry! 
                    with mp.Pool(processes=ncores) as pool:
                        results = list(tqdm(pool.imap(get_relevant_files_scarlet, all_input_dicts), total = len(all_input_dicts)  ))
                        
                print_stage("Beginning to run scarlet photometry!!")
    
                if run_parr:
                    with mp.Pool(processes= ncores ) as pool:
                        results = list(tqdm(pool.imap(run_scarlet_pipe, all_inputs_scarlet_v2), total = len(all_inputs_scarlet_v2)  ))
                else:
                    results = []
                    for i in trange(len(all_inputs_scarlet)):
                        results.append( run_scarlet_pipe(all_inputs_scarlet[i]) )
                        
                ### saving the results of the photometry pipeline
                results = np.array(results)
                
                print_stage("Done running all the scarlet photometry!!")
        
                final_new_mags = np.vstack(results[:,0])
                final_org_mags = np.vstack(results[:,1])
                
                final_saveimgs = np.vstack(results[:,2])
    
                final_saveimgs = final_saveimgs[final_saveimgs != ""]
                all_scarlet_saveimgs += list(final_saveimgs)
                
                #check that the org mags make sense
                print("Maximum Abs difference between org mags = ",np.max( np.abs(final_org_mags[:,0] - shreds_focus_scarlet["MAG_G"]) ))
                
                shreds_focus_scarlet["MAG_G_SCARLET"] = final_new_mags[:,0]
                shreds_focus_scarlet["MAG_R_SCARLET"] = final_new_mags[:,1]
                shreds_focus_scarlet["MAG_Z_SCARLET"] = final_new_mags[:,2]
    
                ## compute updated stellar mass using scarlet photometry??
                
                 #then save this file!
                if tgids_list is None:
                    if use_clean == False:
                        file_save = "/pscratch/sd/v/virajvm/catalog_dr1_dwarfs/iron_photometry/iron_%s_shreds_catalog_w_scarlet_mags_chunk_%d.fits"%(sample_str,chunk_i)
                    else:
                        file_save = "/pscratch/sd/v/virajvm/catalog_dr1_dwarfs/iron_photometry/iron_%s_clean_catalog_w_scarlet_mags_chunk_%d.fits"%(sample_str, chunk_i) 
    
                    
                    save_table( shreds_focus_i, file_save)   
                    print_stage("Saved scarlet summary files at %s!"%file_save)
        
    
    ##################
    ##PART 5: Once all the chunks are done, combine them all!
    ##################

    yes_save = not no_save
    if tgids_list is None and (run_aper | run_scarlet) and yes_save:
        print_stage("Consolidating all the saved chunks!")
        
        #files were saved and so we will consolidate them!
        if use_clean == False:
            clean_flag = "shreds"
        else:
            clean_flag = "clean"


        file_template_aper = "/pscratch/sd/v/virajvm/catalog_dr1_dwarfs/iron_photometry/iron_%s_%s_catalog_w_aper_mags_%s"%(sample_str, clean_flag, pz_flag)
        
        if run_scarlet:
            file_template_scarlet = "/pscratch/sd/v/virajvm/catalog_dr1_dwarfs/iron_photometry/iron_%s_%s_catalog_w_scarlet_mags"%(sample_str, clean_flag)
        
        if nchunks == 1:
            #we just need to rename the file!
            os.rename(file_template_aper + "_chunk_0.fits", file_template_aper + "%s.fits"%end_name)
            if run_scarlet:
                os.rename(file_template_scarlet + "_chunk_0.fits", file_template_scarlet + "%s.fits"%end_name)
            
        else:
            #we need to consolidate files!  
            shreds_focus_combine_aper = []
            
            for ni in trange(nchunks):
                shreds_focus_part = Table.read(file_template_aper + "_chunk_%d.fits"%ni  )
                shreds_focus_combine_aper.append(shreds_focus_part)
                #and then we delete that file!
                os.remove(file_template_aper + "_chunk_%d.fits"%ni)
                
            shreds_focus_combine_aper = vstack(shreds_focus_combine_aper)

            print_stage("Total number of objects in consolidated aperture file = %d"%len(shreds_focus_combine_aper))

            save_table( shreds_focus_combine_aper, file_template_aper + "%s.fits"%end_name ) 
            
            print_stage("Consolidated aperture chunk saved at %s"%(file_template_aper + "%s.fits"%end_name) )


            if run_scarlet:
                #we need to consolidate files!  
                shreds_focus_combine_scarlet = []
                for ni in trange(nchunks):
                    shreds_focus_part = Table.read(file_template_scarlet + "_chunk_%d.fits"%ni  )
                    shreds_focus_combine_scarlet.append(shreds_focus_part)
                    #and then we delete that file!
                    os.remove(file_template_scarlet + "_chunk_%d.fits"%ni)
                    
                shreds_focus_combine_scarlet = vstack(shreds_focus_combine_scarlet)
                print_stage("Total number of objects in consolidated scarlet file = %d"%len(shreds_focus_combine_scarlet))
                
                save_table( shreds_focus_combine_scarlet, file_template_scarlet + "%s.fits"%end_name) 
                
                print_stage("Consolidated scarlet chunk saved at %s"%(file_template_scarlet + "%s.fits"%end_name) )

    ##make a scrollable pdf to view the final results!
    ##only make for some objects?
    from PIL import Image
    from pathlib import Path
    
    if run_aper:
        # Path to images
        if summary_scroll_file_name == "":
            summary_scroll_file_name = "images_" + generate_random_string(5) + ".pdf"
        
        output_pdf = save_sample_path + "%s"%summary_scroll_file_name

        #maximum number of objects to store in a file
        max_num = 500
        all_aper_saveimgs = np.array(all_aper_saveimgs)
        
        # Load images
        image_files = all_aper_saveimgs[:max_num]
        
        images = [Image.open(img).convert("RGB") for img in image_files]
        
        # Save as PDF without extra white space
        images[0].save(output_pdf, save_all=True, append_images=images[1:])
        print(f"Aperture photo summary images saved at {output_pdf}")

        #now specifically plot the special objects we selected
        special_plot_mask = np.array(special_plot_mask)
        
        # Load images
        image_files = all_aper_saveimgs[special_plot_mask][:max_num]
        images = [Image.open(img).convert("RGB") for img in image_files]
        # Save as PDF without extra white space
        from datetime import datetime
        # Get the current date in your preferred format
        date_str = datetime.now().strftime("%Y-%m-%d")
        output_special_pdf = save_sample_path + "images_special_%s_%s.pdf"%(date_str, generate_random_string(3)) 

        images[0].save( output_special_pdf , save_all=True, append_images=images[1:])
        
        print(f"Aperture photo summary images saved at {output_special_pdf}")

        

    
    
    ##################
    ##PART 5: Using the aperture photometry footprint, find the desi sources that lie around on it and similar redshift as well??
    ##################





    ##### CODE FOR TESTING APERTURE PHOTOMETRY ON BAD RCHISQ OBJECTS
    # rchisq_bins = np.arange(0,9,1)
    # shreds_focus = []
    # np.random.seed(42)
    # for i,ri in enumerate(rchisq_bins[:-1]):
    #     # print(ri, rchisq_bins[i+1])
    #     temp_i = shreds_all[ (shreds_all["RCHISQ_R"] > ri) & (shreds_all["RCHISQ_R"] < rchisq_bins[i+1]) & (shreds_all["FRACFLUX_G"] < 0.01) & (shreds_all["FRACFLUX_R"] < 0.01) & (shreds_all["FRACFLUX_Z"] < 0.01) & (shreds_all["MASKBITS"] == 0) & (shreds_all["SAMPLE"] == sample_str)  ]
    #     #pick random 100 objects!
    #     print(len(temp_i))
    #     temp_ij = temp_i[ np.random.randint( len(temp_i) ,size = np.minimum( 500 , len(temp_i))   ) ]
    #     shreds_focus.append(temp_ij)
    # shreds_focus = vstack(shreds_focus)

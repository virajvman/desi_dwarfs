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
from astropy.table import Column
from aperture_photo import run_aperture_pipe
from aperture_cogs import run_cog_pipe
# from get_sga_distances import get_sga_info
from desi_lowz_funcs import save_subimage, fetch_psf, generate_random_string, add_paths_to_catalog, save_cutouts, get_stellar_mass
from desiutil import brick
import fitsio
from easyquery import Query, QueryMaker
import shutil
from process_tractors import get_nearby_source_catalog, are_more_bricks_needed, return_sources_wneigh_bricks, read_source_pzs, read_source_cat
from functools import partial
from concurrent.futures import ThreadPoolExecutor, as_completed
from PIL import Image
from pathlib import Path
import random, string
from datetime import datetime
from shred_photometry_maskbits import create_shred_maskbits
import glob
from shred_classifier import get_pcnn_data_inputs
from aperture_cogs import make_empty_tractor_cog_dict


def stack_results(results, key):
    '''
    Stack outputs from different sources into a common array to be fed to astropy table
    '''
    arrs = [np.atleast_1d(r[key]) for r in results]
    out = np.vstack(arrs)
    if out.shape[1] == 1:  # scalar case
        out = out[:, 0]    # flatten to (N,)
    return out

    
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
    result.add_argument('-use_sample',dest='use_sample', type = str, default = "")
    result.add_argument('-overwrite',dest='overwrite',action='store_true')
    result.add_argument('-make_cats',dest='make_cats', action='store_true')
    result.add_argument('-run_aper',dest='run_aper', action='store_true')
    result.add_argument('-run_cog',dest='run_cog', action='store_true')
    result.add_argument('-nchunks',dest='nchunks', type=int,default = 1)
    result.add_argument('-no_save',dest='no_save', action = "store_true")
    result.add_argument('-make_main_cats',dest='make_main_cats', action = "store_true")
    result.add_argument('-end_name',dest='end_name', type = str, default = "")
    result.add_argument('-no_cnn_cut',dest='no_cnn_cut', action='store_true')
    result.add_argument('-run_simple_photo',dest='run_simple_photo', action='store_true')
    result.add_argument('-get_cnn_inputs',dest='get_cnn_inputs', action='store_true')
    
    return result




def parse_tgids(value):
    if not value:
        return None
    return [int(x) for x in value.split(',')]


def generate_random_string(length=5):
    '''
    Function that generates random string to add on to summary figure pdf file
    '''
    return ''.join(random.choices(string.ascii_lowercase + string.digits, k=length))


def stack_images_vertically(img_path1, img_path2):
    '''
    Function that stacks two summary images into one so can be all in a single scrollable pdf
    '''
    img1 = Image.open(img_path1).convert("RGB")
    img2 = Image.open(img_path2).convert("RGB")

    # Resize to match widths if necessary
    if img1.width != img2.width:
        new_width = min(img1.width, img2.width)
        img1 = img1.resize((new_width, int(img1.height * new_width / img1.width)))
        img2 = img2.resize((new_width, int(img2.height * new_width / img2.width)))

    total_height = img1.height + img2.height
    new_img = Image.new("RGB", (img1.width, total_height))
    new_img.paste(img1, (0, 0))
    new_img.paste(img2, (0, img1.height))
    return new_img


def clean_image_lists(imgs1, imgs2=None, mask=None):
    '''
    Function that cleans lists of image paths
    '''
    imgs1 = np.array(imgs1)

    if imgs2 is not None:
        imgs2 = np.array(imgs2)

        valid_mask = (imgs1 != None) & (imgs2 != None)

        imgs1 = imgs1[valid_mask]
        imgs2 = imgs2[valid_mask]
        
        if mask is not None:
            mask = np.array(mask)[valid_mask]
    else:
        valid_mask = (imgs1 != None)
        imgs1 = imgs1[valid_mask]
        if mask is not None:
            mask = np.array(mask)[valid_mask]
        imgs2 = None

    return imgs1, imgs2, mask



def make_scroll_pdf(all_aper_saveimgs1, save_sample_path, summary_scroll_file_name,
                    special_plot_mask, max_num=500, type_str="aper", all_aper_saveimgs2=None):
    '''
    Function that make scrollable pdf given two lists containing the paths to summary images
    '''
    
    # Clean input arrays
    all_aper_saveimgs1, all_aper_saveimgs2, special_plot_mask = clean_image_lists(
        all_aper_saveimgs1, all_aper_saveimgs2, special_plot_mask
    )

    # Set default name if needed
    if summary_scroll_file_name == "":
        summary_scroll_file_name = f"images_{type_str}_" + generate_random_string(5) + ".pdf"
    output_pdf = save_sample_path + f"{summary_scroll_file_name}"

    # Prepare image pairs or singles
    if all_aper_saveimgs2 is not None:

        if len(all_aper_saveimgs1) != len(all_aper_saveimgs2):
            raise ValueError("Input image files lists to make_scroll_pdf function are not of equal length")
        
        image_pairs = list(zip(all_aper_saveimgs1[:max_num], all_aper_saveimgs2[:max_num]))
        images = [stack_images_vertically(img1, img2) for img1, img2 in image_pairs]
    else:
        images = [Image.open(img).convert("RGB") for img in all_aper_saveimgs1[:max_num]]

    # Save full summary
    if images:
        images[0].save(output_pdf, save_all=True, append_images=images[1:])
        print(f"Photo summary images saved at {output_pdf}")
    else:
        print("No images to save.")

    # Special subset
    if special_plot_mask is not None and np.any(special_plot_mask):
        special_imgs1 = all_aper_saveimgs1[special_plot_mask][:max_num]

        if all_aper_saveimgs2 is not None:
            special_imgs2 = all_aper_saveimgs2[special_plot_mask][:max_num]
            special_pairs = list(zip(special_imgs1, special_imgs2))
            special_images = [stack_images_vertically(img1, img2) for img1, img2 in special_pairs]
        else:
            special_images = [Image.open(img).convert("RGB") for img in special_imgs1]

        if special_images:
            date_str = datetime.now().strftime("%Y-%m-%d")
            output_special_pdf = save_sample_path + f"images_special_{type_str}_{date_str}_{generate_random_string(3)}.pdf"
            special_images[0].save(output_special_pdf, save_all=True, append_images=special_images[1:])
            print(f"Special photo summary images saved at {output_special_pdf}")
        else:
            print("No special images to save.")

    return

    


def get_relevant_files_aper(input_dict):
    '''
    Function that gets the relevant, useful files for running the aperture photometry pipeline.
    '''

    #TARGETID of DESI source
    tgid_k = input_dict["TARGETID"]
    #SAMPLE object is part of: BGS_BRIGHT, LOWZ etc.
    samp_k = input_dict["SAMPLE"]
    #RA, DEC, redshift of source
    ra_k = input_dict["RA"]
    dec_k = input_dict["DEC"]
    redshift_k = input_dict["Z"]

    #Tractor object and brickid of the source
    objid_k = input_dict["OBJID"]
    brickid_k = input_dict["BRICKID"]

    #folder info in which object info will be stored
    top_folder = input_dict["top_folder"]
    sweep_folder = input_dict["sweep_folder"]
    brick_i = input_dict["brick_i"]
    #is it in south or north catalog?
    wcat = input_dict["wcat"]
    #table for source photo-zs, but no longer needed
    source_pzs_i = input_dict["source_pzs_i"]
    #table for source catalog
    source_cat = input_dict["source_cat"]
    #integer for the size of the image cutput
    box_size = input_dict["image_size"]
    
    top_path_k = f"{top_folder}/{wcat}/{sweep_folder}/{brick_i}/{samp_k}_tgid_{tgid_k}"
    
    check_path_existence(all_paths=[top_path_k])
    #inside this folder, we will save all the relevant files and info!!
    image_path =  top_folder + f"_cutouts/image_tgid_{tgid_k}_ra_{ra_k:.3f}_dec_{dec_k:.3f}.fits"


    ##for now, we will only do this for objects that do not already have a source catalog file!!
    if os.path.exists(top_path_k + "/source_cat_f.fits"):
        pass
    else:
        print(f"Source catalog did not exist! Making one: {top_path_k}")
        get_nearby_source_catalog(ra_k, dec_k, objid_k, brickid_k, box_size, wcat, brick_i, top_path_k, source_cat, source_pzs_i)
        
        ## check if the source is at the edge of the brick, if so we will need to combine stuff
        more_bricks, more_wcats, more_sweeps = are_more_bricks_needed(ra_k,dec_k,radius_arcsec = int(box_size*0.262/2)  )
    
        if len(more_bricks) == 0:
            #there are no neighboring bricks needed
            pass
        else:
            return_sources_wneigh_bricks(top_path_k, ra_k, dec_k, objid_k, brickid_k, box_size, more_bricks, more_wcats, more_sweeps,use_pz = False)
            
    if os.path.exists(image_path):
        pass
    else:
        print(f"IMAGE PATH DOES NOT EXIST: {image_path}")

        # file_search = glob.glob(image_path)
        # print(file_search)
        # print("---")
        # #if this image path does not exist, does it exist in the other folder?
        # if "all_deshreds" in image_path:
        #     image_path_other = image_path.replace("all_deshreds","all_good")
        # if "all_good" in image_path:
        #     image_path_other = image_path.replace("all_good","all_deshreds")
        # if "all_sga" in image_path:
        #     image_path_other = image_path.replace("all_sga","all_good")
            
        # if os.path.exists(image_path_other):
        #     shutil.copy(image_path_other, image_path)
        # else:
            #we need to obtain the cutout data!

        #as we are multi-threading the processing for each galaxy, and sessions are not thread safe
        #we will be creating a session per image and then closing it right away!
        with requests.Session() as session:
            save_cutouts(ra_k, dec_k, image_path, session, size=box_size)
        
    ##done saving all the relevant, useful files!
    return


def select_unique_objs(catalog):
    '''
    Function that selects only the unique TARGETIDs from the catalog
    '''
    _,uni_idx = np.unique(catalog["TARGETID"],return_index=True)
    catalog_uni = catalog[uni_idx]
    print(f"Original Count = {len(catalog)}, After Unique Count = {len(catalog_uni)}")
    return catalog_uni
    
    

def make_clean_shreds_catalogs():
    '''
    Function that combines the initial DESI catalogs from each sample into clean and likely shredded dwarf galaxy catalogs.
    Useful information is added to the catalog in the process.
    '''

    ##THESE ARE ALL THE SOURCES THAT ARE NOT IN SGA! 
    bgsb_list = Table.read("/pscratch/sd/v/virajvm/catalog_dr1_dwarfs/iron_bgs_bright_filter_zsucc_zrr02_allfracflux.fits")
    bgsf_list = Table.read("/pscratch/sd/v/virajvm/catalog_dr1_dwarfs/iron_bgs_faint_filter_zsucc_zrr03_allfracflux.fits")
    elg_list = Table.read("/pscratch/sd/v/virajvm/catalog_dr1_dwarfs/iron_elg_filter_zsucc_zrr05_allfracflux.fits")
    lowz_list = Table.read("/pscratch/sd/v/virajvm/catalog_dr1_dwarfs/iron_lowz_filter_zsucc_zrr03.fits")

    #these are sources to be added to the final clean catalog from SGA
    sga_bad_trac_clean = Table.read("/pscratch/sd/v/virajvm/catalog_dr1_dwarfs/iron_desi_sga_bad_trac_good_dwarfs_CLEAN.fits")

    ##select only unique objects
    bgsb_list = select_unique_objs(bgsb_list)
    bgsf_list = select_unique_objs(bgsf_list)
    elg_list = select_unique_objs(elg_list)
    lowz_list = select_unique_objs(lowz_list)
    sga_bad_trac_clean = select_unique_objs(sga_bad_trac_clean)

    
    from desi_lowz_funcs import get_sweep_filename, save_table, is_target_in_south, get_sga_norm_dists, get_sga_norm_dists_FAST
    from construct_dwarf_galaxy_catalogs import bright_star_filter

    #invert the above to get the clean mask!
    clean_mask_bgsb = (bgsb_list["PHOTO_REPROCESS"] == 0)
    clean_mask_bgsf = (bgsf_list["PHOTO_REPROCESS"] == 0)
    clean_mask_elg = (elg_list["PHOTO_REPROCESS"] == 0)
    clean_mask_lowz = (lowz_list["PHOTO_REPROCESS"] == 0)
    
    ##CLEAN CATALOG
    dwarf_mask_bgsb = (bgsb_list["LOGM_SAGA_FIDU"] < 9.25) 
    dwarf_mask_bgsf = (bgsf_list["LOGM_SAGA_FIDU"] < 9.25) 
    dwarf_mask_lowz = (lowz_list["LOGM_SAGA_FIDU"] < 9.25) 
    dwarf_mask_elg = (elg_list["LOGM_SAGA_FIDU"] < 9.25)   

    bgsb_clean_dwarfs = bgsb_list[ clean_mask_bgsb  & (dwarf_mask_bgsb) ] 
    bgsf_clean_dwarfs = bgsf_list[ clean_mask_bgsf  & (dwarf_mask_bgsf) ]
    lowz_clean_dwarfs = lowz_list[ clean_mask_lowz  & dwarf_mask_lowz ]
    elg_clean_dwarfs = elg_list[ clean_mask_elg & dwarf_mask_elg  ]

    #adding the sample column for reference
    bgsb_clean_dwarfs["SAMPLE"] = np.array(len(bgsb_clean_dwarfs)*["BGS_BRIGHT"])
    bgsf_clean_dwarfs["SAMPLE"] = np.array(len(bgsf_clean_dwarfs)*["BGS_FAINT"])
    elg_clean_dwarfs["SAMPLE"] = np.array(len(elg_clean_dwarfs)*["ELG"])
    lowz_clean_dwarfs["SAMPLE"] = np.array(len(lowz_clean_dwarfs)*["LOWZ"])
    
    #the sga_bad_trac_clean catalog already has SAMPLE column!
    print(f"Number of clean sources being added from SGA matches as they have no SGA photometry = {len(sga_bad_trac_clean)}")

    all_clean_dwarfs = vstack( [ bgsb_clean_dwarfs, bgsf_clean_dwarfs, lowz_clean_dwarfs, elg_clean_dwarfs, sga_bad_trac_clean] )

    #removing columns that were present in the ELG column from whole catalog
    all_clean_dwarfs.remove_column("OII_FLUX")
    all_clean_dwarfs.remove_column("OII_FLUX_IVAR")
    all_clean_dwarfs.remove_column("Z_HPX")
    all_clean_dwarfs.remove_column("TARGETID_2")
    all_clean_dwarfs.remove_column("OII_SNR")
    all_clean_dwarfs.remove_column("ZSUCC")

    print("Total number of galaxies in clean catalog=",len(all_clean_dwarfs))

    ## We add information about nearest SGA objects here
    all_clean_dwarfs = get_sga_norm_dists_FAST(all_clean_dwarfs, siena_path="/global/cfs/cdirs/cosmo/data/sga/2020/SGA-2020.fits")

    #keys to be added to catalog regarding bright star information
    bstar_keys = [ "STARFDIST", "STARDIST_DEG","STARMAG", "STAR_RADIUS_ARCSEC", "STAR_RA","STAR_DEC"]

    # Check if all bright star keys exist
    if all(key in all_clean_dwarfs.colnames for key in bstar_keys):
        print("Bright star information already exists!")
    else:
        # Recompute if missing
        print("Bright star information did not exist and will be computed.")
        all_clean_dwarfs = bright_star_filter(all_clean_dwarfs)

    #adding dummy PCNN for now!
    all_clean_dwarfs["PCNN_FRAGMENT"] = -99*np.ones(len(all_clean_dwarfs))
    
    
    # save the clean dwarfs now!
    save_table(all_clean_dwarfs,"/pscratch/sd/v/virajvm/catalog_dr1_dwarfs/desi_y1_dwarf_clean_catalog_v4.fits",comment="This is a compilation of dwarf galaxy candidates in DESI Y1 data from the BGS Bright, BGS Faint, ELG and LOW-Z samples. Only galaxies with LogMstar < 9.25 (w/SAGA based stellar masses) and that have robust photometry are included.")
        
    ##LIKELY SHREDDED CATALOG FROM THE NON SGA sources. The shredded sources in SGA have been compiled separately in construct_dwarf .. py script

    bgsb_shreds = bgsb_list[ ~clean_mask_bgsb & dwarf_mask_bgsb]
    bgsf_shreds = bgsf_list[  ~clean_mask_bgsf & dwarf_mask_bgsf]
    elg_shreds = elg_list[ ~clean_mask_elg & dwarf_mask_elg]
    lowz_shreds = lowz_list[ ~clean_mask_lowz & dwarf_mask_lowz]

    #adding the sample column
    bgsb_shreds["SAMPLE"] = np.array(["BGS_BRIGHT"]*len(bgsb_shreds))
    bgsf_shreds["SAMPLE"] = np.array(["BGS_FAINT"]*len(bgsf_shreds))
    elg_shreds["SAMPLE"] = np.array(["ELG"]*len(elg_shreds))
    lowz_shreds["SAMPLE"] = np.array(["LOWZ"]*len(lowz_shreds))

    #stacking all the shreds now 
    shreds_all = vstack( [bgsb_shreds, bgsf_shreds, elg_shreds, lowz_shreds ])
    
    shreds_all.remove_column("OII_FLUX")
    shreds_all.remove_column("OII_FLUX_IVAR")
    shreds_all.remove_column("Z_HPX")
    shreds_all.remove_column("TARGETID_2")
    shreds_all.remove_column("OII_SNR")
    shreds_all.remove_column("ZSUCC")
    
    print("Total number of objects identified as likely shreds = ", len(shreds_all))

    #We add information about nearest SGA objects here
    shreds_all = get_sga_norm_dists_FAST(shreds_all, siena_path="/global/cfs/cdirs/cosmo/data/sga/2020/SGA-2020.fits")

    # Check if the bright star key already exist
    if all(key in shreds_all.colnames for key in bstar_keys):
        print("Bright star information already exists!")
    else:
        # Recompute if missing
        print("Bright star information did not exist and will be computed.")
        shreds_all = bright_star_filter(shreds_all)


    shreds_all["PCNN_FRAGMENT"] = -99*np.ones(len(shreds_all))
    
    save_table(shreds_all,"/pscratch/sd/v/virajvm/catalog_dr1_dwarfs/desi_y1_dwarf_shreds_catalog_v4.fits")

    print_stage("MAKE SURE TO ADD THE PCNN VALUES TO DO THIS CATALOG! ")
        
    return 



def create_brick_jobs(brick_i, shreds_focus_w, wcat, top_folder):
    '''
    In this function, we parallelize the reading of the source cat and creation of the dictionaries per galaxy containing variables for running photometry pipeline!!
    We will be multi-threading here so it will share the memory of 'shreds_focus_w' catalog!!

    The brick_i is the only variable that is changing, the others can be kept constant over north/south                 
    '''

    source_cat = read_source_cat(wcat, brick_i)

    #select galaxies corresponding to this brick!!
    shreds_focus_brick_i = shreds_focus_w[shreds_focus_w["BRICKNAME"] == brick_i]

    # Prepare input dictionaries
    brick_dicts = []

    #usually there are not that many galaxies in a brick so this is fast for-loop
    for i in range(len(shreds_focus_brick_i)):

        sweep_i = shreds_focus_brick_i["SWEEP"][i]
        sweep_folder = sweep_i.replace("-pz.fits", "")

        galaxy_dict = {
            "TARGETID": shreds_focus_brick_i["TARGETID"][i],
            "SAMPLE": shreds_focus_brick_i["SAMPLE"][i],
            "RA": shreds_focus_brick_i["RA"][i],
            "DEC": shreds_focus_brick_i["DEC"][i],
            "Z": shreds_focus_brick_i["Z"][i],
            "OBJID": shreds_focus_brick_i["OBJID"][i],
            "BRICKID": shreds_focus_brick_i["BRICKID"][i],
            "wcat": wcat,
            "brick_i": brick_i,
            "sweep_folder": sweep_folder,
            "top_folder": top_folder,
            "source_cat": source_cat,
            "source_pzs_i": None,
            "image_size": shreds_focus_brick_i["IMAGE_SIZE_PIX"][i]
        }
        
        brick_dicts.append(galaxy_dict)

    #brick_dicts is a list of dictionaries where one dictionary corresponds to one galaxy!
    return brick_dicts
    

def process_bricks_parallel(brick_dict):
    '''
    This function takes in a single list of dictionaries (brick_dict) corresponding to all galaxies in that brick!!
    '''
    with ThreadPoolExecutor(max_workers=8) as executor:
        #if there are a lot of galaxies then only we use tqdm!
        if len(brick_dict) > 50:
             list(tqdm(executor.map(get_relevant_files_aper, brick_dict), total=len(brick_dict), desc=f"Galaxies in brick {brick_dict[0]['brick_i']}" ) )
        else:
            list(executor.map(get_relevant_files_aper, brick_dict))



def compute_aperture_masses(
    shreds_table, 
    rband_key="MAG_R_APERTURE_R375", 
    gband_key="MAG_G_APERTURE_R375", 
    z_key="Z_CMB", 
    dmpc_key = "DIST_MPC_FIDU",
    output_key="LOGM_SAGA_APERTURE_R375"
):
    """
    Compute new aperture-photometry-based stellar masses using g - r color, r-band magnitude, and redshift.
    """
    rmag_aper = shreds_table[rband_key].data
    gr_aper = shreds_table[gband_key].data - rmag_aper

    nan_mask = np.isnan(rmag_aper) | np.isnan(shreds_table[gband_key].data)

    all_mstar_aper = np.full(len(shreds_table), np.nan)

    valid_mask = ~nan_mask
    if valid_mask.sum() == 0:
        # Nothing to compute
        print(f"Warning: no valid entries for {output_key}, filling with NaN.")
    elif len(shreds_table) == 1:
        # Single object special case
        print("Note: computing for a single object, skipping stellar mass calculation.")
    else:
        mstar_aper_nonan = get_stellar_mass(
            gr_aper[valid_mask],
            rmag_aper[valid_mask],
            shreds_table[z_key][valid_mask],
            d_in_mpc = shreds_table[dmpc_key][valid_mask],
            input_zred = False
        )
        all_mstar_aper[valid_mask] = mstar_aper_nonan

    shreds_table[output_key] = all_mstar_aper
    return shreds_table

def filter_saveimgs_paths(results, flag):
    final_cog_saveimgs = []
    
    for r in results:
        if r[flag] is None:
            final_cog_saveimgs.append(None)
        else:
            final_cog_saveimgs.append(r[flag])
            
    return final_cog_saveimgs

    
if __name__ == '__main__':
    import warnings
    from astropy.units import UnitsWarning
    from astropy.wcs import FITSFixedWarning
    import numpy as np
    np.seterr(invalid='ignore')
    warnings.simplefilter('ignore', category=UnitsWarning)
    warnings.simplefilter('ignore', FITSFixedWarning)
    warnings.filterwarnings("ignore", message="Warning: converting a masked element to nan")

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

    use_tqdm = sys.stdout.isatty()

    # read in command line arguments
    args = argument_parser().parse_args()

    #sample_str could also be multiple samples together!
    sample_str = args.sample

    sample_list = False
    if "," in sample_str:
        sample_list = True
        all_samples  = sample_str.split(",")

    #reading all the input flags
    min_ind = args.min
    max_ind = args.max
    run_parr = args.parallel
    #use clean, shreds, or sga catalog.
    use_sample = args.use_sample

    if use_sample not in ["clean","sga","shred"]:
        raise ValueError('Incorrect entry for use_sample. Correct entries are ["clean","sga","shred"]')
    
    no_cnn_cut = args.no_cnn_cut
    ncores = args.ncores
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
    #should the aperture photometry pipeline be run?
    run_aper = args.run_aper
    #should the cog pipeline be run?
    run_cog = args.run_cog
    run_simple_photo = args.run_simple_photo
    get_cnn_inputs = args.get_cnn_inputs
    

    #the fiducial parameters for the image segmentation!
    npixels_min = 10
    threshold_rms_scale = 1.5
    
    c_light = 299792 #km/s

    ##################
    ##PART 1: Make, prep the catalogs!
    ##################

    if make_main_cats:
        make_clean_shreds_catalogs()

    #################
    #run the many_cutouts.py script to generate cutouts!!
    #################

    ##add the columns on image path and file_path to these catalogs!!
    shreds_file = "/pscratch/sd/v/virajvm/catalog_dr1_dwarfs/desi_y1_dwarf_shreds_catalog_v4.fits"
    clean_file = "/pscratch/sd/v/virajvm/catalog_dr1_dwarfs/desi_y1_dwarf_clean_catalog_v4.fits"
    clean_file_2 = "/pscratch/sd/v/virajvm/catalog_dr1_dwarfs/desi_y1_dwarf_clean_catalog_v4_RUN_W_APER.fits"
    sga_file = "/pscratch/sd/v/virajvm/catalog_dr1_dwarfs/iron_desi_SGA_matched_dwarfs_REPROCESS.fits"


    ##Adding the image path and file path (where all the outputs are saved) columns to the catalog if they do not exist
    shreds_cat = Table.read(shreds_file)
    clean_cat = Table.read(clean_file)
    clean_cat_2 = Table.read(clean_file_2)
    sga_cat = Table.read(sga_file)

    ##we need to be careful here in how we are defining the top folder!! 
    if "IMAGE_PATH" in shreds_cat.colnames and "FILE_PATH" in shreds_cat.colnames:
        print("image_path and file_path columns already exist in shreds catalog!")        
    else:
        print("Adding image_path and file_path to shreds catalog!")
        add_paths_to_catalog(org_file = shreds_file, out_file = shreds_file,top_folder="/pscratch/sd/v/virajvm/redo_photometry_plots/all_deshreds")

    ##
    
    if "IMAGE_PATH" in clean_cat.colnames and "FILE_PATH" in clean_cat.colnames:
        print("image_path and file_path columns already exist in clean catalog!")
    else:
        print("Adding image_path and file_path to clean catalog!")
        add_paths_to_catalog(org_file = clean_file, out_file = clean_file,top_folder="/pscratch/sd/v/virajvm/redo_photometry_plots/all_good")

    ###
    
    if "IMAGE_PATH" in clean_cat_2.colnames and "FILE_PATH" in clean_cat_2.colnames:
        print("image_path and file_path columns already exist in clean TOTAL catalog!")
    else:
        print("Adding image_path and file_path to clean TOTAL catalog!")
        add_paths_to_catalog(org_file = clean_file_2, out_file = clean_file_2,top_folder="/pscratch/sd/v/virajvm/redo_photometry_plots/all_good")

    ##
        
    if "IMAGE_PATH" in sga_cat.colnames and "FILE_PATH" in sga_cat.colnames:
        print("image_path and file_path columns already exist in sga catalog!")
    else:
        print("Adding image_path and file_path to sga catalog!")
        add_paths_to_catalog(org_file = sga_file, out_file = sga_file,top_folder="/pscratch/sd/v/virajvm/redo_photometry_plots/all_sga")


    ##in the shreds file make sure there is a PCNN column
    if "PCNN_FRAGMENT" in shreds_cat.colnames:
        print("PCNN Fragment column already exist!")
    else:
        print("PCNN column does not exist in the catalog. ADD ITT!!")
        # add_pcnn_to_shred_catalog(shreds_file,get_all_data=True)

    #delete these variables as no longer needed!
    del shreds_cat, clean_cat, sga_cat, clean_cat_2
     
    ##################
    ##PART 2: Generate nested folder structure with relevant files for doing photometry
    ##################

    if use_sample == "shred":
        top_folder = "/pscratch/sd/v/virajvm/redo_photometry_plots/all_deshreds"
    if use_sample == "sga":
        top_folder = "/pscratch/sd/v/virajvm/redo_photometry_plots/all_sga"
    if use_sample == "clean":
        top_folder = "/pscratch/sd/v/virajvm/redo_photometry_plots/all_good"


    ##load the relevant catalogs!
    if use_sample == "shred":
        file_to_read = shreds_file
    if use_sample == "sga":
        file_to_read = sga_file
    if use_sample == "clean":
        file_to_read = clean_file

    shreds_all = Table.read(file_to_read)

    #let us check that the number of sources is unique
    print(f"Total number of objects in catalog = {len(shreds_all)}")
    print(f"Total number of UNIQUE objects in catalog = {len(np.unique(shreds_all['TARGETID'].data)) }")
    
    ##let us do a quick tally on how many of the image paths are blank! that is, they need new image paths!
    print("In this catalog, " + str(len(shreds_all[shreds_all['IMAGE_PATH'] == ""])) + " objects do not have image paths!")
    
    ## filtering for the sample now 
    if sample_list:
        #we have provided a list of samples
        data_samples = shreds_all['SAMPLE'].data.astype(str)
        all_samples = np.array(all_samples).astype(str)
        print("Samples being read today:", all_samples)
        sample_mask = np.isin(data_samples, all_samples) 
    else:
        #only a single sample is given!
        sample_mask = (shreds_all["SAMPLE"] ==  sample_str) # & (shreds_all["Z"] < 0.007)

    shreds_focus = shreds_all[sample_mask]
    
    #we also apply a mask on the PCNN probabilities if relevant
    if no_cnn_cut:
        print("NO PCNN cut is being applied")
        pass
    else:
        #filter for sources that CNN thinks are shreds
        print("PCNN cut is being applied")
        shreds_focus = shreds_focus[ shreds_focus["PCNN_FRAGMENT"] >= 0.5]

    print("Number of objects in catalog so far: ", len(shreds_focus))

    ##finally, if a list of targetids is provided, then we only select those
    if tgids_list is not None:
        print("List of targetids to process:",tgids_list)
        shreds_focus = shreds_focus[np.isin(shreds_focus['TARGETID'], np.array(tgids_list) )]
        
        print("Number of targetids to process =", len(shreds_focus))

    #apply the max_ind cut if relevant
    max_ind = np.minimum( max_ind, len(shreds_focus) )
    shreds_focus = shreds_focus[min_ind:max_ind]

    if use_sample == "clean":
        print(f"Initial size = {len(shreds_focus)}")
        #need to filter for robust rchisq as we recently updated that and just want to make sure
        shreds_focus = shreds_focus[ (shreds_focus["RCHISQ_G"] < 4) & (shreds_focus["RCHISQ_R"] < 4) & (shreds_focus["RCHISQ_Z"] < 4) ]
        print(f"Final size = {len(shreds_focus)}")
        

    print("Number of objects in catalog so far: ", len(shreds_focus))

    if make_cats == True:    
        print_stage("Generating relevant files for doing aperture photometry")
        print(f"Using catalog being used: {use_sample}")
    
        print("Number of objects whose photometry will be redone = ", len(shreds_focus) )
    
        print(f"Number of objects in south: {len(shreds_focus[shreds_focus['is_south'] == 1])}" )
        print(f"Number of objects in north: {len(shreds_focus[shreds_focus['is_south'] == 0])}" )
        
        for wcat_ind, wcat in enumerate(["north","south"]):
            check_path_existence(all_paths=[top_folder + "/%s"%wcat])
            
            ##we can invert this if we want to make sure all cats are being made without long code times
            unique_sweeps = np.unique( shreds_focus[shreds_focus["is_south"] == wcat_ind]["SWEEP"])
            print(len(unique_sweeps), f"unique sweeps found in {wcat}")

            shreds_focus_w = shreds_focus[(shreds_focus["is_south"] == wcat_ind)]
    
            unique_bricks_w = np.unique(shreds_focus_w["BRICKNAME"])

            print(f"{len(unique_bricks_w)} unique bricks found in {wcat}")
            print(f"{len(shreds_focus_w)} number of galaxies found in {wcat}")

            ##this is the number of bricks we will process at one time!
            brick_chunk = 256
            
            print(f"Bricks will be processed in chunks of {brick_chunk}")

            #the number of brick jobs we have to process
            num_bricks = len(unique_bricks_w)

             # Fix everything except brick_i
            create_brick_jobs_fixed = partial(create_brick_jobs, shreds_focus_w=shreds_focus_w, wcat=wcat, top_folder=top_folder)

            for min_brick_id in trange(0, num_bricks, brick_chunk):
                #get the relevant brick names in this chunk
                unique_bricks_chunki = unique_bricks_w[ min_brick_id : min_brick_id + brick_chunk]

                # with ThreadPoolExecutor(max_workers=64) as executor:
                    # all_brick_jobs = list(tqdm( executor.map(create_brick_jobs_fixed, unique_bricks_chunki),
                    #                            total=len(unique_bricks_chunki),
                    #                             desc="Creating brick jobs!"   ) )
                
                #we do not care about the order in the brick jobs!!
                #doing the prep work before running the actual job!!
                all_brick_jobs = []
                with ThreadPoolExecutor(max_workers=64) as executor:
                    futures = [executor.submit(create_brick_jobs_fixed, brick_i) for brick_i in unique_bricks_chunki]
                
                    for future in tqdm(as_completed(futures), total=len(futures), desc="Creating brick jobs!"):
                        all_brick_jobs.append(future.result())

                                          
                ##all_brick_jobs is the list we will be sending to different cores. One item is this list corresponds to list of all galaxies in a single brick!
            
                ##each brick will be processed on a core, and within each brick, the galaxies will be multi-threading!
                with mp.Pool(processes=ncores) as pool:
                    results = list(tqdm(pool.imap(process_bricks_parallel, all_brick_jobs), total=len(all_brick_jobs), desc="Bricks"))


    ##################
    ##PART 3: Preparing inputs for the aperture and cog photometry functions
    ##################
    
    if run_aper | run_cog:
        
        print_stage("Constructing input files for the photometry functions!")
    
        all_wcats = ["north","south"]
    
        ##prepare the catalogs on which photo pipeline will be run
        if use_sample == "shred":
            #we deal with the shredded catalogs
            
            #the path where we will also collect all the final summary figures
            top_folder = "/pscratch/sd/v/virajvm/redo_photometry_plots/all_deshreds"
            save_sample_path = "/pscratch/sd/v/virajvm/redo_photometry_plots/all_redo_figures/%s/"%sample_str
        
            check_path_existence(all_paths=[save_sample_path])
    
        if use_sample == "clean":
            #we deal with the nice clean catalogs
            #this is to check for robustness for our aperture photo pipeline
     
            top_folder = "/pscratch/sd/v/virajvm/redo_photometry_plots/all_good"
            
            #the path where we will also collect all the final summary figures
            save_sample_path = "/pscratch/sd/v/virajvm/redo_photometry_plots/all_redo_figures/%s_good/"%sample_str
        
            check_path_existence(all_paths=[save_sample_path])

        if use_sample == "sga":
            top_folder = "/pscratch/sd/v/virajvm/redo_photometry_plots/all_sga"
            save_sample_path = "/pscratch/sd/v/virajvm/redo_photometry_plots/all_redo_figures/%s/"%sample_str
        
            check_path_existence(all_paths=[save_sample_path])
            
                    
        def produce_input_dicts(k):
            '''
            Function that produces the input dictionaries that will be fed to the aperture photometry function
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
            bstar_mag = shreds_focus["STARMAG"][k]

            box_size = shreds_focus["IMAGE_SIZE_PIX"][k]
            
            if use_sample == "clean":
                pcnn_val_k = -99
            else:
                pcnn_val_k = shreds_focus["PCNN_FRAGMENT"][k]
                # pcnn_val_k = -99
            
            sample_str_i = shreds_focus["SAMPLE"][k]


            if use_sample == "sga":
                sga_dist = 0
                sga_ndist = 0
            else:
                #first one is SGA distance in degrees
                sga_dist = shreds_focus["SGA_DIST_DEG"][k]
                sga_ndist = shreds_focus["SGA_D26_NORM_DIST"][k]
            
            wcat_k = all_wcats[int(shreds_focus["is_south"][k])]
    
            sweep_folder = sweep_k.replace("-pz.fits","")
    
    
            if use_sample == "shred":
                img_path_k = f"/pscratch/sd/v/virajvm/redo_photometry_plots/all_deshreds_cutouts/image_tgid_{tgid_k}_ra_{ra_k:.3f}_dec_{dec_k:.3f}.fits" 
            if use_sample == "clean":
                img_path_k = f"/pscratch/sd/v/virajvm/redo_photometry_plots/all_good_cutouts/image_tgid_{tgid_k}_ra_{ra_k:.3f}_dec_{dec_k:.3f}.fits"
            if use_sample == "sga":
                img_path_k = f"/pscratch/sd/v/virajvm/redo_photometry_plots/all_sga_cutouts/image_tgid_{tgid_k}_ra_{ra_k:.3f}_dec_{dec_k:.3f}.fits"

            save_path_k = top_folder + "/%s/"%wcat_k + sweep_folder + "/" + brick_k + "/%s_tgid_%d"%(sample_str_i, tgid_k)
    
            if tgids_list is not None:
                print(save_path_k)

            ##read the relevant source catalog files!
            source_file = save_path_k+"/source_cat_f_more.fits"
            if os.path.exists(source_file):
                pass
            else:
                source_file = save_path_k + "/source_cat_f.fits"

            source_cat_f = Table.read(source_file)

            ##remove the potential duplicates!
            coords = np.array(list(zip(source_cat_f["ra"].data, source_cat_f["dec"].data )))
    
            # Find unique rows based on RA and DEC
            _, unique_indices = np.unique(coords, axis=0, return_index=True)
            
            # Keep only the unique rows
            source_cat_f = source_cat_f[unique_indices]
            
            ##read the relevant image files!!
            if os.path.exists(img_path_k):
                img_data = fits.open(img_path_k)
                data_arr = img_data[0].data
                invvar_arr = img_data[1].data
                mask_arr = img_data[2].data
                wcs = WCS(fits.getheader( img_path_k ))
            else:
                #in case image is not downloaded, we download it!
                print("Image is being downloaded as did not exist!")
                
                with requests.Session() as session:
                    save_cutouts(ra_k,dec_k,img_path_k,session,size=box_size)
                    
                img_data = fits.open(img_path_k)
                data_arr = img_data[0].data
                invvar_arr = img_data[1].data
                mask_arr = img_data[2].data
                wcs = WCS(fits.getheader( img_path_k ))
    
            if np.shape(data_arr[0])[0] != box_size:
                raise ValueError(f"Issue with image size here={img_path_k}. Size should be {box_size}, but is {np.shape(data_arr[0])[0]}")
        
                # import shutil
                # shutil.copy(img_path_k, save_path_k + "/")
    
            temp_dict = {"tgid":tgid_k, "ra":ra_k, "dec":dec_k, "redshift":redshift_k, "save_path":save_path_k, "img_path":img_path_k, "wcs": wcs , "image_data": data_arr, "mask_data": mask_arr, "invvar_data": invvar_arr, "source_cat": source_cat_f, "index":k , "org_mags": [ shreds_focus["MAG_G"][k], shreds_focus["MAG_R"][k], shreds_focus["MAG_Z"][k] ] , "overwrite": overwrite_bool, "image_size" :  box_size,
                        "bright_star_info": (bstar_ra, bstar_dec, bstar_radius, bstar_fdist, bstar_mag), "sga_info": (sga_dist, sga_ndist), 
                        "pcnn_val": pcnn_val_k,  "npixels_min": npixels_min, "threshold_rms_scale": threshold_rms_scale, "run_simple_photo": run_simple_photo}
    
            return temp_dict
            
        print("Number of cores used is =",ncores)
    
        #to avoid memory issues, we will split up the dataset into nchunks 
        all_ks = np.arange(len(shreds_focus))
        print(len(all_ks))
        
        all_ks_chunks = np.array_split(all_ks, nchunks)
    
        all_aper_saveimgs = []
        all_cog_saveimgs = []
        all_jaccard_saveimgs = []
        #a boolean mask for objects that we want to save in a different pdf!
        special_plot_mask = []
        all_jaccard_num_seg = []
        
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
                    all_inputs = list(tqdm(pool.imap(produce_input_dicts, all_ks_i), total = len(all_ks_i), disable=not use_tqdm   ) )
            else:
                all_inputs = []
                for i in trange(len(all_ks_i)):
                    all_inputs.append( produce_input_dicts(all_ks[i]) )
        
            all_inputs = np.array(all_inputs)
            print(len(all_inputs))
        
            ##This is the length of the list that contains info on all the objects on which aperture photometry will be run!
        
            ##################
            ##PART 4a: Run aperture photometry
            ##################

            if use_sample == "shred":
                file_clean_flag = "shreds"
            if use_sample == "clean": 
                file_clean_flag = "clean"
            if use_sample == "sga":
                file_clean_flag = "sga"
                
            file_save = f"/pscratch/sd/v/virajvm/catalog_dr1_dwarfs/iron_photometry/iron_{sample_str}_{file_clean_flag}_catalog_w_aper_mags_chunk_{chunk_i}.fits"
            
            if run_aper == True:
                if run_parr:
                    with mp.Pool(processes= ncores ) as pool:
                        results = list(tqdm(pool.imap(run_aperture_pipe, all_inputs), total = len(all_inputs), disable=not use_tqdm  ))
                else:
                    results = []
                    for i in trange(len(all_inputs)):
                        results.append( run_aperture_pipe(all_inputs[i]) )
        
                ### saving the results of the photometry pipeline
                # results = np.array(results)
            
                print_stage("Done running aperture photometry!!")

                final_close_star_dists = np.array([r["closest_star_norm_dist"] for r in results])
                final_close_star_maxmags  = np.array([r["closest_star_mag"] for r in results])
                final_aper_r3_mags     = np.vstack([r["aper_r3_mags"] for r in results])  # if same shape
               
                final_trac_mags           = np.vstack([r["tractor_dr9_mags"] for r in results]) 
                
                final_save_paths         = [r["save_path"] for r in results]
                on_segment_island        = np.array([r["lie_on_segment_island"] for r in results])
                final_image_paths       = [r["save_summary_png"] for r in results]

                aper_frac_mask_badpix = [r["aper_frac_mask_badpix"] for r in results]
                img_frac_mask_badpix = [r["img_frac_mask_badpix"] for r in results]

                if chunk_i == 0:
                    print_stage("Example outputs from aperture pipeline")
                    print("fidu aper mags shape : ",final_aper_r3_mags.shape)
                    print("final close star_dists shape : ", final_close_star_dists.shape)
                    #print some values for testing
                    print("aper_mags : ", final_aper_r3_mags[0])
                    print("aper frac mask badpix : ", aper_frac_mask_badpix[0] )
                    print("img frac mask badpix : ", img_frac_mask_badpix[0] )
                    
                    print("---"*5)
                    
                #these final image paths will be used to make a scrollable png file!
                #this is the summary aperture photo images
                all_aper_saveimgs += final_image_paths

                ##what criterion to use here when showing the special objects?
                ##if within 1.5 of the stellar radius and the star is within 45 arcsecs of ths source
                special_plot_mask += list( (shreds_focus_i["STARFDIST"] < 1.5) & (shreds_focus_i["STARDIST_DEG"]*3600 < 45) )
                

                #check that the org mags make sense
                print("Maximum Abs difference between org mags = ",np.max( np.abs(final_trac_mags[:,0] - shreds_focus_i["MAG_G"]) ))
                print("Median Abs difference between org mags = ",np.median( np.abs(final_trac_mags[:,0] - shreds_focus_i["MAG_G"]) ))
                
                #check that the org mags make sense
                # print("Maximum Abs difference between org mags = ",np.max( np.abs(final_org_mags[:,0] - shreds_focus_i["MAG_G"]) ))
                
                shreds_focus_i["NEAREST_STAR_NORM_DIST"] = final_close_star_dists
                shreds_focus_i["NEAREST_STAR_MAX_MAG"] = final_close_star_maxmags

                shreds_focus_i["APER_ORG_R3_MAG_G"] = final_aper_r3_mags[:,0]
                shreds_focus_i["APER_ORG_R3_MAG_R"] = final_aper_r3_mags[:,1]
                shreds_focus_i["APER_ORG_R3_MAG_Z"] = final_aper_r3_mags[:,2]

                shreds_focus_i["SAVE_PATH"] = final_save_paths 
                #the file path to the aper summary image
                shreds_focus_i["SAVE_SUMMARY_APER_PATH"] = final_image_paths
                
                shreds_focus_i["APER_SOURCE_ON_ORG_BLOB"] = on_segment_island.astype(bool)

                shreds_focus_i["APER_ORG_R3_FRAC_MASK_PIX"] = aper_frac_mask_badpix
                shreds_focus_i["IMAGE_MASK_PIX_FRAC"] = img_frac_mask_badpix

                print("Compute aperture-photometry based stellar masses now!")

                shreds_focus_i = compute_aperture_masses(shreds_focus_i, rband_key="APER_ORG_R3_MAG_R", gband_key="APER_ORG_R3_MAG_G", z_key="Z", output_key="LOGM_SAGA_APER_ORG_R3")
                    
                #then save this file!
                if tgids_list is None:
                    ##when saving it, let us make sure the SAVE_SUMMARY_APER_PATH is clean!
                    #also better to save it as we might not be running aper and just cog!
                    col = shreds_focus_i["SAVE_SUMMARY_APER_PATH"]
                    clean_col = np.array([x if x is not None else "" for x in col], dtype="U")
                    shreds_focus_i.replace_column("SAVE_SUMMARY_APER_PATH", clean_col)
                                    
                    save_table( shreds_focus_i, file_save)   
                    print_stage("Saved aperture summary files at %s!"%file_save)

 
            ##################
            ##PART 4b: Run cog photometry pipeline!
            ##################

            
            if run_cog == True:
                #we run the curve of growth analysis here!
                
                if run_aper == False and tgids_list is None:
                    #in this case, we want to check if the source has lie_on_segment = 1, If not we will not run this in COG
                    shreds_focus_i = Table.read(file_save)
                    lie_on_segment_i = shreds_focus_i["APER_SOURCE_ON_ORG_BLOB"].data
                    print(f"Chunk {chunk_i}: Number of sources that lie on segment = {np.sum(lie_on_segment_i)}/{len(shreds_focus_i)}")
                    #we will add something to the inputs! inputs is a list of dictionary
                    if len(lie_on_segment_i) != len(all_inputs):
                        raise ValueError("The number of rows in shreds_focus_i did not match with input dictionary length")

                    for i,input_dict_i in enumerate(all_inputs):
                        input_dict_i["LIE_ON_APER_SEGMENT"] = lie_on_segment_i[i]

                else:
                    print("As doing only a few select targetids, we will force it to run COG")
                    for input_dict_i in all_inputs:
                        input_dict_i["LIE_ON_APER_SEGMENT"] = 1

                
                if "LIE_ON_APER_SEGMENT" in all_inputs[0].keys():
                    pass
                else:
                    raise ValueError("The LIE_ON_APER_SEGMENT clause did not get added properly to the dictionary")
                #and to the input dictionary, add info on whether cog should be run or not!
    

                ##I CAN CHECK IF THE CHUNK I AM AT HAS ALREADY SAVED ALL THE COG PARAMS??
                #Do not need to repeat the analysis then
                if os.path.exists(file_save) and tgids_list is None:
                    temp = Table.read(file_save)
                    #check if the COG columns are there! Just check one of them!
                    if "COG_MAG_Z_NO_ISOLATE" in temp.colnames:
                        print("--"*5)
                        print(f"THIS CHUNK {chunk_i} ALREADY HAS COG COLUMNS AND SO SKIPPING THIS AND GOING TO NEXT FOR LOOP")
                        print("Note: if you changed something in the COG function and want it to be reflected, comment this out as it will just take previous file")
                        print(file_save)
                        print("--"*5)
                        continue
                    
                if run_parr:
                    with mp.Pool(processes= ncores ) as pool:
                        results = list(tqdm(pool.imap(run_cog_pipe, all_inputs), total = len(all_inputs), disable=not use_tqdm  ))
                else:
                    results = []
                    for i in trange(len(all_inputs)):
                        results.append( run_cog_pipe(all_inputs[i]) )
        
                ### saving the results of the photometry pipeline
        
                # Each element of results is:
                # "aper_r4_mags": all_aper_mags_r4,
                # "cog_mags": cog_mags,
                # "cog_mags_err": final_cog_mags_err,
                # "cog_params_g":  cog_output_dict["cog_params"]["g"],
                # "cog_params_r":  cog_output_dict["cog_params"]["r"],
                # "cog_params_z":  cog_output_dict["cog_params"]["z"],
                # "cog_params_g_err": cog_output_dict["cog_params_err"]["g"],
                # "cog_params_r_err": cog_output_dict["cog_params_err"]["r"],
                # "cog_params_z_err": cog_output_dict["cog_params_err"]["z"],
                # "img_path": cog_output_dict["save_img_path"],
                # "cog_chi2" : cog_output_dict["cog_chi2"], 
                # "cog_dof" : cog_output_dict["cog_dof"],
                # "aper_r4_frac_in_image": cog_output_dict["aper_r4_frac_in_image"],
                # "cog_decrease_len": cog_output_dict["cog_decrease_len"], 
                # "cog_decrease_mag": cog_output_dict["cog_decrease_len"],
                # "aper_ra_cen": cog_output_dict["aper_radec_cen"][0],
                # "aper_dec_cen": cog_output_dict["aper_radec_cen"][1],
                # "aper_xpix_cen": cog_output_dict["aper_xy_pix_cen"][0],
                # "aper_ypix_cen": cog_output_dict["aper_xy_pix_cen"][1],
                # "aper_params": cog_output_dict["aper_params"],
                # "jaccard_path": cog_output_dict["jaccard_img_path"],
                # "deblend_smooth_num_seg": cog_output_dict["deblend_smooth_num_seg"],
                # "deblend_smooth_dist_pix": cog_output_dict["deblend_smooth_dist_pix"],
                # "tractor_parent_mask_mags": cog_output_dict["parent_tractor_only_mags"],
                # "revert_to_org_tractor": cog_output_dict["revert_to_org_tractor"] 

                #ADDING COG MAGS WHEN USING ISOLATE MASK
                final_aper_r4_mags    = np.vstack([r["aper_r4_mags"] for r in results])
                final_cog_mags        = np.vstack([r["cog_mags"] for r in results])
                final_cog_mags_err    = np.array([r["cog_mags_err"] for r in results])
                final_cog_params_g     = np.vstack([r["cog_params_g"] for r in results])
                final_cog_params_r     = np.vstack([r["cog_params_r"] for r in results])
                final_cog_params_z     = np.vstack([r["cog_params_z"] for r in results])
                final_cog_params_g_err = np.vstack([r["cog_params_g_err"] for r in results])
                final_cog_params_r_err = np.vstack([r["cog_params_r_err"] for r in results])
                final_cog_params_z_err = np.vstack([r["cog_params_z_err"] for r in results])
                final_aper_r4_frac_in_image = np.array([r["aper_r4_frac_in_image"] for r in results])
                final_cog_chi2 = np.vstack( [r["cog_chi2"] for r in results] )
                final_cog_dof = np.vstack( [r["cog_dof"] for r in results] )
                final_cog_decrease_len = np.vstack( [r["cog_decrease_len"] for r in results] )
                final_cog_decrease_mag = np.vstack( [r["cog_decrease_mag"] for r in results] )
                final_aper_params = np.vstack( [r["aper_params"] for r in results] )
                final_apercen_radec = np.vstack([ r["aper_radec_cen"] for r in results])
                final_apercen_xy_pix = np.vstack([ r["aper_xy_pix_cen"] for r in results])
                final_mask_frac_r4 = np.array([ r["mask_frac_r4"] for r in results])
                

                #ADDING COG MAGS WITHOUT ISOLATE MASK
                final_aper_r4_mags_no_isolate_mask    = np.vstack([r["aper_r4_mags_no_isolate"] for r in results])
                final_cog_mags_no_isolate_mask        = np.vstack([r["cog_mags_no_isolate"] for r in results])
                final_cog_mags_err_no_isolate_mask    = np.array([r["cog_mags_err_no_isolate"] for r in results])
                final_cog_params_g_no_isolate_mask     = np.vstack([r["cog_params_g_no_isolate"] for r in results])
                final_cog_params_r_no_isolate_mask     = np.vstack([r["cog_params_r_no_isolate"] for r in results])
                final_cog_params_z_no_isolate_mask     = np.vstack([r["cog_params_z_no_isolate"] for r in results])
                final_cog_params_g_err_no_isolate_mask = np.vstack([r["cog_params_g_err_no_isolate"] for r in results])
                final_cog_params_r_err_no_isolate_mask = np.vstack([r["cog_params_r_err_no_isolate"] for r in results])
                final_cog_params_z_err_no_isolate_mask = np.vstack([r["cog_params_z_err_no_isolate"] for r in results])
                final_aper_r4_frac_in_image_no_isolate_mask = np.array([r["aper_r4_frac_in_image_no_isolate"] for r in results])
                final_cog_chi2_no_isolate_mask = np.vstack( [r["cog_chi2_no_isolate"] for r in results] )
                final_cog_dof_no_isolate_mask = np.vstack( [r["cog_dof_no_isolate"] for r in results] )
                final_cog_decrease_len_no_isolate_mask = np.vstack( [r["cog_decrease_len_no_isolate"] for r in results] )
                final_cog_decrease_mag_no_isolate_mask = np.vstack( [r["cog_decrease_mag_no_isolate"] for r in results] )
                final_aper_params_no_isolate_mask = np.vstack( [r["aper_params_no_isolate"] for r in results] )
                final_apercen_radec_no_isolate_mask = np.vstack([ r["aper_radec_cen_no_isolate"] for r in results])
                final_apercen_xy_pix_no_isolate_mask = np.vstack([ r["aper_xy_pix_cen_no_isolate"] for r in results])
                final_mask_frac_r4_no_isolate_mask = np.array([ r["mask_frac_r4_no_isolate"] for r in results])

                ##get the tractor cog params!!
                empty_tractor_cog_dict = make_empty_tractor_cog_dict
                
                
                

                ##THESE ARE COG OUTPUTS THAT ARE COMMON TO ALL!
                final_deblend_smooth_num_seg = np.array([ r["deblend_smooth_num_seg"] for r in results])
                final_revert_to_org_tractor = np.array([ r["revert_to_org_tractor"] for r in results])
                final_tractor_parent_isolate_mags = np.vstack([ r["tractor_parent_isolate_mags"] for r in results])
                final_tractor_parent_no_isolate_mags = np.vstack([ r["tractor_parent_no_isolate_mags"] for r in results])
                
                #the mu grz values useful in identifying when a galaxy is too LSB to use the smooth deblending params
                final_aper_r2_mu_r_ellipse_tractor= np.array([ r["aper_r2_mu_r_ellipse_tractor"] for r in results ]  )
                final_aper_r2_mu_r_island_tractor = np.array([ r["aper_r2_mu_r_island_tractor"] for r in results ]  )
                

                final_deblend_blob_dist_pix = np.array([ r["deblend_blob_dist_pix"] for r in results])
                final_cog_segment_nseg = np.array([ r["cog_segment_nseg"] for r in results])
                final_cog_segment_nseg_smooth = np.array([ r["cog_segment_nseg_smooth"] for r in results])
                final_cog_segment_on_blob = np.array([ r["cog_segment_on_blob"] for r in results])
                final_num_trac_source_no_isolate = np.array([ r["num_trac_source_no_isolate"] for r in results])
                final_num_trac_source_isolate = np.array([ r["num_trac_source_isolate"] for r in results])
                
                final_areafrac_in_image_r4 = np.array([ r["areafrac_in_image_r4"] for r in results])
                
                final_simple_mags     = np.vstack([r["simple_photo_mags"] for r in results]) 
                final_simple_island_dist_pix     = np.array([r["simple_photo_island_dist_pix"] for r in results])
                final_simple_aper_r4_frac_in_image     = np.array([r["simplest_photo_aper_frac_in_image"] for r in results])
                
                  
                #add the image paths to the total lists

                final_cog_saveimgs = filter_saveimgs_paths(results, "img_path")
                final_jaccard_saveimgs = filter_saveimgs_paths(results, "jaccard_path")
                
                all_cog_saveimgs += final_cog_saveimgs
                all_jaccard_saveimgs += final_jaccard_saveimgs
                
                if chunk_i == 0:
                    print_stage("Example outputs from cog pipeline")
                    print("final aper mags R4 shape : ",final_aper_r4_mags.shape)
                    print("final cog mags shape : ",final_cog_mags.shape)
                    print("final cog chi2 mags shape : ",final_cog_chi2.shape)
                    print("final cog params g-band shape : ",final_cog_mags.shape)
                    #print some values for testing
                    print("cog_mags : ", final_cog_mags[0])
                    print("cog_mags_err : ", final_cog_mags_err[0])
                    print("cog params g : ",final_cog_params_g[0])
                    print("cog params g err  : ", final_cog_params_g_err[0])
                    print("cog aper frac-in-image : ", final_aper_r4_frac_in_image[0] )
                    print("cog saveimg : ", final_cog_saveimgs[0])
                    print("---"*5)
                    
                print_stage("Done running curve of growth pipeline!")

                #if run_aper was not run before then we read in the files again
                #this is in the rare case (maybe only the first time) where we will have run cog separately from run_aper as we do not have the tractor models yet
                if run_aper == False:
                    shreds_focus_i = Table.read(file_save)
                #if run_aper = True, then we already have defined shreds_focus_i variable!
               
                # Note: If running on only a few TGIDs, always run `run_aper()` first,
                # otherwise 'shreds_focus_i' will not be defined and this step will fail.
                try:
                    _ = shreds_focus_i
                except NameError:
                    raise RuntimeError("Missing variable: 'shreds_focus_i'. Did you forget to run `run_aper()`? This would be the case if you are running this on a few TGIDs.")

                ##these are columns to be added per band
                shreds_focus_i["COG_MAG_G_ISOLATE"] = final_cog_mags[:,0]
                shreds_focus_i["COG_MAG_R_ISOLATE"] = final_cog_mags[:,1]
                shreds_focus_i["COG_MAG_Z_ISOLATE"] = final_cog_mags[:,2]

                shreds_focus_i["COG_MAG_G_NO_ISOLATE"] = final_cog_mags_no_isolate_mask[:,0]
                shreds_focus_i["COG_MAG_R_NO_ISOLATE"] = final_cog_mags_no_isolate_mask[:,1]
                shreds_focus_i["COG_MAG_Z_NO_ISOLATE"] = final_cog_mags_no_isolate_mask[:,2]
                
                shreds_focus_i["APER_R4_MAG_G_ISOLATE"] = final_aper_r4_mags[:,0]
                shreds_focus_i["APER_R4_MAG_R_ISOLATE"] = final_aper_r4_mags[:,1]
                shreds_focus_i["APER_R4_MAG_Z_ISOLATE"] = final_aper_r4_mags[:,2]

                shreds_focus_i["APER_R4_MAG_G_NO_ISOLATE"] = final_aper_r4_mags_no_isolate_mask[:,0]
                shreds_focus_i["APER_R4_MAG_R_NO_ISOLATE"] = final_aper_r4_mags_no_isolate_mask[:,1]
                shreds_focus_i["APER_R4_MAG_Z_NO_ISOLATE"] = final_aper_r4_mags_no_isolate_mask[:,2]

                shreds_focus_i["APER_R4_FRAC_IN_IMG_ISOLATE"] = final_aper_r4_frac_in_image
                shreds_focus_i["APER_R4_FRAC_IN_IMG_NO_ISOLATE"] = final_aper_r4_frac_in_image_no_isolate_mask
                
                #these columns are common to both! This is with the isolate mask applied
                shreds_focus_i["TRACTOR_PARENT_MAG_G_ISOLATE"] = final_tractor_parent_isolate_mags[:,0]
                shreds_focus_i["TRACTOR_PARENT_MAG_R_ISOLATE"] = final_tractor_parent_isolate_mags[:,1]
                shreds_focus_i["TRACTOR_PARENT_MAG_Z_ISOLATE"] = final_tractor_parent_isolate_mags[:,2]
                
                shreds_focus_i["TRACTOR_PARENT_MAG_G_NO_ISOLATE"] = final_tractor_parent_no_isolate_mags[:,0]
                shreds_focus_i["TRACTOR_PARENT_MAG_R_NO_ISOLATE"] = final_tractor_parent_no_isolate_mags[:,1]
                shreds_focus_i["TRACTOR_PARENT_MAG_Z_NO_ISOLATE"] = final_tractor_parent_no_isolate_mags[:,2]

                shreds_focus_i["DEBLEND_SMOOTH_NUM_BLOB"] = final_deblend_smooth_num_seg
                shreds_focus_i["REVERT_TO_OLD_TRACTOR"] = final_revert_to_org_tractor

                shreds_focus_i["APER_R2_MU_R_ELLIPSE_TRACTOR"] = final_aper_r2_mu_r_ellipse_tractor
                shreds_focus_i["APER_R2_MU_R_ISLAND_TRACTOR"] = final_aper_r2_mu_r_island_tractor
        
                shreds_focus_i["DEBLEND_BLOB_DIST_PIX"] = final_deblend_blob_dist_pix
                shreds_focus_i["COG_NUM_SEG"] = final_cog_segment_nseg
                shreds_focus_i["COG_NUM_SEG_SMOOTH"] = final_cog_segment_nseg_smooth
                shreds_focus_i["COG_SEG_ON_BLOB"] = final_cog_segment_on_blob
                
                shreds_focus_i["NUM_TRACTOR_SOURCES_NO_ISOLATE"] = final_num_trac_source_no_isolate
                shreds_focus_i["NUM_TRACTOR_SOURCES_ISOLATE"] = final_num_trac_source_isolate

                #this is for aperture estimated on the grz data main segment (check beginning of aperture cog function)
                shreds_focus_i["APER_R4_DATA_FRAC_IN_IMAGE_NO_ISOLATE"] =  final_areafrac_in_image_r4
                     
                shreds_focus_i["SIMPLE_PHOTO_MAG_G"] = final_simple_mags[:,0]
                shreds_focus_i["SIMPLE_PHOTO_MAG_R"] = final_simple_mags[:,1]
                shreds_focus_i["SIMPLE_PHOTO_MAG_Z"] = final_simple_mags[:,2]
                
                shreds_focus_i["SIMPLE_APER_R4_FRAC_IN_IMG"] = final_simple_aper_r4_frac_in_image
                shreds_focus_i["SIMPLE_BLOB_DIST_PIX"] = final_simple_island_dist_pix
                


                ##these are columns that will arrays 
                column_dict = {"COG_CHI2_ISOLATE": final_cog_chi2,
                              "COG_DOF_ISOLATE": final_cog_dof,
                              "COG_MAG_ERR_ISOLATE" : final_cog_mags_err,
                               "COG_PARAMS_G_ISOLATE": final_cog_params_g,
                               "COG_PARAMS_R_ISOLATE": final_cog_params_r,
                               "COG_PARAMS_Z_ISOLATE": final_cog_params_z,
                                "COG_PARAMS_G_ERR_ISOLATE": final_cog_params_g_err,
                               "COG_PARAMS_R_ERR_ISOLATE": final_cog_params_r_err,
                               "COG_PARAMS_Z_ERR_ISOLATE": final_cog_params_z_err,
                               "COG_DECREASE_MAX_L3EN_ISOLATE": final_cog_decrease_len,
                               "COG_DECREASE_MAX_MAG_ISOLATE": final_cog_decrease_mag,
                               'APER_CEN_RADEC_ISOLATE': final_apercen_radec,
                               'APER_CEN_XY_PIX_ISOLATE': final_apercen_xy_pix,
                               'APER_PARAMS_ISOLATE':final_aper_params,   
                               'APER_R4_MASK_FRAC_ISOLATE': final_mask_frac_r4
                              }
                
                column_dict_no_isolate = {
                               "COG_CHI2_NO_ISOLATE": final_cog_chi2_no_isolate_mask,
                              "COG_DOF_NO_ISOLATE": final_cog_dof_no_isolate_mask,
                              "COG_MAG_ERR_NO_ISOLATE" : final_cog_mags_err_no_isolate_mask,
                               "COG_PARAMS_G_NO_ISOLATE": final_cog_params_g_no_isolate_mask,
                               "COG_PARAMS_R_NO_ISOLATE": final_cog_params_r_no_isolate_mask,
                               "COG_PARAMS_Z_NO_ISOLATE": final_cog_params_z_no_isolate_mask,
                                "COG_PARAMS_G_ERR_NO_ISOLATE": final_cog_params_g_err_no_isolate_mask,
                               "COG_PARAMS_R_ERR_NO_ISOLATE": final_cog_params_r_err_no_isolate_mask,
                               "COG_PARAMS_Z_ERR_NO_ISOLATE": final_cog_params_z_err_no_isolate_mask,
                               "COG_DECREASE_MAX_LEN_NO_ISOLATE": final_cog_decrease_len_no_isolate_mask,
                               "COG_DECREASE_MAX_MAG_NO_ISOLATE": final_cog_decrease_mag_no_isolate_mask,
                               'APER_CEN_RADEC_NO_ISOLATE': final_apercen_radec_no_isolate_mask,
                               'APER_CEN_XY_PIX_NO_ISOLATE': final_apercen_xy_pix_no_isolate_mask,
                               'APER_PARAMS_NO_ISOLATE':final_aper_params_no_isolate_mask,
                               'APER_R4_MASK_FRAC_NO_ISOLATE': final_mask_frac_r4_no_isolate_mask
                              }

                TODO: ADD A SINGLE DIAGNOSTIC CUTOUT PLOT SHOWING THE ORIGINAL GRZ IMAGE WITH CENTER AND APERTURE, along with the final masked reconstructed iamge .
                THIS WILL BE USED FOR A QUICK DISPLAY FOR HOW THE OBJECT IS ! And make an associated function that given ra,dec or targetid, will pull up the diagnostic plot if relevant
                

                #combining the above dictionaries into one!
                column_dict_total = column_dict | column_dict_no_isolate

                #this should automatically add columns!
                for ki, values in column_dict_total.items():
                    shreds_focus_i[ki] = values

                #Get the cog based stellar mass, get the 2 kinds of stellar masses!!    
                
                shreds_focus_i = compute_aperture_masses(shreds_focus_i, rband_key="COG_MAG_R_ISOLATE", gband_key="COG_MAG_G_ISOLATE", z_key="Z_CMB", dmpc_key = "DIST_MPC_FIDU", output_key="LOGM_SAGA_COG_ISOLATE")

                shreds_focus_i = compute_aperture_masses(shreds_focus_i, rband_key="COG_MAG_R_NO_ISOLATE", gband_key="COG_MAG_G_NO_ISOLATE", z_key="Z_CMB", dmpc_key = "DIST_MPC_FIDU", output_key="LOGM_SAGA_COG_NO_ISOLATE")
                
                if tgids_list is None:
                    save_table( shreds_focus_i, file_save)   
                    print_stage("Saved aperture summary files at %s!"%file_save)
                    
    ##################
    ##PART 5: Once all the chunks are done, combine them all!
    ##################

    yes_save = not no_save
    # if tgids_list is None and (run_aper | run_cog) and yes_save:
    if tgids_list is None and (run_cog) and yes_save:
        
        print_stage("Consolidating all the saved chunks!")
        
        #files were saved and so we will consolidate them!
        if use_sample == "shred":
            clean_flag = "shreds"
        if use_sample == "clean":
            clean_flag = "clean"
        if use_sample == "sga":
            clean_flag = "sga"

        file_template_aper = "/pscratch/sd/v/virajvm/catalog_dr1_dwarfs/iron_photometry/iron_%s_%s_catalog_w_aper_mags"%(sample_str, clean_flag)
 
        if nchunks == 1:
            shreds_focus_combine_aper = Table.read(file_template_aper + "_chunk_0.fits")

            final_save_name = file_template_aper + "%s.fits"%end_name
            
            #we just need to rename the file!
            shutil.copy(file_template_aper + "_chunk_0.fits", final_save_name)
            if run_cog == True:
                #see the comments below for why this if statement is here
                os.rename(file_template_aper + "_chunk_0.fits", final_save_name)

        else:
            #we need to consolidate files!  
            shreds_focus_combine_aper = []

            final_save_name = file_template_aper + "%s.fits"%end_name
            
            for ni in trange(nchunks):
                shreds_focus_part = Table.read(file_template_aper + "_chunk_%d.fits"%ni  )
                shreds_focus_combine_aper.append(shreds_focus_part)
                #and then we delete that file!
                if run_cog == True:
                    #we only delete this file if cog is also run. COG needs this file and so we do not want to remove it if COG not run yet
                    os.remove(file_template_aper + "_chunk_%d.fits"%ni)
                
            shreds_focus_combine_aper = vstack(shreds_focus_combine_aper)

            print_stage("Total number of objects in consolidated aperture file = %d"%len(shreds_focus_combine_aper))

            save_table( shreds_focus_combine_aper, final_save_name ) 
            
            print_stage("Consolidated aperture chunk saved at %s"%(final_save_name) )

    ##make a scrollable pdf to view the final results!
    ##only make for some objects?


    # ##################
    # ##PART 5: Get the CNN inputs ready!!
    # ##################

    if get_cnn_inputs and tgids_list is None and run_cog:
        print("Getting the PCNN shred classifier input files")
        get_pcnn_data_inputs(sample_str, sample_cat_path = final_save_name)

    if tgids_list is None and run_aper:
        print("Saving aperture photometry imgs")
        make_scroll_pdf(all_aper_saveimgs, save_sample_path, summary_scroll_file_name, None, max_num=250, type_str="aper",all_aper_saveimgs2=None)

    if tgids_list is None and run_cog:
        print("Saving COG aperture photometry imgs")
        
        make_scroll_pdf(all_cog_saveimgs, save_sample_path, summary_scroll_file_name, None, max_num=250, type_str="cog",all_aper_saveimgs2=None)
        
    ##save a scrollable pdf of all the jaccard pngs!
    if tgids_list is None and run_cog:
        print("Saving jaccard score imgs")
        make_scroll_pdf(all_jaccard_saveimgs, save_sample_path, summary_scroll_file_name, None, max_num=250, type_str="jaccard",all_aper_saveimgs2=None)

        
    if tgids_list is None and run_cog:
        print("Saving special jaccard score imgs")
        
        special_jaccard_mask = (shreds_focus_combine_aper["DEBLEND_SMOOTH_NUM_BLOB"].data > 1)

        make_scroll_pdf(all_jaccard_saveimgs, save_sample_path, summary_scroll_file_name, None, max_num=250, type_str="jaccard",all_aper_saveimgs2=None)


    # if tgids_list is None and run_cog:
    #     special_plot_mask = (shreds_focus_combine_aper["STARFDIST"] < 1.5) & (shreds_focus_combine_aper["STARDIST_DEG"]*3600 < 45)
    #     all_aper_saveimgs = shreds_focus_combine_aper["SAVE_SUMMARY_APER_PATH"].data
    #     #we retranslate this into None!
    #     all_aper_saveimgs = np.array([x if x != "" else None for x in all_aper_saveimgs], dtype=object)
                
    #     print("Saving COG and aperture photometry imgs")
    #     make_scroll_pdf(all_aper_saveimgs, save_sample_path, summary_scroll_file_name, special_plot_mask, max_num=250, type_str="apercog",all_aper_saveimgs2=all_cog_saveimgs)

    # ##### CODE FOR TESTING APERTURE PHOTOMETRY ON BAD RCHISQ OBJECTS
    # # rchisq_bins = np.arange(0,9,1)
    # # shreds_focus = []
    # # np.random.seed(42)
    # # for i,ri in enumerate(rchisq_bins[:-1]):
    # #     # print(ri, rchisq_bins[i+1])
    # #     temp_i = shreds_all[ (shreds_all["RCHISQ_R"] > ri) & (shreds_all["RCHISQ_R"] < rchisq_bins[i+1]) & (shreds_all["FRACFLUX_G"] < 0.01) & (shreds_all["FRACFLUX_R"] < 0.01) & (shreds_all["FRACFLUX_Z"] < 0.01) & (shreds_all["MASKBITS"] == 0) & (shreds_all["SAMPLE"] == sample_str)  ]
    # #     #pick random 100 objects!
    # #     print(len(temp_i))
    # #     temp_ij = temp_i[ np.random.randint( len(temp_i) ,size = np.minimum( 500 , len(temp_i))   ) ]
    # #     shreds_focus.append(temp_ij)
    # # shreds_focus = vstack(shreds_focus)

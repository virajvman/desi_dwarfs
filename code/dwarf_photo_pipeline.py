'''
The different things that are parallelized:

When creating the relevant_files_for_aper

1) The creating 

'''
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

def parse_tgids(value):
    if not value:
        return None
    return [int(x) for x in value.split(',')]



def generate_random_string(length=5):
    return ''.join(random.choices(string.ascii_lowercase + string.digits, k=length))


def stack_images_vertically(img_path1, img_path2):
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

def make_scroll_pdf(
    all_aper_saveimgs1,
    save_sample_path,
    summary_scroll_file_name,
    special_plot_mask,
    max_num=500,
    type_str="aper",
    all_aper_saveimgs2=None
):
    if summary_scroll_file_name == "":
        summary_scroll_file_name = f"images_{type_str}_" + generate_random_string(5) + ".pdf"
    output_pdf = save_sample_path + "%s" % summary_scroll_file_name

    all_aper_saveimgs1 = np.array(all_aper_saveimgs1)
    if all_aper_saveimgs2 is not None:
        all_aper_saveimgs2 = np.array(all_aper_saveimgs2)
        assert len(all_aper_saveimgs1) == len(all_aper_saveimgs2), "Both image lists must be the same length."

    # Prepare main images
    if all_aper_saveimgs2 is not None:
        image_pairs = list(zip(all_aper_saveimgs1[:max_num], all_aper_saveimgs2[:max_num]))
        images = [stack_images_vertically(img1, img2) for img1, img2 in image_pairs]
    else:
        image_files = all_aper_saveimgs1[:max_num]
        images = [Image.open(img).convert("RGB") for img in image_files]

    # Save main PDF
    if len(images) > 0:
        images[0].save(output_pdf, save_all=True, append_images=images[1:])
        print(f"Photo summary images saved at {output_pdf}")
    else:
        print("No images to save.")

    # Handle special subset
    if special_plot_mask is not None:
        special_plot_mask = np.array(special_plot_mask)
        special_imgs1 = all_aper_saveimgs1[special_plot_mask][:max_num]

        if all_aper_saveimgs2 is not None:
            special_imgs2 = all_aper_saveimgs2[special_plot_mask][:max_num]
            special_pairs = list(zip(special_imgs1, special_imgs2))
            special_images = [stack_images_vertically(img1, img2) for img1, img2 in special_pairs]
        else:
            special_images = [Image.open(img).convert("RGB") for img in special_imgs1]

        if len(special_images) > 0:
            date_str = datetime.now().strftime("%Y-%m-%d")
            output_special_pdf = save_sample_path + f"images_special_{type_str}_{date_str}_{generate_random_string(3)}.pdf"
            special_images[0].save(output_special_pdf, save_all=True, append_images=special_images[1:])
            print(f"Special photo summary images saved at {output_special_pdf}")
        else:
            print("No special images to save.")

    return


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
    result.add_argument('-run_cog',dest='run_cog', action='store_true')
    result.add_argument('-nchunks',dest='nchunks', type=int,default = 1)
    result.add_argument('-no_save',dest='no_save', action = "store_true")
    result.add_argument('-make_main_cats',dest='make_main_cats', action = "store_true")
    result.add_argument('-end_name',dest='end_name', type = str, default = "")
    result.add_argument('-no_cnn_cut',dest='no_cnn_cut', action='store_true')
    
    return result




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
    
    # top_path_k = top_folder + "/%s/"%wcat + sweep_folder + "/" + brick_i + "/%s_tgid_%d"%(samp_k, tgid_k)
    top_path_k = f"{top_folder}/{wcat}/{sweep_folder}/{brick_i}/{samp_k}_tgid_{tgid_k}"

    # print(top_path_k)

    check_path_existence(all_paths=[top_path_k])
    #inside this folder, we will save all the relevant files and info!!
    image_path =  top_folder + "_cutouts/image_tgid_%d_ra_%f_dec_%f.fits"%(tgid_k,ra_k,dec_k)
    

    get_nearby_source_catalog(ra_k, dec_k, wcat, brick_i, top_path_k, source_cat, source_pzs_i)
    
    ## check if the source is at the edge of the brick, if so we will need to combine stuff
    more_bricks, more_wcats, more_sweeps = are_more_bricks_needed(ra_k,dec_k,radius_arcsec = 45)

    if len(more_bricks) == 0:
        #there are no neighboring bricks needed
        pass
    else:
        return_sources_wneigh_bricks(top_path_k, ra_k, dec_k, more_bricks, more_wcats, more_sweeps,use_pz = False)
        

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

            #as we are multi-threading the processing for each galaxy, and sessions are not thread safe
            #we will be creating a session per image and then closing it right away!
            with requests.Session() as session:
                save_cutouts(ra_k, dec_k, image_path, session, size=350)
        
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
    
    ##construct the clean catalog
    ## to make things easy in the cleaning stage, we will use the optical based colors
    ## we will probably stick with these as our fiducial stellar masses

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

    bstar_keys = [ "STARFDIST", "STARDIST_DEG","STARMAG", "STAR_RADIUS_ARCSEC", "STAR_RA","STAR_DEC"]

    # Check if all bright star keys exist
    if all(key in all_clean_dwarfs.colnames for key in bstar_keys):
        print("Bright star information already exists!")
    else:
        # Recompute if missing
        print("Bright star information did not exist and will be computed.")
        all_clean_dwarfs = bright_star_filter(all_clean_dwarfs)
    

    # save the clean dwarfs now!
    save_table(all_clean_dwarfs,"/pscratch/sd/v/virajvm/catalog_dr1_dwarfs/desi_y1_dwarf_clean_catalog_v3.fits",comment="This is a compilation of dwarf galaxy candidates in DESI Y1 data from the BGS Bright, BGS Faint, ELG and LOW-Z samples. Only galaxies with LogMstar < 9.5 (w/SAGA based stellar masses) and that have robust photometry are included.")
        
    ##applying the mask and then stacking them!!

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

    ##if the existing file already has SGA info, then we just add that
    shreds_all = get_sga_norm_dists_FAST(shreds_all, siena_path="/global/cfs/cdirs/cosmo/data/sga/2020/SGA-2020.fits")

    # Check if the bright star key already exist
    if all(key in shreds_all.colnames for key in bstar_keys):
        print("Bright star information already exists!")
    else:
        # Recompute if missing
        print("Bright star information did not exist and will be computed.")
        shreds_all = bright_star_filter(shreds_all)
    
    save_table(shreds_all,"/pscratch/sd/v/virajvm/catalog_dr1_dwarfs/desi_y1_dwarf_shreds_catalog_v3.fits")
        
    return 



def create_brick_jobs(brick_i, shreds_focus_w, wcat, top_folder):
    '''
    In this function, we parallelize the reading of the source cat and creation of the galaxy dicts!!
    We will be multi-threading here so it will share the memory of shreds_focus_w catalog!!

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
            "wcat": wcat,
            "brick_i": brick_i,
            "sweep_folder": sweep_folder,
            "top_folder": top_folder,
            "source_cat": source_cat,
            "source_pzs_i": None
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

    
def compute_aperture_masses(shreds_table, rband_key="MAG_R_APERTURE_R375", gband_key="MAG_G_APERTURE_R375", z_key="Z", output_key="LOGM_SAGA_APERTURE_R375"):
    """
    Compute aperture-photometry-based stellar masses using g - r color, r-band magnitude, and redshift.

    Parameters
    ----------
    shreds_table : astropy Table or numpy structured array
        Input table with aperture magnitudes and redshift.

    rband_key : str
        Column name for r-band aperture magnitude.

    gband_key : str
        Column name for g-band aperture magnitude.

    z_key : str
        Column name for redshift.

    output_key : str
        Name of the output column to store log stellar masses.

    Returns
    -------
    shreds_table : same as input
        Modified in-place (and also returned) with a new column containing log stellar masses.
    """
    rmag_aper = shreds_table[rband_key].data
    gr_aper = shreds_table[gband_key].data - rmag_aper

    nan_mask = np.isnan(rmag_aper) | np.isnan(shreds_table[gband_key].data)

    all_mstar_aper = np.full(len(shreds_table), np.nan)
    mstar_aper_nonan = get_stellar_mass(gr_aper[~nan_mask], rmag_aper[~nan_mask], shreds_table[z_key][~nan_mask])
    all_mstar_aper[~nan_mask] = mstar_aper_nonan

    shreds_table[output_key] = all_mstar_aper
    return shreds_table
    
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
    no_cnn_cut = args.no_cnn_cut
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
    
    run_aper = args.run_aper
    run_cog = args.run_cog

    ## can I come up with a robust way to choose box size?
    box_size = 350

    npixels_min = 10
    threshold_rms_scale = 1.5
    
    c_light = 299792 #km/s

    ##################
    ##PART 1: Make the clean and shredded catalogs!
    ##################

    if make_main_cats:
        make_clean_shreds_catalogs()

    ##add the columns on image path and file_path to these catalogs!!
    shreds_file = "/pscratch/sd/v/virajvm/catalog_dr1_dwarfs/desi_y1_dwarf_shreds_catalog_v3.fits"
    # shreds_file = "/pscratch/sd/v/virajvm/catalog_dr1_dwarfs/TEMPORARY_desi_y1_dwarf_shreds_catalog_v3.fits"
    
    clean_file = "/pscratch/sd/v/virajvm/catalog_dr1_dwarfs/desi_y1_dwarf_clean_catalog_v3.fits"

    if use_clean == False:
        top_folder = "/pscratch/sd/v/virajvm/redo_photometry_plots/all_deshreds"
    else:
        top_folder = "/pscratch/sd/v/virajvm/redo_photometry_plots/all_good"

    shreds_cat = Table.read(shreds_file)
    clean_cat = Table.read(clean_file)

    ##we need to be careful here in how we are defining the top folder!! 
    if "IMAGE_PATH" in shreds_cat.colnames and "FILE_PATH" in shreds_cat.colnames:
        print("image_path and file_path columns already exist in shreds catalog!")        
    else:
        print("Adding image_path and file_path to shreds catalog!")
        
        add_paths_to_catalog(org_file = shreds_file, out_file = shreds_file,top_folder="/pscratch/sd/v/virajvm/redo_photometry_plots/all_deshreds")
        #add those two paths to the file!

    if "IMAGE_PATH" in clean_cat.colnames and "FILE_PATH" in clean_cat.colnames:
        print("image_path and file_path columns already exist in clean catalog!")
    else:
        print("Adding image_path and file_path to clean catalog!")
        add_paths_to_catalog(org_file = clean_file, out_file = clean_file,top_folder="/pscratch/sd/v/virajvm/redo_photometry_plots/all_good")

    #delete these variables as no longer needed!
    del shreds_cat, clean_cat
    
    ##################
    ##PART 2: Generate nested folder structure with relevant files for doing photometry
    ##################

    ##load the relevant catalogs!
    if use_clean==False:
        shreds_all = Table.read(shreds_file)
        # shreds_all = shreds_all[  shreds_all["SGA_D26_NORM_DIST"] > 1.5 ]
    else:
        shreds_all = Table.read(clean_file)

    ##let us do a quick tally on how many of the image paths are blank! that is, they need new image paths!
    print("In this catalog, " + str(len(shreds_all[shreds_all['IMAGE_PATH'] == ""])) + " objects do not have image paths!")
    
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


    #we also apply a mask on the PCNN probabilities if relevant
    if no_cnn_cut:
        pass
    else:
        #filter for sources that CNN thinks are fragments
        shreds_focus = shreds_focus[ shreds_focus["PCNN_FRAGMENT"] >= 0.4]

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
    
        print(f"Number of objects in south: {len(shreds_focus[shreds_focus['is_south'] == 1])}" )
        print(f"Number of objects in north: {len(shreds_focus[shreds_focus['is_south'] == 0])}" )
        
        ##I do not need the photo-z file and so we will not bother loading it!
    
        for wcat_ind, wcat in enumerate(["north","south"]):
            check_path_existence(all_paths=[top_folder + "/%s"%wcat])

            #if we are not using the photo-z file, we can parallize directly over the bricks!

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

            if use_clean:
                pcnn_val_k = -99
            else:
                pcnn_val_k = shreds_focus["PCNN_FRAGMENT"][k]
            
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
                wcs = WCS(fits.getheader( img_path_k ))
            else:
                #in case image is not downloaded, we download it!
                print("Image is being downloaded as did not exist!")
                
                with requests.Session() as session:
                    save_cutouts(ra_k,dec_k,img_path_k,session,size=350)
                    
                img_data = fits.open(img_path_k)
                data_arr = img_data[0].data
                wcs = WCS(fits.getheader( img_path_k ))
    
            if np.shape(data_arr[0])[0] != 350:
                raise ValueError("Issue with image size here=%s"%img_path_k)
        
                # import shutil
                # shutil.copy(img_path_k, save_path_k + "/")
    
            temp_dict = {"tgid":tgid_k, "ra":ra_k, "dec":dec_k, "redshift":redshift_k, "save_path":save_path_k, "img_path":img_path_k, "wcs": wcs , "image_data": data_arr, "source_cat": source_cat_f, "index":k , "org_mag_g": shreds_focus["MAG_G"][k], "overwrite": overwrite_bool, "box_size" : box_size,
                        "bright_star_info": (bstar_ra, bstar_dec, bstar_radius, bstar_fdist), "sga_info": (sga_dist, sga_ndist), 
                        "pcnn_val": pcnn_val_k,  "npixels_min": npixels_min, "threshold_rms_scale": threshold_rms_scale}
    
            return temp_dict
            
        print("Number of cores used is =",ncores)
    
        #to avoid memory issues, we will split up the dataset into nchunks 
        all_ks = np.arange(len(shreds_focus))
        print(len(all_ks))
        
        all_ks_chunks = np.array_split(all_ks, nchunks)
    
        all_aper_saveimgs = []
        all_cog_saveimgs = []
        #a boolean mask for objects that we want to save in a different pdf!
        special_plot_mask = []
            
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
            ##################

            if use_clean == False:
                file_clean_flag = "shreds"
            else:
                file_clean_flag = "clean"
            file_save = f"/pscratch/sd/v/virajvm/catalog_dr1_dwarfs/iron_photometry/iron_{sample_str}_{file_clean_flag}_catalog_w_aper_mags_chunk_{chunk_i}.fits"
            
            if run_aper == True:
                if run_parr:
                    with mp.Pool(processes= ncores ) as pool:
                        results = list(tqdm(pool.imap(run_aperture_pipe, all_inputs), total = len(all_inputs)  ))
                else:
                    results = []
                    for i in trange(len(all_inputs)):
                        results.append( run_aperture_pipe(all_inputs[i]) )
        
                ### saving the results of the photometry pipeline
                # results = np.array(results)
            
                print_stage("Done running aperture photometry!!")

                final_close_star_dists = np.array([r["closest_star_norm_dist"] for r in results])
                final_close_star_maxmags  = np.array([r["closest_star_mag"] for r in results])
                final_fidu_aper_mags     = np.vstack([r["fidu_aper_mags"] for r in results])  # if same shape
                final_org_mags           = np.vstack([r["org_mags"] for r in results])  
                
                final_save_paths         = [r["save_path"] for r in results]
                final_image_paths       = [r["save_summary_png"] for r in results]
                all_img_data_paths          = [r["img_path"] for r in results]

                if chunk_i == 0:
                    print_stage("Example outputs from aperture pipeline")
                    print("fidu aper mags shape : ",final_fidu_aper_mags.shape)
                    print("final close star_dists shape : ", final_close_star_dists.shape)
                    #print some values for testing
                    print("aper_mags : ", final_fidu_aper_mags[0])
                    print("org_mags : ", final_org_mags[0])
                    print("---"*5)
                    
                #these final image paths will be used to make a scrollable png file!
                #this is the summary aperture photo images
                all_aper_saveimgs += final_image_paths

                ##what criterion to use here when showing the special objects?
                ##if within 1.5 of the stellar radius and the star is within 45 arcsecs of ths source
                special_plot_mask += list( (shreds_focus_i["STARFDIST"] < 1.5) & (shreds_focus_i["STARDIST_DEG"]*3600 < 45) )
                
        
                #check that the org mags make sense
                print("Maximum Abs difference between org mags = ",np.max( np.abs(final_org_mags[:,0] - shreds_focus_i["MAG_G"]) ))
                
                shreds_focus_i["NEAREST_STAR_NORM_DIST"] = final_close_star_dists
                shreds_focus_i["NEAREST_STAR_MAX_MAG"] = final_close_star_maxmags
                
                shreds_focus_i["MAG_G_APERTURE_R375"] = final_fidu_aper_mags[:,0]
                shreds_focus_i["MAG_R_APERTURE_R375"] = final_fidu_aper_mags[:,1]
                shreds_focus_i["MAG_Z_APERTURE_R375"] = final_fidu_aper_mags[:,2]

                shreds_focus_i["SAVE_PATH"] = final_save_paths 
                shreds_focus_i["IMAGE_FITS_PATH"] = all_img_data_paths
                            
                print("Compute aperture-photometry based stellar masses now!")

                shreds_focus_i = compute_aperture_masses(shreds_focus_i, rband_key="MAG_R_APERTURE_R375", gband_key="MAG_G_APERTURE_R375", z_key="Z", output_key="LOGM_SAGA_APERTURE_R375")
                    
                #then save this file!
                if tgids_list is None:
                    save_table( shreds_focus_i, file_save)   
                    print_stage("Saved aperture summary files at %s!"%file_save)

 
            ##################
            ##PART 4b: Run cog photometry pipeline!
            ##################

            
            if run_cog == True:
                #we run the curve of growth analysis here!

                if run_parr:
                    with mp.Pool(processes= ncores ) as pool:
                        results = list(tqdm(pool.imap(run_cog_pipe, all_inputs), total = len(all_inputs)  ))
                else:
                    results = []
                    for i in trange(len(all_inputs)):
                        results.append( run_cog_pipe(all_inputs[i]) )
        
                ### saving the results of the photometry pipeline
        
                # Each element of results is:
                # [0] cog_mags             
                # [1] final_cog_mags_err    
                # [2] final_cog_params_g    --> list of cog fit parameters  
                # [3] final_cog_params_r     
                # [4] final_cog_params_z     
                # [5] final_cog_params_g_err  
                # [6] final_cog_params_r_err  
                # [7] final_cog_params_z_err 
                # [8] save_img_path   

                # Stack the elements appropriately
                final_aper_mags_r4    = np.vstack([r["aper_mags_r4"] for r in results])
                final_cog_mags        = np.vstack([r["cog_mags"] for r in results])
                final_cog_mags_err    = np.array([r["cog_mags_err"] for r in results])
                
                final_cog_params_g     = np.vstack([r["params_g"] for r in results])
                final_cog_params_r     = np.vstack([r["params_r"] for r in results])
                final_cog_params_z     = np.vstack([r["params_z"] for r in results])
                
                final_cog_params_g_err = np.vstack([r["params_g_err"] for r in results])
                final_cog_params_r_err = np.vstack([r["params_r_err"] for r in results])
                final_cog_params_z_err = np.vstack([r["params_z_err"] for r in results])
                
                final_cog_saveimgs     = np.array([r["img_path"] for r in results], dtype=str)

                all_cog_saveimgs += list(final_cog_saveimgs)

                if chunk_i == 0:
                    print_stage("Example outputs from cog pipeline")
                    print("final aper mags R4 shape : ",final_aper_mags_r4.shape)
                    print("final cog mags shape : ",final_cog_mags.shape)
                    print("final cog params g-band shape : ",final_cog_mags.shape)
                    #print some values for testing
                    print("cog_mags : ", final_cog_mags[0])
                    print("cog_mags_err : ", final_cog_mags_err[0])
                    print("cog params g : ",final_cog_params_g[0])
                    print("cog params g err  : ", final_cog_params_g_err[0])
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

                shreds_focus_i["MAG_G_APERTURE_COG"] = final_cog_mags[:,0]
                shreds_focus_i["MAG_R_APERTURE_COG"] = final_cog_mags[:,1]
                shreds_focus_i["MAG_Z_APERTURE_COG"] = final_cog_mags[:,2]

                shreds_focus_i["MAG_G_APERTURE_R4"] = final_aper_mags_r4[:,0]
                shreds_focus_i["MAG_R_APERTURE_R4"] = final_aper_mags_r4[:,1]
                shreds_focus_i["MAG_Z_APERTURE_R4"] = final_aper_mags_r4[:,2]

                shreds_focus_i["MAG_G_APERTURE_COG_ERR"] = final_cog_mags_err[:,0]
                shreds_focus_i["MAG_R_APERTURE_COG_ERR"] = final_cog_mags_err[:,1]
                shreds_focus_i["MAG_Z_APERTURE_COG_ERR"] = final_cog_mags_err[:,2]

                ##adding the lists to the astropy table!

                # Create an astropy Column with object dtype to hold variable-length array
                col_cog_params_g = Column(final_cog_params_g, name='MAG_G_APERTURE_COG_PARAMS')
                col_cog_params_r = Column(final_cog_params_r, name='MAG_R_APERTURE_COG_PARAMS')
                col_cog_params_z = Column(final_cog_params_z, name='MAG_Z_APERTURE_COG_PARAMS')

                # Add the column to the table
                shreds_focus_i.add_column(col_cog_params_g)
                shreds_focus_i.add_column(col_cog_params_r)
                shreds_focus_i.add_column(col_cog_params_z)

                col_cog_params_err_g = Column(final_cog_params_g_err, name='MAG_G_APERTURE_COG_PARAMS_ERR')
                col_cog_params_err_r = Column(final_cog_params_r_err, name='MAG_R_APERTURE_COG_PARAMS_ERR')
                col_cog_params_err_z = Column(final_cog_params_z_err, name='MAG_Z_APERTURE_COG_PARAMS_ERR')

                # Add the column to the table
                shreds_focus_i.add_column(col_cog_params_err_g)
                shreds_focus_i.add_column(col_cog_params_err_r)
                shreds_focus_i.add_column(col_cog_params_err_z)

                #Get the cog based stellar mass
                shreds_focus_i = compute_aperture_masses(shreds_focus_i, rband_key="MAG_R_APERTURE_COG", gband_key="MAG_G_APERTURE_COG", z_key="Z", output_key="LOGM_SAGA_APERTURE_COG")
                
                if tgids_list is None:
                    save_table( shreds_focus_i, file_save)   
                    print_stage("Saved aperture summary files at %s!"%file_save)
                    
    ##################
    ##PART 5: Once all the chunks are done, combine them all!
    ##################

    yes_save = not no_save
    if tgids_list is None and (run_aper | run_cog) and yes_save:
        print_stage("Consolidating all the saved chunks!")
        
        #files were saved and so we will consolidate them!
        if use_clean == False:
            clean_flag = "shreds"
        else:
            clean_flag = "clean"


        file_template_aper = "/pscratch/sd/v/virajvm/catalog_dr1_dwarfs/iron_photometry/iron_%s_%s_catalog_w_aper_mags"%(sample_str, clean_flag)
 
        if nchunks == 1:
            #we just need to rename the file!
            shutil.copy(file_template_aper + "_chunk_0.fits", file_template_aper + "%s.fits"%end_name)
            if run_cog == True:
                #see the comments below for why this if statement is here
                os.rename(file_template_aper + "_chunk_0.fits", file_template_aper + "%s.fits"%end_name)

        else:
            #we need to consolidate files!  
            shreds_focus_combine_aper = []
            
            for ni in trange(nchunks):
                shreds_focus_part = Table.read(file_template_aper + "_chunk_%d.fits"%ni  )
                shreds_focus_combine_aper.append(shreds_focus_part)
                #and then we delete that file!
                if run_cog == True:
                    #we only delete this file if cog is also run. COG needs this file and so we do not want to remove it if COG not run yet
                    os.remove(file_template_aper + "_chunk_%d.fits"%ni)
                
            shreds_focus_combine_aper = vstack(shreds_focus_combine_aper)

            print_stage("Total number of objects in consolidated aperture file = %d"%len(shreds_focus_combine_aper))

            save_table( shreds_focus_combine_aper, file_template_aper + "%s.fits"%end_name ) 
            
            print_stage("Consolidated aperture chunk saved at %s"%(file_template_aper + "%s.fits"%end_name) )

    ##make a scrollable pdf to view the final results!
    ##only make for some objects?

    if tgids_list is None and run_aper:
        make_scroll_pdf(all_aper_saveimgs, save_sample_path, summary_scroll_file_name, special_plot_mask, max_num=500, type_str="aper",all_aper_saveimgs2=None)

    if tgids_list is None and run_cog:
        make_scroll_pdf(all_cog_saveimgs, save_sample_path, summary_scroll_file_name, special_plot_mask, max_num=500, type_str="cog",all_aper_saveimgs2=None)
        
    if tgids_list is None and run_cog and run_aper:
        make_scroll_pdf(all_aper_saveimgs, save_sample_path, summary_scroll_file_name, special_plot_mask, max_num=500, type_str="apercog",all_aper_saveimgs2=all_cog_saveimgs)

    

        

    
    
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

'''
Script that contains relevant functions for running the dwarf galaxy photometry pipeline!

Basic outline:

Contain functions that download the relevant data needed to run the scarlet or aperture photometry pipelines (the make catalogs funcs)

The functions that do the actual photometry are kept in other .py files for organization

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
from scarlet_photo import run_scarlet_pipe
from aperture_photo import run_aperture_pipe
# from get_sga_distances import get_sga_info
from desi_lowz_funcs import save_subimage, fetch_psf


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
    result.add_argument('-sample', dest='sample', type=str, default = "BGSB") 
    result.add_argument('-file_ending', dest='file_ending', type=str, default = "") 
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




def get_nearby_source_catalog(ra_k, dec_k, wcat, brick_i, save_path_k, source_cat, source_pzs):
    '''
    Function that gets source catalog within 45 arcsec radius of object
    '''

    # getting relevant sources within 45 arcsecs of source and filtering by 5sigma more detections in all 3 bands
    source_cat_f = source_cat[ (np.abs(source_cat["ra"] - ra_k) < 45/3600) & (np.abs(source_cat["dec"] - dec_k) < 45/3600) ]

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
    
    ##compute the errors in the mag assuming they are small :) 
    source_cat_f["mag_g_err"] = 1.087*(np.sqrt(1/source_cat_f["flux_ivar_g"]) / source_cat_f["flux_g"]) 
    source_cat_f["mag_r_err"] = 1.087*(np.sqrt(1/source_cat_f["flux_ivar_r"]) / source_cat_f["flux_r"])
    source_cat_f["mag_z_err"] = 1.087*(np.sqrt(1/source_cat_f["flux_ivar_z"]) / source_cat_f["flux_z"]) 
    
    source_cat_f["g-r_err"] = np.sqrt(  source_cat_f["mag_g_err"]**2 + source_cat_f["mag_r_err"]**2)
    source_cat_f["r-z_err"] = np.sqrt(  source_cat_f["mag_r_err"]**2 + source_cat_f["mag_z_err"]**2)

    #remove if there are any nans in the data!
    source_cat_f = source_cat_f[ ~np.isnan(source_cat_f["g-r_err"]) &  ~np.isnan(source_cat_f["r-z_err"])  ]
    source_cat_f = source_cat_f[ ~np.isnan(source_cat_f["g-r"]) &  ~np.isnan(source_cat_f["r-z"])  ]

    
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
            print("Hmm. The target source was not found. Check for differences in target catalog maybe?")
        source_cat_f = vstack( [source_cat_f,  tgt_source ] )
    else:
        pass    
    
    #save this file
    source_cat_f.write(save_path_k + "/source_cat_f.fits",overwrite=True)

    return



def get_relevant_files_scarlet(input_dict):
    '''
    Get the relevant files for a single object!
    '''

    tgid_k = input_dict["TARGETID"]
    #sample is a string saying BGSB, BGSF, or ELG
    samp_k = input_dict["SAMPLE"]
    ra_k = input_dict["RA"]
    dec_k = input_dict["DEC"]
    redshift_k = input_dict["Z"]
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
    #sample is a string saying BGSB, BGSF, or ELG
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

    check_path_existence(all_paths=[top_path_k])
    #inside this folder, we will save all the relevant files and info!!
    image_path =  top_folder + "_cutouts/image_tgid_%d_ra_%f_dec_%f.fits"%(tgid_k,ra_k,dec_k)
    ## save the source catalog 
    if os.path.exists(top_path_k + "/source_cat_f.fits"):
        pass
    else:
        get_nearby_source_catalog(ra_k, dec_k, wcat, brick_i, top_path_k, source_cat, source_pzs_i)
    ## get the nearby galex source and save info on aperture center
    
    if os.path.exists(image_path):
        pass
    else:
        #we need to obtain the cutout data!
        save_cutouts(ra_k,dec_k,image_path,session,size=350)

        # data = fits.open(image_path)[0].data
        # rgb_stuff = sdss_rgb([data[0],data[1],data[2]], ["g","r","z"], scales=dict(g=(2,6.0), r=(1,3.4), z=(0,2.2)), m=0.03)
        
        # wcs = WCS(fits.getheader( image_path ))
        # get_galex_source(ra_k, dec_k, rgb_stuff, wcs, top_path_k)

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

    from desi_lowz_funcs import get_sweep_filename, save_table, is_target_in_south

    # ##add the sweep, catalog info
    # bgsb_list = add_sweeps_column(bgsb_list, "/pscratch/sd/v/virajvm/catalog_dr1_dwarfs/iron_bgs_bright_filter_zsucc_zrr02_allfracflux.fits")
    # bgsf_list = add_sweeps_column(bgsf_list, "/pscratch/sd/v/virajvm/catalog_dr1_dwarfs/iron_bgs_faint_filter_zsucc_zrr03_allfracflux.fits")
    # elg_list = add_sweeps_column(elg_list, "/pscratch/sd/v/virajvm/catalog_dr1_dwarfs/iron_elg_filter_zsucc_zrr05_allfracflux.fits")
    # lowz_list = add_sweeps_column(lowz_list, "/pscratch/sd/v/virajvm/catalog_dr1_dwarfs/iron_lowz_filter_zsucc_zrr03.fits")

    #identifying the objects that have the 
    # fracflux_grz = [f"FRACFLUX_{b}" for b in "GRZ"]
    # rchisq_grz = [f"RCHISQ_{b}" for b in "GRZ"]

    # remove_queries = [Query(_n_or_more_lt(fracflux_grz, 2, 0.35), _n_or_more_lt(rchisq_grz, 2, 2) )]
    # remove_queries = [Query(_n_or_more_lt(fracflux_grz, 2, 0.35), _n_or_more_lt(rchisq_grz, 2, 2) )]
    #note that the this is n_or_more_LT!! so be careful about that!
    # bgsb_mask = get_remove_flag(bgsb_list, remove_queries) == 0
    # bgsf_mask = get_remove_flag(bgsf_list, remove_queries) == 0
    # elg_mask = get_remove_flag(elg_list, remove_queries) == 0

    # clean_mask_bgsb = (~bgsb_mask)
    # clean_mask_bgsf = (~bgsf_mask)
    # clean_mask_elg = (~elg_mask)

    ## MAYBE HERE, WE CAN PRESELECT FOR GALAXIES THAT SHOW MAIN SF SIGNATURE. What if agn signature is apparent in spectra?
    fracflux_limit = 0.25
    
    clean_mask_bgsb = (bgsb_list["FRACFLUX_R"] < fracflux_limit) & (bgsb_list["FRACFLUX_G"] < fracflux_limit) & (bgsb_list["FRACFLUX_Z"] < fracflux_limit)
    clean_mask_bgsf = (bgsf_list["FRACFLUX_R"] < fracflux_limit) & (bgsf_list["FRACFLUX_G"] < fracflux_limit) & (bgsf_list["FRACFLUX_Z"] < fracflux_limit)
    clean_mask_elg = (elg_list["FRACFLUX_R"] < fracflux_limit) & (elg_list["FRACFLUX_G"] < fracflux_limit) & (elg_list["FRACFLUX_Z"] < fracflux_limit)
    
    ##construct the clean catalog
    ## to make things easy in the cleaning stage, we will use the optical based colors
    
    # dwarf_mask_bgsb = ((bgsb_list["LOGM_CIGALE"] < 9.5)  | (bgsb_list["LOGM_SAGA"] < 9.5) )
    # dwarf_mask_bgsf = ((bgsf_list["LOGM_CIGALE"] < 9.5)  | (bgsf_list["LOGM_SAGA"] < 9.5) )
    # dwarf_mask_lowz = ((lowz_list["LOGM_CIGALE"] < 9.5)  | (lowz_list["LOGM_SAGA"] < 9.5) ) 
    # dwarf_mask_elg = (elg_list["LOGM_CIGALE"] < 10.5)   

    dwarf_mask_bgsb = (bgsb_list["LOGM_SAGA"] < 9.5) 
    dwarf_mask_bgsf = (bgsf_list["LOGM_SAGA"] < 9.5) 
    dwarf_mask_lowz = (lowz_list["LOGM_SAGA"] < 9.5) 
    dwarf_mask_elg = (elg_list["LOGM_SAGA"] < 10)   

    #the log_mstar > 7 cut as I am more confident in them
    bgsb_clean_dwarfs = bgsb_list[ clean_mask_bgsb & (bgsb_list["LOGM_SAGA"] > 7) & (dwarf_mask_bgsb) ]

    bgsf_clean_dwarfs = bgsf_list[ clean_mask_bgsf & (bgsf_list["LOGM_SAGA"] > 7) & (dwarf_mask_bgsf) ]
    
    lowz_clean_dwarfs = lowz_list[ dwarf_mask_lowz ]

    elg_clean_dwarfs = elg_list[ clean_mask_elg & dwarf_mask_elg & (elg_list["LOGM_SAGA"] > 7) ]

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
    # all_clean_dwarfs = get_sga_info(all_clean_dwarfs)
    
    #save the clean dwarfs now!
    save_table(all_clean_dwarfs,"/pscratch/sd/v/virajvm/catalog_dr1_dwarfs/desi_y1_dwarf_clean_catalog_v2.fits",comment="This is a compilation of dwarf galaxy candidates in DESI Y1 data from the BGS Bright, BGS Faint, ELG and LOW-Z samples. Only galaxies with 7 < LogMstar < 9.5 (in LOGM_SAGA or CIGALE, 10.5 for ELGs) and that have robust photometry are included. Galaxies that we have identified as shreds are not included here and their photometry is currently being measured. See redo_photometry.py script for more details on how this made.")
        
    ##applying the mask and then stacking them!!

    bgsb_shreds = bgsb_list[ ~clean_mask_bgsb & dwarf_mask_bgsb ]
    #even including clean dwarfs that are below LogMstar < 7 to confirm their existence!
    bgsb_shreds_p1 = bgsb_list[ clean_mask_bgsb & (bgsb_list["LOGM_SAGA"] <= 7) ]

    bgsf_shreds = bgsf_list[  ~clean_mask_bgsf & (bgsf_list["LOGM_SAGA"] < 9.5) ]
    bgsf_shreds_p1 = bgsf_list[ clean_mask_bgsf & (bgsf_list["LOGM_SAGA"] <= 7)  ]
    
    lowz_shreds = lowz_list[lowz_list["LOGM_SAGA"] < 7]

    elg_shreds = elg_list[ ~clean_mask_elg]
    elg_shreds_p1 = elg_list[ clean_mask_elg & dwarf_mask_elg & (elg_list["LOGM_SAGA"] <= 7) ]
    
    #adding the sample column
    bgsb_shreds["SAMPLE"] = np.array(["BGS_BRIGHT"]*len(bgsb_shreds))
    bgsf_shreds["SAMPLE"] = np.array(["BGS_FAINT"]*len(bgsf_shreds))
    elg_shreds["SAMPLE"] = np.array(["ELG"]*len(elg_shreds))
    lowz_shreds["SAMPLE"] = np.array(["LOWZ"]*len(lowz_shreds))
    
    bgsb_shreds_p1["SAMPLE"] = np.array(["BGS_BRIGHT"]*len(bgsb_shreds_p1))
    bgsf_shreds_p1["SAMPLE"] = np.array(["BGS_FAINT"]*len(bgsf_shreds_p1))
    elg_shreds_p1["SAMPLE"] = np.array(["ELG"]*len(elg_shreds_p1))
    

    #stacking all the shreds now 
    shreds_all = vstack( [bgsb_shreds, bgsf_shreds, elg_shreds, lowz_shreds, bgsb_shreds_p1,bgsf_shreds_p1, elg_shreds_p1 ] )

    shreds_all.remove_column("OII_FLUX")
    shreds_all.remove_column("OII_FLUX_IVAR")
    shreds_all.remove_column("Z_HPX")
    shreds_all.remove_column("TARGETID_2")
    shreds_all.remove_column("OII_SNR")
    shreds_all.remove_column("ZSUCC")
    
    print("Total number of objects whose photometry needs to be redone = ", len(shreds_all))

    ## we add the SGA information to the objects now!
    from desi_lowz_funcs import get_sga_norm_dists

    shreds_ra, shreds_dec, shreds_z = shreds_all["RA"].data, shreds_all["DEC"].data, shreds_all["Z"].data

    matched_sgaids, matched_norm_dists = get_sga_norm_dists(shreds_ra,shreds_dec, shreds_z, 
                                                            Nmax = 100,run_parr = False,ncores = 128,siena_path="/global/cfs/cdirs/cosmo/data/sga/2020/SGA-2020.fits",verbose=True)

    shreds_all["SGA_ID_MATCH"] = matched_sgaids
    shreds_all["SGA_D26_NORM_DIST"] = matched_norm_dists

    save_table(shreds_all,"/pscratch/sd/v/virajvm/catalog_dr1_dwarfs/desi_y1_dwarf_shreds_catalog_v2.fits")
    
    ### PLOTTING THE FRACTION OF GALAXIES THAT ARE LIKELY SHREDS/BAD PHOTOMETRY
    # zgrid = np.arange(0.00,0.3,0.02)

    # def get_shred_frac(tot_cat, sample,zlow, zhi, shred_mask):
    #     shreds_count = len(tot_cat[ (tot_cat["SAMPLE"] == sample) & (tot_cat["Z"] < zhi) & (tot_cat["Z"] > zlow) & shred_mask ] )
    #     tot_count = len(tot_cat[ (tot_cat["SAMPLE"] == sample) & (tot_cat["Z"] < zhi) & (tot_cat["Z"] > zlow) ] )
    
    #     if tot_count > 0:
    #         return shreds_count / tot_count
    #     else:
    #         return None

    # shred_frac_all_bgsb = []
    # shred_frac_all_bgsf = []
    # shred_frac_all_elg = []

    # shred_frac_jf_bgsb = []
    # shred_frac_jf_bgsf = []
    # shred_frac_jf_elg = []

    # bgsb_list["SAMPLE"] = np.array(len(bgsb_list)*["BGS_BRIGHT"])
    # bgsf_list["SAMPLE"] = np.array(len(bgsf_list)*["BGS_FAINT"])
    # elg_list["SAMPLE"] = np.array(len(elg_list)*["ELG"])
    # #also fitlering for dwarf candidates as we only care about those!
    # all_list = vstack( [bgsb_list[dwarf_mask_bgsb], bgsf_list[dwarf_mask_bgsf], elg_list[dwarf_mask_elg] ] )

    # remove_queries_all = [Query(_n_or_more_lt(fracflux_grz, 2, 0.35), _n_or_more_lt(rchisq_grz, 2, 2) )]
    
    # # remove_queries_only_frac = [Query(_n_or_more_lt(fracflux_grz, 2, 0.35))]
    
    # mask_shred_all = get_remove_flag(all_list, remove_queries_all) == 0
    # mask_only_frac = get_remove_flag(all_list, remove_queries_only_frac) == 0

    # # print( len(all_list[ (all_list["SAMPLE"] == "ELG") & mask_only_frac ]) )
    # # print( len(all_list[ (all_list["SAMPLE"] == "ELG") & mask_shred_all ]) )
    
    # for i in trange(len(zgrid)-1):
    #     zlow = zgrid[i]
    #     zhi = zgrid[i+1]

    #     shred_frac_all_bgsb.append(   get_shred_frac(all_list, "BGS_BRIGHT",zlow, zhi, mask_shred_all)  ) 
    #     shred_frac_all_bgsf.append(   get_shred_frac(all_list, "BGS_FAINT",zlow, zhi, mask_shred_all)  ) 
    #     shred_frac_all_elg.append(   get_shred_frac(all_list, "ELG",zlow, zhi, mask_shred_all)  ) 

    #     shred_frac_jf_bgsb.append(   get_shred_frac(all_list, "BGS_BRIGHT",zlow, zhi, mask_only_frac)  ) 
    #     shred_frac_jf_bgsf.append(   get_shred_frac(all_list, "BGS_FAINT",zlow, zhi, mask_only_frac)  ) 
    #     shred_frac_jf_elg.append(   get_shred_frac(all_list, "ELG",zlow, zhi, mask_only_frac)  ) 
    

    # bgs_col = "#648FFF" #DC267F
    # elg_col = "#FFB000"
    
    # zcens = 0.5*(zgrid[1:] + zgrid[:-1])
    
    # plt.figure(figsize = (5,5))
    # plt.plot(zcens, shred_frac_all_bgsb,label = "BGS Bright",lw = 3,color = bgs_col,ls = "-",alpha = 0.75)
    # plt.plot(zcens, shred_frac_jf_bgsb,lw = 3,color = bgs_col,ls = "--")
    
    # plt.plot(zcens, shred_frac_all_bgsf,label = "BGS Faint",lw = 3,color = "r",ls = "-",alpha = 0.75)
    # plt.plot(zcens, shred_frac_jf_bgsf,lw = 3,color = "r",ls = "--")
    
    # plt.plot(zcens, shred_frac_all_elg,label = "ELG",lw = 3,color = elg_col,ls = "-",alpha = 0.75)
    # plt.plot(zcens, shred_frac_jf_elg,lw = 3,color = elg_col,ls = "--")

    # plt.plot( [-10,-1],[-10,-1], color = "k", ls = "-", alpha = 0.75,label = r"FRACFLUX + RCHISQ cut")
    # plt.plot( [-10,-1],[-10,-1], color = "k", ls = "--",label = r"FRACFLUX cut")
    
    # plt.legend(fontsize = 12)
    # plt.xlim([0,0.2])
    # plt.ylim([0,1])
    # plt.xlabel("z (Redshift)",fontsize = 15)
    # plt.ylabel(r"N(candidate shreds | z) / N(z)",fontsize = 15)
    # plt.grid(ls=":",color = "lightgrey",alpha = 0.5)
    # plt.savefig("paper_plots/frac_shreds.pdf", bbox_inches="tight")
    # plt.show()
        
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

    source_cat = Table.read("/global/cfs/cdirs/cosmo/data/legacysurvey/dr9/%s/tractor/%s/tractor-%s.fits"%(wcat, brick_folder,brick_i)  )

    return source_cat



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

    sample_str = args.sample
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
    file_ending = args.file_ending
    save_other_image_path = args.save_img
    
    run_w_source = args.run_w_source
    
    run_aper = args.run_aper
    run_scarlet = args.run_scarlet


    run_own_detect = not run_w_source
    print(run_w_source, run_own_detect)

    ## can I come up with a robust way to choose box size?
    box_size = 350
    
    if save_other_image_path != "":
        #confirm that this folder exists
        check_path_existence(all_paths=[save_other_image_path])
        
    c_light = 299792 #km/s

    ##################
    ##PART 1: Make the clean and shredded catalogs!
    ##################

    # make_clean_shreds_catalogs()

    #make sure the get_sga_distances.py script has been run!!

    ##################
    ##PART 2: Generate nested folder structure with relevant files for doing photometry
    ##################

    # creating a single session!
    session = requests.Session()

    ##load the relevant catalogs!
    if use_clean==False:
        shreds_all = Table.read("/pscratch/sd/v/virajvm/catalog_dr1_dwarfs/desi_y1_dwarf_shreds_catalog_v2.fits")
    else:
        shreds_all = Table.read("/pscratch/sd/v/virajvm/catalog_dr1_dwarfs/desi_y1_dwarf_clean_catalog_v2.fits")
        #to get the final clean part we remove all objects above LogMstar > 7 
        
    shreds_focus = shreds_all[(shreds_all["SAMPLE"] ==  sample_str)]

    # shreds_focus = Table.read("/pscratch/sd/v/virajvm/download_south_temp.fits")

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
    
        print("Number of objects whose photometry will be redone = ", len(shreds_focus))
    
        if use_clean == False:
            top_folder = "/pscratch/sd/v/virajvm/redo_photometry_plots/all_deshreds"
        else:
            top_folder = "/pscratch/sd/v/virajvm/redo_photometry_plots/all_good"
            
        print(len(shreds_focus[shreds_focus["is_south"] == 1]))
        print(len(shreds_focus[shreds_focus["is_south"] == 0]))

        for wcat_ind, wcat in enumerate(["north","south"]):
            check_path_existence(all_paths=[top_folder + "/%s"%wcat])
    
            unique_sweeps = np.unique( shreds_focus[shreds_focus["is_south"] == wcat_ind]["SWEEP"])
            print(len(unique_sweeps), "unique sweeps found in %s"%wcat)

            #this is just to make sure the very end sweeps have their files made as well
            unique_sweeps = unique_sweeps[::-1]
    
            #counter for number of sweeps done
            sweeps_done=0
            
            for sweep_i in unique_sweeps:
                
                print("Sweeps done = %d/%d"%(sweeps_done, len(unique_sweeps)))
                sweeps_done += 1
                #make a sweep folder if it does not exist
                sweep_folder = sweep_i.replace("-pz.fits","")
                check_path_existence(all_paths=[top_folder + "/%s/"%wcat + sweep_folder ])
                
                ##load the relevant sweep photo-z file
                source_pzs_i = Table.read( "/global/cfs/cdirs/cosmo/data/legacysurvey/dr9/%s/sweep/9.1-photo-z/"%wcat + sweep_i)
        
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
                with mp.Pool(processes=ncores) as pool:
                    results = list(tqdm(pool.imap(get_relevant_files_aper, all_input_dicts), total = len(all_input_dicts)  ))

                # if serial
                # for i in trange(len(all_input_dicts)):
                #     get_relevant_files_single(all_input_dicts[i])


    ##################
    ##PART 3: Preparing inputs for the aperture and/or scarlet photometry functions
    ##################

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

        wcat_k = all_wcats[int(shreds_focus["is_south"][k])]

        sweep_folder = sweep_k.replace("-pz.fits","")

        if use_clean == False:
            
            save_path_k = top_folder + "/%s/"%wcat_k + sweep_folder + "/" + brick_k + "/%s_tgid_%d"%(sample_str, tgid_k)
            
            img_path_k = "/pscratch/sd/v/virajvm/redo_photometry_plots/all_deshreds_cutouts/image_tgid_%d_ra_%f_dec_%f.fits"%(tgid_k,ra_k,dec_k) 
        else:
    
            save_path_k = top_folder + "/%s/"%wcat_k + sweep_folder + "/" + brick_k + "/%s_tgid_%d"%(sample_str, tgid_k)
                
            img_path_k = "/pscratch/sd/v/virajvm/redo_photometry_plots/all_good_cutouts/image_tgid_%d_ra_%f_dec_%f.fits"%(tgid_k,ra_k,dec_k) 
        
        if os.path.exists(save_path_k + "/source_cat_f.fits"):
            source_cat_f = Table.read(save_path_k + "/source_cat_f.fits")
        else:
            #in case, we have arrived at this point and stil some files are not made, we make them!
            source_pzs_i = Table.read( "/global/cfs/cdirs/cosmo/data/legacysurvey/dr9/%s/sweep/9.1-photo-z/"%wcat_k + sweep_k)
            source_cat_i = read_source_cat(wcat, brick_i)
            
            get_nearby_source_catalog(ra_k, dec_k, wcat_k, brick_k, save_path_k, source_cat_i, source_pzs_i)

        if os.path.exists(img_path_k):
            img_data = fits.open(img_path_k)
            data_arr = img_data[0].data
            wcs = WCS(fits.getheader( img_path_k ))
        else:
            #in case image is not downloaded, we download it!
            save_cutouts(ra_k,dec_k,img_path_k,session,size=350)
            img_data = fits.open(img_path_k)
            data_arr = img_data[0].data
            wcs = WCS(fits.getheader( img_path_k ))

        if np.shape(data_arr[0])[0] != 350:
            raise ValueError("Issue with image size here=%s"%img_path_k)
    
            # import shutil
            # shutil.copy(img_path_k, save_path_k + "/")

        temp_dict = {"tgid":tgid_k, "ra":ra_k, "dec":dec_k, "redshift":redshift_k, "save_path":save_path_k, "img_path":img_path_k, "save_sample_path" : save_sample_path, "wcs": wcs , "image_data": data_arr, 
                     "source_cat": source_cat_f, "index":k , "org_mag_g": shreds_focus["MAG_G"][k], "overwrite": overwrite_bool, "save_other_image_path":save_other_image_path, "run_own_detect":run_own_detect, "box_size" : box_size, 
                     "session":session }

        return temp_dict
        
        # else:
        #     return

    print("Number of cores used is =",ncores)
    
    all_ks = np.arange(len(shreds_focus))

    print(len(all_ks))
    #run with multi-pool
    if run_parr:
        with mp.Pool(processes = ncores ) as pool:
            all_inputs = list(tqdm(pool.imap(produce_input_dicts, all_ks), total = len(all_ks)  ))
    else:
        all_inputs = []
        for i in trange(len(all_ks)):
            all_inputs.append( produce_input_dicts(all_ks[i]) )

    all_inputs = np.array(all_inputs)
    # all_inputs = all_inputs[all_inputs != None]
    print(len(all_inputs))

    ##This is the length of the list that contains info on all the objects on which aperture photometry will be run!

    ##################
    ##PART 4a: Run aperture photometry
    ##First step before scarlet photometry to weed out massive galaxies
    ##################
    
    if run_aper == True:

        shreds_focus["MAG_G_APER"] = np.ones(len(shreds_focus)) * -99
        shreds_focus["MAG_R_APER"] = np.ones(len(shreds_focus)) * -99
        shreds_focus["MAG_Z_APER"] = np.ones(len(shreds_focus)) * -99
        shreds_focus["NEAREST_STAR_DIST"] = np.ones(len(shreds_focus)) * -99

        if run_parr:
            with mp.Pool(processes= ncores ) as pool:
                results = list(tqdm(pool.imap(run_aperture_pipe, all_inputs), total = len(all_inputs)  ))
        else:
            results = []
            for i in trange(len(all_inputs)):
                results.append( run_aperture_pipe(all_inputs[i]) )

        
        ### saving the results of the photometry pipeline
        results = np.array(results)
    
        print_stage("Done running all the aperture photometry!!")
    
        final_index = results[:,0].astype(int)
        final_star_dists = results[:,1].astype(float)
        final_new_mags = np.vstack(results[:,2])
        final_org_mags = np.vstack(results[:,3])

        final_save_paths = results[:,4].astype(str)
        
        shreds_focus_f = shreds_focus[final_index]
    
        #check that the org mags make sense
        print("Maximum Abs difference between org mags = ",np.max( np.abs(final_org_mags[:,0] - shreds_focus_f["MAG_G"]) ))
        
        shreds_focus_f["NEAREST_STAR_DIST"] = final_star_dists
        shreds_focus_f["MAG_G_APER"] = final_new_mags[:,0]
        shreds_focus_f["MAG_R_APER"] = final_new_mags[:,1]
        shreds_focus_f["MAG_Z_APER"] = final_new_mags[:,2]
        shreds_focus_f["SAVE_PATH"] = final_save_paths 

        print("Compute aperture-photometry based stellar masses now!")
            
        #compute the aperture photometry based stellar masses!
        rmag_aper = shreds_focus_f["MAG_R_APER"].data
        gr_aper = shreds_focus_f["MAG_G_APER"].data - shreds_focus_f["MAG_R_APER"].data

        #are any of the aperture magnitudes nans?
        nan_mask = (np.isnan(shreds_focus_f["MAG_G_APER"]) | np.isnan(shreds_focus_f["MAG_R_APER"]) )
        #if any of the bands are nan, we automatically include it in scarlet method to try to do a better job at estimating photometry
        
        from desi_lowz_funcs import get_stellar_mass

        all_mstar_aper = np.ones(len(shreds_focus_f))*np.nan

        mstar_aper_nonan = get_stellar_mass( gr_aper[~nan_mask],rmag_aper[~nan_mask], shreds_focus["Z"][~nan_mask]  )

        all_mstar_aper[~nan_mask] = mstar_aper_nonan
        
        shreds_focus_f["LOGM_SAGA_APERTURE"] = all_mstar_aper 
    
        #then save this file!
        if tgids_list is None and (run_aper | run_scarlet):
            print_stage("Saving summary files!")
            if use_clean == False:
                save_table( shreds_focus_f,"/pscratch/sd/v/virajvm/catalog_dr1_dwarfs/iron_photometry/iron_%s_shreds_catalog_w_new_mags%s.fits"%(sample_str,file_ending) )
            else:
                save_table( shreds_focus_f,"/pscratch/sd/v/virajvm/catalog_dr1_dwarfs/iron_photometry/iron_%s_clean_catalog_w_new_mags%s.fits"%(sample_str, file_ending) )


    ##################
    ##PART 4b: Run scarlet photometry for sources that remain as candidate dwarfs after aperture photometry
    ##################

    # check how scarlet does for more massive galaxies?
    if run_scarlet == True:

        ##load the relevant file from aperture photometry
        ##re-estimate the stellar mass and make a cut on that!
        #also be aware of the -99s, if background cannot be estimated, that likely means it is not a dwarf galaxy!!        
        # if use_clean == False:
        #     shreds_focus = Table.read("/pscratch/sd/v/virajvm/catalog_dr1_dwarfs/iron_photometry/iron_%s_shreds_catalog_w_new_mags%s.fits"%(sample_str,file_ending))
        # else:
        #     shreds_focus = Table.read("/pscratch/sd/v/virajvm/catalog_dr1_dwarfs/iron_photometry/iron_%s_clean_catalog_subset_w_new_mags%s.fits"%(sample_str, file_ending) )

        #     shreds_focus_f_p1 = shreds_focus[~nan_mask][mstar_aper_nonan <= 9.5)]
        # shreds_focus_f_p2 = shreds_focus[nan_mask]

        # shreds_focus_f = vstack([shreds_focus_f_p1, shreds_focus_f_p2])

            
        # #compute the new stellar mass using the update stellar mass!
        
        shreds_focus["MAG_G_SCARLET"] = np.ones(len(shreds_focus)) * -99
        shreds_focus["MAG_R_SCARLET"] = np.ones(len(shreds_focus)) * -99
        shreds_focus["MAG_Z_SCARLET"] = np.ones(len(shreds_focus)) * -99
        shreds_focus["NEAREST_STAR_DIST"] = np.ones(len(shreds_focus)) * -99


        ##generate the relevant files for scarlet photometry
        if make_cats== True:
            print_stage("Generating relevant files for doing scarlet photometry")
            print("Using cleaned catalogs =",use_clean==True)
    
            print("Number of objects whose scarlet files will be obtained = ", len(shreds_focus))
    
            if use_clean == False:
                top_folder = "/pscratch/sd/v/virajvm/redo_photometry_plots/all_deshreds"
            else:
                top_folder = "/pscratch/sd/v/virajvm/redo_photometry_plots/all_good"
                
            print(len(shreds_focus[shreds_focus["is_south"] == 1]))
            print(len(shreds_focus[shreds_focus["is_south"] == 0]))


            ##this can be sped up as we are not reading files or anyting!
            for wcat_ind, wcat in enumerate(["north","south"]):
                check_path_existence(all_paths=[top_folder + "/%s"%wcat])
        
                unique_sweeps = np.unique( shreds_focus[shreds_focus["is_south"] == wcat_ind]["SWEEP"])
                print(len(unique_sweeps), "unique sweeps found in %s"%wcat)
        
                #counter for number of sweeps done
                sweeps_done=0
                
                for sweep_i in unique_sweeps:
                    
                    print("Sweeps done = %d/%d"%(sweeps_done, len(unique_sweeps)))
                    sweeps_done += 1
                    #make a sweep folder if it does not exist
                    sweep_folder = sweep_i.replace("-pz.fits","")
                                
                    #get all the galaxies that fall in this sweep
                    shreds_focus_i = shreds_focus[(shreds_focus["SWEEP"] == sweep_i) & (shreds_focus["is_south"] == wcat_ind)]
            
                    #get all the unique brick names for this source
                    brick_names_i = np.unique( shreds_focus_i["BRICKNAME"] )
    
                    ## for a given sweep can we parallelize this??
                    print("Total number of bricks in this sweep = ", len(brick_names_i) )
                    print("Total number of galaxies in this sweep = ", len(shreds_focus_i) )
    
                    ##prepare all inputs now!
        
                    #looping over all the bricks and saving the source cat files!
                    for brick_i in tqdm(brick_names_i):
                        #make a folder for each brick
                        check_path_existence(all_paths=[top_folder + "/%s/"%wcat + sweep_folder  + "/" + brick_i])
        
                        #inside this brick folder, we will have folders for each galaxy then
        
                        shreds_focus_ij = shreds_focus_i[ shreds_focus_i["BRICKNAME"] == brick_i]
        
                        #read the source catalog for this brick
                        if "p" in brick_i:
                            brick_folder = brick_i[:brick_i.find("p")-1]
                        else:
                            brick_folder = brick_i[:brick_i.find("m")-1]
        
    
                    #list of dictionaries
                    all_input_dicts = []
    
                    ##looping over all the galaxies in this sweep file to get get the input dict file
                    for i in range(len(shreds_focus_i)):
                        temp = { "TARGETID": shreds_focus_i["TARGETID"][i], "SAMPLE": shreds_focus_i["SAMPLE"][i], "RA": shreds_focus_i["RA"][i], 
                                "DEC": shreds_focus_i["DEC"][i], "Z": shreds_focus_i["Z"][i],
                                "wcat":wcat, "brick_i":shreds_focus_i["BRICKNAME"][i], "sweep_folder":sweep_folder, "top_folder":top_folder,
                               "session":session }
    
                        all_input_dicts.append(temp)
    
                    ## get the relevant files for the the 
                    with mp.Pool(processes=ncores) as pool:
                        results = list(tqdm(pool.imap(get_relevant_files_scarlet, all_input_dicts), total = len(all_input_dicts)  ))
                
        if run_parr:
            with mp.Pool(processes= ncores ) as pool:
                results = list(tqdm(pool.imap(run_scarlet_pipe, all_inputs), total = len(all_inputs)  ))
        else:
            results = []
            for i in trange(len(all_inputs)):
                results.append( run_scarlet_pipe(all_inputs[i]) )
        ### saving the results of the photometry pipeline
        results = np.array(results)

        print(results)
    
        print_stage("Done running all the scarlet photometry!!")
    
        final_index = results[:,0].astype(int)
        final_star_dists = results[:,1].astype(float)
        final_new_mags = np.vstack(results[:,2])
        final_org_mags = np.vstack(results[:,3])

        final_save_paths = results[:,4].astype(str)
        
        shreds_focus_f = shreds_focus[final_index]
    
        #check that the org mags make sense
        print("Maximum Abs difference between org mags = ",np.max( np.abs(final_org_mags[:,0] - shreds_focus_f["MAG_G"]) ))
        
        shreds_focus_f["NEAREST_STAR_DIST"] = final_star_dists
        shreds_focus_f["MAG_G_SCARLET"] = final_new_mags[:,0]
        shreds_focus_f["MAG_R_SCARLET"] = final_new_mags[:,1]
        shreds_focus_f["MAG_Z_SCARLET"] = final_new_mags[:,2]
        shreds_focus_f["SAVE_PATH"] = final_save_paths 

   
    #then save this file!
    # if tgids_list is None and (run_aper | run_scarlet):
    #     print_stage("Saving summary files!")
    #     if use_clean == False:
    #         save_table( shreds_focus_f,"/pscratch/sd/v/virajvm/catalog_dr1_dwarfs/iron_shreds_mags/iron_%s_shreds_catalog_w_new_mags%s.fits"%(sample_str,file_ending) )
    #     else:
    #         save_table( shreds_focus_f,"/pscratch/sd/v/virajvm/catalog_dr1_dwarfs/iron_clean_mags/iron_%s_clean_catalog_subset_w_new_mags%s.fits"%(sample_str, file_ending) )


    ##################
    ##PART 5: Using the aperture photometry footprint, find the desi sources that lie around on it and similar redshift as well??
    ##################
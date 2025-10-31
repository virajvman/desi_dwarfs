'''
shifterimg pull docker:legacysurvey/legacypipe:DR10.3.4
shifter --image docker:legacysurvey/legacypipe:DR10.3.4 python3 desi_dwarfs/code/tractor_model.py

In this script, we will loop over all the shred galaxies and save their model image, and residual image, and residual image!
'''

# from desi_lowz_funcs import fetch_psf
# import requests
import numpy as np
from astropy.io import fits
from tractor.tractortime import TAITime
from astrometry.util.util import Tan
import warnings
from legacypipe.survey import LegacySurveyWcs, ConstantFitsWcs
from astropy.wcs import WCS
from legacypipe.survey import wcs_for_brick, BrickDuck
from astrometry.util.fits import fits_table
import matplotlib.pyplot as plt
import glob
from astropy.table import Table, vstack
import os
from scipy.ndimage import gaussian_filter
import astropy.units as u
from astropy.coordinates import SkyCoord
import multiprocessing as mp
from astropy.wcs import WCS
import time
import sys
import argparse

def simple_progress_bar(iteration, total, length=30):
    if total == 0:
        total = 1
    percent = iteration / total
    bar = '=' * int(length * percent) + '-' * (length - int(length * percent))
    sys.stdout.write(f'\r|{bar}| {percent:.3%}')
    sys.stdout.flush()
    

def compute_separations(ra_ref = None, dec_ref = None, ra_all = None, dec_all = None):
    '''
    Function that computes the separation in arcsecs between reference ra,dec and a list/array of ra,decs
    '''

    ref_pos = SkyCoord(ra= ra_ref* u.degree, dec= dec_ref*u.degree )
    all_pos = SkyCoord(ra= ra_all*u.degree, dec=dec_all*u.degree )
    seps = ref_pos.separation(all_pos).arcsec
    return seps

def sdss_rgb(imgs, bands, scales=None,m = 0.02):
    rgbscales = {'u': (2,1.5), #1.0,
                 'g': (2,2.5),
                 'r': (1,1.5),
                 'i': (0,1.0),
                 'z': (0,0.4), #0.3
                 }
    
    if scales is not None:
        rgbscales.update(scales)

    I = 0
    for img,band in zip(imgs, bands):
        plane,scale = rgbscales[band]
        img = np.maximum(0, img * scale + m)
        I = I + img
    I /= len(bands)
        
    Q = 20
    fI = np.arcsinh(Q * I) / np.sqrt(Q)
    I += (I == 0.) * 1e-6
    H,W = I.shape
    rgb = np.zeros((H,W,3), np.float32)
    for img,band in zip(imgs, bands):
        plane,scale = rgbscales[band]
        rgb[:,:,plane] = (img * scale + m) * fI / I
        
    rgb = np.clip(rgb, 0, 1)
    
    return rgb


def custom_brickname(ra, dec):
    brickname = '{:06d}{}{:05d}'.format(
        int(1000*ra), 'm' if dec < 0 else 'p',
        int(1000*np.abs(dec)))
    return brickname


def srcs2image(cat, wcs, band='r', allbands='grz', pixelized_psf=None, psf_sigma=1.0):
    """Build a model image from a Tractor catalog or a list of sources.

    issrcs - if True, then cat is already a list of sources.

    """
    import tractor, legacypipe, astrometry
    from legacypipe.catalog import read_fits_catalog

    if type(wcs) is tractor.wcs.ConstantFitsWcs or type(wcs) is legacypipe.survey.LegacySurveyWcs:
        shape = wcs.wcs.shape
    else:
        shape = wcs.shape
        
    model = np.zeros(shape)
    invvar = np.ones(shape)

    if pixelized_psf is None:
        vv = psf_sigma**2
        psf = tractor.GaussianMixturePSF(1.0, 0., 0., vv, vv, 0.0)
    else:
        psf = pixelized_psf

    tim = tractor.Image(model, invvar=invvar, wcs=wcs, psf=psf,
                        photocal=tractor.basics.LinearPhotoCal(1.0, band=band.lower()),
                        sky=tractor.sky.ConstantSky(0.0),
                        name='model-{}'.format(band))

    # Do we have a tractor catalog or a list of sources?
    if type(cat) is astrometry.util.fits.tabledata:
        srcs = legacypipe.catalog.read_fits_catalog(cat, bands=[band.lower()])
    else:
        srcs = cat

    try:
        tr = tractor.Tractor([tim], srcs)
        mod = tr.getModelImage(0)
    except:
        ##this error can happen if one of the psfsizes is zero if there is missing data!!
        print("SOME ERROR HAS HAPPENED!!")
        print(cat.get("psfsize_g"), cat.get("psfsize_r"), cat.get("psfsize_z") )
        print("--"*10)
        print(psf)
        print("--"*10)
        print(srcs)
        raise

    return mod


def load_tractor(file_path):
    """Load tractor catalog and remove duplicates based on (RA, Dec).    
    """
    
    tractor_file = file_path+"/source_cat_f_more.fits"
    
    if os.path.exists(tractor_file):
        pass
    else:
        tractor_file = file_path + "/source_cat_f.fits"
    
    cols = ['ra', 'dec', 'bx', 'by', 'type', 'ref_cat', 'ref_id',
           'sersic', 'shape_r', 'shape_e1', 'shape_e2',
           'flux_g', 'flux_r', 'flux_z',
           'flux_ivar_g', 'flux_ivar_r', 'flux_ivar_z',
           'nobs_g', 'nobs_r', 'nobs_z',
           'psfdepth_g', 'psfdepth_r', 'psfdepth_z',
           'psfsize_g', 'psfsize_r', 'psfsize_z', "BRICKID","OBJID"]

    tractor = fits_table(tractor_file, columns=cols)

    coords = np.array(list(zip(tractor.get('ra'), tractor.get('dec'))))
    _, unique_indices = np.unique(coords, axis=0, return_index=True)
        
    return tractor[unique_indices]


def make_custom_wcs(ra, dec, width, pixscale):
    brickname = f'custom-{custom_brickname(ra, dec)}'
    brick = BrickDuck(ra, dec, brickname)
    targetwcs = wcs_for_brick(brick, W=float(width), H=float(width), pixscale=pixscale)
    return ConstantFitsWcs(targetwcs)


# def compute_psf_sigma(tractor_subset, band, average_psfsize, mean=True):
#     # if len(tractor_subset.get(f"psfsize_{band}")) == 1 and tractor_subset.get(f"psfsize_{band}") == 0:
#     if tractor_subset.get(f"psfsize_{band}") == 0: 
#         #if we are only reading a single source and it has no psfsize, we take the average value that inputted to function
#         fwhm = average_psfsize
#     else:
#         if mean:
#             fwhm = np.mean(tractor_subset.get(f"psfsize_{band}"))
#         else:
#             # fwhm = tractor_subset.get(f"psfsize_{band}")[0]
#             fwhm = tractor_subset.get(f"psfsize_{band}")
            
#     return (fwhm / 2.3548) / 0.262

def compute_psf_sigma(tractor_subset, band, average_psfsize, mean=True):
    psfsize = tractor_subset.get(f"psfsize_{band}")
    
    if psfsize is None:
        raise ValueError(f"psfsize_{band} not found in tractor_subset.")
    
    if np.isscalar(psfsize):
        if psfsize == 0:
            fwhm = average_psfsize
        else:
            fwhm = psfsize
    else:
        psfsize = np.asarray(psfsize)
        if np.all(psfsize == 0):
            fwhm = average_psfsize
        else:
            fwhm = np.mean(psfsize) if mean else psfsize[0]

    return (fwhm / 2.3548) / 0.262  # convert FWHM (arcsec) to sigma (pixels)



def build_model_image(tractor_subset, wcs, average_psfsize, mean_psf=False):
    return np.array([
        srcs2image(tractor_subset, wcs, band=band, allbands='grz', pixelized_psf=None, psf_sigma=compute_psf_sigma(tractor_subset, band, average_psfsize[band], mean=mean_psf))
        for band in "grz"
    ])
    
def save_rgb_tripanel(mod, data_arr, file_path, tgid=None, width=None, testing=False,use_center_only=False):
    
    rgb_model = sdss_rgb(mod, ["g","r","z"], scales=dict(g=(2,6.0), r=(1,3.4), z=(0,2.2)), m=0.03)
    rgb_data = sdss_rgb(data_arr, ["g","r","z"], scales=dict(g=(2,6.0), r=(1,3.4), z=(0,2.2)), m=0.03)
    resis = data_arr - mod
    rgb_resis = sdss_rgb(resis, ["g","r","z"], scales=dict(g=(2,6.0), r=(1,3.4), z=(0,2.2)), m=0.03)

    start, size = (width - 64) // 2, 64
    slice_ = slice(start, start + size)

    fig, ax = plt.subplots(1, 3, figsize=(12, 4))

    ax[0].set_title("MODEL",fontsize = 13)
    ax[1].set_title("IMG",fontsize = 13)
    ax[2].set_title("IMG - MODEL",fontsize =13)
    
    if use_center_only:
        ax[0].imshow(rgb_model[slice_, slice_, :], origin="lower")
        ax[1].imshow(rgb_data[slice_, slice_, :], origin="lower")
        ax[2].imshow(rgb_resis[slice_, slice_, :], origin="lower")
    else:
        ax[0].imshow(rgb_model, origin="lower")
        ax[1].imshow(rgb_data, origin="lower")
        ax[2].imshow(rgb_resis, origin="lower")

    plt.savefig(f"{file_path}/tractor_model_image.png")
    
    if testing or (tgid is not None and int(tgid) < 500):
        plt.savefig(f"/pscratch/sd/v/virajvm/temp_tractor_models/tractor_model_{tgid}.png")
    plt.close()


def save_rgb_single_panel(data_arr, file_path, image_name=None):
    
    rgb_data = sdss_rgb(data_arr, ["g","r","z"], scales=dict(g=(2,6.0), r=(1,3.4), z=(0,2.2)), m=0.03)

    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    ax.set_title("MODEL",fontsize = 13)
    ax.imshow(rgb_data, origin="lower")
    plt.savefig(f"{file_path}/{image_name}.png")
    plt.close()

    return


def get_img_source(i, ra, dec, tgid, zred, file_path, img_path, width, pixscale=0.262, testing=True):
    '''
    Getting the model of the targeted DESI source
    '''
    
    if os.path.exists(f"{file_path}/tractor_source_model.npy"):
        return
    else:
        tractor = load_tractor(file_path)
        
        seps = compute_separations(ra, dec, tractor.get("ra"), tractor.get("dec"))
    
        #this is not a reliable way to get the source because there can be floating point errors? So maybe 0.1 arcsecs
        tractor_source = tractor[np.argmin(seps)]


        img_data = fits.open(img_path)[0].data
        
        if np.min(seps) > 1:
            print(f"FYI, this object has rather large separation of {np.min(seps)} arcsec. Probably due to the LOWZ target catalog issue. RA={ra}, DEC={dec}, PATH={file_path}")


        if ~np.isfinite(tractor_source.get("shape_r")):
            print(f"The source size {tgid} was not finite! And we will not save the source. It will just be an empty array ")

            mod = np.zeros_like(img_data)
            
            np.save(f"{file_path}/tractor_source_model.npy",  mod) 

            # save_rgb_tripanel(mod, img_data, file_path, tgid, testing,use_center_only=False)
            
        else:
        
            # if len(tractor_source) != 1:
            #     print(f"Ambiguity for index {i} and coords ({ra}, {dec})")
            #     raise ValueError("Tractor source match not unique.")
        
            #if this source has a missing psfsize then we assume an average psfsize from the catalog
            ave_psfsize_dict = { "g": np.mean(tractor.get("psfsize_g")), "r": np.mean(tractor.get("psfsize_r")),  "z": np.mean(tractor.get("psfsize_z"))    }
        
            wcs = make_custom_wcs(ra, dec, width, pixscale)
            mod = build_model_image(tractor_source, wcs, ave_psfsize_dict)
            np.save(f"{file_path}/tractor_source_model.npy", mod)
        
            img_data = fits.open(img_path)[0].data

            # save_rgb_tripanel(mod, img_data, file_path, tgid, testing,use_center_only=False)
            
    return


def get_bkg_sources(i, ra, dec, tgid, zred, file_path, img_path, width, pixscale=0.262, testing=True):
    '''
    Getting the tractor model of the background
    '''
    # if os.path.exists(f"{file_path}/tractor_background_model.npy"):
    #     return

    tractor = load_tractor(file_path)
    
    #we want to also remove sources that do not have finite shape_r
    finite_shaper_mask = np.isfinite(tractor.get("shape_r"))
    tractor = tractor[finite_shaper_mask]

    if os.path.exists(f"{file_path}/main_segment_map.npy"):
        
        segm = np.load(f"{file_path}/main_segment_map.npy")
    
        if np.max(segm) == 0:
            print(f"Main segment map does not exist: {tgid}")
            return
            
        else: 
            img_data = fits.open(img_path)[0].data
            wcs_cutout = WCS(fits.getheader(img_path))
        
            xpix, ypix, _ = wcs_cutout.all_world2pix(tractor.get("ra"), tractor.get("dec"), 0, 1)
        
            #we want to remove sources that are at the edge of the box
            on_edge_mask = (ypix.astype(int) == width) |  (xpix.astype(int) == width)
        
            tractor = tractor[~on_edge_mask]
            xpix = xpix[~on_edge_mask]
            ypix = ypix[~on_edge_mask]
            
            on_main = ~np.isnan(segm[ypix.astype(int), xpix.astype(int)])
            bkg_sources = tractor[~on_main]
        
            wcs = make_custom_wcs(ra, dec, width, pixscale)
        
            #if this source has a missing psfsize then we assume an average psfsize from the catalog
            ave_psfsize_dict = { "g": np.mean(tractor.get("psfsize_g")), "r": np.mean(tractor.get("psfsize_r")),  "z": np.mean(tractor.get("psfsize_z"))    }
            
            mod = build_model_image(bkg_sources, wcs, ave_psfsize_dict, mean_psf=True)
            
            np.save(f"{file_path}/tractor_background_model.npy", mod)
        
            
            return

    else:
        return


def get_blended_remove_sources(i, ra, dec, tgid, zred, file_path, img_path, width,pixscale=0.262, testing=True):
    '''
    Getting the model of the sources that lie on the main segment (blended sources) but that are deemed to be not be part of the parent galaxy
    '''

    # if os.path.exists(file_path + "/tractor_blend_remove_model.npy"):
    #     return

    tractor = load_tractor(file_path)

    #we want to also remove sources that do not have finite shape_r
    finite_shaper_mask = np.isfinite(tractor.get("shape_r"))
    tractor = tractor[finite_shaper_mask]
    

    #load the source catalog that we are removing
    if os.path.exists(file_path + "/blended_source_remove_cat.fits"):
        
        blend_remove_cat = Table.read(file_path + "/blended_source_remove_cat.fits")
    
        if len(blend_remove_cat) == 0:
            #there were no sources to subtract and so we can save an empty array!
            np.save(f"{file_path}/tractor_blend_remove_model.npy", np.zeros((3, width, width))  )
        else: 
            #make sure these columns do not have any units!
            br_ras = blend_remove_cat["ra"].data
            br_decs = blend_remove_cat["dec"].data
        
            #we need to get all the pixel locations of these sources given the wcs and see which ones lie on the main segment
            #sources not on the main segment will be on a zero!
            ra_all = tractor.get("ra")
            dec_all = tractor.get("dec")
    
            #find all the sources that match the blend_remove_cat objects!
            c = SkyCoord(ra= br_ras* u.degree, dec= br_decs*u.degree )
            catalog = SkyCoord(ra=ra_all*u.degree, dec=dec_all*u.degree )
            idx, d2d, d3d = c.match_to_catalog_sky(catalog)
    
        
            #get the indices of objects in the ra_all catalog that match and have zero separation!
            blend_remove_inds = idx[d2d.arcsec == 0]
        
            tractor_blend_re = tractor[blend_remove_inds]
    
            #if this source has a missing psfsize then we assume an average psfsize from the catalog
            ave_psfsize_dict = { "g": np.mean(tractor.get("psfsize_g")), "r": np.mean(tractor.get("psfsize_r")),  "z": np.mean(tractor.get("psfsize_z"))    }
            
            if len(tractor_blend_re) != 0:    
                wcs = make_custom_wcs(ra, dec, width, pixscale)
    
                
                mod = build_model_image(tractor_blend_re, wcs, ave_psfsize_dict, mean_psf=True)
                
                np.save(f"{file_path}/tractor_blend_remove_model.npy", mod)
            
                img_data = fits.open(img_path)[0].data
            else:
                #there were no sources to subtract and so we can save an empty array!
                np.save(f"{file_path}/tractor_blend_remove_model.npy", np.zeros((3, width, width))  )


        return
        
    else:
        return
        
    


# def save_main_tractor_models(tractor_parent, parent_source_cat, wcs, ave_psfsize_dict,
#                         tractor_save_dir, file_path, tgid, width):
#     """
#     Saves individual or HDF5-packed tractor source models depending on source count.
#     """
#     if len(tractor_parent) == 0:
#         print(f"The parent source catalog existed for {tgid}, but no sources remain after parent mask.")
#         return

#     max_ra_diff = np.max(np.abs(tractor_parent.get("ra") - parent_source_cat["ra"]))
#     if max_ra_diff > 0:
#         raise ValueError(f"Inconsistent RA values in get_main_blob_sources : {max_ra_diff}")
    
#     total_model = np.zeros((3, width, width))
#     use_hdf5 = False #len(tractor_parent) > 100

#     if use_hdf5:
#         h5_path = f"{tractor_save_dir}/tractor_parent_source_models.h5"
#         print(f"FYI: Writing {len(tractor_parent)} models to combined HDF5 file: {h5_path}")
#         with h5py.File(h5_path, "w") as f:
#             for k in range(len(tractor_parent)):
#                 objid = int(parent_source_cat["source_objid_new"].data[k])
#                 mod = build_model_image(tractor_parent[k], wcs, ave_psfsize_dict, mean_psf=True)
#                 total_model += mod
#                 f.create_dataset(str(objid), data=mod, compression="gzip", chunks=True)
#     else:
#         for k in range(len(tractor_parent)):
#             objid = int(parent_source_cat["source_objid_new"].data[k])
#             mod = build_model_image(tractor_parent[k], wcs, ave_psfsize_dict, mean_psf=True)
#             total_model += mod
#             np.save(f"{tractor_save_dir}/tractor_parent_source_model_{objid}.npy", mod)

#     # Save the combined model and visualization
#     np.save(f"{file_path}/tractor_main_segment_model.npy", total_model)
#     save_rgb_single_panel(total_model, file_path, image_name="tractor_main_segment_galaxy_model")

#     return


def get_matched_subset(cat, tractor, tgid):
    '''
    Helper function for get_main_blob_sources
    '''

    ps_ras = cat["ra"].data
    ps_decs = cat["dec"].data
    
    #we need to get all the pixel locations of these sources given the wcs and see which ones lie on the main segment
    #sources not on the main segment will be on a zero!
    ra_all = tractor.get("ra")
    dec_all = tractor.get("dec")

    #find all the sources that match the blend_remove_cat objects!
    c = SkyCoord(ra= ps_ras* u.degree, dec= ps_decs*u.degree )
    catalog = SkyCoord(ra=ra_all*u.degree, dec=dec_all*u.degree )
    idx, d2d, d3d = c.match_to_catalog_sky(catalog)

    #get the indices of objects in the ra_all catalog that match and have zero separation!
    parent_source_inds = idx[d2d.arcsec == 0]

    if len(parent_source_inds) != len(ps_ras):
        raise ValueError(f"Inconsistent number of sources in parent galaxy : {len(parent_source_inds)}, {len(ps_ras)} for {tgid}")

    tractor_parent = tractor[parent_source_inds]

    ave_psfsize_dict = { "g": np.mean(tractor_parent.get("psfsize_g")), "r": np.mean(tractor_parent.get("psfsize_r")),  "z": np.mean(tractor_parent.get("psfsize_z"))    }
    
    return tractor_parent, ave_psfsize_dict
    


def get_main_blob_sources(i, ra, dec, tgid, zred, file_path, img_path, width, pixscale=0.262,testing=False):
    '''
    Function that gets all the tractor sources for things on the main blob. This will contain sources that we also removed via a color cut. But we will do that step in the cog function. Here for the simple photo purposes as well, we will save all the non-stars sources on the segment! 
    
    The models for individual sources are stored separately and in a folder
    '''

    #doing the latter is definitely, better, the former is more efficient.
    parent_source_file = file_path + "/parent_galaxy_sources.fits"
    source_cat_all_file = file_path + "/source_cat_all_main_segment.fits"
    
    if os.path.exists(source_cat_all_file):
        #load the entire tractor catalog
        tractor = load_tractor(file_path)

        #we want to also remove sources that do not have finite shape_r
        finite_shaper_mask = np.isfinite(tractor.get("shape_r"))
        tractor = tractor[finite_shaper_mask]
    
        #we will be saving models for all these sources!
        parent_source_cat = Table.read(parent_source_file)
        source_all_main_cat = Table.read(source_cat_all_file)

        tractor_parent, ave_psfsize_dict = get_matched_subset(parent_source_cat, tractor, tgid)
        tractor_all_main_seg, ave_psfsize_dict = get_matched_subset(source_all_main_cat, tractor, tgid)
        
        tractor_save_dir = file_path + "/tractor_models"
        
        os.makedirs(tractor_save_dir, exist_ok=True)

        #if files already exist, we want to remove them to avoid any confusions
        for filename in os.listdir(tractor_save_dir):
            if filename.endswith(".npy") or filename.endswith(".npz"):
                file_path_i = os.path.join(tractor_save_dir, filename)
                try:
                    os.remove(file_path_i)
                except Exception as e:
                    print(f"Warning: failed to delete {file_path_i}. Reason: {e}")


        #also remove the .npy total model files
        file1 = file_path + "/tractor_main_segment_model.npy"
        file2 = file_path + "/tractor_parent_sources_model.npy"
        
        if os.path.exists(file1):
            os.remove(file1)
        if os.path.exists(file2):
            os.remove(file2)

        wcs = make_custom_wcs(ra, dec, width, pixscale)
               
        if len(tractor_parent) != 0:

            max_ra_diff = np.max(np.abs(tractor_parent.get("ra") - parent_source_cat["ra"]))
            if max_ra_diff > 0:
                raise ValueError(f"Inconsistent RA values in get_main_blob_sources : {max_ra_diff}")
                
            total_model = np.zeros((3, width, width))

            tot_count = len(tractor_parent)
            
            if tot_count > 100:
                plot_progress=True
            else:
                plot_progress=False

            ##WE ONLY SAVE THE INDIVIDUAL SOURCES IF Z > 0.005
            ##ELSE WE JUST SAVE THE ENTIRE MODEL IMAGE DIRECTLY!

            if zred >= 0.005:
                #loop through each source in the parent galaxy source catalog
                #we need the individual source models for the galaxies where we apply the isolate mask and those are only at z>0.005
                for k in range(len(tractor_parent)):
                    if plot_progress:
                        if k % 100 == 0:
                            print(f"TGID:{tgid}, progress={k/tot_count:.2f}")
                        
                    mod = build_model_image(tractor_parent[k], wcs, ave_psfsize_dict, mean_psf=True)
                    total_model += mod
                    np.save(f"{tractor_save_dir}/tractor_parent_source_model_{parent_source_cat['source_objid_new'].data[k]:d}.npy", mod)
    
            else:
                print(f"TGID:{tgid}, Z < 0.005 and just saving total model image as too many sources!")
                #we do not need to save each source individually, and we just save the direct image!!
                total_model = build_model_image(tractor_parent, wcs, ave_psfsize_dict, mean_psf=True)

        
            #save the total image of the main segment 
            total_model_main_seg = build_model_image(tractor_all_main_seg, wcs, ave_psfsize_dict, mean_psf=True)
                
            #save the total model of parent sources
            np.save(f"{file_path}/tractor_parent_sources_model.npy", total_model)

            #save the total model of all main segment sources
            np.save(f"{file_path}/tractor_main_segment_model.npy", total_model_main_seg)
                
            #save the total combined image for reference!
            save_rgb_single_panel(total_model, file_path, image_name="tractor_main_segment_galaxy_model")
        else:
            print(f"The parent source catalog existed for {tgid}, but no sources remain after parent mask.")
            
            #there were no sources to subtract and so we can save an empty array!
            # np.save(f"{file_path}/tractor_parent_galaxy_model.npy", np.zeros((3, width, width))  )
            pass

    else:
        print(f"The parent source catalog file did not exist for {tgid}! Skipping tractor model generation")
    
    return


def worker(args):
    i, dwarf_cat, func = args
    func(
        i,
        dwarf_cat["RA"][i],
        dwarf_cat["DEC"][i],
        dwarf_cat["TARGETID"][i],
        dwarf_cat["Z"][i],
        dwarf_cat["FILE_PATH"][i],
        dwarf_cat["IMAGE_PATH"][i],
        dwarf_cat["IMAGE_SIZE_PIX"][i],
        testing=False
    )
    return 1  # for progress counting


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
    result.add_argument('-use_sample', dest='use_sample', type=str, default = "clean") 
    result.add_argument('-sample', dest='sample', type=str, default = "BGS_BRIGHT") 
    result.add_argument('-img_source',dest='img_source', action = "store_true")  
    result.add_argument('-bkg_source',dest='bkg_source', action = "store_true")    
    result.add_argument('-max_num',dest='max_num',type = int, default= 100000 )  
    result.add_argument('-blend_remove_source',dest='blend_remove_source', action = "store_true")  
    result.add_argument('-parent_galaxy',dest='parent_galaxy', action = "store_true")
    result.add_argument('-tgids',dest="tgids_list", type=parse_tgids) 
    
    return result


if __name__ == '__main__': 

    from astropy.units import UnitsWarning
    warnings.filterwarnings("ignore", category=UnitsWarning)

    # read in command line arguments
    args = argument_parser().parse_args()

    use_sample = args.use_sample
    sample = args.sample
    img_source = args.img_source
    bkg_source = args.bkg_source
    blend_remove_source = args.blend_remove_source
    parent_galaxy = args.parent_galaxy
    max_num = args.max_num
    tgids_list = args.tgids_list
    
    print(f"Reading the sample = {use_sample}")

    if use_sample == "sga":
        dwarf_cat = Table.read("/pscratch/sd/v/virajvm/catalog_dr1_dwarfs/iron_desi_SGA_matched_dwarfs_REPROCESS.fits")
    if use_sample == "clean":
        dwarf_cat = Table.read("/pscratch/sd/v/virajvm/catalog_dr1_dwarfs/desi_y1_dwarf_clean_catalog_v4_RUN_W_APER.fits")
        #we added the below condition as we changed our definition recently!
        #this goes from 40k -> 39.4k
        dwarf_cat = dwarf_cat[(dwarf_cat["RCHISQ_R"] < 4 ) & (dwarf_cat["RCHISQ_G"] < 4 )  & (dwarf_cat["RCHISQ_Z"] < 4 )   ]
        
    if use_sample == "shred":
        dwarf_cat = Table.read("/pscratch/sd/v/virajvm/catalog_dr1_dwarfs/desi_y1_dwarf_shreds_catalog_v4.fits")

    print(f"Reading the sample = {sample}")
    dwarf_cat = dwarf_cat[dwarf_cat["SAMPLE"] == sample]


    if tgids_list is not None:
        print("List of targetids to process:",tgids_list)
        dwarf_cat = dwarf_cat[np.isin(dwarf_cat['TARGETID'], np.array(tgids_list) )]
        print("Number of targetids to process =", len(dwarf_cat))

    dwarf_cat = dwarf_cat[:max_num]
    
    print(len(dwarf_cat))
    total = len(dwarf_cat)

    if img_source:
        print("Getting img source models")
        pool = mp.Pool(62)
        completed = 0
        for _ in pool.imap_unordered(worker, [(i, dwarf_cat, get_img_source ) for i in range(total)], chunksize = 500 ):
            completed += 1
            if completed % 1000 == 0 or completed == total:
                simple_progress_bar(completed, total-1)
        pool.close()
        pool.join()


    if bkg_source:
        print("Getting bkg source models")
        pool = mp.Pool(62)
        
        completed = 0
        for _ in pool.imap_unordered(worker, [(i, dwarf_cat, get_bkg_sources ) for i in range(total)], chunksize = 500 ):
            completed += 1
            if completed % 1000 == 0 or completed == total:
                simple_progress_bar(completed, total-1)
    
        pool.close()
        pool.join()

    if blend_remove_source:
        print("Getting blend remove source models")
        pool = mp.Pool(62)
        
        completed = 0
        for _ in pool.imap_unordered(worker, [(i, dwarf_cat, get_blended_remove_sources) for i in range(total)], chunksize = 500 ):
            completed += 1
            if completed % 1000 == 0 or completed == total:
                simple_progress_bar(completed, total-1)
    
        pool.close()
        pool.join()


    if parent_galaxy:
        print("Getting the parent galaxy source models")
        pool = mp.Pool(62)
        
        completed = 0
        for _ in pool.imap_unordered(worker, [(i, dwarf_cat, get_main_blob_sources) for i in range(total)], chunksize = 500 ):
            completed += 1
            if completed % 1000 == 0 or completed == total:
                simple_progress_bar(completed, total-1)
    
        pool.close()
        pool.join()
        
    ###########################################
    ##getting the model for the temporary source!

    # dwarf_cat = Table.read("/pscratch/sd/v/virajvm/catalog_dr1_dwarfs/TEMPORARY_desi_y1_dwarf_shreds_catalog_v3.fits")
    
    # ra = dwarf_cat["RA"][-1]
    # dec = dwarf_cat["DEC"][-1]
    # tgid = dwarf_cat["TARGETID"][-1]
    # file_path = dwarf_cat["FILE_PATH"][-1]
    # img_path = dwarf_cat["IMAGE_PATH"][-1]

    # print(ra,dec,tgid)
    # print(file_path)
    # i = 0
    
    # # get_img_source(i, ra, dec, tgid, file_path, img_path, pixscale=0.262, width=350, testing=False)
    # get_blended_remove_sources(i, ra, dec, tgid, file_path, img_path, pixscale=0.262, width=350, testing=False)
    # get_bkg_sources(i, ra, dec, tgid, file_path, img_path, pixscale=0.262, width=350, testing=False)
    


    

    

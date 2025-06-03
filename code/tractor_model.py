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
from legacypipe.survey import LegacySurveyWcs, ConstantFitsWcs
from astropy.wcs import WCS
from legacypipe.survey import wcs_for_brick, BrickDuck
from astrometry.util.fits import fits_table
import matplotlib.pyplot as plt
import glob
from astropy.table import Table, vstack
import os
import astropy.units as u
from astropy.coordinates import SkyCoord
import multiprocessing as mp
from astropy.wcs import WCS
import time
import sys


def simple_progress_bar(iteration, total, length=30):
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


def compute_psf_sigma(tractor_subset, band, average_psfsize, mean=True):
    if len(tractor_subset.get(f"psfsize_{band}")) == 1 and tractor_subset.get(f"psfsize_{band}") == 0:
        #if we are only reading a single source and it has no psfsize, we take the average value that inputted to function
        fwhm = average_psfsize
    else:
        if mean:
            fwhm = np.mean(tractor_subset.get(f"psfsize_{band}"))
        else:
            fwhm = tractor_subset.get(f"psfsize_{band}")[0]
        
    return (fwhm / 2.3548) / 0.262


def build_model_image(tractor_subset, wcs, average_psfsize, mean_psf=False):
    return np.array([
        srcs2image(tractor_subset, wcs, band=band, allbands='grz', pixelized_psf=None, psf_sigma=compute_psf_sigma(tractor_subset, band, average_psfsize[band], mean=mean_psf))
        for band in "grz"
    ])
    
def save_rgb_tripanel(mod, data_arr, file_path, tgid=None, testing=False,use_center_only=False):
    
    rgb_model = sdss_rgb(mod, ["g","r","z"], scales=dict(g=(2,6.0), r=(1,3.4), z=(0,2.2)), m=0.03)
    rgb_data = sdss_rgb(data_arr, ["g","r","z"], scales=dict(g=(2,6.0), r=(1,3.4), z=(0,2.2)), m=0.03)
    resis = data_arr - mod
    rgb_resis = sdss_rgb(resis, ["g","r","z"], scales=dict(g=(2,6.0), r=(1,3.4), z=(0,2.2)), m=0.03)

    start, size = (350 - 64) // 2, 64
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


def get_img_source(i, ra, dec, tgid, file_path, img_path, pixscale=0.262, width=350, testing=True):
    '''
    Getting the model of the targeted DESI source
    '''
    
    # if os.path.exists(f"{file_path}/tractor_source_model.npy"):
    #     return

    tractor = load_tractor(file_path)
    
    seps = compute_separations(ra, dec, tractor.get("ra"), tractor.get("dec"))
    
    tractor_source = tractor[seps == 0]

    if len(tractor_source) != 1:
        print(f"Ambiguity for index {i} and coords ({ra}, {dec})")
        raise ValueError("Tractor source match not unique.")

    #if this source has a missing psfsize then we assume an average psfsize from the catalog
    ave_psfsize_dict = { "g": np.mean(tractor.get("psfsize_g")), "r": np.mean(tractor.get("psfsize_r")),  "z": np.mean(tractor.get("psfsize_z"))    }

    wcs = make_custom_wcs(ra, dec, width, pixscale)
    mod = build_model_image(tractor_source, wcs, ave_psfsize_dict)
    np.save(f"{file_path}/tractor_source_model.npy", mod)

    img_data = fits.open(img_path)[0].data
    
    save_rgb_tripanel(mod, img_data, file_path, tgid, testing,use_center_only=True)

    return


def get_bkg_sources(i, ra, dec, tgid, file_path, img_path, pixscale=0.262, width=350, testing=True):
    '''
    Getting the tractor model of the background
    '''
    # if os.path.exists(f"{file_path}/tractor_background_model.npy"):
    #     return

    tractor = load_tractor(file_path)
    
    segm = np.load(f"{file_path}/main_segment_map.npy")
    img_data = fits.open(img_path)[0].data
    wcs_cutout = WCS(fits.getheader(img_path))

    xpix, ypix, _ = wcs_cutout.all_world2pix(tractor.get("ra"), tractor.get("dec"), 0, 1)
    on_main = ~np.isnan(segm[ypix.astype(int), xpix.astype(int)])
    bkg_sources = tractor[~on_main]

    wcs = make_custom_wcs(ra, dec, width, pixscale)

    #if this source has a missing psfsize then we assume an average psfsize from the catalog
    ave_psfsize_dict = { "g": np.mean(tractor.get("psfsize_g")), "r": np.mean(tractor.get("psfsize_r")),  "z": np.mean(tractor.get("psfsize_z"))    }
    
    mod = build_model_image(bkg_sources, wcs, ave_psfsize_dict, mean_psf=True)
    
    np.save(f"{file_path}/tractor_background_model.npy", mod)

    save_rgb_tripanel(mod, img_data, file_path, tgid, testing,use_center_only=False)
    
    return


def get_blended_remove_sources(i, ra, dec, tgid, file_path, img_path, pixscale=0.262, width=350, testing=True):
    '''
    Getting the model of the sources that lie on the main segment (blended sources) but that are deemed to be not be part of the parent galaxy
    '''

    # if os.path.exists(file_path + "/tractor_blend_remove_model.npy"):
    #     return

    tractor = load_tractor(file_path)

    #load the source catalog that we are removing
    blend_remove_cat = Table.read(file_path + "/blended_source_remove_cat.fits")

    if len(blend_remove_cat) == 0:
        #there were no sources to subtract and so we can save an empty array!
        np.save(f"{file_path}/tractor_blend_remove_model.npy", np.zeros((3, 350, 350))  )
    else: 
        br_ras = blend_remove_cat["ra"]
        br_decs = blend_remove_cat["dec"]
    
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
        
        if len(tractor_blend_re) != 0 or ():
                
            wcs = make_custom_wcs(ra, dec, width, pixscale)
            
            mod = build_model_image(tractor_blend_re, wcs, ave_psfsize_dict, mean_psf=True)
            
            np.save(f"{file_path}/tractor_blend_remove_model.npy", mod)
        
            img_data = fits.open(img_path)[0].data
            save_rgb_tripanel(mod, img_data, file_path, tgid, testing,use_center_only=False)
        else:
            #there were no sources to subtract and so we can save an empty array!
            np.save(f"{file_path}/tractor_blend_remove_model.npy", np.zeros((3, 350, 350))  )
        
    
    return

    
## another function to consider, if it is fast enough in the future is to store the tractor model for each object and then just add the ones we want at the end !!


def worker(args):
    i, dwarf_cat, func = args
    func(
        i,
        dwarf_cat["RA"][i],
        dwarf_cat["DEC"][i],
        dwarf_cat["TARGETID"][i],
        dwarf_cat["FILE_PATH"][i],
        dwarf_cat["IMAGE_PATH"][i],
        testing=False
    )
    return 1  # for progress counting


    

if __name__ == '__main__':

    #need to run this for some of the clean sources too as we are testing our pipeline on them ...
    #get the tractor model and their backgrounds for these sources

    ##load the file 

    # dwarf_cat = Table.read("/pscratch/sd/v/virajvm/catalog_dr1_dwarfs/desi_y1_dwarf_clean_catalog_v3.fits")
    # dwarf_cat_1 = dwarf_cat[dwarf_cat["SAMPLE"] == "BGS_BRIGHT"][:12000]
    # dwarf_cat_2 = dwarf_cat[dwarf_cat["SAMPLE"] == "BGS_FAINT"][:2400]
    # dwarf_cat_3 = dwarf_cat[dwarf_cat["SAMPLE"] == "LOWZ"][:500]
    # dwarf_cat_4 = dwarf_cat[dwarf_cat["SAMPLE"] == "ELG"][:24000]
    # dwarf_cat = vstack([dwarf_cat_1, dwarf_cat_2, dwarf_cat_3, dwarf_cat_4])

    # print(len(dwarf_cat))

    # total = len(dwarf_cat)

    # ###########################################
    # pool = mp.Pool(128)

    # worker_func = make_worker(get_img_sources)

    # completed = 0
    # for _ in pool.imap_unordered(worker, [(i, dwarf_cat) for i in range(total)], chunksize = 500 ):
    #     completed += 1
    #     simple_progress_bar(completed, total-1)

    # pool.close()
    # pool.join()

    # ###########################################
    
    # pool = mp.Pool(128)
    
    # completed = 0
    # for _ in pool.imap_unordered(worker, [(i, dwarf_cat, get_bkg_sources ) for i in range(total)], chunksize = 500 ):
    #     completed += 1
    #     simple_progress_bar(completed, total-1)

    # pool.close()
    # pool.join()

    ############################################
    
    # pool = mp.Pool(128)
    
    # completed = 0
    # for _ in pool.imap_unordered(worker, [(i, dwarf_cat, get_blended_remove_sources) for i in range(total)], chunksize = 500 ):
    #     completed += 1
    #     simple_progress_bar(completed, total-1)

    # pool.close()
    # pool.join()

    ###########################################

    #### load for shredded cat
    # dwarf_cat = Table.read("/pscratch/sd/v/virajvm/catalog_dr1_dwarfs/desi_y1_dwarf_shreds_catalog_v3.fits")
    # dwarf_cat = dwarf_cat[(dwarf_cat["PCNN_FRAGMENT"] >= 0.4) & (dwarf_cat["SAMPLE"] == "ELG")]

    # print(len(dwarf_cat))

    # total = len(dwarf_cat)

    # pool = mp.Pool(128)
    
    # completed = 0
    # for _ in pool.imap_unordered(worker, [(i, dwarf_cat, get_bkg_sources ) for i in range(total)], chunksize = 500 ):
    #     completed += 1
    #     simple_progress_bar(completed, total-1)

    # pool.close()
    # pool.join()


    ###########################################

    # pool = mp.Pool(128)

    # completed = 0
    # for _ in pool.imap_unordered(worker, [(i, dwarf_cat, get_blended_remove_sources) for i in range(total)], chunksize = 500 ):
    #     completed += 1
    #     simple_progress_bar(completed, total-1)

    # pool.close()
    # pool.join()

    ###########################################
    ##getting the model for the temporary source!

    dwarf_cat = Table.read("/pscratch/sd/v/virajvm/catalog_dr1_dwarfs/TEMPORARY_desi_y1_dwarf_shreds_catalog_v3.fits")
    
    ra = dwarf_cat["RA"][-1]
    dec = dwarf_cat["DEC"][-1]
    tgid = dwarf_cat["TARGETID"][-1]
    file_path = dwarf_cat["FILE_PATH"][-1]
    img_path = dwarf_cat["IMAGE_PATH"][-1]

    print(ra,dec,tgid)
    print(file_path)
    i = 0
    
    # get_img_source(i, ra, dec, tgid, file_path, img_path, pixscale=0.262, width=350, testing=False)
    get_blended_remove_sources(i, ra, dec, tgid, file_path, img_path, pixscale=0.262, width=350, testing=False)
    get_bkg_sources(i, ra, dec, tgid, file_path, img_path, pixscale=0.262, width=350, testing=False)
    


    

    

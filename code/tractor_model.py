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
from astropy.table import Table
import os
import astropy.units as u
from astropy.coordinates import SkyCoord
import multiprocessing as mp

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

    tr = tractor.Tractor([tim], srcs)
    mod = tr.getModelImage(0)

    return mod
    

def get_img_source(i, ra,dec,tgid,file_path, img_path, pixscale=0.262,width=350,testing=True):
    '''
    Function that constructs the tractor model of source of interest to get IMG-S

    the objid, brickid is used to identify the unique source!
    '''

    if os.path.exists(file_path + "/tractor_source_model.npy"):
        pass
    else:
    
        # hdulist = fits.open("/pscratch/sd/v/virajvm/catalog_dr1_dwarfs/temp/copsf_44.3842_0.6444.fits")
        # psf = {'grz'[i]: hdulist[i].data for i in range(3)}
        # from tractor.psf import PixelizedPSF
        # ppsf_g = PixelizedPSF(psf["g"])
        # ppsf_r = PixelizedPSF(psf["r"])
        # ppsf_z = PixelizedPSF(psf["z"])
    
        brickname = 'custom-{}'.format(custom_brickname(ra, dec))
        brick = BrickDuck(ra, dec, brickname)
        targetwcs = wcs_for_brick(brick, W=float(width), H=float(width), pixscale=pixscale)
        #the above targetwcs is in the correct tan format needed for below function!
        wcs = ConstantFitsWcs(targetwcs)
    
        #the below code is taken from: https://github.com/moustakas/legacyhalos/blob/main/py/legacyhalos/virgofilaments.py#L693C1-L694C1
        tractor_file = file_path+"/source_cat_f_more.fits"
        if os.path.exists(tractor_file):
            pass
        else:
            tractor_file = file_path + "/source_cat_f.fits"
    
    
    
    
        #select the columns of relevance!
        cols = ['ra', 'dec', 'bx', 'by', 'type', 'ref_cat', 'ref_id',
                'sersic', 'shape_r', 'shape_e1', 'shape_e2',
                'flux_g', 'flux_r', 'flux_z',
                'flux_ivar_g', 'flux_ivar_r', 'flux_ivar_z',
                'nobs_g', 'nobs_r', 'nobs_z',
                'psfdepth_g', 'psfdepth_r', 'psfdepth_z',
                'psfsize_g', 'psfsize_r', 'psfsize_z', "BRICKID","OBJID"]
    
        tractor = fits_table(tractor_file, columns=cols)
    
        ##sometimes there is are repeated objects due to overlapping bricks and we will be removing those now!!
    
        coords = np.array(list(zip(tractor.get('ra'), tractor.get('dec'))))
    
        # Find unique rows based on RA and DEC
        _, unique_indices = np.unique(coords, axis=0, return_index=True)
        
        # Keep only the unique rows
        tractor = tractor[unique_indices]
    
        #however, we only need to the feed the source of interest to the src2image
        #given all the ras/decs in the catalog, find the one that is our source, this is the source we will be subtracting!
        #compute the seps and find the one with seps == 0!
        # our_source = (tractor.get("BRICKID") == brickid_i) & (tractor.get("OBJID") == objid)
        seps = compute_separations(ra_ref = ra, dec_ref = dec, ra_all = tractor.get("ra"), dec_all = tractor.get("dec"))
    
        tractor_source = tractor[seps == 0]
    
        if len(tractor_source) != 1:
            print(i)
            print(tractor_source.get("ra"))
            print(tractor_source.get("dec"))
            print(tractor_source.get("BRICKID"))
            print(tractor_source.get("OBJID"))
            print(tractor_source.get("OBJID"))
            print(ra, dec)        
            raise ValueError("Number of sources found in tractor not equal to 1!")
    
        #a relevant column in the tractor catalog
        # psfsize_g = arcsec Weighted average PSF FWHM in the g band
        #we have to convert it to sigma and to pixels (using the adopted pixel scale)
        psf_sigma_g = (tractor_source.get("psfsize_g")[0]/2.3548)/0.262
        psf_sigma_r = (tractor_source.get("psfsize_r")[0]/2.3548)/0.262
        psf_sigma_z = (tractor_source.get("psfsize_z")[0]/2.3548)/0.262
    
        #when providing with pixelized psf
        # mod_g = srcs2image(tractor_source, wcs, band='g', allbands='grz', pixelized_psf=ppsf_g, psf_sigma=1.0)
        # mod_r = srcs2image(tractor_source, wcs, band='r', allbands='grz', pixelized_psf=ppsf_r, psf_sigma=1.0)
        # mod_z = srcs2image(tractor_source, wcs, band='z', allbands='grz', pixelized_psf=ppsf_z, psf_sigma=1.0)
    
        #constructing the model image
        mod_g = srcs2image(tractor_source, wcs, band='g', allbands='grz', pixelized_psf=None, psf_sigma=psf_sigma_g)
        mod_r = srcs2image(tractor_source, wcs, band='r', allbands='grz', pixelized_psf=None, psf_sigma=psf_sigma_r)
        mod_z = srcs2image(tractor_source, wcs, band='z', allbands='grz', pixelized_psf=None, psf_sigma=psf_sigma_z)
        mod_tot = np.array([mod_g, mod_r, mod_z])
    
        #constructing rgb model image
        rgb_model = sdss_rgb(mod_tot, ["g","r","z"], scales=dict(g=(2,6.0), r=(1,3.4), z=(0,2.2)), m=0.03)
    
        ##read the actual image data
        img_data = fits.open(img_path)
        data_arr = img_data[0].data
        rgb_data = sdss_rgb(data_arr, ["g","r","z"], scales=dict(g=(2,6.0), r=(1,3.4), z=(0,2.2)), m=0.03)
    
        ##get the image - source image
        resis = data_arr - mod_tot   
        rgb_resis = sdss_rgb(resis, ["g","r","z"], scales=dict(g=(2,6.0), r=(1,3.4), z=(0,2.2)), m=0.03)
        
        #let us make the color image and zoom in on 64x64
        #let us zoom in 64x64
        size = 64
        start = (width - size) // 2
        end = start + size
    
        fig,ax = plt.subplots(1,3,figsize = (12,4))
        ax[0].imshow(rgb_model[start:end, start:end,:],origin="lower")
        ax[1].imshow(rgb_data[start:end, start:end,:],origin="lower")
        ax[2].imshow(rgb_resis[start:end, start:end,:],origin="lower")
    
        #save these fits file in the file_path so we can load them and add to our aperture photo summary plot!
    
        np.save(file_path + "/tractor_source_model.npy", mod_tot)
        
        plt.savefig(file_path + "/tractor_model_image.png")
        if testing or i < 500:
            plt.savefig(f"/pscratch/sd/v/virajvm/temp_tractor_models/tractor_model_{tgid}.png")
            
        plt.close()

    return


def worker(args):
    i, dwarf_cat = args
    get_img_source(
        i, 
        dwarf_cat["RA"][i], 
        dwarf_cat["DEC"][i], 
        dwarf_cat["TARGETID"][i], 
        dwarf_cat["FILE_PATH"][i], 
        dwarf_cat["IMAGE_PATH"][i],
        testing=False
    )
    
    return 1  # just for counting finished tasks

if __name__ == '__main__':

    ##load the file 

    dwarf_cat = Table.read("/pscratch/sd/v/virajvm/catalog_dr1_dwarfs/desi_y1_dwarf_shreds_catalog_v3.fits")
    dwarf_cat = dwarf_cat

    total = len(dwarf_cat)
    pool = mp.Pool(128)

    completed = 0
    for _ in pool.imap_unordered(worker, [(i, dwarf_cat) for i in range(total)], chunksize = 500 ):
        completed += 1
        simple_progress_bar(completed, total-1)

    pool.close()
    pool.join()


    # for i in range(100):
    #     simple_progress_bar( i, len(dwarf_cat) ) 
        
    #     get_img_source(i, dwarf_cat["RA"][i], dwarf_cat["DEC"][i], dwarf_cat["TARGETID"][i], dwarf_cat["FILE_PATH"][i], dwarf_cat["IMAGE_PATH"][i], testing=False)

        

 
    

    

    

'''
shifterimg pull docker:legacysurvey/legacypipe:DR10.3.4
shifter --image docker:legacysurvey/legacypipe:DR10.3.4 python3 tractor_sources.py

In this script, I try to figure out how to construct a model image given some tractor sources and then subtract it from the real image!

'''


# session = requests.Session()
# psf_dict = fetch_psf(ra_k,dec_k, session)



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
    
ra_k = 39.07210225914647
dec_k = 8.368685407411268	
tgid = 39627984176418324
pixscale = 0.262
width = 350


rel_path = "/pscratch/sd/v/virajvm/redo_photometry_plots/all_deshreds/south/sweep-030p005-040p010/0390p082/LOWZ_tgid_39627984176418324"

if __name__ == '__main__':


    # img_path = "/pscratch/sd/v/virajvm/redo_photometry_plots/all_deshreds_cutouts/image_tgid_39627803590660098_ra_44.384196_dec_0.644421.fits"
    img_path = glob.glob(f"/pscratch/sd/v/virajvm/redo_photometry_plots/all_deshreds_cutouts/image_tgid_{tgid}*")[0]
    print(img_path)

    # hdulist = fits.open("/pscratch/sd/v/virajvm/catalog_dr1_dwarfs/temp/copsf_44.3842_0.6444.fits")
    # psf = {'grz'[i]: hdulist[i].data for i in range(3)}
    # from tractor.psf import PixelizedPSF
    # ppsf_g = PixelizedPSF(psf["g"])
    # ppsf_r = PixelizedPSF(psf["r"])
    # ppsf_z = PixelizedPSF(psf["z"])

    brickname = 'custom-{}'.format(custom_brickname(ra_k, dec_k))
    brick = BrickDuck(ra_k, dec_k, brickname)
    targetwcs = wcs_for_brick(brick, W=float(width), H=float(width), pixscale=pixscale)
    #the above targetwcs is in the correct tan format needed for below function!
    wcs = ConstantFitsWcs(targetwcs)

    #the below code is taken from: https://github.com/moustakas/legacyhalos/blob/main/py/legacyhalos/virgofilaments.py#L693C1-L694C1
    # tractorfile = os.path.join(galaxydir, '{}-{}.fits'.format(galaxy, filt2imfile['tractor']))

    tractorfile = rel_path + "/source_cat_f_more.fits"
         
    cols = ['ra', 'dec', 'bx', 'by', 'type', 'ref_cat', 'ref_id',
            'sersic', 'shape_r', 'shape_e1', 'shape_e2',
            'flux_g', 'flux_r', 'flux_z',
            'flux_ivar_g', 'flux_ivar_r', 'flux_ivar_z',
            'nobs_g', 'nobs_r', 'nobs_z',
            'psfdepth_g', 'psfdepth_r', 'psfdepth_z',
            'psfsize_g', 'psfsize_r', 'psfsize_z']

    tractor = fits_table(tractorfile, columns=cols)

    #however, we only need to the feed the source of interest to the src2image
    #given all the ras/decs in the catalog, find the one that is our source!!
    our_source = tractor.get("ra") == ra_k
    ##maybe a better way to select this?

    tractor_source = tractor[our_source]

    #a relevant column in the tractor catalog
    # psfsize_g float32 arcsec Weighted average PSF FWHM in the g band
    #we have to convert it to sigma and to pixels (using the adopted pixel scale)

    psf_sigma_g = (tractor_source.get("psfsize_g")[0]/2.3548)/0.262
    psf_sigma_r = (tractor_source.get("psfsize_r")[0]/2.3548)/0.262
    psf_sigma_z = (tractor_source.get("psfsize_z")[0]/2.3548)/0.262
    
    print(psf_sigma_g, psf_sigma_r,psf_sigma_z)

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
    
    #let us make the color image!
    #let us zoom in 64x64
    size = 64
    start = (width - size) // 2
    end = start + size

    fig,ax = plt.subplots(1,3,figsize = (12,4))
    ax[0].imshow(rgb_model[start:end, start:end,:],origin="lower")
    ax[1].imshow(rgb_data[start:end, start:end,:],origin="lower")
    ax[2].imshow(rgb_resis[start:end, start:end,:],origin="lower")
    
    plt.savefig(rel_path + "/temp_image.png")
    
    plt.close()
    

    

    

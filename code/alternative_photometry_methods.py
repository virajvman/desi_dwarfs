'''
In this script, we contain scripts and functions for getting alternative photometry, that is for the fraction of sources that have something wonky identified from the basic COG ones! This should be a small-ish fraction!
'''

import os
import sys
import joblib
import numpy as np
from tqdm import trange
from tqdm import tqdm
from astropy.io import fits
from astropy.table import Table, vstack, join
from scipy.ndimage import binary_dilation
from astropy.table import Column
from astropy import units as u
from astropy.coordinates import SkyCoord
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import LogNorm
from astropy.wcs import WCS
import matplotlib.patches as patches
from matplotlib.patches import Circle
from photutils.segmentation import detect_sources, deblend_sources
import matplotlib.cm as cm
url_prefix = 'https://www.legacysurvey.org/viewer/'
# Third-party imports 
import concurrent.futures
import pickle
from photutils.segmentation import make_2dgaussian_kernel
from astropy.convolution import convolve
from desi_lowz_funcs import make_subplots, sdss_rgb, get_elliptical_aperture


def mags_to_flux(mags, zeropoint=22.5):
    return 10**((zeropoint - mags) / 2.5)


def flux_to_mag(flux, zeropoint=22.5):
    # Protect against zero/negative flux
    if flux > 0:
        return zeropoint - 2.5*np.log10(flux)
    else:
        return np.nan  # or some sentinel value



def find_nearest_island(segment_map, fiber_xpix,fiber_ypix):
    '''
    Function that finds the nearest segment and returns its index!!

    Is a general function that we will use nuemerous times!
    '''
    
    all_xpixs, all_ypixs = np.meshgrid( np.arange(np.shape(segment_map)[0]), np.arange(np.shape(segment_map)[1]) )
    all_dists = np.sqrt ( ( all_xpixs - fiber_xpix)**2 + ( all_ypixs - fiber_ypix)**2 )

    #get all the distances to the pixels that are not background
    all_segs_notbg = segment_map[ (segment_map != 0) ]
    all_dists_segpixs = all_dists[ (segment_map != 0)  ]

    #the id of the closest segment that we will call our galaxy!
    close_island_num = all_segs_notbg[ np.argmin(all_dists_segpixs) ]

    #the closest distance in puxels
    closest_dist_pix = np.min(all_dists_segpixs)
    
    return close_island_num, closest_dist_pix
    


def get_simplest_photometry(data_arr, r_rms, fiber_xpix, fiber_ypix, source_cat_no_stars, save_path,source_zred=None):
    '''
    This is the photometry method where we forego all color based association and bad pixel masking. In this step, we smooth the r-band image a lot, identify the main deblended blob that contains our galaxy of interest, get all the tractor sources on that blob, and sum their photometry up!!

    This is the simplest kind of photometry, purely based on spatial information, and is a good backup for photometry when multiple segments are detected in the COG aperture map or some sort of negative, downward trend in COG mags indicative of over subtraction! 
    
    Note that below a certain redshift, we will just ignore the deblending step for now, as we find it tends to still overshred. Based on whatever was the final photometry method, we will estimate the final aperture shape and see what fraction is outside the image. and construct a maskbit based on that.

    Note, in case the desi fiber does not line up with any segment, we consider the closest possible segment! And just make a note of this.

    '''

    r_band_data = data_arr[1]

    #these are the same parameters as the isolate galaxy mask!!
    kernel = make_2dgaussian_kernel(15, size=29) 

    #these are the same parameters as our original segmentation
    threshold_rms_scale = 1.5
    npixels_min =  10

    threshold = threshold_rms_scale * r_rms
    convolved_tot_data = convolve( r_band_data, kernel )
    
    segment_map = detect_sources(convolved_tot_data, threshold, npixels=npixels_min) 

    if segment_map is None:        
        np.save(save_path + "/simplest_photometry_binary_mask.npy",  np.zeros_like(r_band_data ) )
        
        np.save(save_path + "/simplest_photo_rgb_mask_image.npy", np.ones_like(data_arr) )
        
        return 3*[np.nan], np.nan, 0

    else:
        #note no deblending here!! except for the large step at the end!    
        ## note that these 2d arrays are [y-coord, x-coord]
        island_num = segment_map.data[int(fiber_ypix),int(fiber_xpix)]
    
        segment_map_smooth = np.copy(segment_map.data)
        
        if island_num == 0:
            #very similarly to to the aperture photometry step, here, we find the nearest possible source
            new_island_num, closest_island_dist_pix = find_nearest_island(segment_map_smooth, fiber_xpix,fiber_ypix)
        else:
            #the fiber is already on a segment!
            new_island_num = island_num
            closest_island_dist_pix = 0

        #make everything that is not on the main blob a 0!
        segment_map_smooth[segment_map_smooth != new_island_num] = 0

    
        ##TODO: ONCE THE MAIN SEGMENT IS IDENTIFIED, WE DO THE DEBLEND STEP. We do not do this for now.
        ##This is the jaccard score stuff! But there are stars in the way!!
        # ncontrast_opt, jaccard_img_path = find_optimal_ncontrast(img_rgb, img_rgb_mask, convolved_tot_data, segment_map, nlevel_val = 4,save_path = None, pcnn_val=None,radec=None, tgid=None, mu_rough=None, source_zred=None, mu_r_smooth = None)
            
        #estimate the aperture of this simplest photo and measure its outside fraction!
        aperture_for_simple_phot, areafrac_in_image_simple_phot, _, _ = get_elliptical_aperture( segment_map_smooth , sigma = 4.25 )

        #use this new segmentation mask to filter the tractor sources! and save their photometry!!!
        xpix_all = source_cat_no_stars["xpix"].astype(int)
        ypix_all = source_cat_no_stars["ypix"].astype(int)
        
        # mask of which sources fall on a segment (nonzero)
        on_segment_mask = (segment_map_smooth[ypix_all, xpix_all] != 0)    
        # filter catalog
        source_cat_no_stars_simple_model = source_cat_no_stars[on_segment_mask]
    
        #save this catalog!
        source_cat_no_stars_simple_model.write( save_path + "/simplest_photometry_parent_sources.fits", overwrite=True)

        #TODO: include a step in the tractor source download part where we download this!
    
        #let us save this mask
        np.save(save_path + "/simplest_photometry_binary_mask.npy", segment_map_smooth)
    
        #let us make a simple rgb image, where we apply this mask to it!
        fig,ax = make_subplots(ncol = 2, nrow = 1, return_fig = True)
        rgb_image_full = sdss_rgb(data_arr)
        #make the masked image
        data_arr[:, (segment_map_smooth == 0)] = 0
        rgb_image_mask = sdss_rgb(data_arr)
        ax[0].set_title("Full grz image",fontsize = 15)
        ax[0].imshow(rgb_image_full, origin="lower")
        ax[1].set_title("Simplest Photo grz mask image",fontsize = 15)
        ax[1].imshow(rgb_image_mask, origin="lower")

        #save this rgb image so we can load it in a future step!
        np.save(save_path + "/simplest_photo_rgb_mask_image.npy", rgb_image_mask)

        for axi in ax:
            axi.set_xticks([])
            axi.set_yticks([])
        fig.savefig(save_path + "/simplest_photometry_mask_image.png")
        plt.close(fig)
    
        #we will make a tractor model image with this sources later!
    
        #in the mean time, measure the simplest photometry!
        g_flux_corr = mags_to_flux(source_cat_no_stars_simple_model["mag_g"]) / source_cat_no_stars_simple_model["mw_transmission_g"]
        r_flux_corr = mags_to_flux(source_cat_no_stars_simple_model["mag_r"]) / source_cat_no_stars_simple_model["mw_transmission_r"]
        z_flux_corr = mags_to_flux(source_cat_no_stars_simple_model["mag_z"]) / source_cat_no_stars_simple_model["mw_transmission_z"]
        
        tot_g_mag = flux_to_mag(np.sum(g_flux_corr))
        tot_r_mag = flux_to_mag(np.sum(r_flux_corr))
        tot_z_mag = flux_to_mag(np.sum(z_flux_corr))
        
        return np.array([tot_g_mag, tot_r_mag, tot_z_mag]), closest_island_dist_pix,  areafrac_in_image_simple_phot
        
        
def get_galaxy_close_photometry_mask():
    '''
    This is the photometry method where we have identified one segment in the COG aperture map (so all good) but multiple galaxies are identified in the smooth deblending step. As it can difficult to identify whether the galaxy of interest is within the light of a much larger galaxy, we will do the simplest possible thing, which is, based on the newest deblended blob mask, we will consider only the tractor sources in that blob and sum up their photometry and call it end of story!

    Another approach here could have been to use the other deblended blobs as a mask during COG stage, but it gets tricky to mask the ICL contriibution e.g., in some rare cases and thus to have the most general, simplest approach, we simply take the tractor approach**

    Will give us a reasonably okay photometry! 

    Note that in general, we will be comparing the tractor based only photometry reconstruction of our sources with our COG method so we will have demonstrated that they are some what consistent even in shredding case. We have already demonstrated this for our clean sources!

    **Note in general, we will always prefer the COG approach, as tractor does not perfectly model the outskirts of shredded galaxies and so for accurate colors and photometry (without any systematic residuals), we will prefer the COG baseed magnitudes whenever possible!!

    Note, in case the desi fiber does not line up with any segment, we consider the closest possible segment! And just make a note of this
    
    '''
    
    
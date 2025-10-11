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
from desi_lowz_funcs import make_subplots, sdss_rgb, get_elliptical_aperture, find_nearest_island
from desi_lowz_funcs import mags_to_flux, flux_to_mag


def process_deblend_image(segment_map, segm_deblend, fiber_xpix, fiber_ypix):
    '''
    Bare bones function where we process the deblended image
    '''
        
    segment_map_v2 = np.copy(segment_map.data)
    segment_map_v2_copy = np.copy(segment_map_v2)
    
    island_num = segment_map.data[int(fiber_ypix),int(fiber_xpix)]

    #if the island num is zero (on bkg), we get the closest one!
    closest_island_dist_pix = 0
    if island_num == 0:
        island_num, closest_island_dist_pix = find_nearest_island(segment_map_v2_copy, fiber_xpix,fiber_ypix)
    
    #pixels that are part of main blob are called 2
    segment_map_v2[segment_map_v2_copy == island_num] = 2
    #all other segments that are not background are called 1
    segment_map_v2[(segment_map_v2_copy != island_num) & (segment_map_v2_copy > 0)] = 1
    
     #make a copy of the deblended array, this is the one where even the main segment is split into different deblended components
    segm_deblend_v2 = np.copy(segm_deblend.data)
    #create an array of nans with same shape
    segm_deblend_v3 = np.zeros_like(segm_deblend.data) * np.nan
    
    #we will populating segm_deblend_v3 with the different segments that are part of main segment island 
    #get deblend segments ids that are part of the main segment island
    deblend_ids = np.unique(segm_deblend_v2[segment_map_v2 == 2])
    
    #create another deblend image where we relabel the deblend ids of the main segment
    #note the 0 pixel means background and hence we have i+1 here
    for i,di in enumerate(deblend_ids):
        #we skip the i=0 step as that is the background
        segm_deblend_v3[(segm_deblend_v2 == di)] = i+1

    #setting all the pixels that are in background to be zero.
    segm_deblend_v3[np.isnan(segm_deblend_v3)] = 0

    #what is the deblend id where our main source is in?
    deblend_island_num = segm_deblend_v3[int(fiber_ypix),int(fiber_xpix)]
    
    if deblend_island_num == 0:
        deblend_island_num, closest_island_dist_pix = find_nearest_island(segm_deblend_v3, fiber_xpix,fiber_ypix)

    #given this deblend island num, get the mask of just the parent galaxy of interest
    parent_galaxy_mask = (segm_deblend_v3 == deblend_island_num)

    #this is the mask that we think contains the parent galaxy
    #we will then only select sources that lie on this mask
    
    return parent_galaxy_mask.astype(int), closest_island_dist_pix

    
    

def get_simplest_photometry(rgb_img, r_rms, fiber_xpix, fiber_ypix, save_path, ncontrast_opt=None):
    '''
    This is the photometry method where we forego all color based association and bad pixel masking. In this step, we smooth the r-band image a lot, identify the main deblended blob that contains our galaxy of interest, get all the tractor sources on that blob, and sum their photometry up!!

    This is the simplest kind of photometry, purely based on spatial information, and is a good backup for photometry when multiple segments are detected in the COG aperture map or some sort of negative, downward trend in COG mags indicative of over subtraction! 
    
    Note that below a certain redshift, we will just ignore the deblending step for now, as we find it tends to still overshred. Based on whatever was the final photometry method, we will estimate the final aperture shape and see what fraction is outside the image. and construct a maskbit based on that.

    Note, in case the desi fiber does not line up with any segment, we consider the closest possible segment! And just make a note of this.
    '''

    #read the source catalog!
    source_cat = Table.read(save_path + "/source_cat_all_main_segment.fits")

    ##load the tractor model of the main segment!
    tractor_model_main_seg = np.load(save_path + "/tractor_main_segment_model.npy")
    r_band_tractor = tractor_model_main_seg[1]
    
    #these are the same parameters as the isolate galaxy mask!!
    kernel = make_2dgaussian_kernel(15, size=29) 

    #these are the same parameters as our original segmentation
    threshold_rms_scale = 1.5
    npixels_min =  10

    threshold = threshold_rms_scale * r_rms
    convolved_tot_data = convolve( r_band_tractor, kernel )
    
    segment_map = detect_sources(convolved_tot_data, threshold, npixels=npixels_min) 

    if segment_map is None or len(source_cat) == 0:        
        np.save(save_path + "/simplest_photometry_binary_mask.npy",  np.zeros_like(r_band_tractor ) )
        
        # np.save(save_path + "/simplest_photo_rgb_mask_image.npy", np.ones_like(data_arr) )
        
        return 3*[np.nan], np.nan, np.nan, np.zeros_like(rgb_img)

    else:

        if np.isnan(ncontrast_opt):
            #we do not do any deblending here and take it as as
            segment_map_data = segment_map.data

            island_num = segment_map_data[int(fiber_ypix),int(fiber_xpix)]
            closest_island_dist_pix = 0
            if island_num == 0:
                island_num, closest_island_dist_pix = find_nearest_island(segment_map_data, fiber_xpix,fiber_ypix)
    
            parent_mask = (segment_map_data == island_num)  
            parent_mask = parent_mask.astype(int)
        
        else:
            segm_deblend_opt = deblend_sources(convolved_tot_data,segment_map,
                                           npixels=10,nlevels=4, contrast=ncontrast_opt,
                                           progress_bar=False)


            parent_mask, closest_island_dist_pix = process_deblend_image(segment_map, segm_deblend_opt, fiber_xpix, fiber_ypix)



        #estimate the aperture of this simplest photo and measure its outside fraction!
        aperture_for_simple_phot, areafrac_in_image_simple_phot, _, _ = get_elliptical_aperture( parent_mask , sigma = 4 )

        #use this new segmentation mask to filter the tractor sources! and save their photometry!!!
        xpix_all = source_cat["xpix"].astype(int)
        ypix_all = source_cat["ypix"].astype(int)
        
        # mask of which sources fall on a segment (nonzero)
        on_segment_mask = (parent_mask[ypix_all, xpix_all] != 0)    
        source_target = (source_cat["separations"].data  == np.min(source_cat["separations"].data) )
    
        # filter catalog
        source_cat_simple_model = source_cat[on_segment_mask | source_target]
    
        #save this catalog!
        source_cat_simple_model.write( save_path + "/simplest_photometry_parent_sources.fits", overwrite=True)

        #let us save this mask
        np.save(save_path + "/simplest_photometry_binary_mask.npy", parent_mask)
    
        #let us make a simple rgb image, where we apply this mask to it!
        fig,ax = make_subplots(ncol = 2, nrow = 1, return_fig = True)
        
        ax[0].set_title("Full grz image",fontsize = 15)
        ax[0].imshow(rgb_img, origin="lower")

        #make the masked image
        tractor_model_main_seg[:, (parent_mask == 0)] = 0
        rgb_trac_parent = sdss_rgb(tractor_model_main_seg)

        ax[1].set_title("Simplest Photo grz image",fontsize = 15)
        ax[1].imshow(rgb_trac_parent, origin="lower")

        for axi in ax:
            axi.set_xticks([])
            axi.set_yticks([])
        fig.savefig(save_path + "/simplest_photometry_mask_image.png")
        plt.close(fig)
    
        #we will make a tractor model image with this sources later!
    
        #in the mean time, measure the simplest photometry!
        g_flux_corr = mags_to_flux(source_cat_simple_model["mag_g"]) / source_cat_simple_model["mw_transmission_g"]
        r_flux_corr = mags_to_flux(source_cat_simple_model["mag_r"]) / source_cat_simple_model["mw_transmission_r"]
        z_flux_corr = mags_to_flux(source_cat_simple_model["mag_z"]) / source_cat_simple_model["mw_transmission_z"]
        
        tot_g_mag = flux_to_mag(np.sum(g_flux_corr))
        tot_r_mag = flux_to_mag(np.sum(r_flux_corr))
        tot_z_mag = flux_to_mag(np.sum(z_flux_corr))
        
        return np.array([tot_g_mag, tot_r_mag, tot_z_mag]), closest_island_dist_pix,  areafrac_in_image_simple_phot, rgb_trac_parent
        
        
    
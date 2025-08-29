import numpy as np
import scarlet
import astropy.io.fits as fits
import sep
import pickle
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import astropy.units as u
from scarlet.detect_pybind11 import get_footprints
from mpl_toolkits.axes_grid1 import make_axes_locatable
# Define normalization
from astropy.visualization import (MinMaxInterval, AsinhStretch, ImageNormalize)
from photutils.background import Background2D, MedianBackground
from scarlet.display import AsinhMapping
from photutils.aperture import aperture_photometry
import os
import sys
import joblib
import scipy.optimize as opt
from astropy import units as u
from astropy.coordinates import SkyCoord
import random
import argparse
import requests
from astropy.io import fits
from astropy.wcs import WCS
from astropy.table import Table, vstack, join
import multiprocessing as mp
from tqdm import tqdm
url_prefix = 'https://www.legacysurvey.org/viewer/'
from io import BytesIO
import matplotlib.patches as patches
from photutils.segmentation import make_2dgaussian_kernel
from astropy.convolution import convolve
from reproject import reproject_interp
from desi_lowz_funcs import save_subimage, fetch_psf
from desi_lowz_funcs import print_stage, check_path_existence, get_remove_flag, _n_or_more_lt, is_target_in_south, match_c_to_catalog, calc_normalized_dist, get_sweep_filename, get_random_markers, save_table, make_subplots, sdss_rgb
from photutils.background import StdBackgroundRMS, MADStdBackgroundRMS
from astropy.stats import SigmaClip
from matplotlib.colors import Normalize
from tqdm import trange

def mask_radius_for_mag(mag):
    # Returns a masking radius in degrees for a star of the given magnitude.
    # Used for Tycho-2 and Gaia stars.
    
    # This is in degrees, and is from Rongpu in the thread [decam-chatter 12099].
    return 1630./3600. * 1.396**(-mag)

# def mask_star(invar_c, center, radius,mask_val = 0):
#     """
#     Masks pixels within a given radius from a center coordinate by setting them to np.inf.
    
#     Parameters:
#     - invar: (3, H, W) NumPy array
#     - center: (x, y) tuple specifying the pixel coordinate
#     - radius: Radius of the region to mask
    
#     Returns:
#     - Modified array with masked region
#     """

#     invar = np.copy(invar_c)

#     H, W = invar.shape[1:]  # Get height and width
#     x0, y0 = center  # Center coordinates

#     # Create a distance grid
#     y, x = np.ogrid[:H, :W]
#     mask = (x - x0)**2 + (y - y0)**2 <= radius**2  # Boolean mask of pixels within radius

#     # Apply the mask to all channels
#     invar[:, mask] = mask_val #np.inf  # Assign np.inf
    
#     return invar

def mask_star(arr, center, radius, mask_val=0):
    """
    Masks pixels within a given radius from a center coordinate.

    Parameters:
    - arr: NumPy array of shape (H, W) or (C, H, W)
    - center: (x, y) tuple specifying the pixel coordinate
    - radius: Radius of the region to mask
    - mask_val: Value to assign to the masked pixels (default: 0)

    Returns:
    - Modified array with masked region
    """

    arr = np.copy(arr)
    
    # Determine if input is 2D or 3D
    if arr.ndim == 2:
        H, W = arr.shape
        is_3d = False
    elif arr.ndim == 3:
        C, H, W = arr.shape
        is_3d = True
    else:
        raise ValueError("Input array must be 2D or 3D with shape (C, H, W)")

    x0, y0 = center

    # Create a distance grid
    y, x = np.ogrid[:H, :W]
    mask = (x - x0)**2 + (y - y0)**2 <= radius**2  # Boolean mask of pixels within radius

    # Apply the mask
    if is_3d:
        arr[:, mask] = mask_val
    else:
        arr[mask] = mask_val

    return arr


def conf_interval(x, pdf, conf_level):
    return np.sum(pdf[pdf > x])-conf_level


def plot_colorbar(fig, ax, im):
    '''
    Function that plots colorbar for a given subplot
    '''
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)  # Keep colorbars the same size
    # Add colorbar
    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label("Flux")
    return

 
def add_comps_back(scar_comp_wlsb_nostars, scar_comp_stars, center_star_inds):
    '''
    This is the function that adds the removed star components back to the same location so the order of the centers and scar_comps is consistent!!
    Note that it is being added back to a modified scar comp list as the LSB component has been added to the end

    scar_comp_wlsb_nostars : list of scarlet component that has all extended components with the LSB component tacked on at the end
    scar_comp_stars : list of ONLY STAR scarlet components that need to be inserted into the scar_comp_wlsb_nostars at appropriate locations
    center_star_inds: the indices in the centers_arr that corresponding to star sources!
    
    '''

    star_items = [(idx, scar_comp_stars[i]) for i,idx in enumerate(center_star_inds) ]
    
    # Reinsert removed elements in sorted order
    for idx, star_comp in sorted(star_items):
        scar_comp_wlsb_nostars = np.insert(scar_comp_wlsb_nostars, idx, star_comp)

    ##this list now contains the stars components even though the variable name might not suggest :)
    return scar_comp_wlsb_nostars 

def transform_coords(all_x, all_y, xmin, xmax, ymin, ymax ):
    '''
    My initial image is 350x350. I am cutting a box around region of interest and then transforming pixel-coordinates
    accordingly. This is becausse the wcs is anchored to the entire 350x350 image
    
    all_x : x-coords of points in original 350x350 box
    all_y : y-coords of points in original 350x350 box
    
    xmin,xmax,ymin,ymax : corners of the rectangle of interest
    
    Think about: does making the box not square make it bad for plotting?
    
    '''
    
    W = np.abs(xmax - xmin)
    H = np.abs(ymin - ymax)
    
    all_x_new = all_x - xmin
    all_y_new = all_y - ymin
    
    #we will use the 0<x<W and 0<y<H constraints to filters sources outside the box! 
    
    return all_x_new, all_y_new, W, H
    
    

def embed_psf(old_psf):
    '''
    Function that will embed a 31x31 array into a 63x63 array
    '''
    # Create a 63x63 array filled with zeros (or any other background value)
    large_psf = np.zeros((63,63))
    # Compute the start indices for centering
    start_idx = (63 - 31) // 2  # This gives 16
    end_idx = start_idx + 31  # This gives 47
    # Place the small array in the center of the large array
    large_psf[start_idx:end_idx, start_idx:end_idx] = old_psf
    
    return large_psf

# Assuming you have:
def combine_sbimgs(sbimgs_hdus, data_wcs_2d):
    '''
    Function to combine the sub-images onto the common data shape

    I just have to project these sub-images into a common 350x350 grid and then add them up! 
    I can add inverse variances directly to get co-added inverse variance map
    
    '''
    data_shape = (350, 350)
    
    sbimgs_len = len(sbimgs_hdus) - 1

    #number of things to combine
    num_comb = int(sbimgs_len/6)

    final_inv_maps = []
    
    for bi in range(len((["g","r","z"]))):
        #Step 1: Store the individual inverse variance maps and associated WCS into lists
        chunks = []
        chunks_wcs = []
        for ni in range(num_comb):
            #accesing the ni'th small part from the bi'th band
            rel_hdu = sbimgs_hdus[2 + 2*bi + 6*ni]
            
            chunks.append( rel_hdu.data )
            
            chunks_wcs.append( WCS(rel_hdu.header ) )

        all_reprojs = []
        # Step 2: Reproject each image
        for j in range(num_comb):
            chunks_j = chunks[j]
            chunks_wcs_j = chunks_wcs[j]
            
            # Reproject to the target WCS
            reproj_small, _ = reproject_interp((chunks_j, chunks_wcs_j), data_wcs_2d, shape_out=data_shape)

            #replacing all the nan pixels with 0 value!

            reproj_small[np.isnan(reproj_small)] = 0
            
            all_reprojs.append(reproj_small)


        # Step 3: Combine/Sum all these reprojected maps to produce the final inverse variance map
        final_map  = np.sum( np.array(all_reprojs), axis = 0)

        final_inv_maps.append(final_map)

    return np.array(final_inv_maps)



def multi_panel_image(grz_img, model_img, residual_img, save_img_path):
    '''
    Function that makes a multi-panel image 
    '''

    fig,ax = make_subplots(ncol =3, nrow =1, col_spacing = 0.75, return_fig=True)

    grz_rgb = sdss_rgb(grz_img)
    model_rgb = sdss_rgb(model_img)
    resi_rgb = sdss_rgb(residual_img)

    ax[0].imshow(grz_rgb,origin="lower")
    ax[1].imshow(model_rgb,origin="lower")
    ax[2].imshow(resi_rgb,origin="lower")
    
    for axi in ax:
        axi.set_xticks([])
        axi.set_yticks([])
    
    plt.savefig(save_img_path,bbox_inches="tight")
    plt.close()

    
def run_scarlet_pipe(input_dict):
    '''
    TO DO:
    1) Use other redshifts in cutout to help choosing which sources are part of dwarf or not
    
    Issues appear to be avoided if stars are removed hehehe. Can mask out those pixels ... 

    In the fitting, I could potentially mask out other sources that are not being "blended in?" I could just do this via the weights array?
    In other words, can I get away with masking out sources that lie outside the elliptical aperture 

    '''
    
    save_path = input_dict["save_path"]
    source_tgid  = input_dict["tgid"]
    source_ra  = input_dict["ra"] 
    source_dec  = input_dict["dec"]
    source_redshift  = input_dict["redshift"]
    wcs  = input_dict["wcs"]
    #data_arr is the C x N x N matrix with image data
    data_arr  = input_dict["image_data"]
    invar_weights = input_dict["invvar_data"]
    source_cat_f = input_dict["source_cat"]
    print(len(source_cat_f))
    ## update this to remove duplicate stars!!
    # extract gaia ids of the stars
    signi_pm = ( np.abs(source_cat_f["pmra"]) * np.sqrt(source_cat_f["pmra_ivar"]) > 3) | ( np.abs(source_cat_f["pmdec"]) * np.sqrt(source_cat_f["pmdec_ivar"]) > 3)
    is_star = (source_cat_f["ref_cat"] == "G2")  &  ( 
    (source_cat_f['type'] == "PSF")  | signi_pm )
    star_gaia_ids = np.array(source_cat_f["ref_id"])[is_star]
    # find the *first* index for each unique gaia_id in the star subset
    _, keep_idx_in_subset = np.unique(star_gaia_ids, return_index=True)
    # convert those subset indices back into indices in the full table
    star_indices = np.where(is_star)[0]
    keep_indices = star_indices[keep_idx_in_subset]
    # now build a new full-mask where duplicate-gaia stars are dropped
    final_mask = np.ones(len(source_cat_f), dtype=bool)
    # drop everything in (star_indices - keep_indices)
    drop_indices = np.setdiff1d(star_indices, keep_indices)
    final_mask[drop_indices] = False
    # optionally create the filtered table
    source_cat_f = source_cat_f[final_mask]

    print(len(source_cat_f))

    ######
    
    overwrite = input_dict["overwrite"]
    #do we run our own peak finder, or use LS DR9 sources as peaks?
    run_own_detect = input_dict["run_own_detect"]
    box_size = input_dict["box_size"]

    ## load the newly computed aperture mags for plotting
    aper_mag_g = input_dict["MAG_G_APERTURE_COG"]
    aper_mag_r = input_dict["MAG_R_APERTURE_COG"]
    aper_mag_z = input_dict["MAG_Z_APERTURE_COG"]

    aper_xcen = input_dict["APER_XCEN"]
    aper_ycen = input_dict["APER_YCEN"]
    aper_semi_maj = input_dict["APER_SEMI_MAJ"]

    aper_ba = input_dict["APER_BA"]
    aper_theta = input_dict["APER_THETA"]


    use_main_blob = False
    print("USE MAIN BLOB IS", use_main_blob)

    verbose=True

    if verbose:
        print(save_path)    
    
    ##load the psf now

    if os.path.exists(save_path +"/psf_data_z.npy"):
        pass
    else:
        print("Psfs did not exixt before and are being downloaded!")
        with requests.Session() as session:
            psf_dict = fetch_psf(source_ra,source_dec, session)
            
        if psf_dict is not None:
            #as psf can be of different sizes and so saving each band separately!
            np.save(save_path  + "/psf_data_g.npy",psf_dict["g"])
            np.save(save_path  + "/psf_data_r.npy",psf_dict["r"])
            np.save(save_path  + "/psf_data_z.npy",psf_dict["z"])
        else:
            print("Psfs could not be downloaded :(")
            return [np.nan,np.nan,np.nan], [np.nan,np.nan, np.nan], ""

    psf_g = np.load(save_path +"/psf_data_g.npy")
    psf_r = np.load(save_path +"/psf_data_r.npy")
    psf_z = np.load(save_path +"/psf_data_z.npy")

    old_psfs = { "g": psf_g, "r": psf_r, "z": psf_z }
    new_psfs = { "g": psf_g, "r": psf_r, "z": psf_z }
            
    psf = np.array( [ new_psfs["g"], new_psfs["r"], new_psfs["z"] ] )


    if np.shape(invar_weights) != np.shape(data_arr):
        print(np.shape(invar_weights),  np.shape(data_arr) )
        print(save_path)
        raise ValueError("invariance maps and data arrays not of same size!")
        
    if psf is None:
        #try download the psf again

        return [np.nan,np.nan,np.nan], [np.nan,np.nan, np.nan], ""
    else:
        
        #fontsize for all the xlabels in titles in plots
        fontsize = 15
        
        ########################################################
        ##IDENTIFY THE MAIN ISLAND OF SOURCE TO DRAW BOX AROUND
        ########################################################
                
        ## we do not want a very large box as it unnecessary more computation :)
        #we will define a rectangular/square box that encapsulates the whole R4 aperture times some factor! 
        
         ## plot these sources on an image segmentation map to find the scarlet components associated with the dwarf

        if use_main_blob:
            image_tot = np.sum(data_arr,axis=0)
    
            noise_dict = {}
            rms_estimator = MADStdBackgroundRMS()
            ##estimate the background rms in each band!
            for filter_ind, bii in enumerate(["g","r","z"]):        
                # Apply sigma clipping
                sigma_clip = SigmaClip(sigma=3.0,maxiters=5)
                clipped_data = sigma_clip(data_arr[filter_ind])
                # Estimate RMS
                background_rms = rms_estimator(clipped_data)
                noise_dict[bii] = background_rms
    
        
            #the rms in the total image will be the rms in the 3 bands added in quadrature
            tot_rms = np.sqrt( noise_dict["g"]**2 + noise_dict["r"]**2 + noise_dict["z"]**2 )
            
            threshold = 1.5 * tot_rms
            
            ##do image segmentation per band image
            npixels_min = 10
            from photutils.segmentation import detect_sources
    
            #load in the R4 aperture and its co-ordinate and size!
    
            #convolving the data
            kernel = make_2dgaussian_kernel(3.0, size=5)  # FWHM = 3.0
            convolved_image_tot = convolve( image_tot, kernel )
    
            segment_map = detect_sources(convolved_image_tot, threshold, npixels=npixels_min) 
            
            #this also includes the zero, we will remove it later
            all_segment_nums = np.unique(segment_map.data)
                
            ## find the sources that lie on the main segment
            #get the segment number where the main source of interest lies in
            fiber_xpix, fiber_ypix,_ = wcs.all_world2pix(source_ra, source_dec,0,1)
                
            island_num = segment_map.data[int(fiber_ypix),int(fiber_xpix)]
                
            #getting the ids of the other sources that are not the main island!
            all_segment_nums_notmain = all_segment_nums[ (all_segment_nums != island_num) & (all_segment_nums != 0)]
                
            #any segment not part of this main segment is considered to be a different source 
            #make a copy of the segment array
            segment_map_v2 = np.copy(segment_map.data)
        
            #pixels that are part of main segment island are called 2
            segment_map_v2[segment_map.data == island_num] = 2
            #all other segments that are not background are called 1
            segment_map_v2[(segment_map.data != island_num) & (segment_map.data > 0)] = 1
            #rest all remains 0
    
            plt.figure(figsize = (5,5))
            plt.imshow(segment_map_v2,cmap="viridis",origin="lower")
            plt.scatter( [fiber_xpix], [fiber_ypix],color = "r", marker = "x",s=50)
            plt.savefig(save_path + "/scarlet_segment_map_v2.pdf",bbox_inches="tight")
            plt.close()
                
            #get the bounding box
            scale_fac = 2
            x_min_new = int(aper_xcen - int(scale_fac*2*aper_semi_maj))
            x_max_new = int(aper_xcen + int(scale_fac*2*aper_semi_maj))
            y_min_new = int(aper_ycen - int(scale_fac*2*aper_semi_maj))
            y_max_new = int(aper_ycen + int(scale_fac*2*aper_semi_maj))
    
            aper_xcen_local = aper_xcen - x_min_new
            aper_ycen_local = aper_ycen - y_min_new
    
            #draw a box showing this             
            
            fig,ax = plt.subplots(figsize = (7,7))
            
            ##plot the rgb image of the data
            img_rgb =  sdss_rgb([data_arr[0],data_arr[1],data_arr[2]], ["g","r","z"], scales=dict(g=(2,6.0), r=(1,3.4), z=(0,2.2)), m=0.03)
            
            rect = patches.Rectangle(
                    (x_min_new - 0.5, y_min_new - 0.5),  # Rectangle lower-left corner
                    x_max_new - x_min_new + 1,  # Width
                    y_max_new - y_min_new + 1,  # Height
                    linewidth=2, edgecolor='lime', facecolor='none' )
    
            # Add the rectangle to the plot
            ax.add_patch(rect)
    
            ax.imshow(img_rgb,origin="lower")
            fig.savefig(save_path + "/bounding_box.pdf",bbox_inches="tight")
            plt.close(fig)
    
            #update the arrays as well that we will be using
            data_arr = data_arr[:, y_min_new : y_max_new, x_min_new : x_max_new ]
            invar_weights = invar_weights[ :, y_min_new : y_max_new, x_min_new : x_max_new  ]
    
            ##we will save hte bounding box co-ordinates for later use!
            np.save(save_path + "/bounding_box_coords.npy",[y_min_new, y_max_new, x_min_new, x_max_new]  )
            
            image_tot = np.sum(data_arr,axis=0)
            segment_map_v2 = segment_map_v2[y_min_new : y_max_new, x_min_new : x_max_new]
            segment_data_new = segment_map.data[y_min_new : y_max_new, x_min_new : x_max_new]
            
            #updating the segment nums
            all_segment_nums = np.unique(segment_data_new)
            #getting the ids of the other sources that are not the main island!
            all_segment_nums_notmain = all_segment_nums[ (all_segment_nums != island_num) & (all_segment_nums != 0)]
            
            sources_f_xpix,sources_f_ypix,_ = wcs.all_world2pix(source_cat_f['ra'].data, source_cat_f['dec'].data, 0,1)    
            sources_f_xpix,sources_f_ypix,W,H = transform_coords(sources_f_xpix,sources_f_ypix, x_min_new, x_max_new, y_min_new, y_max_new )
    
                    
            box_mask = (sources_f_xpix >= 0) & (sources_f_xpix <= W) & (sources_f_ypix >= 0) & (sources_f_ypix <= H)
            
            #only keep the sources that lie in the box
            source_cat_f = source_cat_f[box_mask]
            sources_f_xpix = sources_f_xpix[box_mask]
            sources_f_ypix = sources_f_ypix[box_mask]
        
        else:
            ##IN THIS CASE, WE JUST MODEL THE FULL IMAGE AND JUST DIFFERENTIATE BASED ON COLOR
            ##no main blob is used to identify the main segment
            new_cutout_size = 300
            
            #we use the DESI source as the center and just obtain some square cutout
            x_min_new = int(aper_xcen - int(new_cutout_size))
            x_max_new = int(aper_xcen + int(new_cutout_size))
            y_min_new = int(aper_ycen - int(new_cutout_size))
            y_max_new = int(aper_ycen + int(new_cutout_size))
    
            aper_xcen_local = aper_xcen - x_min_new
            aper_ycen_local = aper_ycen - y_min_new

            ##plot the rgb image of the data
            fig,ax = plt.subplots(figsize = (7,7))
            img_rgb =  sdss_rgb([data_arr[0],data_arr[1],data_arr[2]], ["g","r","z"], scales=dict(g=(2,6.0), r=(1,3.4), z=(0,2.2)), m=0.03)
            rect = patches.Rectangle(
                    (x_min_new - 0.5, y_min_new - 0.5),  # Rectangle lower-left corner
                    x_max_new - x_min_new + 1,  # Width
                    y_max_new - y_min_new + 1,  # Height
                    linewidth=2, edgecolor='lime', facecolor='none' )
            # Add the rectangle to the plot
            ax.add_patch(rect)
            ax.imshow(img_rgb,origin="lower")
            fig.savefig(save_path + "/bounding_box.pdf",bbox_inches="tight")
            plt.close(fig)

             ##we will save hte bounding box co-ordinates for later use!
            np.save(save_path + "/bounding_box_coords.npy",[y_min_new, y_max_new, x_min_new, x_max_new]  )

            #update the arrays as well that we will be using
            data_arr = data_arr[:, y_min_new : y_max_new, x_min_new : x_max_new ]
            invar_weights = invar_weights[ :, y_min_new : y_max_new, x_min_new : x_max_new  ]

            #the whole image is part of the main blob
            segment_map_v2 = np.ones_like(data_arr[0]) * 2

            all_segment_nums = np.array([2])
            #getting the ids of the other sources that are not the main island!
            all_segment_nums_notmain = np.array([])

            sources_f_xpix,sources_f_ypix,_ = wcs.all_world2pix(source_cat_f['ra'].data, source_cat_f['dec'].data, 0,1)    
            sources_f_xpix,sources_f_ypix,W,H = transform_coords(sources_f_xpix,sources_f_ypix, x_min_new, x_max_new, y_min_new, y_max_new )
    
            box_mask = (sources_f_xpix >= 0) & (sources_f_xpix <= W) & (sources_f_ypix >= 0) & (sources_f_ypix <= H)
            
            #only keep the sources that lie in the box
            source_cat_f = source_cat_f[box_mask]
            sources_f_xpix = sources_f_xpix[box_mask]
            sources_f_ypix = sources_f_ypix[box_mask]

        
        ##identifying the stars in the footprint
        ##these will be the stars that will be modeled as a psf source!
        #should I also include a magnitude limit as we want to specifically model the bright ones?

        gaia_max_mag = np.min(  np.array([source_cat_f["gaia_phot_bp_mean_mag"].data, source_cat_f["gaia_phot_rp_mean_mag"].data, source_cat_f["gaia_phot_g_mean_mag"].data]) ,axis=0) 

        ##procedure for selecting a star
        signi_pm = ( np.abs(source_cat_f["pmra"]) * np.sqrt(source_cat_f["pmra_ivar"]) > 3) | ( np.abs(source_cat_f["pmdec"]) * np.sqrt(source_cat_f["pmdec_ivar"]) > 3)
        
        is_star = (source_cat_f["ref_cat"] == "G2")  &  ( 
            (source_cat_f['type'] == "PSF")  | signi_pm 
        ) 

        
        #what is the distance to the closest star from us?
        all_stars = source_cat_f[is_star]

        print("Number of stars", len(all_stars))
        
        if len(all_stars) > 0:
            ref_coord = SkyCoord(ra=source_ra * u.deg, dec=source_dec * u.deg)
            catalog_coords = SkyCoord(ra=all_stars["ra"].data * u.deg, dec=all_stars["dec"].data * u.deg)
            # Compute separations
            separations = ref_coord.separation(catalog_coords).arcsec
            closest_star_dist = np.min(separations)
        else:
            closest_star_dist = -99

        ## the axes object for our main plot!!
        fig_tot, ax = make_subplots(ncol = 5, nrow = 2, row_spacing = 0.5,col_spacing=0.5, label_font_size = fontsize,plot_size = 4,direction = "horizontal", return_fig = True)
        
        #making a RGB image with a given norm 
        stretch = 0.1 #1
        Q = 5
        norm = AsinhMapping(minimum=0, stretch=stretch, Q=Q)
        img_rgb = scarlet.display.img_to_rgb(data_arr, norm=norm)
        
        ##IDENTIFYING THE SOURCE PEAKS IN THE IMAGE

        if verbose:
            print("Identifying source peaks in the image using wavelets")

        ax[0].imshow(img_rgb)

        try:
            # Create a detection image by summing the images in all bands
            # (a more rigorous approach would be to create a chi^2 coadd).
            detect_image = np.sum(data_arr, axis=0)
            # Define a rough standard deviation for the image.
            # This does not have to be exact, as it is fit by the
            # get_multiresolution_support algorithm below.
            sigma = 0.1
            # Find the wavelet coefficients
            coeffs = scarlet.wavelet.starlet_transform(detect_image, scales=3)
            # Determine the significant coefficients
            # (as defined in Starck et. al 2011)
            M = scarlet.wavelet.get_multiresolution_support(detect_image, coeffs, sigma, K=3, epsilon=1e-1, max_iter=20)
            # Use the multi-resolution support to select only
            # the relevant pixels for detection
            detect = M * coeffs
            # We are only detecting positive peaks, so there
            # is no need to include the negative coefficients
            detect[detect<0] = 0
            
            # Calculate isolated footprints and their maxima
            # in the 2nd wavelet scale.
            
            #what is a reasonable min_separation to choose here??
            footprints = get_footprints(detect[1], min_separation=7, min_area=10, thresh=0)
            
            # Display all of the footprints
            footprint_img = np.zeros(detect.shape[1:])
            peaks = []
            for fp in footprints:
                bbox = scarlet.detect.bounds_to_bbox(fp.bounds)
                footprint_img[bbox.slices] = fp.footprint
                peaks += list(fp.peaks)
            
            # Now display the peaks on the original image
            ax[0].contour(footprint_img, [0.5,], colors='w', linewidths=0.5)
        except:
            print("Scarlet: Extracting sources using wavelets failed, so using DR9 source locations")
            run_own_detect = False
        
        if run_own_detect:
            for k, peak in enumerate(peaks):
                ax[0].text(peak.x, peak.y, str(k), color="w", ha='center', va='center', weight="bold", size=8)
    
        ax[0].set_xlim([0,W])
        ax[0].set_ylim([0,H])

        if verbose:
            print("Identifying the source centers")
                
        if run_own_detect == True:
            #if running own peak finding algorithm   
            centers = [(peak.y, peak.x) for peak in peaks]
        else:
            #running on significant sources in LS DR9

            stacked_mags = np.vstack([source_cat_f["mag_g"].data, source_cat_f["mag_r"].data, source_cat_f["mag_z"].data])  # Shape (3, N)
            max_mags = np.nanmin(stacked_mags, axis=0)
            sources_f_xpix_f =  sources_f_xpix[ max_mags < 23 ]
            sources_f_ypix_f = sources_f_ypix[ max_mags < 23 ]
            
            ## we do not run our own detection, but rather use the LS DR9 sources as the centers!
            centers = [ (sources_f_ypix_f[i],sources_f_xpix_f[i]) for i in range(len(sources_f_xpix_f))  ]
            
            ax[0].scatter( sources_f_xpix_f, sources_f_ypix_f, color = "w", marker = "^",s = 15)
            
            ##these sources are labelled 0,1,2,3, ...
            
            #plotting the labels for the sources!
            for k, peak in enumerate(centers):
                ax[0].text(peak[1]*1.02, peak[0]*1.02,  str(k), color="w", ha='center', va='center', weight="bold", size=8)



        ##note that in centers the first axis is already y and second axis is already x
        centers_arr = np.array(centers)

        if verbose:
            print(np.max(centers_arr,axis=0))
            print(np.max(centers_arr,axis=1))
            print(np.shape(data_arr))


        ## GET A MASK FOR WHETHER THE SCARLET FOOTPRINTS ARE ON THE IDENTIFIED SEGMENT OR NOT!
        
        
        
        if np.sum(is_star) > 0:
            #if there exists a star!!
            
            #given the "centers", find the centers closest to stars (hopefully within 5 pixels an)
            star_f_xpix = sources_f_xpix[is_star]
            star_f_ypix = sources_f_ypix[is_star]

            source_cat_is_star = source_cat_f[is_star]
            
            source_star_max_mags = np.nanmin( (  np.array(source_cat_is_star["gaia_phot_bp_mean_mag"]), np.array(source_cat_is_star["gaia_phot_rp_mean_mag"]),np.array(source_cat_is_star["gaia_phot_g_mean_mag"])    )       ,axis = 0)
            
        
            ax[0].scatter( star_f_xpix, star_f_ypix, color = "w", marker = "*",s = 30)

            print("STAR_F_XPIX ", star_f_xpix)
            print("STAR_F_YPIX ", star_f_ypix)
            
            # print(len(sources_f_xpix), len(source_cat_f), len(centers))
            ## what should the order be here ... ?
            all_stars = np.array([star_f_ypix, star_f_xpix]).T
            
            all_cens = np.array([ centers_arr[:,0], centers_arr[:,1] ]).T

            #UPDATE CODE
            # Compute pairwise distances between all stars and all centers:
            #   distances[i, j] = distance from star i to center j
            distances = np.linalg.norm(all_stars[:, None, :] - all_cens[None, :, :], axis=-1)
            
            # Mask out distances >= 15
            distances[distances >= 15] = np.inf
            
            # Now, for each star -> find its closest center
            closest_center_for_star = np.argmin(distances, axis=1)
            min_distance_for_star   = np.min(distances, axis=1)
            
            # Filter valid star/center matches
            valid = min_distance_for_star < np.inf
            center_star_inds = closest_center_for_star[valid]
            centers_stars = all_cens[center_star_inds]
                        
            # # Compute pairwise distances between all points
            # distances = np.linalg.norm(all_stars[:, None, :] - all_cens[None, :, :], axis=-1)
            
            # # Find closest center that is within 5 pixels of a star
            # # Mask out distances >= 5
            # distances[distances >= 15] = np.inf

            # # Find the closest valid match
            # closest_indices = np.argmin(distances, axis=0)
            # min_distances = np.min(distances, axis=0)

            
            # # Filter valid matches
            # valid_matches = min_distances < np.inf
            # centers_stars = all_cens[valid_matches]
            # #I need the indices of these centers so can replace sources with psf
            # center_star_inds = np.where(valid_matches)[0]

            #centers_stars are the centers that are stars and thus should be fit as stars
            if len(center_star_inds) != len(star_f_xpix):
                print(len(centers_stars),len(center_star_inds), len(star_f_xpix))
                raise ValueError( "All stars are not matched to a center!" )
                

        ##I want to make a separate plot just showing the identified peaks!
        fig_detect, axd = make_subplots(ncol = 1, nrow = 1, label_font_size = fontsize,plot_size = 4, return_fig = True)
        
        axd[0].imshow(img_rgb,origin="lower")
        axd[0].contour(footprint_img, [0.5,], colors='w', linewidths=0.5,origin="lower")
  
        for k, peak in enumerate(peaks):
            axd[0].text(peak.x, peak.y, str(k), color="w", ha='center', va='center', weight="bold", size=8)
        axd[0].set_xlim([0,W])
        axd[0].set_ylim([0,H])
        if np.sum(is_star) > 0:
            #if there exists a star!!
            
            #given the "centers", find the centers closest to stars (hopefully within 5 pixels an)
            star_f_xpix = sources_f_xpix[is_star]
            star_f_ypix = sources_f_ypix[is_star]

            star_f_mags = sources_f_xpix[is_star]

            axd[0].scatter( star_f_xpix, star_f_ypix, color = "w", marker = "*",s = 30)
            axd[0].scatter( star_f_xpix, star_f_ypix, edgecolor = "r", facecolor= "none", marker = "o",s = 600,lw = 2)
            
        fig_detect.savefig(save_path + "/wavelet_source_detect.png",bbox_inches="tight")
        
        #########################################
        #########################################
        #RUN THE SCARLET OPTIMIZER TO FIT SOURCES
        #Note that if a star is saturated, there is no hope of fitting it with a psf. Easiest solution is just to mask it ... 
        #########################################
        #########################################

    
        filters = ["g","r","z"]
        ##defining the model frame
        model_psf = scarlet.GaussianPSF(sigma=(0.8,)*len(filters))

        model_frame = scarlet.Frame(
            data_arr.shape,
            psf=model_psf,
            channels=filters)

        #-------
        #Step 1: First run scarlet masking all the saturated stars so as to better constrain the LSB component
        #-------

        ##plotting the r-band inverse variance mask before doing any star masking 
        fig_invar0,ax_invar0 = make_subplots(ncol =1, nrow = 1, return_fig = True)
        ax_invar0[0].imshow(invar_weights[1],origin="lower")
        # ax_invar0[0].set_xticks([])
        # ax_invar0[0].set_yticks([])
        fig_invar0.savefig(save_path + "/invar_rband_prestar.png",bbox_inches="tight")
        
        #if there are stars, make the pixels in some region around the stars with weights = 0
        #the region we define is going to be a 3 arcsec region (change this size?)
        #we only masking stars if they are saturated
            
        if np.sum(is_star) > 0:
            print("Starting star masking!")

            invar_weights_starmask = invar_weights.copy()
            
            for si in range(np.sum(is_star)):

                ##use rongpu stellar radius here!
                star_mag = source_star_max_mags[si]
                print(star_mag)
                radius_pix = 3600*mask_radius_for_mag(star_mag)/0.262

                print(si, star_mag, star_f_xpix[si], star_f_ypix[si], radius_pix)
                
                #in general, need to be very careful when copying and modifying arrays
                invar_weights_starmask = mask_star(invar_weights_starmask, ( star_f_xpix[si], star_f_ypix[si]) , radius_pix )
                ## try to set weights/data to nan
        else:
            invar_weights_starmask = invar_weights

        print("Making an external source mask!")
        invar_weights_bkgmask = invar_weights.copy()
        
        #the pixels we will mask is where segment map = 1
        #could binary grow these masks a bit more if wanted to do some more masking!
        invar_weights_bkgmask[:, segment_map_v2 == 1] = 0
        #we want to the bkg mask to be both stars and bkg sources!!
        invar_weights_bkgmask[invar_weights_starmask == 0] = 0
        
        ##plotting the r-band inverse variance mask after doing the star masking 
        fig_invar1,ax_invar1 = make_subplots(ncol =1, nrow = 1, return_fig = True)
        ax_invar1[0].imshow(invar_weights_starmask[1],origin="lower")
        fig_invar1.savefig(save_path + "/invar_rband_poststar.png",bbox_inches="tight")

        fig_invar2,ax_invar2 = make_subplots(ncol =1, nrow = 1, return_fig = True)
        ax_invar2[0].imshow(invar_weights_bkgmask[1],origin="lower")
        fig_invar2.savefig(save_path + "/invar_rband_bkgmask.png",bbox_inches="tight")

        print("Fitting only extended sources, no stars and no lsbs")
        
        ##making the observation frame 
        observation_nostar = scarlet.Observation(
            data_arr,
            psf=scarlet.ImagePSF(psf),
            weights = invar_weights_starmask,
            channels=filters).match(model_frame)

        #we want to update the fact that we are dropping sources on the stellar footprints!
        #could we use the stellar footprint as a mask instead of just a circular mask?
        #center_star_inds are the index location of identified stars in the centers list
        if np.sum(is_star) > 0: 
            nostars_mask = ~np.isin(np.arange(len(centers_arr)), center_star_inds)
            centers_arr_nostars = centers_arr[nostars_mask]
        else:
            centers_arr_nostars = centers_arr


        ##where is the observation frame being used in the scarlet component object??
        scar_comps_nostars, skipped = scarlet.initialization.init_all_sources(model_frame,
                                                               centers_arr_nostars,
                                                               observation_nostar,
                                                               max_components=1,
                                                               min_snr=50,
                                                               thresh=1,
                                                               fallback=True,
                                                               silent=False,
                                                               set_spectra=True
                                                              )

        scarlet.initialization.set_spectra_to_match(scar_comps_nostars, observation_nostar)
    
        #run the fitting optimizer
        blend = scarlet.Blend(scar_comps_nostars, observation_nostar)
        #should I make this lower?
        it, logL = blend.fit(200, e_rel=1e-8)
        print(f"scarlet ran for {it} iterations to logL = {logL}")
        scarlet.display.show_likelihood(blend)

        ##display the fitting results
        fig_nostar_nolsb = scarlet.display.show_scene(scar_comps_nostars,
                                   norm=norm,
                                   observation=observation_nostar,
                                   show_rendered=True,
                                   show_observed=True,
                                   show_residual=True,
                                  )
        
        fig_nostar_nolsb.savefig(save_path + "/scarlet_scene_results_nolsb_nostar.png",bbox_inches="tight")
        plt.close(fig_nostar_nolsb)

        model_nolsb = blend.get_model()
        # Render it in the observed frame
        model_nolsb_ = observation_nostar.render(model_nolsb)
        # Compute residual
        residual_nolsb = data_arr-model_nolsb_
        #make a plot the grz image, the model and then the residuals 
        multi_panel_image(data_arr, model_nolsb, residual_nolsb, save_path + "/scarlet_model_img_nolsb.pdf")
        
        #-------
        #Step 2: Update scarlet model by adding a LSB component, but still masking stars
        #We fix the colors and position of the bkg sources, we also mask them in invar so the LSB does not fit them
        #we will unmask them in the end
        #-------
        
        print("Fitting only extended sources and lsb, no stars")
            
        ##like the point source, why does this not need an observation frame in its definition???

        ## I want to fix the colors of the background sources as do not want them to influence the LSB component flux? 
        ##define all components fresh, everyone component should have the same observation
        ##we ensure that the position remains fixed!

        observation_nostar_wlsb_bkgfix = scarlet.Observation(
                data_arr,
                psf=scarlet.ImagePSF(psf),
                weights = invar_weights_bkgmask,
                channels=filters).match(model_frame)

        scar_comps_nostars_v2 = np.copy(scar_comps_nostars)

        print(len(scar_comps_nostars_v2), len(centers_arr_nostars))

        ##only loop over the sources that are not starlet
        for ci,comp in enumerate(scar_comps_nostars_v2):
            #if the source is outside the main segment, then fix its color!!
            comp_y = int(centers_arr_nostars[ci][0])
            comp_x = int(centers_arr_nostars[ci][1])
            
            c_seg_num = segment_map_v2[comp_y, comp_x]
            
            if c_seg_num != 2:
                print(f"Source {ci} is not on main blob")
                # print(comp.parameters[0].__dict__["name"])
                # print(comp.parameters[1].__dict__["name"])
                # print(comp.parameters[2].__dict__["name"])
                # print("--")
                #fix spectrum and the center!
                comp.parameters[0].fixed=True #spectrum
                comp.parameters[1].fixed=True #image
                comp.parameters[2].fixed=True #shift
                
                ## if LSB, just set the wavelet documentation
                ##inputs to a models are defined as a class called parameter
                # https://pmelchior.github.io/scarlet/1-concepts.html#
                ## look at the parameter fixed

        scar_comps_nostars_v2 = list(scar_comps_nostars_v2)
        #adding the LSB source now
        scar_comps_nostars_v2.append(scarlet.source.StarletSource(model_frame))

        scarlet.initialization.set_spectra_to_match(scar_comps_nostars_v2, observation_nostar_wlsb_bkgfix)

        #run the fitting optimizer
        blend = scarlet.Blend(scar_comps_nostars_v2, observation_nostar_wlsb_bkgfix)
        it, logL = blend.fit(200, e_rel=1e-10)
        print(f"scarlet ran for {it} iterations to logL = {logL}")
        fig3 = scarlet.display.show_likelihood(blend)
        
        fig3.savefig(save_path + "/logL_iterations_wlsb_nostar.png",bbox_inches="tight")
        plt.close(fig3)
            
        # How to get uncertainties on flux parameter?
    
        ##display the fitting results
        fig4 = scarlet.display.show_scene(scar_comps_nostars_v2,
                                   norm=norm,
                                   observation=observation_nostar_wlsb_bkgfix,
                                   show_rendered=True,
                                   show_observed=True,
                                   show_residual=True,
                                  )
        
        fig4.savefig(save_path + "/scarlet_scene_results_wlsb_nostar.png",bbox_inches="tight")
        plt.close(fig4)

        fig_sources = scarlet.display.show_sources(scar_comps_nostars_v2,
                                   norm=norm,
                                   observation=observation_nostar_wlsb_bkgfix,
                                    show_model=True,
                                 show_rendered=True,
                                 show_observed=True,
                                 add_markers=False,
                                 add_boxes=True
                                  )
        
        fig_sources.savefig(save_path + "/scarlet_sources_results_wlsb_nostar.png",bbox_inches="tight")
        plt.close(fig_sources)
    
        #-------
        #Step 3: Update scarlet model unmasking stars. 
        #In this process, we will keep the Gaia position of the source fixed, and also keep the color of the LSB component fixed!
        #-------

        if np.sum(is_star) > 0:
        # if False:
            ##making the new observation frame with masked weights removed
            observation_wstar = scarlet.Observation(
                data_arr,
                psf=scarlet.ImagePSF(psf),
                weights = invar_weights,
                channels=filters).match(model_frame)

            ##one idea to test:
            ## continue with this approach of fitting stars last, but fix the lsb spectrum component?? so it does not change color??
            ## that might not do it ...
            ## I should just consider binary masking stars that are saturated

            ##example info on this: Starting with DR9, objects that appear in the Gaia catalogs are always retained in the Tractor catalogs, even if they would normally be cut by the model-selection criteria used to detect sources. This is because Gaia sources are often so bright that they saturate in Legacy Surveys imaging. Since such "retained" Gaia sources have no model fits, their flux_g, flux_r and flux_z values are estimated in the catalogs, using polynomial fits to Gaia-to-DECam color transformations for stars. Transformations to DECam are used even in areas of the Legacy Surveys footprint that are only covered by BASS and MzLS. The flux_ivar_[grz] values for these "retained" Gaia sources are set to zero.
            ##how to fix spectrum of the wavelet component??
            ## better to fit stars together and then fit LSB at last, need to think carefully about
            ##if star is saturated, check decals documentation for saturation limit
            ## if saturated, star will have to be extended source ...
            

            ##define all components fresh, everyone component should have the same observation
            ##we ensure that the position remains fixed!

            scar_comps_nostars_new = np.copy(scar_comps_nostars_v2)

            ##only loop over the sources that are not starlet
            for comp in scar_comps_nostars_new[:-1]:
                #for all non-LSB,non-star components, fix color/spectrum and center
                comp.parameters[0].fixed=False
                comp.parameters[1].fixed=False
                comp.parameters[2].fixed=True
                ## if LSB, just set the wavelet documentation
                ##inputs to a models are defined as a class called parameter
                # https://pmelchior.github.io/scarlet/1-concepts.html#
                ## look at the parameter fixed
    
            ##fix the spectrum/color of LSB component
            scar_comps_nostars_new[-1].parameters[0].fixed = True
            #should we also fix the coeffs?
            # scar_comps_nostars_new[-1].parameters[1].fixed = F

            #I have confirmed via print statements that the fixing parameters does change


            star_comps = []  
            for star_ind_i in center_star_inds:
                star_comp_i = scarlet.PointSource(model_frame, centers[star_ind_i], observation_wstar)
                ##we will be fixing the position of the star component
                # star_comp_i.parameters[1].fixed = True
                star_comps.append( star_comp_i )
                    
            #we will add stars location to places in the list such that the original list order is maintained!
            scar_comps_wlsb_wstars = add_comps_back(scar_comps_nostars_new, star_comps, center_star_inds)
    
            scarlet.initialization.set_spectra_to_match(scar_comps_wlsb_wstars, observation_wstar)
    
            #run the fitting optimizer
            blend = scarlet.Blend(scar_comps_wlsb_wstars, observation_wstar)
            it, logL = blend.fit(200, e_rel=1e-10)
            print(f"scarlet ran for {it} iterations to logL = {logL}")
            fig5 = scarlet.display.show_likelihood(blend)
            
            fig5.savefig(save_path + "/logL_iterations_wlsb_wstar.png",bbox_inches="tight")
            plt.close(fig5)
                
            # How to get uncertainties on flux parameter?
        
            ##display the fitting results
            fig6 = scarlet.display.show_scene(scar_comps_wlsb_wstars,
                                       norm=norm,
                                       observation=observation_wstar,
                                       show_rendered=True,
                                       show_observed=True,
                                       show_residual=True,
                                      )
            
            fig6.savefig(save_path + "/scarlet_scene_results_wlsb_wstar.png",bbox_inches="tight")
            plt.close(fig6)

            fig_sources = scarlet.display.show_sources(scar_comps_wlsb_wstars,
                                       norm=norm,
                                       observation=observation_wstar,
                                       show_model=True,
                                         show_rendered=True,
                                         show_observed=True,
                                         add_markers=False,
                                         add_boxes=True
                                      )
            
            fig_sources.savefig(save_path + "/scarlet_sources_results_wlsb_wstar.png",bbox_inches="tight")
            plt.close(fig_sources)
        else:
            scar_comps_wlsb_wstars = scar_comps_nostars_v2
            observation_wstar = observation_nostar_wlsb_bkgfix
    
        # for k, src in enumerate(scar_comps):
        #     print (f"{k}: {src.__class__.__name__}")
            
        #make sure the stars are fit by a psf!
        #find the scar_comps that correspond to stars
        
        ##add the sourece that will model LSB component in the entire cutout
#         scar_comps.append(scarlet.source.StarletSource(model_frame))
     
        ##I do not think I am actually begining with the earlier model
        ##has scar comps changed???
        # https://pmelchior.github.io/scarlet/0-quickstart.html#Initialize-sources
        ## look at this link!!

        ##display the fitting results for each band 
        # Compute model
        model = blend.get_model()
        # Render it in the observed frame
        model_ = observation_wstar.render(model)
        # Compute residual
        residual = data_arr-model_

        fp = open(save_path + "/scarlet_model_wlsb_wstars.sca", "wb")
        pickle.dump(scar_comps_wlsb_wstars, fp)
        fp.close()
        
        #########################################
        #MASK REGIONS OF LSB COMPONENT
        #########################################
    
        scar_comps_wlsb_wstars = np.array(scar_comps_wlsb_wstars)
    
        all_comp_inds = []
        all_xcens = []
        all_ycens = []
        
        for k,src in enumerate(blend):
            if hasattr(src, "center"):
                y,x = src.center
                # ax[0].text(x, y, k, color="k")
                ax[1].text(x, y, k, color="k")
                all_comp_inds.append(k)
                all_xcens.append(int(x))
                all_ycens.append(int(y))
                
            
        ## this source inds list does not include the last component which models the LSB componetn
        
        all_xcens = np.array(all_xcens)
        all_ycens = np.array(all_ycens)
        all_comp_inds = np.array(all_comp_inds)
           
        ax[1].set_title("g+r+z segment map",fontsize = fontsize)
        
        #firstly get the associated segment number for our sources in all_xcens, all_ycens
        all_seg_nums = segment_map_v2[all_ycens, all_xcens]
        
        ##print the object ids that lie on the main segment!
        
        all_xcens_inseg = all_xcens[ all_seg_nums == 2 ]
        all_ycens_inseg = all_ycens[ all_seg_nums == 2 ]
        all_comp_inds_inseg = all_comp_inds[ all_seg_nums == 2 ]

        
        for i,k in enumerate(all_comp_inds_inseg):
            ax[1].text(all_xcens_inseg[i], all_ycens_inseg[i], k, color="k")
                
        ## the sources that are printed on bold in third panel are the sources that lie on the main segment
                
        
        from photutils.morphology import data_properties
            
        segment_map_v3 = np.copy(segment_map_v2)
        
        #setting main segment to 1, background to 0 and other sources to -1
        segment_map_v3[segment_map_v3 == 0] = 0
        segment_map_v3[(segment_map_v3 !=2) & (segment_map_v3 > 0)  ] = -1
        segment_map_v3[segment_map_v3 == 2 ] = 1
        
        ax[1].imshow(segment_map_v3,cmap = "PiYG",alpha = 0.75,origin="lower")
        
        
        def get_elliptical_aperture(segment_data, id_num,sigma = 3):
            segment_data_v2 = np.copy(segment_data)
            ##we set everything else to zero except for our id num of reference
            segment_data_v2[segment_data != id_num] = 0
            segment_data_v2[segment_data == id_num] = 1
            
            #use this trick to get properties of the main segment 
            cat = data_properties(segment_data_v2, mask=None)
        
            columns = ['label', 'xcentroid', 'ycentroid', 'semimajor_sigma',
                       'semiminor_sigma', 'orientation']
            tbl = cat.to_table(columns=columns)
        
            from photutils.aperture import EllipticalAperture
            xypos = (cat.xcentroid, cat.ycentroid)
            r = sigma # choose an appropriate value here
            a = cat.semimajor_sigma.value * r
            b = cat.semiminor_sigma.value * r
            theta = cat.orientation.to(u.rad).value
                
            aperture = EllipticalAperture(xypos, a, b, theta=theta)
        
            return aperture
            
            
        mainseg_aperture = get_elliptical_aperture(segment_map_v3, 1)
        
        mainseg_aperture.plot(ax=ax[1], color='forestgreen', lw=3)
        
        all_notmain_apers = []


        ## we want to only mask the segments that appear in the footprints detected above 
        ## essentially, we want to ask the question whether the elliptical aperture we have drawn around the 
        ## all_segment_nums_notmain contains a center.x, center.y or not

        # Function to check if a point is inside the aperture
        def is_inside_ellipse(ys, xs, aperture):
            #the order of x and y is flipped because that is how the centers_arr is constructed
            mask = aperture.to_mask(method='center').to_image(segment_map_v2.shape) # Create a mask
            ##now I just need to check if any of these xs,ys fall inside this mask!
            mask_vals = mask[ ys.astype(int), xs.astype(int)] 
            if np.max(mask_vals) == 1:
                #there at least one footprint inside this mask and so we will mask it 
                return True
            else:
                return False

        ##SOMETHING CONFUSING IS HAPPENING HERE
        for i in range(len(all_segment_nums_notmain)):
            aper_notseg_i = get_elliptical_aperture(segment_data_new, all_segment_nums_notmain[i],sigma = 2)

            #check if this ellipse contains any of the peaks detected
            #if no peak is detected, then we do not mask this!
            # Check if any of the points fall inside
            footprints_inside_ellipse = is_inside_ellipse( centers_arr[:,0], centers_arr[:,1],  aper_notseg_i )

            if ~footprints_inside_ellipse:
                pass
            else:
                all_notmain_apers.append(aper_notseg_i)
                aper_notseg_i.plot(ax=ax[1], color='r', lw=1)
        
        ## let us also construct aperture for all the other sources and stars so we can mask them!!!
        ##this will be useful when calculating the LSB contribution
        ## let us do this for the identified sources, but are not part of the main segment
        ## but we do the aperture estimation on the original segment map, not the 0,1,2 map
        
        # Create a mask of the same shape as the image
        lsb_mask = np.zeros(segment_map_v2.shape, dtype=bool)
        # Convert apertures to masks and combine them
        
        if len(all_notmain_apers) > 0:            
            for api in all_notmain_apers:
                mask_ap = api.to_mask(method='center').to_image(segment_map_v2.shape)  # Convert to mask
                lsb_mask |= mask_ap.astype(bool)

        #I want to also mask the stars here!!
            
        #add to the mask all stuff outside the main aperture too
        mask_ap_outside = mainseg_aperture.to_mask(method='center').to_image(segment_map_v2.shape)
        lsb_mask |= ~mask_ap_outside.astype(bool)

        #I want to also mask the stars in the LSB component here!!

        lsb_star_mask = np.zeros(segment_map_v2.shape, dtype=bool)
        
        print(segment_map_v2.shape)
        print(lsb_star_mask.shape)
        
        for si in range(np.sum(is_star)):
            print(f"Masking star {si}")
            star_mag = source_star_max_mags[si]
            radius_pix = 0.5 * 3600*mask_radius_for_mag(star_mag)/0.262
            
            #in general, need to be very careful when copying and modifying arrays
            lsb_star_mask = mask_star( lsb_star_mask , ( star_f_xpix[si], star_f_ypix[si]) , radius_pix, mask_val = True )

        lsb_mask |= lsb_star_mask


        ## UPDATED lsb_mask
        ##it is simply going to mask the locations of the stars we were at!
        lsb_mask = np.zeros(segment_map_v2.shape, dtype=bool)
        lsb_mask[segment_map_v2 == 1] = True

        ##make an aperture here based on the R4.25 ellipse 
        from photutils.aperture import EllipticalAperture
        xypos = (aper_xcen_local, aper_ycen_local)
        b = aper_semi_maj * aper_ba * 4.25
        a = aper_semi_maj * 4.25
        aperture_r4 = EllipticalAperture(xypos, a, b, theta=aper_theta)
        aperture_r4.plot(ax=ax[1], color='limegreen', lw=3)
        #make a mask based on this!
        aper_r4_mask = aperture_r4.to_mask(method='center')  # or 'exact' if you want subpixel accuracy
        # now rasterize it onto an array the same shape as your image
        aper_r4_mask = aper_r4_mask.to_image(segment_map_v2.shape).astype(bool)
        
        #################################################
        ##plotting the LSB component in each band and its masked bits

        ## TODO: MASK THE STAR LOCATIONS IN THIS!
        #################################################
    
        #separate out the LSB scar_comps. 
        scar_comps_nolsb = scar_comps_wlsb_wstars[:-1]
        scar_comps_lsb = scar_comps_wlsb_wstars[-1]

        #assume nothing is a star as a baseline
        scar_comp_nolsb_isstar_bool = np.zeros(len(scar_comps_nolsb)).astype(bool)

        if np.sum(is_star) > 0:
            #if stars are there ...
            center_star_inds = np.array(center_star_inds)            
            scar_comp_nolsb_isstar_bool[ center_star_inds ] = True
                  # for star_ind_i in center_star_inds:
            #     #replacing extended source with point source
            #     scar_comps[star_ind_i]
        else:
            pass    
            
        #get the lsb model frame
        lsb_model = scar_comps_lsb.get_model(frame = model_frame)
        #this is the observed frame 
        lsb_model_ = observation_wstar.render(lsb_model)

      
        # if np.min(lsb_model) > 0:
        if True:
            ax[2].set_title("r-band LSB model",fontsize = fontsize)
            #apply the mask
            lsb_model[1][lsb_mask] = 0
            #do a linear model here
            # Linear normalization from min to max
            # norm = Normalize(vmin=lsb_model[1].min(), vmax=lsb_model[1].max())
            
            ax[2].imshow(lsb_model[1],norm=LogNorm(),origin="lower")
            ax[2].set_xticks([])
            ax[2].set_yticks([])
        
            lsb_aper_flux = []
        
            for i in range(3):
                #let us measure the total flux and the flux in the aperture and print it in the plot
        
                phot_table_i = aperture_photometry(lsb_model[i], mainseg_aperture, mask = lsb_mask)
                # phot_table_i = aperture_photometry(lsb_model[i], mainseg_aperture)
                
                aper_flux_i = float(phot_table_i["aperture_sum"] )
                lsb_aper_flux.append( aper_flux_i )
        
                #total flux is the entire LSB component flux in the cutout
                #aper flux is the LSB flux within aperture anchored on the main segment 
                tot_flux_i = np.sum(lsb_model[i])

                if i == 1:
                    mainseg_aperture.plot(ax=ax[2], color='k', lw=2)
                    
                    ax[2].text(0.05, 0.95, "Aper Flux = %.2f"%aper_flux_i, transform=ax[i].transAxes,
                        fontsize=12, color='k', verticalalignment='top',weight="bold")
            
                    ax[2].text(0.05, 0.875, "Total Flux = %.2f"%tot_flux_i, transform=ax[i].transAxes,
                        fontsize=12, color='k', verticalalignment='top',weight="bold")
        

            lsb_aper_flux = np.array(lsb_aper_flux)
        
            lsb_aper_mag = 22.5 - 2.5*np.log10(lsb_aper_flux)
            
            ## we need to make the scar comp model zero outside the aperture!!
        
            
        else:
            #there is no flux in the LSB component!
            lsb_aper_flux = np.array([0,0,0])
            
            lsb_aper_mag = np.array([np.nan, np.nan, np.nan])
        
        #########################################
        #ISOLATED DWARF GALAXY COMPONENTS
        #TODO: How to deal with color gradients in galaxies?
        #########################################

        gmm_file_zgrid = np.arange(0.001, 0.525,0.025)
        
        ##below part is not needed everytime! Can just directly load the dictionary of conf levels
        file_index = np.where( source_redshift > gmm_file_zgrid )[0][-1]
        #load the relevant gmm file
        gmm = joblib.load("/pscratch/sd/v/virajvm/redo_photometry_plots/gmm_color_models/gmm_model_idx_%d.pkl"%file_index)
    
        # Create a grid for evaluation
        xmin, xmax = -1, 2
        ymin, ymax = -1, 1.5
        X, Y = np.meshgrid(np.linspace(xmin, xmax, 200), np.linspace(ymin, ymax, 200))
        positions = np.vstack([X.ravel(), Y.ravel()])
        # Evaluate the density on the grid
        Z_gmm = np.exp(gmm.score_samples(positions.T)).reshape(X.shape)
        Znorm = np.sum(Z_gmm)
        
        #we normalize the gaussian mixture sum. However, to get the confidence levels in terms of the absolute probability returned by gmm, we will multiply by Znorm again
        Z_gmm = Z_gmm/Znorm       
        # plot contours if contour levels are specified in clevs 
        lvls = []
        clevs = [0.38,0.68,0.86,0.954,0.987]
        for cld in clevs:  
            sig = opt.brentq( conf_interval, 0., 1., args=(Z_gmm,cld) )   
            lvls.append(sig)
            
        conf_levels = { "38":lvls[0]*Znorm,"68":lvls[1]*Znorm,"86":lvls[2]*Znorm,"95.4":lvls[3]*Znorm,"98.7":lvls[4]*Znorm} #,"99.7":lvls[5]*Znorm}
    
            ## we ignore the fluxs and magnitudes of all the other sources. We just consider the sources that lie on this main segmen
        
        #get the scarlet components that lie on the main segment (excluding the total LSB component)
        scar_comps_inseg = scar_comps_nolsb[ all_seg_nums == 2]
        scar_comp_inseg_isstar_bool =  scar_comp_nolsb_isstar_bool[all_seg_nums == 2]
            
        #we only compute the colors for components in the main segment 
        
        scar_comps_inseg_mags = []
        scar_comps_inseg_fluxs = []
        
        for i, src in enumerate(scar_comps_inseg):
            fluxs = scarlet.measure.flux(src)
            
            #this does not appear to be it for getting statistical errors
#             p = src.parameters[1]
#             perr = np.sqrt(np.sum(p.std**2 ))
            

            mags= 22.5- 2.5*np.log10(fluxs)
            scar_comps_inseg_mags.append(mags)
            scar_comps_inseg_fluxs.append(fluxs)
            
        scar_comps_inseg_mags =np.array(scar_comps_inseg_mags)
        scar_comps_inseg_fluxs =np.array(scar_comps_inseg_fluxs)
    
        ##MAKING THE COLOR-COLOR PLOT
          #getting colors for the scarlet components
        all_grs = scar_comps_inseg_mags[:,0] - scar_comps_inseg_mags[:,1]
        all_rzs = scar_comps_inseg_mags[:,1] - scar_comps_inseg_mags[:,2]
        
        ##plot the g-r vs. r-z color contours from the redshift bin of this object        
        ax[3].contour(X, Y, Z_gmm, levels=sorted(lvls), cmap="viridis_r",alpha = 1,zorder = 1)
        
        ##I want to be able to see all the points even if they are outside the plotting grids. 
        all_grs_clip = np.clip(all_grs, -1, 2)
        all_rzs_clip = np.clip(all_rzs, -1, 1.5)
        
        ax[3].scatter(all_grs_clip,all_rzs_clip,color = "k")
        
        ##plot the errors for these sources too!
        
        #plotting LSB colors if they are not zer0
        if lsb_aper_mag[0] != np.nan:
            ax[3].scatter( lsb_aper_mag[0] - lsb_aper_mag[1], lsb_aper_mag[1] - lsb_aper_mag[2],edgecolor = "hotpink",lw = 2,
                       facecolor = "k",marker = "p",s = 100  )
        

        # Iterate only over filtered elements
        for gr, rz, k in zip(all_grs_clip, all_rzs_clip, all_comp_inds_inseg):
            ax[3].text(gr * 1.02, rz * 1.02, k, color="r", weight="bold")
        
        #using the LSB colors as the reference color!
        #do I need the 0.1 leniency factor? -> seems to work well
        #what happens if the LSB is way too far off like it is very blue??
        if lsb_aper_mag[0] != np.nan:
            col_lenient = 0.1 
            lsb_gr = lsb_aper_mag[0] - lsb_aper_mag[1]
            lsb_rz = lsb_aper_mag[1] - lsb_aper_mag[2]

            print(f"LSB colors = {lsb_gr, lsb_rz}")
        
            #set a minimum amount
            lsb_gr = np.maximum( lsb_gr, 0.3 )
            lsb_rz = np.maximum( lsb_rz, 0.3 )
            
            gr_cut = lsb_gr + col_lenient
            rz_cut =  lsb_rz + col_lenient
        else:
            print("LSB COLOR IS NAN!")
            gr_cut = 0.1 + 0.1
            rz_cut = -0.01 + 0.1

        print(f"reference colors = {gr_cut, rz_cut}")
        
        ##plot the bluer boundary lines
        ax[3].hlines(y = rz_cut,xmin = -1, xmax= gr_cut,color = "k",lw = 1, ls = "dotted")
        ax[3].vlines(x = gr_cut,ymin = -1, ymax= rz_cut,color = "k",lw = 1, ls = "dotted")    
        ax[3].set_xlim([-1,2])
        ax[3].set_ylim([-1,1.5])
    
        #########################################
        #COMPUTE NEW MAGNITUDES AND MAKE DWARF IMAGE
        #########################################
    
        ## any source that is in that blue box is automatically considered to be part of the galaxy
        #the lsb component with additional masks is by default always included
        #components that lie along this redshift contour are considered to be part of the galaxy!
        
        col_positions = np.vstack([ all_grs , all_rzs ])
        Z_likely_comps = np.exp(gmm.score_samples(col_positions.T))
        
        #is in the blue box or is along the contour!
        # has to have good color and not be a star!
        ## FOR THE SCARLET PIPELINE, WE ARE AS OF NOW DECIDING NOT USE THESE COLOR CONTOURS ... 
        dwarf_mask = ( ((all_grs < gr_cut) & (all_rzs < rz_cut)) | (Z_likely_comps >= conf_levels["98.7"])) & (~scar_comp_inseg_isstar_bool)
        # dwarf_mask = ( ((all_grs < gr_cut) & (all_rzs < rz_cut))) & (~scar_comp_inseg_isstar_bool)
        
        ##we can plot this star mask in the color-color plot
        star_mask = scar_comp_inseg_isstar_bool

        ###VISUALIZING THE SCARLET DWARF MODEL
        
        ## just subtract the masked locations for the final image
        ## we only need to do this for the visualization part as for the flux part we already take care of it!
        all_dwarf_sources = np.concatenate( [scar_comps_inseg[dwarf_mask], [scar_comps_lsb]] )
    
        dwarf = sum( source_j.get_model(frame=model_frame) for source_j in all_dwarf_sources)
        
        #we need to subtract the masked parts of LSB from the dwarf model
        for i in range(3):
            #the lsb mask is True on the regions that want to be masked!
            #those are the regions we want to subtract!
            lsb_model = scar_comps_lsb.get_model(frame = model_frame)
            subtract_mask_image = np.zeros(segment_map_v2.shape)
            subtract_mask_image[lsb_mask] = lsb_model[i][lsb_mask]    
            dwarf[i] -= subtract_mask_image
    
        #the image model with no dwarf
        nondwarf_model = blend.get_model(frame=model_frame) - dwarf
    
        ## COMPUTE THE FINAL DWARF MAGNITUDES
        new_mag_g = 22.5 - 2.5*np.log10(np.sum(dwarf[0]))
        new_mag_r = 22.5 - 2.5*np.log10(np.sum(dwarf[1]))
        new_mag_z = 22.5 - 2.5*np.log10(np.sum(dwarf[2]))

        new_mag_r4_g = 22.5 - 2.5*np.log10(np.sum(dwarf[0][aper_r4_mask] ))
        new_mag_r4_r = 22.5 - 2.5*np.log10(np.sum(dwarf[1][aper_r4_mask] ))
        new_mag_r4_z = 22.5 - 2.5*np.log10(np.sum(dwarf[2][aper_r4_mask] ))
    
        ######################################
        
        #get the original source mag!
    
         #find difference between source and all the other catalog objects
        ref_coord = SkyCoord(ra=source_ra * u.deg, dec=source_dec * u.deg)
        catalog_coords = SkyCoord(ra=source_cat_f["ra"].data * u.deg, dec=source_cat_f["dec"].data * u.deg)
        # Compute separations
        separations = ref_coord.separation(catalog_coords).arcsec
    
        #storing the separations of all sources from our main fiber source so we can remove it from consideration later
        source_cat_f["separations"] = separations
        
        source_cat_obs = source_cat_f[np.argmin(separations)]
        
        if np.min(separations) != 0:
            print("Minimum separation = ", np.min(separations))
            # raise ValueError("Distance between source and closest object in the photometry catalog in area =",np.min(separations))

        #this magnitude is not milky way corrected
        source_mag_g = source_cat_obs["mag_g"] 
        source_mag_r = source_cat_obs["mag_r"] 
        source_mag_z = source_cat_obs["mag_z"] 
        
        source_mag_g_mwc = source_mag_g  + 2.5 * np.log10(source_cat_obs["mw_transmission_g"])
        source_mag_r_mwc = source_mag_r  + 2.5 * np.log10(source_cat_obs["mw_transmission_r"])
        source_mag_z_mwc = source_mag_z  + 2.5 * np.log10(source_cat_obs["mw_transmission_z"])
    
        ######################################
    
    
        dwarf_model_ = observation_wstar.render(dwarf)
        nondwarf_model_ = observation_wstar.render(nondwarf_model)


        ##save these non-dwarf models for future use

        
        img_rgb =  sdss_rgb([data_arr[0],data_arr[1],data_arr[2]], ["g","r","z"], scales=dict(g=(2,6.0), r=(1,3.4), z=(0,2.2)), m=0.03)
        
        total_model_rgb =  sdss_rgb([model_[0],model_[1],model_[2]], ["g","r","z"], scales=dict(g=(2,6.0), r=(1,3.4), z=(0,2.2)), m=0.03)

        np.save(save_path + "/total_model_rgb.npy", total_model_rgb)

        dwarf_model_rgb =  sdss_rgb([dwarf_model_[0],dwarf_model_[1],dwarf_model_[2]], ["g","r","z"], scales=dict(g=(2,6.0), r=(1,3.4), z=(0,2.2)), m=0.03)

        np.save(save_path + "/dwarf_model_rgb.npy", dwarf_model_rgb)

        
        nondwarf_model_rgb =  sdss_rgb([nondwarf_model_[0],nondwarf_model_[1],nondwarf_model_[2]], ["g","r","z"], scales=dict(g=(2,6.0), r=(1,3.4), z=(0,2.2)), m=0.03)

        np.save(save_path + "/nondwarf_model_rgb.npy", nondwarf_model_rgb)

        resi_rgb =  sdss_rgb([residual[0],residual[1],residual[2]], ["g","r","z"], scales=dict(g=(2,6.0), r=(1,3.4), z=(0,2.2)), m=0.03)

        np.save(save_path + "/model_residual_rgb.npy", resi_rgb)
        
        ax[5].set_title("grz data",fontsize = fontsize)
        
        #draw a text box at the top that indicates, ra,dec and redshift
        # ax[5].text(0.5, 0.95,"(%.3f,%.3f, z=%.3f)"%(source_ra,source_dec, source_redshift) ,color = "yellow",fontsize = 12,
        #           transform=ax[5].transAxes, ha = "center", verticalalignment='top')

        ax[5].imshow(img_rgb,origin="lower")
        
        ax[6].set_title("grz model",fontsize = fontsize)
        ax[6].imshow(total_model_rgb,origin="lower")
        
        ax[7].set_title("grz dwarf model",fontsize = fontsize)
        ax[7].imshow(dwarf_model_rgb,origin="lower")
        
        ax[8].set_title("grz wo/dwarf model",fontsize = fontsize)
        ax[8].imshow(nondwarf_model_rgb,origin="lower")
        
        ax[9].set_title("grz residuals",fontsize = fontsize)
        ax[9].imshow(resi_rgb,origin="lower")
        
        
        ax[4].set_title("summary",fontsize = fontsize)

        spacing = 0.08
        ax[4].text(0.05,0.95,"Org-mag g = %.2f"%source_mag_g,size =11,transform=ax[4].transAxes, verticalalignment='top')
        ax[4].text(0.05,0.95 - spacing*1,"AperCog-mag g = %.2f"%aper_mag_g,size = 11,transform=ax[4].transAxes, verticalalignment='top')
        ax[4].text(0.05,0.95 - spacing*2,"Scarlet-mag g = %.2f"%new_mag_g,size = 11,transform=ax[4].transAxes, verticalalignment='top')
        ax[4].text(0.05,0.95 - spacing*3,"Scarlet-mag R4 g = %.2f"%new_mag_r4_g,size = 11,transform=ax[4].transAxes, verticalalignment='top')
        
        # ax[4].text(0.05,0.95 - spacing*3,"fracflux_g = %.2f, %.2f"%(source_cat_obs["fracflux_g"]),size = 11,transform=ax[4].transAxes, verticalalignment='top')
        
        ax[4].text(0.05,0.95 - spacing*4,"Org-mag r = %.2f"%source_mag_r,size =11,transform=ax[4].transAxes, verticalalignment='top')
        ax[4].text(0.05,0.95 - spacing*5,"AperCog-mag r = %.2f"%aper_mag_r,size = 11,transform=ax[4].transAxes, verticalalignment='top')
        ax[4].text(0.05,0.95 - spacing*6,"Scarlet-mag r = %.2f"%new_mag_r,size = 11,transform=ax[4].transAxes, verticalalignment='top')
        ax[4].text(0.05,0.95 - spacing*7,"Scarlet-mag R4 r = %.2f"%new_mag_r4_r,size = 11,transform=ax[4].transAxes, verticalalignment='top')
        
        # ax[4].text(0.05,0.95 - spacing*7,"fracflux_r = %.2f, %.2f"%(source_cat_obs["fracflux_r"]),size = 11,transform=ax[4].transAxes, verticalalignment='top')
        
    
        ax[4].text(0.05,0.95 - spacing*8,"Org-mag z = %.2f"%source_mag_z,size =11,transform=ax[4].transAxes, verticalalignment='top')
        ax[4].text(0.05,0.95 - spacing*9,"AperCog-mag z = %.2f"%aper_mag_z,size = 11,transform=ax[4].transAxes, verticalalignment='top')
        ax[4].text(0.05,0.95 - spacing*10,"Scarlet-mag z = %.2f"%new_mag_z,size = 11,transform=ax[4].transAxes, verticalalignment='top')
        ax[4].text(0.05,0.95 - spacing*11,"Scarlet-mag R4 z = %.2f"%new_mag_r4_z,size = 11,transform=ax[4].transAxes, verticalalignment='top')
        
        # ax[4].text(0.05,0.95 - spacing*11,"fracflux_z = %.2f, %.2f"%(source_cat_obs["fracflux_z"]),size = 11,transform=ax[4].transAxes, verticalalignment='top')
    
        ax[4].set_xlim([0,1])
        ax[4].set_ylim([0,1])
        
        ## add the color-color info now!

       
        for i,axi in enumerate(ax):
            if i != 3:
                axi.set_xticks([])
                axi.set_yticks([])
                
        save_summary_png = save_path + "/grz_scarlet_tgid_%d_ra_%.3f_dec_%.3f.pdf"%(source_tgid, source_ra, source_dec)
        fig_tot.savefig(save_summary_png ,bbox_inches="tight")

        plt.close(fig_tot)
        
        new_mag_g_mwc = new_mag_g + 2.5 * np.log10(source_cat_obs["mw_transmission_g"])
        new_mag_r_mwc = new_mag_r + 2.5 * np.log10(source_cat_obs["mw_transmission_r"])
        new_mag_z_mwc = new_mag_z + 2.5 * np.log10(source_cat_obs["mw_transmission_z"])

        new_mag_R4_g_mwc = new_mag_r4_g + 2.5 * np.log10(source_cat_obs["mw_transmission_g"])
        new_mag_R4_r_mwc = new_mag_r4_r + 2.5 * np.log10(source_cat_obs["mw_transmission_r"])
        new_mag_R4_z_mwc = new_mag_r4_z + 2.5 * np.log10(source_cat_obs["mw_transmission_z"])

        #save these magnitudes for future reference!

        new_mags = [ new_mag_g_mwc, new_mag_r_mwc,new_mag_z_mwc ]
        np.save( save_path + "/scarlet_mags.npy", new_mags )

        new_mags_R4 = [ new_mag_R4_g_mwc, new_mag_R4_r_mwc,new_mag_R4_z_mwc ]
        np.save( save_path + "/scarlet_mags_R4.npy", new_mags_R4 )
        # org_mags =   [ source_mag_g_mwc,source_mag_r_mwc,source_mag_z_mwc  ]
        
        return


import numpy as np
from astropy.coordinates import SkyCoord
import astropy.units as u

def filter_close_sources(ra, dec, radius_arcsec=30):
    """
    Remove sources that lie within <radius_arcsec of another source.
    Keeps the first instance encountered.

    ra, dec : arrays in degrees
    returns : indices of the sources to keep
    """
    coords = SkyCoord(ra=ra*u.degree, dec=dec*u.degree, frame='icrs')
    keep = []
    mask = np.ones(len(coords), dtype=bool)

    for i in range(len(coords)):
        if not mask[i]:
            continue  # this one was already excluded
        keep.append(i)
        # mark all others within the radius as excluded
        sep = coords[i].separation(coords)
        mask[sep < (radius_arcsec*u.arcsec)] = False

    return np.array(keep)



def basic_filter_scarlet(data):
    data_scarlet = data[(data["Z"] < 0.01) & (data["LOGM_SAGA_APERTURE_COG"] < 8) & (data["LOGM_SAGA_APERTURE_COG"] > 6) 
                        & (data["MASKBITS"]==0) & (data["SGA_D26_NORM_DIST"] > 2) & (data["is_south"] == 1)  ]

    return data_scarlet
    

def get_entire_scarlet():

    data_bgsb = Table.read("/pscratch/sd/v/virajvm/catalog_dr1_dwarfs/iron_photometry/iron_BGS_BRIGHT_shreds_catalog_w_aper_mags.fits")
    data_bgsf = Table.read("/pscratch/sd/v/virajvm/catalog_dr1_dwarfs/iron_photometry/iron_BGS_FAINT_shreds_catalog_w_aper_mags.fits")
    data_lowz = Table.read("/pscratch/sd/v/virajvm/catalog_dr1_dwarfs/iron_photometry/iron_LOWZ_shreds_catalog_w_aper_mags.fits")
    data_elg = Table.read("/pscratch/sd/v/virajvm/catalog_dr1_dwarfs/iron_photometry/iron_ELG_shreds_catalog_w_aper_mags.fits")
    
    data_bgsb = basic_filter_scarlet(data_bgsb)
    data_bgsf = basic_filter_scarlet(data_bgsf)
    data_lowz = basic_filter_scarlet(data_lowz)
    data_elg = basic_filter_scarlet(data_elg)

    print(len(data_bgsb), len(data_bgsf), len(data_lowz), len(data_elg))

    data_all = vstack([data_bgsb, data_bgsf, data_lowz, data_elg])

    print(len(data_all))
    
    # example usage:
    keep_indices = filter_close_sources( data_all["RA"].data, data_all["DEC"].data, radius_arcsec = 60 )

    data_all = data_all[keep_indices]

    print(len(data_all))

    return data_all

    



if __name__ == '__main__':

    # data_scarlet = get_entire_scarlet()
    # print(f"Number of galaxies for scarlet model = {len(data_scarlet)}")
    

    data_scarlet = Table.read("/pscratch/sd/v/virajvm/catalog_dr1_dwarfs/iron_photometry/iron_BGS_BRIGHT_shreds_catalog_w_aper_mags.fits")
    index = np.where(data_scarlet["TARGETID"] == 39627824100806546 )[0][0]

    SKIP_INDICES = []
    
    def process_index(index):
        print(index)
        if index in SKIP_INDICES:
            return
    
        try:
            tgid_i = data_scarlet["TARGETID"].data[index]
            ra_i = data_scarlet["RA"].data[index]
            dec_i = data_scarlet["DEC"].data[index]
            zred_i = data_scarlet["Z"].data[index]
            cutout_path = data_scarlet["IMAGE_PATH"].data[index]
            hdus = fits.open(cutout_path)
            image_data = hdus[0].data
            invvar_data = hdus[1].data
    
            wcs = WCS(hdus[0])
    
            save_folder = data_scarlet["FILE_PATH"].data[index]
            if isinstance(save_folder, bytes):
                save_folder = save_folder.decode("utf-8")
    
    
            source_file = os.path.join(save_folder, "source_cat_f_more.fits")
            if not os.path.exists(source_file):
                source_file = os.path.join(save_folder, "source_cat_f.fits")
            source_cat = Table.read(source_file)
    
            input_dict = {
                "save_path": save_folder,
                "tgid": tgid_i,
                "ra": ra_i,
                "dec": dec_i,
                "redshift": zred_i,
                "wcs": wcs,
                "image_data": image_data,
                "invvar_data" : invvar_data,
                "source_cat": source_cat,
                "overwrite": True,
                "run_own_detect": True,
                "box_size": np.shape(image_data)[1],
                "APER_XCEN": data_scarlet["COG_APER_CEN_XPIX"].data[index] ,
                "APER_YCEN": data_scarlet["COG_APER_CEN_YPIX"].data[index],
                "APER_SEMI_MAJ": data_scarlet["COG_APER_PARAMS"].data[:,0][index], 
                "APER_BA": data_scarlet["COG_APER_PARAMS"].data[:,1][index],
                "APER_THETA": data_scarlet["COG_APER_PARAMS"].data[:,2][index]     
            }
    
            for b in "GRZ":
                input_dict[f"MAG_{b}_APERTURE_COG"] = data_scarlet[f"MAG_{b}_APERTURE_COG"].data[index]
    
            print(f"Processing: {index}, RA={ra_i}, DEC={dec_i}, Folder={save_folder}")
            run_scarlet_pipe(input_dict)
            
        except Exception as e:
            print(f"Error processing index {index}: {e}")

    ##index = 26 is an interesting case to study as the tractor model was not perfect but the scarlet model is better. We can discuss this as an interestin case!

    # indices_to_process = [i for i in range(len(data_scarlet)) if i not in SKIP_INDICES]
    # indices_to_process = [i for i in range(len(data_scarlet))]
    
    # with ProcessPoolExecutor(max_workers=8) as executor:
    #     list(tqdm(executor.map(process_index, indices_to_process), total=len(indices_to_process)))

    # index = 1
    # process_index(index)

    process_index(index)


   

    


import os
import sys
from tqdm import trange
from astropy.io import fits
from astropy.table import Table, vstack
from astropy.wcs import WCS
import matplotlib.pyplot as plt
from scipy.ndimage import binary_dilation
from photutils.segmentation import detect_sources, deblend_sources
from sklearn.metrics import jaccard_score
from photutils.background import Background2D, MedianBackground
import matplotlib.patches as patches
from matplotlib.patches import Circle
from photutils.background import StdBackgroundRMS, MADStdBackgroundRMS
from astropy.stats import SigmaClip
from photutils.segmentation import make_2dgaussian_kernel
from astropy.convolution import convolve
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
from desi_lowz_funcs import make_subplots, find_nearest_island
from scipy.ndimage import generic_filter
from matplotlib.colors import LogNorm
from scipy.ndimage import gaussian_filter
from desi_lowz_funcs import get_elliptical_aperture, sdss_rgb
from photutils.aperture import aperture_photometry, EllipticalAperture
from photutils.morphology import data_properties
import numpy.ma as ma

def make_custom_cmap(n_colors, cmap_name="rainbow"):
    """
    Make a custom discrete colormap where 0 is black and the rest are distinct colors.
    
    Parameters
    ----------
    n_colors : int
        Total number of colors (including black for 0).
    cmap_name : str
        Name of base colormap to sample (e.g. "rainbow", "tab20", "hsv", etc.)
    
    Returns
    -------
    cmap : ListedColormap
        Custom colormap.
    """
    # start with black
    colors = [(0, 0, 0)]  

    # sample evenly spaced colors from chosen cmap
    base_cmap = plt.get_cmap(cmap_name)
    sampled_colors = base_cmap(np.linspace(0, 1, n_colors - 1))
    
    # add them after black
    colors.extend(sampled_colors)
    
    return mcolors.ListedColormap(colors)

cmap_cstm = make_custom_cmap(8, cmap_name="tab10")

def jaccard(a, b):
    # a, b: boolean masks
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return inter / union if union > 0 else 1.0

    
def get_desi_segment_score(current_deblend, prev_deblend):
    
    current_deblend, prev_deblend = current_deblend.data, prev_deblend.data
    
    center_coord = int(current_deblend.shape[0]/2)

    if center_coord < 175:
        raise ValueError(f"Weird center location: {center_coord}")
        
    # the component that contains our fiber. Our fiber will be in the center!
    fiber_label = current_deblend[center_coord,center_coord]
    #in both cases, if fiber label happens to be off-center, we find the closest deblend segment!
    if fiber_label == 0:
        fiber_label, _ = find_nearest_island(current_deblend, center_coord,center_coord)
    top_mask = (current_deblend == fiber_label)
    
    fiber_label = prev_deblend[center_coord,center_coord]
    if fiber_label == 0:
        fiber_label, _ = find_nearest_island(current_deblend, center_coord,center_coord)
    prev_mask = (prev_deblend == fiber_label)

    score = jaccard(prev_mask, top_mask)
    
    return score
    
def get_largest_segment_score(current_deblend, prev_deblend):
    curr_labels, curr_areas = current_deblend.labels, current_deblend.areas
    prev_labels, prev_areas = prev_deblend.labels, prev_deblend.areas
    
    # largest component by area
    top_idx = np.argmax(curr_areas)
    top_label = curr_labels[top_idx]
    top_mask = current_deblend.data == top_label
    
    top_idx = np.argmax(prev_areas)
    top_label = prev_labels[top_idx]
    prev_mask = prev_deblend.data == top_label
    
    score = jaccard(prev_mask, top_mask)
    
    return score
    
    
def measure_jaccard_scores(current_deblend, prev_deblend):
    ## THIS IS THE SCORE OF THE DESI SEGMENT
    desi_score = get_desi_segment_score(current_deblend, prev_deblend)

    ##THIS IS THE SCORE OF THE LARGEST SEGMENT
    large_score = get_largest_segment_score(current_deblend, prev_deblend)

    return desi_score, large_score


def find_contrast_run(ncontrast, all_jacc_desi, all_jacc_largest, thresh=0.95, run_len=5):
    all_jacc_desi = np.array(all_jacc_desi)
    all_jacc_largest = np.array(all_jacc_largest)
    
    n = len(all_jacc_desi)
    assert len(all_jacc_largest) == n
    assert len(ncontrast) == n + 1  # you mentioned this
    
    # mask where both are above threshold
    good = (all_jacc_desi > thresh) & (all_jacc_largest > thresh)
    
    # slide a window of length run_len
    for i in range(n - run_len + 1):
        if np.all(good[i:i+run_len]):
            # take the last index in this run
            idx = i + run_len - 1
            # map back to ncontrast (+1 offset)
            chosen_contrast = ncontrast[idx+1]
            return idx, chosen_contrast
    
    return None, np.nan  # nothing found



def find_optimal_ncontrast(img_rgb, img_rgb_mask, convolved_tot_data, segment_map, nlevel_val = 4,save_path = None, pcnn_val=None,radec=None, tgid=None, mu_aperture=None, source_zred=None):
    '''
    In this function, we find the optimal ncontrast value with which we will deblend.
    We also save a plot regarding this!!
    '''
    
    all_segm_deblends = []
    all_jacc_desi = []
    all_jacc_largest = []

    #this originally used to be 0.05
    ncontrast = np.arange(0.05,0.2,0.01)

    fig,ax = make_subplots(ncol = 4 , nrow = 5, return_fig = True,row_spacing = 0.1,col_spacing = 0.1,plot_size = 1.5)

    for axi in ax:
        axi.set_xticks([])
        axi.set_yticks([])

    ##plot the rgb image too
    ax[-4].set_title(f"{tgid}",fontsize = 8)
    ax[-4].imshow(img_rgb,origin="lower")

    #enter ra,dec text here!
    ax[-4].text(0.5,0.9, "(%.3f,%.3f)"%(radec[0], radec[1]) ,color = "white",fontsize = 8,
               ha="center",va="center",transform = ax[-4].transAxes)
    
    ax[-3].text(0.5,0.9, "(%.4f)"%(source_zred) ,color = "white",fontsize = 9,
               ha="center",va="center",transform = ax[-3].transAxes)

    ax[-3].set_title(f"MU_R = {mu_aperture:.2f}",fontsize = 8)
    ax[-3].imshow(img_rgb_mask,origin="lower")

# ,    ax[-2].set_title(f"PCNN = {pcnn_val:.2f}",fontsize = 8)
    #plot the smoothed segment that is being deblended
    ax[-2].imshow(segment_map.data,cmap = "tab10",origin="lower")

    for i in range(len(ncontrast)):
        segm_deblend_ni_map = deblend_sources(convolved_tot_data,segment_map,
                                   npixels=10,nlevels=nlevel_val, contrast=ncontrast[i],
                                   progress_bar=False)
        
        ax[i].text(0.5,0.85, f"nlevel = {nlevel_val}\nncontrast = {ncontrast[i]:.3f}", transform=ax[i].transAxes,
                   ha="center",va="center",fontsize = 10 )
        
        ax[i].imshow(segm_deblend_ni_map.data,cmap ="tab20",origin="lower")

        if i > 0:
            desi_jacc, large_jacc = measure_jaccard_scores(segm_deblend_ni_map, all_segm_deblends[-1] )
            
            all_jacc_desi.append(desi_jacc)
            all_jacc_largest.append(large_jacc)

            ax[i].text(0.5,0.1, f"$j_{{\\rm desi}},j_{{\\rm large}} = {desi_jacc:.2f}, {large_jacc:.2f}$", transform=ax[i].transAxes,
                       ha="center",va="center",fontsize = 9 )

        all_segm_deblends.append( segm_deblend_ni_map )

        ##make a plot with this!
        
    all_jacc_desi = np.array(all_jacc_desi)
    all_jacc_largest = np.array(all_jacc_largest)
    
    #we want to identify the first instance of 5 consecutive values of J > 0.95.
    #we stop at hte last one and choose that ncontrast value!!
    # if verbose:
    #     print(all_jacc_desi)
    #     print(all_jacc_largest)
    #     print(ncontrast)


    #if we do not find any, we should just stick with the largest 0.2 value! we need to find at least 6 or more consecutive 
    _, ncontrast_opt = find_contrast_run(ncontrast, all_jacc_desi, all_jacc_largest, thresh=0.95, run_len=6)

    segm_deblend_optimal = deblend_sources(convolved_tot_data,segment_map,
                                       npixels=10,nlevels=nlevel_val, contrast=ncontrast_opt,
                                       progress_bar=False)

    if _ is None:
        ncontrast_opt = 0.2

    #now plot the most optimal ncontrast value!!
    ax[-1].imshow(segm_deblend_optimal.data,cmap ="tab20",origin="lower")

    jaccard_img_path = save_path + f"/jaccard_smoothing_nlevel_{nlevel_val}.png"
    
    plt.savefig(jaccard_img_path,bbox_inches="tight")
    plt.close()
    
    return ncontrast_opt, jaccard_img_path


def process_deblend_image(segment_map, segm_deblend, fiber_xpix, fiber_ypix, save_path):
    '''
    Function where we process the deblended image so easier to plot!! Sets all other deblended segments in image to zero and just focuses on the deblended segment of the main blob.

    Note here we do not need to worry about no segment being detected as we have already taken care of that situation before running this function!
    '''
        
    segment_map_v2 = np.copy(segment_map.data)
    segment_map_v2_copy = np.copy(segment_map_v2)
    
    island_num = segment_map.data[int(fiber_ypix),int(fiber_xpix)]

    #if the island num is zero (on bkg), we get the closest one!
    nearest_blob_dist_pix = 0
    if island_num == 0:
        island_num, nearest_blob_dist_pix = find_nearest_island(segment_map_v2_copy, fiber_xpix,fiber_ypix)
    
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

    #save this deblended image for plotting purposes later!!
    np.save(save_path + "/deblend_segments_isolate.npy", segm_deblend_v3)

    #what is the deblend id where our main source is in?
    deblend_island_num = segm_deblend_v3[int(fiber_ypix),int(fiber_xpix)]
    
    if deblend_island_num == 0:
        deblend_island_num, _ = find_nearest_island(segm_deblend_v3, fiber_xpix,fiber_ypix)

    #given this deblend island num, get the mask of just the parent galaxy of interest
    parent_galaxy_mask = (segm_deblend_v3 == deblend_island_num)

    # not_parent_galaxy_mask = (segm_deblend_v3 != deblend_island_num) & (segm_deblend_v3 > 0)
    #we want to mask other deblended blobs in the main segment or mask other blobs (distinct from main blob) and hence the last condition
    #this makes sure that we do not mask out the sky so we can still run COG
    not_parent_galaxy_mask = ( (segm_deblend_v3 != deblend_island_num) & (segm_deblend_v3 > 0) ) | (segment_map_v2 == 1)
    
    ##how many deblended segments are on the main blob?
    num_deblend_segs_main_blob = len(deblend_ids)

    return segm_deblend_v3, num_deblend_segs_main_blob, parent_galaxy_mask.astype(int), not_parent_galaxy_mask, nearest_blob_dist_pix


# def simple_interpolate_mask(image, mask, footprint_size=5):
#     filled = image.copy()
#     temp = filled.astype(float)
#     temp[mask] = np.nan

#     # local mean ignoring NaNs
#     def mean_ignore_nan(values):
#         return np.nanmean(values)

#     footprint = np.ones((footprint_size, footprint_size))
#     interpolated = generic_filter(temp, mean_ignore_nan, footprint=footprint, mode='nearest')

#     filled[mask] = interpolated[mask]
#     return filled

# from scipy.interpolate import griddata

# def linear_interpolate_mask_SMOOTH(org_image, smooth_image, segment_map_v2, mask):
#     '''
#     We will use the smoothed image for interpolation to fill into the unsmoothed image! We only want to interpolate on pixels that are on the main blob. Rest will be left as is. Thus the only pixels tha we will be updating are those that are masked and segment_map_v2 = 2.
#     '''
#     filled = org_image.copy()
#     y, x = np.indices(org_image.shape)
    
#     points = np.column_stack((x[~mask], y[~mask]))
#     values = smooth_image[~mask]
    
#     filled[mask & (segment_map_v2 == 2)] = griddata(points, values, (x[mask & (segment_map_v2 == 2)], y[mask & (segment_map_v2 == 2)]), method='linear')
    
#     return filled

# def linear_interpolate_mask(org_image, segment_map_v2, mask):
#     '''
#     We will use the smoothed image for interpolation to fill into the unsmoothed image! We only want to interpolate on pixels that are on the main blob. Rest will be left as is. Thus the only pixels tha we will be updating are those that are masked and segment_map_v2 = 2.
#     '''
#     filled = org_image.copy()
#     y, x = np.indices(org_image.shape)
    
#     points = np.column_stack((x[~mask], y[~mask]))
#     values = org_image[~mask]

    
#     # Target (masked + blob=2) pixels
#     sel = mask & (segment_map_v2 == 2)
#     # Safety check: if no valid input, skip
#     if points.size == 0 or values.size == 0 or np.sum(sel) == 0:
#         return filled

#     filled[mask & (segment_map_v2 == 2)] = griddata(points, values, (x[mask & (segment_map_v2 == 2)], y[mask & (segment_map_v2 == 2)]), method='linear')
    
#     return filled
    
# def mean_surface_brightness(mag, ellip_area):
#     '''
#     # Function to compute surface brightness. We will use this in the r-band data and the input is the area in arcsec2 of the ellipse in which we are measuring the rough photo
#     # '''
    # return mag + 2.5*np.log10(ellip_area)

    
def get_isolate_galaxy_mask(img_rgb=None, img_rgb_mask=None, r_band_trac_model=None, r_rms=None, fiber_xpix=None, fiber_ypix=None, file_path=None,  tgid=None, radec=None, r_mu_aperture = None, segment_map_v2 = None, source_zred=None, pcnn_val=None):
    '''
    Function that returns the deblended segment used for isolating the parent galaxy. 
    Note that this function is only run when z > 0.01, and that flag is applied in the aperture_cogs.py script when this function is called

    The r_band_data here is the r-band tractor model image. A few benefits of this: avoids the weirdness of having to interpolate over masked regions and deal with stars. Also avoids issues with bright stars. Secondly, tractor naturally smooths out the strucutre at some level so make over-deblending less of an issue, but still possible. Furthermore, more meaningful computation of surface brightness (no artifical inflation of aperture due to bad pixels). Secondly, in case in a dense cluter environment, with strong light background from outside, that is not always included in the model image as the source is outside. So a natural way to deal with this. 

    Will need to VI to check that the smoothing scale and jaccard method is working well! 
    The other potential issue is that if tractor rchisq is very poor, then this may not work well? Need to VI the ones with bad rchisq to check.

    I could find that the smoothing scale is too aggresive .. 
    '''
    
    npixels_min = 10
    threshold_rms_scale = 1.5

    #we are 

    ###get the updated deblend values!!
    kernel = make_2dgaussian_kernel(15, size=29)  # FWHM = 3.0

    threshold = threshold_rms_scale * r_rms
    convolved_tot_data = convolve( r_band_trac_model, kernel)
    
    segment_map = detect_sources(convolved_tot_data, threshold, npixels=npixels_min) 

    #we want sources where we detect something!! If we do not detect anything, we return a non valyue
    if segment_map is None:
        print(f"Nothing found in this smooth segmentation: {tgid}! Returning Nones.")

        # segm_deblend_opt, num_deblend_segs_main_blob, parent_galaxy_mask, not_parent_galaxy_mask, nearest_deblend_blob_dist_pix, nearest_main_blob_dist_pix, jaccard_img_path
        
        return None, 0, None, None, np.nan, None, np.nan
    else:
        segment_map_data = segment_map.data

        ncontrast_opt, jaccard_img_path = find_optimal_ncontrast(img_rgb, img_rgb_mask, convolved_tot_data, segment_map, nlevel_val = 4, save_path = file_path, pcnn_val=pcnn_val, radec=radec, tgid=tgid, source_zred=source_zred, mu_aperture = r_mu_aperture)
        
        segm_deblend_opt = deblend_sources(convolved_tot_data,segment_map,
                                           npixels=npixels_min,nlevels=4, contrast=ncontrast_opt,
                                           progress_bar=False)

        
        segm_deblend_opt, num_deblend_segs_main_blob, parent_galaxy_mask, not_parent_galaxy_mask, nearest_blob_dist_pix = process_deblend_image(segment_map, segm_deblend_opt, fiber_xpix, fiber_ypix, file_path)

        #segm_deblend_opt is the optimally deblended image with just the deblended segments on main blob highlighted
        #num_deblend_segs_main_blob is the number of deblended segments identified in the main blob
        #if it is equal to 1, nothing to mask
        
        #parent_galaxy_mask is the mask that contains only the parent galaxy of interest. This is used to estimate the aperture!
        
        #no_parent_galaxy_mask is a mask for the other deblended blobs identified in the main blob and is the mask that we will apply in COG pipeline.

        if num_deblend_segs_main_blob == 0:
            raise ValueError(f"Incorrect number of deblended blobs found in main blob in isolate galaxy mask! -> {tgid} ")

    ##let us do a binary dilation on the not_parent_galaxy_mask to avoid some of the extremeities!!
    structure = np.ones((3, 3), dtype=bool)
    not_parent_galaxy_mask = binary_dilation(not_parent_galaxy_mask, structure=structure, iterations=2)

    ###save the summary plot, which we will use for the VI
    
    g_fake = np.zeros_like(r_band_trac_model)
    z_fake = np.zeros_like(r_band_trac_model)
    r_only_grz_data = np.array([g_fake, r_band_trac_model, z_fake ])
    
    fig, ax = make_subplots(ncol = 4, nrow = 1,col_spacing = 0.05, return_fig=True)
    ax[0].imshow(img_rgb, origin="lower")
    ax[1].imshow(img_rgb_mask, origin="lower")
    
    trac_r_rgb = sdss_rgb(r_only_grz_data)
    ax[2].set_title(f"mu_r = {r_mu_aperture:.2f}")
    ax[2].imshow(trac_r_rgb, origin="lower")

    #plot the deblended image?
    ax[3].imshow(segm_deblend_opt, origin="lower", cmap = cmap_cstm, interpolation=None)

    for axi in ax:    
        axi.set_xticks([])
        axi.set_yticks([])
    plt.savefig(file_path + "/parent_isolate_VI.png",bbox_inches="tight")
    plt.close(fig)
    
    return segm_deblend_opt, num_deblend_segs_main_blob, parent_galaxy_mask, not_parent_galaxy_mask, nearest_blob_dist_pix, jaccard_img_path, ncontrast_opt
    

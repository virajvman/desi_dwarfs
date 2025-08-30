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
from alternative_photometry_methods import find_nearest_island
from desi_lowz_funcs import make_subplots

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


def find_optimal_ncontrast(img_rgb, img_rgb_mask, convolved_tot_data, segment_map, nlevel_val = 4,save_path = None):
    '''
    In this function, we find the optimal ncontrast value with which we will deblend.
    We also save a plot regarding this!!
    '''
    
    all_segm_deblends = []
    all_jacc_desi = []
    all_jacc_largest = []

    #this originally used to be 0.05
    ncontrast = np.arange(0.001,0.2,0.01)

    fig,ax = make_subplots(ncol = 4 , nrow = 6, return_fig = True,row_spacing = 0.1,col_spacing = 0.1,plot_size = 1.5)

    for axi in ax:
        axi.set_xticks([])
        axi.set_yticks([])

    ##plot the rgb image too
    ax[-4].set_title(f"grz image",fontsize = 10)
    ax[-4].imshow(img_rgb,origin="lower")

    ax[-3].set_title(f"latest reconstruct grz image",fontsize = 10)
    ax[-3].imshow(img_rgb_mask,origin="lower")

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


    #if we do not find any, we should just stick with the largest 0.2 value!
    _, ncontrast_opt = find_contrast_run(ncontrast, all_jacc_desi, all_jacc_largest, thresh=0.95, run_len=5)

    segm_deblend_optimal = deblend_sources(convolved_tot_data,segment_map,
                                       npixels=10,nlevels=nlevel_val, contrast=ncontrast_opt,
                                       progress_bar=False)

    if _ is None:
        ncontrast_opt = 0.2

    #now plot the most optimal ncontrast value!!
    ax[-1].imshow(segm_deblend_optimal.data,cmap ="tab20",origin="lower")

    plt.savefig(save_path + f"/jaccard_smoothing_nlevel_{nlevel_val}.png",bbox_inches="tight")
    plt.close()
    
    return ncontrast_opt




def process_deblend_image(segment_map, segm_deblend, fiber_xpix, fiber_ypix):
    '''
    Function where we process the deblended image so easier to plot!! Sets all other deblended segments in image to zero and just focuses on the deblended segment of the main blob.

    Note here we do not need to worry about no segment being detected as we have already taken care of that situation before running this function!
    '''
        
    segment_map_v2 = np.copy(segment_map.data)
    segment_map_v2_copy = np.copy(segment_map_v2)
    
    island_num = segment_map.data[int(fiber_ypix),int(fiber_xpix)]

    #if the island num is zero (on bkg), we get the closest one!
    if island_num == 0:
        island_num, _ = find_nearest_island(segment_map_v2_copy, fiber_xpix,fiber_ypix)
    
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
        deblend_island_num, nearest_deblend_dist_pix = find_nearest_island(segm_deblend_v3, fiber_xpix,fiber_ypix)
        lie_on_smooth_segment=False
    else:
        nearest_deblend_dist_pix = 0
        lie_on_smooth_segment=True

    #given this deblend island num, get the mask of just the parent galaxy of interest
    parent_galaxy_mask = (segm_deblend_v3 == deblend_island_num)
    not_parent_galaxy_mask = (segm_deblend_v3 != deblend_island_num) & (segm_deblend_v3 > 0)

    ##how many deblended segments are on the main blob?
    num_deblend_segs_main_blob = len(deblend_ids)

    return segm_deblend_v3, num_deblend_segs_main_blob, parent_galaxy_mask.astype(int), not_parent_galaxy_mask,  nearest_deblend_dist_pix, lie_on_smooth_segment


def get_isolate_galaxy_mask(img_rgb=None, img_rgb_mask=None, r_band_data=None, r_rms=None, fiber_xpix=None, fiber_ypix=None, file_path=None,  tgid=None, aperture_mask=None ):
    '''
    Function that returns the deblended segment used for isolating the parent galaxy. 
    Note that this function is only run when z > 0.01, and that flag is applied in the aperture_cogs.py script when this function is called

    Note that the r_band_data here is not the original r band image, but is from the final reconstructed image where some parts are already masked
    '''
    
    npixels_min = 10
    threshold_rms_scale = 1.5

    r_band_data[aperture_mask] = 0 

    ###get the updated deblend values!!
    kernel = make_2dgaussian_kernel(15, size=29)  # FWHM = 3.0

    threshold = threshold_rms_scale * r_rms
    convolved_tot_data = convolve( r_band_data, kernel )
    
    segment_map = detect_sources(convolved_tot_data, threshold, npixels=npixels_min) 
    segment_map_data = segment_map.data

    #we want sources where we detect something!! If we do not detect anything, we return a non valyue
    if np.max(segment_map_data) == 0:
        print(f"Nothing found in this smooth segmentation: {tgid}! Returning Nones. In such cases, we will revert back to the tractor photometry!")
        
        return None, None, None, None, None
    else:
        ncontrast_opt = find_optimal_ncontrast(img_rgb, img_rgb_mask, convolved_tot_data, segment_map, nlevel_val = 4, save_path = file_path)
        
        segm_deblend_opt = deblend_sources(convolved_tot_data,segment_map,
                                           npixels=npixels_min,nlevels=4, contrast=ncontrast_opt,
                                           progress_bar=False)

        segm_deblend_opt, num_deblend_segs_main_blob, parent_galaxy_mask,not_parent_galaxy_mask, nearest_deblend_dist_pix, lie_on_smooth_segment = process_deblend_image(segment_map, segm_deblend_opt, fiber_xpix, fiber_ypix)

        #segm_deblend_opt is the optimally deblended image with just the deblended segments on main blob highlighted
        #num_deblend_segs_main_blob is the number of deblended segments identified in the main blob
        #if it is equal to 1, we do not change anything
        #parent_galaxy_mask is the mask that contains only the parent galaxy of interest 
        #no_parent_galaxy_mask is a mask for the other deblended blobs identified in the main blob and is the mask that we will apply in COG pipeline

        if num_deblend_segs_main_blob == 0:
            raise ValueError(f"Incorrect number of deblended blobs found in main blob in isolate galaxy mask! -> {tgid} ")

    ##let us do a binary dilation on the not_parent_galaxy_mask to avoid some of the extremeities!!
    structure = np.ones((3, 3), dtype=bool)
    not_parent_galaxy_mask = binary_dilation(not_parent_galaxy_mask, structure=structure, iterations=2)
    
    return segm_deblend_opt, num_deblend_segs_main_blob, parent_galaxy_mask, not_parent_galaxy_mask, nearest_deblend_dist_pix
    

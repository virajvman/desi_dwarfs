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

TODO: In this step, we need to update what is being considered the center of the source and use source_ra, source_dec based pixel coordinates

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
    
    
    # the component that contains our fiber
    fiber_label = current_deblend[175,175]
    top_mask = (current_deblend == fiber_label)
    
    fiber_label = prev_deblend[175,175]
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

def find_optimal_ncontrast(convolved_tot_data, segment_map, nlevel_val = 4,verbose=False):
    '''
    In this function, we find the optimal ncontrast value with which we will deblend
    '''
    

    all_segm_deblends = []
    all_jacc_desi = []
    all_jacc_largest = []

    ncontrast = np.arange(0.05,0.2,0.01)
    
    for i in range(len(ncontrast)):
        segm_deblend_ni_map = deblend_sources(convolved_tot_data,segment_map,
                                   npixels=10,nlevels=nlevel_val, contrast=ncontrast[i],
                                   progress_bar=False)
        
        # segm_deblend_ni, segm_deblend_ni_map = get_deblend_segments(convolved_tot_data, segment_map, nlevels = nlevel_val, ncontrast = ncontrast[i])

        if i > 0:
            desi_jacc, large_jacc = measure_jaccard_scores(segm_deblend_ni_map, all_segm_deblends[-1] )
            
            all_jacc_desi.append(desi_jacc)
            all_jacc_largest.append(large_jacc)

        all_segm_deblends.append( segm_deblend_ni_map )
        
        
    all_jacc_desi = np.array(all_jacc_desi)
    all_jacc_largest = np.array(all_jacc_largest)
    
    #we want to identify the first instance of 5 consecutive values of J > 0.95.
    #we stop at hte last one and choose that ncontrast value!!
    if verbose:
        print(all_jacc_desi)
        print(all_jacc_largest)
        print(ncontrast)
    
    _, ncontrast_opt = find_contrast_run(ncontrast, all_jacc_desi, all_jacc_largest, thresh=0.95, run_len=5)
    
    return ncontrast_opt



def process_deblend_image(segment_map, segm_deblend, fiber_xpix, fiber_ypix):
    '''
    Function where we process the deblended image so easier to plot!! Sets all other deblended segments in image to zero and just focuses on the deblended segment of the main blob.
    '''
        
    segment_map_v2 = np.copy(segment_map.data)
    segment_map_v2_copy = np.copy(segment_map_v2)
    
    island_num = segment_map.data[int(fiber_ypix),int(fiber_xpix)]
    
    #pixels that are part of main segment island are called 2
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

    #what if there are blobs in the image, but the desi fiber lies on the background? That is definitely possible
    if deblend_island_num == 0:
        lie_on_smooth_segment=True
    
    else:
        lie_on_smooth_segment=False
        #get the id of the closest segment    
    

    #deblend_seg_id is the segment number associated with the desi fiber! If it lies on top of a deblend segment, it is just the id of that segment
    #if it does not lie on top of any segment, then we will take the closest galaxy
    
    return segm_deblend_v3, deblend_island_num, lie_on_smooth_segment

def find_main_deblend_segment(segm_deblend_opt, fiber_xpix, fiber_ypix):
    '''
    Given all the deblended segments in the main blob, find the one that contains the DESI fiber or the closest one!
    '''

    segm_desi_iso = np.copy(segm_deblend_opt)
    new_desi_blob_id = segm_deblend_opt[ int(fiber_ypix), int(fiber_xpix) ]
    ##identify the one that contains the DESI source and set everything else to zero!!
    segm_desi_iso[segm_desi_iso != new_desi_blob_id] = 0

    if np.max(segm_desi_iso) == 0:
        #that is, all the segments were masked, then we need to find the closest deblend segment to desi fiber
        segm_desi_iso = 

    #this is the mask we will use to identify the tractor sources to consider!
    return segm_desi_iso


def get_isolate_galaxy_mask(r_band_data=None, r_rms=None, fiber_xpix=None, fiber_ypix=None, file_path=None, use_final_reconstruction=False, tgid=None ):
    '''
    Function that returns the deblended segment used for isolating the parent galaxy 
    '''
  # data_row = data_shred_all[data_shred_all["TARGETID"] == plot_tgids[plot_ind]]
    
    # img_path = data_row["IMAGE_PATH"]
    # file_path = data_row["FILE_PATH"]
    
    # source_ra, source_dec = data_row["RA"], data_row["DEC"]
    # data_arr = fits.open(img_path)[0].data
    # wcs = WCS(fits.getheader( img_path))
    # fiber_xpix, fiber_ypix,_ = wcs.all_world2pix(source_ra, source_dec,0,1)

    # #first estimate the background error to use in aperture photometry
    # noise_dict = {}
    
    # data = {"g": data_arr[0],"r":data_arr[1], "z":data_arr[2]}

    #     rms_estimator = MADStdBackgroundRMS()
    # ##estimate the background rms in each band!
    # for bii in ["g","r","z"]:        
    #     # Apply sigma clipping
    #     sigma_clip = SigmaClip(sigma=3.0,maxiters=5)
    #     clipped_data = sigma_clip(data[bii])
    #     # Estimate RMS
    #     background_rms = rms_estimator(clipped_data)
    #     noise_dict[bii] = background_rms

    #the rms in the total image will be the rms in the 3 bands added in quadrature
    # tot_rms = noise_dict["r"]

    npixels_min = 10
    threshold_rms_scale = 1.5

    if use_final_reconstruction:
        #instead of loading the full original image, we load in the final reconstruction!
        #overwrite the previosu data_arr!!
        data_arr = np.load(file_path + "/final_reconstruct_galaxy.npy")
        r_band_data = data_arr[1]
    else:
        #we use the provided r_band_data variable
        #note that here the bad pixel mask and stuff is not applied yet, and so will have to be applied ... 
        pass
    
    ###get the updated deblend values!!
    kernel = make_2dgaussian_kernel(15, size=29)  # FWHM = 3.0

    threshold = threshold_rms_scale * r_rms
    convolved_tot_data = convolve( r_band_data, kernel )
    
    segment_map = detect_sources(convolved_tot_data, threshold, npixels=npixels_min) 
    segment_map_data = segment_map.data

    #we want sources where we detect something!! If we do not detect anything, we return a non valyue
    if np.max(segment_map_data) == 0:
        print(f"Nothing found in this smooth segmentation: {tgid}! Returning Nones")
        return 
    else:
        ncontrast_opt = find_optimal_ncontrast(convolved_tot_data, segment_map, nlevel_val = 4)
        
        segm_deblend_opt = deblend_sources(convolved_tot_data,segment_map,
                                           npixels=npixels_min,nlevels=4, contrast=ncontrast_opt,
                                           progress_bar=False)
        
        segm_deblend_opt = process_deblend_image(segment_map, segm_deblend_opt, fiber_xpix, fiber_ypix)
    
    
        
        #isolate the deblend segment of interest
        find_main_deblend_segment(segm_deblend_opt, fiber_xpix, fiber_ypix)
    

    return segm_deblend_opt
    

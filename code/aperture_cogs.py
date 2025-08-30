'''
In this script, we will do a curve of growth analysis on our objects that are really shreds!

We are working in a different script here as we need the tractor/astrometry packages to construct the psf model!

Basic steps are following what SGA catalog did:
1) Identify the range of apertures within which we will do our photometry
2) Mask relevant pixels (if star or residuals after subtracting model image are very large?)
3) If we have identified sources within aperture that we want to subtract, we can create an image with all the masked pixels and subtracted sources for reference?
'''

from scipy.ndimage import gaussian_filter
from desi_lowz_funcs import make_subplots, sdss_rgb, process_img
import numpy as np
from photutils.aperture import aperture_photometry, EllipticalAperture
from photutils.morphology import data_properties
from astropy import units as u
from matplotlib.colors import LogNorm
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import pickle
from astropy.table import Table
from matplotlib.colors import Normalize
from photutils.background import Background2D, MedianBackground
from photutils.segmentation import detect_sources
from photutils.segmentation import make_2dgaussian_kernel
from astropy.convolution import convolve
import matplotlib.patches as patches
from isolate_galaxy_mask import get_isolate_galaxy_mask
from alternative_photometry_methods import mags_to_flux, flux_to_mag

def set_pixels_within_distance(arr, center, radius, value):
    """
    Set all pixels in `arr` within `radius` of `center` to `value`.

    Parameters:
    - arr (2D np.array): The input array to modify.
    - center (tuple): (row, col) coordinates of the central pixel.
    - radius (float): The radius within which to set the value.
    - value: The value to assign to pixels within the radius.
    """
    rows, cols = arr.shape
    y, x = np.ogrid[:rows, :cols]
    cy, cx = center

    # Compute the distance from each point to the center
    distance = np.sqrt((y - cy)**2 + (x - cx)**2)

    # Mask of points within the specified radius
    mask = distance <= radius

    # Set the value
    arr[mask] = value

    return arr

    
def find_nearest_island(segment_map_v2,fiber_xpix, fiber_ypix):
    '''
    This function is used in the rare case that the DESI fiber is not on top of a segment island detected by photutils. This happens only for the ELGs that are very faint sources

    segment_map_v2 is the segment map by photutils and fiber_xpix, fiber_ypix are the pixel locations of the DESI fiber
    '''

    #if the source fiber lies on a pixel classified as background
    #and if a source island is not found within 10'', we then manually drop a source!
    
    all_xpixs, all_ypixs = np.meshgrid( np.arange(np.shape(segment_map_v2)[0]), np.arange(np.shape(segment_map_v2)[1]) )
    
    all_dists = np.sqrt ( ( all_xpixs - fiber_xpix)**2 + ( all_ypixs - fiber_ypix)**2 )
    
    #get all the distances to the pixels that are not background
    all_segs_notbg = segment_map_v2[ (segment_map_v2 != 0) ]
    all_dists_segpixs = all_dists[ (segment_map_v2 != 0)  ]

    #get distance in arcsec
    if np.min(all_dists_segpixs)*0.262 > 10:
        #if the closest island is more than 10 arcsecs away, we think this source is not associated with our source
        #we manually drop a source and call that our source!

        #we use this island number as will possibly be conflating with any other source in the image!
        island_num = 999999
        segment_map_v2 = set_pixels_within_distance(segment_map_v2, (fiber_ypix, fiber_xpix), 2/0.262, island_num)
        #the source center remains the same

        #TODO :IF THIS IS FLAGGED, WE JUST USE THE PHOTOMETRY OF THE TRACTOR SOURCE!! DO WE NEED THIS??
        
    else:
        island_num = all_segs_notbg[ np.argmin(all_dists_segpixs) ]

        #get the positions of hte closest co-ordinate
        xpix_new = all_xpixs[ (segment_map_v2 != 0) ][np.argmin(all_dists_segpixs) ]
        ypix_new = all_ypixs[ (segment_map_v2 != 0) ][np.argmin(all_dists_segpixs) ]
        #we update the source center with this new co-ordinate
        fiber_xpix = xpix_new
        fiber_ypix = ypix_new
        

    return segment_map_v2, island_num, fiber_xpix, fiber_ypix


def measure_elliptical_aperture_area_fraction(image_shape, ell_aper_obj):
    # aperture: EllipticalAperture object
    # image_shape: (ny, nx)
    
    mask = ell_aper_obj.to_mask(method='exact')  # or method='center' for faster, coarser estimate
    aperture_mask = mask.to_image(image_shape)  # same shape as image
    
    # Compute area fraction
    area_in_image = np.nansum(aperture_mask)         # pixel values are fractions from 0 to 1
    total_aperture_area = ell_aper_obj.area  # exact analytic ellipse area: πab
    
    fraction = area_in_image / total_aperture_area

    return fraction


def get_elliptical_aperture(segment_mask, sigma = 3, aperture_mask = None, id_num = None ):
    '''
    Function that takes in a segment mask and fits an aperture to it!

    Note that the sizes etc. obtained here does not have much to do with actual size of the galaxy. We are just using this to get some fiducial size and 
    and to scale the aperture. If we wanted light weighted aperture, we would have to work with the actual image and not the pixel mask.
    
    '''

    if id_num is not None:
        segment_mask_v2 = np.copy(segment_mask)
        segment_mask_v2[segment_mask != id_num] = 0
        segment_mask_v2[segment_mask == id_num] = 1
        segment_mask = segment_mask_v2
    
    if aperture_mask is not None:
        segment_mask[aperture_mask] = 0
        if np.sum( segment_mask) == 0:
            #in case source is too close to the star, then we do not code to crash so we unmask it again
            segment_mask[aperture_mask] = 1
        
    
    #segment_mask is a binary mask.
    
    
    #use this trick to get properties of the main segment 
    cat = data_properties(segment_mask, mask=None)

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

    area_fraction = measure_elliptical_aperture_area_fraction(segment_mask.shape, aperture)

    return aperture, area_fraction, xypos, [cat.semimajor_sigma.value, b/a, theta]
    

def build_residual_mask(data_images, model_images, main_seg_mask, sigma_threshold=5, smooth_sigma=2):
    """
    Build a residual mask flagging pixels with >5-sigma residuals in any band.
    
    Parameters:
    -----------
    data_images : list of 2D np.ndarray
        List of data images (e.g., [data_g, data_r, data_z]).
    model_images : list of 2D np.ndarray
        List of corresponding model images (e.g., [model_g, model_r, model_z]).
    main_seg_mask : 2d np.ndarray
        Mask corresponding to pixels in the main segment. They are not included in the background model and so might skew the residual mask!
    sigma_threshold : float
        Threshold in sigma units for flagging pixels.
    smooth_sigma : float
        Sigma for Gaussian smoothing (in pixels).
    
    Returns:
    --------
    mask : 2D np.ndarray (bool)
        Boolean mask where True indicates a flagged (masked) pixel.

    Notes:
    ---------
    This tractor model image will have to be for sources that are not in the main segment. That would require us to rerun the tractor pipeline. We will have to thus mask pixels that are in the main segment!

    
    """
    assert len(data_images) == len(model_images), "Mismatch in number of bands"
    
    # Initialize mask
    mask = np.zeros_like(data_images[0], dtype=bool)

    bkg_sigmas = []
    bkg_meds = []
    
    for data, model in zip(data_images, model_images):
        # Compute residual image
        #setting the pixels in the main segment to zero so they dont mess up the residual mask generation!
        data[main_seg_mask] = 0
        residual = data - model

        # Smooth the residual
        smoothed_residual = gaussian_filter(residual, sigma=smooth_sigma)
        
        # Estimate per-pixel sigma using the smoothed residual's standard deviation
        # (Alternative: if you have per-pixel noise maps, you should use them here)
        sigma_estimate = np.std(smoothed_residual)
        
        # Flag pixels where the absolute smoothed residual exceeds threshold
        band_mask = np.abs(smoothed_residual) > (sigma_threshold * sigma_estimate)
        
        # Combine mask across bands (OR logic: mask if flagged in any band)
        
        bkg_sigmas.append(np.std(residual[~band_mask]))
        bkg_meds.append(np.median(residual[~band_mask]))
        
        
        mask |= band_mask

        #compute the std in each background for each filter
    
    return mask, bkg_sigmas, bkg_meds



def cog_function(r,mtot, m0, alpha_1, r_0, alpha_2):
    '''
    This is empirical model for the curve of growth
    '''

    return mtot + m0 * np.log(1 + alpha_1 * ( (r/r_0)**(-alpha_2) ) )



def fit_cog(r_data, m_data, p0, bounds=None, maxfev=1200, filler=np.nan):
    '''
    Function that fits radius and magnitude
    '''
    # Filter out NaNs, infinite values
    valid =  np.isfinite(r_data) & np.isfinite(m_data)
    r_clean = r_data[valid]
    m_clean = m_data[valid]

    # If too few points, return filler
    if len(r_clean) < len(p0):
        print("Not enough valid data points. Returning filler values.")
        return np.full(len(p0), filler), np.full(len(p0), filler), np.nan, np.nan

    try:
        popt, pcov = curve_fit(
            cog_function, r_clean, m_clean, p0=p0,
            bounds=bounds or (-np.inf, np.inf),
            maxfev=maxfev
        )

        #convert the covariance matrix into parameter uncertainties
        perr = np.sqrt(np.diag(pcov))

        #use these fitted parameters to measure a chi2
        residuals = m_clean - cog_function(r_clean, *popt)
        chi2 = np.sum((residuals) ** 2)
        dof = len(r_clean) - len(popt)
        chi2_red = chi2 / dof if dof > 0 else np.nan
        
        return popt, perr, chi2, dof
        
    except RuntimeError:
        print("Fit failed. Returning filler values.")
        return np.full(len(p0), filler), np.full(len(p0), filler), np.nan, np.nan



def find_largest_blob(segment_map):
    '''
    Function that finds the largest blob!!
    '''

    # The areas (number of pixels) of each segment
    areas = segment_map.areas  # numpy array of pixel counts per segment
    
    # Find index of the largest one
    largest_idx = areas.argmax()

    # Get the corresponding segment ID number
    largest_id = segment_map.labels[largest_idx]

    if largest_id == 0:
        raise ValueError("The background is being assigned the largest segment!")

    #use this to construct the mask of just the largest blob
    segment_map_data = segment_map.data
    largest_blob_mask = (segment_map_data == largest_id)

    return largest_blob_mask.astype(int), largest_id
    

def get_new_segment(data, fiber_xpix, fiber_ypix, tot_noise_rms,  npixels_min, threshold_rms_scale, aperture_mask):
    '''
    In this function, the goal is to identify the number of segments that are found in the latest reconstructed segment. If lots of segments are found, we flag that as possible bad photo
    For z<0.01 sources, we also use this to estimate the aperture!
    '''
    #combine the g+r+z 
    tot_data = np.sum(data, axis=0)
    
    masked_data = np.copy(tot_data)
    #we set the aperture mask values to be very low so masked pixels are not identified as part of the segment
    masked_data[aperture_mask] = 0 #-1e10

    threshold = threshold_rms_scale * tot_noise_rms

    kernel = make_2dgaussian_kernel(3.0, size=5)  # FWHM = 3.0
    kernel_smooth = make_2dgaussian_kernel(15, size=29)  # FWHM = 3.0

    convolved_tot_data = convolve( masked_data, kernel )
    convolved_tot_data_smooth = convolve( masked_data, kernel_smooth )

    #we will definitely find somewhere here!
    segment_map_new = detect_sources(convolved_tot_data, threshold, npixels=npixels_min) 
    #if we do not find anything, that is good to know!!
    if segment_map_new is None:
        cog_num_seg = 0
    else:
        #we subtract 1 as there is the background
        cog_num_seg = len(np.unique(segment_map_new.data)) - 1

    segment_map_smooth_new = detect_sources(convolved_tot_data_smooth, threshold, npixels=npixels_min) 
    #if we do not find anything, that is good to know!!
    if segment_map_smooth_new is None:
        cog_num_seg_smooth = 0
    else:
        #we subtract 1 as there is the background
        cog_num_seg_smooth = len(np.unique(segment_map_smooth_new.data)) - 1

    #####
    #FIND THE BLOB ON WHICH WE WILL DO THE APERTURE FITTING
    if cog_num_seg_smooth > 0:
        #if a smooth component is found, we choose the largest smooth component
        largest_blob, _ = find_largest_blob(segment_map_smooth_new)
    else:
        if cog_num_seg != 0:
            #in this case as well, we will eventuall revert back to, but just push it forward through the aperture pipeline in case!
            largest_blob, _ = find_largest_blob(segment_map_new)
        else:
            #we do not run COG at all and simply revert back to the tractor at the end!
            largest_blob = None
  

    return largest_blob, cog_num_seg_smooth, cog_num_seg, segment_map_smooth_new


def longest_run(mask):
    # pad with False at both ends to detect run boundaries
    padded = np.pad(mask.astype(int), (1, 1), constant_values=0)
    changes = np.diff(padded)
    starts = np.where(changes == 1)[0]
    ends   = np.where(changes == -1)[0]
    lengths = ends - starts
    if len(lengths) == 0:
        return 0, None, None
    max_idx = np.argmax(lengths)
    return lengths[max_idx] + 1, starts[max_idx], ends[max_idx]  # +1 to include full streak

def detect_cog_decrease(cog_mags_bi):
    '''
    Function that identifies whether magnitudes are decreasing in the COG
    '''
    #we first invert it as are dealing with mags and drop in mag is an actaul increase in the number

    cog_mags_bi *= -1
    
    diffs = np.diff(cog_mags_bi)

    # A decrease means next value is smaller
    is_decreasing = diffs < 0
    
    # Apply
    length, start, end = longest_run(is_decreasing)

    if length > 0:
        drop_mag = cog_mags_bi[start] - cog_mags_bi[end]
    else:
        drop_mag = 0
    
    return length, drop_mag


def return_blank_values():
    '''
    Function that will return blank values for the run_cogs function
    '''

    aper_mags_r4_blank = [np.nan] * 3
    final_cog_mtot_blank = [np.nan] * 3
    final_cog_mtot_err_blank = [np.nan] * 3
    final_cog_params_blank = None
    final_cog_params_err_blank = None

    final_cog_chi2_blank = [np.nan] * 3
    final_cog_dof_blank = [np.nan] * 3
    areafrac_in_image_largest_aper_blank = np.nan
    areafrac_in_image_largest_aper_blank = np.nan
    save_img_path_blank = ""
    decrease_cog_len_blank = [np.nan] * 3
    decrease_cog_mag_blank = [np.nan] * 3
    aper_world_pos_blank = [np.nan, np.nan]
    aper_pix_pos_blank = [np.nan, np.nan]

    aper_params_blank = [np.nan, np.nan, np.nan]
    

    return aper_mags_r4_blank, final_cog_mtot_blank, final_cog_mtot_err_blank, final_cog_params_blank, final_cog_params_err_blank, final_cog_chi2_blank, final_cog_dof_blank, areafrac_in_image_largest_aper_blank,  save_img_path_blank, decrease_cog_len_blank, decrease_cog_mag_blank, aper_world_pos_blank, aper_pix_pos_blank, aper_params_blank

    
    
def get_tractor_only_mag(non_parent_galaxy_mask, save_path):
    '''
    Function where we use the not-parent galaxy mask to get a tractor only mag!! If no not parent objects were found in the smooth deblended stage, we just add source flux without removing any more!
    '''

    ##using the non-parent galaxy mask, remove all the tractor sources that lie on that to get a tractor only photometry!!
    parent_source_catalog = Table.read(save_path + "/parent_galaxy_sources.fits")

    print(len(parent_source_catalog))
    
    #remove sources that lie on the non parent galaxy mask

    #TODO: ONLY FOCUS ON SOURCES ON THE PARENT SEGMENT AS IT WILL CONTAIN MAJORITY OF THE SOURCES, DO NOT NEED TO JUST REMOVE OUTISDE MAIN DEBLEND BLOB
    #TODO: In a step, save all the tractor source models of sources on the parent segment and so we can at the end flexibly make model iamges :) 
    
    not_parent_remove_mask = (non_parent_galaxy_mask[parent_source_catalog["ypix"].data.astype(int), parent_source_catalog["xpix"].data.astype(int)] == True)

    parent_source_catalog = parent_source_catalog[~not_parent_remove_mask]

    print(len(parent_source_catalog))
    
    parent_source_catalog.write( save_path + "/parent_galaxy_sources_FINAL.fits", overwrite=True)
    
    #in the mean time, measure the simplest photometry!
    g_flux_corr = mags_to_flux(parent_source_catalog["mag_g"]) / parent_source_catalog["mw_transmission_g"]
    r_flux_corr = mags_to_flux(parent_source_catalog["mag_r"]) / parent_source_catalog["mw_transmission_r"]
    z_flux_corr = mags_to_flux(parent_source_catalog["mag_z"]) / parent_source_catalog["mw_transmission_z"]
    
    tot_g_mag = flux_to_mag(np.sum(g_flux_corr))
    tot_r_mag = flux_to_mag(np.sum(r_flux_corr))
    tot_z_mag = flux_to_mag(np.sum(z_flux_corr))

    return [tot_g_mag, tot_r_mag, tot_z_mag], parent_source_catalog


def run_cogs(data_arr, segment_map_v2, star_mask, aperture_mask, tgid, tractor_mags,save_path, subtract_source_pos, pcnn_val, npixels_min, threshold_rms_scale, wcs, source_zred=None):
    '''
    In this function, we run the curve-of-growth (COG) analysis!
    '''
    #load the tractor background model, main segment map, and the model of blended sources that is to be removed
    tractor_bkg_model = np.load(save_path + "/tractor_background_model.npy")
    segm_deblend_v3  = np.load(save_path + "/main_segment_map.npy")
    tractor_blend_model = np.load(save_path + "/tractor_blend_remove_model.npy")
    #note the blend_model can just be an array of zeroes if nothing is to be removed
    #we loading the original position of the fiber
    fiber_xpix, fiber_ypix = np.load( save_path + "/fiber_pix_pos_org.npy" )
    tot_noise_rms = np.load(save_path + "/tot_noise_rms.npy")
    #this is the estimated background rms in the image!
    noise_rms_perband = np.load(save_path + "/noise_per_band_rms.npy")

    #when computing the residual mask, we will be setting the pixels in the main segment to zero
    #another approach is to set pixels within the aperture to zero, but that seems a bit more arbitrary
    data_arr_copy = np.copy(data_arr)
    main_seg_pix_locs = ~np.isnan(segm_deblend_v3)
    data_arr_copy[:, main_seg_pix_locs] = 0

    #this is a mask where True means it is a masked pixel
    #we need to mask the main segment when doing this! segm_deblend_v3 will not be nan in the main segment pixels!
    main_seg_mask = (segm_deblend_v3 != np.nan)
    resid_5sig_mask, _, _ = build_residual_mask(data_arr_copy, tractor_bkg_model, main_seg_mask, sigma_threshold=5, smooth_sigma=2)

    #we will apply this mask to our data when computing the curve of growth analysis!
    #we need to combine this mask with our existing aperture mask
    #we have to invert it though as were definigin aperture_mask=0 as pixels to be masked
    aperture_mask = ~aperture_mask.astype(bool)
    #then we combine such that if one of these masks indicates it should be masked we mask it
    aperture_mask |= resid_5sig_mask

    ##compute the image which has the background model subtracted
    data_arr_no_bkg_no_blend = data_arr - tractor_bkg_model - tractor_blend_model
    #subtract the model of the blended sources that are not part of the dwarf galaxy!
    # data_no_bkg = {"g": data_arr_no_bkg[0], "r": data_arr_no_bkg[1] , "z": data_arr_no_bkg[2] }
    data_no_bkg_no_blend = {"g": data_arr_no_bkg_no_blend[0], "r": data_arr_no_bkg_no_blend[1] , "z": data_arr_no_bkg_no_blend[2] }

    #it is possible that the subtracted sources resulted in a very negative value and that will mess up with the aperture mags
    #so we mask pixels that have pixel values that are 5 sigma below the background value!
    # aperture_mask_2 = (data_arr_no_bkg_no_blend < -5 * noise_rms_perband[:, None, None]).any(axis=0)
    aperture_mask_2 = (data_arr_no_bkg_no_blend[0] < -5 * noise_rms_perband[0]) | (data_arr_no_bkg_no_blend[1] < -5 * noise_rms_perband[1])  | (data_arr_no_bkg_no_blend[2] < -5 * noise_rms_perband[2]) 
    
    aperture_mask |= aperture_mask_2

    #Note that the aperture mask already contains the star mask. Check the aperture_photo.py script

    reconstruct_data = np.copy(data_arr_no_bkg_no_blend)
    reconstruct_data[:,aperture_mask] = 0
    np.save(save_path + "/final_reconstruct_galaxy_no_cog_mask.npy", reconstruct_data)
   
    ##get the current best estimate of the aperture blob
    largest_blob, cog_num_seg_smooth, cog_num_seg, segment_map_smooth_new = get_new_segment(reconstruct_data, fiber_xpix, fiber_ypix, tot_noise_rms,  npixels_min, threshold_rms_scale, aperture_mask)

    if source_zred < 0.01:
        #we do not apply any isolate galaxy mask
        #and just continue as usual with our COG approach
        smooth_segment_found = True
        parent_galaxy_mask = largest_blob
        final_cog_mask = star_mask | aperture_mask

    else:
        rgb_img = sdss_rgb([data_arr[0],data_arr[1],data_arr[2]])
        rgb_img_mask =  sdss_rgb([reconstruct_data[0], reconstruct_data [1], reconstruct_data [2]])
        
        #we construct the smooth galaxy mask using the r-band image of the latest reconstruction so far!
        segm_smooth_deblend_opt, num_deblend_segs_main_blob, parent_galaxy_mask, non_parent_galaxy_mask, nearest_deblend_dist_pix = get_isolate_galaxy_mask(img_rgb = rgb_img, img_rgb_mask = rgb_img_mask, r_band_data=reconstruct_data[1], r_rms=noise_rms_perband[1], fiber_xpix=fiber_xpix, fiber_ypix=fiber_ypix, file_path=save_path, tgid=tgid, aperture_mask = aperture_mask )

        if segm_smooth_deblend_opt is not None:
            #a smooth segment was found!
            smooth_segment_found = True
            
            if num_deblend_segs_main_blob > 1:
                #if more than 1 deblended blobs were found, we use the non_parent_galaxy_mask for masking in COG
                #and we will use the parent_galaxy_mask to isolate tractor sources!
                
                #we will use the parent_galaxy_mask to estimate the aperture!!

                #only pixels that are in other deblended blobs and not the main deblend blob are true 
                non_parent_galaxy_mask_for_cog = non_parent_galaxy_mask.astype(bool)
            else:
                #only 1 is found and we do not do anything and keep everything as is. 
                #We just construct a mask full of Falses, as when we apply this, we will not be masking anything!
                non_parent_galaxy_mask_for_cog = np.zeros_like(parent_galaxy_mask, dtype = bool) 
                #we will still use the parent_galaxy_mask to estimate the aperture
        else:
            #no smooth segment was found. This means that we will be reverting to Tractor source photometry as our final source magnitude!            
            smooth_segment_found = False

            non_parent_galaxy_mask_for_cog = np.zeros_like(parent_galaxy_mask, dtype = bool) 

            parent_galaxy_mask = largest_blob


        #the star mask is already in the aperture_mask, but just included here for completeness.
        #the non_parent_galaxy_mask_for_cog is the mask we will apply to the reconstruct data to get the final version!
        final_cog_mask = star_mask | aperture_mask | non_parent_galaxy_mask_for_cog

    #NOTE: the parent galaxy mask is a binary mask. 1 is on segment, 0 is bkg

    tractor_only_mags, parent_source_catalog = get_tractor_only_mag(non_parent_galaxy_mask_for_cog, save_path)

    reconstruct_data[:,final_cog_mask] = 0

    #construct the reconstruct_data dictionary
    reconstruct_data_dict = { "g": reconstruct_data[0], "r": reconstruct_data[1], "z": reconstruct_data[2] }

    #save this all!!
    rgb_512 = process_img(reconstruct_data, cutout_size = 512, org_size = np.shape(data_arr)[1] )
    rgb_128 = process_img(reconstruct_data, cutout_size = 128, org_size = np.shape(data_arr)[1] )
    rgb_reconstruct_full = process_img(reconstruct_data, cutout_size = None, org_size = np.shape(data_arr)[1] )
        
    np.save(save_path + "/final_mask_cog.npy", final_cog_mask  )

    if parent_galaxy_mask is None:
        #basically, we will not run COG and just return here!!
        np.save(save_path + "/parent_galaxy_segment_mask.npy", np.zeros_like(parent_galaxy_mask)   )

        print("--"*6)
        print(f"TARGETID={tgid} has no detections and thus the segmentation map for COG is None. COG will not be run.")
        print(f"{save_path}")
        print("--"*6)
        return_nans = return_blank_values()
        return return_nans
    
    else:        
        parent_galaxy_mask = parent_galaxy_mask.astype(int)
        
        np.save(save_path + "/parent_galaxy_segment_mask.npy", parent_galaxy_mask)  

        # TO DO: Write a function that makes some diagnostic plots like these given a list of target ids!!    
        # Also, then construct an alternative photometry using the similar color sources on main blob and that just lie on the deblended segment mask!
        
        ##save some diagnostic images!
        ax_256 = make_subplots(ncol=1,nrow = 1)
        ax_256[0].set_title("Reconstructed Image 512x512",fontsize = 12)
        ax_256[0].imshow(rgb_512, origin='lower')
        ax_256[0].set_xlim([0,512])
        ax_256[0].set_ylim([0,512])
        ax_256[0].set_xticks([])
        ax_256[0].set_yticks([])
        plt.savefig(save_path + "/reconstruct_image_512.png",bbox_inches="tight")
        plt.close()
    
        ax_128 = make_subplots(ncol=1,nrow = 1)
        ax_128[0].set_title("Reconstructed Image 128x128",fontsize = 12)
        ax_128[0].imshow(rgb_128, origin='lower')
        ax_128[0].set_xlim([0,128])
        ax_128[0].set_ylim([0,128])
        ax_128[0].set_xticks([])
        ax_128[0].set_yticks([])
        plt.savefig(save_path + "/reconstruct_image_128.png",bbox_inches="tight")
        plt.close()

        
        fig,ax = make_subplots(ncol = 5, nrow = 2, row_spacing = 0.5,col_spacing=1.2, label_font_size = 17,plot_size = 3,direction = "horizontal", return_fig=True)
    
        radii_scale = np.arange(1.25,4.25,0.25)
                
        cog_mags = {"g":[], "r": [], "z": []}

        for scale_i in radii_scale:
            aperture_for_phot_i, areafrac_in_image_i, xypos, aper_params = get_elliptical_aperture( parent_galaxy_mask , sigma = scale_i )
    
            if scale_i == radii_scale[-1]:
                areafrac_in_image_largest_aper = areafrac_in_image_i
    
                #for the xy pos should not change at all between different scale and we just save it at the end
                aper_xpos = xypos[0]
                aper_ypos = xypos[1]
                
            #we only plot it at the edge cases
            if scale_i == 2:
                aperture_for_phot_i.plot(ax = ax[0], color = "k", lw = 1.25, ls = "-",alpha = 1)
            if scale_i == 3:
                aperture_for_phot_i.plot(ax = ax[0], color = "k", lw = 1.25, ls = "--",alpha = 1)
            if scale_i == 4:
                aperture_for_phot_i.plot(ax = ax[0], color = "k", lw = 1.25, ls = "dotted",alpha = 1)
    
            for bi in "grz":
                phot_table_i = aperture_photometry( reconstruct_data_dict[bi] , aperture_for_phot_i, mask = aperture_mask)
                #note that unlike the case in the aperture_photo, we are no longer subtracting the total flux of blended objects from aperture
                #instead, we subtract the blended source model so we accurately doing the COG analysis!!
                new_mag_i = 22.5 - 2.5*np.log10( phot_table_i["aperture_sum"].data[0] )
                cog_mags[bi].append(new_mag_i)
    
    
        #convert this xy position into a ra,dec position using wcs of the image. Also, save the xy position
        aper_ra_cen, aper_dec_cen, _ = wcs.all_pix2world(aper_xpos, aper_ypos, 0, 1)
        
        ##fit the parametric form to these data points
        final_cog_params = {}
        final_cog_params_err = {}
        final_cog_chi2 = []
        final_cog_dof = []
        decrease_cog_len = []
        decrease_cog_mag = []
        
        
        for bi in "grz":
            #for the initial guess, mtot, m0, alpha_1, r_0, alpha_2
            if np.isnan(cog_mags[bi][-1]):
                guess_mag = 17
                guess_low = 10
                guess_high = 25
            else:
                guess_mag = cog_mags[bi][-1]
                guess_low = guess_mag - 1
                guess_high = guess_mag + 1
            
            p0 = [guess_mag, 2.0, 0.5, 1.5, 3]
            
            bounds = ([guess_low, 0, 0, 0.1, 0], [guess_high, 10, 10, 10, 10])
    
            #construct a flag if the cog mags are decreasing!
            cog_decrease_len, cog_decrease_mag = detect_cog_decrease(np.array(cog_mags[bi]))
    
            decrease_cog_len.append(cog_decrease_len)
            decrease_cog_mag.append(cog_decrease_mag)
    
            popt, perr, chi2, dof = fit_cog(radii_scale, np.array(cog_mags[bi]), p0, bounds=bounds, maxfev=1200, filler=np.nan)
    
            #note that popt returns the all the parameter values!
            #we will separately save the final mags too
            
            final_cog_params[bi] = popt
            final_cog_params_err[bi] = perr
            final_cog_chi2.append(chi2)
            final_cog_dof.append(dof)
        
        #this list contains the final magnitudes!
        #if the cog optimization did not work, we just look at the magnitude in the largest aperture radius
        final_cog_mtot = []
        final_cog_mtot_err = []
            
        for bi in "grz":
            if np.isnan(final_cog_params[bi][0]):
                #the cog optimization did not work
                final_cog_mtot.append( cog_mags[bi][-1] )
                final_cog_mtot_err.append( np.nan )
            else:
                final_cog_mtot.append( final_cog_params[bi][0] )
                final_cog_mtot_err.append( final_cog_params_err[bi][0] )
                
        ##if the optimal fits were not found, 
        
        box_size = np.shape(data_arr)[1] 
    
        ##plot the rgb image of actual data 
        ax_id = 5
        ax[ax_id].set_title("IMG",fontsize=12)
        rgb_img = sdss_rgb([data_arr[0],data_arr[1],data_arr[2]], ["g","r","z"], scales=dict(g=(2,6.0), r=(1,3.4), z=(0,2.2)), m=0.03)
        ax[ax_id].imshow(rgb_img, origin='lower',zorder = 0)
        
        circle = patches.Circle( (fiber_xpix, fiber_ypix),7, color='limegreen', fill=False, linewidth=1,ls ="--",zorder = 1)
        ax[ax_id].add_patch(circle)
        ax[ax_id].set_xlim([0,box_size])
        ax[ax_id].set_ylim([0,box_size])
        ax[ax_id].set_xticks([])
        ax[ax_id].set_yticks([])
    
        ##plot rgb data of the bkg tractor model
        ax_id = 6
        ax[ax_id].set_title("Tractor BKG model",fontsize = 12)
        rgb_bkg_model = sdss_rgb([tractor_bkg_model[0],tractor_bkg_model[1],tractor_bkg_model[2]], ["g","r","z"], scales=dict(g=(2,6.0), r=(1,3.4), z=(0,2.2)), m=0.03)
        ax[ax_id].imshow(rgb_bkg_model, origin='lower')
        ax[ax_id].set_xlim([0,box_size])
        ax[ax_id].set_ylim([0,box_size])
        ax[ax_id].set_xticks([])
        ax[ax_id].set_yticks([])
        
        
        ##plot the rgb data - bkg tractor model
        ax_id = 7
        rgb_img_m_bkg_m_blend_model = sdss_rgb([data_arr_no_bkg_no_blend[0],data_arr_no_bkg_no_blend[1],data_arr_no_bkg_no_blend[2]], ["g","r","z"], scales=dict(g=(2,6.0), r=(1,3.4), z=(0,2.2)), m=0.03)
        ax[ax_id].set_title("IMG - BKG - BLEND",fontsize = 12)
        ax[ax_id].imshow(rgb_img_m_bkg_m_blend_model, origin='lower')
        ax[ax_id].set_xlim([0,box_size])
        ax[ax_id].set_ylim([0,box_size])
        ax[ax_id].set_xticks([])
        ax[ax_id].set_yticks([])
    
        ##plotting the new segmentation map!
        ax_id = 8
        ax[ax_id].set_title("Smooth Segment Map",fontsize = 12)
        if segment_map_smooth_new is None:
            segment_map_smooth_new = np.ones_like(segment_map_smooth_new)
        ax[ax_id].imshow(segment_map_smooth_new, origin='lower', cmap="tab20",
                       interpolation='nearest')
        ax[ax_id].set_xlim([0,box_size])
        ax[ax_id].set_ylim([0,box_size])
        ax[ax_id].set_xticks([])
        ax[ax_id].set_yticks([])

        ax_id = 9
        ax[ax_id].set_title("Final Parent Segment Map",fontsize = 12)
        ax[ax_id].imshow(parent_galaxy_mask, origin='lower', cmap="Greys",
                       interpolation='nearest',vmin=0,vmax=1, zorder = 0)

        #overplot the tractor sources that lie on the final segment!
        for i in range(len(parent_source_catalog)):
            ax[ax_id].scatter( parent_source_catalog["xpix"][i] , parent_source_catalog["ypix"][i], s = 10, color = "r",zorder = 1)
    
        ax[ax_id].set_xlim([0,box_size])
        ax[ax_id].set_ylim([0,box_size])
        ax[ax_id].set_xticks([])
        ax[ax_id].set_yticks([])

        
        ##for reference, make the g+r+z plot with the masked pixels and different apertures on it
        ax_id = 0
    
        ax[ax_id].set_title(r"Sum$_{grz}$(Parent Reconstruct)",fontsize = 12)
        combine_img = reconstruct_data[0] + reconstruct_data[1] + reconstruct_data[2]
        #setting the masked pixels to nans
        combine_img[final_cog_mask] = np.nan
        
        #instead of lognorm, let us do linear scaling
        # norm_obj = LogNorm()
        norm_obj = Normalize(vmin=np.nanmin(combine_img), vmax=np.nanmax(combine_img)) 
        ax[ax_id].imshow(combine_img,origin="lower",norm=norm_obj,cmap = "viridis",zorder = 0)
        ax[ax_id].scatter( aper_xpos, aper_ypos, color = "darkorange",marker = "x")
                
        ax[ax_id].set_xlim([0,box_size])
        ax[ax_id].set_ylim([0,box_size])
        ax[ax_id].set_xticks([])
        ax[ax_id].set_yticks([])
        
        ##make the cog plot!
        ax_id = 1
    
        all_cogs = np.concatenate( (cog_mags["g"],cog_mags["r"],cog_mags["z"]) )
        all_cogs = all_cogs[ ~np.isinf(all_cogs) & ~np.isnan(all_cogs)]
    
        rgrid = np.linspace(1,5,100)
    
        if len(all_cogs) > 0:
            ax[ax_id].scatter(radii_scale, cog_mags["g"],color = "#1E88E5",zorder = 1)
            if ~np.isnan(final_cog_params["g"][0]):
                ax[ax_id].plot(rgrid, cog_function(rgrid, *final_cog_params["g"]), color = "#1E88E5",lw =1,zorder = 1  )
    
            ax[ax_id].scatter(radii_scale, cog_mags["r"],color = "#FFC107",zorder = 1)
            
            if ~np.isnan(final_cog_params["r"][0]):
                ax[ax_id].plot(rgrid, cog_function(rgrid, *final_cog_params["r"]), color = "#FFC107",lw =1,zorder = 1  )
            
            ax[ax_id].scatter(radii_scale, cog_mags["z"],color = "#D81B60",zorder = 1)
            
            if ~np.isnan(final_cog_params["z"][0]):
                ax[ax_id].plot(rgrid, cog_function(rgrid, *final_cog_params["z"]), color = "#D81B60",lw =1 ,zorder = 1 )
    
            spacing = 0.3
            ypos = np.min(all_cogs) - 0.125 + 0.1
    
            ax[ax_id].set_title(f"Chi2 : {final_cog_chi2[0]:.2f},{final_cog_chi2[1]:.2f},{final_cog_chi2[2]:.2f}")
            
            ax[ax_id].text(1.15, ypos, "g", weight="bold", color = "#1E88E5",fontsize = 16)
            ax[ax_id].text(1.15 + 0.32, ypos, "r", weight="bold", color = "#FFC107",fontsize = 16)
            ax[ax_id].text(1.15 + 2*spacing, ypos, "z", weight="bold", color = "#D81B60",fontsize = 16)
    
            ax[ax_id].set_ylabel(r"$m(<r)$ mag",fontsize = 14)
            ax[ax_id].set_xlim([1, 5])
            ax[ax_id].vlines(x = 2, ymin=np.min(all_cogs) - 0.25, ymax = np.max(all_cogs) + 0.25, color = "k",ls = "-",lw = 1, alpha = 0.75, zorder = 0)
            ax[ax_id].vlines(x = 3, ymin=np.min(all_cogs) - 0.25, ymax = np.max(all_cogs) + 0.25, color = "k",ls = "--",lw = 1, alpha = 0.75,zorder = 0)
            ax[ax_id].vlines(x = 4, ymin=np.min(all_cogs) - 0.25, ymax = np.max(all_cogs) + 0.25, color = "k",ls = "dotted",lw = 1, alpha = 0.75, zorder = 0)
            
            ax[ax_id].set_ylim([ np.max(all_cogs) + 0.125,np.min(all_cogs) - 0.125  ] )
    

        ##show the cog summary numbers!!
        
        
        ax_id = 2
    
        spacing = 0.085
        start = 0.97
        fsize = 11.5
        ax[ax_id].text(0.05,start,f"Tractor-mag g = {tractor_mags['g']:.2f}",size =fsize,transform=ax[ax_id].transAxes, verticalalignment='top')
        ax[ax_id].text(0.05,start - spacing*1,f"Aper-mag (R4) g = {cog_mags['g'][-1]:.2f}",size = fsize,transform=ax[ax_id].transAxes, verticalalignment='top')
        ax[ax_id].text(0.05,start - spacing*2,f"COG-mag g = {final_cog_params['g'][0]:.2f}",size = fsize,transform=ax[ax_id].transAxes, verticalalignment='top')
        ax[ax_id].text(0.05,start - spacing*3,f"Trac-only-mag g = {tractor_only_mags[0]:.2f}",size = fsize,transform=ax[ax_id].transAxes, verticalalignment='top')
        
        
        ax[ax_id].text(0.05,start - spacing*4,f"Tractor-mag r = {tractor_mags['r']:.2f}",size =fsize,transform=ax[ax_id].transAxes, verticalalignment='top')
        ax[ax_id].text(0.05,start - spacing*5,f"Aper-mag (R4) r = {cog_mags['r'][-1]:.2f}",size = fsize,transform=ax[ax_id].transAxes, verticalalignment='top')
        ax[ax_id].text(0.05,start - spacing*6,f"COG-mag r= {final_cog_params['r'][0]:.2f}",size = fsize,transform=ax[ax_id].transAxes, verticalalignment='top')
        ax[ax_id].text(0.05,start - spacing*7,f"Trac-only-mag r= {tractor_only_mags[1]:.2f}",size = fsize,transform=ax[ax_id].transAxes, verticalalignment='top')
        
    
        ax[ax_id].text(0.05,start - spacing*8,f"Tractor-mag z = {tractor_mags['z']:.2f}",size =fsize,transform=ax[ax_id].transAxes, verticalalignment='top')
        ax[ax_id].text(0.05,start - spacing*9,f"Aper-mag (R4) z = {cog_mags['z'][-1]:.2f}",size = fsize,transform=ax[ax_id].transAxes, verticalalignment='top')
        ax[ax_id].text(0.05,start - spacing*10,f"COG-mag z = {final_cog_params['z'][0]:.2f}",size = fsize,transform=ax[ax_id].transAxes, verticalalignment='top')
        ax[ax_id].text(0.05,start - spacing*11,f"Trac-only-mag z = {tractor_only_mags[2]:.2f}",size = fsize,transform=ax[ax_id].transAxes, verticalalignment='top')
        
        # ax[ax_id].text(0.05,start - spacing*9,f"pCNN = {pcnn_val:.2f}",size = fsize,transform=ax[ax_id].transAxes, verticalalignment='top')
    
        ##show the final summary figure
        ax_id = 3
        ax[ax_id].set_title("Reconstructed Galaxy",fontsize = 12)
        ax[ax_id].imshow(rgb_reconstruct_full, origin='lower',zorder = 0)
        ax[ax_id].scatter( aper_xpos, aper_ypos, color = "darkorange",marker = "x",zorder = 1)
        ax[ax_id].set_xlim([0,box_size])
        ax[ax_id].set_ylim([0,box_size])
        ax[ax_id].set_xticks([])
        ax[ax_id].set_yticks([])

        ##show the final summary figure
        ax_id = 4
        ax[ax_id].set_title("Simple-Photo Galaxy",fontsize = 12)
        #read in the simple photo mask!
        simple_photo_mask = np.load(save_path + "/simplest_photometry_binary_mask.npy")
        data_arr[:, simple_photo_mask == 0] = 0
        rgb_img_simple = sdss_rgb([data_arr[0],data_arr[1],data_arr[2]], ["g","r","z"], scales=dict(g=(2,6.0), r=(1,3.4), z=(0,2.2)), m=0.03)
       
        ax[ax_id].imshow(rgb_img_simple, origin='lower',zorder = 0)
        ax[ax_id].set_xlim([0,box_size])
        ax[ax_id].set_ylim([0,box_size])
        ax[ax_id].set_xticks([])
        ax[ax_id].set_yticks([])
    
        ##saving this image :0
        save_img_path = save_path + "/cog_summary.png"
        plt.savefig( save_img_path ,bbox_inches="tight")
        # plt.savefig(f"/pscratch/sd/v/virajvm/temp_cog_plots/cog_{tgid}.png" ,bbox_inches="tight")
        plt.close()
    
        aper_mags_r4 = [ cog_mags["g"][-1], cog_mags["r"][-1], cog_mags["z"][-1] ] 
    
        #we are returning dictionaries here
        return aper_mags_r4, final_cog_mtot, final_cog_mtot_err, final_cog_params, final_cog_params_err, final_cog_chi2, final_cog_dof, areafrac_in_image_largest_aper,  save_img_path, decrease_cog_len, decrease_cog_mag, [aper_ra_cen, aper_dec_cen], [aper_xpos, aper_ypos], aper_params




def run_cog_pipe(input_dict):
    '''
    Function to run the curve of growth analysis once the initial aperture photometry pipe has been run!
    '''
    save_path = input_dict["save_path"]
    source_tgid  = input_dict["tgid"]
    data_arr  = input_dict["image_data"]
    pcnn_val = input_dict["pcnn_val"]
    npixels_min = input_dict["npixels_min"]
    threshold_rms_scale = input_dict["threshold_rms_scale"]
    wcs = input_dict["wcs"]
    source_zred  = input_dict["redshift"]
    
    #loading all the relevant data!
    segment_map_v2 = np.load(save_path + "/segment_map_v2.npy")
    star_mask = np.load(save_path + "/star_mask.npy")
    aperture_mask = np.load(save_path + "/aperture_mask.npy") #this aperture mask already contains the mask for the saturated pixels.
    tractor_mags_arr = np.load(save_path + "/tractor_mags.npy")
    tractor_mags = { "g": tractor_mags_arr[0], "r": tractor_mags_arr[1], "z": tractor_mags_arr[2]  }
    mw_trans = np.load(save_path + "/source_cat_obs_transmission.npy")

    with open(save_path + '/subtract_source_pos.pkl', 'rb') as f:
        subtract_source_pos = pickle.load(f)


    #feeding it to the cog function!
    ## MAYBE THIS RUN COGS FUNCTION IS NOT NEDED!
    aper_mags_r4, final_cog_mags, final_cog_mags_err, final_cog_params, final_cog_params_err, final_cog_chi2, final_cog_dof, areafrac_in_image, save_img_path,  decrease_cog_len, decrease_cog_mag, aper_world_coord, aper_pix_coord, aper_params = run_cogs(data_arr, segment_map_v2, star_mask, aperture_mask, source_tgid, tractor_mags, save_path, subtract_source_pos, pcnn_val, npixels_min, threshold_rms_scale, wcs, source_zred = source_zred )

    if np.isnan(aper_world_coord[0]) and np.isnan(final_cog_mags[0]):     
        #the outputs was None!!
        return {
        "aper_mags_r4": 3*[np.nan],
        "cog_mags": 3*[np.nan],
        "cog_mags_err": 3*[np.nan],
        "params_g": 5*[np.nan],
        "params_r": 5*[np.nan],
        "params_z": 5*[np.nan],
        "params_g_err": 5*[np.nan],
        "params_r_err": 5*[np.nan],
        "params_z_err": 5*[np.nan],
        "img_path": None,
        "cog_chi2" : 3*[np.nan], 
        "cog_dof" : 3*[np.nan],
        "areafrac_in_image": np.nan,
        "cog_decrease_len": 3*[np.nan], 
        "cog_decrease_mag": 3*[np.nan],
        "aper_ra_cen": np.nan,
        "aper_dec_cen": np.nan,
        "aper_xpix_cen": np.nan,
        "aper_ypix_cen": np.nan,
        "aper_params": 3*[np.nan]}


    else:
        #mw extinction correcting the magnitudes!
        cog_mag_g = final_cog_mags[0] + 2.5 * np.log10(mw_trans[0])
        cog_mag_r = final_cog_mags[1] + 2.5 * np.log10(mw_trans[1])
        cog_mag_z = final_cog_mags[2] + 2.5 * np.log10(mw_trans[2])
    
        all_aper_mags_r4 = []
        for i in range(3):
            aper_mag_i = aper_mags_r4[i] + 2.5 * np.log10(mw_trans[i])
            all_aper_mags_r4.append(aper_mag_i)
    
        #saving the magnitudes locally in folder
        cog_mags = [cog_mag_g, cog_mag_r, cog_mag_z]
        np.save(save_path + "/new_cog_mags.npy", cog_mags)
    
        np.save(save_path + "/aperture_mags_R4.npy", all_aper_mags_r4)
        
        #we return these final cog mags and append them to the catalog
        #we are expanding out the dictionaries as we do not want to returns dicts
        return {
        "aper_mags_r4": all_aper_mags_r4,
        "cog_mags": cog_mags,
        "cog_mags_err": final_cog_mags_err,
        "params_g": final_cog_params["g"],
        "params_r": final_cog_params["r"],
        "params_z": final_cog_params["z"],
        "params_g_err": final_cog_params_err["g"],
        "params_r_err": final_cog_params_err["r"],
        "params_z_err": final_cog_params_err["z"],
        "img_path": save_img_path,
        "cog_chi2" : final_cog_chi2, 
        "cog_dof" : final_cog_dof,
        "areafrac_in_image": areafrac_in_image,
        "cog_decrease_len": decrease_cog_len, 
        "cog_decrease_mag": decrease_cog_mag,
        "aper_ra_cen": aper_world_coord[0],
        "aper_dec_cen": aper_world_coord[1],
        "aper_xpix_cen": aper_pix_coord[0],
        "aper_ypix_cen": aper_pix_coord[1],
        "aper_params": aper_params}
     
    


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

def mean_surface_brightness(mag, ellip_area):
    '''
    Function to compute surface brightness. We will use this in the r-band data and the input is the area in arcsec2 of the ellipse in which we are measuring the rough photo
    '''
    return mag + 2.5*np.log10(ellip_area)


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
        #IF THE DISTANCE IS MORE THAN 10'', then we just revert back to the original tractor source!
        #This is an arbitrary distance cutoff. For smaller distances, we will be keeping tabs on the distance for reference

        return None, None, None, None, np.nan
        
    else:
        island_num = all_segs_notbg[ np.argmin(all_dists_segpixs) ]

        #get the positions of hte closest co-ordinate
        xpix_new = all_xpixs[ (segment_map_v2 != 0) ][np.argmin(all_dists_segpixs) ]
        ypix_new = all_ypixs[ (segment_map_v2 != 0) ][np.argmin(all_dists_segpixs) ]
        #we update the source center with this new co-ordinate
        fiber_xpix = xpix_new
        fiber_ypix = ypix_new
        
    return segment_map_v2, island_num, fiber_xpix, fiber_ypix, np.min(all_dists_segpixs)


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


    ##estimate a very simple apertyre on this largest blob
    if largest_blob is not None:
        #there does exist a smooth blob.
        largest_blob_temp = np.copy(largest_blob)
        
        #estimate aperture and then scale it by 2. Measure average surface brightness within this
        aperture_temp, _,_,aper_param_temp = get_elliptical_aperture(largest_blob_temp, sigma = 2)

        #if I save this aperture, how easy is it to plot it back?
        
        #measure photometry within this rough aperture!
        phot_temp_g = aperture_photometry( data[0] , aperture_temp, mask = aperture_mask)
        phot_temp_r = aperture_photometry( data[1] , aperture_temp, mask = aperture_mask)
        phot_temp_z = aperture_photometry( data[2] , aperture_temp, mask = aperture_mask)

        g_mag_temp = 22.5 - 2.5*np.log10( phot_temp_g["aperture_sum"].data[0] )
        r_mag_temp = 22.5 - 2.5*np.log10( phot_temp_r["aperture_sum"].data[0] )
        z_mag_temp = 22.5 - 2.5*np.log10( phot_temp_z["aperture_sum"].data[0] )

        
        #get the area of this aperture in arcsec^2 so convert the pixel sizes to arcsec! Area = pi*a*b
        aper_temp_area = np.pi * (aper_param_temp[0]*0.262) * (aper_param_temp[0]*aper_param_temp[1]*0.262)
        
        g_mu_rough = mean_surface_brightness( g_mag_temp,  aper_temp_area )
        r_mu_rough = mean_surface_brightness( r_mag_temp,  aper_temp_area )
        z_mu_rough = mean_surface_brightness( z_mag_temp,  aper_temp_area )
        
    else:
        r_mur_rough = 0
        z_mur_rough = 0

    return largest_blob, cog_num_seg_smooth, cog_num_seg, segment_map_smooth_new, [g_mu_rough, r_mu_rough, z_mu_rough]


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

    
def get_tractor_only_mag(parent_galaxy_mask, save_path, width, tgid):
    '''
    Function where we use the not-parent galaxy mask to get a tractor only mag!! If no not parent objects were found in the smooth deblended stage, we just add source flux without removing any more!
    '''

    ##using the non-parent galaxy mask, remove all the tractor sources that lie on that to get a tractor only photometry!!
    parent_source_catalog = Table.read(save_path + "/parent_galaxy_sources.fits")

    #only include sources that lie on the main segment as identified by the parent galaxy_mask

    parent_keep_mask = (parent_galaxy_mask[parent_source_catalog["ypix"].data.astype(int), parent_source_catalog["xpix"].data.astype(int)] == True)

    #and always include the source of target!
    source_target = (parent_source_catalog["separations"] < 1)

    parent_source_catalog_f = parent_source_catalog[parent_keep_mask | source_target]

    if len(parent_source_catalog_f) == 0:
        print(f"No parent sources found hmm : {len(parent_source_catalog_f)}, {tgid}, {len(parent_source_catalog)}")
        #there is this one weird example 39627783936151447 where the source that is targeted is in Gaia DR2, but is clearly a small, extended galaxy.
        #just return nans :) 
        return [np.nan, np.nan, np.nan], parent_source_catalog_f
        
    ##MAKE RGB IMAGE OF THE TRACTOR ONLY RECONSTRUCTION
    ##with these remaining sources, we can combine their source models in the folder to get the model!
    total_model = np.zeros((3, width, width))
    for pi in range(len(parent_source_catalog_f)):
        objidi = parent_source_catalog_f["source_objid_new"].data[pi]
        tractor_model_path = save_path + f"/tractor_models/tractor_parent_source_model_{objidi}.npy"
        #load it!
        model_i = np.load(tractor_model_path)
        total_model += model_i
    ##save the final parent galaxy tractor only model
    rgb_data = sdss_rgb(total_model, ["g","r","z"])
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    ax.set_title("Parent Galaxy Tractor Only Model",fontsize = 13)
    ax.imshow(rgb_data, origin="lower")
    fig.savefig(f"{save_path}/parent_galaxy_tractor_only_reconstruction.png")
    plt.close(fig)
    
    parent_source_catalog_f.write( save_path + "/parent_galaxy_sources_FINAL.fits", overwrite=True)
    
    #in the mean time, measure the simplest photometry!
    g_flux_corr = mags_to_flux(parent_source_catalog_f["mag_g"]) / parent_source_catalog_f["mw_transmission_g"]
    r_flux_corr = mags_to_flux(parent_source_catalog_f["mag_r"]) / parent_source_catalog_f["mw_transmission_r"]
    z_flux_corr = mags_to_flux(parent_source_catalog_f["mag_z"]) / parent_source_catalog_f["mw_transmission_z"]
    
    tot_g_mag = flux_to_mag(np.sum(g_flux_corr))
    tot_r_mag = flux_to_mag(np.sum(r_flux_corr))
    tot_z_mag = flux_to_mag(np.sum(z_flux_corr))

    return [tot_g_mag, tot_r_mag, tot_z_mag], parent_source_catalog_f



def cog_fitting_subfunction(same_input_dict,reconstruct_galaxy_dict, parent_galaxy_mask, final_cog_mask,  img_flag = None):
    '''
    The function that actually runs the cog fitting. We will compute two kinds of COG mags: both with and without the smooth parent blob mask.

    The reconstruct_data_dict is the specific kind of data we are running with !

    The parent_galaxy_mask is the mask on which we estimate the aperture
    final_cog_mask here is just for visualization purposes, but is the total mask of all pixels that we mask in our cog analysis. It is already applied to the data_dict

    '''

    data_arr = same_input_dict["data_arr"]
    tgid = same_input_dict["tgid"]
    save_path = same_input_dict["save_path"]
    parent_source_catalog = same_input_dict["parent_source_catalog"]
    tractor_only_mags = same_input_dict["tractor_only_mags"]
    tractor_dr9_mags = same_input_dict["tractor_dr9_mags"]
    tractor_bkg_model = same_input_dict["tractor_bkg_model"]
    data_arr_no_bkg_no_blend = same_input_dict["data_arr_no_bkg_no_blend"]
    segment_map_smooth_new = same_input_dict["segment_map_smooth_new"]
    wcs = same_input_dict["wcs"]
    fiber_xpix = same_input_dict["fiber_xpix"]
    fiber_ypix = same_input_dict["fiber_ypix"]
        
    # TO DO: Write a function that makes some diagnostic plots like these given a list of target ids!!    

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
            phot_table_i = aperture_photometry( reconstruct_galaxy_dict[bi] , aperture_for_phot_i)
            #mask is not needed here as the data is already masked!
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
    final_cog_mtot = []
    final_cog_mtot_err = []

    for bi in "grz":
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
        segment_map_smooth_new = np.ones_like(data_arr[0])
    else:
        segment_map_smooth_new = segment_map_smooth_new.data
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
    combine_img = reconstruct_galaxy_dict["g"] + reconstruct_galaxy_dict["r"] + reconstruct_galaxy_dict["z"]
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
    ax[ax_id].text(0.05,start,f"Trac-DR9-mag g = {tractor_dr9_mags['g']:.2f}",size =fsize,transform=ax[ax_id].transAxes, verticalalignment='top')
    ax[ax_id].text(0.05,start - spacing*1,f"Aper-mag (R4) g = {cog_mags['g'][-1]:.2f}",size = fsize,transform=ax[ax_id].transAxes, verticalalignment='top')
    ax[ax_id].text(0.05,start - spacing*2,f"COG-mag g = {final_cog_params['g'][0]:.2f}",size = fsize,transform=ax[ax_id].transAxes, verticalalignment='top')
    ax[ax_id].text(0.05,start - spacing*3,f"Trac-only-mag g = {tractor_only_mags[0]:.2f}",size = fsize,transform=ax[ax_id].transAxes, verticalalignment='top')
    
    
    ax[ax_id].text(0.05,start - spacing*4,f"Trac-DR9-mag r = {tractor_dr9_mags['r']:.2f}",size =fsize,transform=ax[ax_id].transAxes, verticalalignment='top')
    ax[ax_id].text(0.05,start - spacing*5,f"Aper-mag (R4) r = {cog_mags['r'][-1]:.2f}",size = fsize,transform=ax[ax_id].transAxes, verticalalignment='top')
    ax[ax_id].text(0.05,start - spacing*6,f"COG-mag r= {final_cog_params['r'][0]:.2f}",size = fsize,transform=ax[ax_id].transAxes, verticalalignment='top')
    ax[ax_id].text(0.05,start - spacing*7,f"Trac-only-mag r= {tractor_only_mags[1]:.2f}",size = fsize,transform=ax[ax_id].transAxes, verticalalignment='top')
    

    ax[ax_id].text(0.05,start - spacing*8,f"Trac-DR9-mag z = {tractor_dr9_mags['z']:.2f}",size =fsize,transform=ax[ax_id].transAxes, verticalalignment='top')
    ax[ax_id].text(0.05,start - spacing*9,f"Aper-mag (R4) z = {cog_mags['z'][-1]:.2f}",size = fsize,transform=ax[ax_id].transAxes, verticalalignment='top')
    ax[ax_id].text(0.05,start - spacing*10,f"COG-mag z = {final_cog_params['z'][0]:.2f}",size = fsize,transform=ax[ax_id].transAxes, verticalalignment='top')
    ax[ax_id].text(0.05,start - spacing*11,f"Trac-only-mag z = {tractor_only_mags[2]:.2f}",size = fsize,transform=ax[ax_id].transAxes, verticalalignment='top')
    
    # ax[ax_id].text(0.05,start - spacing*9,f"pCNN = {pcnn_val:.2f}",size = fsize,transform=ax[ax_id].transAxes, verticalalignment='top')

    ##show the final summary figure
    ax_id = 3
    ax[ax_id].set_title("Reconstructed Galaxy",fontsize = 12)
    rgb_reconstruct_full = process_img( [reconstruct_galaxy_dict["g"],reconstruct_galaxy_dict["r"],reconstruct_galaxy_dict["z"]], cutout_size = None, org_size = np.shape(data_arr)[1] )
    
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
    save_img_path = save_path + f"/cog_summary{img_flag}.png"
    plt.savefig( save_img_path ,bbox_inches="tight")
    # plt.savefig(f"/pscratch/sd/v/virajvm/temp_cog_plots/cog_{tgid}.png" ,bbox_inches="tight")
    plt.close()

    aper_mags_r425 = [ cog_mags["g"][-1], cog_mags["r"][-1], cog_mags["z"][-1] ] 


    ##MW EXTINCTION CORRECTING THE MAGNITUDES!

    #if the mags are nans, they will just carry through everything!!
    #Correcting mags for MW extinction

    mw_trans = np.load(save_path + "/source_cat_obs_transmission.npy")

    cog_mags_mwc = []
    for i in range(3):
        cog_mag_i = final_cog_mtot[i] + 2.5 * np.log10(mw_trans[i])
        cog_mags_mwc.append(cog_mag_i)
        
    aper_mags_r425_mwc = []
    for i in range(3):
        aper_mag_i = aper_mags_r425[i] + 2.5 * np.log10(mw_trans[i])
        aper_mags_r425_mwc.append(aper_mag_i)

    #saving the magnitudes locally in folder
    np.save(save_path + f"/new_cog_mags{img_flag}.npy", cog_mags_mwc)

    np.save(save_path + f"/aperture_mags_R4{img_flag}.npy", aper_mags_r425_mwc)

    #we are returning dictionaries here
    return_dict = {
        "aper_r425_mags": aper_mags_r425_mwc,
        "cog_mags": cog_mags_mwc,
        "cog_mags_err": final_cog_mtot_err,
        "cog_params_g": final_cog_params["g"],
        "cog_params_g_err": final_cog_params_err["g"],
        "cog_params_r": final_cog_params["r"],
        "cog_params_r_err": final_cog_params_err["r"], 
        "cog_params_z": final_cog_params["z"],
        "cog_params_z_err": final_cog_params_err["z"],
        "cog_chi2" :final_cog_chi2, 
        "cog_dof" :final_cog_dof,
        "cog_decrease_len" : decrease_cog_len,
        "cog_decrease_mag" : decrease_cog_mag, 
        "aper_r425_frac_in_image" :areafrac_in_image_largest_aper,
        "img_path" :save_img_path,
        "aper_radec_cen" : [aper_ra_cen, aper_dec_cen],
        "aper_xy_pix_cen" :  [aper_xpos, aper_ypos], 
        "aper_params" : aper_params}
    
    return return_dict

        #     "cog_params_g":  cog_output_dict["cog_params"]["g"],
    #     "cog_params_r":  cog_output_dict["cog_params"]["r"],
    #     "cog_params_z":  cog_output_dict["cog_params"]["z"],
    #     "cog_params_g_err": cog_output_dict["cog_params_err"]["g"],
    #     "cog_params_r_err": cog_output_dict["cog_params_err"]["r"],
    #     "cog_params_z_err": cog_output_dict["cog_params_err"]["z"],


def run_cogs(data_arr, segment_map_v2, star_mask, aperture_mask, tgid, tractor_dr9_mags,save_path, subtract_source_pos, pcnn_val, npixels_min, threshold_rms_scale, wcs, source_zred=None, source_radec=None):
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
    largest_blob, cog_num_seg_smooth, cog_num_seg, segment_map_smooth_new, mu_rough = get_new_segment(reconstruct_data, fiber_xpix, fiber_ypix, tot_noise_rms,  npixels_min, threshold_rms_scale, aperture_mask)

    #note that it is possible for the largest blob to be None!

    ##estimate the surface brightness of this segment_map_smooth_new aperture galaxy! With a simple cut in surface brightness, only galaxies below that will get this 
    #additional optimal deblending cut!

    if source_zred < 0.01:
        #we do not apply any isolate galaxy mask
        #and just continue as usual with our COG approach
        smooth_segment_found = True
        parent_galaxy_mask = largest_blob
        final_cog_mask = star_mask | aperture_mask
        jaccard_img_path = None
        num_deblend_segs_main_blob = 1
        nearest_deblend_dist_pix = 0

    else:
        rgb_img = sdss_rgb([data_arr[0],data_arr[1],data_arr[2]])
        rgb_img_mask =  sdss_rgb([reconstruct_data[0], reconstruct_data [1], reconstruct_data [2]])

        
        #we construct the smooth galaxy mask using the r-band image of the latest reconstruction so far!
        segm_smooth_deblend_opt, num_deblend_segs_main_blob, parent_galaxy_mask, non_parent_galaxy_mask, nearest_deblend_dist_pix, jaccard_img_path = get_isolate_galaxy_mask(img_rgb = rgb_img, img_rgb_mask = rgb_img_mask, r_band_data=reconstruct_data[1], r_rms=noise_rms_perband[1], fiber_xpix=fiber_xpix, fiber_ypix=fiber_ypix, file_path=save_path, tgid=tgid, aperture_mask = aperture_mask, pcnn_val = pcnn_val, radec=source_radec, mu_rough = mu_rough )

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
            #no smooth segment was found. This means that we will be reverting to Tractor source photometry OR the original unsmoothed segment map as our final source magnitude!            
            smooth_segment_found = False

            non_parent_galaxy_mask_for_cog = np.zeros_like(parent_galaxy_mask, dtype = bool) 

            parent_galaxy_mask = largest_blob


        #the star mask is already in the aperture_mask, but just included here for completeness.
        #the non_parent_galaxy_mask_for_cog is the mask we will apply to the reconstruct data to get the final version!
        final_cog_mask = star_mask | aperture_mask | non_parent_galaxy_mask_for_cog

    #NOTE: the parent galaxy mask is a binary mask. 1 is on segment, 0 is bkg

    #this is a mask without any of the deblending step
    final_cog_mask_no_isolate_mask = star_mask | aperture_mask 

    #we make two copies of image, one with the smooth blob mask applied and one with it!
    reconstruct_data_no_smooth_mask = np.copy(reconstruct_data)
    reconstruct_data_no_smooth_mask_dict = { "g": reconstruct_data_no_smooth_mask[0], "r": reconstruct_data_no_smooth_mask[1], "z": reconstruct_data_no_smooth_mask[2] }

    reconstruct_data[:,final_cog_mask] = 0
    #construct the reconstruct_data dictionary
    reconstruct_data_dict = { "g": reconstruct_data[0], "r": reconstruct_data[1], "z": reconstruct_data[2] }

    #save this all!!
    rgb_512 = process_img(reconstruct_data, cutout_size = 512, org_size = np.shape(data_arr)[1] )
    rgb_128 = process_img(reconstruct_data, cutout_size = 128, org_size = np.shape(data_arr)[1] )

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
        
    np.save(save_path + "/final_mask_cog.npy", final_cog_mask  )


    #example template empty dictionary used in case outputs are nans
    EMPTY_COG_DICT = {
            "aper_r425_mags": 3*[np.nan],
            "cog_mags": [np.nan] * 3,
            "cog_mags_err": [np.nan] * 3,
            "cog_params_g": 5*[np.nan],
            "cog_params_g_err": 5*[np.nan],
            "cog_params_r": 5*[np.nan],
            "cog_params_r_err": 5*[np.nan], 
            "cog_params_z": 5*[np.nan],
            "cog_params_z_err": 5*[np.nan],
            "cog_chi2" :[np.nan] * 3, 
            "cog_dof" :[np.nan] * 3,
            "aper_r425_frac_in_image" :np.nan,
            "img_path" : None,
            "cog_decrease_len" : [np.nan] * 3,
            "cog_decrease_mag" : [np.nan] * 3, 
            "aper_radec_cen" : [np.nan, np.nan],
            "aper_xy_pix_cen" :  [np.nan, np.nan], 
            "aper_params" : [np.nan] * 3 ,
        }


    if parent_galaxy_mask is None:
        #basically, we will not run COG and just return here!! 
        np.save(save_path + "/parent_galaxy_segment_mask.npy", np.zeros_like(parent_galaxy_mask)   )

        print("--"*6)
        print(f"TARGETID={tgid} has no detections and thus the segmentation map for COG is None. COG will not be run. We will revert to original Tractor mags!")
        print(f"{save_path}")
        print("--"*6)

        FINAL_COG_DICT = {}

        ##adding the cog_isolate params
        for ki in EMPTY_COG_DICT.keys():
            FINAL_COG_DICT[ki] = EMPTY_COG_DICT[ki]

        #adding the cog no isoalte params
        for ki in EMPTY_COG_DICT.keys():
            FINAL_COG_DICT[ki + "_no_isolate_mask"] = EMPTY_COG_DICT[ki]

        #adding the other params
        FINAL_COG_DICT["jaccard_path"] = None
        FINAL_COG_DICT["deblend_smooth_num_seg"] = num_deblend_segs_main_blob 
        FINAL_COG_DICT["deblend_smooth_dist_pix"] =  np.nan
        FINAL_COG_DICT["tractor_parent_mask_mags"] = [np.nan, np.nan, np.nan]
        FINAL_COG_DICT["revert_to_org_tractor"] = True
        FINAL_COG_DICT["aper_r2_mus_no_isolate_mask"] = mu_rough
        
        return FINAL_COG_DICT
    
    else:       
        ##save the tractor only mags!!
        tractor_only_mags, parent_source_catalog = get_tractor_only_mag(parent_galaxy_mask, save_path, np.shape(data_arr)[1], tgid )
        
        parent_galaxy_mask = parent_galaxy_mask.astype(int)
        
        np.save(save_path + "/parent_galaxy_segment_mask.npy", parent_galaxy_mask)  

    
        ##RUNNING THE TWO KINDS OF COG PHOTOMETRY!! One with mask, and one without mask!
        #Note, only the last 3 functions and img_flag will be changing. everything else will be kept the same.

        same_input_dict = {
            "data_arr": data_arr,
            "tgid": tgid,
            "save_path": save_path,
            "parent_source_catalog": parent_source_catalog,
            "tractor_only_mags": tractor_only_mags,
            "tractor_dr9_mags": tractor_dr9_mags,
            "tractor_bkg_model": tractor_bkg_model,
            "data_arr_no_bkg_no_blend": data_arr_no_bkg_no_blend,
            "segment_map_smooth_new": segment_map_smooth_new,
            "wcs": wcs,
            "fiber_xpix": fiber_xpix,
            "fiber_ypix": fiber_ypix
        }
        
        cog_isolate_dict = cog_fitting_subfunction(same_input_dict, reconstruct_data_dict, parent_galaxy_mask, final_cog_mask, img_flag = "")
        
        #as a reminder, larget_blob is the pixel mask of the largest blob identified in the g+r+z smooth segmentation done in get_new_segment. It does not have
        #if source_zred < 0.01, we are not doing the separate deblending anyways, and so need to run it all over again!
        
        if largest_blob is not None and source_zred > 0.01:
            cog_NO_isolate_dict = cog_fitting_subfunction(same_input_dict,reconstruct_data_no_smooth_mask_dict, largest_blob, final_cog_mask_no_isolate_mask,  img_flag = "_no_isolate")
        else:
            cog_NO_isolate_dict = {}
            #we want to return nans
            for ki in EMPTY_COG_DICT.keys():
                cog_NO_isolate_dict[ki] = EMPTY_COG_DICT[ki]
            
        #initializing the final dictionary to be returned
        ##vv the keys in the above dicts returning by the cog_fitting_subfunction command
        # "aper_r425_mags": aper_mags_r4,
        # "cog_mags": final_cog_mtot,
        # "cog_mags_err": final_cog_mtot_err,
        # "cog_params": final_cog_params,
        # "cog_params_err": final_cog_params_err,
        # "cog_chi2" :final_cog_chi2, 
        # "cog_dof" :final_cog_dof,
        # "cog_decrease_len" : decrease_cog_len,
        # "cog_decrease_mag" : decrease_cog_mag, 
        # "aper_r425_frac_in_image" :areafrac_in_image_largest_aper,
        # "save_img_path" :save_img_path,
        # "aper_radec_cen" : [aper_ra_cen, aper_dec_cen],
        # "aper_xy_pix_cen" :  [aper_xpos, aper_ypos], 
        # "aper_params" : aper_params}
        
        FINAL_COG_DICT = {}

        ##adding the cog_isolate params
        for ki in EMPTY_COG_DICT.keys():
            FINAL_COG_DICT[ki] = cog_isolate_dict[ki]

        for ki in EMPTY_COG_DICT.keys():
            FINAL_COG_DICT[ki + "_no_isolate_mask"] = cog_NO_isolate_dict[ki]

        FINAL_COG_DICT["jaccard_path"] = jaccard_img_path
        FINAL_COG_DICT["deblend_smooth_num_seg"] = num_deblend_segs_main_blob 
        FINAL_COG_DICT["deblend_smooth_dist_pix"] =  nearest_deblend_dist_pix
        FINAL_COG_DICT["tractor_parent_mask_mags"] = tractor_only_mags
        FINAL_COG_DICT["revert_to_org_tractor"] = False
        FINAL_COG_DICT["aper_r2_mus_no_isolate_mask"] = mu_rough
        
        return FINAL_COG_DICT

        

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
    source_ra  = input_dict["ra"] 
    source_dec  = input_dict["dec"]
    
    #loading all the relevant data!
    segment_map_v2 = np.load(save_path + "/segment_map_v2.npy")
    star_mask = np.load(save_path + "/star_mask.npy")
    aperture_mask = np.load(save_path + "/aperture_mask.npy") #this aperture mask already contains the mask for the saturated pixels.
    tractor_dr9_mags_arr = np.load(save_path + "/tractor_mags.npy")
    tractor_dr9_mags = { "g": tractor_dr9_mags_arr[0], "r": tractor_dr9_mags_arr[1], "z": tractor_dr9_mags_arr[2]  }


    with open(save_path + '/subtract_source_pos.pkl', 'rb') as f:
        subtract_source_pos = pickle.load(f)

    #feeding it to the cog function!
    cog_output_dict = run_cogs(data_arr, segment_map_v2, star_mask, aperture_mask, source_tgid, tractor_dr9_mags, save_path, subtract_source_pos, pcnn_val, npixels_min, threshold_rms_scale, wcs, source_zred = source_zred, source_radec = [source_ra, source_dec] )
    
    ##NOTE: ALL THE MAGS RETURNED HERE ARE MW EXTINCTION CORRECTED

    return cog_output_dict
    
    # return {
    #     "aper_r425_mags": all_aper_mags_r425,
    #     "cog_mags": cog_mags,
    #     "cog_mags_err": cog_output_dict["cog_mtot_err"],
    #     "cog_params_g":  cog_output_dict["cog_params"]["g"],
    #     "cog_params_r":  cog_output_dict["cog_params"]["r"],
    #     "cog_params_z":  cog_output_dict["cog_params"]["z"],
    #     "cog_params_g_err": cog_output_dict["cog_params_err"]["g"],
    #     "cog_params_r_err": cog_output_dict["cog_params_err"]["r"],
    #     "cog_params_z_err": cog_output_dict["cog_params_err"]["z"],
    #     "img_path": cog_output_dict["save_img_path"],
    #     "cog_chi2" : cog_output_dict["cog_chi2"], 
    #     "cog_dof" : cog_output_dict["cog_dof"],
    #     "aper_r425_frac_in_image": cog_output_dict["aper_r425_frac_in_image"],
    #     "cog_decrease_len": cog_output_dict["cog_decrease_len"], 
    #     "cog_decrease_mag": cog_output_dict["cog_decrease_len"],
    #     "aper_ra_cen": cog_output_dict["aper_radec_cen"][0],
    #     "aper_dec_cen": cog_output_dict["aper_radec_cen"][1],
    #     "aper_xpix_cen": cog_output_dict["aper_xy_pix_cen"][0],
    #     "aper_ypix_cen": cog_output_dict["aper_xy_pix_cen"][1],
    #     "aper_params": cog_output_dict["aper_params"],
    #     "jaccard_path": cog_output_dict["jaccard_img_path"],
    #     "deblend_smooth_num_seg": cog_output_dict["deblend_smooth_num_seg"],
    #     "deblend_smooth_dist_pix": cog_output_dict["deblend_smooth_dist_pix"],
    #     "tractor_parent_mask_mags": cog_output_dict["parent_tractor_only_mags"],
    #     "revert_to_org_tractor": cog_output_dict["revert_to_org_tractor"] 
    # }


    
    
    
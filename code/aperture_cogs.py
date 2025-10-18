'''
In this script, we will do a curve of growth analysis on our objects that are really shreds!

We are working in a different script here as we need the tractor/astrometry packages to construct the psf model!

Basic steps are following what SGA catalog did:
1) Identify the range of apertures within which we will do our photometry
2) Mask relevant pixels (if star or residuals after subtracting model image are very large?)
3) If we have identified sources within aperture that we want to subtract, we can create an image with all the masked pixels and subtracted sources for reference?
'''

from scipy.ndimage import gaussian_filter
from scipy import ndimage as ndi
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
from alternative_photometry_methods import  get_simplest_photometry
from desi_lowz_funcs import mags_to_flux, flux_to_mag
from desi_lowz_funcs import get_elliptical_aperture, measure_elliptical_aperture_area_fraction_masked, find_nearest_island
from desi_lowz_funcs import measure_elliptical_aperture_area_in_image_pix
import copy
import os
from measure_fiber_flux import measure_simple_fiberflux

def make_empty_tractor_cog_dict():
    '''
    Make a template empty tractor dict
    '''
    return {
        "tractor_cog_mags": [np.nan] * 3,
        "tractor_fiber_mags": [np.nan] * 3,
        "tractor_cog_mags_err": [np.nan] * 3,
        "tractor_cog_params_g": [np.nan] * 5,
        "tractor_cog_params_g_err": [np.nan] * 5,
        "tractor_cog_params_r": [np.nan] * 5,
        "tractor_cog_params_r_err": [np.nan] * 5, 
        "tractor_cog_params_z": [np.nan] * 5,
        "tractor_cog_params_z_err": [np.nan] * 5,
        "tractor_cog_chi2": [np.nan] * 3, 
        "tractor_aper_radec_cen": [np.nan, np.nan], 
        "tractor_aper_params": [np.nan] * 3,
        "tractor_aper_cen_masked_bool": False,
        "tractor_aperfrac_in_image": np.nan,
        "tractor_rad_mus": [np.nan] * 4
    }

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


## some functions to compute the surface brightness!!
def surface_brightness_mu(r_scales, aper_rad_pix, cog_params , ba_ratio):
    
    mtot, m0, alpha_1, r_0, alpha_2 = cog_params

    #the cog function needs r in the scale factors. Not in absolute pixels or arcseconds yet
    m_r = cog_function(r_scales, mtot, m0, alpha_1, r_0, alpha_2)
    
    r_arcsec = r_scales * aper_rad_pix * 0.262

    #according to the SGA2020 paper, this is computed by just using the semi-major axis
    area = np.pi * (r_arcsec**2) * ba_ratio  # elliptical area correction    
    return r_arcsec, m_r + 2.5 * np.log10(area), m_r


def half_light_radius_cog(cog_params):
    mtot, m0, alpha_1, r_0, alpha_2 = cog_params
    
    return r_0 * ( (1/alpha_1) * (np.exp( -np.log10(0.5) / (0.4*m0)  ) - 1) )**(-1/alpha_2)

    
def radius_at_mu(tgid, file_path, mu_target, cog_params, ba_ratio=None, aper_rad_pix=None, plot_name=None):
    """
    Compute the radius (in arcsec) where surface brightness μ = mu_target,
    and return a compact list of those radii plus the half-light radius.

    Returns
    -------
    list
        [r(mu_1), r(mu_2), ..., r(mu_n), r_half] in arcsec
        NaN where not bracketed.
    """

    # Compute the half-light radius in arcsec
    r12_arcsec = half_light_radius_cog(cog_params) * aper_rad_pix * 0.262
    if not np.isfinite(r12_arcsec) or r12_arcsec <= 0:
        print(f"[WARN] Nonpositive half-light radius for TGID={tgid}: r12={r12_arcsec}")
        r12_arcsec = np.nan

    # Ensure inputs are safe
    mu_target = np.atleast_1d(mu_target).astype(float)
    if not (0 < ba_ratio <= 1) or aper_rad_pix is None or aper_rad_pix <= 0:
        print(f"[WARN] Invalid input for TGID={tgid}")
        return [np.nan] * len(mu_target) + [r12_arcsec]

    # Radial grid (empirically chosen range)
    r_grid = np.linspace(0.5, 15, 100)

    # Compute μ(r)
    r_arcsec, mu_grid, m_r = surface_brightness_mu(r_grid, aper_rad_pix, cog_params, ba_ratio)

    # Check bracketing
    mu_min, mu_max = mu_grid.min(), mu_grid.max()
    r_mu = np.full_like(mu_target, np.nan, dtype=float)
    mask_valid = (mu_target >= mu_min) & (mu_target <= mu_max)
    if np.any(mask_valid):
        r_mu[mask_valid] = np.interp(mu_target[mask_valid], mu_grid, r_arcsec)
    else:
        print(f"[WARN] mu_target range not bracketed for TGID={tgid}, ({mu_min:.2f}, {mu_max:.2f})")

    # Plot for diagnostics
    plt.figure(figsize=(4, 4))
    plt.plot(r_arcsec, mu_grid, color="k", lw=2)
    for mu in mu_target:
        plt.hlines(y=mu, xmin=r_arcsec.min(), xmax=r_arcsec.max(), color="r", ls="dotted", lw=1)
    plt.vlines(x=r12_arcsec, ymin=mu_grid.min(), ymax=mu_grid.max(), color="forestgreen", ls="--", lw=1)
    plt.xlim([r_arcsec.min(), r_arcsec.max()])
    plt.ylim([20, 28])
    plt.savefig(f"{file_path}/mu_radii_curve_{plot_name}.png", bbox_inches="tight")
    plt.close()

    r_mu = np.where((r_mu > 0) & np.isfinite(r_mu), r_mu, np.nan)
  
    # Compact output list
    return list(r_mu) + [r12_arcsec]


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
        # chi2_red = chi2 / dof if dof > 0 else np.nan
        
        return popt, perr, chi2, dof
        
    except RuntimeError:
        print("Fit failed. Returning filler values.")
        return np.full(len(p0), filler), np.full(len(p0), filler), np.nan, np.nan


def make_tractor_r_band_model_img(tgid, org_grz_data, save_path, wcs):
    '''
    In this function, we make the r-band tractor model image of the sources currently deemed to be part of the parent galaxy.

    Note that this function also returns the magnitude of the brightest tractor source (the parent source). This will allow us to see what how well does 
    the parent tractor source capture the photometry
    '''

    parent_source_cat = Table.read(save_path + "/parent_galaxy_sources.fits")
    
    if len(parent_source_cat) == 0:
        print(f"No sources in the parent source cat = {tgid}. Returning nans")

        np.save(f"{save_path}/parent_galaxy_tractor_no_isolate_model.npy", np.zeros_like(org_grz_data) )

        return np.zeros_like(org_grz_data[1]), np.zeros_like(org_grz_data), 3*[np.nan], None, len(parent_source_cat), parent_source_cat, [np.nan,np.nan,np.nan]

    else:
        total_model = np.zeros_like(org_grz_data)
    
        if len(parent_source_cat) > 100:
            print(f"FYI: Reading more than 100 sources in the tractor catalog so may take some time: {tgid}")

        #get the object with the brightest r-band mag in the source cat
        bright_ind = np.argmin(parent_source_cat["mag_r"].data)
        tractor_brightest_source_mags = np.array([ parent_source_cat["mag_g"].data[bright_ind], parent_source_cat["mag_r"].data[bright_ind], parent_source_cat["mag_z"].data[bright_ind] ])
 
        for pi in range(len(parent_source_cat)):
            objidi = parent_source_cat["source_objid_new"].data[pi]
            tractor_model_path = save_path + f"/tractor_models/tractor_parent_source_model_{objidi}.npy"
            #load it!
            model_i = np.load(tractor_model_path)
            total_model += model_i

        r_total_model = total_model[1]

        g_flux_corr = mags_to_flux(parent_source_cat["mag_g"]) / parent_source_cat["mw_transmission_g"]
        r_flux_corr = mags_to_flux(parent_source_cat["mag_r"]) / parent_source_cat["mw_transmission_r"]
        z_flux_corr = mags_to_flux(parent_source_cat["mag_z"]) / parent_source_cat["mw_transmission_z"]
        
        tot_g_mag = flux_to_mag(np.sum(g_flux_corr))
        tot_r_mag = flux_to_mag(np.sum(r_flux_corr))
        tot_z_mag = flux_to_mag(np.sum(z_flux_corr))
    
        np.save(f"{save_path}/parent_galaxy_tractor_no_isolate_model.npy",total_model)
        rgb_data = sdss_rgb(total_model, ["g","r","z"])
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        ax.set_title(f"Parent Galaxy No Isolate Tractor Model",fontsize = 13)
        ax.imshow(rgb_data, origin="lower")
        fig.savefig(f"{save_path}/parent_galaxy_no_isolate_tractor_reconstruction.png")
        plt.close(fig)

        return r_total_model, total_model, [tot_g_mag, tot_r_mag, tot_z_mag], rgb_data, len(parent_source_cat), parent_source_cat, tractor_brightest_source_mags



def find_parent_segment_blob(segment_map, fiber_xpix, fiber_ypix):
    '''
    Function that finds the segment we will call parent!
    '''

    island_num = segment_map.data[int(fiber_ypix),int(fiber_xpix)]

    on_blob = True
    
    if island_num == 0:
        on_blob=False
        #INSTEAD OF FINDING THE LARGEST MAIN BLOB, WE WILL FIND THE CLOSEST MAIN BLOB
        island_num, _ = find_nearest_island(segment_map.data, fiber_xpix,fiber_ypix)
        
        # on_blob = False
        # #that is the segment is on background. Then we choose the largest blob!
        # # The areas (number of pixels) of each segment
        # areas = segment_map.areas  # numpy array of pixel counts per segment
        # # Find index of the largest one
        # largest_idx = areas.argmax()
        # # Get the corresponding segment ID number
        # island_num = segment_map.labels[largest_idx]

        # if island_num == 0:
        #     raise ValueError("The background is being assigned the largest segment!")

    #if it is not zero, then it is on a segment, then we are all good!
    segment_map_data = segment_map.data
    parent_galaxy_mask = (segment_map_data == island_num)

    #the other segments will be blocked as part of the not_galaxy_no_isolate_mask. We do not mask the background
    not_parent_galaxy_mask = (segment_map_data != island_num) & (segment_map_data != 0)

    return parent_galaxy_mask, on_blob, not_parent_galaxy_mask



def get_new_segment_tractor(r_band_trac_model, fiber_xpix, fiber_ypix, r_noise_rms,  npixels_min, threshold_rms_scale, img_type, save_path):
    '''
    An updated version of the get new segment function, where the parent galaxy segment is based on the tractor model image!
    Also, in the past, we had worked with the g+r+z image, but should it make a big difference? This already becomes the no-isolate mask so this is okay for now ... 

    Need to also return a mask of other segments to be blocked in case they are identified!
    
    '''

    if np.any(r_band_trac_model):
        kernel = make_2dgaussian_kernel(3.0, size=5)  # FWHM = 3.0
        kernel_smooth = make_2dgaussian_kernel(15, size=29) 

        convolved_tot_data = convolve( r_band_trac_model, kernel )
        convolved_tot_data_smooth = convolve(r_band_trac_model, kernel_smooth)

        threshold = threshold_rms_scale * r_noise_rms
    
        #we will definitely find somewhere here!
        segment_map_trac = detect_sources(convolved_tot_data, threshold, npixels=npixels_min) 
        
        #if we do not find anything, that is good to know!!
        if segment_map_trac is None:
            cog_num_seg = 0
        else:
            #we subtract 1 as there is the background
            cog_num_seg = len(np.unique(segment_map_trac.data)) - 1

        segment_map_trac_smooth = detect_sources(convolved_tot_data_smooth, threshold, npixels=npixels_min) 
        #if we do not find anything, that is good to know!!
        if segment_map_trac_smooth is None:
            print(f"No smooth tractor component detected: {save_path}")
            cog_num_seg_smooth = 0
            segment_map_trac_smooth_data = None
        else:
            #we subtract 1 as there is the background
            cog_num_seg_smooth = len(np.unique(segment_map_trac_smooth.data)) - 1
            segment_map_trac_smooth_data = segment_map_trac_smooth.data
        #####
        #FIND THE BLOB ON WHICH WE WILL DO THE APERTURE ESTIMATION
        if cog_num_seg_smooth > 0:
            #if a smooth component is found, we choose the component that contains the DESI fiber!
            #It does not have to be the largest component as there can be a true large galaxy nearby!
            #and if indeed the largest component is part of the parent galaxy, then that is an over-shredding problem ... 
            #which we would be trying to fix with the smoothing scale etc.
            parent_galaxy_no_isolate_mask, on_blob, not_galaxy_no_isolate_mask = find_parent_segment_blob(segment_map_trac_smooth, fiber_xpix, fiber_ypix)
        else:
            if cog_num_seg > 0:
                #hope we find a segment in not the very smoothed version
                parent_galaxy_no_isolate_mask, on_blob, not_galaxy_no_isolate_mask = find_parent_segment_blob(segment_map_trac, fiber_xpix, fiber_ypix)
            else:
                #we do not run COG at all and simply revert back to the tractor at the end!
                parent_galaxy_no_isolate_mask = None
                on_blob = False
                not_galaxy_no_isolate_mask = np.zeros_like(r_band_trac_model).astype(bool)

        ##estimate a very simple aperture on this largest blob
        if parent_galaxy_no_isolate_mask is not None:
            #there does exist a smooth blob.
            parent_galaxy_no_isolate_mask_copy = np.copy(parent_galaxy_no_isolate_mask)
            
            #estimate aperture and then scale it by 2. Measure average surface brightness within this
            aperture_temp, _,_,aper_param_temp = get_elliptical_aperture( binary_segment_mask = parent_galaxy_no_isolate_mask_copy, sigma = 2, aper_light_image = convolved_tot_data_smooth, img_type = img_type )

            aper_in_image_area_pix = measure_elliptical_aperture_area_in_image_pix( parent_galaxy_no_isolate_mask.shape, aperture_temp )
            aper_in_image_area_arcsec = aper_in_image_area_pix * (0.262)**2
            
            #measure photometry within this rough aperture! No mask needed as only on tractor model image
            phot_temp_r = aperture_photometry( r_band_trac_model , aperture_temp)
    
            r_mag_temp = flux_to_mag(phot_temp_r["aperture_sum"].data[0])
                
            #get the area of this aperture in arcsec^2 so convert the pixel sizes to arcsec! Area = pi*a*b
            #This aperture has not been scaled yet by the factor of two so we will do that!
            #the factor of 2 is because we want the MU_R within the R2 aperture
            aper_temp_area = np.pi * (2 * aper_param_temp[0]*0.262) * (2 * aper_param_temp[0]*aper_param_temp[1]*0.262)

            aper_area_f = np.minimum(aper_temp_area,  aper_in_image_area_arcsec)
            
            r_mu_rough_ellipse = mean_surface_brightness( r_mag_temp,   aper_area_f)

            #we can also measure the mu by taking the area of the island instead of the ellipse!
            #the sum gives the total number of pixels, and the area of each pixel is the 0.262^2
            area_island_arcsec = np.sum(parent_galaxy_no_isolate_mask_copy) * (0.262)**2

            #we need to measure the magnitude of the island??
            r_mag_island = 22.5 - 2.5*np.log10( np.sum(r_band_trac_model[ parent_galaxy_no_isolate_mask_copy]) )

            r_mu_rough_island = mean_surface_brightness( r_mag_island,  area_island_arcsec)

            # print("DEBUGGING MODE:")
            # print(f"ELLIPSE RMAG = {r_mag_temp }")
            # print(f"ISLAND RMAG = {r_mag_island }")
            # print(f"ELLIPSE AREA = {aper_temp_area,  aper_in_image_area_arcsec}")
            # print(f"ISLAND AREA = {area_island_arcsec}")
            # print(f"ELLIPSE MUR = {r_mu_rough_ellipse}")
            # print(f"ISLAND MUR = {r_mu_rough_island}")
            
        else:
            r_mu_rough_ellipse = np.nan
            r_mu_rough_island = np.nan
            

        return parent_galaxy_no_isolate_mask, r_mu_rough_ellipse, r_mu_rough_island , cog_num_seg, cog_num_seg_smooth, on_blob, segment_map_trac_smooth_data, not_galaxy_no_isolate_mask
            
    else:
        not_galaxy_no_isolate_mask = np.zeros_like(r_band_trac_model).astype(bool)

        
        return None, np.nan, np.nan, np.nan, np.nan, False, None, not_galaxy_no_isolate_mask



def load_tractor_models(tractor_save_dir, parent_source_cat):
    """
    Load individual or HDF5-stored tractor source models.
    Returns the summed total_model and a dict for individual access.
    """
    h5_path = f"{tractor_save_dir}/tractor_parent_source_models.h5"
    total_model = None
    models = {}

    if os.path.exists(h5_path):
        with h5py.File(h5_path, "r") as f:
            for key in f.keys():
                mod = f[key][:]
                models[int(key)] = mod
                if total_model is None:
                    total_model = np.zeros_like(mod)
                total_model += mod
    else:
        for objid in parent_source_cat["source_objid_new"].data:
            path = f"{tractor_save_dir}/tractor_parent_source_model_{int(objid)}.npy"
            mod = np.load(path)
            models[int(objid)] = mod
            if total_model is None:
                total_model = np.zeros_like(mod)
            total_model += mod

    return total_model, models


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



def base_cog_measure(radii_scale, parent_galaxy_mask, reconstruct_galaxy_dict, aper_light_image = None, img_type=None, ax = None):
    '''
    This is the function that actually makes the COG calculations

    aper_light_image is the 2d image we will use to measure the aperture properties in case img_type = "light", which will be majority of the cases.
    '''

    grz_image = reconstruct_galaxy_dict["g"] + reconstruct_galaxy_dict["r"] + reconstruct_galaxy_dict["z"]

    
    cog_mags = {"g":[], "r": [], "z": []}

    for scale_i in radii_scale:
        #note that in case img_type=light, the aper_light_image (ideally tractor model) is used to estimate the aperture center/size
        aperture_for_phot_i, aperfrac_in_image_i, xypos, aper_params = get_elliptical_aperture( binary_segment_mask = parent_galaxy_mask , sigma = scale_i, aper_light_image = aper_light_image, img_type = img_type)
        #again note that img_type just switches between using the binary_segment_mask or aper_light_image to estimate the aperture!
    
        if scale_i == radii_scale[-1]:
            aperfrac_in_image_largest_aper = aperfrac_in_image_i
    
            #for the xy pos should not change at all between different scale and we just save it at the end
            aper_xpos = xypos[0]
            aper_ypos = xypos[1]
            
        if ax is not None:
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
            new_mag_i = flux_to_mag( phot_table_i["aperture_sum"].data[0] )
            # new_mag_i = 22.5 - 2.5*np.log10( phot_table_i["aperture_sum"].data[0] )
            
            cog_mags[bi].append(new_mag_i)



    ##this is also a good location to compute fiber mags! We know the aperture center
    fiber_mags = measure_simple_fiberflux(aper_xpos, aper_ypos, reconstruct_galaxy_dict, aperture_diam_arcsec = 1.5)

    return cog_mags, fiber_mags, aperfrac_in_image_largest_aper, aper_xpos, aper_ypos, aper_params



def base_cog_fit(cog_mags, radii_scale):
    '''
    Function that does the cog fitting!
    '''

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

    #Return all the stuff!
    return final_cog_params, final_cog_params_err, final_cog_chi2, final_cog_dof, decrease_cog_len, decrease_cog_mag, final_cog_mtot, final_cog_mtot_err
            

def basic_cog_fitting_subfunction(reconstruct_tractor_galaxy_dict, parent_mask, wcs, org_basic_mask, img_type=None, aper_light_image=None, tgid=None,save_path=None):
    '''
    The very simplified way to measure the cog parameters. Used for the tractor based image  

    org_basic_mask: the mask of just aperture and star pixels. Used to identify whether the aperture center lies on a masked pixel!
    '''

    #make a blank empty dict
    tractor_cog_dict = make_empty_tractor_cog_dict()

    if parent_mask is not None:
         ##get the tractor cog stuff here!!
        #with the total model image made, we would like to also measure the elliptical parameters on this?
        radii_scale = np.arange(1.25,4.25,0.25)
        
        cog_mags, fiber_mags, aperfrac_in_image, aper_xpos, aper_ypos, aper_params = base_cog_measure(radii_scale, parent_mask.astype(int), reconstruct_tractor_galaxy_dict, ax = None, aper_light_image = aper_light_image, img_type = img_type)

         #check if the aper_xpos, aper_ypos lies on the org_basic_mask! This mask is True or 1 if it is being masked
        aper_cen_mask_pix_val = org_basic_mask[ int(aper_ypos), int(aper_xpos) ]
        aper_cen_mask_bool = aper_cen_mask_pix_val.astype(bool)
            
        #convert this xy position into a ra,dec position using wcs of the image. Also, save the xy position
        aper_ra_cen, aper_dec_cen, _ = wcs.all_pix2world(aper_xpos, aper_ypos, 0, 1)
    
        #fit model to cog    
        final_cog_params, final_cog_params_err, final_cog_chi2, _, _, _, final_cog_mtot, final_cog_mtot_err = base_cog_fit(cog_mags, radii_scale)

        
        if ~np.isnan(final_cog_params["r"][0]):
            #now compute the semi-major axis of different surface brightnesses
            radius_mus = radius_at_mu(tgid, save_path,  np.array([24,25,26]), final_cog_params["r"], ba_ratio=aper_params[1], aper_rad_pix = aper_params[0], plot_name = "tractor")
        else:
            radius_mus = 4*[np.nan]
            
        #update the dictionary parameters
        #these are parameters of the tractor based photometry that would be useful to have in case the fiducial photometry refers to this!
        #note that we do not really need the tractor cog mags here. We already have the tractor total photometry, but this is just for reference
        tractor_cog_dict["tractor_cog_mags"] = final_cog_mtot
        tractor_cog_dict["tractor_fiber_mags"] = fiber_mags
        tractor_cog_dict["tractor_cog_mags_err"] = final_cog_mtot_err
        for bi in "grz":
            tractor_cog_dict[f"tractor_cog_params_{bi}]"] = final_cog_params[bi]
            tractor_cog_dict[f"tractor_cog_params_{bi}_err"] = final_cog_params_err[bi]
        tractor_cog_dict["tractor_cog_chi2"] = final_cog_chi2
        
        tractor_cog_dict["tractor_aper_radec_cen"] = [ aper_ra_cen, aper_dec_cen]        
        tractor_cog_dict["tractor_aper_params"] = aper_params
        tractor_cog_dict["tractor_aper_cen_masked_bool"] = aper_cen_mask_bool
        tractor_cog_dict["tractor_aperfrac_in_image"] =  aperfrac_in_image
        tractor_cog_dict["tractor_rad_mus"] =  radius_mus
        
        return tractor_cog_dict
    
    else:
        return tractor_cog_dict


def get_tractor_only_isolate_mag_helper(tgid, parent_mask, parent_source_cat, save_path, flag = "isolate", wcs=None, org_basic_mask=None, img_type=None):
    '''
    Helper function that in general deals with both of the two isolate or no isolate parent masks.

    With a given input mask, it returns the tractor only photometry, the corresponding source catalog, and the tractor parent model
    '''

    if parent_mask is not None:
        #only include sources that lie on the main segment as identified by the parent galaxy_mask
        parent_keep_mask = (parent_mask[parent_source_cat["ypix"].data.astype(int), parent_source_cat["xpix"].data.astype(int)] == True)
    
        #and always include the source of target!
        source_target = (parent_source_cat["separations"] < 1)
    
        parent_source_cat_f = parent_source_cat[parent_keep_mask | source_target]

        #get the brightest source mags here
        bright_ind = np.argmin(parent_source_cat_f["mag_r"].data)
        tractor_brightest_source_mags = [ parent_source_cat_f["mag_g"].data[bright_ind], parent_source_cat_f["mag_r"].data[bright_ind], parent_source_cat_f["mag_z"].data[bright_ind]  ]
        
        if len(parent_source_cat_f) == 0:
            print(f"No parent sources found hmm : {len(parent_source_cat_f)}, {tgid}, {len(parent_source_cat)}")
            #there is this one weird example 39627783936151447 where the source that is targeted is in Gaia DR2, but is clearly a small, extended galaxy.
            #just return nans :) 
            empty_tractor_cog_dict = make_empty_tractor_cog_dict()
            return  3*[np.nan], None, None, None, empty_tractor_cog_dict, 3*[np.nan], None
                   
            
        ##MAKE RGB IMAGE OF THE TRACTOR ONLY RECONSTRUCTION
        ##with these remaining sources, we can combine their source models in the folder to get the model!
        width = parent_mask.shape[1]
        total_model = np.zeros((3, width, width))
    
        if len(parent_source_cat_f) > 100:
            print(f"FYI: Reading more than 100 sources in the tractor catalog so may take some time: {tgid}")
        
        for pi in range(len(parent_source_cat_f)):
            objidi = parent_source_cat_f["source_objid_new"].data[pi]
            tractor_model_path = save_path + f"/tractor_models/tractor_parent_source_model_{objidi}.npy"
            #load it!
            model_i = np.load(tractor_model_path)
            total_model += model_i
    
        #compute the magnitudes by summing the photometry
        #these magnitudes were not originally corrected for extinction so we are doing so here!
        g_flux_corr = mags_to_flux(parent_source_cat_f["mag_g"]) / parent_source_cat_f["mw_transmission_g"]
        r_flux_corr = mags_to_flux(parent_source_cat_f["mag_r"]) / parent_source_cat_f["mw_transmission_r"]
        z_flux_corr = mags_to_flux(parent_source_cat_f["mag_z"]) / parent_source_cat_f["mw_transmission_z"]

        #summing all the individual fluxes to get the total extinction corrected tractor based photometry
        tot_g_mag = flux_to_mag(np.sum(g_flux_corr))
        tot_r_mag = flux_to_mag(np.sum(r_flux_corr))
        tot_z_mag = flux_to_mag(np.sum(z_flux_corr))

        np.save(f"{save_path}/parent_galaxy_tractor_{flag}_model.npy",total_model)
        rgb_data = sdss_rgb(total_model, ["g","r","z"])
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        ax.set_title(f"Parent {flag} Galaxy Tractor Model",fontsize = 13)
        ax.imshow(rgb_data, origin="lower")
        fig.savefig(f"{save_path}/parent_galaxy_{flag}_tractor_reconstruction.png")
        plt.close(fig)

        parent_source_cat_f.write( save_path + f"/parent_galaxy_sources_{flag}_FINAL.fits", overwrite=True)

        #with the total model image made, we would like to also measure the elliptical parameters on this?
        reconstruct_tractor_galaxy_dict = {"g": total_model[0], "r": total_model[1], "z": total_model[2] }

        tractor_isolate_aper_light_img = total_model[0] + total_model[1] + total_model[2]
        
        tractor_cog_dict = basic_cog_fitting_subfunction(reconstruct_tractor_galaxy_dict, parent_mask.astype(int), wcs, org_basic_mask, img_type = img_type, aper_light_image=tractor_isolate_aper_light_img, tgid = tgid , save_path = save_path)

        return [tot_g_mag, tot_r_mag, tot_z_mag], parent_source_cat_f, total_model, rgb_data, tractor_cog_dict,  tractor_brightest_source_mags, tractor_isolate_aper_light_img
    
    else:

        empty_tractor_cog_dict = make_empty_tractor_cog_dict()
        
        return 3*[np.nan], None, None, None, empty_tractor_cog_dict, 3*[np.nan], None



def get_tractor_isolate_mag(parent_galaxy_isolate_mask, save_path, width, tgid, wcs, org_basic_mask, img_type=None):
    '''
    Function where we use the not-parent galaxy mask to get a tractor only mag!! If no not parent objects were found in the smooth deblended stage, we just add source flux without removing any more!

    We need to do this for both with and without isolate mask
    '''

    ##using the non-parent galaxy mask, remove all the tractor sources that lie on that to get a tractor only photometry!!
    parent_source_cat = Table.read(save_path + "/parent_galaxy_sources.fits")

    #run the helper functions
    tractor_parent_isolate_mags, parent_isolate_cat, parent_isolate_trac_model, isolate_rgb, tractor_cog_dict, tractor_brightest_source_mags, tractor_isolate_aper_light_img = get_tractor_only_isolate_mag_helper(tgid, parent_galaxy_isolate_mask, parent_source_cat.copy(), save_path, flag = "isolate", wcs = wcs, org_basic_mask=org_basic_mask, img_type = img_type)

    #we also return the rgb image of the isolate 
    return tractor_parent_isolate_mags, parent_isolate_cat, isolate_rgb, tractor_cog_dict, tractor_brightest_source_mags, tractor_isolate_aper_light_img



    
def cog_fitting_subfunction(same_input_dict,reconstruct_galaxy_dict, parent_galaxy_mask, final_cog_mask, org_basic_mask, img_flag = None, img_type = None, aper_light_image = None):
    '''
    The function that actually runs the cog fitting. We will compute two kinds of COG mags: both with and without the smooth parent blob mask.

    The reconstruct_data_dict is the specific kind of data we are running with !

    The parent_galaxy_mask is the mask on which we estimate the aperture
    final_cog_mask here is just for visualization purposes, but is the total mask of all pixels that we mask in our cog analysis. It is already applied to the data_dict

    org_basic_mask = star_mask | aperture_mask that contains the original star pixels and unassociated color pixel masks
    ^This is without any of the isolate or not isolate mask. We use this to measure the the masked fraction of our final aperture.
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

    #get the plotting target_ra,dec zred
    ra = same_input_dict["ra"]
    dec = same_input_dict["dec"]
    zred = same_input_dict["zred"]

    fig,ax = make_subplots(ncol = 5, nrow = 2, row_spacing = 0.5,col_spacing=1.2, label_font_size = 17,plot_size = 3,direction = "horizontal", return_fig=True)

    radii_scale = np.arange(1.25,4.25,0.25)

    #we are estimating the mags in varying apertures now. 

    cog_mags, fiber_mags, aperfrac_in_image_largest_aper, aper_xpos, aper_ypos, aper_params = base_cog_measure(radii_scale, parent_galaxy_mask, reconstruct_galaxy_dict, aper_light_image= aper_light_image, ax = ax, img_type=img_type)

    #check if the aper_xpos, aper_ypos lies on the org_basic_mask! This mask is True or 1 if it is being masked
    aper_cen_mask_pix_val = org_basic_mask[ int(aper_ypos), int(aper_xpos) ]
    aper_cen_mask_bool = aper_cen_mask_pix_val.astype(bool)

    #convert this xy position into a ra,dec position using wcs of the image. Also, save the xy position
    aper_ra_cen, aper_dec_cen, _ = wcs.all_pix2world(aper_xpos, aper_ypos, 0, 1)

    #fit model to cog    
    final_cog_params, final_cog_params_err, final_cog_chi2, final_cog_dof, decrease_cog_len, decrease_cog_mag, final_cog_mtot, final_cog_mtot_err = base_cog_fit(cog_mags, radii_scale)


    if ~np.isnan(final_cog_params["r"][0]):
        #now compute the semi-major axis of different surface brightnesses
        radius_mus = radius_at_mu(tgid, save_path, np.array([24,25,26]), final_cog_params["r"], ba_ratio=aper_params[1], aper_rad_pix = aper_params[0], plot_name = "cog")
    else:
        radius_mus = [np.nan] * 4
    
    box_size = np.shape(data_arr)[1] 
    
    ##plot the rgb image of actual data 
    ax_id = 5
    ax[ax_id].set_title("IMG",fontsize=12)
    ax[ax_id].text(0.5,0.95,f"{ra:.3f}, {dec:.3f}, z = {zred:.3f}",size =12,transform=ax[ax_id].transAxes, va='center',ha="center",color = "yellow")
    
    ax[ax_id].text(0.5,0.05,f"{tgid}",size = 12,transform=ax[ax_id].transAxes, va='center',ha="center",color = "yellow")
    
    
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

    spacing = 0.08
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

    #let us save this reconstructed galaxy!!
    np.save(save_path + f"/final_reconstruct_galaxy_subfunction{img_flag}.npy", np.array([reconstruct_galaxy_dict["g"],reconstruct_galaxy_dict["r"],reconstruct_galaxy_dict["z"]]))

    ax[ax_id].imshow(rgb_reconstruct_full, origin='lower',zorder = 0)

    ax[ax_id].set_xticks([])
    ax[ax_id].set_yticks([])

    if np.isfinite(aper_xpos) and np.isfinite(aper_ypos) and np.isfinite(aper_params[0]) and aper_params[0] > 0:
        #draw the aperture
        if np.isnan(radius_mus[-1]):
            draw_a_pix = aper_params[0] * 2.5
        else:
            #this is in arcsecs, so we need to convert it back to pixels!
            draw_a_pix = 2.5 * radius_mus[-1] / 0.262
            
        draw_b_pix = draw_a_pix  * aper_params[1]
        draw_theta = aper_params[2]
        draw_ellip_aper = aperture = EllipticalAperture( (aper_xpos, aper_ypos) , draw_a_pix, draw_b_pix, theta=draw_theta)
    
        draw_ellip_aper.plot(ax = ax[ax_id], color = "red", lw = 1.25, ls = "-",alpha = 1)
        ax[ax_id].scatter( aper_xpos, aper_ypos, color = "red",marker = "x",zorder = 1,s=10)
    
        #plotting the xy limits
        hw = 1.5 * draw_a_pix  # half-width
        xlo, xhi = aper_xpos - hw, aper_xpos + hw
        ylo, yhi = aper_ypos - hw, aper_ypos + hw

        if np.all(np.isfinite([xlo, xhi, ylo, yhi])):
            ax[ax_id].set_xlim([xlo, xhi])
            ax[ax_id].set_ylim([ylo, yhi])
        else:
            print(f"Skipping axis limits for TGID {tgid}: invalid values")

    ##show the final summary figure
    ax_id = 4
    # ax[ax_id].set_title("Simple-Photo Galaxy",fontsize = 12)
    ax[ax_id].set_title("Tractor g+r+z model",fontsize = 12)
    #read in the simple photo mask!
    # if os.path.exists(save_path + "/simplest_photometry_binary_mask.npy"):
    #     simple_photo_mask = np.load(save_path + "/simplest_photometry_binary_mask.npy")
    #     data_arr[:, simple_photo_mask == 0] = 0
    #     rgb_img_simple = sdss_rgb([data_arr[0],data_arr[1],data_arr[2]], ["g","r","z"], scales=dict(g=(2,6.0), r=(1,3.4), z=(0,2.2)), m=0.03)
    # else:
    #     rgb_img_simple = np.zeros_like(rgb_img)
    norm_obj = Normalize(vmin=np.nanmin(aper_light_image), vmax=np.nanmax(aper_light_image)) 
    ax[ax_id].imshow(aper_light_image, origin='lower',zorder = 0,norm=norm_obj,cmap = "viridis")
    ax[ax_id].set_xlim([0,box_size])
    ax[ax_id].set_ylim([0,box_size])
    ax[ax_id].set_xticks([])
    ax[ax_id].set_yticks([])

    ##saving this image :0
    save_img_path = save_path + f"/cog_summary{img_flag}.png"
    plt.savefig( save_img_path ,bbox_inches="tight")
    # plt.savefig(f"/pscratch/sd/v/virajvm/temp_cog_plots/cog_{tgid}.png" ,bbox_inches="tight")
    plt.close()

    aper_mags_r4 = [ cog_mags["g"][-1], cog_mags["r"][-1], cog_mags["z"][-1] ] 


    ##MW EXTINCTION CORRECTING THE MAGNITUDES!

    #if the mags are nans, they will just carry through everything!!
    #Correcting mags for MW extinction

    mw_trans = np.load(save_path + "/source_cat_obs_transmission.npy")

    cog_mags_mwc = []
    for i in range(3):
        cog_mag_i = final_cog_mtot[i] + 2.5 * np.log10(mw_trans[i])
        cog_mags_mwc.append(cog_mag_i)
        
    aper_mags_r4_mwc = []
    for i in range(3):
        aper_mag_i = aper_mags_r4[i] + 2.5 * np.log10(mw_trans[i])
        aper_mags_r4_mwc.append(aper_mag_i)

    #saving the magnitudes locally in folder
    np.save(save_path + f"/new_cog_mags{img_flag}.npy", cog_mags_mwc)

    np.save(save_path + f"/aperture_mags_R4{img_flag}.npy", aper_mags_r4_mwc)

    #we use the tractor model based aperture to measure what fraction of pixels are masked!
    try:
        ell_aper_temp, _, _, _ = get_elliptical_aperture( binary_segment_mask = parent_galaxy_mask , sigma = 4, aper_light_image = aper_light_image , img_type = img_type)
        mask_frac_r4 = measure_elliptical_aperture_area_fraction_masked( parent_galaxy_mask.shape , ~org_basic_mask , ell_aper_temp)
    except:
        print(f"Failed to estimate aperture after applying aperture+star mask = {tgid}")
        mask_frac_r4 = 1
        
    #we are returning dictionaries here
    #note that the fiber mags will not be corrected for extinction
    return_dict = {
        "aper_r4_mags": aper_mags_r4_mwc,
        "cog_mags": cog_mags_mwc,
        "fiber_mags": fiber_mags,
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
        "aper_r4_frac_in_image" :aperfrac_in_image_largest_aper,
        "img_path" :save_img_path,
        "aper_radec_cen" : [aper_ra_cen, aper_dec_cen],
        "aper_xy_pix_cen" :  [aper_xpos, aper_ypos], 
        "aper_params" : aper_params, 
        "mask_frac_r4": mask_frac_r4,
        "aper_cen_masked_bool": aper_cen_mask_bool,
        "aper_rad_mus": radius_mus}
    
    return return_dict


#example template empty dictionary used in case outputs are nans
EMPTY_COG_DICT = {
        "aper_r4_mags": 3*[np.nan],
        "cog_mags": [np.nan] * 3,
        "fiber_mags": [np.nan] * 3,
        "cog_mags_err": [np.nan] * 3,
        "cog_params_g": 5*[np.nan],
        "cog_params_g_err": 5*[np.nan],
        "cog_params_r": 5*[np.nan],
        "cog_params_r_err": 5*[np.nan], 
        "cog_params_z": 5*[np.nan],
        "cog_params_z_err": 5*[np.nan],
        "cog_chi2" :[np.nan] * 3, 
        "cog_dof" :[np.nan] * 3,
        "aper_r4_frac_in_image" :np.nan,
        "img_path" : None,
        "cog_decrease_len" : [np.nan] * 3,
        "cog_decrease_mag" : [np.nan] * 3, 
        "aper_radec_cen" : [np.nan, np.nan],
        "aper_xy_pix_cen" :  [np.nan, np.nan], 
        "aper_params" : [np.nan] * 3 ,
        "mask_frac_r4": 1,
        "aper_cen_masked_bool": False,
        "aper_rad_mus": [np.nan] * 4 }


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


    #decide what kind of aperture to estimate. For computational simplicity, for the nearest galaxies, we do the binary image
    #AH HA! This is the tricky business here. When we are the img_type, we need to make sure we are using the tractor model
    if source_zred > 0.003:
        img_type = "light"
    else:
        img_type = "binary"
        

    #get the parent galaxy no isolate mask. This is using tractor r-band model image to estimate the aperture
    r_band_trac_model, grz_band_trac_model, tractor_only_no_isolate_mags, trac_only_w_isolate_rgb, num_trac_sources_no_isolate, parent_no_isolate_cat, tractor_brightest_source_mags_no_isolate = make_tractor_r_band_model_img(tgid, data_arr, save_path, wcs)

    #note that grz_band_trac_model is a list of the tractor models in grz bands respectivelyy
    
    parent_galaxy_no_isolate_mask, r_mu_aperture, r_mu_island, cog_num_seg, cog_num_seg_smooth, on_main_blob, tractor_smooth_segment, not_galaxy_no_isolate_mask = get_new_segment_tractor(r_band_trac_model, fiber_xpix, fiber_ypix, noise_rms_perband[1],  npixels_min, threshold_rms_scale, img_type, save_path)

    #for diagnostic purposes, also estimate the aperture on the actual reconstruct image and store how much outside the image?

    try:
        grz_image_aper_est = reconstruct_data[0] + reconstruct_data[1] + reconstruct_data[2] 
        _, aperfrac_in_grz_image_r4, _, _ = get_elliptical_aperture(binary_segment_mask = main_seg_mask.astype(int) , sigma = 4, aper_light_image = grz_image_aper_est, img_type = img_type  )
    except:
        print(f"Could not estimate aperfrac in image on real data: {tgid}")
        aperfrac_in_grz_image_r4 = np.nan
        
    rgb_img = sdss_rgb([data_arr[0],data_arr[1],data_arr[2]])

    #this is the redshift limit below which we do not bother applying the isolate parent mask
    zred_lim_no_isolate_lim = 0.005
    
    if source_zred < zred_lim_no_isolate_lim:
        #we do not apply any isolate galaxy mask. Will be nans
        
        parent_galaxy_isolate_mask = None

        #for these systems, it is the same! but we will not use the latter one
        final_cog_mask_no_isolate_mask = star_mask | aperture_mask | not_galaxy_no_isolate_mask
        final_cog_mask_isolate_mask = star_mask | aperture_mask | not_galaxy_no_isolate_mask
            
        jaccard_img_path = None
        num_deblend_segs_main_blob = 1
        ncontrast_opt = np.nan
        nearest_blob_dist_pix = np.nan
        
    else:
        rgb_img_mask =  sdss_rgb([reconstruct_data[0], reconstruct_data [1], reconstruct_data [2]])

        #we construct the smooth galaxy mask using the r-band image of the latest reconstruction so far!
        segm_smooth_deblend_opt, num_deblend_segs_main_blob, parent_galaxy_isolate_mask, non_parent_galaxy_mask, nearest_blob_dist_pix, jaccard_img_path, ncontrast_opt = get_isolate_galaxy_mask(img_rgb = rgb_img, img_rgb_mask = rgb_img_mask, r_band_trac_model = r_band_trac_model, r_rms=noise_rms_perband[1], fiber_xpix=fiber_xpix, fiber_ypix=fiber_ypix, file_path=save_path, tgid=tgid, pcnn_val = pcnn_val, radec=source_radec,  segment_map_v2 = np.copy(segment_map_v2), source_zred = source_zred,  r_mu_aperture = r_mu_aperture, r_mu_island = r_mu_island )

        #note: non_parent_galaxy_mask is the mask we apply to the image to mask out secondary galaxies in the image!

        if segm_smooth_deblend_opt is not None:
            
            if num_deblend_segs_main_blob > 1:
                #if more than 1 deblended blobs were found, we use the non_parent_galaxy_mask for masking in COG
                #and we will use the parent_galaxy_mask to isolate tractor sources and define the aperture!
                
                #we will use the parent_galaxy_isolate_mask to estimate the aperture!!

                #only pixels that are in other deblended blobs and not the main deblend blob are true 
                non_parent_galaxy_mask_for_cog = non_parent_galaxy_mask.astype(bool)
            else:
                #only 1 is found and we do not do anything and keep everything as is. 
                #We just construct a mask full of Falses, as when we apply this, we will not be masking anything!
                non_parent_galaxy_mask_for_cog = np.zeros_like(aperture_mask , dtype = bool) 
                #we will still use the parent_galaxy_isolate_mask to estimate the aperture. 
            
        else:
            #no smooth segment was found. This means that we will be reverting to the original no isolate mask        
            non_parent_galaxy_mask_for_cog = np.zeros_like(aperture_mask, dtype = bool) 

            #if segm_smooth_deblend_opt is None, this will also be None, but just to make sure 
            parent_galaxy_isolate_mask = None
            

        #the star mask is already in the aperture_mask, but just included here for completeness.
        #the non_parent_galaxy_mask_for_cog is the mask we will apply to the reconstruct data to get the final version!

        final_cog_mask_no_isolate_mask = star_mask | aperture_mask | not_galaxy_no_isolate_mask
        final_cog_mask_isolate_mask = star_mask | aperture_mask | non_parent_galaxy_mask_for_cog | not_galaxy_no_isolate_mask
    #NOTE: the parent galaxy mask are binary images. 1 is on segment, 0 is bkg

    #this is just the simple star + different color pixel mask
    org_basic_mask = star_mask | aperture_mask
        
    #QUICK DETOUR: USING THE OPTIMAL DEBLENDING PARAMETER, MAKE THE SIMPLEST PHOTO MEASUREMENT
    simplest_photo_mags, simplest_photo_island_dist_pix, simplest_photo_aper_frac_in_image,  simple_rgb_trac_parent  = get_simplest_photometry(rgb_img,  noise_rms_perband[1], fiber_xpix, fiber_ypix, save_path, ncontrast_opt = ncontrast_opt )
    #we add these in the final catalog
    
    #we make two copies of image, one with the smooth blob mask applied and one with it!
    reconstruct_data_no_isolate = np.copy(reconstruct_data)
    reconstruct_data_no_isolate[:, final_cog_mask_no_isolate_mask] = 0
    reconstruct_data_no_isolate_dict = { "g": reconstruct_data_no_isolate[0], "r": reconstruct_data_no_isolate[1], "z": reconstruct_data_no_isolate[2] }

    reconstruct_data[:,final_cog_mask_isolate_mask] = 0
    #construct the reconstruct_data dictionary
    reconstruct_data_dict = { "g": reconstruct_data[0], "r": reconstruct_data[1], "z": reconstruct_data[2] }

    #save this all!!
    rgb_350 = process_img(reconstruct_data, cutout_size = 350, org_size = np.shape(data_arr)[1] )
    rgb_128 = process_img(reconstruct_data, cutout_size = 128, org_size = np.shape(data_arr)[1] )

    ##save some diagnostic images!
    ax_256 = make_subplots(ncol=1,nrow = 1)
    ax_256[0].set_title("Reconstructed Image (with isolate) 350x350",fontsize = 12)
    ax_256[0].imshow(rgb_350, origin='lower')
    ax_256[0].set_xlim([0,350])
    ax_256[0].set_ylim([0,350])
    ax_256[0].set_xticks([])
    ax_256[0].set_yticks([])
    plt.savefig(save_path + "/reconstruct_image_350.png",bbox_inches="tight")
    plt.close()

    ax_128 = make_subplots(ncol=1,nrow = 1)
    ax_128[0].set_title("Reconstructed Image (with isolate) 128x128",fontsize = 12)
    ax_128[0].imshow(rgb_128, origin='lower')
    ax_128[0].set_xlim([0,128])
    ax_128[0].set_ylim([0,128])
    ax_128[0].set_xticks([])
    ax_128[0].set_yticks([])
    plt.savefig(save_path + "/reconstruct_image_128.png",bbox_inches="tight")
    plt.close()
        
    np.save(save_path + "/final_mask_cog_with_isolate.npy", final_cog_mask_isolate_mask  )
    np.save(save_path + "/final_mask_cog_no_isolate.npy", final_cog_mask_no_isolate_mask  )

    if parent_galaxy_isolate_mask is None and parent_galaxy_no_isolate_mask is None:
        #basically, we will NOT run cog at all and just return nans here!! 
        np.save(save_path + "/parent_galaxy_segment_mask.npy", np.zeros_like(parent_galaxy_isolate_mask)   )

        print("--"*6)
        print(f"TARGETID={tgid} has no detections and thus the segmentation map for COG is None. COG will not be run. We will revert to original Tractor mags!")
        print(f"{save_path}")
        print("--"*6)

        FINAL_COG_DICT = {}
    
        ##adding the cog_isolate params
        for ki,val in EMPTY_COG_DICT.items():
            FINAL_COG_DICT[ki] = copy.deepcopy(val)

        #adding the cog no isoalte params
        for ki,val in EMPTY_COG_DICT.items():
            FINAL_COG_DICT[ki + "_no_isolate"] = copy.deepcopy(val)

        ##make an empty tractor cog dict
        empty_tractor_cog_dict = make_empty_tractor_cog_dict()

        for ki, val in empty_tractor_cog_dict.items():
            FINAL_COG_DICT[ki + "_no_isolate"] = copy.deepcopy(val)

        for ki, val in empty_tractor_cog_dict.items():
            FINAL_COG_DICT[ki + "_w_isolate"] = copy.deepcopy(val)


        #adding the other params
        FINAL_COG_DICT["jaccard_path"] = None
        FINAL_COG_DICT["deblend_smooth_num_seg"] = num_deblend_segs_main_blob 
        FINAL_COG_DICT["tractor_parent_isolate_mags"] = [np.nan, np.nan, np.nan]
        FINAL_COG_DICT["tractor_parent_no_isolate_mags"] = [np.nan, np.nan, np.nan]
        FINAL_COG_DICT["revert_to_org_tractor"] = True
        FINAL_COG_DICT["aper_r2_mu_r_ellipse_tractor"] = r_mu_aperture
        FINAL_COG_DICT["aper_r2_mu_r_island_tractor"] = r_mu_island
        FINAL_COG_DICT["deblend_blob_dist_pix"] = nearest_blob_dist_pix
        FINAL_COG_DICT["cog_segment_nseg"] = cog_num_seg
        FINAL_COG_DICT["cog_segment_nseg_smooth"] = cog_num_seg_smooth
        FINAL_COG_DICT["cog_segment_on_blob"] = on_main_blob
        FINAL_COG_DICT["num_trac_source_no_isolate"] = num_trac_sources_no_isolate
        FINAL_COG_DICT["num_trac_source_isolate"] = np.nan
        FINAL_COG_DICT["aperfrac_in_image_data_r4"] = aperfrac_in_grz_image_r4
        FINAL_COG_DICT["simple_photo_mags"] =  simplest_photo_mags
        FINAL_COG_DICT["simple_photo_island_dist_pix"] =  simplest_photo_island_dist_pix
        FINAL_COG_DICT["simplest_photo_aper_frac_in_image"] =  simplest_photo_aper_frac_in_image
        
        FINAL_COG_DICT["tractor_brightest_source_mags_no_isolate"] =  tractor_brightest_source_mags_no_isolate
        FINAL_COG_DICT["tractor_brightest_source_mags_w_isolate"] =  [np.nan, np.nan,np.nan]
        
        trac_only_w_isolate_rgb = np.ones_like(rgb_img)

    else:           
        ## measure the cog stuff for the tractor no isolate parent reconstruction
        tractor_no_isolate_model_dict = {"g": grz_band_trac_model[0], "r": grz_band_trac_model[1], "z": grz_band_trac_model[2] }
        tractor_no_isolate_aper_light_img = grz_band_trac_model[0] + grz_band_trac_model[1] + grz_band_trac_model[2]
        
        tractor_cog_dict_no_isolate = basic_cog_fitting_subfunction(tractor_no_isolate_model_dict, parent_galaxy_no_isolate_mask, wcs, org_basic_mask, aper_light_image = tractor_no_isolate_aper_light_img, img_type= img_type, tgid = tgid, save_path = save_path)
        ##^^these are the elliptical aperture parameters and the cog parameters on the tractor only model! 

        ##save the tractor only mags!!
        tractor_only_isolate_mags, parent_isolate_cat, trac_only_w_isolate_rgb, tractor_cog_dict_w_isolate, tractor_brightest_source_mags_isolate, tractor_isolate_aper_light_img = get_tractor_isolate_mag(parent_galaxy_isolate_mask, save_path, np.shape(data_arr)[1], tgid, wcs, org_basic_mask, img_type = img_type)

        #the above tractor isolate and tractor no isolate also return the r band image which we will used below for aperture estimation

        if parent_isolate_cat is not None:
            num_trac_sources_isolate = len(parent_isolate_cat)
        else:
            num_trac_sources_isolate = np.nan

        if trac_only_w_isolate_rgb is None:
            trac_only_w_isolate_rgb = np.ones_like(rgb_img)

        if parent_galaxy_isolate_mask is not None:
            parent_galaxy_isolate_mask = parent_galaxy_isolate_mask.astype(int) 
            np.save(save_path + "/parent_galaxy_isolate_mask.npy", parent_galaxy_isolate_mask ) 
        else:
            np.save(save_path + "/parent_galaxy_isolate_mask.npy", np.zeros_like(aperture_mask) ) 
            

        if parent_galaxy_no_isolate_mask is not None:
            parent_galaxy_no_isolate_mask = parent_galaxy_no_isolate_mask.astype(int)
            np.save(save_path + "/parent_galaxy_no_isolate_mask.npy", parent_galaxy_no_isolate_mask )  
        else:
            
            np.save(save_path + "/parent_galaxy_no_isolate_mask.npy", np.zeros_like(aperture_mask) )  
            
        ##RUNNING THE TWO KINDS OF COG PHOTOMETRY!! One with mask, and one without mask!

        isolate_input_dict = {
            "data_arr": np.copy(data_arr),
            "tgid": tgid,
            "save_path": save_path,
            "parent_source_catalog": parent_isolate_cat,
            "tractor_only_mags": tractor_only_isolate_mags,
            "tractor_dr9_mags": tractor_dr9_mags,
            "tractor_bkg_model": tractor_bkg_model,
            "data_arr_no_bkg_no_blend": data_arr_no_bkg_no_blend,
            "segment_map_smooth_new": tractor_smooth_segment,
            "wcs": wcs,
            "fiber_xpix": fiber_xpix,
            "fiber_ypix": fiber_ypix,
            "ra": source_radec[0],
            "dec": source_radec[1], 
            "zred": source_zred
        }

        #making a deep copy of this dict as it will be modified inside one of these functions
        no_isolate_input_dict = copy.deepcopy(isolate_input_dict)
        no_isolate_input_dict["parent_source_catalog"] = parent_no_isolate_cat
        no_isolate_input_dict["tractor_only_mags"] = tractor_only_no_isolate_mags
 

        #if we are in the redshift regime, where we just do not run the separate ISOLATE mask step
        if source_zred < zred_lim_no_isolate_lim:
            #run the no isolate photo
            if parent_galaxy_no_isolate_mask is not None and tractor_no_isolate_aper_light_img is not None:
                cog_NO_isolate_dict = cog_fitting_subfunction(no_isolate_input_dict, reconstruct_data_no_isolate_dict, parent_galaxy_no_isolate_mask, final_cog_mask_no_isolate_mask, org_basic_mask,  img_flag = "_no_isolate", img_type = img_type, aper_light_image = tractor_no_isolate_aper_light_img)
            else:
                cog_NO_isolate_dict = {}
                #we want to return nans
                for ki,val in EMPTY_COG_DICT.items():
                    cog_NO_isolate_dict[ki] = copy.deepcopy(val)
                
            #run the isolate photo
            cog_isolate_dict = {}
            #we want to return nans
            for ki,val in EMPTY_COG_DICT.items():
                cog_isolate_dict[ki] = copy.deepcopy(val)

        else:
            #we are in the redshift regime where we will apply both isolate and no isolate mask!

            #run the isolate photo
            if parent_galaxy_isolate_mask is not None and tractor_isolate_aper_light_img is not None:
                cog_isolate_dict = cog_fitting_subfunction(isolate_input_dict, reconstruct_data_dict, parent_galaxy_isolate_mask, final_cog_mask_isolate_mask, org_basic_mask, img_flag = "", img_type = img_type, aper_light_image = tractor_isolate_aper_light_img)
            else:
                cog_isolate_dict = {}
                #we want to return nans
                for ki,val in EMPTY_COG_DICT.items():
                    cog_isolate_dict[ki] = copy.deepcopy(val)

    
            #run the no isolate photo
            if parent_galaxy_no_isolate_mask is not None and tractor_no_isolate_aper_light_img is not None:
                cog_NO_isolate_dict = cog_fitting_subfunction(no_isolate_input_dict,reconstruct_data_no_isolate_dict, parent_galaxy_no_isolate_mask, final_cog_mask_no_isolate_mask,  org_basic_mask, img_flag = "_no_isolate", img_type = img_type, aper_light_image = tractor_no_isolate_aper_light_img)
            else:            
                cog_NO_isolate_dict = {}
                #we want to return nans
                for ki,val in EMPTY_COG_DICT.items():
                    cog_NO_isolate_dict[ki] = copy.deepcopy(val)


        #initialize the empty output dict!
        FINAL_COG_DICT = {}
    
        ##adding the cog_isolate params
        for ki in EMPTY_COG_DICT.keys():
            FINAL_COG_DICT[ki] = cog_isolate_dict[ki]

        for ki in EMPTY_COG_DICT.keys():
            FINAL_COG_DICT[ki + "_no_isolate"] = cog_NO_isolate_dict[ki]


        ##add the tractor only based cog propeties
        for ki, val in tractor_cog_dict_no_isolate.items():
            FINAL_COG_DICT[ki + "_no_isolate"] = copy.deepcopy(val)

        for ki, val in tractor_cog_dict_w_isolate.items():
            FINAL_COG_DICT[ki + "_w_isolate"] = copy.deepcopy(val)
    
        FINAL_COG_DICT["jaccard_path"] = jaccard_img_path
        FINAL_COG_DICT["deblend_smooth_num_seg"] = num_deblend_segs_main_blob 
        FINAL_COG_DICT["tractor_parent_isolate_mags"] = tractor_only_isolate_mags
        FINAL_COG_DICT["tractor_parent_no_isolate_mags"] = tractor_only_no_isolate_mags
        FINAL_COG_DICT["revert_to_org_tractor"] = False
        FINAL_COG_DICT["aper_r2_mu_r_ellipse_tractor"] = r_mu_aperture
        FINAL_COG_DICT["aper_r2_mu_r_island_tractor"] = r_mu_island
        FINAL_COG_DICT["deblend_blob_dist_pix"] = nearest_blob_dist_pix
        FINAL_COG_DICT["cog_segment_nseg"] = cog_num_seg
        FINAL_COG_DICT["cog_segment_nseg_smooth"] = cog_num_seg_smooth
        FINAL_COG_DICT["cog_segment_on_blob"] = on_main_blob
        FINAL_COG_DICT["num_trac_source_no_isolate"] = num_trac_sources_no_isolate
        FINAL_COG_DICT["num_trac_source_isolate"] = num_trac_sources_isolate
        FINAL_COG_DICT["aperfrac_in_image_data_r4"] = aperfrac_in_grz_image_r4
        FINAL_COG_DICT["simple_photo_mags"] =  simplest_photo_mags
        FINAL_COG_DICT["simple_photo_island_dist_pix"] =  simplest_photo_island_dist_pix
        FINAL_COG_DICT["simplest_photo_aper_frac_in_image"] =  simplest_photo_aper_frac_in_image

        FINAL_COG_DICT["tractor_brightest_source_mags_no_isolate"] =  tractor_brightest_source_mags_no_isolate
        FINAL_COG_DICT["tractor_brightest_source_mags_w_isolate"] =   tractor_brightest_source_mags_isolate
        
    ###MAKE A MULTI PANEL PLOT SAVING THE DIFFERENT RECONSTRUCTIONS IN ONE PLACE!
    ## -> IMG, cog with isolate, cog without mask, tractor only parent, and then simple reconstruct

    fig,ax = make_subplots(ncol = 5, nrow = 1, return_fig=True, col_spacing = 0.1)
    
    ax[0].set_title(f"{tgid}",fontsize = 12)
    ax[0].imshow(rgb_img, origin="lower")
    
    ax[1].set_title(r"COG w/isolate",fontsize = 10)
    #let us save this reconstructed galaxy!!
    try:
        data_cog_wiso = np.load(save_path + f"/final_reconstruct_galaxy_subfunction.npy")
        rgb_cog_wiso = sdss_rgb( data_cog_wiso)
    except:
        rgb_cog_wiso = np.ones_like(rgb_img)
    ax[1].imshow(rgb_cog_wiso,origin="lower")
            
    ax[2].set_title(r"COG wo/isolate",fontsize = 10)
    try:
        data_cog_wo_iso = np.load(save_path + f"/final_reconstruct_galaxy_subfunction_no_isolate.npy")
        rgb_cog_wo_iso = sdss_rgb( data_cog_wo_iso)
    except:
        rgb_cog_wo_iso = np.ones_like(rgb_img)
    ax[1].imshow(rgb_cog_wo_iso,origin="lower")  
    
    ax[3].set_title(r"Trac-only w/isolate",fontsize = 10)
    ax[3].imshow(trac_only_w_isolate_rgb, origin="lower")
    
    ax[4].set_title(r"Simple Photo",fontsize = 10)
    ax[4].imshow(simple_rgb_trac_parent, origin="lower")
    
    for axi in ax:
        axi.set_xticks([])
        axi.set_yticks([])
     
    plt.savefig(save_path + "/different_parent_reconstruct_panel.png",bbox_inches="tight")
    plt.close(fig)
    
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

    lie_on_segment = input_dict["LIE_ON_APER_SEGMENT"]


    if lie_on_segment == 1:
    
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

    else:
        print(f"COG WILL NOT BE RUN FOR {source_tgid}. Returning blank empty dictionary")

        FINAL_COG_DICT = {}
    
        ##adding the cog_isolate params
        for ki,val in EMPTY_COG_DICT.items():
            FINAL_COG_DICT[ki] = copy.deepcopy(val)

        #adding the cog no isoalte params
        for ki,val in EMPTY_COG_DICT.items():
            FINAL_COG_DICT[ki + "_no_isolate"] = copy.deepcopy(val)

    
        ##make an empty tractor cog dict
        empty_tractor_cog_dict = make_empty_tractor_cog_dict()

        for ki, val in empty_tractor_cog_dict.items():
            FINAL_COG_DICT[ki + "_no_isolate"] = copy.deepcopy(val)

        for ki, val in empty_tractor_cog_dict.items():
            FINAL_COG_DICT[ki + "_w_isolate"] = copy.deepcopy(val)

        #adding the other params
        FINAL_COG_DICT["jaccard_path"] = None
        FINAL_COG_DICT["deblend_smooth_num_seg"] = 0
        FINAL_COG_DICT["deblend_smooth_dist_pix"] =  np.nan
        FINAL_COG_DICT["tractor_parent_isolate_mags"] = [np.nan, np.nan, np.nan]
        FINAL_COG_DICT["tractor_parent_no_isolate_mags"] = [np.nan, np.nan, np.nan]
        FINAL_COG_DICT["revert_to_org_tractor"] = True
        FINAL_COG_DICT["aper_r2_mu_r_ellipse_tractor"] = np.nan
        FINAL_COG_DICT["aper_r2_mu_r_island_tractor"] = np.nan
        FINAL_COG_DICT["deblend_blob_dist_pix"] = np.nan
        FINAL_COG_DICT["cog_segment_nseg"] = np.nan
        FINAL_COG_DICT["cog_segment_nseg_smooth"] = np.nan
        FINAL_COG_DICT["cog_segment_on_blob"] = False
        FINAL_COG_DICT["num_trac_source_no_isolate"] = np.nan
        FINAL_COG_DICT["num_trac_source_isolate"] = np.nan
        FINAL_COG_DICT["aperfrac_in_image_data_r4"] = np.nan
        FINAL_COG_DICT["simple_photo_mags"] =  [np.nan, np.nan, np.nan]
        FINAL_COG_DICT["simple_photo_island_dist_pix"] =  np.nan
        FINAL_COG_DICT["simplest_photo_aper_frac_in_image"] =  np.nan

        FINAL_COG_DICT["tractor_brightest_source_mags_no_isolate"] = [np.nan, np.nan, np.nan]
        FINAL_COG_DICT["tractor_brightest_source_mags_w_isolate"] = [np.nan, np.nan, np.nan]
        
        return FINAL_COG_DICT

    
    
    
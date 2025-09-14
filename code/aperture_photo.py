##In this script, we redo the photometry for shredded objects 

from photutils.background import Background2D, MedianBackground
import os
import sys
import joblib
import scipy.optimize as opt
from photutils.aperture import ApertureStats
import numpy as np
from tqdm import trange
import multiprocessing as mp
from tqdm import tqdm
from astropy.io import fits
from astropy.table import Table, vstack, join
import astropy.coordinates as coord
from scipy.ndimage import binary_dilation
from astropy.table import Column
from astropy import units as u
from astropy.coordinates import SkyCoord
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import LogNorm
import mastcasjobs
from astropy.io import fits
from astropy.wcs import WCS
from photutils.aperture import aperture_photometry, SkyEllipticalAperture, CircularAperture, EllipticalAperture
import matplotlib.patches as patches
from matplotlib.patches import Circle
from photutils.segmentation import detect_sources, deblend_sources
import matplotlib.cm as cm
url_prefix = 'https://www.legacysurvey.org/viewer/'
# Third-party imports 
import requests
from io import BytesIO
from astropy.io import fits
import matplotlib.colors as mcolors
from matplotlib.collections import PatchCollection
from easyquery import Query, QueryMaker
reduce_compare = QueryMaker.reduce_compare
import random
import argparse
import concurrent.futures
import pickle
from photutils.segmentation import make_2dgaussian_kernel
from astropy.convolution import convolve
from aperture_cogs import get_elliptical_aperture, find_nearest_island
from photutils.background import StdBackgroundRMS, MADStdBackgroundRMS
from astropy.stats import SigmaClip
from scipy.stats import skew, kurtosis
from alternative_photometry_methods import get_simplest_photometry
from desi_lowz_funcs import get_elliptical_aperture, measure_elliptical_aperture_area_fraction_masked

rootdir = '/global/u1/v/virajvm/'
sys.path.append(os.path.join(rootdir, 'DESI2_LOWZ'))
from desi_lowz_funcs import print_stage, check_path_existence, get_remove_flag, _n_or_more_lt, is_target_in_south, match_c_to_catalog, calc_normalized_dist, get_sweep_filename, get_random_markers, save_table, make_subplots, sdss_rgb

def conf_interval(x, pdf, conf_level):
    return np.sum(pdf[pdf > x])-conf_level

def mask_bad_flux(flux_vals):
    good_flux_vals = np.where(flux_vals > 0, flux_vals, np.nan)
    return good_flux_vals


def flux_to_mag(flux, zeropoint=22.5):
    # Protect against zero/negative flux
    if flux > 0:
        return zeropoint - 2.5*np.log10(flux)
    else:
        return np.nan  # or some sentinel value


def substitute_bad_mag(source_cat_f):
    """
    Replace bad mags/errors with safe values for GMM input.

    What happens if very bright sources have inf mags?
    """
    for BAND in ("g", "r", "z"):
        # Replace NaN/Inf mags with faint limit
        source_cat_f[f"mag_{BAND}"] = np.nan_to_num(
            source_cat_f[f"mag_{BAND}"], nan=25, posinf=25, neginf=25
        )

        # Replace NaN/Inf errors with large value, not zero
        source_cat_f[f"mag_{BAND}_err"] = np.nan_to_num(
            source_cat_f[f"mag_{BAND}_err"], nan=0, posinf=0, neginf=0
        )

    # Compute colors
    source_cat_f["g-r"] = source_cat_f["mag_g"] - source_cat_f["mag_r"]
    source_cat_f["r-z"] = source_cat_f["mag_r"] - source_cat_f["mag_z"]

    # Errors: avoid zeros
    source_cat_f["g-r_err"] = np.sqrt(source_cat_f["mag_g_err"]**2 + source_cat_f["mag_r_err"]**2)
    source_cat_f["r-z_err"] = np.sqrt(source_cat_f["mag_r_err"]**2 + source_cat_f["mag_z_err"]**2)

    # Clip extreme values to a plausible range
    source_cat_f["g-r"] = np.clip(source_cat_f["g-r"], -3, 3)
    source_cat_f["r-z"] = np.clip(source_cat_f["r-z"], -3, 3)

    return source_cat_f


    
def mask_radius_for_mag(mag):
    '''
    Masking radius of the bright star in degrees
    '''
    # Returns a masking radius in degrees for a star of the given magnitude.
    # Used for Tycho-2 and Gaia stars.
    
    # This is in degrees, and is from Rongpu in the thread [decam-chatter 12099].
    return 1630./3600. * 1.396**(-mag)

def mask_circle(array, x0, y0, r,value = 0):
    '''
    This will be used to mask the circular bright star region!
    '''
    ny, nx = array.shape
    y, x = np.ogrid[:ny, :nx]
    mask = (x - x0)**2 + (y - y0)**2 <= r**2
    array[mask] = value
    return array


def get_binary_mask(mask_data_array, set_maskbits = [2,3,4]):
    '''
    Constructs a binary pixel mask based on whether the given maskbits are set or not
    
    This function is originally taken from 
    https://github.com/MultimodalUniverse/MultimodalUniverse/blob/d20de9b5d50564ca740e170030f807fd870d7f77/scripts/legacysurvey/build_parent_sample.py#L176C5-L176C13
    '''

    # Ensure input is an integer array suitable for bitwise operations
    mask_data_array = np.asarray(mask_data_array)
    if np.ma.isMaskedArray(mask_data_array):
        mask_data_array = mask_data_array.filled(0)
    mask_data_array = mask_data_array.astype(np.uint32)
    
    maskclean = np.ones_like(mask_data_array, dtype=bool)
    #the 14,15 maskbits are related to i-band which we are not using so removed from below list
    for bit in set_maskbits:
        maskclean &= (mask_data_array & 2**bit)==0
    
    maskclean = maskclean.astype(mask_data_array.dtype)

    return maskclean.astype(bool)
    

def make_new_saturated_mask(invvar_data):
    '''
    Makes the new bad pixel mask based on inverse variance = 0 in any one band.
    '''
    
    #we also mask pixels where the inverse variance is zero in at least one of the 3 bands
    final_mask = (invvar_data[0]== 0) | (invvar_data[1]== 0) | (invvar_data[2]== 0)

    ##once I have mask, how to grow it a bit!
    # Apply dilation (repeat to grow by 4 pixels)
    structure = np.ones((3, 3), dtype=bool)
    final_mask_pd = binary_dilation(final_mask, structure=structure, iterations=4)
    
    #we invert it as we want to return a mask of all the good pixels!    
    return ~final_mask_pd
    

def fmt_mag(mag):
    return "NaN" if not np.isfinite(mag) else f"{mag:.2f}"





def run_aperture_pipe(input_dict):
    '''
    Main function that runs the initial aperture photometry pipeline. Identifying the main blob, applying the color associated criterion etc.
    '''

    save_path = input_dict["save_path"]
    source_tgid  = input_dict["tgid"]
    source_ra  = input_dict["ra"] 
    source_dec  = input_dict["dec"]
    source_redshift  = input_dict["redshift"]
    wcs  = input_dict["wcs"]
    data_arr  = input_dict["image_data"]
    mask_arr = input_dict["mask_data"]
    invvar_arr = input_dict["invvar_data"]
    source_cat_f = input_dict["source_cat"]
    overwrite = input_dict["overwrite"]
    pcnn_val = input_dict["pcnn_val"]
    npixels_min = input_dict["npixels_min"]
    threshold_rms_scale = input_dict["threshold_rms_scale"]
    image_size = input_dict["image_size"]
    run_simple_photo = input_dict["run_simple_photo"]
    
    verbose=False

    if verbose:
        print(source_ra, source_dec)
        
    bstar_tuple = input_dict["bright_star_info"]

    sga_tuple = input_dict["sga_info"]

    sga_dist, sga_ndist = sga_tuple[0], sga_tuple[1]
    
    #this is the bright star info that was used to compute the STARDIST, STARFDIST etc.
    #the radius is given in arcsecs and will be plotted as a circle for reference
    bstar_ra, bstar_dec, bstar_radius, bstar_fdist, bstar_mag = bstar_tuple[0], bstar_tuple[1], bstar_tuple[2], bstar_tuple[3], bstar_tuple[4]

    #if the photometry is nan, we make it very faint so it is propagated through the whole pipeline!
    source_cat_f = substitute_bad_mag(source_cat_f)
            
    # source_cat_f = source_cat_f[ ~np.isnan(source_cat_f["g-r_err"]) &  ~np.isnan(source_cat_f["r-z_err"]) &  ~np.isinf(source_cat_f["g-r_err"]) &  ~np.isinf(source_cat_f["r-z_err"]) ]
    # source_cat_f = source_cat_f[ ~np.isnan(source_cat_f["g-r"]) &  ~np.isnan(source_cat_f["r-z"]) &  ~np.isinf(source_cat_f["g-r"]) &  ~np.isinf(source_cat_f["r-z"]) ]

    #get the pixel locations of these sources 
    sources_f_xpix,sources_f_ypix,_ = wcs.all_world2pix(source_cat_f['ra'].data, source_cat_f['dec'].data, 0,1)

    source_cat_f["xpix"] = sources_f_xpix
    source_cat_f["ypix"] = sources_f_ypix

    #now we remove the soruces that are lie on the edge
    edge_remove_mask = (source_cat_f["xpix"].data.astype(int) == image_size) | (source_cat_f["ypix"].data.astype(int) == image_size)

    source_cat_f = source_cat_f[~edge_remove_mask]

    sources_f_xpix = sources_f_xpix[~edge_remove_mask]
    sources_f_ypix = sources_f_ypix[~edge_remove_mask]
    
    gmm_file_zgrid = np.arange(0.001, 0.525,0.025)

    #bool variable that decides if the rephoto pipeline should be run
    do_i_run = False

    ##################
    # get catalog of nearby DR9 sources along with their photo-zs info. Nearby is defined as within 45 arcsecs
    ##################

    ref_coord = SkyCoord(ra=source_ra * u.deg, dec=source_dec * u.deg)
    # sources_coords = SkyCoord(ra=source_cat_f["ra"].data * u.deg, dec=source_cat_f["dec"].data * u.deg)
    # # Compute separations
    # source_seps = ref_coord.separation(sources_coords).arcsec

    ##procedure for selecting a star
    signi_pm = ( np.abs(source_cat_f["pmra"]) * np.sqrt(source_cat_f["pmra_ivar"]) > 2) | ( np.abs(source_cat_f["pmdec"]) * np.sqrt(source_cat_f["pmdec_ivar"]) > 2)
    star_model = ((source_cat_f['type'] == "PSF") | (source_cat_f['type'] == "DUP") )

    ##some stuff choices here. There are some birght HII regions that are also in Gaia and tractor often models them as PSF. So to be lax on star definition. I think I will relax the PMRA cuts
    # is_star = (source_cat_f["ref_cat"] == "G2")  &  ( 
    #     (source_cat_f['type'] == "PSF")  | signi_pm 
    # )     
    is_star = (source_cat_f["ref_cat"] == "G2")  &  star_model  & signi_pm  

    #could also include the DUPLICATE TYPE?

    #what is the distance to the closest star from us?
    all_stars = source_cat_f[is_star]
    if len(all_stars) > 0:
        catalog_coords = SkyCoord(ra=all_stars["ra"].data * u.deg, dec=all_stars["dec"].data * u.deg)
        # Compute separations
        separations = ref_coord.separation(catalog_coords).arcsec
        closest_star_dist = np.min(separations)
        #this will be useful to determine if there is a saturated star in the field or not
        #we get the maximum magnitude in any of the 3 gaia bands
        closest_star_tab = all_stars[np.argmin(separations)]

        star_mags = [
            closest_star_tab["gaia_phot_bp_mean_mag"],
            closest_star_tab["gaia_phot_rp_mean_mag"],
            closest_star_tab["gaia_phot_g_mean_mag"]
        ]

        # Filter out masked values and convert to float
        valid_star_mags = [float(m) for m in star_mags if not np.ma.is_masked(m)]

        # Compute nanmin only if there are valid magnitudes
        if valid_star_mags:
            closest_star_mag = np.nanmin(valid_star_mags)
        else:
            closest_star_mag = np.nan  # Or some fallback value
        
        # closest_star_mag = np.nanmin( [ float(closest_star_tab["gaia_phot_bp_mean_mag"]), float(closest_star_tab["gaia_phot_rp_mean_mag"]),float(closest_star_tab["gaia_phot_g_mean_mag"]) ] )
        
    else:
        closest_star_dist = np.inf
        closest_star_mag = np.inf

    #given the closest star dist, what is it normalized in units of the star radius?
    if closest_star_mag != np.inf:
        closest_star_radius = mask_radius_for_mag( closest_star_mag ) * 3600

        closest_star_norm_dist = closest_star_dist / closest_star_radius
    else:
        closest_star_norm_dist = np.inf

        
    save_summary_png = save_path + "/grz_bands_segments.png"

    if overwrite == False:
        try:
            new_mags = np.load(save_path + "/aper_r3_mags.npy")

            raise ValueError("The correct outputs will not be outputted if overwrite is False!")

            output_dict = {
                    "closest_star_dist": closest_star_dist,
                    "closest_star_mag": closest_star_mag,
                    "aper_r3_mags": 3*[np.nan],
                    "save_path": save_path,
                    "save_summary_png": None,
                    "closest_star_norm_dist": closest_star_norm_dist,
                    "lie_on_segment_island": False,
                    "aper_frac_mask_badpix": np.nan, 
                    "img_frac_mask_badpix":  np.nan,
                    "simple_photo_mags": 3*[np.nan], 
                    "simple_photo_island_dist_pix": np.nan,
                    "simplest_photo_aper_frac_in_image" : np.nan
                }

            return output_dict
            
        except:
            do_i_run = True
    if overwrite == True:
        do_i_run = True
            
    if do_i_run:
        # try:    
        file_index = np.where( source_redshift > gmm_file_zgrid )[0][-1]
        #load the relevant gmm file
        gmm = joblib.load("/pscratch/sd/v/virajvm/redo_photometry_plots/gmm_color_models/gmm_model_idx_%d.pkl"%file_index)
    
        # Create a grid for evaluation
        xmin, xmax = -0.5, 2
        ymin, ymax = -0.5, 1.5
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
    
        #let us save these levels for future reference!
        box_size = image_size
    
        data = { "g": data_arr[0], "r": data_arr[1], "z": data_arr[2]}

        #make the rgb image
        rgb_stuff = sdss_rgb([data["g"],data["r"],data["z"]], ["g","r","z"], scales=dict(g=(2,6.0), r=(1,3.4), z=(0,2.2)), m=0.03)


        #first estimate the background error to use in aperture photometry
        noise_dict = {}
        bkg_properties_dict = {}

        rms_estimator = MADStdBackgroundRMS()
        ##estimate the background rms in each band!
        for bii in ["g","r","z"]:        
            # Apply sigma clipping
            sigma_clip = SigmaClip(sigma=3.0,maxiters=5)
            clipped_data = sigma_clip(data[bii])
            # Estimate RMS
            background_rms = rms_estimator(clipped_data)
            noise_dict[bii] = background_rms

            ##getting some properties of the bkg
            sigma_clip = SigmaClip(sigma=5.0,maxiters=3)
            clipped_ma = sigma_clip(data[bii])
            clipped_data_v2 = clipped_ma.compressed() 
   
            bkg_properties_dict[bii] = {
                    "mean": np.mean(clipped_data_v2),
                    "median": np.mean(clipped_data_v2),
                    "std": np.std(clipped_data_v2),
                    "skewness": skew(clipped_data_v2),
                    "kurtosis": kurtosis(clipped_data_v2),
                    "p5": np.percentile(clipped_data_v2, 5),
                    "p95": np.percentile(clipped_data_v2, 95),
                }

        ##save this dictionary of background properties!
        with open(save_path + "/background_properties_dict.pkl", "wb") as f:
            pickle.dump(bkg_properties_dict, f)
        
        #the rms in the total image will be the rms in the 3 bands added in quadrature
        tot_rms = np.sqrt( noise_dict["g"]**2 + noise_dict["r"]**2 + noise_dict["z"]**2 )
        
        #parameters to think a bit carefully about
        threshold = threshold_rms_scale * tot_rms

        kernel = make_2dgaussian_kernel(3.0, size=5)  # FWHM = 3.0
        
    
        tot_data = np.sum(data_arr, axis=0)

        if np.shape(tot_data) != (image_size,image_size):
            raise ValueError(f"Dimensions of summed grz image are not ({image_size},{image_size})")
        
        convolved_tot_data = convolve( tot_data, kernel )

        segment_map = detect_sources(convolved_tot_data, threshold, npixels=npixels_min) 
        
        segm_deblend = deblend_sources(convolved_tot_data, segment_map,
                                   npixels=npixels_min,nlevels=16, contrast=0.01,
                                   progress_bar=False)
        
        #get the segment number where the main source of interest lies in
        #we maintain 2 copies because one of them will be changed in case the desi fiber does not lie on a island segment
        fiber_xpix, fiber_ypix,_ = wcs.all_world2pix(source_ra, source_dec,0,1)
        fiber_xpix_org, fiber_ypix_org,_ = wcs.all_world2pix(source_ra, source_dec,0,1)

        ## note that these 2d arrays are [y-coord, x-coord]
        island_num = segment_map.data[int(fiber_ypix),int(fiber_xpix)]
                
        #any segment not part of this main segment is considered to be a different source 
        #make a copy of the segment array
        segment_map_v2 = np.copy(segment_map.data)

        #it is possible that the source lies on background and not in one of the segmented islands.
        #this is possible if the source was located in the very faint extremeties which is possible for ELGs

        lie_on_segment_island = 1
        min_dist_pix = 0
        if island_num == 0:
            lie_on_segment_island = 0
            print_stage(f"Following source does not lie on a segment island, TGID:{source_tgid}")
            #finds the nearest island within 10'' and associates it with that
            segment_map_v2, island_num, fiber_xpix, fiber_ypix, min_dist_pix = find_nearest_island(segment_map_v2, fiber_xpix, fiber_ypix)

            if island_num is None:

                np.save(save_path + "/main_segment_map.npy", np.zeros_like(tot_data) )
                
                print_stage(f"Following source being skipped for aperture photo: TGID:{source_tgid}")
                #we will just be returning to the original tractor mag from DR9 and no aperture and COG for this!
                output_dict = {
                        "closest_star_dist": closest_star_dist,
                        "closest_star_mag": closest_star_mag,
                        "aper_r3_mags": 3*[np.nan],
                        "tractor_dr9_mags": 3*[np.nan],
                        "save_path": save_path,
                        "save_summary_png": None,
                        "closest_star_norm_dist": closest_star_norm_dist,
                        "lie_on_segment_island": lie_on_segment_island,
                        "first_min_dist_island_pix" : min_dist_pix,
                        "aper_frac_mask_badpix": np.nan, 
                        "img_frac_mask_badpix":  np.nan,
                        "simple_photo_mags": 3*[np.nan], 
                        "simple_photo_island_dist_pix": np.nan,
                        "simplest_photo_aper_frac_in_image" : np.nan
                    }
 
                return output_dict
            

                
    
        #define the aperture that will be used for plotting the bright star
        #these should be in the pixel coordinates
        #converting star center ra,dec into pixels
        if bstar_ra != 99:
            bstar_radius_pix = bstar_radius/ 0.262 #converting the radius in arcseconds to pixels
            bstar_xpix,bstar_ypix,_ = wcs.all_world2pix(bstar_ra, bstar_dec, 0,1)
            aperture_for_bstar_1 = CircularAperture( (float(bstar_xpix), float(bstar_ypix)) , r =  bstar_radius_pix)
            aperture_for_bstar_34 = CircularAperture( (float(bstar_xpix), float(bstar_ypix)) , r =  0.75*bstar_radius_pix)
            aperture_for_bstar_12 = CircularAperture( (float(bstar_xpix), float(bstar_ypix)) , r =  0.5*bstar_radius_pix)

        source_cat_all_stars = source_cat_f[is_star]


        ##compute the max mag of this star
        source_cat_all_stars["max_mag"] = np.min( (  np.array(source_cat_all_stars["gaia_phot_bp_mean_mag"]), np.array(source_cat_all_stars["gaia_phot_rp_mean_mag"]),np.array(source_cat_all_stars["gaia_phot_g_mean_mag"])    )       ,axis = 0)

        ##constructing the stellar mask. This can be used to mask the appropriate pixels downstream
        #we want the location of the stars to be 1s or Trues!
        star_mask = np.zeros_like(tot_data)

        #just in case of the wild possibility that our main bright star is not located in the catalog, we just mask it again here!
        if bstar_ra != 99:
            mask_radius_frac = 0.5
    
            star_mask = mask_circle(star_mask, bstar_xpix,bstar_ypix, mask_radius_frac * bstar_radius_pix, value = 1)
            #this ensures that we are just masking the stellar region and not the whole deblended segment!
        else:
            #if no bright star exists, then we do nothing
            pass

        #separation between our bstar and
        if bstar_ra != 99:
            all_star_seps = SkyCoord(bstar_ra, bstar_dec, unit='deg').separation(SkyCoord( source_cat_all_stars["ra"].data , source_cat_all_stars["dec"].data, unit='deg')).arcsec

        #we will iteratively update the star_mask!
        for j in range(len(source_cat_all_stars)):
            #these are all the stars in the image!
            ##masking the brightest, most influential star!
            if bstar_ra != 99 and all_star_seps[j] < 1:
                #if the bright star exists in this catalog, as we have already masked it, we do not need to do anything
                pass           
            else:
                #compute its radius and mask it! It is returned in degrees and so convert to arcseconds
                star_radius_i_as =  mask_radius_for_mag( float(source_cat_all_stars["max_mag"][j])) * 3600

                #same thing as before
                mask_radius_frac = 0.5
    
                star_radius_i_pix = mask_radius_frac*star_radius_i_as/0.262
                #mask it now!
                star_mask = mask_circle(star_mask, source_cat_all_stars["xpix"][j], source_cat_all_stars["ypix"][j], star_radius_i_pix, value = 1 )                 

        star_mask = star_mask.astype(bool)
        ## the star mask is ready!!!

        #construct the mask from the saturated pixels from bright spikes around stars
        good_pixel_mask = make_new_saturated_mask(invvar_arr)

        #note that in the below case, if the segment_map_v2 has been updated to be 99999, we will need to use the 
        #updated segment_map_v2 and not segment_map.data
        #when the island_num == 0 condition is not triggered, then this is the same and does not matter
        segment_map_v2_copy = np.copy(segment_map_v2)
        
        #pixels that are part of main segment island are called 2
        segment_map_v2[segment_map_v2_copy == island_num] = 2
        #all other segments that are not background are called 1
        segment_map_v2[(segment_map_v2_copy != island_num) & (segment_map_v2_copy > 0)] = 1
        #rest all remains 0

        #constructing the fiducial aperture for doing photometry
        aperture_for_phot, _, _,_ = get_elliptical_aperture( segment_map_v2, sigma = 3, aperture_mask = star_mask, id_num = 2  )
        aperture_for_phot_noscale, _, _,_ = get_elliptical_aperture( segment_map_v2, sigma = 1, aperture_mask = star_mask, id_num = 2  )

        #given the saturated pixel mask and above aperture, what fraction of the pixels in the image are masked by this, and what fraction of pixels within the original R3 aperture are masked by this?
        aper_frac_mask_badpix = measure_elliptical_aperture_area_fraction_masked(data_arr[0].shape, good_pixel_mask, aperture_for_phot)

        img_frac_mask_badpix = np.sum(~good_pixel_mask)/np.sum( np.ones_like(good_pixel_mask) )

        #make a copy of the deblended array, this is the one where even the main segment is split into different deblended components
        segm_deblend_v2 = np.copy(segm_deblend.data)
        #create an array of nans with same shape
        segm_deblend_v3 = np.zeros_like(segm_deblend.data) * np.nan

        #we will populating segm_deblend_v3 with the different segments that are part of main segment island 
        #get deblend segments ids that are part of the main segment island
        deblend_ids = np.unique(segm_deblend_v2[segment_map_v2 == 2])

        #what is the deblend id where our main source is in?
        deblend_island_num = segm_deblend_v2[int(fiber_ypix),int(fiber_xpix)]

        #create another deblend image where we relabel the deblend ids of the main segment
        #note the 0 pixel means background and hence we have i+1 here
        new_deblend_ids = []
        for i,di in enumerate(deblend_ids):
            #we skip the i=0 step as that is the background
            segm_deblend_v3[(segm_deblend_v2 == di)] = i+1
            new_deblend_ids.append(i+1)
        new_deblend_ids = np.array(new_deblend_ids)

        ## let us save the main segment map for tractor model use later!
        np.save(save_path + "/main_segment_map.npy",segm_deblend_v3 )

        new_deblend_island_num = int(new_deblend_ids[deblend_ids == deblend_island_num])
 
        source_cat_f["new_deblend_id"] = -99*np.ones_like(sources_f_xpix)

        #what are the deblend segment ids of our DR9 sources in our region?
        for k in range(len(source_cat_f)):
            #if the source is not part of any deblend island part of main segment,then we assign -99 as id
            if np.isnan( segm_deblend_v3[ int(sources_f_ypix[k]), int(sources_f_xpix[k]) ]  ):    
                source_cat_f["new_deblend_id"][k] =  -99
            else:
                source_cat_f["new_deblend_id"][k] = int(segm_deblend_v3[ int(sources_f_ypix[k]), int(sources_f_xpix[k]) ])

        ## compute the colors of each deblend segment that is part of the main segment
        all_segs_grs = []
        all_segs_rzs = []
        
        all_segs_grs_err = []
        all_segs_rzs_err = []

        ## construct the total star mask which will allow us to get the segment colors that are not affected by star colors.
        # we should mask some pixels to prevent negative values here ... 

        data_for_colors = {}
        
        for biii in ["g","r","z"]:
            data[biii][ np.isnan(data[biii]) ] = 0
            data[biii][ np.isinf(data[biii]) ] = 0
            data[biii][ data[biii] < -5*noise_dict[biii] ] = 0

            #copying it to different array
            data_for_colors_i = np.copy(data[biii])
            #making all the stars = 0
            data_for_colors_i[star_mask] = 0

            #we will also mask the spike regions in this part!!
            data_for_colors_i[~good_pixel_mask] = 0
            
            #saving this to the updated dictionary!
            data_for_colors[biii] = data_for_colors_i

        ##when colors are being computed we will mask the stellar pixels
        for din in new_deblend_ids:
            gband_flux_i = np.sum(data_for_colors["g"][segm_deblend_v3 == din ])
            rband_flux_i = np.sum(data_for_colors["r"][segm_deblend_v3 == din ])
            zband_flux_i = np.sum(data_for_colors["z"][segm_deblend_v3 == din ])

            #as all the error is assumed to be the same so we add them in quadrature
            #this is a very approximate error based on the estimated bkg rms. To do this perfectly, we would need to use the inverse variance maps
            gflux_err_i = np.sqrt( np.sum(segm_deblend_v3 == din) * noise_dict["g"]**2 )
            rflux_err_i = np.sqrt( np.sum(segm_deblend_v3 == din) * noise_dict["r"]**2 )
            zflux_err_i = np.sqrt( np.sum(segm_deblend_v3 == din) * noise_dict["z"]**2 )

            #make the 0 or negative flux values nans to avoid the many warning messages
            gband_flux_i = mask_bad_flux(gband_flux_i)
            rband_flux_i = mask_bad_flux(rband_flux_i)
            zband_flux_i = mask_bad_flux(zband_flux_i)
            
            #convert the flux to mags to compute colors
            gband_mag_i = 22.5- 2.5*np.log10(gband_flux_i)
            rband_mag_i = 22.5- 2.5*np.log10(rband_flux_i)
            zband_mag_i = 22.5- 2.5*np.log10(zband_flux_i)

            #conver to error assuming small error
            gmag_err_i = 1.087*( gflux_err_i / gband_flux_i)
            rmag_err_i = 1.087*( rflux_err_i / rband_flux_i) 
            zmag_err_i = 1.087*( zflux_err_i / zband_flux_i) 

            #compute the colors and its error
            gr_i = gband_mag_i - rband_mag_i
            gr_err_i = np.sqrt( gmag_err_i**2 + rmag_err_i**2)
            rz_i = rband_mag_i - zband_mag_i
            rz_err_i = np.sqrt( rmag_err_i**2 + zmag_err_i**2)

            ##adding info to lists
            all_segs_grs.append(gr_i)
            all_segs_rzs.append(rz_i)
            all_segs_grs_err.append(gr_err_i)
            all_segs_rzs_err.append(rz_err_i)

        #converting to arrays
        all_segs_grs = np.array(all_segs_grs)
        all_segs_rzs = np.array(all_segs_rzs)
        all_segs_grs_err = np.array(all_segs_grs_err)
        all_segs_rzs_err = np.array(all_segs_rzs_err)

        #note that if there are segments that include the star, the colors are determined from pixels that are not masked by the star!

        #for a sanity check, plot these sources to see how they are doing!
        in_main_islands = (source_cat_f["new_deblend_id"] != -99)
        
        #find difference between source and all the other catalog objects
        ref_coord = SkyCoord(ra=source_ra * u.deg, dec=source_dec * u.deg)
        catalog_coords = SkyCoord(ra=source_cat_f["ra"].data * u.deg, dec=source_cat_f["dec"].data * u.deg)
        # Compute separations
        separations = ref_coord.separation(catalog_coords).arcsec

        #storing the separations of all sources from our main fiber source so we can remove it from consideration later
        source_cat_f["separations"] = separations
        
        source_cat_obs = source_cat_f[np.argmin(separations)]

        #Due to some mistake in the LOWZ source catalog, and how the overlapping region was handled, for LOWZ the separations can be 
        #non-zero in the overlapping region
        if np.min(separations) > 1:
            print("---")
            print("Distance between source and closest object in the photometry catalog in area =",np.min(separations))
            print(source_ra, source_dec)
            print(save_path)
            print("---")
            
        #get the original magnitude of the object

        #this magnitude is not milky way extinction corrected
        source_mag_g = source_cat_obs["mag_g"] 
        source_mag_r = source_cat_obs["mag_r"] 
        source_mag_z = source_cat_obs["mag_z"] 
        
        source_mag_g_mwc = source_mag_g  + 2.5 * np.log10(source_cat_obs["mw_transmission_g"])
        source_mag_r_mwc = source_mag_r  + 2.5 * np.log10(source_cat_obs["mw_transmission_r"])
        source_mag_z_mwc = source_mag_z  + 2.5 * np.log10(source_cat_obs["mw_transmission_z"])

        ####
        source_cat_inseg_signi = source_cat_f[ in_main_islands ]
        #updating the source type masks
        is_star_inseg_signi = is_star[ in_main_islands ]
        
        markers_rnd = get_random_markers(len(source_cat_inseg_signi) )
        source_cat_inseg_signi["marker"] = markers_rnd

        #how lenient we are in defining the color-color box
        col_lenient = 0.1
      
        #generating empty mask array that has same shape as data
        #note that when passing the aperture masking array to the function, 1 = mask, 0 = no mask
        #so we will invert it at end. For our purposes till then, 1 means include/no mask, 0 = mask 
        aperture_mask = np.ones_like( tot_data)

        #this is the mask for only the star or different deblend segments in the main segment if one exists!
        #we will not be masking yet sources in the exterior!
        #we will be passing this to the curve of growth analysis later
        aperture_mask_no_bkg = np.ones_like( tot_data)
        
        #firstly mask all the identified segments that are not part of our main segment island!
        aperture_mask[segment_map_v2 == 1] = 0

        #colors of reference deblend segment
        ref_deblend_gr = float(all_segs_grs[new_deblend_island_num - 1])
        ref_deblend_rz = float(all_segs_rzs[new_deblend_island_num - 1])

        #these minimum choices roughly line up with the lower end of the color-color contour
        #they were originally 0.2 and 0, but becayse of the +0.1 later, we subtract -0.1 here to make sure the final cut is 0.2 and 0!
        ref_deblend_gr = np.maximum(ref_deblend_gr, 0.1) 
        ref_deblend_rz = np.maximum(ref_deblend_rz, -0.05) 

        #select objects that are very blue that we want to keep
        very_blue_keep_mask = (all_segs_rzs < -0.1) & (all_segs_grs < 0.5)

        #identifying segs that are redder than ref deblend seg by 0.1 or more in either direction. 0.1 is how much we are being lenient by
        #and has no nan and inf values 
        likely_other_seg_mask =  ( (all_segs_grs > (col_lenient + ref_deblend_gr)) | (all_segs_rzs > (col_lenient + ref_deblend_rz)) ) & ( ~np.isnan(all_segs_grs) ) & (~np.isnan(all_segs_rzs)) & ( ~np.isinf(all_segs_grs) ) & (~np.isinf(all_segs_rzs) )

        # exclude from this mask the sources with r-z < -0.1 and g-r < 0.5. These are objects that we find are associated 
        #with very blue HII regions in galaxies and we want to keep them and not mask them!!
        likely_other_seg_mask &= ~very_blue_keep_mask

        new_deblend_ids_likely_other_segs = new_deblend_ids[likely_other_seg_mask]        
        #this is an array of new deblend ids of the likely other deblend segments. This maps on to segm_deblend_v3 it seems
             
        #what is the probabiltiy that these other segments agree with the color-distribution??
        #score_samples returns the log likelihood and so we do np.exp to get the normal probabilities

        if len(all_segs_grs[likely_other_seg_mask]) == 0:
            #that is there are no components that are candidate other segments
            #no need to put additional masking
            pass

        else:
            ## what do we do if something has a negative flux?? We ignore it?
            all_likely_other_segs_gr = all_segs_grs[likely_other_seg_mask]
            all_likely_other_segs_rz = all_segs_rzs[likely_other_seg_mask]
                        
            col_positions = np.vstack([ all_likely_other_segs_gr , all_likely_other_segs_rz ])
    
            Z_likely_segs = np.exp(gmm.score_samples(col_positions.T))

            #make a plot of the final segments that are included in the calculation, the ones that are masked and of the sources that will be removed!
            for i,indi in enumerate(new_deblend_ids_likely_other_segs):
                ## we loop over each likely other segment and decide whether it is part of our galaxy or not 
                
                if Z_likely_segs[i] >= conf_levels["98.7"]:
                    #yay include!
                    #do nothing as by default every pixel is included, we just have to mask pixels that we will not be counting
                    pass
                else:
                    #we reject this segment as part of our galaxy! 
                    aperture_mask[ segm_deblend_v3 == indi] = 0
                    aperture_mask_no_bkg[segm_deblend_v3 == indi] = 0


        ## MAKING
        ax = make_subplots(ncol = 5, nrow = 2, row_spacing = 0.5,col_spacing=0.4, label_font_size = 17,plot_size = 3,direction = "horizontal")

        ##we have so far dealt with deblended segments being part of main galaxy segment. What about individual sources?

        #the counter for total flux we will be subtracting at the end!
        tot_subtract_sources_g = 0
        tot_subtract_sources_r = 0
        tot_subtract_sources_z = 0

        subtract_source_pos = {"x":[] , "y":[], "marker":[]  }

        #we will be saving the indices of the sources that are being removed and saving them in a separate catalog!
        source_cat_nostars_inseg_inds = []

        source_cat_nostars_inseg = source_cat_inseg_signi[~is_star_inseg_signi]

        source_cat_nostars_inseg_COPY = source_cat_nostars_inseg.copy()
        #we include only the ==0 objects here because those are the ones being removed below
        #there sometimes be floating point errors resulting in non-zero separations, but they will persist through the catalog 
        source_cat_nostars_inseg_own = source_cat_nostars_inseg_COPY[ source_cat_nostars_inseg_COPY["separations"] == 0]
            
        if len(source_cat_inseg_signi) == 0:
            #nothing to do here!! That is there are no significant sources that we have to deal with in the color-color space
            pass
        else:
            #remove the very object that we are targeting from this!
            source_cat_nostars_inseg = source_cat_nostars_inseg[ source_cat_nostars_inseg["separations"] > 0]

            ##LOOKING AT NON-STELALR SOURCES
            if len(source_cat_nostars_inseg) > 0:

                gr_err_nostars_inseg = source_cat_nostars_inseg["g-r_err"].data
                gr_err_nostars_inseg[ np.isnan(gr_err_nostars_inseg) ] = 0

                rz_err_nostars_inseg = source_cat_nostars_inseg["r-z_err"].data
                rz_err_nostars_inseg[ np.isnan(rz_err_nostars_inseg) ]= 0
                
                ##computing the color-color probabilites for reference
                col_posi_sources_pp = np.vstack([ source_cat_nostars_inseg["g-r"].data + gr_err_nostars_inseg , source_cat_nostars_inseg["r-z"].data + rz_err_nostars_inseg  ])
                col_posi_sources_pm = np.vstack([ source_cat_nostars_inseg["g-r"].data + gr_err_nostars_inseg, source_cat_nostars_inseg["r-z"].data - rz_err_nostars_inseg])
                col_posi_sources_mp = np.vstack([ source_cat_nostars_inseg["g-r"].data - gr_err_nostars_inseg , source_cat_nostars_inseg["r-z"].data + rz_err_nostars_inseg])
                col_posi_sources_mm = np.vstack([ source_cat_nostars_inseg["g-r"].data - gr_err_nostars_inseg , source_cat_nostars_inseg["r-z"].data - rz_err_nostars_inseg])

                Z_likely_sources_pp = np.exp(gmm.score_samples(col_posi_sources_pp.T))
                Z_likely_sources_pm = np.exp(gmm.score_samples(col_posi_sources_pm.T))
                Z_likely_sources_mp = np.exp(gmm.score_samples(col_posi_sources_mp.T))
                Z_likely_sources_mm = np.exp(gmm.score_samples(col_posi_sources_mm.T))
 
                Z_likely_sources_max = np.max( (Z_likely_sources_pp,Z_likely_sources_pm,Z_likely_sources_mp,Z_likely_sources_mm), axis=0)
    
                ##is the source bluer than than reference source segment? If so, we will accept it!!
                is_in_blue_box = ( source_cat_nostars_inseg["g-r"].data - gr_err_nostars_inseg  < ref_deblend_gr + col_lenient ) & (source_cat_nostars_inseg["r-z"].data - rz_err_nostars_inseg < ref_deblend_rz + col_lenient)

                #we will add the similar r-z<-0.1, g-r<0.5 color cut here for the HII regions!
                very_blue_keep_mask_sources = (source_cat_nostars_inseg["r-z"].data- rz_err_nostars_inseg < -0.1) & (source_cat_nostars_inseg["g-r"].data - gr_err_nostars_inseg  < 0.5)

                is_in_blue_box |= very_blue_keep_mask_sources

                for w in range(len(source_cat_nostars_inseg) ):
             
                    if is_in_blue_box[w]:
                        pass
                    else:
                       
                        if Z_likely_sources_max[w] >= conf_levels["98.7"]: 
                            pass
                        else:
                            # print(w, source_cat_nostars_inseg["g-r"].data[w] - gr_err_nostars_inseg[w], source_cat_nostars_inseg["r-z"].data[w] - rz_err_nostars_inseg[w] ,  Z_likely_sources_max[w],conf_levels["98.7"]   ) 

                            ##the sources we want to subtract, we want to make sure they are not already being masked by the bright star masks?
                            ##we can identify this based on whether the center of the source lies on top of the masked pixel region or not

                            star_mask_val_w = star_mask[int(source_cat_nostars_inseg["ypix"][w]), int(source_cat_nostars_inseg["xpix"][w]) ]

                            if star_mask_val_w == True:
                                #do nothing, that is, do not subtract as it is already being subtracted by virtue of the stellar mass
                                pass
                            else:    
                                aperture_mask[ segm_deblend_v3 == source_cat_nostars_inseg["new_deblend_id"][w] ] = 1
                                aperture_mask_no_bkg[ segm_deblend_v3 == source_cat_nostars_inseg["new_deblend_id"][w] ] = 1
                                
                                ##adding flux of that source to the 
                                
                                tot_subtract_sources_g += 10**( (22.5  -  source_cat_nostars_inseg["mag_g"][w] )/2.5 )
                                tot_subtract_sources_r += 10**( (22.5  -  source_cat_nostars_inseg["mag_r"][w] )/2.5 )
                                tot_subtract_sources_z += 10**( (22.5  -  source_cat_nostars_inseg["mag_z"][w] )/2.5 )
                                
    
                                 #plot this source for reference on the mask plot!!
                                ax[4].scatter( [source_cat_nostars_inseg["xpix"][w]] , [source_cat_nostars_inseg["ypix"][w] ],  color = "k", marker = source_cat_nostars_inseg["marker"][w],s= 20,  zorder = 1)
    
                                subtract_source_pos["x"].append(source_cat_nostars_inseg["xpix"][w])
                                subtract_source_pos["y"].append(source_cat_nostars_inseg["ypix"][w])
                                subtract_source_pos["marker"].append(source_cat_nostars_inseg["marker"][w])
    
                                #saving the row index in the catalog corresponding to subtracted source
                                source_cat_nostars_inseg_inds.append(w) 
                            
                            
            #########
            ## DEALING WITH STARS :) 
            #########

            #easy! In one step we just mask all the stars, specifically half of the radius-magnitude relation made by Rongpu
            aperture_mask[star_mask] = 0
            aperture_mask_no_bkg[star_mask] = 0            


        # if os.path.exists(save_path + "/tractor_source_model.npy"):
        #     tractor_model = np.load(save_path + "/tractor_source_model.npy")
        # else:
        tractor_model = np.zeros_like(data_arr)
        
        #we will mask all the nans in the data
        aperture_mask[ np.isnan(tot_data) ] = 0
        aperture_mask_no_bkg[ np.isnan(tot_data) ] = 0

        
        aperture_mask[ ~good_pixel_mask ] = 0
        aperture_mask_no_bkg[ ~good_pixel_mask ] = 0
        
        #what about pixels that anomalous negative values?
        #we will mask pixels that are 5sigma lower than the background!
        aperture_mask[ data["g"] < -5*noise_dict["g"] ] = 0
        aperture_mask[ data["r"] < -5*noise_dict["r"] ] = 0
        aperture_mask[ data["z"] < -5*noise_dict["z"] ] = 0
        
        #for plotting let us make everything outside the aperture nans
        aperture_mask_plot = np.copy(aperture_mask)
        tot_data_plot = np.copy(tot_data)
            
        ##instead of plotting just the aperture mask, let us plot the aperture mask applied to the log log data plot!
        tot_data_plot[~aperture_mask_plot.astype(bool)] = np.nan

        ##this is the magnitude at a fixed aperture
        phot_table_g = aperture_photometry(data["g"] , aperture_for_phot, mask = ~aperture_mask.astype(bool))
        phot_table_r = aperture_photometry(data["r"] , aperture_for_phot, mask = ~aperture_mask.astype(bool))
        phot_table_z = aperture_photometry(data["z"] , aperture_for_phot, mask = ~aperture_mask.astype(bool))

        fidu_aper_mag_g = flux_to_mag( phot_table_g["aperture_sum"].data[0] - tot_subtract_sources_g )
        fidu_aper_mag_r = flux_to_mag( phot_table_r["aperture_sum"].data[0] - tot_subtract_sources_r )
        fidu_aper_mag_z = flux_to_mag( phot_table_z["aperture_sum"].data[0] - tot_subtract_sources_z )

        #these mags are extinction corrected!
        org_mags = [source_mag_g_mwc,source_mag_r_mwc,source_mag_z_mwc] 
        
        np.save(save_path + "/segment_map_v2.npy",  segment_map_v2)
        np.save(save_path + "/star_mask.npy",  star_mask)
        np.save(save_path + "/aperture_mask.npy",  aperture_mask)
        #these mags are not extinction corrected
        np.save(save_path + "/tractor_mags.npy", np.array([source_mag_g, source_mag_r, source_mag_z]  ))

        np.save(save_path + "/fiber_pix_pos.npy", np.array([fiber_xpix, fiber_ypix]) )
        np.save(save_path + "/fiber_pix_pos_org.npy", np.array([fiber_xpix_org, fiber_ypix_org]) )
        
        np.save(save_path + "/tot_noise_rms.npy", np.array([tot_rms]) )

        np.save(save_path + "/noise_per_band_rms.npy", np.array([ noise_dict["g"], noise_dict["r"],  noise_dict["z"] ]) )

        with open(save_path + '/subtract_source_pos.pkl', 'wb') as f:
            pickle.dump(subtract_source_pos, f)

        np.save(save_path + "/source_cat_obs_transmission.npy", np.array([  source_cat_obs["mw_transmission_g"] , source_cat_obs["mw_transmission_r"], source_cat_obs["mw_transmission_z"]  ]) )

        #saving the blended source catalog that needs to be subtracted
        source_cat_nostars_inseg_inds = np.array(source_cat_nostars_inseg_inds)

        #print(f"SOURCE_CAT 7: {len(source_cat_nostars_inseg_inds)}")

        source_cat_nostars_inseg_remove = source_cat_nostars_inseg[source_cat_nostars_inseg_inds]

        #print(f"SOURCE_CAT 8: {len(source_cat_nostars_inseg_remove)}")

        source_cat_nostars_inseg_remove.write( save_path + "/blended_source_remove_cat.fits", overwrite=True)
        #this catalog will be used to get the tractor model of the blended sources

        ##save the catalog of sources that are considered to be part of the galaxy!
        #-> save the source catalog for objects that lie on the main segment are not being removed by the color cuts!
        # Build boolean mask: True means "keep"
        
        galaxy_sources_mask = np.ones(len(source_cat_nostars_inseg), dtype=bool)
        
        if len(source_cat_nostars_inseg_inds) > 0:
            galaxy_sources_mask[source_cat_nostars_inseg_inds] = False
            
        source_cat_galaxy_objs = source_cat_nostars_inseg[galaxy_sources_mask]
        source_cat_galaxy_objs = vstack([source_cat_galaxy_objs, source_cat_nostars_inseg_own])

        # if len(source_cat_nostars_inseg_own) != 1:
        #     print(f"Number of targeted sources not equal to 1! : {source_tgid}, {len(source_cat_nostars_inseg_own)}, {np.min(source_cat_galaxy_objs['separations'])}")

        #save this catalog!
        #before saving this let us add a column of unique object ids to identify them 
        source_cat_galaxy_objs["source_objid_new"] = np.arange(len(source_cat_galaxy_objs))
        source_cat_galaxy_objs.write( save_path + "/parent_galaxy_sources.fits", overwrite=True)
        
        ##NOTE!! we will have to update this again based on the second deblending step we do in the last step!
        ##These catalogs contain the pixel positions so we can use that if need be!
        #^the above catalog contains sources that satisfy the color cuts! The simple photo sources are saved as part of the below function
        
        #we run the simplest photometry here!!
        if run_simple_photo:
            simplest_photo_mags, simplest_photo_island_dist_pix, simplest_photo_aper_frac_in_image  = get_simplest_photometry(data_arr,  noise_dict["r"], fiber_xpix, fiber_ypix, source_cat_f[~is_star], save_path,source_zred=None)
        else:
            simplest_photo_mags = 3*[np.nan]
            simplest_photo_island_dist_pix = 0
            simplest_photo_aper_frac_in_image = 0

        ##########################################
        ### PLOTTING CODE
        ##########################################

        ##GRZ IMAGE
        ax_id = 5
        ax[ax_id].set_title("grz image w/aperture",fontsize = 13)
        ax[ax_id].imshow(rgb_stuff,origin="lower",zorder = 0)
        #th box size is image_sizeximage_size
        ax[ax_id].text(0.5 , 0.95 , "(%.3f,%.3f, z=%.3f)"%(source_ra,source_dec, source_redshift) ,color = "yellow",fontsize = 10,
                      ha="center", va= "center",transform = ax[ax_id].transAxes )
        

        #get pixel co-ordinates of the source galaxy
        circle = patches.Circle( (fiber_xpix_org, fiber_ypix_org),3, color='orange', fill=False, linewidth=1,ls ="-")
        #these two circles will overlap in majority of cases but in the rare case where the DESI fiber was not on top of a island segment, we will get different circles
        circle_2 = patches.Circle( (fiber_xpix, fiber_ypix),3, color='orange', fill=False, linewidth=1,ls ="--")
        
        ax[ax_id].add_patch(circle)
        ax[ax_id].add_patch(circle_2)
        
        ax[ax_id].set_xlim([0,box_size])
        ax[ax_id].set_ylim([0,box_size])
        ax[ax_id].set_xticks([])
        ax[ax_id].set_yticks([])
        if bstar_ra != 99:
            aperture_for_bstar_1.plot(ax = ax[ax_id], color = "r", lw = 1.5, ls = "dotted")
            aperture_for_bstar_34.plot(ax = ax[ax_id], color = "r", lw = 1.5, ls = "dotted")
            aperture_for_bstar_12.plot(ax = ax[ax_id], color = "r", lw = 1.5, ls = "dotted")

        ##G+R+Z DATA LOG IMAGE
        ax_id = 6
        ax[ax_id].set_title("g+r+z data (log scaling)",fontsize = 13)
        # Create a norm object and inspect the vmin/vmax
        norm_obj = LogNorm()
        ax[ax_id].imshow(tot_data,origin="lower",norm=norm_obj,cmap = "viridis",zorder = 0)
        tot_data_vmin = norm_obj.vmin
        tot_data_vmax = norm_obj.vmax
        ax[ax_id].set_xlim([0,box_size])
        ax[ax_id].set_ylim([0,box_size])
        ax[ax_id].set_xticks([])
        ax[ax_id].set_yticks([])
        if bstar_ra != 99:
            aperture_for_bstar_1.plot(ax = ax[ax_id], color = "r", lw = 1.5, ls = "dotted")
            aperture_for_bstar_34.plot(ax = ax[ax_id], color = "r", lw = 1.5, ls = "dotted")
            aperture_for_bstar_12.plot(ax = ax[ax_id], color = "r", lw = 1.5, ls = "dotted")


        ##G+R+Z SEGMENTED IMAGE
        ax_id = 7
        ##the color in these plots is from the tab20 color map
        cmap = cm.tab20
        # Normalize values to the range [0,1]
        norm = mcolors.Normalize(vmin=min(new_deblend_ids), vmax=max(new_deblend_ids))
        # Map values to colors
        colors = cmap(norm(np.array(new_deblend_ids)))
        
        ax[ax_id].set_title("g+r+z data segmentation",fontsize = 13)
        ax[ax_id].imshow(segment_map, origin='lower', cmap=segment_map.cmap,
                   interpolation='nearest',zorder = 0)
        # aperture_for_phot.plot(ax = ax[7], color = "r", lw = 2.5, ls = "-")
        # aperture_for_phot_noscale.plot(ax = ax[7], color = "r", lw = 1, ls = "dotted")
        ax[ax_id].scatter( sources_f_xpix,sources_f_ypix, s=5,color = "white",marker="^") 
        
        #print(f"SOURCE_CAT 9: {len( sources_f_xpix)}")

        
        #sources that are stars!
        ax[ax_id].scatter( sources_f_xpix[is_star],sources_f_ypix[is_star], s=50,color = "white",marker="*",zorder = 3) 
        ax[ax_id].set_xlim([0,box_size])
        ax[ax_id].set_ylim([0,box_size])
        ax[ax_id].set_xticks([])
        ax[ax_id].set_yticks([])
        if bstar_ra != 99:
            aperture_for_bstar_1.plot(ax = ax[7], color = "r", lw = 1.5, ls = "dotted")
            aperture_for_bstar_34.plot(ax = ax[7], color = "r", lw = 1.5, ls = "dotted")
            aperture_for_bstar_12.plot(ax = ax[7], color = "r", lw = 1.5, ls = "dotted")


        ##G+R+Z MAIN SEGMENT
        ax_id = 8
        ax[ax_id].set_title("g+r+z band main segment",fontsize = 13)
        ax[ax_id].scatter( sources_f_xpix[is_star],sources_f_ypix[is_star], s=50,color = "white",marker="*",zorder = 3) 
        ax[ax_id].imshow(segm_deblend_v3, origin='lower', cmap="tab20",
                   interpolation='nearest')
        #plotting the sources in the main segment

        #print(f"SOURCE_CAT 10.5: {len( source_cat_inseg_signi)}")

        
        plot_now = source_cat_inseg_signi[(~is_star_inseg_signi)]

        #print(f"SOURCE_CAT 10: {len( plot_now)}")

        
        for p in range(len(plot_now)):
            ax[ax_id].scatter( [ plot_now["xpix"][p]] , [ plot_now["ypix"][p]], s=10,color = "k",marker=plot_now["marker"][p]) 
        ax[ax_id].set_xlim([0,box_size])
        ax[ax_id].set_ylim([0,box_size])
        ax[ax_id].set_xticks([])
        ax[ax_id].set_yticks([])
        if bstar_ra != 99:
            aperture_for_bstar_1.plot(ax = ax[ax_id], color = "r", lw = 1.5, ls = "dotted")
            aperture_for_bstar_34.plot(ax = ax[ax_id], color = "r", lw = 1.5, ls = "dotted")
            aperture_for_bstar_12.plot(ax = ax[ax_id], color = "r", lw = 1.5, ls = "dotted")
        

        # aperture_for_phot.plot(ax = ax[ax_id], color = "r", lw = 2.5, ls = "-")


        ##ALL DEBLENDED SEGMENTS
        ax_id = 9

        ax[ax_id].set_title("g+r+z segmentation+deblend",fontsize = 13)
        
        # aperture_for_phot.plot(ax = ax[ax_id], color = "r", lw = 2.5, ls = "-")
        # aperture_for_phot_noscale.plot(ax = ax[ax_id], color = "r", lw = 1, ls = "dotted")
        
        ax[ax_id].imshow(segm_deblend, origin='lower', cmap=segm_deblend.cmap,
                   interpolation='nearest',zorder = 0)
           
        ax[ax_id].scatter( sources_f_xpix[is_star],sources_f_ypix[is_star], s=50,color = "white",marker="*",zorder = 3) 
        
        plot_now = source_cat_inseg_signi[(~is_star_inseg_signi)]

        #print(f"SOURCE_CAT 11: {len( plot_now)}")

        
        for p in range(len(plot_now)):
            ax[ax_id].scatter( [ plot_now["xpix"][p]] , [ plot_now["ypix"][p]], s=10,color = "k",marker=plot_now["marker"][p]) 
        
        ax[ax_id].set_xlim([0,box_size])
        ax[ax_id].set_ylim([0,box_size])
        ax[ax_id].set_xticks([])
        ax[ax_id].set_yticks([])
    
        
        ##COLOR-COLOR PLOT
        ax_id = 0
        ##plot the g-r vs. r-z color contours from the redshift bin of this object        
        ax[ax_id].contour(X, Y, Z_gmm, levels=sorted(lvls), cmap="viridis_r",alpha = 1,zorder = 1)

        #plotting all the segment colors with error on the color-color plot
        for i in range(len(new_deblend_ids)):
            ax[ax_id].scatter([all_segs_grs[i]], [all_segs_rzs[i]], s=50, marker="x",lw=2,zorder=3,color = colors[i])
            ax[ax_id].errorbar([all_segs_grs[i]], [all_segs_rzs[i]], zorder=3,ecolor = colors[i],
                                  xerr=[ all_segs_grs_err[i] ], yerr= [ all_segs_rzs_err[i] ], fmt='none',alpha=1, capsize=3)

        ##to avoid some sources being outside the plotting limits, we will clip their values 
        def get_updated_gr_rz(gr_vals, rz_vals):
            gr_vals_up = np.clip(gr_vals, -1, 2)
            rz_vals_up = np.clip(rz_vals, -1, 1.5)
            return gr_vals_up, rz_vals_up

        gr_new, rz_new = get_updated_gr_rz(source_cat_inseg_signi[is_star_inseg_signi]["g-r"].data, source_cat_inseg_signi[is_star_inseg_signi]["r-z"].data)
        ax[ax_id].scatter( gr_new, rz_new, color =  "r", marker = "*",s= 40,zorder = 2 ) 

        plot_now = source_cat_inseg_signi[(~is_star_inseg_signi) ]

        #print(f"SOURCE_CAT 12: {len( plot_now)}")

        
        gr_new, rz_new = get_updated_gr_rz(plot_now["g-r"].data, plot_now["r-z"].data)
        for p in range(len(plot_now)):
            ax[ax_id].scatter( [ gr_new[p]], [rz_new[p]], color = "k" , marker = plot_now["marker"][p], s= 20,zorder = 2 ) 
            
        gr_new, rz_new = get_updated_gr_rz(source_cat_inseg_signi["g-r"].data, source_cat_inseg_signi["r-z"].data)
        ## plot the color-color errorbars
        ax[ax_id].errorbar(gr_new, rz_new, 
            xerr=source_cat_inseg_signi["g-r_err"], yerr=source_cat_inseg_signi["r-z_err"], fmt='none', ecolor='k', alpha=0.3, capsize=3,zorder = 2)
  
        ##plot the bluer boundary lines
        ax[ax_id].hlines(y = all_segs_rzs[new_deblend_island_num - 1] + col_lenient,xmin = -0.5, xmax= all_segs_grs[new_deblend_island_num - 1] + col_lenient,color = "k",lw = 1, ls = "-")
        ax[ax_id].vlines(x = all_segs_grs[new_deblend_island_num - 1] + col_lenient,ymin = -0.5, ymax= all_segs_rzs[new_deblend_island_num - 1] + col_lenient,color = "k",lw = 1, ls = "-")

        ax[ax_id].hlines(y = ref_deblend_rz+col_lenient,xmin = -0.5, xmax= ref_deblend_gr + col_lenient,color = "k",lw = 0.75, ls = "--")
        ax[ax_id].vlines(x = ref_deblend_gr + col_lenient,ymin = -0.5, ymax= ref_deblend_rz + col_lenient,color = "k",lw = 0.75, ls = "--")

        

        ax[ax_id].set_xlim([ -0.5, 2])
        ax[ax_id].set_ylim([  -0.5, 1.5])
        ax[ax_id].tick_params(axis='both', labelsize=10)
        ax[ax_id].set_xlabel("g-r",fontsize = 17)
        ax[ax_id].set_ylabel("r-z",fontsize = 17)


        ##TRACTOR IMG AND TRACTOR IMG-S
        ax_id = 1
        #getting difference between source of interest model and entire image        
        resis = data_arr - tractor_model 
        rgb_resis = sdss_rgb(resis, ["g","r","z"], scales=dict(g=(2,6.0), r=(1,3.4), z=(0,2.2)), m=0.03)
        #we crop to the center 64 as that is our CNN input
        size = 64
        start = (image_size - size) // 2
        end = start + size

        ax[ax_id].set_title(r"IMG")
        ax[ax_id].imshow(rgb_stuff[start:end, start:end,:])
        ax[ax_id].set_xticks([])
        ax[ax_id].set_yticks([])
        ax[ax_id].set_xlim([0,63])
        ax[ax_id].set_ylim([0,63])
        
        ax_id = 2
        ax[ax_id].text(0.25,0.96,f"pCNN = {pcnn_val:.2f}",size = 12,transform=ax[ax_id].transAxes, verticalalignment='top',color = "white")
        ax[ax_id].set_title(r"IMG - S")
        ax[ax_id].imshow(rgb_resis[start:end, start:end,:])
        ax[ax_id].set_xticks([])
        ax[ax_id].set_yticks([])


        ## APERTURE MASK PLOT
        ax_id = 4

        ax[ax_id].set_title("g+r+z band aperture mask",fontsize = 13)
        ax[ax_id].set_xticks([])
        ax[ax_id].set_yticks([])
        ax[ax_id].set_xlim([0,box_size])
        ax[ax_id].set_ylim([0,box_size])

        ##getting the aperture mask      
        ax[ax_id].imshow(tot_data_plot,origin="lower",norm=LogNorm(vmin=tot_data_vmin, vmax = tot_data_vmax),cmap = "viridis",zorder = 0)

        ax[ax_id].set_xlim([0,box_size])
        ax[ax_id].set_ylim([0,box_size])

        aperture_for_phot.plot(ax = ax[ax_id], color = "r", lw = 2.5, ls = "-")
        
        if bstar_ra != 99:
            aperture_for_bstar_1.plot(ax = ax[ax_id], color = "r", lw = 1.5, ls = "dotted")
            aperture_for_bstar_34.plot(ax = ax[ax_id], color = "r", lw = 1.5, ls = "dotted")
            aperture_for_bstar_12.plot(ax = ax[ax_id], color = "r", lw = 1.5, ls = "dotted")
            

        
        ### SUMMARY PLOT WITH INITIAL MAGNITUDE
        ax_id = 3
        ax[ax_id].set_title(f"summary, {source_tgid}",fontsize = 11)

        ax[ax_id].set_xlim([0,1])
        ax[ax_id].set_ylim([0,1])
        ax[ax_id].set_xticks([])
        ax[ax_id].set_yticks([])

        spacing = 0.08
        start = 0.97
        fsize = 11
        ax[ax_id].text(0.05,start,"Tractor-mag g = %.2f"%source_mag_g,size =fsize,transform=ax[ax_id].transAxes, verticalalignment='top')

        
        ax[ax_id].text(0.05,start - spacing*1,f"Aper-mag-R3 g = {fmt_mag(fidu_aper_mag_g)}",size = fsize,transform=ax[ax_id].transAxes, verticalalignment='top')
        
        ax[ax_id].text(0.05,start - spacing*2,"fracflux_g = %.2f"%(source_cat_obs["fracflux_g"]),size = 12,transform=ax[ax_id].transAxes, verticalalignment='top')
        
        ax[ax_id].text(0.05,start - spacing*3,"Tractor-mag r = %.2f"%source_mag_r,size =fsize,transform=ax[ax_id].transAxes, verticalalignment='top')
        ax[ax_id].text(0.05,start - spacing*4,f"Aper-mag-R3 r = {fmt_mag(fidu_aper_mag_r)}",size = fsize,transform=ax[ax_id].transAxes, verticalalignment='top')
        ax[ax_id].text(0.05,start - spacing*5,"fracflux_r= %.2f"%(source_cat_obs["fracflux_r"]),size = fsize,transform=ax[ax_id].transAxes, verticalalignment='top')

        ax[ax_id].text(0.05,start - spacing*6,"Tractor-mag z = %.2f"%source_mag_z,size =fsize,transform=ax[ax_id].transAxes, verticalalignment='top')
        ax[ax_id].text(0.05,start - spacing*7,f"Aper-mag-R3 z = {fmt_mag(fidu_aper_mag_z)}",size = fsize,transform=ax[ax_id].transAxes, verticalalignment='top')
        ax[ax_id].text(0.05,start - spacing*8,"fracflux_z = %.2f"%(source_cat_obs["fracflux_z"]),size = fsize,transform=ax[ax_id].transAxes, verticalalignment='top')

        ax[ax_id].text(0.05,start - spacing*9,"Closest Star fdist = %.2f"%(closest_star_norm_dist),size = fsize,transform=ax[ax_id].transAxes, verticalalignment='top')
        ax[ax_id].text(0.05,start - spacing*10,"Bright star fdist = %.2f"%(bstar_fdist),size = fsize,transform=ax[ax_id].transAxes, verticalalignment='top')
        ax[ax_id].text(0.05,start - spacing*11,"SGA Dist (deg), NDist = %.2f, %.2f"%(sga_dist, sga_ndist),size = fsize,transform=ax[ax_id].transAxes, verticalalignment='top')

        ##saving this image :P
        plt.savefig( save_summary_png ,bbox_inches="tight")
        plt.close()
    
        fidu_aper_mag_g = fidu_aper_mag_g + 2.5 * np.log10(source_cat_obs["mw_transmission_g"])
        fidu_aper_mag_r = fidu_aper_mag_r + 2.5 * np.log10(source_cat_obs["mw_transmission_r"])
        fidu_aper_mag_z = fidu_aper_mag_z + 2.5 * np.log10(source_cat_obs["mw_transmission_z"])

        fidu_aper_mags = [fidu_aper_mag_g, fidu_aper_mag_r, fidu_aper_mag_z]
        
        ##save these as a file for future reference
        np.save(save_path + "/aper_r3_mags.npy", fidu_aper_mags)        
        np.save(save_path + "/org_mags.npy", org_mags)


        return {
                "closest_star_dist": closest_star_dist,
                "closest_star_mag": closest_star_mag,
                "aper_r3_mags": fidu_aper_mags,
                "save_path": save_path,
                "save_summary_png": save_summary_png,
                "tractor_dr9_mags": org_mags,
                "closest_star_norm_dist": closest_star_norm_dist,
                "lie_on_segment_island": lie_on_segment_island,
                "first_min_dist_island_pix" : min_dist_pix,
                "aper_frac_mask_badpix": aper_frac_mask_badpix, 
                "img_frac_mask_badpix":  img_frac_mask_badpix,
                "simple_photo_mags": simplest_photo_mags, 
                "simple_photo_island_dist_pix": simplest_photo_island_dist_pix,
                "simplest_photo_aper_frac_in_image": simplest_photo_aper_frac_in_image
            }
        




    

        
        
        
        


        

        

    



    

                                    
   
    

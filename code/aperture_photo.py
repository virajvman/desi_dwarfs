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

rootdir = '/global/u1/v/virajvm/'
sys.path.append(os.path.join(rootdir, 'DESI2_LOWZ'))
from desi_lowz_funcs import print_stage, check_path_existence, get_remove_flag, _n_or_more_lt, is_target_in_south, match_c_to_catalog, calc_normalized_dist, get_sweep_filename, get_random_markers, save_table, make_subplots, sdss_rgb

def conf_interval(x, pdf, conf_level):
    return np.sum(pdf[pdf > x])-conf_level

def mask_bad_flux(flux_vals):
    good_flux_vals = np.where(flux_vals > 0, flux_vals, np.nan)
    return good_flux_vals

def mask_radius_for_mag(mag):
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
                    
def run_aperture_pipe(input_dict):
    '''
    Function to redo the aperture via aperture photometry

    Parameters:
    ---------------
    save_path : str, path where the final outputs are being stored
    img_path : str, path of the fits file that stores the img and wcs data
    source_ra : float, RA location of source
    source_dec : float, DEC location of source
    source_redshift : float, DESI redshift of source


    KNOWN ISSUES:
    NEED TO DEBUG THE CASE WHEN ISLAND_NUM = 99999, E.G. ELG_tgid_39627809651429142
    /pscratch/sd/v/virajvm/redo_photometry_plots/all_good/south/sweep-040p000-050p005/0456p010/ELG_tgid_39627809651429142/

    
    '''

    ## DONE: I THINK SELECTION CUT IS THAT MORE THAN 2 ABOVE 0.2
    ## TO DO: WHY IS THE GAIA STAR RADIUS SLIGHTLY DIFFERENT THAN THE RONGPU FORMULA RADIUS?

    save_path = input_dict["save_path"]
    img_path  = input_dict["img_path"]
    source_tgid  = input_dict["tgid"]
    source_ra  = input_dict["ra"] 
    source_dec  = input_dict["dec"]
    source_redshift  = input_dict["redshift"]
    wcs  = input_dict["wcs"]
    data_arr  = input_dict["image_data"]
    source_cat_f = input_dict["source_cat"]
    org_mag_g = input_dict["org_mag_g"]
    overwrite = input_dict["overwrite"]
    pcnn_val = input_dict["pcnn_val"]
    npixels_min = input_dict["npixels_min"]
    threshold_rms_scale = input_dict["threshold_rms_scale"]
        
    verbose=False

    if verbose:
        print(source_ra, source_dec)
    
    use_photoz = False
    
    bstar_tuple = input_dict["bright_star_info"]

    sga_tuple = input_dict["sga_info"]

    sga_dist, sga_ndist = sga_tuple[0], sga_tuple[1]
    
    #this is the bright star info that was used to compute the STARDIST, STARFDIST etc.
    #the radius is given in arcsecs and will be plotted as a circle for reference
    bstar_ra, bstar_dec, bstar_radius, bstar_fdist = bstar_tuple[0], bstar_tuple[1], bstar_tuple[2], bstar_tuple[3]

    ##filter the source cat for objects with nan values
    ##we do not do this for the scarlet pipeline, and hence we do it separately over here

    #remove if there are any nans in the data!
    source_cat_f = source_cat_f[ ~np.isnan(source_cat_f["g-r_err"]) &  ~np.isnan(source_cat_f["r-z_err"]) &  ~np.isinf(source_cat_f["g-r_err"]) &  ~np.isinf(source_cat_f["r-z_err"]) ]
    source_cat_f = source_cat_f[ ~np.isnan(source_cat_f["g-r"]) &  ~np.isnan(source_cat_f["r-z"]) &  ~np.isinf(source_cat_f["g-r"]) &  ~np.isinf(source_cat_f["r-z"]) ]

    gmm_file_zgrid = np.arange(0.001, 0.525,0.025)

    #bool variable that decides if the rephoto pipeline should be run
    do_i_run = False

    ##################
    # get catalog of nearby DR9 sources along with their photo-zs info. Nearby is defined as within 45 arcsecs
    ##################

    ref_coord = SkyCoord(ra=source_ra * u.deg, dec=source_dec * u.deg)
    sources_coords = SkyCoord(ra=source_cat_f["ra"].data * u.deg, dec=source_cat_f["dec"].data * u.deg)
    # Compute separations
    source_seps = ref_coord.separation(sources_coords).arcsec

    ##procedure for selecting a star
    signi_pm = ( np.abs(source_cat_f["pmra"]) * np.sqrt(source_cat_f["pmra_ivar"]) > 3) | ( np.abs(source_cat_f["pmdec"]) * np.sqrt(source_cat_f["pmdec_ivar"]) > 3)
    
    is_star = (source_cat_f["ref_cat"] == "G2")  &  ( 
        (source_cat_f['type'] == "PSF")  | signi_pm 
    ) 

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
            new_mags = np.load(save_path + "/aper_r35_mags.npy")
            org_mags = np.load(save_path + "/org_mags.npy")

            return {
                    "closest_star_dist": closest_star_dist,
                    "closest_star_mag": closest_star_mag,
                    "fidu_aper_mags": new_mags,
                    "org_mags": org_mags,
                    "save_path": save_path,
                    "save_summary_png": save_summary_png,
                    "img_path": img_path,
                }
            # return closest_star_dist, closest_star_mag, new_mags, org_mags, save_path, save_summary_png, img_path
            
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
        # clevs = [0.16,0.50,0.841,0.977,0.998,0.99994]
        clevs = [0.38,0.68,0.86,0.954,0.987] #,0.997]
        for cld in clevs:  
            sig = opt.brentq( conf_interval, 0., 1., args=(Z_gmm,cld) )   
            lvls.append(sig)
            
        conf_levels = { "38":lvls[0]*Znorm,"68":lvls[1]*Znorm,"86":lvls[2]*Znorm,"95.4":lvls[3]*Znorm,"98.7":lvls[4]*Znorm} #,"99.7":lvls[5]*Znorm}
    
        #let us save these levels for future reference!
        box_size = 350
    
        data = { "g": data_arr[0], "r": data_arr[1], "z": data_arr[2]}

        #make the rgb image
        rgb_stuff = sdss_rgb([data["g"],data["r"],data["z"]], ["g","r","z"], scales=dict(g=(2,6.0), r=(1,3.4), z=(0,2.2)), m=0.03)
    
        #get the pixel locations of these sources 
        sources_f_xpix,sources_f_ypix,_ = wcs.all_world2pix(source_cat_f['ra'].data, source_cat_f['dec'].data, 0,1)

        source_cat_f["xpix"] = sources_f_xpix
        source_cat_f["ypix"] = sources_f_ypix

        #first estimate the background error to use in aperture photometry
        noise_dict = {}

        ##SOMETHING TO CHECK IS IF THE FITS FILE I AM READING HAVE ANY INFORMATION ON THE BACKGROUND RMS LEVEL
        rms_estimator = MADStdBackgroundRMS()
        ##estimate the background rms in each band!
        for bii in ["g","r","z"]:        
            # Apply sigma clipping
            sigma_clip = SigmaClip(sigma=3.0,maxiters=5)
            clipped_data = sigma_clip(data[bii])
            # Estimate RMS
            background_rms = rms_estimator(clipped_data)
            noise_dict[bii] = background_rms

        #the rms in the total image will be the rms in the 3 bands added in quadrature
        tot_rms = np.sqrt( noise_dict["g"]**2 + noise_dict["r"]**2 + noise_dict["z"]**2 )
        
        #parameters to think a bit carefully about
        # npixels_min = 20
        threshold = threshold_rms_scale * tot_rms

        kernel = make_2dgaussian_kernel(3.0, size=5)  # FWHM = 3.0
        
        tot_data = np.sum(data_arr, axis=0)

        if np.shape(tot_data) != (350,350):
            raise ValueError("Dimensions of summed grz image are not (350,350)")
        
        convolved_tot_data = convolve( tot_data, kernel )

        # segment_map = detect_sources(tot_data, threshold, npixels=npixels_min) 
        segment_map = detect_sources(convolved_tot_data, threshold, npixels=npixels_min) 
        
    
        ##do deblending per band image
        segm_deblend = deblend_sources(convolved_tot_data, segment_map,
                                   npixels=npixels_min,nlevels=16, contrast=0.01,
                                   progress_bar=False)
        
        ##choose the aperture radius based on the size of the main segment  
    
        #get the segment number where the main source of interest lies in
        #we maintain 2 copies because one of them will be changed in case the desi fiber does not lie on a island segment
        fiber_xpix, fiber_ypix,_ = wcs.all_world2pix(source_ra, source_dec,0,1)
        fiber_xpix_org, fiber_ypix_org,_ = wcs.all_world2pix(source_ra, source_dec,0,1)
        
        island_num = segment_map.data[int(fiber_ypix),int(fiber_xpix)]
        ## note that these 2d arrays are [y-coord, x-coord]
                
        #any segment not part of this main segment is considered to be a different source 
        #make a copy of the segment array
        segment_map_v2 = np.copy(segment_map.data)

        #is it possible that the source lies on background and not in one of the segmented islands?
        #this is possible if the source was located in the very faint extremeties which is possible for ELGs
        #In this case, I think just keep the aperture as is and just having fixed apertures is the way to go?
        if island_num == 0:
            print_stage(f"Following source does not lie on a segment island, TGID:{source_tgid}")
            segment_map_v2, island_num, fiber_xpix, fiber_ypix = find_nearest_island(segment_map_v2, fiber_xpix, fiber_ypix)
    
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
            # aperture_mask[ segm_deblend_v3 == source_cat_stars_inseg["new_deblend_id"][j] ] = 0  
            #returning an updating aperture mask where the pixels within 0.5 of star radius are masked!
            star_mask = mask_circle(star_mask, bstar_xpix,bstar_ypix, 0.5 * bstar_radius_pix, value = 1)
            #this ensures that we are just masking the stellar region and not the whole deblended segment!
        else:
            #if no bright star exists, then we do not do this!
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
                star_radius_i_pix = 0.5*star_radius_i_as/0.262
                #mask it now!
                star_mask = mask_circle(star_mask, source_cat_all_stars["xpix"][j], source_cat_all_stars["ypix"][j], star_radius_i_pix, value = 1 )                 

        star_mask = star_mask.astype(bool)
        ## the star mask is ready!!!

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
        aperture_for_phot = get_elliptical_aperture( segment_map_v2, star_mask, 2, sigma = 3.5 )
        aperture_for_phot_noscale = get_elliptical_aperture( segment_map_v2, star_mask, 2, sigma = 1 )

        
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

            #saving this to the updated dictionary!
            data_for_colors[biii] = data_for_colors_i

        ##when colors are being computed we will mask the stellar pixels
        for din in new_deblend_ids:
            gband_flux_i = np.sum(data_for_colors["g"][segm_deblend_v3 == din ])
            rband_flux_i = np.sum(data_for_colors["r"][segm_deblend_v3 == din ])
            zband_flux_i = np.sum(data_for_colors["z"][segm_deblend_v3 == din ])

            #as all the error is assumed to be the same so we add them in quadrature
            #TODO: use the actual pixel inv-var map to do this
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
        
        #overplot again the DR9 sources but only if they are significant!
        #however we want to only include if it a signicant flux contributor, like min(grz) < 21 and robust color determination

        #find difference between source and all the other catalog objects
        ref_coord = SkyCoord(ra=source_ra * u.deg, dec=source_dec * u.deg)
        catalog_coords = SkyCoord(ra=source_cat_f["ra"].data * u.deg, dec=source_cat_f["dec"].data * u.deg)
        # Compute separations
        separations = ref_coord.separation(catalog_coords).arcsec

        #storing the separations of all sources from our main fiber source so we can remove it from consideration later
        source_cat_f["separations"] = separations
        
        source_cat_obs = source_cat_f[np.argmin(separations)]
        
        if np.min(separations) != 0:
            raise ValueError("Distance between source and closest object in the photometry catalog in area =",np.min(separations))
        #get the original magnitude of the object

        #this magnitude is not milky way corrected
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

        #identifying segs that are redder than ref deblend seg by 0.1 or more in either direction. 0.1 is how much we are being lenient by
        #and has no nan and inf values 
        likely_other_seg_mask =  ( (all_segs_grs > (col_lenient + ref_deblend_gr)) | (all_segs_rzs > (col_lenient + ref_deblend_rz)) ) & ( ~np.isnan(all_segs_grs) ) & (~np.isnan(all_segs_rzs)) & ( ~np.isinf(all_segs_grs) ) & (~np.isinf(all_segs_rzs) )
        
        new_deblend_ids_likely_other_segs = new_deblend_ids[likely_other_seg_mask]        
        #this is an array of new deblend ids of the likely other deblend segments. This maps on to segm_deblend_v3 it seems
             
        #what is the probabiltiy that these other segments agree with the color-distribution??
        #score_samples returns the log likelihood and so we do np.exp to get the normal probabilities


        ##THIS COLOR COLOR SELECTION CAN BE DONE A BIT BETTER ...

        ##AM I ENFORCING THAT THE TARGETED SOURCE IS NOT BEING REMOVED?
        ## I am imagine a case where if the entire island is the segment (no deblended segment) and if source has weird color then this can happen

        
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
                
                #we will use the criterion that if they are within 4sigma percentile contour of the color distribution, then we accept it
                if Z_likely_segs[i] >= conf_levels["98.7"]:
                    # print("include")
                    #yay include!
                    #do nothing as by default every pixel is included, we just have to mask pixels that we will not be counting
                    pass
                else:
                    # print("reject")
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
            
        if len(source_cat_inseg_signi) == 0:
            #nothing to do here!! That is there are no significant sources that we have to deal with in the color-color space
            pass
        else:
            ##We deal with the other non-stellar sources by looking at their DR9 photo-zs!!    
            
            #remove the very object that we are targeting from this!
            source_cat_nostars_inseg = source_cat_nostars_inseg [ source_cat_nostars_inseg["separations"] > 0]

            ##LOOKING AT NON-STELALR SOURCES
            if len(source_cat_nostars_inseg) > 0:

                #what to do if the errors are nans?
                gr_err_nostars_inseg = source_cat_nostars_inseg["g-r_err"].data
                gr_err_nostars_inseg[ np.isnan(gr_err_nostars_inseg) ] = 0

                rz_err_nostars_inseg = source_cat_nostars_inseg["r-z_err"].data
                rz_err_nostars_inseg[ np.isnan(rz_err_nostars_inseg) ]= 0
                
                ##computing the color-color probabilites for reference
                col_posi_sources_pp = np.vstack([ source_cat_nostars_inseg["g-r"].data + gr_err_nostars_inseg , source_cat_nostars_inseg["r-z"].data + rz_err_nostars_inseg  ])
                col_posi_sources_pm = np.vstack([ source_cat_nostars_inseg["g-r"].data + gr_err_nostars_inseg, source_cat_nostars_inseg["r-z"].data - rz_err_nostars_inseg])
                col_posi_sources_mp = np.vstack([ source_cat_nostars_inseg["g-r"].data - gr_err_nostars_inseg , source_cat_nostars_inseg["r-z"].data + rz_err_nostars_inseg])
                col_posi_sources_mm = np.vstack([ source_cat_nostars_inseg["g-r"].data - gr_err_nostars_inseg , source_cat_nostars_inseg["r-z"].data - rz_err_nostars_inseg])


                ##to do this, I will try all the 4 possible options, ++,+-,-+ etc. and if one hits inside contour, we consider it! 
                Z_likely_sources_pp = np.exp(gmm.score_samples(col_posi_sources_pp.T))
                Z_likely_sources_pm = np.exp(gmm.score_samples(col_posi_sources_pm.T))
                Z_likely_sources_mp = np.exp(gmm.score_samples(col_posi_sources_mp.T))
                Z_likely_sources_mm = np.exp(gmm.score_samples(col_posi_sources_mm.T))
 
                Z_likely_sources_max = np.max( (Z_likely_sources_pp,Z_likely_sources_pm,Z_likely_sources_mp,Z_likely_sources_mm), axis=0)
    
                ##is the source bluer than than reference source segment? If so, we will accept it!!
                is_in_blue_box = ( source_cat_nostars_inseg["g-r"].data - gr_err_nostars_inseg  < ref_deblend_gr + col_lenient ) & (source_cat_nostars_inseg["r-z"].data - rz_err_nostars_inseg < ref_deblend_rz + col_lenient)

                
                for w in range(len(source_cat_nostars_inseg) ):
             
                    if is_in_blue_box[w]:
                        pass
                    else:
                        #if the source is redder, we will investigate its photo-zs
                        #unsure if I should choose 68 or 95 here
                        #also low-redshift photo-zs are often not accurate and so if it within 0.15 within errors we good!
                        #but we also want to look reliable photozs, or ones with, not too large errors
                        #not sure how to be quantitative about this
                        sra = source_cat_nostars_inseg["ra"][w]	
                        sdec = source_cat_nostars_inseg["dec"][w]	
                        
                        ## important note : https://www.legacysurvey.org/dr9/files/#photo-z-sweeps-9-1-photo-z-sweep-brickmin-brickmax-pz-fits
                        ## Although we provide photo-zs for all objects that meet the NOBS cut, 
                        ## the brightest objects have the most reliable photo-zs. 
                        ## As a rule of thumb, objects brighter than 𝑧-band magnitude of 21 are mostly reliable, 


                        ##WE ARE NOT USING PHOTO-ZS ANY LONGER!
                        zphot_low = np.zeros_like(len(source_cat_nostars_inseg))  # source_cat_nostars_inseg["Z_PHOT_L95"][w]
                        if zphot_low <= 0.1:
                            #the idea here is that at low-redshift, photo-zs are not accurate
                            zphot_low = 0
                        zphot_high = 100 + np.zeros_like(len(source_cat_nostars_inseg)) # source_cat_nostars_inseg["Z_PHOT_U95"][w]	
                        
                        # zphot_std = source_cat_nostars_inseg["Z_PHOT_STD"][w]	
                        # zphot_mean = source_cat_nostars_inseg["Z_PHOT_MEAN"][w]	

                        #if object is fainter than z band 21, we just make its zphot range very large so that it is like it its photoz is not being used
                        if source_cat_nostars_inseg["mag_z"][w] >= 21:
                            zphot_low = 0
                            zphot_high = 100

                        
                        if use_photoz:
                            #if we are using photo-zs to separated objects, then 
                            in_zphot_range = (zphot_low <= source_redshift) & (source_redshift <= zphot_high)
                        else:
                            #if we are not using photo-zs to separated objects
                            in_zphot_range = True

                        if (in_zphot_range & (Z_likely_sources_max[w] >= conf_levels["98.7"])) | ( (zphot_low == -99) & (zphot_high == -99) & (Z_likely_sources_max[w] >= conf_levels["98.7"]) ) : 
                            # if the redshift is in 95% photoz interval + good colors
                            #or if the source has no photoz but good colors
                            #then we consider this source as part of our galaxies
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
                            
                            
                            
                            # Annotate the point
                            if use_photoz:
                                ax[4].text( source_cat_nostars_inseg["xpix"][w] , source_cat_nostars_inseg["ypix"][w] + 5,  "[%.2f,%.2f]"%(zphot_low, zphot_high),fontsize = 8, ha = "center")


            #########
            ## DEALING WITH STARS :) 
            #########

            #easy! In one step we just mask all the stars, specifically half of the radius-magnitude relation made by Rongpu
            aperture_mask[star_mask] = 0
            aperture_mask_no_bkg[star_mask] = 0            


        tractor_model = np.load(save_path + "/tractor_source_model.npy")

        #we will mask all the nans in the data
        aperture_mask[ np.isnan(tot_data) ] = 0
        aperture_mask_no_bkg[ np.isnan(tot_data) ] = 0

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

        fidu_aper_mag_g = 22.5 - 2.5*np.log10( phot_table_g["aperture_sum"].data[0] - tot_subtract_sources_g )
        fidu_aper_mag_r = 22.5 - 2.5*np.log10( phot_table_r["aperture_sum"].data[0] - tot_subtract_sources_r )
        fidu_aper_mag_z = 22.5 - 2.5*np.log10( phot_table_z["aperture_sum"].data[0] - tot_subtract_sources_z )

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
        source_cat_nostars_inseg_remove = source_cat_nostars_inseg[source_cat_nostars_inseg_inds]
        source_cat_nostars_inseg_remove.write( save_path + "/blended_source_remove_cat.fits", overwrite=True)
        #this catalog will be used to get the tractor model of the blended sources
        
        ##########################################
        ### PLOTTING CODE
        ##########################################

        ##GRZ IMAGE
        ax_id = 5
        ax[ax_id].set_title("grz image w/aperture",fontsize = 13)
        ax[ax_id].imshow(rgb_stuff,origin="lower",zorder = 0)
        #th box size is 350x350
        ax[ax_id].text(65,325, "(%.3f,%.3f, z=%.3f)"%(source_ra,source_dec, source_redshift) ,color = "yellow",fontsize = 10)

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
        plot_now = source_cat_inseg_signi[(~is_star_inseg_signi)]
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
        start = (350 - size) // 2
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
        ax[ax_id].text(0.05,start - spacing*1,"Aper-mag-R35 g = %.2f"%fidu_aper_mag_g,size = fsize,transform=ax[ax_id].transAxes, verticalalignment='top')
        ax[ax_id].text(0.05,start - spacing*2,"fracflux_g = %.2f"%(source_cat_obs["fracflux_g"]),size = 12,transform=ax[ax_id].transAxes, verticalalignment='top')
        
        ax[ax_id].text(0.05,start - spacing*3,"Tractor-mag r = %.2f"%source_mag_r,size =fsize,transform=ax[ax_id].transAxes, verticalalignment='top')
        ax[ax_id].text(0.05,start - spacing*4,"Aper-mag-R35 r = %.2f"%fidu_aper_mag_r,size = fsize,transform=ax[ax_id].transAxes, verticalalignment='top')
        ax[ax_id].text(0.05,start - spacing*5,"fracflux_r= %.2f"%(source_cat_obs["fracflux_r"]),size = fsize,transform=ax[ax_id].transAxes, verticalalignment='top')

        ax[ax_id].text(0.05,start - spacing*6,"Tractor-mag z = %.2f"%source_mag_z,size =fsize,transform=ax[ax_id].transAxes, verticalalignment='top')
        ax[ax_id].text(0.05,start - spacing*7,"Aper-mag-R35 z = %.2f"%fidu_aper_mag_z,size = fsize,transform=ax[ax_id].transAxes, verticalalignment='top')
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
        np.save(save_path + "/aper_r35_mags.npy", fidu_aper_mags)        
        np.save(save_path + "/org_mags.npy", org_mags)


        return {
                "closest_star_dist": closest_star_dist,
                "closest_star_mag": closest_star_mag,
                "fidu_aper_mags": fidu_aper_mags,
                "org_mags": org_mags,
                "save_path": save_path,
                "save_summary_png": save_summary_png,
                "img_path": img_path,
                "closest_star_norm_dist": closest_star_norm_dist
            }
        
        # return closest_star_dist, closest_star_mag, fidu_aper_mags, org_mags, save_path, save_summary_png, img_path




    

        
        
        
        


        

        

    



    

                                    
   
    

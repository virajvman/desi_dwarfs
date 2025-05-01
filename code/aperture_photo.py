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
from photutils.segmentation import detect_sources, deblend_sources
import matplotlib.cm as cm
url_prefix = 'https://www.legacysurvey.org/viewer/'
# Third-party imports 
import requests
from io import BytesIO
from astropy.io import fits
import matplotlib.colors as mcolors
from matplotlib.patches import Circle
from matplotlib.collections import PatchCollection
from easyquery import Query, QueryMaker
reduce_compare = QueryMaker.reduce_compare
import random
import argparse
import concurrent.futures
from photutils.morphology import data_properties
from photutils.segmentation import make_2dgaussian_kernel
from astropy.convolution import convolve

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


# def fetch_noise(ra, dec,size = 300):
#     """
#     Returns the noise matrix per pixel level. Use to compute errors!
#     """
#     url = url_prefix + f'cutout.fits?ra={ra}&dec={dec}&'
    
#     url += 'layer=ls-dr9&size=%d&subimage'%size
#     print(url)
#     session = requests.Session()
#     resp = session.get(url)
#     cutout = fits.open(BytesIO(resp.content))
#     # ## THIS IS WORKING BUT NEED TO FIGURE OUT WHY THE ARRAY IS ALWAYS SMALLER...
#     # noise_image = {'g'}
#     # return noise_image

def mask_circle(array, x0, y0, r,value = 0):
    '''
    This will be used to mask the circular bright star region!
    '''
    ny, nx = array.shape
    y, x = np.ogrid[:ny, :nx]
    mask = (x - x0)**2 + (y - y0)**2 <= r**2
    array[mask] = value
    return array
    
# use this random state for reproducibility
# rng = np.random.RandomState(14)


                
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

    TO DO:

    1) When star is part of a different de-blended segment, do not mask the whole segment by default. Include that, but maybe just mask a certain aperture around the star?

    STAR LOGIC:

    1) If a star is not in the same island as DESI source, we do not do anything.
    2) If the star is on the main island, we mask the area within 0.5 of the star radius! If the main source is within 0.5, then it is beyond fixing!

    KNOWN ISSUES:
    1) I am not dealing with situations where an external source is split half way between aper rad. This happens if it accidentally connects
    with this external source making the whole main segment larger than aper rad and then if the external source has flux subtracted then whoops
    solution is to insist that aper_rad is larger than the main segment at all times

    3) Make sure the case where a source is in the star segment and thus it will be masked is star is in its own deblended segment. However, do not want to be over-substrating in that case
    '''


    ## DONE: JUST MASK ALL STARS WITHIN 0.5 STAR RADIUS THE WHOLE TIME !
    ## DONE:  GET ALL SOURCES IN GIVEN RADIUS EVEN IN CASE WHERE I AT EDGE OF A BRICK
    ## DONE: NEED TO ALSO IDENTIFY NON-BRIGHT STARS RADII as they can impact blending .. 
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
     
    is_star = (source_cat_f["ref_cat"] == "G2") & (source_cat_f['type'] == "PSF") 
    # ( ( np.abs(source_cat_f["pmra"] * np.sqrt(source_cat_f["pmra_ivar"])) >= 1 ) | ( np.abs(source_cat_f["pmdec"] * np.sqrt(source_cat_f["pmdec_ivar"])) >= 1 )  )
    # is_star = (source_cat_f["ref_cat"] == "G2") &  ( (source_cat_f["pmra"] != 0 ) | (source_cat_f["pmdec"] != 0 )  ) 
# 

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
            new_mags = np.load(save_path + "/new_aperture_mags.npy")
            org_mags = np.load(save_path + "/org_mags.npy")

            ## NEED TO CORRECTLY LOAD THE BRIGHT STAR PIXEL FRAC??
        
            return closest_star_dist, closest_star_mag, new_mags, org_mags, save_path, save_summary_png, True, img_path
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
        
        ##################
        ##PART 4: Read information on the aperture center!
        ##################
        
        # #load the aperture coordinates
        # aper_loc = np.load(save_path + "/aperture_cen_coord.npy")

        ##################
        ##Part 5: Make the diagnostic plots for each band
        ##################
    
        #first estimate the background error to use in aperture photometry
        noise_dict = {}
        bkg_dict = {}
        bkg_median_dict = {}
    
        #in case background estimate fails due to precense of massive galaxy
        #we will use this fiducial background
        noise_dict_fidu = {'g': 0.0020144903630876956, 'r': 0.002699455063905918, 'z': 0.008366939035641905}
        
        for bii in ["g","r","z"]:        
            ##Part 5.1: estimate the RMS in background which is useful for source detection
    
            ## if the background cannot be estimated then that means there is probably a very large source in the area!
            try:
                bkg_estimator = MedianBackground()
                bkg = Background2D(data[bii], (box_size, box_size), filter_size=(3, 3),
                                   bkg_estimator=bkg_estimator, exclude_percentile=10.0)
                
                bii_rms = np.median(bkg.background_rms)
                noise_dict[bii] = bii_rms
                bkg_dict[bii] = bkg.background
                bkg_median_dict[bii] = bkg.background_median
            except:
                ## if background cannot be estimated, this is likely to be not a dwarf galaxy
                # print("The background for %s band cannot be estimated. We use the fiducial noise for this band"%bii)
                noise_dict[bii] = noise_dict_fidu[bii]

        tot_data = np.sum(data_arr, axis=0)

        bkg_estimated = True

        try:
            bkg_estimator = MedianBackground()
            tot_bkg = Background2D(tot_data, (box_size, box_size), filter_size=(3, 3),
                               bkg_estimator=bkg_estimator, exclude_percentile=20.0)
            tot_noise_rms = np.median(tot_bkg.background_rms)
            bkg_estimated = True
            
        except:
            print("-----")
            print("could not get background TARGETID =",source_tgid)
            print("-----")
            tot_noise_rms = np.sqrt(  noise_dict_fidu["g"]**2 + noise_dict_fidu["r"]**2 + noise_dict_fidu["z"]**2 )
            #this variable is returned so we know after when exaclty this happens
            bkg_estimated = False
            
            
        from desi_lowz_funcs import make_subplots
        ax = make_subplots(ncol = 5, nrow = 2, row_spacing = 0.5,col_spacing=0.4, label_font_size = 17,plot_size = 3,direction = "horizontal")
        #2 rows per band and we have 3 bands:grz
    
        ## list stores the aperture size for each band
        all_aper_rads = []

        ##construct total image to identify segments

        npixels_min = 20
        
        threshold = 2 * tot_noise_rms

        kernel = make_2dgaussian_kernel(3.0, size=5)  # FWHM = 3.0
        convolved_tot_data = convolve( tot_data, kernel )

        # segment_map = detect_sources(tot_data, threshold, npixels=npixels_min) 
        segment_map = detect_sources(convolved_tot_data, threshold, npixels=npixels_min) 
        
    
        ##do deblending per band image
        segm_deblend = deblend_sources(convolved_tot_data, segment_map,
                                   npixels=npixels_min,nlevels=16, contrast=0.01,
                                   progress_bar=False)
        
        ##choose the aperture radius based on the size of the main segment  
    
        #get the segment number where the main source of interest lies in
        xpix, ypix,_ = wcs.all_world2pix(source_ra, source_dec,0,1)
        island_num = segment_map.data[int(ypix),int(xpix)]
        ## note that these 2d arrays are [y-coord, x-coord]

        #any segment not part of this main segment is considered to be a different source 
        #make a copy of the segment array
        segment_map_v2 = np.copy(segment_map.data)

        #is it possible that the source lies on background and not in one of the segmented islands?
        ## HMM, FOR ELGs, we need to not do this ... 
        
        
        if island_num == 0:
            #if the source lies on pixel classified as background, we find the nearest segment
            #we need to update island num

            all_xpixs, all_ypixs = np.meshgrid( np.arange(np.shape(segment_map_v2)[0]), np.arange(np.shape(segment_map_v2)[1]) )
            all_dists = np.sqrt ( ( all_xpixs - xpix)**2 + ( all_ypixs - ypix)**2 )
            
            #get all the distances to the pixels that are not background
            all_segs_notbg = segment_map_v2[ (segment_map_v2 != 0) ]
            all_dists_segpixs = all_dists[ (segment_map_v2 != 0)  ]
            #find closest one
            island_num = all_segs_notbg[ np.argmin(all_dists_segpixs) ]

            #we will have to also update the xpix and ypix location then if we end up using it hte future??
            xpix_new = all_xpixs[ (segment_map_v2 != 0) ][np.argmin(all_dists_segpixs) ]
            ypix_new = all_ypixs[ (segment_map_v2 != 0) ][np.argmin(all_dists_segpixs) ]
            
            sources_f_xpix[ (source_cat_f["ra"] == source_ra) &  (source_cat_f["dec"] == source_dec) ] = xpix_new
            sources_f_ypix[ (source_cat_f["ra"] == source_ra) &  (source_cat_f["dec"] == source_dec) ] = ypix_new

            #updating the main variable for future reference
            xpix = xpix_new
            ypix = ypix_new


        ##################
        ##PART 6: Make the star mask!!

        ##if the flux_ivar_grz of a star is 0, then that means it is a retained source and its broad band magnitude is taken from polynomial fits to color transformation
        ##turns out, starting with DR9, objects appearing in Gaia catalogs are now always RETAINED in the tractor catalogs
        ##################

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

        
        #pixels that are part of main segment island are called 2
        segment_map_v2[segment_map.data == island_num] = 2
        #all other segments that are not background are called 1
        segment_map_v2[(segment_map.data != island_num) & (segment_map.data > 0)] = 1
        #rest all remains 0

       

        def get_elliptical_aperture(segment_data, stellar_mask, id_num,sigma = 3):
            '''
            I feel the star mask to do so that some parts of the pixels are masked! 
            '''
            segment_data_v2 = np.copy(segment_data)
            ##we set everything else to zero except for our id num of reference
            segment_data_v2[segment_data != id_num] = 0
            segment_data_v2[segment_data == id_num] = 1

            #all pixels that are in stars are masked so that the aperture is not larger than it should be if it is blended with a star
            segment_data_v2[stellar_mask] = 0

            if np.sum( segment_data_v2) == 0:
                #in case source is too close to the star, then we do not code to crash so we unmask it again
                segment_data_v2[stellar_mask] = 1

            #we can set everything in the star mask to be zero as well
            
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

        
        aperture_for_phot = get_elliptical_aperture( segment_map_v2, star_mask, 2, sigma = 3.5 )
        aperture_for_phot_noscale = get_elliptical_aperture( segment_map_v2, star_mask, 2, sigma = 1 )
        

        aperture_for_phot.plot(ax = ax[7], color = "r", lw = 2.5, ls = "-")
        aperture_for_phot_noscale.plot(ax = ax[7], color = "r", lw = 1, ls = "dotted")
        
    
        ax[5].set_title("grz image w/aperture",fontsize = 13)
        ax[6].set_title("g+r+z data (log scaling)",fontsize = 13)
        ax[7].set_title("g+r+z data segmentation",fontsize = 13)
        # ax[6].set_title("g+r+z segmentation+deblend",fontsize = 13)

        ax[8].set_title("g+r+z band main segment",fontsize = 13)
        # ax[0].set_title("color-color space",fontsize = 13)
        ax[9].set_title("g+r+z band aperture mask",fontsize = 13)
        
        
        ax[3].set_title("summary",fontsize = 13)

        # Create a norm object and inspect the vmin/vmax
        norm_obj = LogNorm()
        
        ax[6].imshow(tot_data,origin="lower",norm=norm_obj,cmap = "viridis",zorder = 0)

        tot_data_vmin = norm_obj.vmin
        tot_data_vmax = norm_obj.vmax

        ax[7].imshow(segment_map, origin='lower', cmap=segment_map.cmap,
                   interpolation='nearest',zorder = 0)
    
        # ax[6].imshow(segm_deblend, origin='lower', cmap=segm_deblend.cmap,
        #            interpolation='nearest',zorder = 0)
        
        ax[5].imshow(rgb_stuff,origin="lower",zorder = 0)
        #recall the box size is 350x350
        ax[5].text(65,325, "(%.3f,%.3f, z=%.3f)"%(source_ra,source_dec, source_redshift) ,color = "yellow",fontsize = 10)
        
        #show the fiber location on the image
        
        #get pixel co-ordinates of the source galaxy
        circle = patches.Circle( (xpix, ypix),7, color='orange', fill=False, linewidth=1,ls ="-")
        ax[5].add_patch(circle)

        #overplot the centers of the DR9 sources for reference
        #all sources
        ax[7].scatter( sources_f_xpix,sources_f_ypix, s=5,color = "white",marker="^") 
        # ax[7].scatter( sources_f_xpix,sources_f_ypix, s=5,color = "white",marker = "^") 


        
        #sources that are a star, that is psf and pmra!=0
        ax[7].scatter( sources_f_xpix[is_star],sources_f_ypix[is_star], s=50,color = "white",marker="*",zorder = 3) 
        # ax[7].scatter( sources_f_xpix[is_star],sources_f_ypix[is_star], s=50,color = "white",marker="*",zorder = 3) 

        #plotting the final aperture
        for axi in [ax[5],ax[6],ax[7],ax[8]]:
            axi.set_xlim([0,box_size])
            axi.set_ylim([0,box_size])
            axi.set_xticks([])
            axi.set_yticks([])
            ##plotting the aperture for photometry.plotting is done below in the for-loop
            
            # aperture_for_phot.plot(ax = axi, color = "r", lw = 2.5, ls = "-")

            if bstar_ra != 99:
                
                aperture_for_bstar_1.plot(ax = axi, color = "r", lw = 1.5, ls = "dotted")
                aperture_for_bstar_34.plot(ax = axi, color = "r", lw = 1.5, ls = "dotted")
                aperture_for_bstar_12.plot(ax = axi, color = "r", lw = 1.5, ls = "dotted")

        
                

        #make a copy of the deblended array, this is the one where even the main segment is split into different deblended components
        segm_deblend_v2 = np.copy(segm_deblend.data)
        #create an array of nans with same shape
        segm_deblend_v3 = np.zeros_like(segm_deblend.data) * np.nan

        #we will populating segm_deblend_v3 with the different segments that are part of main segment island 
        #get deblend segments ids that are part of the main segment island
        deblend_ids = np.unique(segm_deblend_v2[segment_map_v2 == 2])

        #what is the deblend id where our main source is in?
        deblend_island_num = segm_deblend_v2[int(ypix),int(xpix)]

        #create another deblend image where we relabel the deblend ids of the main segment
        #note the 0 pixel means background and hence we have i+1 here
        new_deblend_ids = []
        for i,di in enumerate(deblend_ids):
            #we skip the i=0 step as that is the background
            segm_deblend_v3[(segm_deblend_v2 == di)] = i+1
            new_deblend_ids.append(i+1)
        new_deblend_ids = np.array(new_deblend_ids)

        #this is the new id of the deblend segment that contains our source
        new_deblend_island_num = int(new_deblend_ids[deblend_ids == deblend_island_num])

        source_cat_f["new_deblend_id"] = -99*np.ones_like(sources_f_xpix)

        #what are the deblend segment ids of our DR9 sources in our region?
        for k in range(len(source_cat_f)):
            #if the source is not part of any deblend island part of main segment,then we assign -99 as id
            if np.isnan( segm_deblend_v3[ int(sources_f_ypix[k]), int(sources_f_xpix[k]) ]  ):    
                source_cat_f["new_deblend_id"][k] =  -99
            else:
                source_cat_f["new_deblend_id"][k] = int(segm_deblend_v3[ int(sources_f_ypix[k]), int(sources_f_xpix[k]) ])

        ax[8].imshow(segm_deblend_v3, origin='lower', cmap="tab20",
                   interpolation='nearest')


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

        ##the color in these plots is from the tab20 color map
        cmap = cm.tab20
        # Normalize values to the range [0,1]
        norm = mcolors.Normalize(vmin=min(new_deblend_ids), vmax=max(new_deblend_ids))
        # Map values to colors
        colors = cmap(norm(np.array(new_deblend_ids)))

        ##plot the g-r vs. r-z color contours from the redshift bin of this object        
        ax[0].contour(X, Y, Z_gmm, levels=sorted(lvls), cmap="viridis_r",alpha = 1,zorder = 1)

        #plotting all the segment colors with error on the color-color plot
        for i in range(len(new_deblend_ids)):
            ax[0].scatter([all_segs_grs[i]], [all_segs_rzs[i]], s=50, marker="x",lw=2,zorder=3,color = colors[i])
            ax[0].errorbar([all_segs_grs[i]], [all_segs_rzs[i]], zorder=3,ecolor = colors[i],
                                  xerr=[ all_segs_grs_err[i] ], yerr= [ all_segs_rzs_err[i] ], fmt='none',alpha=1, capsize=3)

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

        ax[8].scatter( source_cat_inseg_signi[is_star_inseg_signi]["xpix"],source_cat_inseg_signi[is_star_inseg_signi]["ypix"], s=20,color = "r",marker="*" ) 

        plot_now = source_cat_inseg_signi[(~is_star_inseg_signi)]
        for p in range(len(plot_now)):
            ax[8].scatter( [ plot_now["xpix"][p]] , [ plot_now["ypix"][p]], s=10,color = "k",marker=plot_now["marker"][p]) 

        
        ax[8].set_xlim([0,box_size])
        ax[8].set_ylim([0,box_size])

        ax[8].set_xticks([])
        ax[8].set_yticks([])
        ax[9].set_xticks([])
        ax[9].set_yticks([])
        
         #note that if some of these psf sources are hii regions, then they can be very blue and go outside of our plotting limits!
        ax[0].scatter( source_cat_inseg_signi[is_star_inseg_signi]["g-r"], source_cat_inseg_signi[is_star_inseg_signi]["r-z"], color =  "r", marker = "*",s= 40,zorder = 2 ) 

        plot_now = source_cat_inseg_signi[(~is_star_inseg_signi) ]
        for p in range(len(plot_now)):
            ax[0].scatter( [ plot_now["g-r"][p]], [ plot_now["r-z"][p]], color = "k" , marker = plot_now["marker"][p], s= 20,zorder = 2 ) 
        
        ## plot the color-color errorbars
        ax[0].errorbar(source_cat_inseg_signi["g-r"], source_cat_inseg_signi["r-z"], 
            xerr=source_cat_inseg_signi["g-r_err"], yerr=source_cat_inseg_signi["r-z_err"], fmt='none', ecolor='k', alpha=0.3, capsize=3,zorder = 2)

        #how lenient we are in defining the color-color box
        col_lenient = 0.1
        
        ##plot the bluer boundary lines
        ax[0].hlines(y = all_segs_rzs[new_deblend_island_num - 1] + col_lenient,xmin = -0.5, xmax= all_segs_grs[new_deblend_island_num - 1] + col_lenient,color = "k",lw = 1, ls = "-")
        ax[0].vlines(x = all_segs_grs[new_deblend_island_num - 1] + col_lenient,ymin = -0.5, ymax= all_segs_rzs[new_deblend_island_num - 1] + col_lenient,color = "k",lw = 1, ls = "-")

        #generating empty mask array that has same shape as data
        #note that when passing the aperture masking array to the function, 1 = mask, 0 = no mask
        #so we will invert it at end. For our purposes till then, 1 means include/no mask, 0 = mask 
        aperture_mask = np.ones_like( tot_data)

        #firstly mask all the identified segments that are not part of our main segment island!
        aperture_mask[segment_map_v2 == 1] = 0

        #colors of reference deblend segment
        ref_deblend_gr = float(all_segs_grs[new_deblend_island_num - 1])
        ref_deblend_rz = float(all_segs_rzs[new_deblend_island_num - 1])

        #identifying segs that are redder than ref deblend seg by 0.1 or more in either direction. 0.1 is how much we are being lenient by
        #and has no nan and inf values 
        likely_other_seg_mask =  ( (all_segs_grs > (col_lenient + ref_deblend_gr)) | (all_segs_rzs > (col_lenient + ref_deblend_rz)) ) & ( ~np.isnan(all_segs_grs) ) & (~np.isnan(all_segs_rzs)) & ( ~np.isinf(all_segs_grs) ) & (~np.isinf(all_segs_rzs) )
        
        # ax[1+8*pind].set_xlim([ np.minimum( -0.5, np.min(all_segs_grs[likely_other_seg_mask])) ,np.maximum(1.5, np.max(all_segs_grs[likely_other_seg_mask]) ) ]  )
        # ax[1+8*pind].set_ylim([ np.minimum( -0.5, np.min(all_segs_grs[likely_other_seg_mask])) ,np.maximum(1.5, np.max(all_segs_rzs[likely_other_seg_mask] ))])
        ax[0].set_xlim([ -0.5, 2])
        ax[0].set_ylim([  -0.5, 1.5])
        ax[0].tick_params(axis='both', labelsize=10)
        ax[0].set_xlabel("g-r",fontsize = 17)
        ax[0].set_ylabel("r-z",fontsize = 17)

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



        ##we have so far dealt with deblended segments being part of main galaxy segment. What about individual sources?

        #the counter for total flux we will be subtracting at the end!
        tot_subtract_sources_g = 0
        tot_subtract_sources_r = 0
        tot_subtract_sources_z = 0
        
        if len(source_cat_inseg_signi) == 0:
            #nothing to do here!! That is there are no significant sources that we have to deal with in the color-color space
            pass
        else:
            ##We deal with the other non-stellar sources by looking at their DR9 photo-zs!!    
            source_cat_nostars_inseg = source_cat_inseg_signi[~is_star_inseg_signi]

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
                            ##adding flux of that source to the 
                            
                            tot_subtract_sources_g += 10**( (22.5  -  source_cat_nostars_inseg["mag_g"][w] )/2.5 )
                            tot_subtract_sources_r += 10**( (22.5  -  source_cat_nostars_inseg["mag_r"][w] )/2.5 )
                            tot_subtract_sources_z += 10**( (22.5  -  source_cat_nostars_inseg["mag_z"][w] )/2.5 )
                            

                             #plot this source for reference on the mask plot!!
                            ax[9].scatter( [source_cat_nostars_inseg["xpix"][w]] , [source_cat_nostars_inseg["ypix"][w] ],  color = "k", marker = source_cat_nostars_inseg["marker"][w],s= 20,  zorder = 1)
                            
                            # Annotate the point
                            if use_photoz:
                                ax[9].text( source_cat_nostars_inseg["xpix"][w] , source_cat_nostars_inseg["ypix"][w] + 5,  "[%.2f,%.2f]"%(zphot_low, zphot_high),fontsize = 8, ha = "center")


            #########
            ## DEALING WITH STARS :) 
            #########

            #easy! In one step we just mask all the stars, specifically half of the radius-magnitude relation made by Rongpu
            aperture_mask[star_mask] = 0

        
        #we will mask all the nans in the data
        aperture_mask[ np.isnan(tot_data) ] = 0

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
        # ax[1].imshow(aperture_mask_plot,origin="lower",cmap = "PiYG",zorder = 0,interpolation='nearest',vmin=0,vmax=1,alpha = 0.6)
        ax[9].imshow(tot_data_plot,origin="lower",norm=LogNorm(vmin=tot_data_vmin, vmax = tot_data_vmax),cmap = "viridis",zorder = 0)

        ax[9].set_xlim([0,box_size])
        ax[9].set_ylim([0,box_size])

        
        ##plotting the aperture again
        aperture_for_phot.plot(ax = ax[9], color = "r", lw = 2.5, ls = "-")
        aperture_for_phot.plot(ax = ax[8], color = "r", lw = 2.5, ls = "-")
        
        if bstar_ra != 99:
            aperture_for_bstar_1.plot(ax = ax[9], color = "r", lw = 1.5, ls = "dotted")
            aperture_for_bstar_34.plot(ax = ax[9], color = "r", lw = 1.5, ls = "dotted")
            aperture_for_bstar_12.plot(ax = ax[9], color = "r", lw = 1.5, ls = "dotted")

            # aperture_for_bstar_1.plot(ax = ax[0], color = "r", lw = 1.5, ls = "dotted")
            # aperture_for_bstar_34.plot(ax = ax[0], color = "r", lw = 1.5, ls = "dotted")
            # aperture_for_bstar_12.plot(ax = ax[0], color = "r", lw = 1.5, ls = "dotted")
            
        # np.save(save_path + "/aperture_mask_%s.npy"%bi, ~aperture_mask.astype(bool) )

        phot_table_g = aperture_photometry(data["g"] , aperture_for_phot, mask = ~aperture_mask.astype(bool))
        phot_table_r = aperture_photometry(data["r"] , aperture_for_phot, mask = ~aperture_mask.astype(bool))
        phot_table_z = aperture_photometry(data["z"] , aperture_for_phot, mask = ~aperture_mask.astype(bool))

        
        new_mag_g = 22.5 - 2.5*np.log10( phot_table_g["aperture_sum"].data[0] - tot_subtract_sources_g )
        new_mag_r = 22.5 - 2.5*np.log10( phot_table_r["aperture_sum"].data[0] - tot_subtract_sources_r )
        new_mag_z = 22.5 - 2.5*np.log10( phot_table_z["aperture_sum"].data[0] - tot_subtract_sources_z )
        
        org_mags = [source_mag_g_mwc,source_mag_r_mwc,source_mag_z_mwc] 

        tractor_model = np.load(save_path + "/tractor_source_model.npy")

        #we want to plot the original iamge 
        resis = data_arr - tractor_model 
        rgb_resis = sdss_rgb(resis, ["g","r","z"], scales=dict(g=(2,6.0), r=(1,3.4), z=(0,2.2)), m=0.03)
        
        size = 64
        start = (350 - size) // 2
        end = start + size

        ax[1].set_title(r"IMG")
        ax[1].imshow(rgb_stuff[start:end, start:end,:])
        ax[1].set_xticks([])
        ax[1].set_yticks([])
        ax[1].set_xlim([0,63])
        ax[1].set_ylim([0,63])
        
        ax[2].set_title(r"IMG - S")
        ax[2].imshow(rgb_resis[start:end, start:end,:])
        ax[2].set_xticks([])
        ax[2].set_yticks([])
        
        ## add text saying what the old magnitude and the new magnitude
        ax[3].set_xlim([0,1])
        ax[3].set_ylim([0,1])
        ax[3].set_xticks([])
        ax[3].set_yticks([])

        spacing = 0.08
        start = 0.99
        fsize = 11
        ax[3].text(0.05,start,"Org-mag g = %.2f"%source_mag_g,size =fsize,transform=ax[3].transAxes, verticalalignment='top')
        ax[3].text(0.05,start - spacing*1,"New-mag g = %.2f"%new_mag_g,size = fsize,transform=ax[3].transAxes, verticalalignment='top')
        ax[3].text(0.05,start - spacing*2,"fracflux_g = %.2f"%(source_cat_obs["fracflux_g"]),size = 12,transform=ax[3].transAxes, verticalalignment='top')
        
        ax[3].text(0.05,start - spacing*3,"Org-mag r = %.2f"%source_mag_r,size =fsize,transform=ax[3].transAxes, verticalalignment='top')
        ax[3].text(0.05,start - spacing*4,"New-mag r = %.2f"%new_mag_r,size = fsize,transform=ax[3].transAxes, verticalalignment='top')
        ax[3].text(0.05,start - spacing*5,"fracflux_r= %.2f"%(source_cat_obs["fracflux_r"]),size = fsize,transform=ax[3].transAxes, verticalalignment='top')

        ax[3].text(0.05,start - spacing*6,"Org-mag z = %.2f"%source_mag_z,size =fsize,transform=ax[3].transAxes, verticalalignment='top')
        ax[3].text(0.05,start - spacing*7,"New-mag z = %.2f"%new_mag_z,size = fsize,transform=ax[3].transAxes, verticalalignment='top')
        ax[3].text(0.05,start - spacing*8,"fracflux_z = %.2f"%(source_cat_obs["fracflux_z"]),size = fsize,transform=ax[3].transAxes, verticalalignment='top')

        ax[3].text(0.05,start - spacing*9,"Closest Star fdist = %.2f"%(closest_star_norm_dist),size = fsize,transform=ax[3].transAxes, verticalalignment='top')
        ax[3].text(0.05,start - spacing*10,"Bright star fdist = %.2f"%(bstar_fdist),size = fsize,transform=ax[3].transAxes, verticalalignment='top')
        ax[3].text(0.05,start - spacing*11,"SGA Dist (deg), NDist = %.2f, %.2f"%(sga_dist, sga_ndist),size = fsize,transform=ax[3].transAxes, verticalalignment='top')
        
        
        new_mag_g = new_mag_g + 2.5 * np.log10(source_cat_obs["mw_transmission_g"])
        new_mag_r = new_mag_r + 2.5 * np.log10(source_cat_obs["mw_transmission_r"])
        new_mag_z = new_mag_z + 2.5 * np.log10(source_cat_obs["mw_transmission_z"])
        
        new_mags = [new_mag_g, new_mag_r, new_mag_z]

        plt.savefig( save_summary_png ,bbox_inches="tight")

        plt.close()
    
        ##save these as a file for future reference
        np.save(save_path + "/new_aperture_mags.npy", new_mags)
        np.save(save_path + "/org_mags.npy", org_mags)

        
        return closest_star_dist, closest_star_mag, new_mags, org_mags, save_path, save_summary_png, bkg_estimated, img_path




    

        
        
        
        


        

        

    



    

                                    
   
    

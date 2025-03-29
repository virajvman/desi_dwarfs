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

    


def fetch_noise(ra, dec,size = 300):
    """
    Returns the noise matrix per pixel level. Use to compute errors!
    """
    url = url_prefix + f'cutout.fits?ra={ra}&dec={dec}&'
    
    url += 'layer=ls-dr9&size=%d&subimage'%size
    print(url)
    session = requests.Session()
    resp = session.get(url)
    cutout = fits.open(BytesIO(resp.content))
    # ## THIS IS WORKING BUT NEED TO FIGURE OUT WHY THE ARRAY IS ALWAYS SMALLER...
    # noise_image = {'g'}
    # return noise_image

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

    KNOWN ISSUES:
    1) I am not dealing with situations where an external source is split half way between aper rad. This happens if it accidentally connects
    with this external source making the whole main segment larger than aper rad and then if the external source has flux subtracted then whoops
    solution is to insist that aper_rad is larger than the main segment at all times

    2) What happens if an external source happens to lie very close to the blue box? Then it is not removed :( Need to fix this

    3) Make sure the case where a source is in the star segment and thus it will be masked is star is in its own deblended segment. However, do not want to be over-substrating in that case
    '''

    save_path = input_dict["save_path"]
    save_sample_path  = input_dict["save_sample_path"]

    save_other_image_path  = input_dict["save_other_image_path"]
    
    img_path  = input_dict["img_path"]
    source_tgid  = input_dict["tgid"]
    source_ra  = input_dict["ra"] 
    source_dec  = input_dict["dec"]
    source_redshift  = input_dict["redshift"]
    wcs  = input_dict["wcs"]
    data_arr  = input_dict["image_data"]
    source_cat_f = input_dict["source_cat"]
    object_index = input_dict["index"]
    org_mag_g = input_dict["org_mag_g"]
    overwrite = input_dict["overwrite"]

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

    ## quick search for stars in our source catalog of interest
    #this is the bright_star definition in SAGA build3.py file
    #     bright_stars = Query(
    #     "SHAPE_R < 1",
    #     "r_mag < 17",
    #     (Query("abs(PMRA * sqrt(PMRA_IVAR)) >= 2") | Query("abs(PMDEC * sqrt(PMDEC_IVAR)) >= 2")),
    # )

    ref_coord = SkyCoord(ra=source_ra * u.deg, dec=source_dec * u.deg)
    sources_coords = SkyCoord(ra=source_cat_f["ra"].data * u.deg, dec=source_cat_f["dec"].data * u.deg)
    # Compute separations
    source_seps = ref_coord.separation(sources_coords).arcsec
    # ( np.abs(source_cat_f["pmra"] * np.sqrt(source_cat_f["pmra_ivar"])) >= 2 ) & ( np.abs(source_cat_f["pmdec"] * np.sqrt(source_cat_f["pmdec_ivar"])) >= 2 )
    # is_star = (source_cat_f["ref_cat"] == "G2") &  (source_seps != 0) 
    is_star = (source_cat_f["ref_cat"] == "G2") &  ( np.abs(source_cat_f["pmra"] * np.sqrt(source_cat_f["pmra_ivar"])) >= 2 ) & ( np.abs(source_cat_f["pmdec"] * np.sqrt(source_cat_f["pmdec_ivar"])) >= 2 ) 
    
    #we do not really care about it being really a star or not. If it is in the gaia catalog and it is not a source we targeted, it is not associated with our source!

    is_psf_nostar = (source_cat_f['type'] == "PSF") & (source_cat_f["ref_cat"] != "G2") 


    #what is the distance to the closest star from us?
    all_stars = source_cat_f[is_star]
    if len(all_stars) > 0:
        catalog_coords = SkyCoord(ra=all_stars["ra"].data * u.deg, dec=all_stars["dec"].data * u.deg)
        # Compute separations
        separations = ref_coord.separation(catalog_coords).arcsec
        closest_star_dist = np.min(separations)
    else:
        closest_star_dist = -99
    
    if overwrite == False:
        try:
            new_mags = np.load(save_path + "/new_aperture_mags.npy")
            org_mags = np.load(save_path + "/org_mags.npy")

            return object_index, closest_star_dist, new_mags, org_mags, save_path
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

        try:
            bkg_estimator = MedianBackground()
            tot_bkg = Background2D(tot_data, (box_size, box_size), filter_size=(3, 3),
                               bkg_estimator=bkg_estimator, exclude_percentile=20.0)
            tot_noise_rms = np.median(tot_bkg.background_rms)
            
        except:
            print("-----")
            print("could not get background TARGETID =",source_tgid)
            print("-----")
            tot_noise_rms = np.sqrt(  noise_dict_fidu["g"]**2 + noise_dict_fidu["r"]**2 + noise_dict_fidu["z"]**2 )
            # raise ValueError("Issue in estimating the background.")
            
        
        from desi_lowz_funcs import make_subplots
        ax = make_subplots(ncol = 4, nrow = 2, row_spacing = 0.5,col_spacing=0.9, label_font_size = 17,plot_size = 3,direction = "horizontal")
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
        #this was originally, but is not the convolved data
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
                
            
            
        #pixels that are part of main segment island are called 2
        segment_map_v2[segment_map.data == island_num] = 2
        #all other segments that are not background are called 1
        segment_map_v2[(segment_map.data != island_num) & (segment_map.data > 0)] = 1
        #rest all remains 0

        ##construct the elliptical aperture now!
        
        # all_xcords, all_ycords = np.meshgrid( np.arange(np.shape(segment_map_v2)[0]), np.arange(np.shape(segment_map_v2)[1]) )
        # all_dists = np.sqrt ( ( all_xcords - aper_loc[0])**2 + ( all_ycords - aper_loc[1])**2 )
        # farthest_pixel = np.max(all_dists[ segment_map_v2 == 2])
        # #use this farthest pixel in the main segment to approximate the aperture radius
        # aper_rad = farthest_pixel * 1.25
        # #making sure the aperture is not bigger than the cutout
        # aper_rad = np.minimum( aper_rad, 175*np.sqrt(2))
        # #making sure the aperture is at least as large as the previous band. Sometimes these dwarfs are too faint in z-band
        # if len(all_aper_rads) > 0:
        #     aper_rad = np.maximum( aper_rad, np.max(all_aper_rads) )
        # else:
        #     all_aper_rads.append(aper_rad)

        #with the aperture radius finalized, define the circular aperture
        # aperture_for_phot = CircularAperture( (float(aper_loc[0]), float(aper_loc[1])) , r =  aper_rad)

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

        aperture_for_phot = get_elliptical_aperture( segment_map_v2, 2, sigma = 3.5 )

        
        ax[4].set_title("grz image w/aperture",fontsize = 13)
        ax[5].set_title("g+r+z data (log scaling)",fontsize = 13)
        ax[6].set_title("g+r+z data segmentation",fontsize = 13)
        ax[7].set_title("g+r+z segmentation+deblend",fontsize = 13)

        ax[0].set_title("g+r+z band main segment",fontsize = 13)
        ax[1].set_title("color-color space",fontsize = 13)
        ax[2].set_title("g+r+z band aperture mask",fontsize = 13)
        
        ax[3].set_title("summary",fontsize = 13)
        

        ax[5].imshow(tot_data,origin="lower",norm=LogNorm(),cmap = "viridis",zorder = 0)
        
        ax[6].imshow(segment_map, origin='lower', cmap=segment_map.cmap,
                   interpolation='nearest',zorder = 0)
    
        ax[7].imshow(segm_deblend, origin='lower', cmap=segm_deblend.cmap,
                   interpolation='nearest',zorder = 0)
        
        ax[4].imshow(rgb_stuff,origin="lower",zorder = 0)
        #recall the box size is 350x350
        ax[4].text(50,325, "(%.3f,%.3f, z=%.3f)"%(source_ra,source_dec, source_redshift) ,color = "yellow",fontsize = 10)
        
        #show the fiber location on the image
        
        #get pixel co-ordinates of the source galaxy
        circle = patches.Circle( (xpix, ypix),7, color='orange', fill=False, linewidth=1,ls ="-")
        ax[4].add_patch(circle)

        #overplot the centers of the DR9 sources for reference
        #all sources
        ax[6].scatter( sources_f_xpix,sources_f_ypix, s=5,color = "white",marker="^") 
        ax[7].scatter( sources_f_xpix,sources_f_ypix, s=5,color = "white",marker = "^") 

        #sources that are a star, that is psf and pmra!=0
        ax[6].scatter( sources_f_xpix[is_star],sources_f_ypix[is_star], s=40,color = "white",marker="*",zorder = 3) 
        ax[7].scatter( sources_f_xpix[is_star],sources_f_ypix[is_star], s=40,color = "white",marker="*",zorder = 3) 

        #sources that are a psf, but not a star!
        ax[6].scatter( sources_f_xpix[is_psf_nostar],sources_f_ypix[is_psf_nostar], s=5,color = "white",marker="o",zorder = 3) 
        ax[7].scatter( sources_f_xpix[is_psf_nostar],sources_f_ypix[is_psf_nostar], s=5,color = "white",marker="o",zorder = 3)

        #plotting the final aperture
        for axi in [ax[4],ax[5],ax[6],ax[7]]:
            axi.set_xlim([0,box_size])
            axi.set_ylim([0,box_size])
            axi.set_xticks([])
            axi.set_yticks([])
            ##plotting the aperture for photometry.plotting is done below in the for-loop

            aperture_for_phot.plot(ax = axi, color = "r", lw = 2, ls = "dotted")
            
            # center = (aperture_for_phot.positions[0],aperture_for_phot.positions[1])  # (x, y) center of the circle
            # radius = aperture_for_phot.r     # Radius of the circle

            # circle = patches.Circle(center, radius, color='r', fill=False, linewidth=1,ls ="dotted")
            # axi.add_patch(circle)

        
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
        source_cat_f["xpix"] = sources_f_xpix
        source_cat_f["ypix"] = sources_f_ypix
        

        #what are the deblend segment ids of our DR9 sources in our region?
        for k in range(len(source_cat_f)):
            #if the source is not part of any deblend island part of main segment,then we assign -99 as id
            if np.isnan( segm_deblend_v3[ int(sources_f_ypix[k]), int(sources_f_xpix[k]) ]  ):    
                source_cat_f["new_deblend_id"][k] =  -99
            else:
                source_cat_f["new_deblend_id"][k] = int(segm_deblend_v3[ int(sources_f_ypix[k]), int(sources_f_xpix[k]) ])

        ax[0].imshow(segm_deblend_v3, origin='lower', cmap="tab20",
                   interpolation='nearest')


        ## compute the colors of each deblend segment that is part of the main segment
        all_segs_grs = []
        all_segs_rzs = []
        
        all_segs_grs_err = []
        all_segs_rzs_err = []

        # we should mask some pixels to prevent negative values here ... 
        for biii in ["g","r","z"]:
            data[biii][ np.isnan(data[biii]) ] = 0
            data[biii][ np.isinf(data[biii]) ] = 0
            data[biii][ data[biii] < -5*noise_dict[biii] ] = 0

        for din in new_deblend_ids:
            gband_flux_i = np.sum(data["g"][segm_deblend_v3 == din])
            rband_flux_i = np.sum(data["r"][segm_deblend_v3 == din])
            zband_flux_i = np.sum(data["z"][segm_deblend_v3 == din])

            #as all the error is assumed to be the same so we add them in quadrature
            #TODO: use the actual pixel inv-var map to do this
            gflux_err_i = np.sqrt( np.sum(segm_deblend_v3 == din) * noise_dict["g"]**2 )
            rflux_err_i = np.sqrt( np.sum(segm_deblend_v3 == din) * noise_dict["r"]**2 )
            zflux_err_i = np.sqrt( np.sum(segm_deblend_v3 == din) * noise_dict["z"]**2 )
                
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


        ##the color in these plots is from the tab20 color map
        cmap = cm.tab20
        # Normalize values to the range [0,1]
        norm = mcolors.Normalize(vmin=min(new_deblend_ids), vmax=max(new_deblend_ids))
        # Map values to colors
        colors = cmap(norm(np.array(new_deblend_ids)))

        ##plot the g-r vs. r-z color contours from the redshift bin of this object        
        ax[1].contour(X, Y, Z_gmm, levels=sorted(lvls), cmap="viridis_r",alpha = 1,zorder = 1)

        #plotting all the segment colors with error on the color-color plot
        for i in range(len(new_deblend_ids)):
            ax[1].scatter([all_segs_grs[i]], [all_segs_rzs[i]], s=50, marker="x",lw=2,zorder=3,color = colors[i])
            ax[1].errorbar([all_segs_grs[i]], [all_segs_rzs[i]], zorder=3,ecolor = colors[i],
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


        ## we want to also remove the source that we are already pointing at!
        # color_plot_source_mask = in_main_islands & ( (source_cat_f["mag_g"] < 22) |  (source_cat_f["mag_r"] < 22) | (source_cat_f["mag_z"] < 22 ))
        color_plot_source_mask = in_main_islands
        
        source_cat_inseg_signi = source_cat_f[ color_plot_source_mask ]
        #updating the source type masks
        is_star_inseg_signi = is_star[ color_plot_source_mask ]
        is_psf_nostar_inseg_signi = is_psf_nostar[ color_plot_source_mask]

        markers_rnd = get_random_markers(len(source_cat_inseg_signi) )
        source_cat_inseg_signi["marker"] = markers_rnd

        ax[0].scatter( source_cat_inseg_signi[is_star_inseg_signi]["xpix"],source_cat_inseg_signi[is_star_inseg_signi]["ypix"], s=20,color = "r",marker="*" ) 

 
        ## sadly plt.plot does not accept a list of markers and so we will have to loop over

        plot_now = source_cat_inseg_signi[is_psf_nostar_inseg_signi]
        for p in range(len(plot_now)):
            ax[0].scatter( [plot_now["xpix"][p]],[plot_now["ypix"][p]], s=10,color = "k",marker=plot_now["marker"][p] )

        plot_now = source_cat_inseg_signi[(~is_psf_nostar_inseg_signi) & (~is_star_inseg_signi)]
        for p in range(len(plot_now)):
            ax[0].scatter( [ plot_now["xpix"][p]] , [ plot_now["ypix"][p]], s=10,color = "k",marker=plot_now["marker"][p]) 

        ax[0].set_xlim([0,box_size])
        ax[0].set_ylim([0,box_size])

        ax[0].set_xticks([])
        ax[0].set_yticks([])
        ax[2].set_xticks([])
        ax[2].set_yticks([])
        
         #note that if some of these psf sources are hii regions, then they can be very blue and go outside of our plotting limits!
        ax[1].scatter( source_cat_inseg_signi[is_star_inseg_signi]["g-r"], source_cat_inseg_signi[is_star_inseg_signi]["r-z"], color =  "r", marker = "*",s= 40,zorder = 2 ) 

        plot_now = source_cat_inseg_signi[is_psf_nostar_inseg_signi]
        for p in range(len(plot_now)):
            ax[1].scatter( [plot_now["g-r"][p] ], [plot_now["r-z"][p]], color = "k", marker = plot_now["marker"][p] , s= 15,zorder = 2 ) 

        plot_now = source_cat_inseg_signi[ (~is_psf_nostar_inseg_signi) & (~is_star_inseg_signi) ]
        for p in range(len(plot_now)):
            ax[1].scatter( [ plot_now["g-r"][p]], [ plot_now["r-z"][p]], color = "k" , marker = plot_now["marker"][p], s= 20,zorder = 2 ) 
        
        ## plot the color-color errorbars
        ax[1].errorbar(source_cat_inseg_signi["g-r"], source_cat_inseg_signi["r-z"], 
            xerr=source_cat_inseg_signi["g-r_err"], yerr=source_cat_inseg_signi["r-z_err"], fmt='none', ecolor='k', alpha=0.3, capsize=3,zorder = 2)

        #how lenient we are in defining the color-color box
        col_lenient = 0.1
        
        ##plot the bluer boundary lines
        ax[1].hlines(y = all_segs_rzs[new_deblend_island_num - 1] + col_lenient,xmin = -0.5, xmax= all_segs_grs[new_deblend_island_num - 1] + col_lenient,color = "k",lw = 1, ls = "dotted")
        ax[1].vlines(x = all_segs_grs[new_deblend_island_num - 1] + col_lenient,ymin = -0.5, ymax= all_segs_rzs[new_deblend_island_num - 1] + col_lenient,color = "k",lw = 1, ls = "dotted")

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
        likely_other_seg_mask =  ( (all_segs_grs > (col_lenient + ref_deblend_gr)) | (all_segs_rzs > (col_lenient + ref_deblend_rz)) ) & ( ~np.isnan(all_segs_grs) ) & (~np.isnan(all_segs_rzs)) & ( ~np.isinf(all_segs_grs) ) & (~np.isinf(all_segs_rzs))
        
        # ax[1+8*pind].set_xlim([ np.minimum( -0.5, np.min(all_segs_grs[likely_other_seg_mask])) ,np.maximum(1.5, np.max(all_segs_grs[likely_other_seg_mask]) ) ]  )
        # ax[1+8*pind].set_ylim([ np.minimum( -0.5, np.min(all_segs_grs[likely_other_seg_mask])) ,np.maximum(1.5, np.max(all_segs_rzs[likely_other_seg_mask] ))])
        ax[1].set_xlim([ -0.5, 2])
        ax[1].set_ylim([  -0.5, 1.5])
        ax[1].tick_params(axis='both', labelsize=10)
        ax[1].set_xlabel("g-r",fontsize = 17)
        ax[1].set_ylabel("r-z",fontsize = 17)

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
        
        
        # source_cat_inseg_signi = source_cat_f[ color_plot_source_mask ]
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

                
                # if len(source_cat_nostars_inseg) > 0:
                for w in range(len(source_cat_nostars_inseg) ):
             
                    if is_in_blue_box[w]:
                        # print("include")
                        #yay include!
                        pass
                    else:
                        #if the source is redder, we will investigate its photo-zs
                        #unsure if I should choose 68 or 95 here
                        #also low-redshift photo-zs are often not accurate and so if it within 0.15 within errors we good!
                        #but we also want to look reliable photozs, or ones with, not too large errors
                        #not sure how to be quantitative about this
                        sra = source_cat_nostars_inseg["ra"][w]	
                        sdec = source_cat_nostars_inseg["dec"][w]	
                        
                        
                        zphot_low = source_cat_nostars_inseg["Z_PHOT_L95"][w]
                        if zphot_low <= 0.1:
                            #the idea here is that at low-redshift, photo-zs are not accurate
                            zphot_low = 0
                        zphot_high = source_cat_nostars_inseg["Z_PHOT_U95"][w]	
                        zphot_std = source_cat_nostars_inseg["Z_PHOT_STD"][w]	
                        zphot_mean = source_cat_nostars_inseg["Z_PHOT_MEAN"][w]	

                        #what do if the zphot is -99?

                        #if it has a reliable photo-zs
                        in_zphot_range = (zphot_low <= source_redshift) & (source_redshift <= zphot_high)


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
                            ax[2].scatter( [source_cat_nostars_inseg["xpix"][w]] , [source_cat_nostars_inseg["ypix"][w] ],  color = "k", marker = source_cat_nostars_inseg["marker"][w],s= 20,  zorder = 1)
                            
                            # Annotate the point
                            ax[2].text( source_cat_nostars_inseg["xpix"][w] , source_cat_nostars_inseg["ypix"][w] + 5,  "[%.2f,%.2f]"%(zphot_low, zphot_high),fontsize = 8, ha = "center")



            ## What if a star is deblended into main segment?
            source_cat_stars_inseg = source_cat_inseg_signi[is_star_inseg_signi]


            if len(source_cat_stars_inseg) > 0:


                for j in range(len(source_cat_stars_inseg)):
                    #there exists a star in our main segment

                    #is this star part of our main deblend segment
                    if source_cat_stars_inseg["new_deblend_id"][j]  == new_deblend_island_num:

                        #star is part of the same deblend segment as our source
                        #as no other way to mask it, we subtract it from total apeture flux
                        aperture_mask[ segm_deblend_v3 == source_cat_stars_inseg["new_deblend_id"][j] ] = 1                        
                        #convert the star magnitude to flux to subtract from final counting

                        ## we do not subtract the source flux, but rather the psf flux?
                        
                        tot_subtract_sources_g +=  10**( (22.5  -  float(source_cat_stars_inseg["mag_g"][j]) )/2.5 ) 
                        tot_subtract_sources_r +=  10**( (22.5  -  float(source_cat_stars_inseg["mag_r"][j]) )/2.5 ) 
                        tot_subtract_sources_z +=  10**( (22.5  -  float(source_cat_stars_inseg["mag_z"][j]) )/2.5 ) 
                        
                        #plot this source for reference on the mask plot!!
                        ax[2].scatter( [source_cat_stars_inseg["xpix"]] , [source_cat_stars_inseg["ypix"]],  color = "r", marker = "*",s= 40,  zorder = 1)
                        
                    else:

                        # print("STAR is part of different deblend seg")
                        # print(np.unique(segm_deblend_v3  ))
                        # print( source_cat_stars_inseg["new_deblend_id"][j])
                        #TODO: make sure all pixels around some radius around the stars is masked because star can be split into 2 sometimes
                        #TODO: objid 1684 idx 21 has this weird case even if there is a star in mains egment, it is not being masked and substracted from source
                        #TODO: what do if star segment is kinda large ... even more than star. that is why below q is imp
                        #can we have a condition on the reliability of the photometry of star to know when to subtract??
            
                        #star is part of a different deblend segment in main island and so we will just mask that deblend segment

                        #there is an interesting failure case to consider here
                        #if there is a source that happens to be part of star segment that is classified as external
                        #then that source deblend is not masked and so the star is not masked again.
                        #need to fix this issue.
                        #if a deblend segment has a star, even if it has other sources, the star supersedes everything

                        ## if reliable size is known, could just carve that region out around star as well
                        aperture_mask[ segm_deblend_v3 == source_cat_stars_inseg["new_deblend_id"][j] ] = 0

                        #it is always safer to mask the star as the stars are sometimes very bright and pixels are not well measured etc.
            

        #we will mask all the nans in the data
        aperture_mask[ np.isnan(tot_data) ] = 0

        #what about pixels that anomalous negative values?
        #we will mask pixels that are 5sigma lower than the background!
        aperture_mask[ data["g"] < -5*noise_dict["g"] ] = 0
        aperture_mask[ data["r"] < -5*noise_dict["r"] ] = 0
        aperture_mask[ data["z"] < -5*noise_dict["z"] ] = 0
        
        #for plotting let us make everything outside the aperture nans
        aperture_mask_plot = np.copy(aperture_mask)
    
        # all_xpixs, all_ypixs = np.meshgrid( np.arange(np.shape(aperture_mask)[0]), np.arange(np.shape(aperture_mask)[1]) )
        # all_dists = np.sqrt ( ( all_xpixs - aper_loc[0])**2 + ( all_ypixs - aper_loc[1])**2 )
        # radius = aperture_for_phot.r
        # aperture_mask_plot[ all_dists > radius ] = np.nan
 
        ax[2].imshow(aperture_mask_plot,origin="lower",cmap = "PiYG",zorder = 0,interpolation='nearest',vmin=0,vmax=1,alpha = 0.6)
        ax[2].set_xlim([0,box_size])
        ax[2].set_ylim([0,box_size])

        
        ##plotting the aperture again
        aperture_for_phot.plot(ax = ax[2], color = "r", lw = 1, ls = "-")
        aperture_for_phot.plot(ax = ax[0], color = "r", lw = 1, ls = "-")
        

        # center = (aperture_for_phot.positions[0],aperture_for_phot.positions[1])  # (x, y) center of the circle
        # radius = aperture_for_phot.r     # Radius of the circle
        # circle = patches.Circle(center, radius, color='r', fill=False, linewidth=1,ls ="-")
        # ax[2+8*pind].add_patch(circle)
        # circle = patches.Circle(center, radius, color='r', fill=False, linewidth=1,ls ="-")
        # ax[0+8*pind].add_patch(circle)

        # #first construct the mask
        ##I believe the mask that aperture photometry needs is the opposite, that is, true means mask it!

        # np.save(save_path + "/aperture_mask_%s.npy"%bi, ~aperture_mask.astype(bool) )

        phot_table_g = aperture_photometry(data["g"] , aperture_for_phot, mask = ~aperture_mask.astype(bool))
        phot_table_r = aperture_photometry(data["r"] , aperture_for_phot, mask = ~aperture_mask.astype(bool))
        phot_table_z = aperture_photometry(data["z"] , aperture_for_phot, mask = ~aperture_mask.astype(bool))
        
        new_mag_g = 22.5 - 2.5*np.log10( phot_table_g["aperture_sum"].data[0] - tot_subtract_sources_g )
        new_mag_r = 22.5 - 2.5*np.log10( phot_table_r["aperture_sum"].data[0] - tot_subtract_sources_r )
        new_mag_z = 22.5 - 2.5*np.log10( phot_table_z["aperture_sum"].data[0] - tot_subtract_sources_z )
        
        org_mags = [source_mag_g_mwc,source_mag_r_mwc,source_mag_z_mwc] 

        #we need to correct this mag for the mw transmission
        #we will use the transmission as our original source


        ## add text saying what the old magnitude and the new magnitude
        ax[3].set_xlim([0,1])
        ax[3].set_ylim([0,1])
        ax[3].set_xticks([])
        ax[3].set_yticks([])
        ax[3].text(0.05,0.95,"Org-mag g = %.2f"%source_mag_g,size =12,transform=ax[3].transAxes, verticalalignment='top')
        ax[3].text(0.05,0.85,"New-mag g = %.2f"%new_mag_g,size = 12,transform=ax[3].transAxes, verticalalignment='top')
        ax[3].text(0.05,0.75,"fracflux_g = %.2f"%(source_cat_obs["fracflux_g"]),size = 12,transform=ax[3].transAxes, verticalalignment='top')
        
        ax[3].text(0.05,0.65,"Org-mag r = %.2f"%source_mag_r,size =12,transform=ax[3].transAxes, verticalalignment='top')
        ax[3].text(0.05,0.55,"New-mag r = %.2f"%new_mag_r,size = 12,transform=ax[3].transAxes, verticalalignment='top')
        ax[3].text(0.05,0.45,"fracflux_r= %.2f"%(source_cat_obs["fracflux_r"]),size = 12,transform=ax[3].transAxes, verticalalignment='top')

        ax[3].text(0.05,0.35,"Org-mag z = %.2f"%source_mag_z,size =12,transform=ax[3].transAxes, verticalalignment='top')
        ax[3].text(0.05,0.25,"New-mag z = %.2f"%new_mag_z,size = 12,transform=ax[3].transAxes, verticalalignment='top')
        ax[3].text(0.05,0.15,"fracflux_z = %.2f"%(source_cat_obs["fracflux_z"]),size = 12,transform=ax[3].transAxes, verticalalignment='top')

        
        new_mag_g = new_mag_g + 2.5 * np.log10(source_cat_obs["mw_transmission_g"])
        new_mag_r = new_mag_r + 2.5 * np.log10(source_cat_obs["mw_transmission_r"])
        new_mag_z = new_mag_z + 2.5 * np.log10(source_cat_obs["mw_transmission_z"])
        
        new_mags = [new_mag_g, new_mag_r, new_mag_z]
        
    
        plt.savefig(save_path + "/grz_bands_segments.png",bbox_inches="tight")
        plt.savefig(save_sample_path + "grz_segs_tgid_%d_ra_%.3f_dec_%.3f.png"%(source_tgid, source_ra, source_dec),bbox_inches="tight")

        #also save the figure in another plot!
        if save_other_image_path != "":
            plt.savefig(save_other_image_path + "grz_segs_tgid_%d_ra_%.3f_dec_%.3f.png"%(source_tgid, source_ra, source_dec),bbox_inches="tight")
        
        plt.close()
    
        ##save these as a file for future reference
        np.save(save_path + "/new_aperture_mags.npy", new_mags)
        np.save(save_path + "/org_mags.npy", org_mags)
    
        return object_index, closest_star_dist, new_mags, org_mags, save_path

        # except:
    
        #     print("ERROR OCCURED IN THE PHOTOMETRY PIPEPLINE")
        #     print("TARGETID=", source_tgid)
        #     print("SAVE PATH=",save_path)
        #     print("data shape=",np.shape(data_arr[0]),np.shape(data_arr[1]),np.shape(data_arr[2])  )
        #     return object_index, -99, [-99,-99,-99], [-99,-99,-99]







    

        
        
        
        


        

        

    



    

                                    
   
    

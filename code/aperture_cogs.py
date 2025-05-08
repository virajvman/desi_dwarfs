'''
In this script, we will do a curve of growth analysis on our objects that are really shreds!

We are working in a different script here as we need the tractor/astrometry packages to construct the psf model!

Basic steps are following what SGA catalog did:
1) Identify the range of apertures within which we will do our photometry
2) Mask relevant pixels (if star or residuals after subtracting model image are very large?)
3) If we have identified sources within aperture that we want to subtract, we can create an image with all the masked pixels and subtracted sources for reference?

'''

from scipy.ndimage import gaussian_filter
from desi_lowz_funcs import make_subplots, sdss_rgb
import numpy as np
from photutils.aperture import aperture_photometry, EllipticalAperture
from photutils.morphology import data_properties
from astropy import units as u
from matplotlib.colors import LogNorm
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

def get_elliptical_aperture(segment_data, stellar_mask, id_num,sigma = 3):
    '''
    Function that takes in the main segment and a star mask and fits an elliptical aperture to it
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

def build_residual_mask(data_images, model_images, sigma_threshold=5, smooth_sigma=2):
    """
    Build a residual mask flagging pixels with >5-sigma residuals in any band.
    
    Parameters:
    -----------
    data_images : list of 2D np.ndarray
        List of data images (e.g., [data_g, data_r, data_z]).
    model_images : list of 2D np.ndarray
        List of corresponding model images (e.g., [model_g, model_r, model_z]).
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
    This tractor model image will have to be for sources that are not in the main segment. That would require us to rerun the tractor pipeline

    
    """
    assert len(data_images) == len(model_images), "Mismatch in number of bands"
    
    # Initialize mask
    mask = np.zeros_like(data_images[0], dtype=bool)
    
    for data, model in zip(data_images, model_images):
        # Compute residual image
        residual = data - model
        
        # Smooth the residual
        smoothed_residual = gaussian_filter(residual, sigma=smooth_sigma)
        
        # Estimate per-pixel sigma using the smoothed residual's standard deviation
        # (Alternative: if you have per-pixel noise maps, you should use them here)
        sigma_estimate = np.std(smoothed_residual)
        
        # Flag pixels where the absolute smoothed residual exceeds threshold
        band_mask = np.abs(smoothed_residual) > (sigma_threshold * sigma_estimate)
        
        # Combine mask across bands (OR logic: mask if flagged in any band)
        mask |= band_mask
    
    return mask



def cog_function(r,mtot, m0, alpha_1, r_0, alpha_2):
    '''
    This is empirical model for the curve of growth
    '''

    return mtot + m0 * np.log(1 + alpha_1 * ( (r/r_0)**(-alpha_2) ) )



def fit_cog(r_data, m_data, p0, bounds=None, maxfev=1200, filler=np.nan):
    # Filter out NaNs
    valid = ~np.isnan(r_data) & ~np.isnan(m_data)
    r_clean = r_data[valid]
    m_clean = m_data[valid]

    # If too few points, return filler
    if len(r_clean) < len(p0):
        print("Not enough valid data points. Returning filler values.")
        return np.full(len(p0), filler)

    try:
        popt, pcov = curve_fit(
            cog_function, r_clean, m_clean, p0=p0,
            bounds=bounds or (-np.inf, np.inf),
            maxfev=maxfev
        )
        return popt
    except RuntimeError:
        print("Fit failed. Returning filler values.")
        return np.full(len(p0), filler)
    
def run_cogs(data_arr, segment_map_v2, star_mask, aperture_mask, tot_subtract_sources, fidu_mag, tgid, save_path):
    '''
    In this function, we run the curve-of-growth (COG) analysis!

    tot_subtract_sources is a dictionary that contains the total flux of sources we need to subtract
    tot_subtract_sources = { "g": tot_subtract_sources_g, "r": tot_subtract_sources_r, "z": tot_subtract_sources_z  }
    '''

    fig,ax = make_subplots(ncol = 3, nrow = 2, row_spacing = 0.5,col_spacing=1.2, label_font_size = 17,plot_size = 3,direction = "horizontal", return_fig=True)


    #load the tractor background model and the main segment map
    tractor_bkg_model = np.load(save_path + "/tractor_background_model.npy")
    segm_deblend_v3  = np.load(save_path + "/main_segment_map.npy")


    #when computing the residual mask, we will be setting the pixels in the main segment to zero
    #another approach is to set pixels within the aperture to zero, but that seems a bit more arbitrary
    data_arr_copy = np.copy(data_arr)
    main_seg_pix_locs = ~np.isnan(segm_deblend_v3)
    data_arr_copy[:, main_seg_pix_locs] = 0

    #this is a mask where True means it is a masked pixel
    resid_5sig_mask = build_residual_mask(data_arr_copy, tractor_bkg_model, sigma_threshold=5, smooth_sigma=2)

    #we will apply this mask to our data when computing the curve of growth analysis!
    #we need to combine this mask with our existing aperture mask
    #we have to invert it though as were definigin aperture_mask=0 as pixels to be masked
    aperture_mask = ~aperture_mask.astype(bool)
    #then we combine such that if one of these masks indicates it should be masked we mask it
    aperture_mask |= resid_5sig_mask

    ##compute the image which has the background model subtracted
    data_arr_no_bkg = data_arr - tractor_bkg_model

    data_no_bkg = {"g": data_arr_no_bkg[0], "r": data_arr_no_bkg[1] , "z": data_arr_no_bkg[2] }

    #we take the initial aperture we have and scale it 
    #this is in units of the semi-major/semi-minor ellipse
    #we are just scaling that ellipse by these values!
    
    radii_scale = np.arange(2,4.25,0.25)
            
    cog_mags = {"g":[], "r": [], "z": []}

    for scale_i in radii_scale:
        aperture_for_phot_i = get_elliptical_aperture( segment_map_v2, star_mask, 2, sigma = scale_i )

        #we only plot it at the edge cases
        if scale_i == 2:
            aperture_for_phot_i.plot(ax = ax[0], color = "k", lw = 1, ls = "-",alpha = 0.75)
        if scale_i == 3:
            aperture_for_phot_i.plot(ax = ax[0], color = "k", lw = 1, ls = "--",alpha = 0.75)
        if scale_i == 4:
            aperture_for_phot_i.plot(ax = ax[0], color = "k", lw = 1, ls = "dotted",alpha = 0.75)

        
        for bi in "grz":
            phot_table_i = aperture_photometry(data_no_bkg[bi] , aperture_for_phot_i, mask = aperture_mask)
            new_mag_i = 22.5 - 2.5*np.log10( phot_table_i["aperture_sum"].data[0] - tot_subtract_sources[bi] )
            cog_mags[bi].append(new_mag_i)

    
    ##fit the parametric form to these data points
    final_fits = {}
    for bi in "grz":
        #for the initial guess, mtot, m0, alpha_1, r_0, alpha_2
        p0 = [cog_mags[bi][-1], 2.0, 0.5, 1.5, 3]
        bounds = ([cog_mags[bi][-1] - 1, 0, 0, 0.1, 0], [cog_mags[bi][-1] + 1, 10, 10, 10, 10])

        popt = fit_cog(radii_scale, np.array(cog_mags[bi]), p0, bounds=bounds, maxfev=1200, filler=np.nan)
        # popt, pcov = curve_fit(cog_function, radii_scale, cog_mags[bi], p0=p0, bounds=bounds)
        
        final_fits[bi] = popt
    
    box_size = 350 
    
    ##plot the rgb image of actual data 
    ax_id = 3
    rgb_img = sdss_rgb([data_arr[0],data_arr[1],data_arr[2]], ["g","r","z"], scales=dict(g=(2,6.0), r=(1,3.4), z=(0,2.2)), m=0.03)
    ax[ax_id].imshow(rgb_img, origin='lower')
    ax[ax_id].set_xlim([0,box_size])
    ax[ax_id].set_ylim([0,box_size])
    ax[ax_id].set_xticks([])
    ax[ax_id].set_yticks([])

    ##plot rgb data of the bkg tractor model
    ax_id = 4
    rgb_bkg_model = sdss_rgb([tractor_bkg_model[0],tractor_bkg_model[1],tractor_bkg_model[2]], ["g","r","z"], scales=dict(g=(2,6.0), r=(1,3.4), z=(0,2.2)), m=0.03)
    ax[ax_id].imshow(rgb_bkg_model, origin='lower')
    ax[ax_id].set_xlim([0,box_size])
    ax[ax_id].set_ylim([0,box_size])
    ax[ax_id].set_xticks([])
    ax[ax_id].set_yticks([])
    
    

    ##plot the rgb data - bkg tractor model
    ax_id = 5
    rgb_img_m_bkg_model = sdss_rgb([data_arr_no_bkg[0],data_arr_no_bkg[1],data_arr_no_bkg[2]], ["g","r","z"], scales=dict(g=(2,6.0), r=(1,3.4), z=(0,2.2)), m=0.03)
    
    ax[ax_id].imshow(rgb_img_m_bkg_model, origin='lower')
    ax[ax_id].set_xlim([0,box_size])
    ax[ax_id].set_ylim([0,box_size])
    ax[ax_id].set_xticks([])
    ax[ax_id].set_yticks([])

    
    ##for reference, make the g+r+z plot with the masked pixels and different apertures on it
    ax_id = 0

    combine_img = data_arr_no_bkg[0] + data_arr_no_bkg[1] + data_arr_no_bkg[2]
    #setting the masked pixels to nans
    combine_img[aperture_mask] = np.nan

    norm_obj = LogNorm()
    ax[ax_id].imshow(combine_img,origin="lower",norm=norm_obj,cmap = "viridis",zorder = 0)
    
    ax[ax_id].set_xlim([0,box_size])
    ax[ax_id].set_ylim([0,box_size])
    ax[ax_id].set_xticks([])
    ax[ax_id].set_yticks([])
    
    ##make the cog plot!
    ax_id = 1

    all_cogs = np.concatenate( (cog_mags["g"],cog_mags["r"],cog_mags["z"]) )
    all_cogs = all_cogs[ ~np.isinf(all_cogs) & ~np.isnan(all_cogs)]

    rgrid = np.linspace(1,5,100)

    # cog_function(r,mtot, m0, alpha_1, r_0, alpha_2)

    if len(all_cogs) > 0:
        ax[ax_id].scatter(radii_scale, cog_mags["g"],color = "mediumblue",zorder = 1)
        if ~np.isnan(final_fits["g"][0]):
            ax[ax_id].plot(rgrid, cog_function(rgrid, *final_fits["g"]), color = "mediumblue",lw =1,zorder = 1  )

        ax[ax_id].scatter(radii_scale, cog_mags["r"],color = "forestgreen",zorder = 1)
        
        if ~np.isnan(final_fits["r"][0]):
            ax[ax_id].plot(rgrid, cog_function(rgrid, *final_fits["r"]), color = "forestgreen",lw =1,zorder = 1  )
        
        ax[ax_id].scatter(radii_scale, cog_mags["z"],color = "firebrick",zorder = 1)
        
        if ~np.isnan(final_fits["z"][0]):
            ax[ax_id].plot(rgrid, cog_function(rgrid, *final_fits["z"]), color = "firebrick",lw =1 ,zorder = 1 )

        ax[ax_id].set_ylabel(r"$m(<r)$ mag",fontsize = 14)
        ax[ax_id].set_xlim([1, 5])
        ax[ax_id].vlines(x = 2, ymin=np.min(all_cogs) - 0.25, ymax = np.max(all_cogs) + 0.25, color = "k",ls = "-",lw = 1, alpha = 0.75, zorder = 0)
        ax[ax_id].vlines(x = 3, ymin=np.min(all_cogs) - 0.25, ymax = np.max(all_cogs) + 0.25, color = "k",ls = "--",lw = 1, alpha = 0.75,zorder = 0)
        ax[ax_id].vlines(x = 4, ymin=np.min(all_cogs) - 0.25, ymax = np.max(all_cogs) + 0.25, color = "k",ls = "dotted",lw = 1, alpha = 0.75, zorder = 0)
        
        ax[ax_id].set_ylim([ np.max(all_cogs) + 0.125,np.min(all_cogs) - 0.125  ] )
    
    ##show the cog summary numbers!!
    ax_id = 2
    # print(final_fits)


    ##saving this image :0
    plt.savefig( save_path + "/cog_summary.png" ,bbox_inches="tight")
    plt.savefig(f"/pscratch/sd/v/virajvm/temp_cog_plots/cog_{tgid}.png" ,bbox_inches="tight")
    
    plt.close()

    


        ###
        #curve of growth analysis
        #TODO: add the fitting function to get asymptotic magnitude
        #TODO: why does z-band magnitude dip down some times?
        #TODO: I am not masking enough pixels of the background sources I think ... do this better ...

        #The COG analysis relies on subtracting tractor models from the outskirts
        #So we will do this in a different script for clarity and ability to working the tractor package 
        #we will save all the relevant files so we can directly just load them!
        ###

        ##Instead of cog, include the 
        
        

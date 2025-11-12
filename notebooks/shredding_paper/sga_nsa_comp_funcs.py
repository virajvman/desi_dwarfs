import matplotlib.pyplot as plt
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
import astropy.units as u
from matplotlib.patches import Ellipse
import cmasher as cmr
import os
import sys
from photutils.aperture import EllipticalAperture
import glob
import numpy as np
from tqdm import trange
from astropy.io import fits
from astropy.table import Table, vstack
from astropy.convolution import convolve, Gaussian1DKernel
import astropy.units as u
import astropy.coordinates as coord
import matplotlib
import matplotlib.pyplot as plt
from astropy.table import Column
from tqdm import trange
import pandas as pd
import fitsio
from astropy.table import Table, vstack
from astropy import units as u
from astropy.coordinates import SkyCoord
from easyquery import Query, QueryMaker
from scipy.stats import binomtest
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import LogNorm
from matplotlib.colors import ListedColormap, BoundaryNorm
import h5py
from astropy.cosmology import Planck18
import glob
from matplotlib.lines import Line2D
from scipy.stats import median_abs_deviation

rootdir = '/global/u1/v/virajvm/'
sys.path.append(os.path.join(rootdir, 'DESI2_LOWZ/desi_dwarfs/code'))

from consolidate_photometry import consolidate_new_photo
from desi_lowz_funcs import make_alternating_plot



mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['axes.linewidth'] = 1.5
mpl.rcParams['axes.xmargin'] = 1
mpl.rcParams['xtick.labelsize'] = 'x-large'
mpl.rcParams['xtick.major.size'] = 5
mpl.rcParams['xtick.major.width'] = 1.5
mpl.rcParams['ytick.labelsize'] = 'x-large'
mpl.rcParams['ytick.major.size'] = 5
mpl.rcParams['ytick.major.width'] = 1.5
mpl.rcParams['legend.frameon'] = False

rootdir = '/global/u1/v/virajvm/'
sys.path.append(os.path.join(rootdir, 'DESI2_LOWZ/desi_dwarfs/code'))

from desi_lowz_funcs import process_img, match_c_to_catalog

import warnings
from astropy.wcs import FITSFixedWarning

# Suppress just FITSFixedWarning
warnings.filterwarnings('ignore', category=FITSFixedWarning)


def cut_2d_array(img_data, cutout_size = 96, org_size = 350,return_shift = False):
    '''
    Function that center crops the image and returns rgb image!
    '''

    if cutout_size is not None:
        start = (org_size - cutout_size) // 2  # assumes square images
        end = start + cutout_size
        img_data = img_data[start:end, start:end]
        
    if return_shift:
        return img_data, start
    else:
        return img_data
    

def undo_extinct(mags, extinct):

    flux = 10**((22.5 - mags)/2.5)

    flux_new = flux * extinct

    mags_new = 22.5 - 2.5*np.log10(flux_new)

    return mags_new


def correct_extinct(mags, extinct):

    flux = 10**((22.5 - mags)/2.5)

    flux_new = flux / extinct

    mags_new = 22.5 - 2.5*np.log10(flux_new)

    return mags_new

def make_fav_cmap():
    cmr_i = cmr.gothic_r.copy()
    cmr_i.set_under(alpha=0.)
    return cmr_i


def make_compare_plot_panel(ax_top, ax_bot, sga_cat, ext_mag, desi_mag, band_name,
                            ylabel_top=None, ylabel_bot=None, xlabel_bot = None, add_cbar=False, 
                            show_cbar_label=False, mag_min = 11, mag_max = 21,bins=75, tickmarks = [12,14,16,18,20],
                           bad_tgids = None):

    delta_mag = ext_mag - desi_mag

    label_font_size = 15
    vmax=50

    cmap = make_fav_cmap()
    
    # top: Δmag vs mag
    # im = ax_top.hist2d(desi_mag, delta_mag, range=((mag_min, mag_max), (-1.5, 1.5)), bins=(bins, int(bins/3)),
    #                    norm=LogNorm(vmin=1, vmax=200), cmap=cmr.dusk_r,rasterized=True)
    im = ax_top.hist2d(desi_mag, delta_mag, range=((mag_min, mag_max), (-1.5, 1.5)), bins=(bins, int(bins/3)),
                       vmin=0,vmax=vmax, cmap=cmap,rasterized=True)


    xgrid = np.linspace(12,21,100)
    
    make_alternating_plot(ax_top,xgrid,0*xgrid,dash_len=1,color_1="yellowgreen",color_2="k",lw=1)
    make_alternating_plot(ax_top,xgrid,0*xgrid+0.75,dash_len=1,color_1="yellowgreen",color_2="k",lw=1,alpha=0.25)
    make_alternating_plot(ax_top,xgrid,0*xgrid-0.75,dash_len=1,color_1="yellowgreen",color_2="k",lw=1,alpha=0.25)

    ax_top.set_ylim(-1.5, 1.5)

    ax_bot.set_yticks(tickmarks)
    ax_bot.set_xticks(tickmarks)

    if ylabel_top:
        ax_top.set_ylabel(ylabel_top, fontsize=13.5)
    else:
        ax_top.set_yticklabels([])
    
    # bottom: mag vs mag
    # im = ax_bot.hist2d(desi_mag, ext_mag, range=((mag_min, mag_max), (mag_min, mag_max)), bins=bins,
                       # norm=LogNorm(vmin=1, vmax=200), cmap=cmr.dusk_r,rasterized=True,zorder=0)

    im = ax_bot.hist2d(desi_mag, ext_mag, range=((mag_min, mag_max), (mag_min, mag_max)), bins=bins,
                       vmin=0, vmax=vmax, cmap=cmap,rasterized=True,zorder=0)


    # ax_bot.plot([mag_min, mag_max], [mag_min, mag_max], lw=1, color="grey")
    # ax_bot.plot([mag_min, mag_max], [mag_min + 0.75, mag_max + 0.75], ls="--", color="grey", lw=0.75)
    # ax_bot.plot([mag_min, mag_max], [mag_min - 0.75, mag_max - 0.75], ls="--", color="grey", lw=0.75)

    xgrid = np.linspace(12,21,100)
    
    make_alternating_plot(ax_bot,xgrid,xgrid,dash_len=1,color_1="yellowgreen",color_2="k",lw=1)
    
    make_alternating_plot(ax_bot,xgrid,xgrid+0.75,dash_len=1,color_1="yellowgreen",color_2="k",lw=1,alpha=0.25)
    make_alternating_plot(ax_bot,xgrid,xgrid-0.75,dash_len=1,color_1="yellowgreen",color_2="k",lw=1,alpha=0.25)


    ax_bot.set_xlim([mag_min, mag_max])
    ax_bot.set_ylim([mag_min, mag_max])

    if bad_tgids is not None:
        #then we plot the VI'ed outliers as crosses!
        #the sga_cat is matching the desi_mag etc. lists in indexing
        idxs = np.isin(sga_cat["TARGETID"].data,  bad_tgids)

        ax_bot.scatter(desi_mag[idxs], ext_mag[idxs], color = "k",marker="x", s=20,zorder=1,alpha=0.75)
    
    if ylabel_bot:
        ax_bot.set_ylabel(ylabel_bot, fontsize=label_font_size)
    else:
        ax_bot.set_yticklabels([])
    
    ax_bot.set_xlabel(xlabel_bot, fontsize=label_font_size)
    ax_bot.text(0.5,0.075, f"{band_name}-band", weight="bold", fontsize=15,  transform=ax_bot.transAxes,va="center",ha="center" )

    # Stats
    
    delta_nmad = median_abs_deviation(delta_mag, scale="normal")
    outlier_frac = np.sum(np.abs(delta_mag) > 0.75) / len(delta_mag)

    if bad_tgids is not None:
        outlier_frac_VI = len(bad_tgids)/len(delta_mag)
    
    spacing = 0.075
    fs =  12.5
    if bad_tgids is not None:
        ax_bot.text(0.03, 0.935 - 3*spacing, rf"f$(|\Delta| > 0.75, \rm VI) = {outlier_frac_VI * 100:.1f}\%$", fontsize=fs,  transform=ax_bot.transAxes)
        
    ax_bot.text(0.03, 0.935 - 2*spacing, rf"f$(|\Delta| > 0.75) = {outlier_frac * 100:.1f}\%$", fontsize=fs,  transform=ax_bot.transAxes)
    ax_bot.text(0.03, 0.935 - spacing, rf"$\sigma_{{\rm NMAD}} = {delta_nmad:.2f}$", fontsize=fs,  transform=ax_bot.transAxes)
    ax_bot.text(0.03, 0.935, rf"$N = {len(ext_mag)}$", fontsize=fs,  transform=ax_bot.transAxes)

    if add_cbar:
        cbar_ax = ax_top.inset_axes([0.55, 0.85, 0.4, 0.075])  # relative to ax_top
        cbar = plt.colorbar(im[3], cax=cbar_ax, orientation='horizontal')
        cbar.ax.tick_params(labelsize=12)
        # if show_cbar_label:
            # cbar.set_label("Numbes", fontsize=10)

    return im




def make_compare_plot_3bands(ext_mags, aper_mags, sga_cat, band_names=("g", "r", "z"),
                             save_path=None, show_plot=False, ylabel_top = r"mag$_{\rm NSA}$- mag$_{\rm aper}$", 
                             ylabel_bot = r"mag$_{\rm NSA}$", xlabel_bot = r"mag$_{aper}$ (this work)", mag_min=11, mag_max=21,
                            tickmarks = [12,14,16,18,20], bad_tgids = None):
    """
    ext_mags, desi_mags : list or tuple of 3 arrays (one per band)
    band_names          : list of 3 band labels
    """

    fig, axes = plt.subplots(2, 3, figsize=(15, 6.15), sharex=True, sharey=False,
                             gridspec_kw={"height_ratios": [1, 3], "hspace": 0.0, "wspace": 0.15})


    if len(aper_mags[1]) != len(sga_cat):
        raise ValueError("the input sga_cat shape does not match aper_mags list shape")
    
    for i in range(3):
        add_cbar = (i == 2)  # only add cbar on last column
        show_cbar_label = (i == 2)
        make_compare_plot_panel(axes[0, i], axes[1, i], sga_cat,
                                ext_mags[i], aper_mags[i], band_names[i],
                                ylabel_top=(ylabel_top if i == 0 else None),
                                ylabel_bot=(ylabel_bot if i == 0 else None),
                                xlabel_bot = xlabel_bot,
                                add_cbar=add_cbar, show_cbar_label=show_cbar_label,mag_min=mag_min, mag_max=mag_max, tickmarks = tickmarks, 
                               bad_tgids=bad_tgids)

        axes[1, i].set_box_aspect(1)
    
    if save_path is not None:
        plt.savefig(save_path, bbox_inches="tight", dpi=150)
    
    if show_plot:
        plt.show()
    else:
        plt.close(fig)






# def make_compare_plot(ext_mag, desi_mag, ylabel_top =r"mag$_{\rm NSA}$- mag$_{\rm aper}$", 
#                       ylabel_bot = r"mag$_{\rm NSA}$", save_path = None, title = None,band_name=None,show_plot=False):

#     delta_mag = ext_mag -  desi_mag # Difference
    
#     # Set up figure with two panels
#     fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(5, 7), sharex=True,
#                                          gridspec_kw={"height_ratios": [1, 3], "hspace": 0.1})
    

#     if title is not None:
#         ax_top.set_title(title,fontsize = 15)
        
#     im = ax_top.hist2d( desi_mag, delta_mag, range=((11, 21), (-1.5, 1.5)), bins=(100,int(100/4)),
#                   norm=LogNorm(vmin=1,vmax=200), cmap=cmr.dusk_r)
    
#     ax_top.axhline(0, color="grey", ls="--", lw=1)
#     ax_top.set_ylabel(ylabel_top,fontsize = 14)
#     ax_top.legend(fontsize=6, loc="upper right")
#     ax_top.set_ylim(-1.5, 1.5)
    
#     # ---------------------
#     # Bottom panel: 2D histogram
#     # ---------------------
    
#     im = ax_bot.hist2d( desi_mag, ext_mag, range=((11, 21), (11, 21)), bins=100,
#                   norm=LogNorm(vmin=1,vmax=200), cmap=cmr.dusk_r)
#     ax_bot.plot([11, 22], [11, 22], lw=1, color="grey")
#     ax_bot.plot([11,22],[11.75,22.75],ls = "--",color = "grey",lw = 0.75)
#     ax_bot.plot([11,22],[10.25,21.25],ls = "--",color = "grey",lw = 0.75)
#     ax_bot.set_xlim([12, 21])
#     ax_bot.set_ylim([12, 21])
#     ax_bot.set_ylabel(ylabel_bot,fontsize = 16)
#     ax_bot.set_xlabel(r"mag$_{\rm aper}$ (this work)",fontsize = 16)

    
#     outlier_frac = np.sum(np.abs(delta_mag) > 0.75)/len(delta_mag)

#     outlier_mask = (np.abs(delta_mag) > 0.75)
    
#     ax_bot.text(16.4, 14.5, rf"$f(|\Delta_{{\rm mag}}| > 0.75) = {outlier_frac * 100:.1f}\%$", fontsize=15)

    
    
#     delta_nmad = median_abs_deviation(delta_mag,scale="normal")
    
#     ax_bot.text(17.75,15.125,r"$\sigma_{\rm NMAD} = %.2f$"%delta_nmad,fontsize = 15)

#     ax_bot.text(18.15,15.125 + 0.625,r"$N = %d$"%int(len(ext_mag)),fontsize = 15)
    
#     ax_bot.text(14.75,18.5,f"{band_name}",weight="bold",fontsize = 20)

#     cbar_ax = fig.add_axes([0.17, 0.62, 0.3, 0.015])  # top-left corner, horizontal bar
#     cbar = plt.colorbar(im[3], cax=cbar_ax, orientation='horizontal')
#     # cbar.ax.tick_params(labelsize=12)


#     if save_path is not None:
#         plt.savefig(save_path,bbox_inches="tight",dpi=150)

#     if show_plot:
#         plt.show()
#     else:
#         plt.close()

#     return outlier_mask


def collect_sga_mags(sga_catlog, g_flag = "MAG_G_BEST", r_flag = "MAG_R_BEST", z_flag = "MAG_Z_BEST"):

    gmags_new = undo_extinct( sga_catlog[g_flag].data, sga_catlog["MW_TRANSMISSION_G"].data ) 
    rmags_new = undo_extinct( sga_catlog[r_flag].data, sga_catlog["MW_TRANSMISSION_R"].data )  
    zmags_new = undo_extinct( sga_catlog[z_flag].data, sga_catlog["MW_TRANSMISSION_Z"].data )  

    # Input arrays
    aper_mags =  [ gmags_new , rmags_new , zmags_new ]
    sga_mags = [sga_catlog["SGA_G_COG_MAG"].data,sga_catlog["SGA_R_COG_MAG"].data,sga_catlog["SGA_Z_COG_MAG"].data]
    all_ff = np.concatenate( [sga_catlog["FRACFLUX_G"].data,sga_catlog["FRACFLUX_R"].data,sga_catlog["FRACFLUX_Z"].data])
    trac_mags = [sga_catlog["MAG_G"].data,sga_catlog["MAG_R"].data,sga_catlog["MAG_Z"].data  ] 

    #need to collect the SGA sizes and the new sizes!!
    
        
    return  sga_mags, aper_mags, trac_mags, all_ff


def plot_ellipse(ax, aper_xcen, aper_ycen, scale_f, r50_pix, ba_ratio, phi_deg, edgecolor, lw, ls, alpha):
    ell = Ellipse((aper_xcen, aper_ycen), 2 * scale_f * r50_pix, 2 * scale_f * r50_pix * ba_ratio, angle=phi_deg - 90,
                  facecolor='none', edgecolor=edgecolor, lw=lw, ls=ls, alpha=alpha)
    ax[0].add_patch(ell)
    return

def draw_label_line(ax, line_start, line_y, color, ls, alpha, lw = 2):
    # line_start = 0.25 - offset
    # line_y = yloc - 0.09
    ax[0].plot([line_start -0.1, line_start+0.1], [line_y,line_y],
               color = color,ls=ls, alpha=alpha, lw = lw,
              transform=ax[0].transAxes)
    return


def plot_one_panel_model_comp(ax,targetid, new_cat, siena_cat,
                              alpha=0.75,linewidth=2,
                              sga_ls = "--",sga_color = "yellowgreen",
                              new_ls = "-", new_color = "white",
                              trac_ls = "-", trac_color = "orchid",
                              scale_f = 2, cutout_size_arcsec = 60,
                              verbose=False,use_sga_r50=True,
                              bar_size_arc=30, bar_size_str="30''", draw_color_label_line=True,
                             compare_sga=True):
    '''
    Function that plots the SGA ellipse and compares it with our reconstructed galaxy
    '''

    index = np.where(new_cat["TARGETID"] == targetid)[0][0]

    #get the 3 different photometry!!
    g_org, r_org, z_org = new_cat[index]["MAG_G"], new_cat[index]["MAG_R"], new_cat[index]["MAG_Z"]
    g_new, r_new, z_new = new_cat[index]["MAG_G_BEST"], new_cat[index]["MAG_R_BEST"], new_cat[index]["MAG_Z_BEST"]

    #get image and wcs infno
    img_path = new_cat[index]["IMAGE_PATH"]
    hdus = fits.open(img_path)
    img_data = hdus[0].data
    wcs = WCS(fits.getheader( img_path))

    if compare_sga:
        #get sga mags
        g_sga, r_sga, z_sga = new_cat[index]["SGA_G_COG_MAG"], new_cat[index]["SGA_R_COG_MAG"], new_cat[index]["SGA_Z_COG_MAG"]
        mw_ex_g,mw_ex_r,mw_ex_z = new_cat[index]["MW_TRANSMISSION_G"], new_cat[index]["MW_TRANSMISSION_R"], new_cat[index]["MW_TRANSMISSION_Z"]
        g_sga, r_sga, z_sga = correct_extinct(g_sga, mw_ex_g), correct_extinct(r_sga, mw_ex_r), correct_extinct(z_sga, mw_ex_z)

        ##GET INFO ON THE SGA APERTURE!
        sga_ra=new_cat["SGA_RA_MOMENT"].data[index]
        sga_dec=new_cat["SGA_DEC_MOMENT"].data[index]
    
        if use_sga_r50:
            sga_sma = siena_cat["R_SMA50"].data[index]/0.262 #in pixels
        else:
            sga_sma = siena_cat["SMA_SB23"].data[index]/0.262 #in pixels
    
        sga_pa = new_cat["SGA_PA"].data[index]
        sga_ba = new_cat["SGA_BA"].data[index]
    
        x_sga, y_sga,_ = wcs.all_world2pix(sga_ra,sga_dec,0,1)

        comp_name = "SGA"

    
    else:
        #comparison with NSA
        nsa_ra = siena_cat["RA"].data[index]
        nsa_dec = siena_cat["DEC"].data[index]

        r_sga = siena_cat["MAG_R_CORR"].data[index]
        
        sga_sma = siena_cat["ELPETRO_TH50_R"].data[index]/0.262 #in pixels
        sga_pa = siena_cat["ELPETRO_PHI"].data[index]
        sga_ba = siena_cat["ELPETRO_BA"].data[index]

        x_sga, y_sga,_ = wcs.all_world2pix(nsa_ra,nsa_dec,0,1)

        comp_name = "NSA"


    #get the tractor source model!
    file_path = new_cat[index]["FILE_PATH"]
    
    ##get the final,updated ra,dec of the galaxy
    ra_new, dec_new = new_cat["RA"].data[index], new_cat["DEC"].data[index]
    zred = new_cat["Z"].data[index]
    
    ##INFO ON NEW APERTURE
    aper_xcen, aper_ycen,_ = wcs.all_world2pix(ra_new,dec_new,0,1)
    aper_params = new_cat["SHAPE_PARAMS"].data[index]
    r50_new = new_cat["R50_R"].data[index]/0.262 #need to convert from pixels to arcseconds

    #INFO ON TRACTOR APERTURE
    ra_tgt, dec_tgt = new_cat["RA_TARGET"].data[index], new_cat["DEC_TARGET"].data[index]
    trac_xcen, trac_ycen,_ = wcs.all_world2pix(ra_tgt,dec_tgt,0,1)
    r50_trac = new_cat["SHAPE_R"].data[index] / 0.262
    if r50_trac == 0:
        r50_trac = 0.75/0.262

    trac_ba = new_cat["BA"].data[index]
    trac_phi = new_cat["PHI"].data[index]
    
    #finalize the cutout size
    cutout_size_pix  = int(cutout_size_arcsec/0.262)

    if verbose:
        print(cutout_size_pix, img_data.shape)
        
    cutout_size_pix = np.minimum(cutout_size_pix, img_data.shape[1])
   
    #get the cutout image
    rgb_img, shift = process_img(img_data, cutout_size = cutout_size_pix, org_size = np.shape(img_data)[1], return_shift=True )

    #however, once we do the rescaling of image, we need to shift the x_center, y_center
    x_sga -= shift
    y_sga -= shift

    trac_xcen -= shift
    trac_ycen -= shift

    aper_xcen -= shift
    aper_ycen -= shift

    fsize = 14
    fs_title = 13
    fs_col = "white"
    yloc = 0.97

    ax[0].text(0.75, 0.1, f"({ra_new:.3f},{dec_new:.3f})",size = 12, transform=ax[0].transAxes, va='top',ha="center",color=fs_col)

    ax[0].imshow(rgb_img, origin="lower")

    offset = 0.05
    ax[0].text(0.25 - offset-0.025,yloc,rf"$r_{{\rm DR9}}$:{r_org:.1f}",
               size = fsize,transform=ax[0].transAxes, va='top',ha="center",color = trac_color,alpha=alpha+0.2)

    ax[0].text(0.5,yloc,rf"$r_{{\rm new}}$:{r_new:.1f}",
               size = fsize,transform=ax[0].transAxes, va='top',ha="center",color = new_color,alpha=alpha+0.2)

    ax[0].text(0.75 + offset+0.025,yloc,rf"$r_{{\rm {comp_name}}}$:{r_sga:.1f}",
           size = fsize,transform=ax[0].transAxes, va='top',ha="center",color = sga_color,alpha=alpha+0.2)

        
    #draw a line right underneath this to show
    if draw_color_label_line:
        draw_label_line(ax, 0.25 - offset, yloc - 0.09, trac_color, trac_ls, alpha,lw = linewidth + 0.5)
        draw_label_line(ax, 0.5 , yloc - 0.09, new_color, new_ls, alpha, lw = linewidth)
        draw_label_line(ax, 0.75 + offset, yloc - 0.09, sga_color, sga_ls, alpha, lw = linewidth - 0.5)
        

    ##plot the SGA aperture
    plot_ellipse(ax, x_sga, y_sga, scale_f, sga_sma, sga_ba, sga_pa, sga_color, linewidth-0.5, sga_ls, alpha)

    ##plot the new aperture
    plot_ellipse(ax, aper_xcen, aper_ycen, scale_f, r50_new, aper_params[0], aper_params[1], new_color, linewidth, new_ls, alpha=1)

    #plot the tractor aperture
    plot_ellipse(ax, trac_xcen, trac_ycen, scale_f, r50_trac, trac_ba, trac_phi, trac_color, linewidth+0.5, trac_ls, alpha)

    for axi in ax:
        axi.set_xticks([])
        axi.set_yticks([])
        axi.set_xlim([0,np.shape(rgb_img)[1]])
        axi.set_ylim([0,np.shape(rgb_img)[1]])
        
    #plot the horizontal bar plotting 30''

    fs=12
    bar_size = bar_size_arc/0.262
    bar_start = 0.07*np.shape(rgb_img)[0]
    y_bar = 0.07*np.shape(rgb_img)[0]
    
    ax[0].plot([bar_start, bar_start+bar_size], [y_bar,y_bar],color = "white",lw = 1)

    if bar_size_str is not None:
        ax[0].text( bar_start + 0.55*bar_size, y_bar*1.15, f"{bar_size_str}",fontsize = fs,color = "white",ha="center"  )
            
    return ax
    

c_light = 300000

def select_close_zreds(zreds_1, zreds_2):
    '''
    function that constructs a mask for objects that have close redshifts (within 500 km/s?)
    '''
    close_mask = (np.abs(zreds_1*c_light - zreds_2*c_light) < 1000)
    return close_mask

    
def get_nsa_matching(nsa_cat, desi_cat):
    
    idx_nsa, d2d_nsa,_ = match_c_to_catalog(c_cat = desi_cat, catalog_cat = nsa_cat)

    #instead of being distance, let the matching be within the Petrosian radius!!

    nsa_matching_th50 = nsa_cat["ELPETRO_TH50_R"][idx_nsa].data

    dist_mask = (d2d_nsa.arcsec < 2*nsa_matching_th50)
    
    desi_nsa_match = desi_cat[dist_mask]
    nsa_match = nsa_cat[idx_nsa][dist_mask]

    print(len(nsa_match))

    #we want to further select by objects that are at the same redshift

    close_mask = select_close_zreds(  nsa_match["Z"].data, desi_nsa_match["Z"].data )
    
    desi_nsa_match = desi_nsa_match[close_mask]
    nsa_match = nsa_match[close_mask]
    
    good_flux = (nsa_match["ELPETRO_FLUX"][:,3].data > 0) & (nsa_match["ELPETRO_FLUX"][:,4].data > 0) & (nsa_match["ELPETRO_FLUX"][:,6].data > 0)

    desi_nsa_match = desi_nsa_match[good_flux]
    nsa_match = nsa_match[good_flux]
    
    print(len(nsa_match))

    #obtaining the NSA magnitudes to compare with
    nsa_petro_gflux = nsa_match["ELPETRO_FLUX"][:,3].data
    nsa_petro_rflux = nsa_match["ELPETRO_FLUX"][:,4].data
    nsa_petro_zflux = nsa_match["ELPETRO_FLUX"][:,6].data


    
    
    nsa_petro_gmag = 22.5 - 2.5*np.log10(nsa_petro_gflux)
    nsa_petro_rmag = 22.5 - 2.5*np.log10(nsa_petro_rflux)
    nsa_petro_zmag = 22.5 - 2.5*np.log10(nsa_petro_zflux)

    #obtaining the DESI mags to compare with
    #let us undo the extinction in MW!
    desi_nsa_match_g = undo_extinct( desi_nsa_match["MAG_G_BEST"].data,  desi_nsa_match["MW_TRANSMISSION_G"].data)
    desi_nsa_match_r = undo_extinct( desi_nsa_match["MAG_R_BEST"].data,  desi_nsa_match["MW_TRANSMISSION_R"].data)
    desi_nsa_match_z = undo_extinct( desi_nsa_match["MAG_Z_BEST"].data,  desi_nsa_match["MW_TRANSMISSION_Z"].data)
    
    nsa_corr_gmag = undo_extinct( nsa_petro_gmag,  desi_nsa_match["MW_TRANSMISSION_G"].data)
    nsa_corr_rmag = undo_extinct( nsa_petro_rmag,  desi_nsa_match["MW_TRANSMISSION_R"].data)
    nsa_corr_zmag = undo_extinct( nsa_petro_zmag,  desi_nsa_match["MW_TRANSMISSION_Z"].data)
    
    nsa_match["MAG_G_CORR"] = nsa_corr_gmag
    nsa_match["MAG_R_CORR"] = nsa_corr_rmag
    nsa_match["MAG_Z_CORR"] = nsa_corr_zmag
    
    nsa_petro_mag = [nsa_petro_gmag, nsa_petro_rmag, nsa_petro_zmag]
    best_mag = [ desi_nsa_match_g , desi_nsa_match_r , desi_nsa_match_z ]

    ##also return the trac mags and fracflux values!

    trac_mag_g = desi_nsa_match["MAG_G"].data
    trac_mag_r = desi_nsa_match["MAG_R"].data
    trac_mag_z = desi_nsa_match["MAG_Z"].data

    ff_g = desi_nsa_match["FRACFLUX_G"].data
    ff_r = desi_nsa_match["FRACFLUX_R"].data
    ff_z = desi_nsa_match["FRACFLUX_Z"].data

    trac_mag = [trac_mag_g, trac_mag_r, trac_mag_z]
    all_ff = np.concatenate( [ff_g, ff_r, ff_z] )

    
    return nsa_petro_mag, best_mag, trac_mag, all_ff, desi_nsa_match, nsa_match

    

def get_all_ff_deltas(sample_name):
    cat_shred = Table.read(f"/pscratch/sd/v/virajvm/catalog_dr1_dwarfs/iron_photometry/iron_{sample_name}_shreds_catalog_w_aper_mags.fits")
    cat_clean = Table.read(f"/pscratch/sd/v/virajvm/catalog_dr1_dwarfs/iron_photometry/iron_{sample_name}_clean_catalog_w_aper_mags.fits")

    #consolidate it
    cat_shred = consolidate_new_photo(cat_shred, sample=sample_name, add_pcnn=False,flag_cog_nan_always=False)
    cat_clean = consolidate_new_photo(cat_clean, sample=sample_name, add_pcnn=False,flag_cog_nan_always=False)
    
    cat_shred = cat_shred[cat_shred["PHOTO_MASKBIT"] == 0]
    cat_clean = cat_clean[cat_clean["PHOTO_MASKBIT"] == 0]

    all_ffs = []
    all_mag_new = []    
    all_mag_trac = []    
    

    for bi in "GRZ":
        
        mag_new_shred = cat_shred[f"MAG_{bi}_BEST"].data
        mag_trac_shred = cat_shred[f"MAG_{bi}"].data
        
        ff_shred = cat_shred[f"FRACFLUX_{bi}"]

        mag_new_clean = cat_clean[f"MAG_{bi}_BEST"].data
        mag_trac_clean = cat_clean[f"MAG_{bi}"].data
    
        ff_clean = cat_clean[f"FRACFLUX_{bi}"]

        all_ffs.append(ff_shred)
        all_ffs.append(ff_clean)

        all_mag_new.append(mag_new_shred)
        all_mag_new.append(mag_new_clean)

        all_mag_trac.append(mag_trac_shred)
        all_mag_trac.append(mag_trac_clean)
                
    return np.concatenate(all_ffs), np.concatenate(all_mag_new), np.concatenate(all_mag_trac)
    



def get_fracflux_change_curve(ff_vals, new_mags, trac_mags, mag_cut=0.75, cumulative=True):
    """
    Returns the fraction of objects in bins (or cumulatively) of fracflux 
    that have significantly different magnitudes.

    Parameters
    ----------
    ff_vals : array
        FRACFLUX values for all objects
    new_mags, trac_mags : arrays
        Magnitudes to compare
    mag_cut : float
        Threshold for |Δmag| to count as 'different'
    cumulative : bool
        If True, compute cumulative fraction above each fracflux threshold.
        If False, compute fraction in bins.
    """
    delta_vals = np.abs(new_mags - trac_mags)
    bins = np.logspace(-2.1, np.log10(7), 20)
    bin_centers = np.sqrt(bins[:-1] * bins[1:])

    frac_bad = []

    if cumulative:
        # For each threshold (bin edge), compute cumulative fraction
        for thr in bins[:-1]:
            mask = ff_vals <= thr
            
            frac_bad.append(np.mean(delta_vals[mask] > mag_cut))
    else:
        # Digitize into bins and compute within-bin fraction
        bin_idx = np.digitize(ff_vals, bins)
        for i in range(1, len(bins)):
            in_bin = bin_idx == i
            if np.sum(in_bin) < 100:
                frac_bad.append(np.nan)
            else:
                frac_bad.append(np.mean(delta_vals[in_bin] > mag_cut))

    return bin_centers, frac_bad





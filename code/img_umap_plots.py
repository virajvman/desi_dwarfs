
import os
import sys

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
import cmasher as cmr
from astropy.cosmology import Planck18
from desi_lowz_funcs import print_stage
from mass_and_photo_corrections import DWARF_CATALOG_SPEC_HDU
from sklearn.preprocessing import StandardScaler

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

from desi_lowz_funcs import make_subplots


def halpha_flux_to_lumi(zreds, ha_flux):
    '''
    Function that converts redshift and observed Halpha flux into Halpha luminosity!
    '''
    lumi_dist_in_cm = Planck18.luminosity_distance(zreds).to(u.cm).value
    ha_lumi = ha_flux * 1e-17 * 4 * np.pi * (lumi_dist_in_cm)**2
    ##this is in units of ergs/s
    return ha_lumi

def get_line_ratios_snr(iron_main):
    '''
    Function that returns table with added SNR columns useful for BPT diagrams
    '''
    sii_all_val = np.array(iron_main["SII_6716_FLUX"]) + np.array(iron_main["SII_6731_FLUX"])

    sii_6716_sig = np.sqrt(1/iron_main["SII_6716_FLUX_IVAR"])
    sii_6731_sig = np.sqrt(1/iron_main["SII_6731_FLUX_IVAR"])

    sii_all_sig = np.sqrt( sii_6716_sig**2 + sii_6731_sig**2 )
    
    iron_main["SII_ALL_FLUX"] = sii_all_val
    
    #computing the line snr
    hbeta_snr = iron_main["HBETA_FLUX"].data * np.sqrt( iron_main["HBETA_FLUX_IVAR"])
    halpha_snr = iron_main["HALPHA_FLUX"].data * np.sqrt( iron_main["HALPHA_FLUX_IVAR"])

    oiii_snr = iron_main["OIII_5007_FLUX"].data * np.sqrt( iron_main["OIII_5007_FLUX_IVAR"])

    nii_snr = iron_main["NII_6584_FLUX"].data * np.sqrt( iron_main["NII_6584_FLUX_IVAR"])
    
    sii_snr = sii_all_val / sii_all_sig

    iron_main["NII_6584_SNR"] = nii_snr
    iron_main["HBETA_SNR"] = hbeta_snr
    iron_main["HALPHA_SNR"] = halpha_snr
    iron_main["OIII_5007_SNR"] = oiii_snr
    iron_main["SII_ALL_SNR"] = sii_snr
    
    return iron_main


def make_umap_plot(fig, ax, spec_temp_cat, mask, quant, 
                  n_bins=150, limits = None,
                  cmap = cmr.chroma,scatter=False, 
                  cb_label = r"$\log\mathrm{[OIII]}/\mathrm{H}\beta$", cb_size = 12, cb_padding=4, cb_pos = [0,1,0.33,0.02]):

    spec_temp_cat = spec_temp_cat[mask]
    quant = quant[mask]

    embedding_x = spec_temp_cat["SPEC_UMAP_0"]
    embedding_y = spec_temp_cat["SPEC_UMAP_1"]
    
    counts, _, _ = np.histogram2d(embedding_x, embedding_y, bins=n_bins)
    
    hist_2, xedges, yedges = np.histogram2d(embedding_x, embedding_y, bins=n_bins, weights = quant) 

    averaged_2 = hist_2/counts

    if limits is None:
        vmin_2 = np.percentile(quant, 2.3 )
        vmax_2 = np.percentile(quant, 97.7)
    else:
        vmin_2 = limits[0]
        vmax_2 = limits[1]
        
    print(f"Plotting limits = {vmin_2}, {vmax_2}")

    if scatter:
        samp_freq = 10
        sc = ax.scatter(embedding_x[::samp_freq], embedding_y[::samp_freq],c= quant[::samp_freq], cmap=cmr.cosmic,vmin=vmin_2,vmax=vmax_2,s=0.5,rasterized=True)

    else:
        sc = ax.pcolormesh(xedges, yedges, averaged_2.T, shading='auto', cmap=cmap,vmin=vmin_2,vmax=vmax_2,rasterized=True)

     # --- Colorbar above the axes, centered ---
    # get the axis position in figure coordinates

    # compute left position so that colorbar is centered
    
    # create a new axes for colorbar
    cax = fig.add_axes(cb_pos)
    cb = fig.colorbar(sc, cax=cax, orientation='horizontal')
    cb.ax.xaxis.set_ticks_position('top')
    cb.ax.xaxis.set_label_position('bottom')
    cb.set_label(cb_label, fontsize=cb_size, labelpad=cb_padding)
    
    ax.set_xlim([4,15])
    ax.set_ylim([-1,10])
    
    ax.set_xticks([])
    ax.set_yticks([])

    # Remove all spines (the box around the plot)
    for spine in ax.spines.values():
        spine.set_visible(False)
        

    return



if __name__ == '__main__':

    filename = "/pscratch/sd/v/virajvm/desi_dwarf_catalogs/dr1/v1.0/desi_dr1_dwarf_catalog.fits"

    # load the MAIN extension directly as an Astropy Table
    data_cat = Table.read(filename, hdu="MAIN")
    fspec_cat = Table.read(filename, hdu=DWARF_CATALOG_SPEC_HDU)
    spec_temp_cat = Table.read(filename, hdu="SPECTRA_TEMPLATE")

    keep_mask = spec_temp_cat["SPEC_UMAP_0"] > -50

    spec_temp_cat = spec_temp_cat[keep_mask]
    data_cat = data_cat[keep_mask]
    fspec_cat = fspec_cat[keep_mask]
        
    # Calculate the 2D histogram, where 'umap_embedding[:, 0]' is x-axis and 'umap_embedding[:, 1]' is y-axis
    # 'Y' is the second parameter for averaging

    if False:
        def get_edges_counts(spec_cat,mask):
            n_bins = 200
            temp = spec_cat[mask]
            hist, xedges, yedges = np.histogram2d(temp["SPEC_UMAP_0"], temp["SPEC_UMAP_1"], bins=n_bins) 
            # Step 3: Calculate the number of points in each bin
            counts, _, _ = np.histogram2d(temp["SPEC_UMAP_0"], temp["SPEC_UMAP_1"], bins=n_bins)
            return xedges, yedges, counts
    
        elg_mask = (data_cat["SAMPLE"] == "ELG")
        no_elg_mask = (data_cat["SAMPLE"] != "ELG")
        
        xedges_elg, yedges_elg, counts_elg = get_edges_counts(spec_temp_cat,elg_mask)
        xedges_noelg, yedges_noelg, counts_noelg = get_edges_counts(spec_temp_cat,no_elg_mask)
        
        ax = make_subplots(ncol=1,nrow=2,plot_size=4,row_spacing=0.5)
        
        ax[0].pcolormesh(xedges_elg, yedges_elg, counts_elg.T, shading='auto', cmap='Blues',norm=LogNorm(),rasterized=True)
        ax[1].pcolormesh(xedges_noelg, yedges_noelg, counts_noelg.T, shading='auto', cmap='Oranges',norm=LogNorm(),rasterized=True)
    
        ax[0].text(0.075,0.075,"ELG",transform=ax[0].transAxes,fontsize = 17)
        ax[1].text(0.075,0.075,"BGS & LOWZ",transform=ax[1].transAxes,fontsize = 17)
        
        for axi in ax:
            axi.set_xlim([2.75,15])
            axi.set_ylim([-1,10])
            ax[0].set_xlabel(r"SPEC_UMAP_0",fontsize = 15)
            axi.set_ylabel(r"SPEC_UMAP_1",fontsize = 15)
    
        plt.savefig("/pscratch/sd/v/virajvm/catalog_dr1_dwarfs/iron_spectra/plots/umap_spectra_sample.pdf",bbox_inches="tight")
        plt.close()


    ##make a multi-panel umap plot color-coded by nmf_fit resid, redshift, oiii/oii, sii/halpha

    fig,ax = make_subplots(ncol=4,nrow=1,plot_size=4,return_fig=True,col_spacing = 0.25)
    
    nbins = 150

    nnmf_mask = np.ones(len(spec_temp_cat),dtype=bool)
    nnmf_rnorm = spec_temp_cat["NNMF_RESID"].data

    bar_size = 0.2
    offset_bar = 0.33
    
    # [left, bottom, width, height] in figure coord
    cb_size = 16
    cb_height = 0.95
    
    make_umap_plot(fig,ax[0], spec_temp_cat, nnmf_mask, nnmf_rnorm, 
                    n_bins=nbins, limits = None,
                  cmap = "Reds",scatter=False, 
                  cb_label = r"NMF Fit Residaul", cb_size = cb_size,cb_pos = 0)

    ####
    
    make_umap_plot(fig,ax[1], spec_temp_cat, nnmf_mask, data_cat["Z"].data, 
                n_bins=nbins, limits = [0.01,0.3],
              cmap = cmr.lilac,scatter=False, 
              cb_label = r"Redshift", cb_size = cb_size,cb_pos = [0.08+ offset_bar,cb_height,bar_size,0.02])


    ####
    
    make_umap_plot(fig,ax[2], spec_temp_cat, snr_mask, oiii_oii_ratio, 
                n_bins=nbins, limits = None,
              cmap = cmr.sapphire,scatter=False, 
              cb_label = r"log([OIII]/[OII])", cb_size = cb_size,cb_pos = [0.08+ 2*offset_bar,cb_height,bar_size,0.02])


    make_umap_plot(fig,ax[3], spec_temp_cat, snr_mask, sii_ha_ratio, 
                n_bins=nbins, limits = None,
              cmap = cmr.rainforest,scatter=False, 
              cb_label = r"log([SII]/[H$\alpha$])", cb_size = cb_size,cb_pos = [0.08+ 3*offset_bar,cb_height,bar_size,0.02])

    plt.savefig(f"/pscratch/sd/v/virajvm/catalog_dr1_dwarfs/iron_spectra/plots/umap_img_multi_panel.pdf",bbox_inches="tight")
    plt.close()


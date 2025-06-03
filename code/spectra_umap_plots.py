
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


def make_umap_plot(embedding_x, embedding_y, quant, 
                   figsize = (5,5),n_bins=150, limits = None,
                  cmap = cmr.chroma,scatter=False, 
                  cb_label = r"$\log\mathrm{[OIII]}/\mathrm{H}\beta$", cb_size = 12, cb_padding = 20, cb_position = [0.7, 0.7, 0.2, 0.02],
                  file_end = "bpt2",dpi=150):
    
    fig_2, ax_2 = plt.subplots(1,1,figsize = figsize) 

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
        sc = ax_2.scatter(embedding_x[::samp_freq], embedding_y[::samp_freq],c= quant[::samp_freq], cmap=cmr.cosmic,vmin=vmin_2,vmax=vmax_2,s=0.5)

    else:
        sc = ax_2.pcolormesh(xedges, yedges, averaged_2.T, shading='auto', cmap=cmap,vmin=vmin_2,vmax=vmax_2)

    #colorbar stuff
    cbar_ax = fig_2.add_axes(cb_position)  # [left, bottom, width, height] in figure coords
    
    cb = fig_2.colorbar(sc, cax=cbar_ax, orientation='horizontal')
    cb.ax.xaxis.set_ticks_position('top')
    cb.ax.xaxis.set_label_position('bottom')
    
    cb.set_label(cb_label,fontsize = cb_size, labelpad = cb_padding)
    
    ax_2.set_xlim([4.25, 14.25])
    ax_2.set_ylim([-2, 12])  
    
    ax_2.set_xticks([])
    ax_2.set_yticks([])

    # Remove all spines (the box around the plot)
    for spine in ax_2.spines.values():
        spine.set_visible(False)
    fig_2.savefig(f"/pscratch/sd/v/virajvm/catalog_dr1_dwarfs/iron_spectra/plots/umap_spectra_{file_end}.png",bbox_inches="tight",dpi=dpi)
    plt.close(fig_2)

    return


def make_umap_sample_plot(embedding_x, embedding_y, sample, 
                   figsize = (5,5),n_bins=150, limits = None,
                  cmap = cmr.chroma,scatter=False, 
                  cb_label = r"$\log\mathrm{[OIII]}/\mathrm{H}\beta$", cb_size = 12, cb_padding = 20, cb_position = [0.7, 0.7, 0.2, 0.02],dpi=150):

    '''
    Sample can be BGS_BRIGHT, BGS_FAINT, LOWZ, or ELG. We will make a panel of 4 density plot 
    '''
    


    titles = [r"BGS Bright, BGS Faint, LOWZ", r"ELG"]
    
    for i in range(2):
        
        # fig, ax = make_subplots(ncol = 1,nrow = 1,return_fig=True, row_spacing = 0.5)
        fig, ax = plt.subplots(1,1,figsize = figsize) 
        
        #we loop through each sample!
        if i == 0:
            sample_mask = (sample == b"BGS_BRIGHT") | (sample == b"BGS_FAINT") | (sample == b"LOWZ")

        else:
            sample_mask = (sample == b"ELG")
            
        counts, xedges, yedges = np.histogram2d(embedding_x[sample_mask], embedding_y[sample_mask], bins=n_bins)

        #let us normalize these counts so that they add to 1
        counts = counts/np.sum(counts)

        if i == 0:
            norm = LogNorm()
        else:
            #using same min/max as before!
            norm = LogNorm(vmin = norm.vmin, vmax=norm.vmax)
        
        sc = ax.pcolormesh(xedges, yedges, counts.T, shading='auto', cmap="Greys",norm=norm)

        ax.set_xlim([4.25, 14.25])
        ax.set_ylim([-2, 12])  
        
        ax.set_xticks([])
        ax.set_yticks([])

        ax.set_title(titles[i],fontsize = cb_size)

        for spine in ax.spines.values():
            spine.set_visible(False)
        
        fig.savefig(f"/pscratch/sd/v/virajvm/catalog_dr1_dwarfs/iron_spectra/plots/umap_spectra_sample_{i}.png",bbox_inches="tight",dpi=dpi)
        plt.close(fig)

    return



if __name__ == '__main__':

    #we load this just to make sure the mapping is done correctly!
    save_path = "/pscratch/sd/v/virajvm/catalog_dr1_dwarfs/iron_spectra/spectra_files/desi_y1_dwarf_combine_nnmf_result.h5"
    with h5py.File(save_path, "r") as f:
        tgids = f["TARGETID"][:]
        zreds = f["Z"][:]
        nnmf_rnorm = f["NNMF_RNORM"][:]    

    #let us order these by targetids just like we do in spectra_anomaly script so they are in same order as 
    zreds =  zreds[np.argsort(tgids)]
    nnmf_rnorm =  nnmf_rnorm[np.argsort(tgids)]
    tgids =  tgids[np.argsort(tgids)]
        
    embedding = np.load("/pscratch/sd/v/virajvm/catalog_dr1_dwarfs/iron_spectra/spectra_files/desi_dwarfs_umap_nnmf_and_pca_v2.npy")

    fastspec_table_ordered = Table.read("/pscratch/sd/v/virajvm/catalog_dr1_dwarfs/iron_spectra/spectra_files/desi_y1_dwarf_combine_fastspec_cols.fits")
    print(f"Number of objects in fspec catalog = {len(fastspec_table_ordered)}")
    ##chdcking that the targetids line up
    print( f"TGID max difference = {np.max( np.abs( tgids - fastspec_table_ordered['TARGETID'].data  ) )}"  )

    print(f"UMAP embedding shape = {embedding.shape}")

    
    data_cat = Table.read("/pscratch/sd/v/virajvm/catalog_dr1_dwarfs/desi_y1_dwarf_combine_catalog.fits")
    #we need to confirm only unique tragetids and select ones that are in this catalog. Some objects are probably removed because they were removed in the NMF analysis
    tgids_cat_unique, tgid_cat_unique_inds = np.unique(data_cat["TARGETID"].data, return_index=True)
    ## find the inds from tgids_cat_unique that appear in tgids_unique
    isin_mask = np.isin(tgids_cat_unique, tgids)
    #we will use the combination of tgid_cat_unique_inds and isin_mask to get the table we finally need!
    data_cat = data_cat[tgid_cat_unique_inds][isin_mask]
    
    print(len(data_cat["TARGETID"].data))
    print(len(np.unique(data_cat["TARGETID"].data)))

    #check that the matching is successful!!
    print( f"MaxAbs difference between catalog tgid and nnmf tgid = {np.max( np.abs( data_cat['TARGETID'].data - tgids) )}" )
    print( f"MaxAbs difference between catalog zred and nnmf zred = {np.max( np.abs( data_cat['Z'].data - zreds) )}" )

    print(len(data_cat), embedding.shape )

    ### Making the UMAP plots!!

    from brokenaxes import brokenaxes
    import matplotlib.pyplot as plt

    # Adjust as needed: [start, break_point], [cluster_start, cluster_end]
    # bax = brokenaxes(xlims=((embedding[:,0].min() - 0.5, 5.5), (11, embedding[:,0].max() + 0.5)), hspace=.05)
    bax = brokenaxes(xlims=((embedding[:,0].min() - 0.5, -0.25), (4.25, embedding[:,0].max() + 0.25)), hspace=.05)
    
    n_bins = 150
    
    # Calculate the 2D histogram, where 'umap_embedding[:, 0]' is x-axis and 'umap_embedding[:, 1]' is y-axis
    # 'Y' is the second parameter for averaging
    hist, xedges, yedges = np.histogram2d(embedding[:, 0], embedding[:, 1], bins=n_bins) 
    
    # Step 3: Calculate the number of points in each bin
    counts, _, _ = np.histogram2d(embedding[:, 0], embedding[:, 1], bins=n_bins)

    bax.set_title("UMAP of Spectra using 30 NMF + PCA Residual Coefficient")
    bax.pcolormesh(xedges, yedges, counts.T, shading='auto', cmap='Purples',norm=LogNorm())
    plt.savefig("/pscratch/sd/v/virajvm/catalog_dr1_dwarfs/iron_spectra/plots/umap_spectra_count_v2.png",bbox_inches="tight",dpi=150)
    plt.show()

    
    

    ## plot the same plot with other color-codings!!
    ## let us plot by SAMPLE, Halpha EW, redshift, line ratio?
    fastspec_table_ordered = get_line_ratios_snr(fastspec_table_ordered)
    
    halpha_ews = fastspec_table_ordered["HALPHA_EW"]

    bpt1_mask = (fastspec_table_ordered["SII_ALL_SNR"] > 3) & (fastspec_table_ordered["HALPHA_SNR"] > 3)
    bpt1_data = fastspec_table_ordered[bpt1_mask]
    sii_ha_ratio = bpt1_data["SII_ALL_FLUX"]/bpt1_data["HALPHA_FLUX"]

    bpt2_mask = (fastspec_table_ordered["OIII_5007_SNR"] > 3) & (fastspec_table_ordered["HBETA_SNR"] > 3)
    bpt2_data = fastspec_table_ordered[bpt2_mask]
    oiii_hb_ratio = bpt2_data["OIII_5007_FLUX"]/bpt2_data["HBETA_FLUX"]

    # [left, bottom, width, height] in figure coord
    cb_position = [0.25, 0.95, 0.5, 0.02]
    dpi = 300
    cb_padding = 4
    cb_size = 16
    
    ### HALPHA EW UMAP PLOT
    ##instead of EW, one could plot the Halpha luminosity??

    halpha_lumi = halpha_flux_to_lumi(zreds,fastspec_table_ordered["HALPHA_FLUX"].data)

    # make_umap_plot(embedding[:,0], embedding[:,1],halpha_ews, 
    #                figsize = (5,5),n_bins=150, limits = [5,150],
    #               cmap = "Blues",scatter=False, 
    #               cb_label = r"H$\alpha$ EW", cb_size = cb_size, cb_padding = cb_padding, cb_position = cb_position,
    #               file_end = "halpha_ew",dpi=dpi)

    make_umap_plot(embedding[:,0][halpha_lumi > 0], embedding[:,1][halpha_lumi > 0], np.log10(halpha_lumi[halpha_lumi > 0]), 
                   figsize = (5,5),n_bins=150, limits = None,
                  cmap = "Blues",scatter=False, 
                  cb_label = r"$\log L_{\mathrm{H}\alpha} [\mathrm{ergs/s}]$", cb_size = cb_size, cb_padding = cb_padding, cb_position = cb_position,
                  file_end = "halpha_lumi",dpi=dpi)
    
    ### BPT 1 PLOT

    make_umap_plot(embedding[:,0][bpt1_mask], embedding[:,1][bpt1_mask],np.log10(sii_ha_ratio), 
                   figsize = (5,5),n_bins=150, limits = None,
                  cmap = cmr.rainforest,scatter=False, 
                  cb_label = r"$\log\mathrm{[SII]}/\mathrm{H}\alpha$", cb_size = cb_size, cb_padding = cb_padding, cb_position = cb_position,
                  file_end = "bpt1",dpi=dpi)

    ### BPT 2 PLOT

    make_umap_plot(embedding[:,0][bpt2_mask], embedding[:,1][bpt2_mask], np.log10(oiii_hb_ratio), 
                   figsize = (5,5),n_bins=150, limits = None,
                  cmap = "inferno",scatter=False, 
                  cb_label = r"$\log\mathrm{[OIII]}/\mathrm{H}\beta$", cb_size = cb_size, cb_padding = cb_padding, cb_position = cb_position,
                  file_end = "bpt2",dpi=dpi)


    ## Reconstruction error plot
    
    make_umap_plot(embedding[:,0], embedding[:,1], nnmf_rnorm, 
                   figsize = (5,5),n_bins=150, limits = None,
                  cmap = "Reds",scatter=False, 
                  cb_label = r"NMF Fit Residaul", cb_size = cb_size, cb_padding = cb_padding, cb_position = cb_position,
                  file_end = "nmf_resids",dpi=dpi)

    
    #to get stellar mass, I will first need to load the catalog and then match by targetid
    #this is also how I will get my sample !



    ##now use the sample and stellar mass!

    make_umap_plot(embedding[:,0], embedding[:,1],data_cat["LOGM_SAGA"].data, 
                   figsize = (5,5),n_bins=150, limits = None,
                  cmap = cmr.bubblegum,scatter=False, 
                  cb_label = r"$\log M_{\rm star}$", cb_size = cb_size, cb_padding = cb_padding, cb_position = cb_position,
                  file_end = "mstar",dpi=dpi)

    make_umap_plot(embedding[:,0], embedding[:,1],data_cat["Z"].data, 
                   figsize = (5,5),n_bins=150, limits = [0.01, 0.3],
                  cmap = cmr.lilac,scatter=False, 
                  cb_label = r"Redshift", cb_size = cb_size, cb_padding = cb_padding, cb_position = cb_position,
                  file_end = "zred",dpi=dpi)


    make_umap_sample_plot(embedding[:,0], embedding[:,1],data_cat["SAMPLE"].data,
                   figsize = (5,5),n_bins=150, limits = None,
                  cmap = cmr.chroma,scatter=False, 
                  cb_label = r"$\log\mathrm{[OIII]}/\mathrm{H}\beta$", cb_size = cb_size, cb_padding = cb_padding, cb_position = [0.7, 0.7, 0.2, 0.02],dpi=dpi)


    ##ca



    

    



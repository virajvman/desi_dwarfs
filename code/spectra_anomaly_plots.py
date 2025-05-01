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

def plot_nnmf_templates(wave_rest, nnmf_temps):

    fig, ax = plt.subplots(5, 2, figsize=(20, 7.5), layout="constrained")
    
    for i in range(10):
        row, col = divmod(i, 2)
        ax[row, col].plot(wave_rest, nnmf_temps[:, i], color="mediumblue", lw=1)
        ax[row, col].set_xlim([3600, 9000])
    
        yloc = 0.3 if i == 1 else 0.8
    
        ax[row, col].text(0.87, yloc, f"NMF Template {i}", ha='center', va='center', fontsize=15,
                          transform=ax[row, col].transAxes, weight="bold")
    
        if row != 4:
            ax[row, col].set_xticks([])
        else:
            ax[row, col].set_xlabel(r"Rest-frame Wavelength $\AA$", fontsize=15)
    
    plt.savefig("/pscratch/sd/v/virajvm/catalog_dr1_dwarfs/iron_spectra/plots/nnmf_templates.pdf", bbox_inches="tight")
    plt.close()
    
    return



def nnmf_resid_plot(nnmf_rnorm):

    ax = make_subplots(ncol = 1, nrow=1)

    ax[0].hist(nnmf_rnorm,bins = 50,range = (20,100))
    ax[0].set_xlim([25,100])
    ax[0].set_yscale("log")
    ax[0].set_xlabel(r"Residual Norm of NMF Fit",fontsize = 13)
    ax[0].set_ylabel(r"N",fontsize = 13)
    
    plt.savefig("/pscratch/sd/v/virajvm/catalog_dr1_dwarfs/iron_spectra/plots/nmf_fit_rnorm.pdf",bbox_inches="tight")
    
    plt.close()

    return

def plot_pca_templates(wave_rest, templates_pca_arr):

    fig, ax = plt.subplots(10, 2, figsize=(20, 15), layout="constrained")
    
    for i in range(20):
        # For a 10x2 grid, we need to calculate the position of each plot
        row, col = divmod(i, 2)
        
        ax[row, col].plot(wave_rest, templates_pca_arr[i], color="darkorange", lw=1)
        ax[row, col].set_xlim([3600, 9000])
    
        yloc = 0.8
    
        ax[row, col].text(0.815, yloc, f"PCA Residual Template {i}", ha='center', va='center', fontsize=15,
                          transform=ax[row, col].transAxes, weight="bold")
        
        if row != 9:
            ax[row, col].set_xticks([])
        else:
            ax[row, col].set_xlabel(r"Rest-frame Wavelength $\AA$", fontsize=15)
    
    # Save or display the plot
    plt.savefig("/pscratch/sd/v/virajvm/catalog_dr1_dwarfs/iron_spectra/plots/pca_templates.pdf", bbox_inches="tight")
    plt.close()
    
    return


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



import torch
import torch.nn as nn
import torch.nn.functional as F

class PCA(nn.Module):
    '''
    This is taken from https://github.com/gngdb/pytorch-pca/blob/main/pca.py    
    '''
    def __init__(self, n_components):
        super().__init__()
        self.n_components = n_components

    @staticmethod
    def _svd_flip(u, v, u_based_decision=True):
        """
        Adjusts the signs of the singular vectors from the SVD decomposition for
        deterministic output.

        This method ensures that the output remains consistent across different
        runs.

        Args:
            u (torch.Tensor): Left singular vectors tensor.
            v (torch.Tensor): Right singular vectors tensor.
            u_based_decision (bool, optional): If True, uses the left singular
              vectors to determine the sign flipping. Defaults to True.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Adjusted left and right singular
              vectors tensors.
        """
        if u_based_decision:
            max_abs_cols = torch.argmax(torch.abs(u), dim=0)
            signs = torch.sign(u[max_abs_cols, range(u.shape[1])])
        else:
            max_abs_rows = torch.argmax(torch.abs(v), dim=1)
            signs = torch.sign(v[range(v.shape[0]), max_abs_rows])
        u *= signs
        v *= signs[:, None]
        return u, v

    @torch.no_grad()
    def fit(self, X):
        n, d = X.size()
        if self.n_components is not None:
            d = min(self.n_components, d)
        self.register_buffer("mean_", X.mean(0, keepdim=True))
        Z = X - self.mean_ # center
        U, S, Vh = torch.linalg.svd(Z, full_matrices=False)
        Vt = Vh
        U, Vt = self._svd_flip(U, Vt)
        self.register_buffer("components_", Vt[:d])
        return self

    def forward(self, X):
        return self.transform(X)

    def transform(self, X):
        assert hasattr(self, "components_"), "PCA must be fit before use."
        return torch.matmul(X - self.mean_, self.components_.t())

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

    def inverse_transform(self, Y):
        assert hasattr(self, "components_"), "PCA must be fit before use."
        return torch.matmul(Y, self.components_) + self.mean_


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
    

    ax_2.set_xlim([-5.5, 6])
    ax_2.set_xlim([-5.5, 6])  
    
    ax_2.set_xticks([])
    ax_2.set_yticks([])

    # Remove all spines (the box around the plot)
    for spine in ax_2.spines.values():
        spine.set_visible(False)
    fig_2.savefig(f"/pscratch/sd/v/virajvm/catalog_dr1_dwarfs/iron_spectra/plots/umap_spectra_{file_end}.pdf",bbox_inches="tight",dpi=dpi)
    plt.close(fig_2)

    return


def make_umap_sample_plot(embedding_x, embedding_y, sample, 
                   figsize = (5,5),n_bins=150, limits = None,
                  cmap = cmr.chroma,scatter=False, 
                  cb_label = r"$\log\mathrm{[OIII]}/\mathrm{H}\beta$", cb_size = 12, cb_padding = 20, cb_position = [0.7, 0.7, 0.2, 0.02],dpi=150):

    '''
    Sample can be BGS_BRIGHT, BGS_FAINT, LOWZ, or ELG. We will make a panel of 4 density plot 
    '''
    
    fig, ax = make_subplots(ncol = 1,nrow = 2,return_fig=True, row_spacing = 0.5)


    titles = [r"BGS Bright, BGS Faint, LOWZ", r"ELG"]
    
    for i in range(2):
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
        
        sc = ax[i].pcolormesh(xedges, yedges, counts.T, shading='auto', cmap="Greys",norm=norm)

        ax[i].set_xlim([-5.5, 6])
        ax[i].set_xlim([-5.5, 6])  
        
        ax[i].set_xticks([])
        ax[i].set_yticks([])

        ax[i].set_title(titles[i],fontsize = cb_size)

        for spine in ax[i].spines.values():
            spine.set_visible(False)
        
    fig.savefig(f"/pscratch/sd/v/virajvm/catalog_dr1_dwarfs/iron_spectra/plots/umap_spectra_sample.pdf",bbox_inches="tight",dpi=dpi)
    plt.close(fig)

    return


def single_spec_cutout(tgid):
    '''
    This a function that plots the spectra of an object with an image cutout (?) to further discuss in the anomaly detection sectionf
    '''



if __name__ == '__main__':


    run_pca = False
    run_fastspec_match = False
    compute_norm_resis = False
    run_umap = False
 
    save_path = "/pscratch/sd/v/virajvm/catalog_dr1_dwarfs/iron_spectra/spectra_files/desi_y1_dwarf_nnmf_result.h5"
    with h5py.File(save_path, "r") as f:
        tgids = f["TARGETID"][:]
        zreds = f["Z"][:]
        # wave_rest = f["WAVE_REST"][:]
        # flux_scale = f["FLUX_NORM"][:]
        # flux_ivar_scale = f["FLUX_IVAR_NORM"][:] 
        # scales = f["NORM_FACTOR"][:] 
        # nnmf_coeffs = f["NNMF_COEFFS"][:]
        # nnmf_rnorm = f["NNMF_RNORM"][:]    
        # is_valid = f["IS_VALIDATION"][:] 


    ##get the unique inds that will match with the fastspecfit catalog in the same order!

    _,unique_nnmf_inds = np.unique(tgids, return_index=True)

    tgids_unique = tgids[unique_nnmf_inds ]
    zreds_unique = zreds[unique_nnmf_inds]

    # print(f"Number of unique targets in nnmf.h5 file = {len(tgids_unique)}")

    # ### get the fastspecfit columns:
    if run_fastspec_match:
        
        line_names = ["OII_3726", "OII_3729", "HGAMMA", "OIII_4363", "HBETA", "OIII_4959", "OIII_5007", "NII_6548", "HALPHA", "NII_6584", "SII_6716", "SII_6731"]
        
        flux_and_ivar_keys = []
        for name in line_names:
            flux_and_ivar_keys.append(f"{name}_FLUX")
            flux_and_ivar_keys.append(f"{name}_FLUX_IVAR")
        
        # If you want to print the final list:
        print(flux_and_ivar_keys)
        
        columns_needed = ["TARGETID","SNR_B","SNR_R","SNR_Z","HALPHA_EW", "HALPHA_EW_IVAR"] + flux_and_ivar_keys

        from desi_lowz_funcs import get_tgids_fastspec

        fastspec_table = get_tgids_fastspec(tgids, columns_needed)

        #once we read the catalog, we first select for unique tgids and then sort them to be in same order as the nnmf targetids!

        fspec_tgids = fastspec_table[0]["TARGETID"]
        #the unique function already returns a sorted array, but we do np.argsort just to be sure
        _,unique_inds_fspec = np.unique(fspec_tgids,return_index=True)
        sort_inds_fspec = np.argsort(fspec_tgids[unique_inds_fspec])

        fastspec_table_f = Table(fastspec_table[0])

        fastspec_table_ordered = fastspec_table_f[ unique_inds_fspec][sort_inds_fspec]

        print(f"Number of unique objects in fspec catalog = {len(fastspec_table_ordered)}")
        
        #and then we confirm that this is good
        tgids_diff = np.max(np.abs(fspec_tgids[unique_inds_fspec][sort_inds_fspec] - tgids_unique))

        print(f"This should be 0 if all tgids matched perfectly = {tgids_diff}" )

        #we save this file now!!
        fastspec_table_ordered.write("/pscratch/sd/v/virajvm/catalog_dr1_dwarfs/iron_spectra/spectra_files/desi_y1_dwarf_fastspec_cols.fits",overwrite=True)
        
    else:
        #the file is already saved and thus
        fastspec_table_ordered = Table.read("/pscratch/sd/v/virajvm/catalog_dr1_dwarfs/iron_spectra/spectra_files/desi_y1_dwarf_fastspec_cols.fits")
        
        print(f"Number of unique objects in fspec catalog = {len(fastspec_table_ordered)}")



    # ### now we do the other!!

    # ## load the nnmf templates !
    # nnmf_temps = np.load("/pscratch/sd/v/virajvm/catalog_dr1_dwarfs/iron_spectra/nnmf_templates/templates_dwarfs.npy")
    # print(nnmf_temps.shape)

    # print("Creating the NMF templates plot!")
    # plot_nnmf_templates(wave_rest, nnmf_temps)

    # print("Creating the NMF residual error plot!")
    # nnmf_resid_plot(nnmf_rnorm)


    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # print(f"device={device}")

    # if run_pca:
    #     if compute_norm_resis:
    #         all_inputs = []
    #         for i in trange(flux_scale.shape[1]):
    #             all_inputs.append(  (flux_scale[:,i], flux_ivar_scale[:,i], nnmf_coeffs[i] )   )
    #         print(all_inputs[0][0].shape, all_inputs[0][1].shape, all_inputs[0][2].shape)
    #         from spectra_encoder import parallel_residual
    #         all_norm_resis = parallel_residual(all_inputs,  n_processes=64)
    #         np.save( "/pscratch/sd/v/virajvm/catalog_dr1_dwarfs/iron_spectra/spectra_files/norm_residuals_dwarfs.npy", all_norm_resis )
    #     else:
    #         all_norm_resis = np.load("/pscratch/sd/v/virajvm/catalog_dr1_dwarfs/iron_spectra/spectra_files/norm_residuals_dwarfs.npy"  )
        
    #     print(f"all_norm_resis shape = {all_norm_resis.shape}")

    #     X = torch.tensor(all_norm_resis, dtype=torch.float32)

    #     pca = PCA(n_components=20).to(device).fit(X)

    #     torch.save(pca, "/pscratch/sd/v/virajvm/catalog_dr1_dwarfs/iron_spectra/spectra_files/desi_dwarfs_pca.pt")     

    #     templates = pca.components_  
    #     templates_pca_arr = templates.cpu().numpy()
        
    # else:
    #     #we load the pre-saved PCA templates and their coefficients! 
    #     torch.serialization.add_safe_globals({'PCA': PCA})
    #     pca = torch.load("/pscratch/sd/v/virajvm/catalog_dr1_dwarfs/iron_spectra/spectra_files/desi_dwarfs_pca.pt", weights_only=False)
    
    #     templates = pca.components_  

    #     templates_pca_arr = templates.cpu().numpy()

    
    
    # print("Plotting PCA templates!")
    # plot_pca_templates(wave_rest, templates_pca_arr)

    if run_umap:
        import umap.umap_ as umap
        all_spec_feats = np.concatenate( [nnmf_coeffs, t_arr], axis = 1 )
        print(all_spec_feats.shape)
        reducer = umap.UMAP()
        scaled_t_arr = StandardScaler().fit_transform(all_spec_feats)
        print(scaled_t_arr[0])
        embedding = reducer.fit_transform(scaled_t_arr)
        np.save("/pscratch/sd/v/virajvm/catalog_dr1_dwarfs/iron_spectra/spectra_files/desi_dwarfs_umap_nnmf_and_pca.npy", embedding)
    else:

        ##loadding the 2D UMAP embedding space of the  
        embedding = np.load("/pscratch/sd/v/virajvm/catalog_dr1_dwarfs/iron_spectra/spectra_files/desi_dwarfs_umap_nnmf_and_pca.npy")


    print(f"UMAP embedding shape = f{embedding.shape}")

    ##now we narrow down the embedding coefficients to match the fastspec columns
    embedding_unique = embedding[unique_nnmf_inds]

    print(f"unique UMAP embedding shape = f{embedding_unique.shape}")
    
    ### Making the UMAP plots!!

    from brokenaxes import brokenaxes
    import matplotlib.pyplot as plt


    # Adjust as needed: [start, break_point], [cluster_start, cluster_end]
    bax = brokenaxes(xlims=((embedding[:,0].min() - 0.5, 5.5), (11, embedding[:,0].max() + 0.5)), hspace=.05)
        
    n_bins = 150
    
    # Calculate the 2D histogram, where 'umap_embedding[:, 0]' is x-axis and 'umap_embedding[:, 1]' is y-axis
    # 'Y' is the second parameter for averaging
    hist, xedges, yedges = np.histogram2d(embedding_unique[:, 0], embedding_unique[:, 1], bins=n_bins) 
    
    # Step 3: Calculate the number of points in each bin
    counts, _, _ = np.histogram2d(embedding_unique[:, 0], embedding_unique[:, 1], bins=n_bins)

    bax.set_title("UMAP of Spectra using 30 NMF + PCA Residual Coefficient")
    bax.pcolormesh(xedges, yedges, counts.T, shading='auto', cmap='Purples',norm=LogNorm())
    plt.savefig("/pscratch/sd/v/virajvm/catalog_dr1_dwarfs/iron_spectra/plots/umap_spectra_count_v2.pdf",bbox_inches="tight",dpi=150)
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

    halpha_lumi = halpha_flux_to_lumi(zreds_unique,fastspec_table_ordered["HALPHA_FLUX"].data)

    # make_umap_plot(embedding_unique[:,0], embedding_unique[:,1],halpha_ews, 
    #                figsize = (5,5),n_bins=150, limits = [5,150],
    #               cmap = "Blues",scatter=False, 
    #               cb_label = r"H$\alpha$ EW", cb_size = cb_size, cb_padding = cb_padding, cb_position = cb_position,
    #               file_end = "halpha_ew",dpi=dpi)

    make_umap_plot(embedding_unique[:,0][halpha_lumi > 0], embedding_unique[:,1][halpha_lumi > 0], np.log10(halpha_lumi[halpha_lumi > 0]), 
                   figsize = (5,5),n_bins=150, limits = None,
                  cmap = "Blues",scatter=False, 
                  cb_label = r"$\log L_{\mathrm{H}\alpha} [\mathrm{ergs/s}]$", cb_size = cb_size, cb_padding = cb_padding, cb_position = cb_position,
                  file_end = "halpha_lumi",dpi=dpi)
    
    ### BPT 1 PLOT

    make_umap_plot(embedding_unique[:,0][bpt1_mask], embedding_unique[:,1][bpt1_mask], np.log10(sii_ha_ratio), 
                   figsize = (5,5),n_bins=150, limits = None,
                  cmap = cmr.rainforest,scatter=False, 
                  cb_label = r"$\log\mathrm{[SII]}/\mathrm{H}\alpha$", cb_size = cb_size, cb_padding = cb_padding, cb_position = cb_position,
                  file_end = "bpt1",dpi=dpi)

    ### BPT 2 PLOT

    make_umap_plot(embedding_unique[:,0][bpt2_mask], embedding_unique[:,1][bpt2_mask], np.log10(oiii_hb_ratio), 
                   figsize = (5,5),n_bins=150, limits = None,
                  cmap = "inferno",scatter=False, 
                  cb_label = r"$\log\mathrm{[OIII]}/\mathrm{H}\beta$", cb_size = cb_size, cb_padding = cb_padding, cb_position = cb_position,
                  file_end = "bpt2",dpi=dpi)


    #to get stellar mass, I will first need to load the catalog and then match by targetid
    #this is also how I will get my sample !

    data_cat = Table.read("/pscratch/sd/v/virajvm/catalog_dr1_dwarfs/desi_y1_dwarf_clean_catalog_v3.fits")
    #apply same filter as we applied for the NNMF analysis
    data_cat = data_cat[ data_cat["LOGM_SAGA"] < 9 ]

    tgids_cat_unique, tgid_cat_unique_inds = np.unique(data_cat["TARGETID"].data, return_index=True)

    ## find the inds from tgids_cat_unique that appear in tgids_unique
    isin_mask = np.isin(tgids_cat_unique, tgids_unique)

    #we will use the combination of tgid_cat_unique_inds and isin_mask to get the table we finally need!
    data_cat = data_cat[tgid_cat_unique_inds][isin_mask]

    #check that the matching is successful!!
    print( f"MaxAbs difference between catalog tgid and nnmf tgid = {np.max( np.abs( data_cat['TARGETID'].data - tgids_unique) )}" )
    print( f"MaxAbs difference between catalog zred and nnmf zred = {np.max( np.abs( data_cat['Z'].data - zreds_unique) )}" )

    print(len(data_cat), embedding_unique.shape )

    ##now use the sample and stellar mass!

    make_umap_plot(embedding_unique[:,0], embedding_unique[:,1],data_cat["LOGM_SAGA"].data, 
                   figsize = (5,5),n_bins=150, limits = None,
                  cmap = cmr.bubblegum,scatter=False, 
                  cb_label = r"$\log M_{\rm star}$", cb_size = cb_size, cb_padding = cb_padding, cb_position = cb_position,
                  file_end = "mstar",dpi=dpi)


    make_umap_sample_plot(embedding_unique[:,0], embedding_unique[:,1],data_cat["SAMPLE"].data,
                   figsize = (5,5),n_bins=150, limits = None,
                  cmap = cmr.chroma,scatter=False, 
                  cb_label = r"$\log\mathrm{[OIII]}/\mathrm{H}\beta$", cb_size = cb_size, cb_padding = cb_padding, cb_position = [0.7, 0.7, 0.2, 0.02],dpi=150)

    




    



 





































    

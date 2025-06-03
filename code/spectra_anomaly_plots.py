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



def plot_nnmf_templates(wave_rest, nnmf_temps):

    fig, ax = plt.subplots(5, 2, figsize=(20, 7.5), layout="constrained")
    
    for i in range(10):
        row, col = divmod(i, 2)
        ax[row, col].plot(wave_rest, nnmf_temps[:, i], color="mediumblue", lw=1)
        ax[row, col].set_xlim([3600, 9000])
    
        yloc = 0.3 if i == 4 else 0.8
    
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

    ax[0].hist(nnmf_rnorm,bins = 50,range = (35,200))
    ax[0].set_xlim([35,200])
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





if __name__ == '__main__':


    run_pca = True
    run_fastspec_match = True
    compute_norm_resis = True
    run_umap = True
    
    save_path = "/pscratch/sd/v/virajvm/catalog_dr1_dwarfs/iron_spectra/spectra_files/desi_y1_dwarf_combine_nnmf_result.h5"
    with h5py.File(save_path, "r") as f:
        tgids = f["TARGETID"][:]
        zreds = f["Z"][:]
        wave_rest = f["WAVE_REST"][:]
        flux_scale = f["FLUX_NORM"][:]
        flux_ivar_scale = f["FLUX_IVAR_NORM"][:] 
        # scales = f["NORM_FACTOR"][:] 
        nnmf_coeffs = f["NNMF_COEFFS"][:]
        nnmf_rnorm = f["NNMF_RNORM"][:]    
        # is_valid = f["IS_VALIDATION"][:] 

    print(np.shape(tgids))
    print(np.shape(zreds))
    print(np.shape(wave_rest))
    print(np.shape(flux_scale))
    print(np.shape(nnmf_coeffs))
    
    # #to have a consistent order across everything, we will have the ordering here be the same as targetids in argsort

    sort_inds = np.argsort(tgids)
    tgids = tgids[sort_inds]
    zreds = zreds[sort_inds]
    flux_scale = flux_scale[:,sort_inds]
    flux_ivar_scale = flux_ivar_scale[:,sort_inds]
    nnmf_coeffs = nnmf_coeffs[sort_inds]
    nnmf_rnorm = nnmf_rnorm[sort_inds]
    
    ##get the unique inds that will match with the fastspecfit catalog in the same order!
    _,unique_nnmf_inds = np.unique(tgids, return_index=True)
    print(len(tgids))
    tgids_unique = tgids[unique_nnmf_inds ]
    print(len(tgids_unique))
    print(np.max(tgids - tgids_unique))

    #this shoudl already be unique and so above should be zero!
    
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
        tgids_diff = np.max(np.abs(fastspec_table_ordered["TARGETID"] - tgids_unique))

        print(f"This should be 0 if all tgids matched perfectly = {tgids_diff}" )

        #we save this file now!!
        fastspec_table_ordered.write("/pscratch/sd/v/virajvm/catalog_dr1_dwarfs/iron_spectra/spectra_files/desi_y1_dwarf_combine_fastspec_cols.fits",overwrite=True)
        
    else:
        #the file is already saved and thus
        fastspec_table_ordered = Table.read("/pscratch/sd/v/virajvm/catalog_dr1_dwarfs/iron_spectra/spectra_files/desi_y1_dwarf_combine_fastspec_cols.fits")
        
        print(f"Number of unique objects in fspec catalog = {len(fastspec_table_ordered)}")

    # ### now we do the other!!

    ## load the nnmf templates !
    nnmf_temps = np.load("/pscratch/sd/v/virajvm/catalog_dr1_dwarfs/iron_spectra/nnmf_templates/templates_dwarfs_v2.npy")
    print(nnmf_temps.shape)

    print("Creating the NMF templates plot!")
    plot_nnmf_templates(wave_rest, nnmf_temps)

    print("Creating the NMF residual error plot!")
    nnmf_resid_plot(nnmf_rnorm)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}")

    if run_pca:
        if compute_norm_resis:
            all_inputs = []
            for i in trange(flux_scale.shape[1]):
                all_inputs.append(  (flux_scale[:,i], flux_ivar_scale[:,i], nnmf_coeffs[i] )   )
            print(all_inputs[0][0].shape, all_inputs[0][1].shape, all_inputs[0][2].shape)
            from spectra_encoder import parallel_residual
            all_norm_resis = parallel_residual(all_inputs,  n_processes=128)
            np.save( "/pscratch/sd/v/virajvm/catalog_dr1_dwarfs/iron_spectra/spectra_files/norm_residuals_dwarfs_v2.npy", all_norm_resis )
        else:
            all_norm_resis = np.load("/pscratch/sd/v/virajvm/catalog_dr1_dwarfs/iron_spectra/spectra_files/norm_residuals_dwarfs_v2.npy"  )
        
        print(f"all_norm_resis shape = {all_norm_resis.shape}")

        X = torch.tensor(all_norm_resis, dtype=torch.float32)

        pca = PCA(n_components=20).to(device).fit(X)

        torch.save(pca, "/pscratch/sd/v/virajvm/catalog_dr1_dwarfs/iron_spectra/spectra_files/desi_dwarfs_pca_v2.pt")     

        templates = pca.components_  
        templates_pca_arr = templates.cpu().numpy()

        t = pca.transform(X)
        t_arr = t.cpu().numpy()
        
    else:
        #we load the pre-saved PCA templates and their coefficients! 
        torch.serialization.add_safe_globals({'PCA': PCA})
        pca = torch.load("/pscratch/sd/v/virajvm/catalog_dr1_dwarfs/iron_spectra/spectra_files/desi_dwarfs_pca_v2.pt", weights_only=False)
    
        templates = pca.components_  

        templates_pca_arr = templates.cpu().numpy()

        ##fit the templates themselves to the data!
        all_norm_resis = np.load("/pscratch/sd/v/virajvm/catalog_dr1_dwarfs/iron_spectra/spectra_files/norm_residuals_dwarfs_v2.npy"  )
        X = torch.tensor(all_norm_resis, dtype=torch.float32)
        t = pca.transform(X)
        t_arr = t.cpu().numpy()

    print(np.shape(t_arr))
    
    print("Plotting PCA templates!")
    plot_pca_templates(wave_rest, templates_pca_arr)

    if run_umap:
        import umap.umap_ as umap
        all_spec_feats = np.concatenate( [nnmf_coeffs, t_arr], axis = 1 )
        print(all_spec_feats.shape)
        reducer = umap.UMAP()
        scaled_t_arr = StandardScaler().fit_transform(all_spec_feats)
        print(scaled_t_arr[0])
        embedding = reducer.fit_transform(scaled_t_arr)
        np.save("/pscratch/sd/v/virajvm/catalog_dr1_dwarfs/iron_spectra/spectra_files/desi_dwarfs_umap_nnmf_and_pca_v2.npy", embedding)
    else:

        ##loadding the 2D UMAP embedding space of the  
        embedding = np.load("/pscratch/sd/v/virajvm/catalog_dr1_dwarfs/iron_spectra/spectra_files/desi_dwarfs_umap_nnmf_and_pca_v2.npy")


 


    



 





































    

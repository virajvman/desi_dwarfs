'''
In this script, we analyze spectra of DESI spectra using NNMF
'''

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable
# Define normalization
import os
import sys
import joblib
import random
import argparse
from astropy.table import Table, vstack, join
from tqdm import tqdm, trange
import matplotlib.patches as patches
from desi_lowz_funcs import print_stage, check_path_existence
from scipy.optimize import nnls
import cupy as cp
from desispec.interpolation import resample_flux
from nearly_nmf import nmf
import desispec.io
from desispec import coaddition  
import time

def get_wave(wavemin=3600, wavemax=10000, dloglam=1e-4):
    """
    Return logarithmic wavelength array from wavemin to wavemax step dloglam

    Args:
        wavemin: minimum wavelength
        wavemax: maximum wavelength
        dloglam: stepsize in log(wave)

    Return: wave array
    """
    n = np.log10(wavemax/wavemin) / dloglam
    wave = 10**(np.log10(wavemin) + dloglam*np.arange(n))
    return wave



def deredshift_resample_desi_spectra(all_waves, all_fluxs, all_ivar, all_zreds, wave_out = get_wave(wavemin=3600, wavemax=8000,dloglam=1e-4)):
    '''
    all_waves: the grid of DESI spectra, this is not de-redshifted wavelenghts bins, directly from data
    all_fluxs: the 2d array of the flux values for all the objects at different wavelenghts
    all_ivar: same story but for ivar!
    wave_out: the common wavelength grid we want to de-redshift all the spectra too!

    We have to do for every spectra one-by-one!    
    '''

    all_fluxs_out = []
    all_ivars_out = []

    for i in trange(len(all_fluxs)):
        waves_i =  all_waves / (1 + all_zreds[i])
        
        flux_i, ivar_i = resample_flux( wave_out,  waves_i, all_fluxs[i], ivar = all_ivar[i])
        all_fluxs_out.append(flux_i)
        all_ivars_out.append(ivar_i)
        
    return cp.array(wave_out), cp.array(all_fluxs_out), cp.array(all_ivars_out)

if __name__ == '__main__':

    rng = np.random.default_rng(42)

    ##################
    ##PART 1: Data preparation: Load,download the DESI spectra. Resample it!
    ##################

    print_stage("Loading the DESI spectra")
    
    # data_cat = Table.read("/pscratch/sd/v/virajvm/catalog_dr1_dwarfs/desi_y1_dwarf_clean_catalog_v2.fits")
    # data_cat = data_cat[(data_cat["SAMPLE"] != "ELG") & (data_cat["LOGM_SAGA"] < 9) ]

    # data_cat = data_cat[:100000]

    # # print("Total number of spectra = %d"%len(data_cat))

    # # ## SAVE THIS IN CHUNK SO LESS MEMORY INTENSIVE!!
    
    # # temp = data_cat["TARGETID","SURVEY","PROGRAM","HEALPIX","Z"][100000:150000]
    # # data_spec = desispec.io.spectra.read_spectra_parallel(temp, nproc=128, prefix='coadd', rdspec_kwargs={ "skip_hdus" : [ "EXP_FIBERMAP", "SCORES", "EXTRA_CATALOG", "MASK", "RESOLUTION"] }, specprod="iron", match_order=True)
    
    # # spec_combined = coaddition.coadd_cameras(data_spec)

    # # ##save this now!
    # # all_fluxs = spec_combined.flux
    # # all_waves = spec_combined.wave
    # # all_ivar = spec_combined.ivar

    # # np.save("/pscratch/sd/v/virajvm/catalog/fluxs_dwarf_spectra_2.npy",  all_fluxs["brz"])
    # # np.save("/pscratch/sd/v/virajvm/catalog/waves_dwarf_spectra_2.npy",  all_waves["brz"])
    # # np.save("/pscratch/sd/v/virajvm/catalog/ivar_dwarf_spectra_2.npy",  all_ivar["brz"])

    ## SELECT ONLY FOR 7000 A

    fluxs_1 = np.load("/pscratch/sd/v/virajvm/catalog/fluxs_dwarf_spectra_1.npy")
    all_waves = np.load("/pscratch/sd/v/virajvm/catalog/waves_dwarf_spectra_1.npy")
    ivars_1 = np.load("/pscratch/sd/v/virajvm/catalog/ivar_dwarf_spectra_1.npy")

    fluxs_0 = np.load("/pscratch/sd/v/virajvm/catalog/fluxs_dwarf_spectra.npy")
    all_waves = np.load("/pscratch/sd/v/virajvm/catalog/waves_dwarf_spectra.npy")
    ivars_0 = np.load("/pscratch/sd/v/virajvm/catalog/ivar_dwarf_spectra.npy")

    #combine these stacks
    print(fluxs_1.shape, fluxs_0.shape)

    all_fluxs = np.vstack( (fluxs_0, fluxs_1) )
    all_ivar = np.vstack( (ivars_0, ivars_1) )

    print(all_fluxs.shape, all_ivar.shape)

    # ################

    print_stage("De-redshifting the spectra and clipping to relevant wavelength range")
    

    # all_zreds = data_cat["Z"][:100000]

    # print(np.shape(all_waves), np.shape(all_fluxs), np.shape(all_ivar), np.shape(all_zreds))
    
    # # all_fluxs = np.load("/pscratch/sd/v/virajvm/catalog/fluxs_spectra.npy")
    # # all_waves = np.load("/pscratch/sd/v/virajvm/catalog/waves_spectra.npy")
    # # all_ivar = np.load("/pscratch/sd/v/virajvm/catalog/ivar_spectra.npy")

    # # #note that these arrays have been cupy formats
    # wave_grid, all_fluxs_f, all_ivar_f =  deredshift_resample_desi_spectra(all_waves, all_fluxs, all_ivar,all_zreds )

    # # #translate them into the appropriate shape
    # all_fluxs_f = all_fluxs_f.T
    # all_ivar_f = all_ivar_f.T

    # print(all_fluxs_f.shape)
    # print(all_ivar_f.shape)
    # print(wave_grid.shape)
    
    #save these matrices
    # cp.save("/pscratch/sd/v/virajvm/catalog/fluxs_spectra_resamp.npy", all_fluxs_f)
    # cp.save("/pscratch/sd/v/virajvm/catalog/waves_spectra_resamp.npy", wave_grid)
    # cp.save("/pscratch/sd/v/virajvm/catalog/ivar_spectra_resamp.npy", all_ivar_f)

    all_fluxs_f = cp.load("/pscratch/sd/v/virajvm/catalog/fluxs_spectra_resamp.npy")
    wave_grid = cp.load("/pscratch/sd/v/virajvm/catalog/waves_spectra_resamp.npy")
    all_ivar_f = cp.load("/pscratch/sd/v/virajvm/catalog/ivar_spectra_resamp.npy")

    print(all_fluxs_f.shape)
    print(all_ivar_f.shape)
    print(wave_grid.shape)


    all_fluxs_f = all_fluxs_f[ wave_grid < 7000, :50000]
    all_ivar_f = all_ivar_f[ wave_grid < 7000, :50000]
    wave_grid = wave_grid[wave_grid < 7000]

    print(np.diff(wave_grid)[0], np.diff(wave_grid)[-1])
    
    ##################
    ##PART 1: Figuring out the appropriate normalization for the spectra
    ##################

    print_stage("Obtaining normalization for all DESI spectra")
    

    ## Idea is to fit a single template to un-normalized data and that serves as the "average" value
    ## The coefficient of this will be the normalization factor!
    
    n_templates = 1
    
    H_shape = (n_templates, all_fluxs_f.shape[1])
    W_shape = (all_fluxs_f.shape[0], n_templates)
    
    H_start = cp.array( rng.uniform(0, 1, H_shape) )
    W_start = cp.array( np.ones(W_shape) )
    
    #obtain the template!
    H_nearly, W_nearly, chi_nearly = nmf.nearly_NMF(all_fluxs_f, all_ivar_f, H_start, W_start, n_iter=50, return_chi_2=True)

    #fit the templates to the data to get the normalization factors!
    V_X = cp.asnumpy(  np.sqrt(all_ivar_f) * all_fluxs_f )
    
    H_nnls = np.zeros((W_nearly.shape[-1], V_X.shape[-1]))

    chi2_nnls = []

    W_nearly = cp.asnumpy( W_nearly )
    all_ivar_f_np = cp.asnumpy(all_ivar_f)
    for i in range(V_X.shape[-1]):
        W = np.sqrt( all_ivar_f_np[:, i][:, None] ) * W_nearly
        H_0, _ = nnls(W, V_X[:,i])
        H_nnls[:, i] = H_0

    #get the coefficients and scaling factors
    coeffs = H_nnls.T

    print(np.max(coeffs), np.min(coeffs))
    coeffs = np.concatenate(coeffs)
    scaling_factors = 1/coeffs

    print(np.shape(all_fluxs_f), np.shape(coeffs))
    print(len(coeffs[coeffs == 0]))

    #rescale the data!!
    all_fluxs_scaled =  all_fluxs_f * cp.array(scaling_factors )
    
    all_ivars_scaled =  all_ivar_f * (1/cp.array(scaling_factors**2 ))

    ##plot this template for reference!

    fig, ax = plt.subplots(1, 1, figsize=(11, 5), layout="constrained")
    ax.plot(cp.asnumpy(wave_grid), W_nearly[:, 0], label=f"Nearly-NMF Template 1/1",color = "r",lw = 2)
    ax.set_xlim([ 3600,7000])
    plt.savefig("/pscratch/sd/v/virajvm/catalog_dr1_dwarfs/iron_spectra/average_nnmf.pdf",bbox_inches="tight")
    plt.close()

    all_fluxs_scaled = all_fluxs_scaled[ :,  (cp.array(coeffs) != 0) ]
    all_ivars_scaled = all_ivars_scaled[ :, (cp.array(coeffs) != 0) ]

    print(cp.shape(all_fluxs_scaled))

    ##################
    ##PART 2: Obtaining the spectra templates !
    ##################

    print_stage("Obtaining the NNMF templates!")
    
    ## now fit the actual templates to the data!!

    n_templates = 7
    
    H_shape = (n_templates, all_fluxs_scaled.shape[1])
    W_shape = (all_fluxs_scaled.shape[0], n_templates)
    
    H_start = cp.array( rng.uniform(0, 1, H_shape) )
    W_start = np.ones(W_shape)
    W_start = cp.array(W_start)

    H_nearly = cp.array(H_start, copy=True)
    W_nearly = cp.array(W_start, copy=True)

    
    print("Starting iteration!")
    for i in range(n_templates):
        print("iteration", i + 1)
        # t1 = time.time()
        # Earlier templates do not get fixed so will get the cumulative amount of
        # iterations to train, so we can speed this up by only templates for
        # a  less amount of iterations to get "close" to the final before 
        # doing the full number for the final templates
        n_iter = 100
        H_itr, W_itr = nmf.nearly_NMF(all_fluxs_scaled, all_ivars_scaled, H_nearly[:(i + 1), :], W_nearly[:, :(i + 1)], n_iter=n_iter)
        # t2 = time.time()
        # print(t2 - t1)
        # Place the template we train into the array for the next iteration
        H_nearly[:(i + 1), :] = H_itr
        W_nearly[:, :(i + 1)] = W_itr
            
    #optimize them all together now
    H_nearly, W_nearly, chi_nearly = nmf.nearly_NMF(all_fluxs_scaled, all_ivars_scaled, H_start, W_start, n_iter=1000, return_chi_2=True)

    ##plot all these templates for reference!

    #saving the templates themselves
    np.save("/pscratch/sd/v/virajvm/catalog/nnmf_templates_spectra.npy", cp.asnumpy(W_nearly) )
    
    fig, ax = plt.subplots(n_templates, 1, figsize=(20, 20), layout="constrained")
    
    for i in range(n_templates):
        ax[i].plot(cp.asnumpy(wave_grid), cp.asnumpy(W_nearly[:, i]), label=f"Nearly-NMF Template {i}{n_templates}",color = "r",lw = 1)
        ax[i].set_xlim([ 3600,7000  ])
        # ax[i].set_yticks([])
        if i != 9:
            ax[i].set_xticks([])
        
    plt.savefig("/pscratch/sd/v/virajvm/catalog_dr1_dwarfs/iron_spectra/all_nnmf_templates.pdf",bbox_inches="tight")
    plt.close()


    print_stage("Fitting NNMF templates to all the spectra!")
    
    #fit the templates to the data to get spectra fitting coefficients!
    V_X = cp.asnumpy( np.sqrt(all_ivars_scaled) * all_fluxs_scaled )
    
    H_nnls = np.zeros((W_nearly.shape[-1], V_X.shape[-1]))

    W_nearly = cp.asnumpy( W_nearly )
    all_ivars_scaled_np = cp.asnumpy(all_ivars_scaled)
    
    for i in range(V_X.shape[-1]):
        W = np.sqrt(all_ivars_scaled_np[:, i][:, None]) * W_nearly
        H_0, _ = nnls(W, V_X[:,i])
        H_nnls[:, i] = H_0

    #get the coefficients and scaling factors
    coeffs_spectra = H_nnls.T

    ##################
    ##PART 3: Saving the data
    ##################

    #saving all the coefficients
    np.save("/pscratch/sd/v/virajvm/catalog/nnmf_coeff_spectra.npy", coeffs_spectra)


    #saving the normalization coefficients for the spectra!
    np.save("/pscratch/sd/v/virajvm/catalog/nnmf_norm_scales_spectra.npy", scaling_factors)

    







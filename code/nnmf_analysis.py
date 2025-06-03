'''
In this script, we analyze spectra of DESI spectra using NNMF
'''

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from multiprocessing import Pool
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
import h5py

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


def _deredshift_one_spectrum(args):
    """
    Helper spectrum deredshifting function for parallel processing.
    """
    wave, flux, ivar, zred, wave_out = args
    rest_wave = wave / (1 + zred)
    flux_out, ivar_out = resample_flux(wave_out, rest_wave, flux, ivar=ivar)
    return flux_out, ivar_out


def deredshift_resample_desi_spectra(all_waves, all_fluxs, all_ivar, all_zreds,
                                     wave_out=None, ncores=4,verbose=True):
    """
    De-redshift and resample DESI spectra onto a common wavelength grid in parallel.

    Parameters
    ----------
    all_waves : array-like
        Observed-frame wavelength grid (same for all spectra).
    all_fluxs : 2D array-like
        Flux values for all spectra (n_spectra, n_pixels).
    all_ivar : 2D array-like
        Inverse variance values for all spectra (n_spectra, n_pixels).
    all_zreds : 1D array-like
        Redshifts for each spectrum (length n_spectra).
    wave_out : 1D array-like, optional
        Output rest-frame wavelength grid. If None, uses default grid.
    ncores : int
        Number of processes to use for parallelization.

    Returns
    -------
    all_fluxs_out : list of arrays
        De-redshifted and resampled fluxes.
    all_ivars_out : list of arrays
        De-redshifted and resampled inverse variances.
    """
    if wave_out is None:
        wave_out = get_wave(wavemin=3600, wavemax=9000, dloglam=1e-4)

    n = len(all_fluxs)
    inputs = [(all_waves, all_fluxs[i], all_ivar[i], all_zreds[i], wave_out) for i in range(n)]

    with Pool(processes=ncores) as pool:
        results = list(tqdm(pool.imap(_deredshift_one_spectrum, inputs), total=n,
                            desc="Deredshifting and resampling"))


    results = np.array(results)
    # unzip results
    all_fluxs_out = results[:,0]
    all_ivars_out = results[:,1]

    if verbose:
        print("flux out shape =", np.shape(all_fluxs_out))
        print("ivar out shape =", np.shape(all_ivars_out))
        print("wave out shape =", np.shape(wave_out))
        print(f"wave out limits = [{np.min(wave_out), np.max(wave_out)}]")
        
    return wave_out, all_fluxs_out, all_ivars_out



if __name__ == '__main__':

    
    rng = np.random.default_rng(42)

    ##################
    ##PART 1: Data preparation: Load,download the DESI spectra. Resample it!
    ##################
    print_stage("Loading the DESI spectra")

    save_dered = "/pscratch/sd/v/virajvm/catalog_dr1_dwarfs/iron_spectra/spectra_files/desi_y1_dwarf_combine_deredshift.h5"

    if os.path.exists(save_dered):
        
         with h5py.File(save_dered, "r") as f:
            all_tgids = f["TARGETID"][:]
            all_zreds = f["Z"][:]
            wave_rest = f["WAVE_REST"][:]
            all_fluxs_out = f["FLUX"][:]
            all_flux_ivars_out = f["FLUX_IVAR"][:]
        
    else:
        #to read the data, one can do
        with h5py.File("/pscratch/sd/v/virajvm/catalog_dr1_dwarfs/iron_spectra/spectra_files/desi_y1_dwarf_combine.h5", "r") as f:
            wave = f["WAVE"][:]
            all_flux = f["FLUX"][:] 
            all_flux_ivar = f["FLUX_IVAR"][:]  # single spectrum
            all_zreds = f["Z"][:]  # single spectrum
            all_tgids = f["TARGETID"][:]  # single spectrum
            
    
        print("wave shape", wave.shape)
        print("flux shape", all_flux.shape)
        print("flux_ivar shape", all_flux_ivar.shape)
        print("zreds shape", all_zreds.shape)
        
        # ################

        ##I should de-redshift this once and then save it!
    
        print_stage("De-redshifting the spectra and clipping to relevant wavelength range")
    
        wave_rest, all_fluxs_out, all_flux_ivars_out = deredshift_resample_desi_spectra(wave, all_flux, all_flux_ivar, all_zreds,
                                         wave_out=None, ncores=128,verbose=True)

        with h5py.File(save_dered, "w") as f:
            f.create_dataset("TARGETID", data=all_tgids, dtype='i8')
            f.create_dataset("Z", data=all_zreds, dtype='f4')
            f.create_dataset("WAVE_REST", data=wave_rest, dtype='f4')
            f.create_dataset("FLUX", data=all_fluxs_out, dtype='f4')
            f.create_dataset("FLUX_IVAR", data=all_flux_ivars_out, dtype='f4')


    print("wave rest shape", wave_rest.shape)
    print("flux out shape", all_fluxs_out.shape)
    print("flux_ivar out shape", all_flux_ivars_out.shape)
    print("zreds out shape", all_zreds.shape)


     ##let us get some random spectra to train the NNMF with!

    #translate them into the appropriate shape and to float32 for memory usage
    print(type(all_fluxs_out[0][0]))
    all_fluxs_out = all_fluxs_out.T
    all_flux_ivars_out = all_flux_ivars_out.T
    wave_rest = wave_rest

    print(np.diff(wave_rest)[0], np.diff(wave_rest)[-1])
    
    ##convert this into cupy arrays
    all_fluxs_out_cp = cp.array(all_fluxs_out)
    all_flux_ivars_out_cp = cp.array(all_flux_ivars_out)

    ##################
    ##PART 1: Figuring out the appropriate normalization for all spectra
    ##################

    print_stage("Obtaining normalization for all DESI spectra")
    
    ## Idea is to fit a single template to un-normalized data and that serves as the "average" value
    ## The coefficient of this will be the normalization factor!
    
    n_templates = 1

    #select a random subset of the spectra database to get the average spectrum! Like ~25% is a good number.
    #this array is already randomly selected and so it is fine to do this
    all_fluxs_out_cp_subsample = all_fluxs_out_cp[ :, : 0.5*(all_fluxs_out_cp.shape[1])]
    all_flux_ivars_out_cp_subsample = all_flux_ivars_out_cp[:, : 0.5*(all_fluxs_out_cp.shape[1])  ]

    print(f"Shape of original spectra array = {all_fluxs_out_cp.shape} ")
    print(f"Shape of sub-sampled spectra array for getting normalization spectra = {all_fluxs_out_cp_subsample.shape} ")
    
    H_shape = (n_templates, all_fluxs_out_cp_subsample.shape[1])
    W_shape = (all_fluxs_out_cp_subsample.shape[0], n_templates)
    
    H_start = cp.array( rng.uniform(0, 1, H_shape) )
    W_start = cp.array( np.ones(W_shape) )
    
    #obtain the template!
    H_nearly, W_nearly, chi_nearly = nmf.nearly_NMF(all_fluxs_out_cp_subsample, all_flux_ivars_out_cp_subsample, H_start, W_start, n_iter=50, return_chi_2=True)
    #W_nearly is the average spectra template

    #fit the templates to ALL the spectra data to get the normalization factors!
    V_X =  np.sqrt(all_flux_ivars_out) * all_fluxs_out
    
    H_nnls = np.zeros((W_nearly.shape[-1], V_X.shape[-1]))

    chi2_nnls = []

    W_nearly = cp.asnumpy( W_nearly )

    for i in range(V_X.shape[-1]):
        W = np.sqrt( all_flux_ivars_out[:, i][:, None] ) * W_nearly
        H_0, _ = nnls(W, V_X[:,i])
        H_nnls[:, i] = H_0

    #get the coefficients and scaling factors
    coeffs = H_nnls.T

    print(np.max(coeffs), np.min(coeffs))
    coeffs = np.concatenate(coeffs)
    scaling_factors = 1/coeffs

    #RESCALE THE ENTIRE DATASET!!!
    all_fluxs_scaled =  all_fluxs_out * scaling_factors
    
    all_ivars_scaled =  all_flux_ivars_out * (1/scaling_factors**2)

    ##plot this template for reference!

    fig, ax = plt.subplots(1, 1, figsize=(11, 5), layout="constrained")
    ax.plot(cp.asnumpy(wave_rest), W_nearly[:, 0], label=f"Nearly-NMF Template 1/1",color = "r",lw = 2)
    ax.set_xlim([ 3600,9000])
    plt.savefig("/pscratch/sd/v/virajvm/catalog_dr1_dwarfs/iron_spectra/average_nnmf.pdf",bbox_inches="tight")
    plt.close()

    #saving the templates themselves
    np.save("/pscratch/sd/v/virajvm/catalog_dr1_dwarfs/iron_spectra/nnmf_templates/normalization_templates_dwarfs_v2.npy", W_nearly)

    print(np.shape(W_nearly))
    print(cp.shape(H_nearly))

    ##################
    ##PART 1.1: Applying normalization to all the spectra and filtering for objects with coeff = 0
    ##################

    ## we do not want to choose spectra where the coefficients are zero
    print(np.shape(all_fluxs_scaled))

    all_fluxs_scaled = all_fluxs_scaled[ :,  (coeffs != 0) ]
    all_ivars_scaled = all_ivars_scaled[ :, (coeffs != 0) ]

    ##also removen the redshifts and tgids
    all_zreds = all_zreds[ coeffs != 0 ]
    all_tgids = all_tgids[ coeffs != 0 ]

    print(np.shape(all_fluxs_scaled))
    print(all_zreds.shape)

    ##and then we split this into training and validations sets

    print("Randomly shuffling the array now!")

    #we are not using the validation array for anything. This is just one way to down-sample the spectra to save memory
    N = len(all_tgids)
    indices = np.random.permutation(N)
    split = int(0.5 * N)
    
    train_inds, valid_inds = indices[:split], indices[split:]

    #training set
    all_fluxs_train = all_fluxs_scaled[:,train_inds]
    all_flux_ivars_train = all_ivars_scaled[:,train_inds]
    all_zreds_train = all_zreds[train_inds]
    all_tgids_train = all_tgids[train_inds]

    ##################
    ##PART 2: Obtaining the spectra templates !
    ##################
    
    print_stage("Obtaining the NNMF templates!")
    
    ## now fit the actual templates to the data!!
    
    n_templates = 10

    H_shape = (n_templates, all_fluxs_train.shape[1])
    W_shape = (all_fluxs_train.shape[0], n_templates)
    
    H_start = cp.array( rng.uniform(0, 1, H_shape) )
    W_start = np.ones(W_shape)
    W_start = cp.array(W_start)

    H_nearly = cp.array(H_start, copy=True)
    W_nearly = cp.array(W_start, copy=True)

    all_fluxs_train_cp = cp.array(all_fluxs_train)
    all_flux_ivars_train_cp = cp.array(all_flux_ivars_train)

    print("Starting iteration!")
    for i in range(n_templates):
        print("iteration", i + 1)
        # t1 = time.time()
        # Earlier templates do not get fixed so will get the cumulative amount of
        # iterations to train, so we can speed this up by only templates for
        # a  less amount of iterations to get "close" to the final before 
        # doing the full number for the final templates
        n_iter = 50
        H_itr, W_itr = nmf.nearly_NMF(all_fluxs_train_cp, all_flux_ivars_train_cp, H_nearly[:(i + 1), :], W_nearly[:, :(i + 1)], n_iter=n_iter)
        # t2 = time.time()
        # print(t2 - t1)
        # Place the template we train into the array for the next iteration
        H_nearly[:(i + 1), :] = H_itr
        W_nearly[:, :(i + 1)] = W_itr
            
    #optimize them all together now
    H_nearly, W_nearly, chi_nearly = nmf.nearly_NMF(all_fluxs_train_cp, all_flux_ivars_train_cp, H_start, W_start, n_iter=1000, return_chi_2=True)


    ##plot all these templates for reference!

    #saving the templates themselves
    np.save("/pscratch/sd/v/virajvm/catalog_dr1_dwarfs/iron_spectra/nnmf_templates/templates_dwarfs_v2.npy", cp.asnumpy(W_nearly) )

    fig, ax = plt.subplots(n_templates, 1, figsize=(20, 20), layout="constrained")
    
    for i in range(n_templates):
        ax[i].plot(wave_rest, cp.asnumpy(W_nearly[:, i]), label=f"Nearly-NMF Template {i}{n_templates}",color = "r",lw = 1)
        ax[i].set_xlim([ 3600,9000  ])
        
        if i != 9:
            ax[i].set_xticks([])
        
    plt.savefig("/pscratch/sd/v/virajvm/catalog_dr1_dwarfs/iron_spectra/all_nnmf_templates.pdf",bbox_inches="tight")
    plt.close()


    ##################
    ##PART 3: Fitting these NNMF templates to the entire dataset!
    ##################

    print_stage("Fitting NNMF templates to all the spectra!")
    
    # #fit the templates to the data to get spectra fitting coefficients!
    V_X =  np.sqrt(all_ivars_scaled) * all_fluxs_scaled
    
    H_nnls = np.zeros((W_nearly.shape[-1], V_X.shape[-1]))

    W_nearly = cp.asnumpy( W_nearly )

    all_rnorms = []
    
    for i in range(V_X.shape[-1]):
        W = np.sqrt(all_ivars_scaled[:, i][:, None]) * W_nearly
        H_0, rnorm = nnls(W, V_X[:,i])
        #rnorm is the 2-norm of the residual : ||Ax - b||_2 where A,b are the inputs. It is this 2-norm residual that is trying to be minimized!
        H_nnls[:, i] = H_0
        all_rnorms.append(rnorm)
        
    #get the coefficients and scaling factors
    coeffs_spectra = H_nnls.T

    print(np.shape(coeffs_spectra))

    ##now we save this! We also save an array indicating whether in validation or not

    valid_array = np.zeros(len(all_tgids))
    valid_array[valid_inds] = 1

    print(np.shape(all_tgids), np.shape(all_zreds) )
    print(np.shape(all_fluxs_scaled))
    print(np.shape(scaling_factors))
    print(np.shape(coeffs_spectra))
    
    save_final = "/pscratch/sd/v/virajvm/catalog_dr1_dwarfs/iron_spectra/spectra_files/desi_y1_dwarf_combine_nnmf_result.h5"

    #note that this will overwrite the file
    with h5py.File(save_final, "w") as f:
        f.create_dataset("TARGETID", data=all_tgids, dtype='i8')
        f.create_dataset("Z", data=all_zreds, dtype='f4')
        f.create_dataset("WAVE_REST", data=wave_rest, dtype='f4')
        #let us store the fitted coefficients
        f.create_dataset("FLUX_NORM", data=all_fluxs_scaled, dtype='f4')
        f.create_dataset("FLUX_IVAR_NORM", data=all_ivars_scaled, dtype='f4')
        f.create_dataset("NORM_FACTOR", data=scaling_factors, dtype='f4')
        f.create_dataset("NNMF_COEFFS", data= coeffs_spectra , dtype='f4')
        f.create_dataset("NNMF_RNORM", data= np.array(all_rnorms) , dtype='f4')
        f.create_dataset("IS_VALIDATION", data= valid_array)
        
        







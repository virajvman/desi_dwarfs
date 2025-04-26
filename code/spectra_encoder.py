from torchvision import transforms
import torch
import torch.nn as nn
import torch.nn.functional as F
import glob
from tqdm.notebook import tqdm
from astropy.table import Table
import multiprocessing as mp
from torch.utils.data import DataLoader, Dataset
import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt
from multiprocessing import Pool, cpu_count


import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Global var for each process
nnmf_temps = None

def init_session(templates):
    global nnmf_temps
    nnmf_temps = templates
    

def get_nnmf_fit(coeffs_i,nnmf_temps):
    return np.dot( coeffs_i, nnmf_temps.T )


def construct_residuals(args):
    '''
    In this function, we will compute the residuals from the NNMF fit and normalize the residuals with the noise!

    flux_scale_i : the scaled flux for a spectra
    flux_scale_ivar_i : the scaled flux ivar for a spectra
    nnmf_coeffs_i : the nnmf_coeffs for a spectra, will be used for the reconstruction
    nnmf_temps : The NNMF templates, this could be a global variable
    '''

    flux_scale_i,  flux_ivar_scale_i,  nnmf_coeffs_i = args
    
    fit_i = get_nnmf_fit(nnmf_coeffs_i, nnmf_temps)
    
    residual = flux_scale_i - fit_i
    
    noise_i = np.sqrt(1/flux_ivar_scale_i)  # Extract noise for this object

    #scale the residual now with the noise
    scaled_residual = residual / noise_i

    return scaled_residual


def parallel_residual(inputs, n_processes=None):
    """
    Runs get_construct_residuals in parallel.

    The input is a tuple contains (flux_scale_i, flux_ivar_scale_i, nnmf_coeff_i, nnmf_temps)!

    """

    ## load the nnmf templates !
    nnmf_temps = np.load("/pscratch/sd/v/virajvm/catalog_dr1_dwarfs/iron_spectra/nnmf_templates/templates_dwarfs.npy")
    print(nnmf_temps.shape)

    with Pool(processes=n_processes, initializer=init_session, initargs=(nnmf_temps,) ) as pool:
        resids = list(tqdm(pool.imap(construct_residuals, inputs), total=len(inputs)))

    print("Shape of residual array =",np.shape(resids) )
    
    return np.array(resids)


# Autoencoder model
class ResidualAutoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim=10):
        '''
        The input dimension is the dimension of the 1D spectra residual. So around ~4000.
        The latent dimension is the dimension we will be compressing it into. 
        '''
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, input_dim)
        )

    def forward(self, x):
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat, z




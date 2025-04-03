'''
This script downloads the DESI spectra locally to do NNMF on it!
'''

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
import desispec.io
from desispec import coaddition   

if __name__ == '__main__':
    #read the catalog
    data_cat = Table.read("/pscratch/sd/v/virajvm/catalog_dr1_dwarfs/desi_y1_dwarf_clean_catalog_v2.fits")
    #only focusing on reasonable high SNR spectra! ELGs very faint!
    data_cat = data_cat[(data_cat["SAMPLE"] != "ELG") & (data_cat["LOGM_SAGA"] < 9) ]

    print("Total number of spectra = %d"%len(data_cat))

    ## this is very fast!!!
    spec_cat = desispec.io.spectra.read_spectra_parallel(data_cat, nproc=16, prefix='coadd', rdspec_kwargs={ "skip_hdus" : [ "EXP_FIBERMAP", "SCORES", "EXTRA_CATALOG", "MASK", "RESOLUTION"] }, specprod="iron", match_order=True)

    #coadd spectra from b,r,z cameras!
    spec_cat_comb = coaddition.coadd_cameras(spec_cat)

    ##getting the flux, wavelength and ivar matrices
    all_fluxs = spec_combined.flux
    all_waves = spec_combined.wave
    all_ivar = spec_combined.ivar

    #for the NNMF spectra stuff, I want to focus on rest-frame wavelength range from 4000A to 7000A for now
    #we need to save only relevant part of the above matrices!

    


    

    

    
    


    

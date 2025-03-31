from desitarget.targetmask import desi_mask, bgs_mask
# import some helpful python packages 
import os
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
import sys
import multiprocessing as mp
from tqdm import tqdm

def is_sga_shred(input_data):
    '''
    This function returns the index of the object and the normalized distance 
    '''
    ## find the closest SGA galaxy in the sky
    ## first find all similar galaxies in redshift

    multiplier=1.0

    index = input_data["index"]
    source_ra = input_data["ra"]
    source_dec = input_data["dec"]
    source_redshift = input_data["redshift"]
    siena20_f = input_data["siena_cat"]

    # siena20_f = siena20_f[ (np.abs(source_redshift - siena20_f["Z_LEDA"]) * c_light < 1000 ) ]
    
    #the columns we need are RA/DEC_MOMENT , D26, BA, PA, Z_LEDA and nothing else!
        
    # if len(siena20_f) > 0:
    #find nearby ones in sky
    c = SkyCoord(ra= source_ra * u.degree, dec= source_dec *u.degree )
    catalog = SkyCoord(ra=siena20_f["RA_MOMENT"].data*u.degree, dec=siena20_f["DEC_MOMENT"].data*u.degree )
    d2d = c.separation(catalog)
    siena20_closest = siena20_f[ np.argmin(d2d.arcsec) ]

    #I need to determine if it is within twice the radius of this galaxy ... 
    norm_dist = calc_normalized_dist(source_ra, source_dec, siena20_closest["RA_MOMENT"], siena20_closest["DEC_MOMENT"],siena20_closest["D26"]*60*0.5, 
                         cen_ba=siena20_closest["BA"], cen_phi=siena20_closest["PA"], multiplier=multiplier)
    
    # if norm_dist <= 1:
    return index, norm_dist, siena20_closest["SGA_ID"]


if __name__ == '__main__':

    c_light = 299792 #km/s
    
     
    rootdir = '/global/u1/v/virajvm/'
    sys.path.append(os.path.join(rootdir, 'DESI2_LOWZ'))
    from desi_lowz_funcs import save_table, get_useful_cat_colms, _n_or_more_gt, _n_or_more_lt, get_remove_flag
    from desi_lowz_funcs import match_c_to_catalog, get_stellar_mass, get_stellar_mass_mia, calc_normalized_dist

    files = [ "/pscratch/sd/v/virajvm/catalog_dr1_dwarfs/desi_y1_dwarf_shreds_catalog.fits",  "/pscratch/sd/v/virajvm/catalog_dr1_dwarfs/desi_y1_dwarf_clean_catalog.fits"]


    for file_path in files:
        
        data_cat = Table.read(file_path)
        # data_cat = data_cat
    
        print(len(data_cat))
    
        siena_path = "/global/cfs/cdirs/cosmo/data/sga/2020/SGA-2020.fits"
        siena20 = Table.read(siena_path,hdu = "ELLIPSE")
    
        catalog = SkyCoord(ra=siena20["RA"].data*u.degree, dec=siena20["DEC"].data*u.degree )
        
        def get_input_dicts(i):
            #around this one galaxy, see if there are SGA galaxies within 2 degrees of it and at consistent redshift!
            c = SkyCoord(ra= data_cat["RA"][i] * u.degree, dec= data_cat["DEC"][i]*u.degree )
            seps = c.separation(catalog).deg
    
            source_redshift = data_cat["Z"][i]
            siena20_f = siena20[ (seps < 2) & (np.abs(source_redshift - siena20["Z_LEDA"]) * c_light < 1000 ) ]
    
            if len(siena20_f) > 0:
                #there is potentially something!
                temp = {"index":i, "ra": data_cat["RA"][i], "dec":data_cat["DEC"][i], "redshift":source_redshift, "siena_cat":siena20_f }
                return temp
            else:
                return
    
    
        print("Number of cores used is =",mp.cpu_count())
        
        all_ks = np.arange(len(data_cat))
    
        print(len(all_ks))
        #run with multi-pool
        with mp.Pool(processes=mp.cpu_count()) as pool:
            all_inputs = list(tqdm(pool.imap(get_input_dicts, all_ks), total = len(all_ks)  ))
    
        all_inputs = np.array(all_inputs)
        all_inputs = all_inputs[all_inputs != None]
        
        print(len(all_inputs))
    
        ### to avoid stress on the memory, we can split this up into some chunks 
        all_results = []

        nchunks = 10
        


        #now let us produce the 
        with mp.Pool(processes=mp.cpu_count()) as pool2:
            results = list(tqdm(pool2.imap(is_sga_shred, all_inputs), total = len(all_inputs)  ))
    
        results = np.array(results)
    
        inds = results[:,0].astype(int)
        norm_dists = results[:,1].astype(float)
        sga_ids = results[:,2].astype(int)
    
        all_norm_dists = np.ones(len(data_cat)) * -99
        all_sga_ids = np.ones(len(data_cat)) * -99
    
        all_norm_dists[inds] = norm_dists
        all_sga_ids[inds] = sga_ids
    
        #add this to the data cat
    
        data_cat["SGA_ID_MATCH"] = all_sga_ids
        data_cat["SGA_D26_NORM_DIST"] = all_norm_dists
    
        #the first index is the 
    
        print(len(data_cat[(data_cat["SGA_D26_NORM_DIST"] > 0) & (data_cat["SGA_D26_NORM_DIST"] < 2)]  ) )
        print(len(data_cat) )
    
        save_table(data_cat,  file_path)
    
    
    

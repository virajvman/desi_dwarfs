'''
python3 desi_dwarfs/code/download_spectra.py -random -nchunks 50 -save_name desi_y1_dwarf_combine > download_spectra.log 2>&1

'''

import numpy as np
import matplotlib.pyplot as plt
# Define normalization
import os
import random
import argparse
from astropy.table import Table
from tqdm import trange
from desi_lowz_funcs import print_stage, check_path_existence, parse_tgids
import desispec.io
from desispec import coaddition  
import h5py

def argument_parser():
    '''
    Function that parses the arguments passed while running a script
    '''
    result = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # this is the catalog file we are using to load the spectra!
    # result.add_argument('-catalog', dest='catalog', type=str, default = "") 
    result.add_argument('-min', dest='min', type=int,default = 0)
    result.add_argument('-max', dest='max', type=int,default = 500000) 
    result.add_argument('-ncores', dest='ncores', type=int,default = 64) 
    result.add_argument('-tgids',dest="tgids_list", type=parse_tgids) 
    result.add_argument('-run_parr', dest='run_parr',  action='store_true') 
    result.add_argument('-random', dest='random',  action='store_true') 
    result.add_argument('-nchunks',dest='nchunks', type=int,default = 1)
    result.add_argument('-save_name',dest='save_name', type = str, default = "spectra")

    return result

if __name__ == '__main__':

    rng = np.random.default_rng(42)

    # read in command line arguments
    args = argument_parser().parse_args()

    #sample_str could also be multiple samples together!
    # cat_path = args.catalog
    min_ind = args.min
    max_ind = args.max
    ncores = args.ncores
    tgids_list = args.tgids_list
    run_parr = args.run_parr
    random = args.random
    nchunks = args.nchunks
    save_name = args.save_name

    ##################
    ##PART 1: Data preparation: Load,download the DESI spectra. Resample it etc.
    ##################

    print_stage("Loading the DESI catalogs")

    cat_path = "/pscratch/sd/v/virajvm/catalog_dr1_dwarfs/desi_y1_dwarf_combine_catalog.fits"

    data_cat = Table.read(cat_path)

    print(f"Number of galaxies for which spectra is being downloaded = {len(data_cat)}")

    # ##add what ever filter you want to apply
    # data_cat = data_cat[ data_cat["LOGM_SAGA"] < 9.25 ]

    ##if there exists a column named TARGETID_1 in the file we rename it!
    cat_cols = data_cat.colnames
    if "TARGETID_1" in cat_cols:
        data_cat.rename_column("TARGETID_1", "TARGETID")
    
    if tgids_list is not None:
        print("List of targetids to process:",tgids_list)
        data_cat = data_cat[np.isin(data_cat['TARGETID'], np.array(tgids_list) )]
        print("Number of targetids to process =", len(data_cat))


    ##do we randomly shuffle spectra?
    if random:
        print("Randomly shuffling the array now!")
        arr_inds = np.arange( len(data_cat) )
        np.random.shuffle(arr_inds)
        data_cat = data_cat[arr_inds]
    
    #apply the max_ind cut if relevant
    # max_ind = np.minimum( max_ind, len(data_cat) )
    # data_cat = data_cat[min_ind:max_ind]

    print("Total number of spectra to download = %d"%len(data_cat))

    #only load the necessary columns
    temp = data_cat["TARGETID","SURVEY","PROGRAM","HEALPIX","Z"]
    
    ##run in chunks so less memory intensive!
    print_stage("Starting to read the spectra in parallel!")

    all_ks = np.arange(len(temp))
    print(len(temp))
    
    all_ks_chunks = np.array_split(all_ks, nchunks)

    file_template = "/pscratch/sd/v/virajvm/catalog_dr1_dwarfs/iron_spectra/spectra_files/" + save_name

    for chunk_i in trange(nchunks):
        print_stage("Started chunk %d/%d"%(chunk_i, nchunks) )
    
        all_ks_i =  all_ks_chunks[chunk_i]
        
        print_stage("Number of objects in this chunk = %d"%len(all_ks_i))

        #getting the table associated with this chunk!
        temp_chunk_i = temp[all_ks_i]

        data_spec = desispec.io.spectra.read_spectra_parallel(temp_chunk_i, nproc=ncores, prefix='coadd', rdspec_kwargs={ "skip_hdus" : [ "EXP_FIBERMAP", "SCORES", "EXTRA_CATALOG", "MASK", "RESOLUTION"] }, specprod="iron", match_order=True)
    
        #we coadd the spectra!
        spec_combined = coaddition.coadd_cameras(data_spec)
    
        # ##save this now!
        all_fluxs = spec_combined.flux
        all_waves = spec_combined.wave
        all_ivars = spec_combined.ivar
    
        all_tgids = temp_chunk_i["TARGETID"].data
        all_zreds = temp_chunk_i["Z"].data

        #then we save this in a hdf5 file!

        save_path = file_template +  "_chunk_%d.h5"%chunk_i
    
        with h5py.File(save_path, "w") as f:
            f.create_dataset("TARGETID", data=all_tgids, dtype='i8')
            f.create_dataset("Z", data=all_zreds, dtype='f4')
            f.create_dataset("WAVE", data=all_waves["brz"], dtype='f4')  # shared
            f.create_dataset("FLUX", data=all_fluxs["brz"], dtype='f4')
            f.create_dataset("FLUX_IVAR", data=all_ivars["brz"], dtype='f4')


    #once we save all the chunks do we try to consolidate all the chunks?
    if nchunks == 1:
        #we just need to rename the file!
        os.rename(file_template + "_chunk_0.h5", file_template + ".h5")
   
    else:
                
        # Prepare empty lists for stacking
        all_targetids = []
        all_zreds = []
        all_fluxs = []
        all_ivars = []
        shared_wave = None
        
        for chunk_i in range(nchunks):
            with h5py.File(file_template + "_chunk_%d.h5"%chunk_i, "r") as f:
                all_targetids.append(f["TARGETID"][:])
                all_zreds.append(f["Z"][:])
                all_fluxs.append(f["FLUX"][:])
                all_ivars.append(f["FLUX_IVAR"][:])
                
                # assuming WAVE is identical across files
                if shared_wave is None:
                    shared_wave = f["WAVE"][:]
                    
            #remove the old files!
            os.remove(file_template + "_chunk_%d.h5"%chunk_i)

        # Stack along the first axis (rows)
        all_targetids = np.concatenate(all_targetids)
        all_zreds = np.concatenate(all_zreds)
        all_fluxs = np.concatenate(all_fluxs)
        all_ivars = np.concatenate(all_ivars)

        print(np.shape(all_fluxs))
        print(np.shape(shared_wave))
        print(np.shape(all_targetids))

        print_stage("Total number of spectra in consolidated spectra file = %d"%len(all_targetids))
                
        # Save to a new HDF5 file
        with h5py.File(file_template + ".h5", "w") as f:
            f.create_dataset("TARGETID", data=all_targetids, dtype='i8')
            f.create_dataset("Z", data=all_zreds, dtype='f4')
            f.create_dataset("WAVE", data=shared_wave, dtype='f4')
            f.create_dataset("FLUX", data=all_fluxs, dtype='f4')
            f.create_dataset("FLUX_IVAR", data=all_ivars, dtype='f4')

        print_stage("Consolidated spectra chunk saved at %s"%(file_template + ".h5") )
        
      
    #to read the data, one can do
    # with h5py.File("spectra.h5", "r") as f:
    #     wave = f["wave"][:]
    #     flux = f["flux"][0]  # single spectrum
    
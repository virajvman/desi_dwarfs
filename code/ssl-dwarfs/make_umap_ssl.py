'''
This script is run as follows after loading the relevant environments as 
conda activate
conda activate ssl-pl
/global/u1/v/virajvm/miniforge3/envs/ssl-pl/bin/python code/ssl-dwarfs/make_umap_ssl.py
'''

from ssl_legacysurvey.data_analysis import dimensionality_reduction
from ssl_legacysurvey.utils import plotting_tools as plt_tools # Plotting images or catalogue info
from matplotlib.colors import LogNorm
import numpy as np
import matplotlib.pyplot as plt
import glob
from ssl_legacysurvey.utils import load_data # Loading galaxy catalogue and image data from hdf5 file(s)

def load_all_data(save = False,include_image=False):
    '''
    In this function, we load all the representations and targetids from the chunk files into common arrays

    Actually to be space efficient, I will not be loading all images at once, but just the targetids. Once I get the associated targetid, I will directly read its image !
    Much more efficient!!
    
    '''

    all_files = glob.glob("/pscratch/sd/v/virajvm/ssl-legacysurvey-dwarfs/representations/represent*")
    print(f"A total of {len(all_files)} representations files to be read!")

    all_tgids_array = []
    all_image_array = []
    all_repr_array = []
    all_image_files_array = []
    all_rmags_array = []
    
    #we have the data split across different chunks and so we will load them one by one!
    for file_i in range(len(all_files)):
    
        h5_data_path = f'/pscratch/sd/v/virajvm/ssl-legacysurvey-dwarfs/h5_datasets/data_chunk_{file_i}.h5'
        print(f"Reading file: {h5_data_path}")
        DDL = load_data.DecalsDataLoader(image_dir=h5_data_path, npix_in=152)
        gals = DDL.get_data(-1, fields=DDL.fields_available,npix_out=152) # -1 to load all galaxies


        print(f"TARGETID SHAPE = {gals['targetid'].shape}")
        all_tgids_array.append(gals["targetid"])
        all_image_files_array.append(gals["image_path"])
        all_rmags_array.append( gals["mag_r"] )

        print(f"IMAGE SHAPE = {gals['images'].shape}")
        all_image_array.append(gals["images"])

        repres_path = f'/pscratch/sd/v/virajvm/ssl-legacysurvey-dwarfs/representations/represent_chunk_{file_i}.npy'
        repres_arr = np.load(repres_path)
        print(f"REPRESENTATION SHAPE = {repres_arr.shape}")
        
        all_repr_array.append(repres_arr)

    all_repr_array = np.array(all_repr_array)
    all_tgids_array = np.array(all_tgids_array)
    all_image_array = np.array(all_image_array)
    all_rmags_array = np.array(all_rmags_array)
    
    all_image_files_array = np.array(all_image_files_array)
    
    
    all_repr_array = np.concatenate( all_repr_array, axis = 0)
    all_tgids_array = np.concatenate( all_tgids_array, axis = 0)
    all_image_array = np.concatenate( all_image_array, axis = 0)
    all_rmags_array = np.concatenate( all_rmags_array, axis = 0)

    all_image_files_array = np.concatenate( all_image_files_array, axis = 0)
    
    
    print(f"Total targetid array shape = {np.shape(all_tgids_array)}")
    print(f"Total representations array shape = {np.shape(all_repr_array)}")
    print(f"Total image array shape = {np.shape(all_image_array)}")
    print(f"Total image file array shape = {np.shape(all_image_files_array)}")
    print(f"Total rmags array shape = {np.shape(all_rmags_array)}")
    
    
    
    if save:
        np.save("/pscratch/sd/v/virajvm/ssl-legacysurvey-dwarfs/representations/total_targetids_arr.npy", all_tgids_array)
        np.save("/pscratch/sd/v/virajvm/ssl-legacysurvey-dwarfs/representations/total_representation_arr.npy", all_repr_array )
        np.save("/pscratch/sd/v/virajvm/ssl-legacysurvey-dwarfs/representations/total_image_files_arr.npy", all_image_files_array )
        np.save("/pscratch/sd/v/virajvm/ssl-legacysurvey-dwarfs/representations/total_rmags_arr.npy", all_rmags_array )
        
    
    return all_tgids_array, all_repr_array, all_image_array



def make_umap_plot(umap_embedding_cos):
    '''
    Function that makes the UMAP plot!!
    '''

    plt.figure()
    plt.hist2d(umap_embedding_cos[:, 0], umap_embedding_cos[:, 1],bins=300,norm=LogNorm())
    plt.axis('off')
    plt.savefig("/pscratch/sd/v/virajvm/ssl-legacysurvey-dwarfs/plots/ssl_umap_dwarfs.png",bbox_inches="tight")
    plt.close()

    return


if __name__ == '__main__':

    #load all the data and representation arrays

    generate_inputs = True
    generate_umap = False

    if generate_inputs:
        _, all_repr_array,_ = load_all_data(save = True, include_image=False)
    else:
        all_tgids_array = np.load("/pscratch/sd/v/virajvm/ssl-legacysurvey-dwarfs/representations/total_targetids_arr.npy")
        all_repr_array = np.load("/pscratch/sd/v/virajvm/ssl-legacysurvey-dwarfs/representations/total_representation_arr.npy")

        print(f"Total targetid array shape = {np.shape(all_tgids_array)}")
        print(f"Total representations array shape = {np.shape(all_repr_array)}")


    if generate_umap:
        #make the UMAP now!    
        umap_embedding_cos, umap_trans_cos = dimensionality_reduction.umap_transform(all_repr_array, n_components=2, metric='cosine')

        print(f"Total umap embedding array shape = {np.shape(umap_embedding_cos)}")
        
    
        np.save("/pscratch/sd/v/virajvm/ssl-legacysurvey-dwarfs/umap_representations/total_umap_embedding_2d.npy",  umap_embedding_cos )
    else:
        umap_embedding_cos = np.load("/pscratch/sd/v/virajvm/ssl-legacysurvey-dwarfs/umap_representations/total_umap_embedding_2d.npy")

        print(f"Total umap embedding array shape = {np.shape(umap_embedding_cos)}")
        
    #UMAP plot!
    make_umap_plot(umap_embedding_cos)
    






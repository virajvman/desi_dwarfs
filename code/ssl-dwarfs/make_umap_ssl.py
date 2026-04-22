'''
This script is run as follows after loading the relevant environments as 
conda activate
conda activate ssl-pl
cd DESI2_LOWZ/desi_dwarfs/code
/global/u1/v/virajvm/miniforge3/envs/ssl-pl/bin/python ssl-dwarfs/make_umap_ssl.py
'''

import os
from ssl_legacysurvey.data_analysis import dimensionality_reduction
from ssl_legacysurvey.utils import plotting_tools as plt_tools # Plotting images or catalogue info
from matplotlib.colors import LogNorm
import numpy as np
import matplotlib.pyplot as plt
import glob
import re
from ssl_legacysurvey.utils import load_data # Loading galaxy catalogue and image data from hdf5 file(s)
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def _data_chunk_index(path):
    m = re.search(r"data_chunk_(\d+)\.h5$", path)
    return int(m.group(1)) if m else -1


def load_all_data(save = False, img_type=None):
    '''
    In this function, we load all the representations and targetids from the chunk files into common arrays

    Actually to be space efficient, I will not be loading all images at once, but just the targetids. Once I get the associated targetid, I will directly read its image !
    Much more efficient!!
    '''
        
    h5_dir = "/pscratch/sd/v/virajvm/catalog_dr1_dwarfs/ssl_shred_data/h5_datasets"
    h5_files = sorted(
        glob.glob(f"{h5_dir}/data_chunk_*.h5"), key=_data_chunk_index
    )
    print(f"A total of {len(h5_files)} data chunk files to align with representations!")

    all_image_array = []
    
    all_tgids_array = []
    all_repr_array = []
    
    all_gmags_array = []
    all_rmags_array = []
    all_zmags_array = []

    all_stardist_array = []

    all_mstar_array = []
    
    all_zreds_array = []
    
    rep_dir = "/pscratch/sd/v/virajvm/catalog_dr1_dwarfs/ssl_shred_data/representations"
    for file_i, h5_data_path in enumerate(h5_files):
        print(f"Reading file: {h5_data_path}")
        DDL = load_data.DecalsDataLoader(image_dir=h5_data_path, npix_in=152)
        gals = DDL.get_data(-1, fields=DDL.fields_available,npix_out=152) # -1 to load all galaxies


        print(f"TARGETID SHAPE = {gals['targetid'].shape}")
        all_tgids_array.append(gals["targetid"])
        
        all_gmags_array.append( gals["mag_g"] )
        all_rmags_array.append( gals["mag_r"] )
        all_zmags_array.append( gals["mag_z"] )

        all_stardist_array.append( gals["star_dist"] )
        
        all_mstar_array.append( gals["mstar"] )
    
        all_zreds_array.append( gals["redshift"] )
        
        print(f"IMAGE SHAPE = {gals['images'].shape}")
        all_image_array.append(gals["images"])

        repres_path = os.path.join(rep_dir, f"represent_chunk_{file_i}.npy")
        repres_arr = np.load(repres_path)
        print(f"REPRESENTATION SHAPE = {repres_arr.shape}")
        
        all_repr_array.append(repres_arr)

    all_repr_array = np.array(all_repr_array)
    all_tgids_array = np.array(all_tgids_array)
    
    all_gmags_array = np.array(all_gmags_array)
    all_rmags_array = np.array(all_rmags_array)
    all_zmags_array = np.array(all_zmags_array)
    
    all_stardist_array = np.array(all_stardist_array)
    
    all_mstar_array = np.array(all_mstar_array)
    
    all_zreds_array = np.array(all_zreds_array)
    all_image_array = np.array(all_image_array)
    
    all_repr_array = np.concatenate( all_repr_array, axis = 0)
    all_tgids_array = np.concatenate( all_tgids_array, axis = 0)
    
    all_gmags_array = np.concatenate( all_gmags_array, axis = 0)
    all_rmags_array = np.concatenate( all_rmags_array, axis = 0)
    all_zmags_array = np.concatenate( all_zmags_array, axis = 0)

    all_stardist_array = np.concatenate( all_stardist_array, axis = 0)
    
    all_mstar_array = np.concatenate( all_mstar_array, axis = 0)
    
    all_zreds_array = np.concatenate( all_zreds_array, axis = 0)
    all_image_array = np.concatenate( all_image_array, axis = 0)
        
    print(f"Total targetid array shape = {np.shape(all_tgids_array)}")
    print(f"Total representations array shape = {np.shape(all_repr_array)}")
    print(f"Total rmags array shape = {np.shape(all_rmags_array)}")
    print(f"Total image array shape = {np.shape(all_image_array)}")
    print(f"Total stardist array shape = {np.shape(all_stardist_array)}")
    
    
    if save:
        np.save("/pscratch/sd/v/virajvm/catalog_dr1_dwarfs/ssl_shred_data/dwarf_dr1/total_targetids_arr.npy", all_tgids_array)
        np.save("/pscratch/sd/v/virajvm/catalog_dr1_dwarfs/ssl_shred_data/dwarf_dr1/total_representation_arr.npy", all_repr_array )
        
        np.save("/pscratch/sd/v/virajvm/catalog_dr1_dwarfs/ssl_shred_data/dwarf_dr1/total_gmags_arr.npy", all_gmags_array )
        np.save("/pscratch/sd/v/virajvm/catalog_dr1_dwarfs/ssl_shred_data/dwarf_dr1/total_rmags_arr.npy", all_rmags_array )
        np.save("/pscratch/sd/v/virajvm/catalog_dr1_dwarfs/ssl_shred_data/dwarf_dr1/total_zmags_arr.npy", all_zmags_array )

        np.save("/pscratch/sd/v/virajvm/catalog_dr1_dwarfs/ssl_shred_data/dwarf_dr1/total_stardist_arr.npy", all_stardist_array )
        
        np.save("/pscratch/sd/v/virajvm/catalog_dr1_dwarfs/ssl_shred_data/dwarf_dr1/total_mstar_arr.npy", all_mstar_array )
        
        np.save("/pscratch/sd/v/virajvm/catalog_dr1_dwarfs/ssl_shred_data/dwarf_dr1/total_zreds_arr.npy", all_zreds_array )

        np.save("/pscratch/sd/v/virajvm/catalog_dr1_dwarfs/ssl_shred_data/dwarf_dr1/total_image_arr.npy", all_image_array )
        
    
    return all_repr_array



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
    generate_umap = True
    generate_pca = True
    N_PCA = 50

    # for img_type in ["recon","org"]:
        
    if generate_inputs:
        all_repr_array = load_all_data(save = True, img_type=None)
    else:
        all_repr_array = np.load(f"/pscratch/sd/v/virajvm/catalog_dr1_dwarfs/ssl_shred_data/total_representation_arr.npy")

        print(f"Total representations array shape = {np.shape(all_repr_array)}")

    #instead of going from 2048 straight to 2 dim for UMAP, it might be easier to go to 50 dim using PCA and then UMAP to 2
    if generate_pca:
        # Optional but recommended for PCA
        scaler = StandardScaler(with_mean=True, with_std=True)
        all_repr_scaled = scaler.fit_transform(all_repr_array)

        pca = PCA(n_components=N_PCA, random_state=42)
        repr_pca = pca.fit_transform(all_repr_scaled)

        print(f"PCA output shape = {repr_pca.shape}")
        print(
            f"Explained variance (first {N_PCA} comps) = "
            f"{np.sum(pca.explained_variance_ratio_):.3f}"
        )

        np.save(
            f"/pscratch/sd/v/virajvm/catalog_dr1_dwarfs/ssl_shred_data/pca/"
            f"total_repr_pca_{N_PCA}.npy",
            repr_pca,
        )

    else:
        repr_pca = np.load(
            f"/pscratch/sd/v/virajvm/catalog_dr1_dwarfs/ssl_shred_data/pca/"
            f"total_repr_pca_{N_PCA}.npy"
        )

        print(f"PCA output shape = {repr_pca.shape}")

    ########

    if generate_umap:
        #make the UMAP now!   
        print(f"UMAP input shape = {repr_pca.shape}")
        
        umap_embedding_cos, umap_trans_cos = dimensionality_reduction.umap_transform(repr_pca, n_components=2, metric='cosine')

        print(f"Total umap embedding array shape = {np.shape(umap_embedding_cos)}")
        
        np.save(f"/pscratch/sd/v/virajvm/catalog_dr1_dwarfs/ssl_shred_data/umap/total_umap_embedding_2d.npy",  umap_embedding_cos )
    else:
        umap_embedding_cos = np.load(f"/pscratch/sd/v/virajvm/catalog_dr1_dwarfs/ssl_shred_data/umap/total_umap_embedding_2d.npy")

        print(f"Total umap embedding array shape = {np.shape(umap_embedding_cos)}")
            
    # #UMAP plot!
    # make_umap_plot(umap_embedding_cos)
    






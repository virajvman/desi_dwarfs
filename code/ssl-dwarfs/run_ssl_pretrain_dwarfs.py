'''
This script is run as follows after loading the relevant environments as 
conda activate
conda activate ssl-pl
/global/u1/v/virajvm/miniforge3/envs/ssl-pl/bin/python code/ssl-dwarfs/run_ssl_pretrain_dwarfs.py 
'''

import os
import numpy as np
import torch
from ssl_legacysurvey.utils import load_data # Loading galaxy catalogue and image data from hdf5 file(s)
from ssl_legacysurvey.utils import plotting_tools as plt_tools # Plotting images or catalogue info

from ssl_legacysurvey.data_loaders import datamodules # Pytorch dataloaders and datamodules
from ssl_legacysurvey.data_loaders import decals_augmentations # Augmentations for training

from ssl_legacysurvey.data_analysis import dimensionality_reduction # PCA/UMAP functionality
import matplotlib.pyplot as plt

import numpy as np
import torchvision
import pytorch_lightning as pl
import argparse
import logging

from pathlib import Path
import sys
import glob
import math

from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.plugins import DDPPlugin
from ssl_legacysurvey.moco.moco2_module import Moco_v2 

from ssl_legacysurvey.data_loaders import datamodules
from ssl_legacysurvey.utils import format_logger
from scripts import predict
from ssl_legacysurvey.finetune import extract_model_outputs
from scripts import similarity_search_nxn

if __name__ == '__main__':

    ##load the model! This checkpoint file was obtained from the Globus endpoint. See the github for more info
    checkpoint_path = '/pscratch/sd/v/virajvm/ssl-legacysurvey-dwarfs/resnet50.ckpt'
    model = Moco_v2.load_from_checkpoint(
        checkpoint_path=checkpoint_path
        )

    print("Model finished loading!")

    all_files = glob.glob("/pscratch/sd/v/virajvm/ssl-legacysurvey-dwarfs/h5_datasets/data_chunk*")
    print(f"A total of {len(all_files)} data chunk files to be read!")
    
    #we have the data split across different chunks and so we will load them one by one!
    for file_i in range(len(all_files)):
    
        h5_data_path = f'/pscratch/sd/v/virajvm/ssl-legacysurvey-dwarfs/h5_datasets/data_chunk_{file_i}.h5'
        print(h5_data_path)
        # Load h5py file into dictionary
        DDL = load_data.DecalsDataLoader(image_dir=h5_data_path, npix_in=152)
        gals = DDL.get_data(-1, fields=DDL.fields_available,npix_out=152) # -1 to load all galaxies
        print("Available keys & data shapes:")
        for k in gals:
            if k != "image_path" and k != "file_path":
                print(f"{k} shape:", gals[k].shape)
            
            
        class Args: # In general codes in this project use argparse. Args() simplifies this for this example 
            # Data location and GPU availability 
            data_path = h5_data_path
            gpu = True # Use GPU?
            gpus = 1 # Number of gpus to use
            num_nodes = 1
            ngals_tot = gals['images'].shape[0]
            # Training
            verbose = True
            ssl_training = True
            batch_size = ngals_tot
            learning_rate = 0.03
            max_epochs = 5
            max_num_samples= ngals_tot
            
            check_val_every_n_epoch = 999 # We haven't provided validation set, so don't use it!
            num_sanity_val_steps = 0
        
            augmentations = 'grrrssgbjcgnrg'
            jitter_lim = 7
            
            strategy = 'dp' # Distributed training strategy,  ddp does not work in ipython notebook, only dp does
            seed = 13579
        
            checkpoint_every_n_epochs = 1
            num_workers = 1 # Number of workers for data loader
        
            # Model architecture and settings
            backbone = 'resnet50' # Encoder architecture to use, Can use any in torchvision, i.e. ['resnet18', 'resnet34', 'resnet50', 'resnet152', .....]
            use_mlp = True # use projection head
        
            emb_dim = 128 # Dimensionality where loss is calculated
            num_negatives = 16 #Number of negative samples to keep in queue for Mocov2
            
            # needed for predict.py script
            out_dir = '/pscratch/sd/v/virajvm/ssl-legacysurvey-dwarfs/representations'
            extract_representations = True
            checkpoint_path = '/pscratch/sd/v/virajvm/ssl-legacysurvey-dwarfs/resnet50.ckpt'
            use_mlp_representation = True
            overwrite = True
            file_head = '/pscratch/sd/v/virajvm/ssl-legacysurvey-dwarfs/'
            chunksize= ngals_tot
            batch_size_per_gpu = int(gals['images'].shape[0]/4)
            num_gpus = 1
            data_dim = 2048
            predict_batch_size = ngals_tot
            representation_directory = '/pscratch/sd/v/virajvm/ssl-legacysurvey-dwarfs/representations'
            representation_file_head = file_head
            umap_file_head = '/pscratch/sd/v/virajvm/ssl-legacysurvey-dwarfs/umap_representations'
            train_umap = True
            
            #For dimensionality reduction script
            n_samples = 1000
            sample_dimensionality = 100
            n_pca_components = 8
            n_umap_components = 2
            umap_embedding_file_path = os.path.join(out_dir, f"{umap_file_head}_{gals['images'].shape[0]}_embedding.npz")
            umap_transform_file_path = os.path.join(out_dir, f"{umap_file_head}_{gals['images'].shape[0]}_transform.pkl")
        
            #for similarity search:
            use_faiss = True
            use_gpu = True
            norm = True
            
            rep_dir = '/pscratch/sd/v/virajvm/ssl-legacysurvey-dwarfs/'
            output_dir = '/pscratch/sd/v/virajvm/ssl-legacysurvey-dwarfs/'
            knearest = 25
            delta_mag = 20
            start_on_chunk = 0
            survey = 'south'
            rep_file_head = file_head
            chunksize_similarity = ngals_tot
            sim_chunksize= ngals_tot
            rep_dim = 128
            nchunks = int(math.ceil(ngals_tot/chunksize))
            nchunks_similarity = int(math.ceil(ngals_tot/chunksize_similarity))
            supervised_training = True
            
        params = vars(Args)
        p = {}
        for k, v in params.items():
            p[k] = v
        params = p
    
    
        backbone = model.encoder_q
    
        # Remove the MLP projection head from the model, so output is now the representaion for each galaxy
        backbone.fc = torch.nn.Identity()
        
        params['ssl_training'] = False
        params['jitter_lim'] = 0
        params['augmentations'] = 'rrjc'#adjust to whatever parameters you want
        
        # Load all images as one batch
        
        transform = datamodules.DecalsTransforms(
            params['augmentations'],
            params
        )
        
        
        decals_dataloader = datamodules.DecalsDataset(
            h5_data_path,
            None,
            transform,
            params,
        )
        
        ngals =  gals['images'].shape[0]
        im, label = decals_dataloader.__getitem__(0)
        images = torch.empty((ngals, im.shape[0], im.shape[1], im.shape[2]), dtype=im.dtype)
        for i in range(ngals):
            images[i], _ = decals_dataloader.__getitem__(i)
            
        # Run images through model to get representations
        representations = backbone(images)
        
        if params['gpu']:
           representations = representations.detach()
            
        representations = representations.numpy()
    
        print(f"Representations shape = {representations.shape}")

        save_rep = f'/pscratch/sd/v/virajvm/ssl-legacysurvey-dwarfs/representations/represent_chunk_{file_i}.npy'

        ##SAVE THESE REPRESENTATIONS AS A FLOAT32 OBJECT
        
        #save this array
        np.save( save_rep, representations )
        


        
        

'''
This script is run as follows after loading the relevant environments as 
conda activate
conda activate ssl-pl
cd DESI2_LOWZ/desi_dwarfs/code
/global/u1/v/virajvm/miniforge3/envs/ssl-pl/bin/python ssl-dwarfs/run_ssl_pretrain_dwarfs.py 
'''

import os
import numpy as np
import torch
from ssl_legacysurvey.utils import load_data
from ssl_legacysurvey.utils import plotting_tools as plt_tools

from ssl_legacysurvey.data_loaders import datamodules
from ssl_legacysurvey.data_loaders import decals_augmentations

from ssl_legacysurvey.data_analysis import dimensionality_reduction
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
import re

from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.plugins import DDPPlugin
from ssl_legacysurvey.moco.moco2_module import Moco_v2 

from ssl_legacysurvey.data_loaders import datamodules
from ssl_legacysurvey.utils import format_logger
from scripts import predict
from ssl_legacysurvey.finetune import extract_model_outputs
from scripts import similarity_search_nxn


def _data_chunk_index(path):
    m = re.search(r"data_chunk_(\d+)\.h5$", path)
    return int(m.group(1)) if m else -1


if __name__ == '__main__':

    checkpoint_path = '/pscratch/sd/v/virajvm/ssl-legacysurvey-dwarfs/resnet50.ckpt'
    model = Moco_v2.load_from_checkpoint(checkpoint_path=checkpoint_path)

    # --- pick device and move model once, outside the chunk loop ---
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    model = model.to(device).eval()

    # sanity check
    print("encoder_q device:", next(model.encoder_q.parameters()).device)

    print("Model finished loading!")

    h5_glob = "/pscratch/sd/v/virajvm/catalog_dr1_dwarfs/ssl_shred_data/h5_datasets/data_chunk_*.h5"
    h5_data_paths = sorted(glob.glob(h5_glob), key=_data_chunk_index)
    print(f"A total of {len(h5_data_paths)} data chunk files to be read!")

    for file_i, h5_data_path in enumerate(h5_data_paths):
        print(h5_data_path)
        
        DDL = load_data.DecalsDataLoader(image_dir=h5_data_path, npix_in=152)
        gals = DDL.get_data(-1, fields=DDL.fields_available, npix_out=152)
        print("Available keys & data shapes:")

        ngals = gals['images'].shape[0]
        print(ngals)
            
        class Args:
            data_path = h5_data_path
            gpu = True
            gpus = 1
            num_nodes = 1
            ngals_tot = gals['images'].shape[0]
            verbose = True
            ssl_training = True
            batch_size = ngals_tot
            learning_rate = 0.03
            max_epochs = 5
            max_num_samples = ngals_tot
            check_val_every_n_epoch = 999
            num_sanity_val_steps = 0
            augmentations = 'grrrssgbjcgnrg'
            jitter_lim = 7
            strategy = 'dp'
            seed = 13579
            checkpoint_every_n_epochs = 1
            num_workers = 1
            backbone = 'resnet50'
            use_mlp = True
            emb_dim = 128
            num_negatives = 16
            out_dir = '/pscratch/sd/v/virajvm/catalog_dr1_dwarfs/ssl_shred_data/representations'
            extract_representations = True
            checkpoint_path = '/pscratch/sd/v/virajvm/ssl-legacysurvey-dwarfs/resnet50.ckpt'
            use_mlp_representation = True
            overwrite = True
            file_head = '/pscratch/sd/v/virajvm/ssl-legacysurvey-dwarfs/'
            chunksize = ngals_tot
            batch_size_per_gpu = int(gals['images'].shape[0] / 4)
            num_gpus = 1
            data_dim = 2048
            predict_batch_size = ngals_tot
            representation_directory = '/pscratch/sd/v/virajvm/catalog_dr1_dwarfs/ssl_shred_data/representations'
            representation_file_head = file_head
            umap_file_head = '/pscratch/sd/v/virajvm/ssl-legacysurvey-dwarfs/umap_representations'
            train_umap = True
            n_samples = ngals
            sample_dimensionality = 100
            n_pca_components = 8
            n_umap_components = 2
            umap_embedding_file_path = os.path.join(out_dir, f"{umap_file_head}_{gals['images'].shape[0]}_embedding.npz")
            umap_transform_file_path = os.path.join(out_dir, f"{umap_file_head}_{gals['images'].shape[0]}_transform.pkl")
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
            sim_chunksize = ngals_tot
            rep_dim = 128
            nchunks = int(math.ceil(ngals_tot / chunksize))
            nchunks_similarity = int(math.ceil(ngals_tot / chunksize_similarity))
            supervised_training = True
            
        params = {k: v for k, v in vars(Args).items()}

        backbone = model.encoder_q
        backbone.fc = torch.nn.Identity()
        # backbone is already on `device` because model was moved above,
        # but being explicit doesn't hurt:
        backbone = backbone.to(device).eval()

        params['ssl_training'] = False
        params['jitter_lim'] = 0
        params['augmentations'] = 'rrjc'

        transform = datamodules.DecalsTransforms(params['augmentations'], params)

        decals_dataloader = datamodules.DecalsDataset(
            h5_data_path,
            None,
            transform,
            params,
        )

        ngals = gals['images'].shape[0]
        im, label = decals_dataloader.__getitem__(0)

        # stage all images on CPU, pinned if GPU is available
        images = torch.empty(
            (ngals, im.shape[0], im.shape[1], im.shape[2]),
            dtype=im.dtype,
            pin_memory=(device.type == 'cuda'),
        )
        for i in range(ngals):
            images[i], _ = decals_dataloader.__getitem__(i)

        # --- mini-batched GPU inference ---
        mb = 128  # drop to 64 or 32 if you OOM on a shared GPU
        reps = []
        with torch.no_grad():
            for start in range(0, ngals, mb):
                batch = images[start:start + mb].to(device, non_blocking=True)
                reps.append(backbone(batch).cpu())

        representations = torch.cat(reps, dim=0).numpy()

        print(f"Representations shape = {representations.shape}")

        save_rep = f'/pscratch/sd/v/virajvm/catalog_dr1_dwarfs/ssl_shred_data/representations/represent_chunk_{file_i}.npy'
        print(f"Saving file = {save_rep}")
        np.save(save_rep, representations)
        
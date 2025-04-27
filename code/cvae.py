from torchvision import transforms
import torch
import torch.nn as nn
import torch.nn.functional as F
import glob
from tqdm.notebook import tqdm
from astropy.table import Table
from desi_lowz_funcs import sdss_rgb
import multiprocessing as mp
from torch.utils.data import DataLoader, Dataset
import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt
from multiprocessing import Pool, cpu_count
from desi_lowz_funcs import save_cutouts, parallel_run
import requests


def print_spec_urls(tgid_list,max_num=10):
    for ti in tgid_list[:max_num]:
        print("https://www.legacysurvey.org/viewer-desi/desi-spectrum/dr1/targetid%d"%ti )
    



    
def add_imgpath_to_catalog(org_file = "/pscratch/sd/v/virajvm/catalog_dr1_dwarfs/desi_y1_dwarf_clean_catalog_v3.fits", new_file = "/pscratch/sd/v/virajvm/catalog_dr1_dwarfs/desi_y1_dwarf_clean_catalog_v3_logm9.fits"):
    
    '''
    This function adds the image path as a column to a given catalog
    '''

    dwarf_good = Table.read("/pscratch/sd/v/virajvm/catalog_dr1_dwarfs/desi_y1_dwarf_clean_catalog_v3.fits")
    dwarf_good = dwarf_good[(dwarf_good["LOGM_SAGA"] < 9) ]

    target_list = []

    all_tgids  = dwarf_good["TARGETID"].data
    all_ras  = dwarf_good["RA"].data
    all_decs  = dwarf_good["DEC"].data
    
    for i in trange(len(dwarf_good)):
        target_list.append( ( all_tgids[i], all_ras[i], all_decs[i] )  )

    all_image_paths = parallel_run(target_list, n_processes=64)

    print(len(all_image_paths), len(target_list), len(dwarf_good))

    #filling in Nones with blank filler values
    all_image_paths[all_image_paths == None] = ""

    #converting to same dtype!
    all_image_paths = all_image_paths.astype(str)

    dwarf_good["IMAGE_PATH"] = all_image_paths
    
    dwarf_good.write("/pscratch/sd/v/virajvm/catalog_dr1_dwarfs/desi_y1_dwarf_clean_catalog_v3_logm9.fits",overwrite=True)
    
    return
    



def read_and_crop_image(match_path,size = 128,plot=False, title =None):
    '''
    Function that takes in image path, reads it and crops it to relevant size!
    '''
    try:
        #read the image
        data = fits.open(match_path)[0].data
        start = (350 - size) // 2
        end = start + size
        data = data[:, start:end, start:end   ]
        rgb_stuff = sdss_rgb([data[0],data[1],data[2]], ["g","r","z"], scales=dict(g=(2,6.0), r=(1,3.4), z=(0,2.2)), m=0.03)
        # Convert to uint8
        rgb_uint8 = (rgb_stuff * 255).astype(np.uint8)
        if plot:            
            if title is not None:
                plt.title(title,fontsize = 12)
            # Now you can display or save
            plt.imshow(rgb_uint8)
            plt.axis('off')
            plt.show()    
        #we transform the axes to make it in the shape we need it to be in!
        rgb_uint8  = np.moveaxis(rgb_uint8, 2, 0)    
        return rgb_uint8   
        
    except:
        print(match_path)
        return None
        # save_cutouts(ra, dec, match_path, session, size=350, timeout=30)
        # read_and_crop_image(ra,dec,match_path)
        
    

def read_dataset(max_sample = 40000,no_ELG=False,filter_stars = True):
    '''
    In this function, we read in the datasets. It returns the image array and the relevant conditional variables!
    '''
    
    dwarf_good = Table.read("/pscratch/sd/v/virajvm/catalog_dr1_dwarfs/desi_y1_dwarf_clean_catalog_v3_logm9.fits")

    ##this is just to test our with no stars in the cutout!

    # 
    
    dwarf_good = dwarf_good[ (dwarf_good["IMAGE_PATH"] != "")]
    
    if no_ELG:
        dwarf_good = dwarf_good[ dwarf_good["SAMPLE"] != "ELG"]

    if filter_stars:
        dwarf_good = dwarf_good[(dwarf_good["STARDIST_DEG"].data*3600 > 16) & (dwarf_good["STARFDIST"] > 2)]
    
    # dwarf_good = dwarf_good["TARGETID","SAMPLE","Z","MAG_R","IMAGE_PATH"]

    path_mask = dwarf_good["IMAGE_PATH"].data.mask
    #removing all the masked elements
    dwarf_good = dwarf_good[~path_mask]

    #let us randomly shuffle this sample
    N = len(dwarf_good)
    indices = np.random.permutation(N)
    dwarf_good = dwarf_good[indices]

    dwarf_cat = dwarf_good[:max_sample]

    ##getting the conditional data!
    all_redshifts = dwarf_cat["Z"].data
    all_mags = dwarf_cat["MAG_R"].data
    all_tgids = dwarf_cat["TARGETID"].data
    all_labels = np.array([all_redshifts, all_mags,all_tgids]).T
    
    #this is some masked array ... 
    all_dwarf_paths = dwarf_cat["IMAGE_PATH"].data.data
    
    ##getting all the imaging data!
    with mp.Pool(processes=64) as pool:
        results = list(tqdm(pool.imap(read_and_crop_image, all_dwarf_paths), total = len(all_dwarf_paths)  ))

    # Filter out failed results (None) and also filter corresponding labels
    valid_results = []
    valid_inds = []
    
    for i,img in tqdm(enumerate(results)):
        if img is not None:
            valid_results.append(img)
            valid_inds.append(i)

    all_dwarf_imgs = np.array(valid_results)
    all_inds = np.array(valid_inds)

    all_labels = all_labels[all_inds]
    dwarf_cat = dwarf_cat[all_inds]
    all_dwarf_paths = all_dwarf_paths[all_inds]
    
    # all_dwarf_imgs = np.array(results)

    print(np.shape(all_dwarf_imgs), np.shape(all_labels))
    ##check if there are any nans or infs in the data
    print(all_dwarf_imgs[np.isinf(all_dwarf_imgs)])
    print( all_dwarf_imgs[np.isnan(all_dwarf_imgs)] )

    return all_dwarf_imgs, all_labels, all_dwarf_paths, dwarf_cat
    


class GalaxyDataset(Dataset):
    def __init__(self, images, conditions):
        """
        :param images: torch.Tensor of shape (N, C, H, W). These are not normalized yet.
        :param conditions: torch.Tensor or np.array of shape (N, cond_dim)
        """
        if not torch.is_tensor(images):
            images = torch.tensor(images, dtype=torch.float32) / 255.0
        if not torch.is_tensor(conditions):
            conditions = torch.tensor(conditions, dtype=torch.float32)

            
        self.images = images
        self.conditions = conditions

    def __len__(self):
        return self.images.shape[0]

    def __getitem__(self, idx):
        return self.images[idx], self.conditions[idx]

        
def display_images(in_, out, conds, nrows=1, ncols=4, label=None, count=False):
    for row in range(nrows):
        if in_ is not None:
            in_pic = np.array(in_.data.cpu() * 255, dtype=int).transpose((0, 2, 3, 1))
            plt.figure(figsize=(18, 4))
            # plt.suptitle(label, color='w', fontsize=16)
            for col in range(ncols):
                plt.subplot(1, ncols, col + 1)
                if conds is None:
                    pass
                else:
                    plt.title(r"%d, %.2f"%( conds[col+ncols*row][2], conds[col+ncols*row][0]   ), fontsize = 7 )
                    
                plt.imshow(in_pic[col + ncols * row])
                plt.axis('off')
        out_pic = np.array(out.data.cpu() * 255, dtype=int).transpose((0, 2, 3, 1))
        plt.figure(figsize=(18, 6))
        for col in range(ncols):
            plt.subplot(1, ncols, col + 1)
            plt.imshow(out_pic[col + ncols * row])
            plt.axis('off')
            if count: plt.title(str(col + ncols * row), color='w')



#### CVAE code

class BaseCVAE(nn.Module):
    def __init__(self, cond_dim, z_dim, flatten_dim, decoder_output_shape):
        super().__init__()
        self.cond_dim = cond_dim
        self.z_dim = z_dim
        self.flatten_dim = flatten_dim
        self.decoder_output_shape = decoder_output_shape

        self.fc_mu = nn.Linear(self.flatten_dim + cond_dim, z_dim)
        self.fc_logvar = nn.Linear(self.flatten_dim + cond_dim, z_dim)
        self.decoder_input = nn.Linear(z_dim + cond_dim, self.flatten_dim)

    def encode(self, x, c):
        x = self.encoder_conv(x)
        x = x.view(x.size(0), -1)
        x = torch.cat([x, c], dim=1)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, c):
        z = torch.cat([z, c], dim=1)
        x = self.decoder_input(z)
        x = x.view(x.size(0), *self.decoder_output_shape)
        return self.decoder_conv(x)

    def forward(self, x, c):
        mu, logvar = self.encode(x, c)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z, c)
        return x_recon, mu, logvar



class CVAE_F256(BaseCVAE):
    def __init__(self, cond_dim=2, z_dim=50):
        flatten_dim = 256 * 4 * 4
        decoder_output_shape = (256, 4, 4)
        super().__init__(cond_dim, z_dim, flatten_dim, decoder_output_shape)

        self.encoder_conv = nn.Sequential(
            nn.Conv2d(3, 32, 4, 2, 1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2, 1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 128, 4, 2, 1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.Conv2d(128, 256, 4, 2, 1), nn.BatchNorm2d(256), nn.ReLU()
        )

        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, 2, 1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, 2, 1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, 2, 1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.ConvTranspose2d(32, 3, 4, 2, 1), nn.Sigmoid()
        )


class CVAE_F64(BaseCVAE):
    def __init__(self, cond_dim=2, z_dim=50):
        flatten_dim = 64 * 8 * 8
        decoder_output_shape = (64, 8, 8)
        super().__init__(cond_dim, z_dim, flatten_dim, decoder_output_shape)

        self.encoder_conv = nn.Sequential(
            nn.Conv2d(3, 16, 4, 2, 1), nn.BatchNorm2d(16), nn.ReLU(),
            nn.Conv2d(16, 32, 4, 2, 1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2, 1), nn.BatchNorm2d(64), nn.ReLU()
        )

        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 4, 2, 1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 4, 2, 1), nn.BatchNorm2d(16), nn.ReLU(),
            nn.ConvTranspose2d(16, 3, 4, 2, 1), nn.Sigmoid()
        )

class BaseVAE(nn.Module):
    def __init__(self, z_dim, flatten_dim, decoder_output_shape):
        super().__init__()
        self.z_dim = z_dim
        self.flatten_dim = flatten_dim
        self.decoder_output_shape = decoder_output_shape

        self.fc_mu = nn.Linear(self.flatten_dim, z_dim)
        self.fc_logvar = nn.Linear(self.flatten_dim, z_dim)
        self.decoder_input = nn.Linear(z_dim, self.flatten_dim)

    def encode(self, x):
        x = self.encoder_conv(x)
        x = x.view(x.size(0), -1)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        x = self.decoder_input(z)
        x = x.view(x.size(0), *self.decoder_output_shape)
        return self.decoder_conv(x)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z)
        return x_recon, mu, logvar


class VAE_F64(BaseVAE):
    def __init__(self, z_dim=50):
        flatten_dim = 64 * 8 * 8
        decoder_output_shape = (64, 8, 8)
        super().__init__(z_dim, flatten_dim, decoder_output_shape)

        self.encoder_conv = nn.Sequential(
            nn.Conv2d(3, 16, 4, 2, 1),  # 64 -> 32
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 32, 4, 2, 1),  # 32 -> 16
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2, 1),  # 16 -> 8
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )

        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 4, 2, 1),  # 8 -> 16
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 4, 2, 1),  # 16 -> 32
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 3, 4, 2, 1),  # 32 -> 64
            nn.Sigmoid()
        )

class VAE_F256(BaseVAE):
    def __init__(self, z_dim=50):
        flatten_dim = 256 * 4 * 4
        decoder_output_shape = (256, 4, 4)
        super().__init__(z_dim, flatten_dim, decoder_output_shape)

        self.encoder_conv = nn.Sequential(
            nn.Conv2d(3, 32, 4, 2, 1),  # 64 -> 32
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2, 1),  # 32 -> 16
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, 4, 2, 1),  # 16 -> 8
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 256, 4, 2, 1),  # 8 -> 4
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )

        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, 2, 1),  # 4 -> 8
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),  # 8 -> 16
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),  # 16 -> 32
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, 4, 2, 1),  # 32 -> 64
            nn.Sigmoid()
        )




### the loss funtions

### here we use the MMD+ELBO loss function that John uses

def compute_kernel(x, y):
    x_size = x.size(0)
    y_size = y.size(0)
    dim = x.size(1)
    x = x.unsqueeze(1)
    y = y.unsqueeze(0) 
    tiled_x = x.expand(x_size, y_size, dim)
    tiled_y = y.expand(x_size, y_size, dim)
    kernel_input = (tiled_x - tiled_y).pow(2).mean(2) 
    return torch.exp(-kernel_input) # (x_size, y_size)
    
def compute_mmd(x, y):
    xx_kernel = compute_kernel(x,x)
    yy_kernel = compute_kernel(y,y)
    xy_kernel = compute_kernel(x,y)
    return torch.mean(xx_kernel) + torch.mean(yy_kernel) - 2*torch.mean(xy_kernel)

def MMD_loss_cvae(model, x_hat, x, mu, logvar, alpha=1.0, beta=1.0):
    """
    MMD-VAE loss: 
    MSE + (1 - alpha)*KL + (beta + alpha - 1)*MMD

    We also return the individual loss terms to plot for reference

    In the physics inspired CVAE, the number of latent dimensions are physics parameters mu,var and the loss function is appropriately chosen to ensure the latent dimensions closely map onto that. 
    If we want our latent space to be regularized, we will make sure the KL term is not zero.

    The basic idea behind the MMD loss is that two distributions are identical if and only if their moments are also the same.
    
    More information here: https://ermongroup.github.io/blog/a-tutorial-on-mmd-variational-autoencoders/
    
    If I do not include the KL term, and just have MMD term, the variance parameter collapses to zero. And the variational aspect is no longer useful. To make sure our final latent distribution is sort of normal, we slowly increase the contribution of the KL term. This process is called
    an annealing process. 

    Let us fix beta = 1, and just change the alpha term such that it decreases from 1 -> 0.5 slowly linearly.
    
    """
    mse_loss = F.mse_loss(x_hat, x, reduction='sum')

    # KL term that will help regularize it to a N(0,1)
    kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    # Sample from standard normal prior
    prior_sample = torch.randn_like(mu)
    ##the treatment here is a little different than the one in John Wu's notebook

    # Get current batch latent sample
    z = model.reparameterize(mu, logvar)

    # Compute MMD between z and prior
    mmd = compute_mmd(z, prior_sample) * x.size(0)
    
    return mse_loss + (1 - alpha)*kl_div + (beta + alpha - 1)*mmd,  kl_div, (beta + alpha - 1) * mmd,  mse_loss


## generate random instances and march through condition variables like redshift and see if it makes sense 
## also test with setting random conditionals (not meaniningful) 


##functions for training!

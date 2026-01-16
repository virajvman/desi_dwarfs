from astropy.io import fits
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy.stats import binned_statistic_2d
import math
import os
import numpy as np
import glob
from matplotlib.colors import LogNorm

def sdss_rgb(imgs, bands, scales=None,
             m = 0.02):
    rgbscales = {'u': (2,1.5), #1.0,
                 'g': (2,2.5),
                 'r': (1,1.5),
                 'i': (0,1.0),
                 'z': (0,0.4), #0.3
                 }
    if scales is not None:
        rgbscales.update(scales)

    I = 0
    for img,band in zip(imgs, bands):
        plane,scale = rgbscales[band]
        img = np.maximum(0, img * scale + m)
        I = I + img
    I /= len(bands)
    Q = 20
    fI = np.arcsinh(Q * I) / np.sqrt(Q)
    I += (I == 0.) * 1e-6
    H,W = I.shape
    rgb = np.zeros((H,W,3), np.float32)
    for img,band in zip(imgs, bands):
        plane,scale = rgbscales[band]
        rgb[:,:,plane] = (img * scale + m) * fI / I
    rgb = np.clip(rgb, 0, 1)
    return rgb


def dr2_rgb(rimgs, bands, **ignored):
    return sdss_rgb(rimgs, bands, scales=dict(g=(2,6.0), r=(1,3.4), z=(0,2.2)), m=0.03)



def scatter_plot_as_images_from_array(imgs, z_emb, nx=8, ny=8, npix_show=152, iseed=13579):
    """
    Sample points from 2D embedding space and display the corresponding galaxy images.

    Returns
    -------
    img_full : np.ndarray
        Composite image showing selected thumbnails in UMAP space bins.
    """
    print(f"Dimensions of input UMAP space : {z_emb.shape}")
    z_emb = z_emb[:, :2]  # Ensure 2D
    nplt = nx * ny
    img_full = np.zeros((ny * npix_show, nx * npix_show, 3)) + 255

    xmin, xmax = z_emb[:, 0].min(), z_emb[:, 0].max()
    ymin, ymax = z_emb[:, 1].min(), z_emb[:, 1].max()

    binx = np.linspace(xmin, xmax, nx + 1)
    biny = np.linspace(ymin, ymax, ny + 1)

    ret = binned_statistic_2d(z_emb[:, 0], z_emb[:, 1], z_emb[:, 1], 'count', bins=[binx, biny], expand_binnumbers=True)
    z_emb_bins = ret.binnumber.T
    
    inds_lin = np.arange(z_emb.shape[0])
    inds_selected = []
    

    n_candidates = 3

    for ix in range(nx):
        for iy in range(ny):
            dm = (z_emb_bins[:, 0] == ix) & (z_emb_bins[:, 1] == iy)
            inds = inds_lin[dm]
            np.random.seed(ix * nx + iy + iseed)
            if len(inds) > 0:
                selected = np.random.choice(inds, size=min(n_candidates, len(inds)), replace=False)
                inds_selected.append(selected)
            else:
                inds_selected.append([])  # no candidates for this bin
            # if len(inds) > 0:
            #     ind_plt = np.random.choice(inds)
            #     inds_selected.append(ind_plt)  # This is an index into image_array

    text_entries = []
    
    # Now build the composite image
    iimg = 0
    for ix in range(nx):
        for iy in range(ny):
            if iimg % 100 == 0 and iimg > 0:
                print(f"{iimg}/{nx*ny}")
                
            # dm = (z_emb_bins[:, 0] == ix) & (z_emb_bins[:, 1] == iy)
            # inds = inds_lin[dm]
            candidates = inds_selected[iimg]  # list of candidate indices

            if len(candidates) > 0:
                ind = candidates[0]
                img = imgs[ind]
    
                # Crop center
                size = npix_show
                start = (img.shape[1] - size) // 2
                end = start + size
                img = img[:, start:end, start:end]
    
                rgb_img = dr2_rgb(img, ['g', 'r', 'z'])[::-1]
    
                img_full[ix * npix_show:(ix + 1) * npix_show,
                         iy * npix_show:(iy + 1) * npix_show] = rgb_img

                # record text position (x, y) and label
                x_text = iy * npix_show + 4
                y_text = ix * npix_show + npix_show - 8
                text_entries.append((x_text, y_text, str(ind)))
    
            iimg += 1
            #####

    return img_full, text_entries


def plot_umap(img_full, text_entries, save_path):

    plt.figure(figsize=(60, 60))
    plt.imshow(img_full, origin="lower")

    for x, y, label in text_entries:
        plt.text(
            x, y, label,
            color="white",
            fontsize=10,
            ha="left",
            va="top",
            bbox=dict(facecolor="black", alpha=0.5, pad=1)
        )

    plt.axis("off")
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()


def make_umap_plot(umap_embedding_cos,nx, ny):
    '''
    Function that makes the UMAP plot!!
    '''

    plt.figure()
    plt.hist2d(umap_embedding_cos[:, 0], umap_embedding_cos[:, 1],bins=10,cmap = "Reds", range= ((-8,15), (-8,15)  ),norm=LogNorm() )
    # plt.axis('off')
    plt.xlim([-8,15])
    plt.ylim([-8,15])
    plt.savefig("/pscratch/sd/v/virajvm/ssl-legacysurvey-dwarfs/plots/ssl_umap_dwarfs_V2.png",bbox_inches="tight")
    plt.close()

    xmin, xmax = -8, 15
    ymin, ymax = -8, 15

    binx = np.linspace(xmin, xmax, nx + 1)
    biny = np.linspace(ymin, ymax, ny + 1)

    plt.figure(figsize=(8, 8))
    plt.hist2d(umap_embedding_cos[:, 0], umap_embedding_cos[:, 1], bins=[binx, biny], cmap='Blues')
    for ix in range(nx):
        for iy in range(ny):
            xcen = 0.5 * (binx[ix] + binx[ix+1])
            ycen = 0.5 * (biny[iy] + biny[iy+1])
            plt.text(xcen, ycen, f"{ix},{iy}", ha='center', va='center', fontsize=6, color='red')
    
    plt.xlabel("UMAP 1")
    plt.ylabel("UMAP 2")
    plt.title("UMAP bins with (ix,iy) labels")
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.savefig("/pscratch/sd/v/virajvm/ssl-legacysurvey-dwarfs/plots/umap_bin_debug_labels.png")
    plt.close()

    return

    
if __name__ == '__main__':

    for size in [25,50,75]:
        for img_type in ["recon"]:
            #load the umap and tgids array
            umap_embedding_cos = np.load(f"/pscratch/sd/v/virajvm/catalog_dr1_dwarfs/ssl_shred_data/umap/total_umap_embedding_2d_{img_type}.npy")
        
            #load all the image file paths of these images!!
            all_images = np.load(f"/pscratch/sd/v/virajvm/catalog_dr1_dwarfs/ssl_shred_data/total_image_{img_type}_arr.npy")
        
            print(all_images.shape)
            
            # make the image collage plot
            nx, ny = size,size
            img_full, text_entries = scatter_plot_as_images_from_array(all_images, umap_embedding_cos, nx=nx, ny=ny, npix_show=128)
        
            plot_umap(img_full, text_entries, save_path =  f"/pscratch/sd/v/virajvm/catalog_dr1_dwarfs/ssl_shred_data/plots/umap_galaxy_imgs_{img_type}_{size}.pdf")
            

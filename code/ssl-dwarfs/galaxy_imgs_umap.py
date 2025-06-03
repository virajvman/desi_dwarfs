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



def scatter_plot_as_images_from_array(img_file_array, z_emb, nx=8, ny=8, npix_show=64, iseed=13579):
    """
    Sample points from 2D embedding space and display the corresponding galaxy images.

    tgids_array and z_emb have a one to one mapping!

    Parameters
    ----------
    tgids_array : array
        Array of shape (N_images) which are the targetids of the images!
    z_emb : array
        (N_total, 2+) array of the galaxy locations in UMAP embedding space.
    nx, ny : int
        Number of tiles in x and y direction for the image grid.
    npix_show : int
        Output size per image tile in pixels (assumed square).
    iseed : int
        Seed for reproducibility in image selection per tile.
    display_image : bool
        If True, show the matplotlib image.

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

    # Now build the composite image
    iimg = 0
    for ix in range(nx):
        for iy in range(ny):
            if iimg % 100 == 0 and iimg > 0:
                print(f"{iimg}/{nx*ny}")
            # dm = (z_emb_bins[:, 0] == ix) & (z_emb_bins[:, 1] == iy)
            # inds = inds_lin[dm]
            candidates = inds_selected[iimg]  # list of candidate indices
    
            for cand in candidates:
                try:
                    tgid_file = img_file_array[cand]
    
                    img = fits.open(tgid_file)[0].data
    
                    # Crop center
                    size = npix_show
                    start = (img.shape[1] - size) // 2
                    end = start + size
                    img = img[:, start:end, start:end]
    
                    rgb_img = dr2_rgb(img, ['g', 'r', 'z'])[::-1]
    
                    img_full[ix * npix_show:(ix + 1) * npix_show,
                             iy * npix_show:(iy + 1) * npix_show] = rgb_img

                    break  # exit candidate loop once one image is loaded
        
                except Exception as e:
                    print(f"Warning: could not load {tgid_file} – {e}")
                    continue
    
            iimg += 1
            #####

            # if len(inds) > 0:
            #     #this whole thing can be made better when we add the image path to the h5 dataset itself! We do not have access to fits in this
            #     tgid_file = img_file_array[inds_selected[iimg]]

            #     img = fits.open(tgid_file)[0].data
            #     #crop the image to center 152
            #     size = 96
            #     start = (img.shape[1] - size) // 2  # assumes square images
            #     end = start + size
            #     img = img[:, start:end, start:end]

            #     #conver to to the rgb image!
            #     rgb_img = dr2_rgb(img,
            #                          ['g','r','z'])[::-1]
                
            #     img_full[iy * npix_show:(iy + 1) * npix_show,
            #              ix * npix_show:(ix + 1) * npix_show] = rgb_img
            #     iimg += 1

    #once this image is returned, we will plot it!
    
    # fig, ax = plt.subplots(figsize=(10, 10))
    # ax.imshow(img_full,origin = "lower")
    # ax.set_xticks(np.arange(0, nx * npix_show, npix_show) )
    # ax.set_yticks(np.arange(0, ny * npix_show, npix_show) )
    # ax.set_xticklabels(np.arange(nx))
    # ax.set_yticklabels(np.arange(ny))

    # ax.set_xlim([-2*npix_show, (nx+2) * npix_show ] )
    # ax.set_ylim([-2*npix_show, (ny+2) * npix_show ] )
            
    # ax.set_xlabel("ix")
    # ax.set_ylabel("iy")
    # ax.set_title("Image grid: (ix, iy) indices")
    # ax.grid(True, color='red', linestyle='--', linewidth=0.5)
    
    # plt.savefig("/pscratch/sd/v/virajvm/ssl-legacysurvey-dwarfs/plots/image_grid_debug_labels.png")
    # plt.close()

    return img_full


# def scatter_plot_as_images_rectangular_grid(img_file_array, z_emb, nx=8, ny=8, npix_show=64, iseed=13579):
#     """
#     Arrange galaxy images in a rectangular grid using UMAP ordering.

#     Parameters
#     ----------
#     tgids_array : array
#         Array of targetids for the galaxy images.
#     z_emb : array
#         (N_total, 2+) array of UMAP locations.
#     nx, ny : int
#         Grid size (number of tiles along x and y).
#     npix_show : int
#         Size (pixels) of each image tile.
#     iseed : int
#         Random seed for reproducibility in selection.

#     Returns
#     -------
#     img_full : np.ndarray
#         Composite RGB image in a rectangular layout.
#     """
    
#     print(f"Dimensions of input UMAP space: {z_emb.shape}")
#     z_emb = z_emb[:, :2]
#     n_total = z_emb.shape[0]
#     n_show = nx * ny

#     # Normalize UMAP coordinates to [0, 1]
#     z_norm = (z_emb - z_emb.min(0)) / (z_emb.max(0) - z_emb.min(0))

#     # Sort by x then y for approximate UMAP ordering
#     sort_idx = np.lexsort((z_norm[:, 1], z_norm[:, 0]))
#     sorted_img_file = img_file_array[sort_idx]

#     # Randomly sample n_show from top sorted indices
#     np.random.seed(iseed)
#     selected_img_file = sorted_img_file[:n_show]

#     # Build the RGB composite image
#     img_full = np.zeros((ny * npix_show, nx * npix_show, 3)) + 255  # white background

#     iimg = 0
#     for iy in range(ny):
#         for ix in range(nx):
#             if iimg % 100 == 0 and iimg > 0:
#                 print(f"{iimg}/{nx * ny}")
#             tgid_file = selected_img_file[iimg]
  
#             img = fits.open(tgid_file)[0].data
#             size = 96
#             start = (img.shape[1] - size) // 2
#             end = start + size
#             img = img[:, start:end, start:end]

#             rgb_img = dr2_rgb(img, ['g', 'r', 'z'])[::-1]

#             img_full[ix * npix_show:(ix + 1) * npix_show,
#                      iy * npix_show:(iy + 1) * npix_show] = rgb_img

#             iimg += 1

#     return img_full
   

def plot_umap(img_full,save_path):
    
    #save plot and save this now!!
    plt.figure(figsize = (10,10))
    plt.imshow(img_full,origin="lower")
    # plt.xlim([-8,15])
    # plt.ylim([-8,15])
    # plt.xticks([])
    # plt.yticks([])
    # plt.axis('off') 
    plt.savefig(save_path,bbox_inches="tight")
    plt.close()


    
    return


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

    #load the umap and tgids array
    umap_embedding_cos = np.load("/pscratch/sd/v/virajvm/ssl-legacysurvey-dwarfs/umap_representations/total_umap_embedding_2d.npy")


    #load all the image file paths of these images!!
    all_image_files = np.load("/pscratch/sd/v/virajvm/ssl-legacysurvey-dwarfs/representations/total_image_files_arr.npy",allow_pickle = True)

    # mask = (umap_embedding_cos[:,0] > -8) &   (umap_embedding_cos[:,1] > -8)  &   (umap_embedding_cos[:,1] < 15) & (umap_embedding_cos[:,0] < 15)
    # umap_embedding_cos = umap_embedding_cos[  mask  ]
    # all_image_files = all_image_files[  mask  ]

    # make the image collage plot
    nx, ny = 50,50
    img_full = scatter_plot_as_images_from_array(all_image_files, umap_embedding_cos, nx=nx, ny=ny)

    make_umap_plot(umap_embedding_cos,nx,ny)

    # # img_full_rect = scatter_plot_as_images_rectangular_grid(all_image_files, umap_embedding_cos, nx=nx, ny=ny)

    plot_umap(img_full, save_path =  "/pscratch/sd/v/virajvm/ssl-legacysurvey-dwarfs/plots/umap_galaxy_imgs.png")
    # plot_umap(img_full, save_path =  "/pscratch/sd/v/virajvm/ssl-legacysurvey-dwarfs/plots/umap_galaxy_rect_imgs.png")
    

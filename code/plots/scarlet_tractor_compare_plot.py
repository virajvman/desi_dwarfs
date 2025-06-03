from matplotlib.patheffects import withStroke
import requests
import os
import sys
import numpy as np
from astropy.io import fits
from astropy.table import Table, vstack
import matplotlib
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import LogNorm
import matplotlib.patches as patches
from matplotlib.patches import Circle


def sdss_rgb(imgs, bands, scales=None,m = 0.02):
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
        
    # b,g,r = [rimg * rgbscales[b] for rimg,b in zip(imgs, bands)]
    # r = np.maximum(0, r + m)
    # g = np.maximum(0, g + m)
    # b = np.maximum(0, b + m)
    # I = (r+g+b)/3.
    Q = 20
    fI = np.arcsinh(Q * I) / np.sqrt(Q)
    I += (I == 0.) * 1e-6
    H,W = I.shape
    rgb = np.zeros((H,W,3), np.float32)
    for img,band in zip(imgs, bands):
        plane,scale = rgbscales[band]
        rgb[:,:,plane] = (img * scale + m) * fI / I

    # R = fI * r / I
    # G = fI * g / I
    # B = fI * b / I
    # # maxrgb = reduce(np.maximum, [R,G,B])
    # # J = (maxrgb > 1.)
    # # R[J] = R[J]/maxrgb[J]
    # # G[J] = G[J]/maxrgb[J]
    # # B[J] = B[J]/maxrgb[J]
    # rgb = np.dstack((R,G,B))
    rgb = np.clip(rgb, 0, 1)
    
    return rgb


def get_tractor_model(ra,dec,size=350,path=None):
    '''
    We extract the model image from Tractor using hte url method
    '''
    url_prefix = 'https://www.legacysurvey.org/viewer/'

    url = url_prefix + f'cutout.fits?ra={ra}&dec={dec}&size=%s&'%size
    
    url += 'layer=ls-dr9-model&pixscale=0.262&bands=grz'
    resp = session.get(url)
    with open(path, "wb") as f:
        f.write(resp.content)
        
    return      

def get_tractor_model_resid(ra,dec,size=350,path=None):
    '''
    We extract the model image from Tractor using hte url method
    '''
    url_prefix = 'https://www.legacysurvey.org/viewer/'

    url = url_prefix + f'cutout.fits?ra={ra}&dec={dec}&size=%s&'%size
    
    url += 'layer=ls-dr9-resid&pixscale=0.262&bands=grz'
    resp = session.get(url)
    with open(path, "wb") as f:
        f.write(resp.content)
        
    return  

def get_new_bounding_box(box, img_data_total):

    
    z1,z2,y1,y2 = box
    z_center = (z1 + z2) // 2
    y_center = (y1 + y2) // 2
    
    # Current sizes
    z_len = z2 - z1
    y_len = y2 - y1
    
    # New square half-size (from center to edge)
    half_size = max(z_len, y_len) // 2
    
    # Update bounding box to be square while keeping center
    z1_new = z_center - half_size
    z2_new = z_center + half_size
    y1_new = y_center - half_size
    y2_new = y_center + half_size
    
    # Optional: clip to avoid going out of image bounds
    z1_new = max(z1_new, 0)
    y1_new = max(y1_new, 0)
    z2_new = min(z2_new, img_data_total.shape[1])  # assuming 3D image with shape (bands, Z, Y)
    y2_new = min(y2_new, img_data_total.shape[2])

    return [z1_new, z2_new, y1_new, y2_new]



def plot(index, data_scarlet):

    file_path = data_scarlet["FILE_PATH"][index]
    img_path = data_scarlet["IMAGE_PATH"][index]
    
    img_data_total = fits.open(img_path)[0].data
    
    print(file_path)
    
    scar_mags = np.load( file_path + "/scarlet_mags.npy")
    
    scarlet_total_model = np.load(file_path + "/total_model_rgb.npy")
    scarlet_model_rgb = np.load(file_path + "/dwarf_model_rgb.npy")
    scarlet_resid_rgb = np.load(file_path + "/model_residual_rgb.npy")
    bounding_box_coords = np.load(file_path + "/bounding_box_coords.npy")
    
    ##### PREPARING THE SCARLET MODEL IMAGES
    ##updating their shapes so they are same size:
    target_shape = (350,350,3)  # assuming shape is (bands, H, W)
    # Initialize arrays o
    fill_val = 0.1267482
    padscar_total_model = fill_val + np.zeros(target_shape)
    padscar_model_rgb = fill_val + np.zeros(target_shape)
    padscar_resid_rgb = fill_val + np.zeros(target_shape)
    
    # Embed the smaller arrays into the appropriate slices
    padscar_total_model[bounding_box_coords[0]:bounding_box_coords[1], bounding_box_coords[2]:bounding_box_coords[3], :] = scarlet_total_model
    padscar_model_rgb[bounding_box_coords[0]:bounding_box_coords[1], bounding_box_coords[2]:bounding_box_coords[3],:] = scarlet_model_rgb
    padscar_resid_rgb[bounding_box_coords[0]:bounding_box_coords[1], bounding_box_coords[2]:bounding_box_coords[3],:] = scarlet_resid_rgb
    
    bounding_box_coords = get_new_bounding_box(bounding_box_coords, img_data_total)
    
    scarlet_total_model = padscar_total_model[bounding_box_coords[0]:bounding_box_coords[1], bounding_box_coords[2]:bounding_box_coords[3]]
    scarlet_model_rgb = padscar_model_rgb[bounding_box_coords[0]:bounding_box_coords[1], bounding_box_coords[2]:bounding_box_coords[3]]
    scarlet_resid_rgb = padscar_resid_rgb[bounding_box_coords[0]:bounding_box_coords[1], bounding_box_coords[2]:bounding_box_coords[3]]
    

    ##### PREPARING THE ACTUAL IMAGES!
    img_data = img_data_total[:, bounding_box_coords[0] : bounding_box_coords[1], bounding_box_coords[2] : bounding_box_coords[3] ]
    img_rgb = sdss_rgb([img_data[0],img_data[1],img_data[2]], ["g","r","z"], scales=dict(g=(2,6.0), r=(1,3.4), z=(0,2.2)), m=0.03)
    

    ##### PREPARING THE TRACTOR MODELS
    get_tractor_model(data_scarlet["RA"][index],data_scarlet["DEC"][index],
                      size=350,path=f"/pscratch/sd/v/virajvm/trash/model_{index}.fits")
    
    get_tractor_model_resid(data_scarlet["RA"][index],data_scarlet["DEC"][index],
                      size=350,path=f"/pscratch/sd/v/virajvm/trash/model_resid_{index}.fits")
    
    tractor_model = fits.open(f"/pscratch/sd/v/virajvm/trash/model_{index}.fits")[0].data
    tractor_model = tractor_model[:, bounding_box_coords[0] : bounding_box_coords[1], bounding_box_coords[2] : bounding_box_coords[3] ]  
    
    tractor_model_resid = fits.open(f"/pscratch/sd/v/virajvm/trash/model_resid_{index}.fits")[0].data
    tractor_model_resid = tractor_model_resid[:, bounding_box_coords[0] : bounding_box_coords[1], bounding_box_coords[2] : bounding_box_coords[3] ]  
    
    tractor_model_rgb = sdss_rgb([tractor_model[0],tractor_model[1],tractor_model[2]], ["g","r","z"], scales=dict(g=(2,6.0), r=(1,3.4), z=(0,2.2)), m=0.03)
    tractor_model_resid_rgb = sdss_rgb([tractor_model_resid[0],tractor_model_resid[1],tractor_model_resid[2]], ["g","r","z"], scales=dict(g=(2,6.0), r=(1,3.4), z=(0,2.2)), m=0.03)
    
    ## load the dwarf only model!!
    
    tractor_bkg_model = np.load(file_path + "/tractor_background_model.npy")
    tractor_blend_model = np.load(file_path + "/tractor_blend_remove_model.npy")
    
    tractor_dwarf_model = img_data_total - tractor_bkg_model - tractor_blend_model
    tractor_dwarf_model = tractor_dwarf_model[ :, bounding_box_coords[0] : bounding_box_coords[1], bounding_box_coords[2] : bounding_box_coords[3] ]
    
    
    tractor_dwarf_model_rgb = sdss_rgb([tractor_dwarf_model[0],tractor_dwarf_model[1],tractor_dwarf_model[2]], ["g","r","z"], scales=dict(g=(2,6.0), r=(1,3.4), z=(0,2.2)), m=0.03)

    tractor_source_model = np.load(file_path + "/tractor_source_model.npy")
    tractor_source_model = tractor_source_model[ :, bounding_box_coords[0] : bounding_box_coords[1], bounding_box_coords[2] : bounding_box_coords[3] ]

    tractor_source_model_rgb = sdss_rgb([tractor_source_model[0],tractor_source_model[1],tractor_source_model[2]], ["g","r","z"], scales=dict(g=(2,6.0), r=(1,3.4), z=(0,2.2)), m=0.03)

    desi_fiber_pos = np.load(file_path + "/fiber_pix_pos_org.npy")
    #we need to update this 
    print(desi_fiber_pos)
    print(bounding_box_coords)
    desi_fiber_pos = desi_fiber_pos - np.array([bounding_box_coords[0] , bounding_box_coords[2]] )
    print(desi_fiber_pos)
    
    fig,ax = plt.subplots(3,3,figsize = (3*3,3.125*3))
    
    plt.subplots_adjust( wspace = 0.0, hspace = 0.1)
    
    ax[0,0].imshow(img_rgb,origin="lower",zorder = 0)

    for ind in range(2):
        circle = patches.Circle( (desi_fiber_pos[1], desi_fiber_pos[0]),3, color='orange', fill=False, linewidth=1,ls ="-")
        ax[0,ind].add_patch(circle)
    

    ax[0,0].text(0.5, 0.95,r"$grz$ image",color = "white",fontsize = 12,
                          transform=ax[0,0].transAxes, ha = "center", verticalalignment='top')

    #plot what the DESI source is by a circle


    ax[0,1].imshow(tractor_source_model_rgb, origin="lower")
    ax[0,1].text(0.5, 0.95,"DESI Targeted Source\n Tractor Model",color = "white",fontsize = 12,
                          transform=ax[0,1].transAxes, ha = "center", verticalalignment='top')
    

    ax[0,2].set_facecolor("white")
    ax[0,2].set_aspect(img_rgb.shape[0] / img_rgb.shape[1])  # height / width
    ax[0,2].axis("off")

    ax[0, 1].text(
    0.5, 0.075,
    rf"$\mathrm{{mag_{{DR9}}}}$: ({data_scarlet['MAG_G'][index]:.1f}, "
    rf"{data_scarlet['MAG_R'][index]:.1f}, "
    rf"{data_scarlet['MAG_Z'][index]:.1f})",
    ha="center", va="center",
    fontsize=10, color="white",
    transform=ax[0, 1].transAxes
)

    ax[1, 2].text(
    0.5, 0.075,
    rf"$\mathrm{{mag_{{aper}}}}$: ({data_scarlet['MAG_G_APERTURE_COG'][index]:.1f}, "
    rf"{data_scarlet['MAG_R_APERTURE_COG'][index]:.1f}, "
    rf"{data_scarlet['MAG_Z_APERTURE_COG'][index]:.1f})",
    ha="center", va="center",
    fontsize=10, color="white",
    transform=ax[1, 2].transAxes
)

    ax[2, 2].text(
    0.5, 0.075,
    rf"$\mathrm{{mag_{{scarlet}}}}$: ({scar_mags[0]:.1f}, "
    rf"{scar_mags[1]:.1f}, "
    rf"{scar_mags[2]:.1f})",
    ha="center", va="center",
    fontsize=10, color="white",
    transform=ax[2, 2].transAxes
)

    

    for i in range(1,3):
        ax[i,0].text(0.5, 0.95,"Entire Model",color = "white",fontsize = 12,
                        transform=ax[i,0].transAxes, ha = "center", verticalalignment='top')
        ax[i,1].text(0.5, 0.95,"Model Residual",color = "white",fontsize = 12,
                          transform=ax[i,1].transAxes, ha = "center", verticalalignment='top')
        ax[i,2].text(0.5, 0.95,"Dwarf Model",color = "white",fontsize = 12,
                          transform=ax[i,2].transAxes, ha = "center", verticalalignment='top')
    
    ax[1,0].imshow(tractor_model_rgb,origin="lower")
    ax[1,1].imshow(tractor_model_resid_rgb,origin="lower")
    ax[1,2].imshow(tractor_dwarf_model_rgb,origin="lower")
    
    ax[2,0].imshow(scarlet_total_model,origin="lower")
    ax[2,1].imshow(scarlet_resid_rgb,origin="lower")
    ax[2,2].imshow(scarlet_model_rgb,origin="lower")
    
    for axi in np.concatenate(ax):
        axi.set_xticks([])
        axi.set_yticks([])


    def add_box_around_row(ax, row_idx,color = "orange"):
        
        # Choose row index (0 = top, 1 = middle, 2 = bottom)
        # Get bounding box of all axes in that row (in figure coordinates)
        bbox = [ax[row_idx, col].get_position() for col in range(3)]
        
        # Compute combined x0, x1, y0, y1 to cover the whole row
        x0 = min(b.x0 for b in bbox)
        x1 = max(b.x1 for b in bbox)
        y0 = min(b.y0 for b in bbox)
        y1 = max(b.y1 for b in bbox)
        
        # Create a rectangle in figure coordinates
        rect = patches.Rectangle(
            (x0, y0), x1 - x0, y1 - y0,
            transform=fig.transFigure,
            facecolor='none', alpha=1, edgecolor=color, zorder=0, lw = 5
        )

        return rect
        
    # Add to figure
    rect_1 = add_box_around_row(ax, 1,color = "#DC3220")
    fig.patches.append(rect_1)

    rect_2 = add_box_around_row(ax, 2,color = "#005AB5")
    fig.patches.append(rect_2)

    ax[1,0].set_ylabel(r"Tractor",fontsize = 15, fontfamily='monospace',color= "#DC3220")
    ax[2,0].set_ylabel(r"Scarlet",fontsize = 15, fontfamily='monospace',color = "#005AB5")
    
    plt.savefig(f"/global/homes/v/virajvm/DESI2_LOWZ/quenched_fracs_nbs/paper_plots/scarlet_tractor_compare_{index}.png",bbox_inches="tight",dpi=150)
    plt.close()


if __name__ == '__main__':
    
    mpl.rcParams['font.family'] = 'serif'
    mpl.rcParams['axes.linewidth'] = 1.5
    mpl.rcParams['axes.xmargin'] = 1
    mpl.rcParams['xtick.labelsize'] = 'x-large'
    mpl.rcParams['xtick.major.size'] = 5
    mpl.rcParams['xtick.major.width'] = 1.5
    mpl.rcParams['ytick.labelsize'] = 'x-large'
    mpl.rcParams['ytick.major.size'] = 5
    mpl.rcParams['ytick.major.width'] = 1.5
    mpl.rcParams['legend.frameon'] = False
        
    session = requests.Session()
    
    data = Table.read("/pscratch/sd/v/virajvm/catalog_dr1_dwarfs/desi_y1_dwarf_shreds_catalog_filter.fits")
    
    #filter for objects that would be good to do a scarlet model for
    data_scarlet = data[(data["Z"] < 0.01) & (data["SAMPLE"] != "ELG") & (data["LOGM_SAGA_APERTURE_COG"] < 9) & (data["MASKBITS"]==0) & (data["STARFDIST"] > 2) & (data["SGA_D26_NORM_DIST"] > 4) & (data["is_south"] == 1)  ]
    
    print(f"Number of galaxies for scarlet model = {len(data_scarlet)}")

    for index in [6]: #28, 26,100, 9]:
    # index = 26 #100 #9
        plot(index, data_scarlet)

    




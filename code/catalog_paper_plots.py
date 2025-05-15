import glob
from astropy.io import fits
from astropy.wcs import WCS
from desi_lowz_funcs import make_subplots
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from astropy.table import Table
from desi_lowz_funcs import make_subplots, sdss_rgb
from easyquery import Query, QueryMaker
from desi_lowz_funcs import get_remove_flag, _n_or_more_lt, make_subplots, _n_or_more_lt
from tqdm import trange
from matplotlib.colors import LogNorm
from astropy.io import ascii
from astropy import units as u
from astropy.coordinates import SkyCoord
from scipy.stats import median_abs_deviation
from healpy.newvisufunc import projview
import healpy as hp
import matplotlib.cm as cm
import matplotlib

def get_image_summary(ax,data_table , cutout_size = 40, img_folder = "/pscratch/sd/v/virajvm/redo_photometry_plots/all_deshreds_cutouts/", fsize = 12,label=0):
    '''
    This function returns the rgb color image overlayed with circles showing the DR9 sources. The source that is targeted by DESI fiber is colored differently for reference
    '''
    ra = data_table["RA"][0]
    dec = data_table["DEC"][0]
    tgid = data_table["TARGETID"][0]
    zred = data_table["Z"][0]
    rmag = data_table["MAG_R"][0]
    mstar = data_table["LOGM_SAGA"][0]
    save_path = data_table["SAVE_PATH"][0]
    
    img_path_k = img_folder + "image_tgid_%d*.fits"%(tgid) 
    img_path_k = glob.glob(img_path_k)[0]
        
    img_data = fits.open(img_path_k)
    data_arr = img_data[0].data
    wcs = WCS(fits.getheader( img_path_k ))


    ## plot the rgb image of this galaxy with some given size
    rgb_stuff = sdss_rgb([data_arr[0],data_arr[1],data_arr[2]], ["g","r","z"], scales=dict(g=(2,6.0), r=(1,3.4), z=(0,2.2)), m=0.03)


    ## load the source catalog in this object:
    source_cat_f = Table.read(save_path + "/source_cat_f.fits")
    
    #get the pixel locations of these sources 
    sources_f_xpix,sources_f_ypix,_ = wcs.all_world2pix(source_cat_f['ra'].data, source_cat_f['dec'].data, 0,1)

    #identify the source that has DESI fiber!
    all_star_seps = SkyCoord(ra, dec, unit='deg').separation(SkyCoord( source_cat_f["ra"].data , source_cat_f["dec"].data, unit='deg')).arcsec
    
    fiber_xpix = sources_f_xpix[np.argmin(all_star_seps)]
    fiber_ypix = sources_f_ypix[np.argmin(all_star_seps)]
              
    ## plot the image
    ax.imshow(rgb_stuff,origin="lower")
    ax.scatter( sources_f_xpix, sources_f_ypix,facecolor="none",edgecolor = "white",lw =0.75,s=400,ls = "dotted" )
    ax.scatter( fiber_xpix, fiber_ypix,facecolor="none",edgecolor = "r",lw =2,s=400 )
    
    cutout_size = int(cutout_size/0.262)
    #this makes the it an integer!
    start = (350 - cutout_size) // 2
    end = start + cutout_size
    
    ax.set_xlim([start,end])
    ax.set_ylim([start,end])
    
    ax.set_xticks([])
    ax.set_yticks([])

    ##indicate some text detialing some properties?
    # ax.text(0.05,0.85,r"$z = %.2f$"%(data_table["Z"][0]),size = fsize,transform=ax.transAxes, verticalalignment='top',color = "white")
    # ax.text(0.05,0.8,r"$r_{\rm mag,DR9} = %.1f$, FRACFLUX_R = %.2f, "%(rmag, data_table["FRACFLUX_R"][0]),size = fsize,transform=ax.transAxes, verticalalignment='top',color = "red",bbox=dict(facecolor='black', alpha=0.75, pad=5) )
    ax.text(0.025,0.95,r"%d"%(label),size = 20,transform=ax.transAxes, verticalalignment='top',color = "white",bbox=dict(facecolor='black', alpha=0.8, pad=2) )
    
    ax.set_title(r"$r_{\rm mag,DR9} = %.1f$, FRACFLUX_R = %.2f "%(rmag, data_table["FRACFLUX_R"][0]),size = fsize, color = "firebrick")
    
    return


def make_shred_panel(bgsb_shreds, bgsf_shreds, elg_shreds):


    mask1 = (bgsb_shreds["RA"]== 133.14261025691368)
    data_1 = bgsb_shreds[mask1]
    
    mask2 = (bgsf_shreds["RA"] == 42.61540995579803)
    data_2 = bgsf_shreds[mask2]
    
    mask3 = (elg_shreds["RA"] == 37.85279871518745)
    data_3 = elg_shreds[mask3]
    
    # mask0 = (bgsb_shreds["RA"] == 265.3569194336881)
    # data_0 = bgsb_shreds[mask0]
    mask0 = (bgsb_shreds["TARGETID"]== 39627752084603180) #39627752084603392)
    data_0 = bgsb_shreds[mask0]

    #this is the super star source that would be good to add as it is clearly fragmented and has a low fracflux value as so extended
    # bgsb_shreds[bgsb_shreds["TARGETID"] == 39627685319676194]
    
    axs = make_subplots(ncol = 4, nrow = 1, col_spacing = 0.4)

    get_image_summary(axs[0], data_0, cutout_size = 60, img_folder = "/pscratch/sd/v/virajvm/redo_photometry_plots/all_deshreds_cutouts/",fsize = 11,label = 1)
    get_image_summary(axs[1], data_1, cutout_size = 40, img_folder = "/pscratch/sd/v/virajvm/redo_photometry_plots/all_deshreds_cutouts/",fsize = 11,label = 2)
    get_image_summary(axs[2], data_2, cutout_size = 40, img_folder = "/pscratch/sd/v/virajvm/redo_photometry_plots/all_deshreds_cutouts/",fsize = 11,label = 3)
    get_image_summary(axs[3], data_3, cutout_size = 40, img_folder = "/pscratch/sd/v/virajvm/redo_photometry_plots/all_deshreds_cutouts/",fsize = 11,label = 4)
    
    plt.savefig("/global/homes/v/virajvm/DESI2_LOWZ/quenched_fracs_nbs/paper_plots/shreds_examples.pdf",bbox_inches="tight")
    
    plt.show()

    return



def make_pcnn_completeness():
    '''
    This function makes the completeness and purity of pCNN threshold plot
    '''

    #this is the entire shredded catalog that also includes PCNN_FRAGMENT COLUMN
    data_main = Table.read("/pscratch/sd/v/virajvm/catalog_dr1_dwarfs/desi_y1_dwarf_shreds_catalog_v3.fits")
    
    frag_comp = np.load("/pscratch/sd/v/virajvm/catalog_dr1_dwarfs/shred_classifier_output/fragment_completeness.npy")
    good_impure = np.load("/pscratch/sd/v/virajvm/catalog_dr1_dwarfs/shred_classifier_output/good_impurity.npy")
    #this grid is defined in the shred_classifier.py file
    thresh_grid = np.linspace(0., 0.999, 40)
        
    # Create the figure and gridspec
    fig = plt.figure(figsize=(4, 4*4/3))
    gs = fig.add_gridspec(2, 1, height_ratios=[1, 3], hspace=0.0)
    
    # Top: histogram
    ax_hist = fig.add_subplot(gs[0])
    
    # Bottom: line plot
    ax_plot = fig.add_subplot(gs[1], sharex=ax_hist)
    
    # Plot histogram
    ax_hist.hist(data_main["PCNN_FRAGMENT"], density=True,bins=np.linspace(0,1,20), color='gray', alpha=0.7)
    # ax_hist.set_ylabel('')
    ax_hist.tick_params(labelbottom=False)  # Hide x labels on hist
    ax_hist.set_yticks([])
    ax_hist.set_ylabel(r"$N(p_{\rm CNN})$",fontsize = 13)
    
    color_frag = "mediumblue"
    color_good = "firebrick"
    
    # Plot line
    ax_plot.plot(thresh_grid, frag_comp,lw = 3,color = color_frag)
    ax_plot.plot(thresh_grid, good_impure,lw = 3,color = color_good)
    
    # ax_plot.set_xlabel('X')
    # ax_plot.set_ylabel('Y')
    ax_plot.set_xlim([0,1])
    ax_plot.set_ylim([0,1])
    ax_plot.set_xlabel(r"Threshold $p_{\rm CNN}$",fontsize = 15)
    
    ax_plot.text(0.325,0.225,r"$\frac{N( > p_{\rm CNN}, \text{Not Fragment}  ) }{N( > p_{\rm CNN}  )}$",fontsize = 15,
                 color = color_good)
    
    ax_plot.text(0.225,0.825,r"$\frac{N( > p_{\rm CNN}, \text{ Fragment}  ) }{N( \text{Fragment}  )}$",fontsize = 15,
                 color = color_frag)
    
    plt.savefig("/global/homes/v/virajvm/DESI2_LOWZ/quenched_fracs_nbs/paper_plots/pcnn_threshold.pdf",dpi = 300,bbox_inches="tight")
    
    plt.close()

    retrun


def get_plotting_inds(main_cat):
    '''
    Function that randomly selects rows from a given galaxy sample in a catalo
    '''

    # Get unique samples
    samples = ['BGS_BRIGHT', 'BGS_FAINT', 'ELG'][::-1]
    
    # Dictionary to hold random indices
    random_indices = []
    
    for samp in samples:
        # Find the row indices where sample matches
        mask = main_cat['SAMPLE'] == samp
        matching_indices = np.where(mask)[0]
        # Randomly pick one index
        rand_idx = np.random.choice(matching_indices)
        random_indices.append(rand_idx)
    
    print(random_indices)

    return np.array(random_indices)

def make_img_s_pcnn_panels():
    '''
    This function makes the pCNN IMG and IMG-S panel image
    '''
    
    #this is the entire shredded catalog that also includes PCNN_FRAGMENT COLUMN
    data_main = Table.read("/pscratch/sd/v/virajvm/catalog_dr1_dwarfs/desi_y1_dwarf_shreds_catalog_v3.fits")

    data_cnn_shred = data_main[data_main["PCNN_FRAGMENT"] >= 0.3]
    data_cnn_clean = data_main[data_main["PCNN_FRAGMENT"] < 0.3]

    #choosing the random plot inds
    clean_inds = np.array([32495, 13180, 1629]) #get_plotting_inds(data_cnn_clean)
    shred_inds = np.array([19400, 12444, 715]) # get_plotting_inds(data_cnn_shred)

    ### making the fragment source panels
    fig, ax = make_subplots(ncol =2, nrow = 3, col_spacing = 0.05, row_spacing = 0.35,plot_size = 2, return_fig=True )
        
    fig.text(0.175, 0.425, r'$p_{\rm CNN} \geq 0.3$', ha='center', va='top', fontsize=20)
    
    ax[4].set_title(r"IMG",fontsize = 16)
    ax[5].set_title(r"IMG - S",fontsize = 16)
    
    shred_tgids = data_cnn_shred[shred_inds]["TARGETID"].data
    
    for j in range(3):
        temp = np.load(f"/pscratch/sd/v/virajvm/catalog_dr1_dwarfs/shred_classifier_output/shred_classifier_input_images/image_{shred_tgids[j]}.npy")
        #we crop to the center 64
        size = 64
        start = (96 - size) // 2
        end = start + size
        temp = temp[:, start:end, start:end]
        #make the rgb image!
        rgb1 = temp[:3]
        #make the sdss rgb image of this!
        rgb1 = sdss_rgb([rgb1[0],rgb1[1],rgb1[2]], ["g","r","z"], scales=dict(g=(2,6.0), r=(1,3.4), z=(0,2.2)), m=0.03)
        rgb2 = temp[3:6]
        rgb2 = sdss_rgb([rgb2[0],rgb2[1],rgb2[2]], ["g","r","z"], scales=dict(g=(2,6.0), r=(1,3.4), z=(0,2.2)), m=0.03)
        ax[2*j].imshow(rgb1)
        ax[2*j+1].imshow(rgb2)
        ax[2*j].scatter( 32, 32,facecolor="none",edgecolor = "r",lw =1,s=400, ls = "--" )
        ax[2*j+1].scatter( 32, 32,facecolor="none",edgecolor = "r",lw =1,s=400, ls = "--" )    
    for axi in ax:
        axi.set_yticks([])
        axi.set_xticks([])
    plt.savefig("/global/homes/v/virajvm/DESI2_LOWZ/quenched_fracs_nbs/paper_plots/shred_egs_pcnn.pdf",dpi=300,bbox_inches="tight")
    plt.close()

    ##making the clean source panels

    fig, ax = make_subplots(ncol =2, nrow = 3, col_spacing = 0.05, row_spacing = 0.35,plot_size = 2, return_fig=True )
    fig.text(0.175, 0.425, r'$p_{\rm CNN} < 0.3$', ha='center', va='top', fontsize=20)
    
    ax[4].set_title(r"IMG",fontsize = 16)
    ax[5].set_title(r"IMG - S",fontsize = 16)

    clean_tgids = data_cnn_clean[clean_inds]["TARGETID"].data
    
    for j in range(3):
        temp = np.load(f"/pscratch/sd/v/virajvm/catalog_dr1_dwarfs/shred_classifier_output/shred_classifier_input_images/image_{clean_tgids[j]}.npy")
    
        #we crop to the center 64
        size = 64
        start = (96 - size) // 2
        end = start + size
        temp = temp[:, start:end, start:end]
    
        #make the rgb image!
        rgb1 = temp[:3]
        #make the sdss rgb image of this!
        rgb1 = sdss_rgb([rgb1[0],rgb1[1],rgb1[2]], ["g","r","z"], scales=dict(g=(2,6.0), r=(1,3.4), z=(0,2.2)), m=0.03)
     
        rgb2 = temp[3:6]
        rgb2 = sdss_rgb([rgb2[0],rgb2[1],rgb2[2]], ["g","r","z"], scales=dict(g=(2,6.0), r=(1,3.4), z=(0,2.2)), m=0.03)
    
        ax[2*j].imshow(rgb1)
        ax[2*j+1].imshow(rgb2)
        ax[2*j].scatter( 32, 32,facecolor="none",edgecolor = "r",lw =1,s=400, ls = "--" )
        ax[2*j+1].scatter( 32, 32,facecolor="none",edgecolor = "r",lw =1,s=400, ls = "--" )
    for axi in ax:
        axi.set_yticks([])
        axi.set_xticks([])
    plt.savefig("/global/homes/v/virajvm/DESI2_LOWZ/quenched_fracs_nbs/paper_plots/clean_egs_pcnn.pdf",dpi=300,bbox_inches="tight")
    plt.close()


def make_shred_frac_plot():
    '''
    This function makes the primary clean and shred catalogs we work with!
    '''

    bgsb_list = Table.read("/pscratch/sd/v/virajvm/catalog_dr1_dwarfs/iron_bgs_bright_filter_zsucc_zrr02_allfracflux.fits")
    bgsf_list = Table.read("/pscratch/sd/v/virajvm/catalog_dr1_dwarfs/iron_bgs_faint_filter_zsucc_zrr03_allfracflux.fits")
    elg_list = Table.read("/pscratch/sd/v/virajvm/catalog_dr1_dwarfs/iron_elg_filter_zsucc_zrr05_allfracflux.fits")
    lowz_list = Table.read("/pscratch/sd/v/virajvm/catalog_dr1_dwarfs/iron_lowz_filter_zsucc_zrr03.fits")

    dwarf_mask_bgsb = (bgsb_list["LOGM_SAGA"] < 9.5) 
    dwarf_mask_bgsf = (bgsf_list["LOGM_SAGA"] < 9.5) 
    dwarf_mask_lowz = (lowz_list["LOGM_SAGA"] < 9.5) 
    dwarf_mask_elg = (elg_list["LOGM_SAGA"] < 9.5)   

    bgsb_dwarfs = bgsb_list[dwarf_mask_bgsb]
    bgsf_dwarfs = bgsf_list[dwarf_mask_bgsf]
    elg_dwarfs = elg_list[dwarf_mask_elg]
    lowz_dwarfs = lowz_list[dwarf_mask_lowz]

    
    #identifying the objects that are likely shreds 
    fracflux_grz = [f"FRACFLUX_{b}" for b in "GRZ"]

    remove_queries = [Query(_n_or_more_lt(fracflux_grz, 2, 0.2)) ]
    
    # note that the this is n_or_more_LT!! so be careful about that!
    #these are masks for objects that did not satisfy the above condition!
    shred_bgsb_mask = get_remove_flag(bgsb_dwarfs, remove_queries) == 0
    shred_bgsf_mask = get_remove_flag(bgsf_dwarfs, remove_queries) == 0
    shred_elg_mask = get_remove_flag(elg_dwarfs, remove_queries) == 0
    shred_lowz_mask = get_remove_flag(lowz_dwarfs, remove_queries) == 0
    
    ## PLOTTING THE FRACTION OF GALAXIES THAT ARE LIKELY SHREDS/BAD PHOTOMETRY
    zgrid = np.arange(0.00,0.125,0.005)
    #the stellar mass bins of 0.5 dex
    mstar_grid = np.arange(5.75, 9.5,0.25)

    def get_shred_frac(tot_cat, subset_mask,  low, hi, rel_col="Z"):
        '''
        Given the total catalog and mask of objects to look at, computes the fraction in an interval of the relevant column (e.g., redshift or stellar mass)
        '''

        tot_count_bin = len(tot_cat[(tot_cat[rel_col]< hi) & (tot_cat[rel_col] > low)  ])
        subset_count_bin = len(tot_cat[(tot_cat[rel_col]< hi) & (tot_cat[rel_col] > low) & subset_mask  ])

        ##adopt some minimum count here, also show some error here??
        
        if tot_count_bin > 10:
            return subset_count_bin / tot_count_bin
        else:
            return None

    #these are the fraction of objects as a function of redshift that are classified as likely fragments by just the FRACFLUX cut!
    shred_frac_bgsb_1 = []
    shred_frac_bgsf_1 = []
    shred_frac_elg_1 = []
    shred_frac_lowz_1 = []
    
    for i in trange(len(zgrid)-1):
        zlow = zgrid[i]
        zhi = zgrid[i+1]

        shred_frac_bgsb_1.append(   get_shred_frac(bgsb_dwarfs, shred_bgsb_mask ,zlow, zhi, rel_col = "Z" )  ) 
        shred_frac_bgsf_1.append(   get_shred_frac(bgsf_dwarfs, shred_bgsf_mask, zlow, zhi, rel_col = "Z")  ) 
        shred_frac_elg_1.append(   get_shred_frac(elg_dwarfs, shred_elg_mask, zlow, zhi, rel_col = "Z")  ) 
        shred_frac_lowz_1.append(   get_shred_frac(lowz_dwarfs, shred_lowz_mask, zlow, zhi, rel_col = "Z")  ) 

    ## now we look at the fraction of objects that are classified as fragments by the fracflux cut AND the PCNN!
    #load the shredded cat and check these numbers add up ... 
    ##THE REDSHIFT DIFFERENCE WITH PCNN FILTER IS NOT VERY NOTICEABLE AND NOT IMPORTANT AND SO WE DO NOT INCLUDE IT HERE
    
    shred_cat = Table.read("/pscratch/sd/v/virajvm/catalog_dr1_dwarfs/desi_y1_dwarf_shreds_catalog_v3.fits")
    print(f"In shredded cat, BGS_BRIGHT={len(shred_cat[shred_cat['SAMPLE'] == 'BGS_BRIGHT'  ])}, BGS_FAINT={len(shred_cat[shred_cat['SAMPLE'] == 'BGS_FAINT'  ])}, ELG={len(shred_cat[shred_cat['SAMPLE'] == 'ELG'  ])}, LOWZ={len(shred_cat[shred_cat['SAMPLE'] == 'LOWZ'  ])}")
    
    print(f"In plotting catalog, BGS_BRIGHT={len(bgsb_dwarfs[shred_bgsb_mask])}, BGS_FAINT={len(bgsf_dwarfs[shred_bgsf_mask])}, ELG={len(elg_dwarfs[shred_elg_mask])}, LOWZ={len(lowz_dwarfs[shred_lowz_mask])} ")

    #let us confirm that they are in the same order!
    print(  np.max(  np.abs( shred_cat[shred_cat['SAMPLE'] == 'BGS_BRIGHT']["TARGETID"].data - bgsb_dwarfs[shred_bgsb_mask]["TARGETID"].data ) ))
    print(  np.max(  np.abs( shred_cat[shred_cat['SAMPLE'] == 'BGS_FAINT']["TARGETID"].data - bgsf_dwarfs[shred_bgsf_mask]["TARGETID"].data ) )) 
    print(  np.max(  np.abs( shred_cat[shred_cat['SAMPLE'] == 'ELG']["TARGETID"].data - elg_dwarfs[shred_elg_mask]["TARGETID"].data ) )) 
    print(  np.max(  np.abs( shred_cat[shred_cat['SAMPLE'] == 'LOWZ']["TARGETID"].data - lowz_dwarfs[shred_lowz_mask]["TARGETID"].data ) )) 

    ## now we measure panel 2,
    # shred_frac_bgsb_2 = []
    # shred_frac_bgsf_2 = []
    # shred_frac_elg_2 = []
    # shred_frac_lowz_2 = []

    # #we need the mask to include the pcnn cut, let us just initialize an array of zeros
    bgsb_dwarfs_pcnn = np.zeros(len(bgsb_dwarfs))
    bgsf_dwarfs_pcnn = np.zeros(len(bgsf_dwarfs))
    elg_dwarfs_pcnn = np.zeros(len(elg_dwarfs))
    lowz_dwarfs_pcnn = np.zeros(len(lowz_dwarfs))

    #now we fill this with pcnn values for just shredded objects
    bgsb_dwarfs_pcnn[ shred_bgsb_mask ] = shred_cat[shred_cat['SAMPLE'] == 'BGS_BRIGHT']["PCNN_FRAGMENT"].data
    bgsf_dwarfs_pcnn[ shred_bgsf_mask ] = shred_cat[shred_cat['SAMPLE'] == 'BGS_FAINT']["PCNN_FRAGMENT"].data
    elg_dwarfs_pcnn[ shred_elg_mask ] = shred_cat[shred_cat['SAMPLE'] == 'ELG']["PCNN_FRAGMENT"].data
    lowz_dwarfs_pcnn[ shred_lowz_mask ] = shred_cat[shred_cat['SAMPLE'] == 'LOWZ']["PCNN_FRAGMENT"].data

    # #and now we can use this in the mask!
    print( len(bgsb_dwarfs[shred_bgsb_mask]), len(bgsb_dwarfs[shred_bgsb_mask & (bgsb_dwarfs_pcnn >= 0.3)] )  )
    print( len(bgsf_dwarfs[shred_bgsf_mask]), len(bgsf_dwarfs[shred_bgsf_mask&(bgsf_dwarfs_pcnn >= 0.3)]  ) )
    print( len(elg_dwarfs[shred_elg_mask]), len(elg_dwarfs[shred_elg_mask&(elg_dwarfs_pcnn >= 0.3)]   ) )
    print( len(lowz_dwarfs[shred_lowz_mask]), len(lowz_dwarfs[shred_lowz_mask&(lowz_dwarfs_pcnn >= 0.3)] ) )
    

    # for i in trange(len(zgrid)-1):
    #     zlow = zgrid[i]
    #     zhi = zgrid[i+1]

    #     shred_frac_bgsb_2.append(   get_shred_frac(bgsb_dwarfs, shred_bgsb_mask&(bgsb_dwarfs_pcnn >= 0.3) ,zlow, zhi, rel_col = "Z" )  ) 
    #     shred_frac_bgsf_2.append(   get_shred_frac(bgsf_dwarfs, shred_bgsf_mask&(bgsf_dwarfs_pcnn >= 0.3), zlow, zhi, rel_col = "Z")  ) 
    #     shred_frac_elg_2.append(   get_shred_frac(elg_dwarfs, shred_elg_mask&(elg_dwarfs_pcnn >= 0.3), zlow, zhi, rel_col = "Z")  ) 
    #     shred_frac_lowz_2.append(   get_shred_frac(lowz_dwarfs, shred_lowz_mask&(lowz_dwarfs_pcnn >= 0.3), zlow, zhi, rel_col = "Z")  ) 

    ### get the stellar mass difference

    shred_frac_bgsb_ms = []
    shred_frac_bgsf_ms = []
    shred_frac_elg_ms = []
    shred_frac_lowz_ms = []

    shred_frac_bgsb_ms_cnn = []
    shred_frac_bgsf_ms_cnn = []
    shred_frac_elg_ms_cnn = []
    shred_frac_lowz_ms_cnn = []

    for i in trange(len(mstar_grid)-1):
        mlow = mstar_grid[i]
        mhi = mstar_grid[i+1]

        shred_frac_bgsb_ms.append(   get_shred_frac(bgsb_dwarfs, shred_bgsb_mask ,mlow, mhi, rel_col = "LOGM_SAGA" )  ) 
        shred_frac_bgsf_ms.append(   get_shred_frac(bgsf_dwarfs, shred_bgsf_mask, mlow, mhi, rel_col = "LOGM_SAGA")  ) 
        shred_frac_elg_ms.append(   get_shred_frac(elg_dwarfs, shred_elg_mask , mlow, mhi, rel_col = "LOGM_SAGA")  ) 
        shred_frac_lowz_ms.append(   get_shred_frac(lowz_dwarfs, shred_lowz_mask, mlow, mhi, rel_col = "LOGM_SAGA")  ) 
        
        shred_frac_bgsb_ms_cnn.append(   get_shred_frac(bgsb_dwarfs, shred_bgsb_mask & (bgsb_dwarfs_pcnn >= 0.3) ,mlow, mhi, rel_col = "LOGM_SAGA" )  ) 
        shred_frac_bgsf_ms_cnn.append(   get_shred_frac(bgsf_dwarfs, shred_bgsf_mask & (bgsf_dwarfs_pcnn >= 0.3), mlow, mhi, rel_col = "LOGM_SAGA")  ) 
        shred_frac_elg_ms_cnn.append(   get_shred_frac(elg_dwarfs, shred_elg_mask & (elg_dwarfs_pcnn >= 0.3), mlow, mhi, rel_col = "LOGM_SAGA")  ) 
        shred_frac_lowz_ms_cnn.append(   get_shred_frac(lowz_dwarfs, shred_lowz_mask & (lowz_dwarfs_pcnn >= 0.3), mlow, mhi, rel_col = "LOGM_SAGA")  ) 


        ## let us make an additional cut of SGA DIST > 2

        

        
    
    bgs_col = "#648FFF" #DC267F
    lowz_col = "#DC267F"
    elg_col = "#FFB000"
    
    zcens = 0.5*(zgrid[1:] + zgrid[:-1])
    ms_cens = 0.5*(mstar_grid[1:] + mstar_grid[:-1])

    print(shred_frac_bgsb_ms)

    # ##this is going to be a 2 panel subplot 
    # ## first panel is the fraction of sources identified as likely fragments in dwarf galaxy catalog
    # ## second panel is going to be fraction of sources that are identified as likely fragments and have further confirmed to be shreds by pcnn>0.3, we will show this per sample as well
    # ##the third panel shows the fration of shred as a function of stellar mass for each sample (a simple color-based stellar mass)

    from desi_lowz_funcs import make_subplots

    fig,ax = make_subplots(ncol = 1, nrow = 2,return_fig=True,row_spacing=0.75)

    ax[1].plot(zcens, shred_frac_bgsb_1,label = "BGS Bright",lw = 3,color = bgs_col,ls = "-",alpha = 0.75)
    ax[1].plot(zcens, shred_frac_bgsf_1,label = "BGS Faint",lw = 3,color = "r",ls = "-",alpha = 0.75)
    ax[1].plot(zcens, shred_frac_lowz_1,label = "LOWZ",lw = 3,color = lowz_col,ls = "-",alpha = 0.75)
    ax[1].plot(zcens, shred_frac_elg_1,label = "ELG",lw = 3,color = elg_col,ls = "-",alpha = 0.75)

    ls = "-"
    lw = 3
    
    ax[0].plot(ms_cens, shred_frac_bgsb_ms,ls= ls,lw = lw,color = bgs_col,alpha = 0.75)
    ax[0].plot(ms_cens, shred_frac_bgsf_ms,ls= ls,lw = lw,color = "r",alpha = 0.75)
    ax[0].plot(ms_cens, shred_frac_lowz_ms,ls= ls,lw = lw,color = lowz_col,alpha = 0.75)
    ax[0].plot(ms_cens, shred_frac_elg_ms,ls= ls,lw = lw,color = elg_col,alpha = 0.75)
    ax[0].plot([0,1],[0,1],ls=ls, lw = lw, color = "k", label = r"FRACFLUX cut")

    # ls = "--"
    # lw = 1.5
    # ax[0].plot(ms_cens, shred_frac_bgsb_ms_cnn,ls= ls,lw = lw,color = bgs_col,alpha = 0.75)
    # ax[0].plot(ms_cens, shred_frac_bgsf_ms_cnn,ls= ls,lw = lw,color = "r",alpha = 0.75)
    # ax[0].plot(ms_cens, shred_frac_lowz_ms_cnn,ls= ls,lw = lw,color = lowz_col,alpha = 0.75)
    # ax[0].plot(ms_cens, shred_frac_elg_ms_cnn,ls= ls,lw = lw,color = elg_col,alpha = 0.75)
    # ax[0].plot([0,1],[0,1],ls=ls, lw = lw, color = "k", label = r"FRACFLUX cut, $p_{\rm CNN} \geq 0.3$")

    
    ax[1].legend(fontsize = 12)
    # ax[0].legend(fontsize = 12)
    
    ax[1].set_xlim([0,0.1])
    
    ax[0].set_xlim([6,9.25])
    
    for axi in ax:
        axi.set_ylim([0,1])
        
    ax[1].set_xlabel("$z$ (Redshift)",fontsize = 15)
    ax[0].set_xlabel(r"$\log_{10} M_{\star}/M_{\odot}$",fontsize = 15)
    
    ax[1].set_ylabel(r"Likely Fragment Fraction",fontsize = 14)
    ax[0].set_ylabel(r"Likely Fragment Fraction",fontsize = 14)
    
    # plt.grid(ls=":",color = "lightgrey",alpha = 0.5)
    plt.savefig("/global/homes/v/virajvm/DESI2_LOWZ/quenched_fracs_nbs/paper_plots/frac_shreds.pdf", bbox_inches="tight")
    plt.close()
        
    return 


def get_offsets(file_path = "/pscratch/sd/v/virajvm/catalog_dr1_dwarfs/iron_photometry/iron_BGS_BRIGHT_shreds_catalog_w_aper_mags.fits"):
    data = Table.read(file_path)
    
    all_fracfluxs = np.concatenate( (data["FRACFLUX_G"],data["FRACFLUX_R"],data["FRACFLUX_Z"]  ) ) 
    all_offsets = np.concatenate( (data["MAG_G_APERTURE"].data - data["MAG_G"].data ,data["MAG_R_APERTURE"].data - data["MAG_R"].data ,data["MAG_Z_APERTURE"].data - data["MAG_Z"].data   ) )

    aper_mags = np.concatenate( (data["MAG_G_APERTURE"].data,data["MAG_R_APERTURE"].data,data["MAG_Z_APERTURE"].data) )
    
    all_fracfluxs = all_fracfluxs[~np.isnan(all_offsets)]
    all_offsets = all_offsets[~np.isnan(all_offsets)]
    aper_mags = all_offsets[~np.isnan(all_offsets)]
    
    return all_fracfluxs, np.array(all_offsets), aper_mags


    
def get_delta_mag_fracflux_plot():
    '''
    We make a plot of change in magnitude as a function of fracflux!!
    '''

    ##load the shred aperture results
    ff_bgsb, dm_bgsb, _ = get_offsets("/pscratch/sd/v/virajvm/catalog_dr1_dwarfs/iron_photometry/iron_BGS_BRIGHT_shreds_catalog_w_aper_mags.fits")
    ff_bgsf, dm_bgsf, _ = get_offsets("/pscratch/sd/v/virajvm/catalog_dr1_dwarfs/iron_photometry/iron_BGS_FAINT_shreds_catalog_w_aper_mags.fits")
    ff_elg, dm_elg, _ = get_offsets("/pscratch/sd/v/virajvm/catalog_dr1_dwarfs/iron_photometry/iron_ELG_shreds_catalog_w_aper_mags.fits")
    ff_lowz, dm_lowz, _ = get_offsets("/pscratch/sd/v/virajvm/catalog_dr1_dwarfs/iron_photometry/iron_LOWZ_shreds_catalog_w_aper_mags.fits")

    #load the clean aperture results. The number of galaxies here is meant to be the same as the shred ones to have a one-to-one comparison 
    ff_bgsb_c, dm_bgsb_c, _ = get_offsets("/pscratch/sd/v/virajvm/catalog_dr1_dwarfs/iron_photometry/iron_BGS_BRIGHT_clean_catalog_w_aper_mags.fits")
    ff_bgsf_c, dm_bgsf_c, _ = get_offsets("/pscratch/sd/v/virajvm/catalog_dr1_dwarfs/iron_photometry/iron_BGS_FAINT_clean_catalog_w_aper_mags.fits")
    ff_elg_c, dm_elg_c, _ = get_offsets("/pscratch/sd/v/virajvm/catalog_dr1_dwarfs/iron_photometry/iron_ELG_clean_catalog_w_aper_mags.fits")
    ff_lowz_c, dm_lowz_c, _ = get_offsets("/pscratch/sd/v/virajvm/catalog_dr1_dwarfs/iron_photometry/iron_LOWZ_clean_catalog_w_aper_mags.fits")
    

    ##we combine this all!
    ff_all = np.concatenate((ff_bgsb, ff_bgsf, ff_elg, ff_lowz))
    dm_all = np.concatenate((dm_bgsb, dm_bgsf, dm_elg, dm_lowz))

    print(len(ff_all))

    ff_all_c = np.concatenate((ff_bgsb_c, ff_bgsf_c, ff_elg_c, ff_lowz_c))
    dm_all_c = np.concatenate((dm_bgsb_c, dm_bgsf_c, dm_elg_c, dm_lowz_c))

    print(len(ff_all_c))

    ff_all = np.concatenate( (ff_all, ff_all_c) )
    dm_all = np.concatenate( (dm_all, dm_all_c) )

    ax = make_subplots(ncol =1, nrow = 1)
    
    ax[0].hlines(y = 0,xmin=1e-2, xmax = 5,ls = "-",color = "k",lw =1 )
    bins = 50
    
    ax[0].hist2d(ff_all, dm_all, range = ( (1e-2, 5,), (-5, 2) ),
                 bins = [ np.logspace(-2, np.log10(5), bins), np.linspace(-5,2,bins) ],norm=LogNorm(vmin=10,vmax = 1000),
                cmap = "BuPu")
    
    ax[0].vlines(x = 0.2, ymin = -5, ymax = 2, color= "k", ls = "--",lw = 1)
    ax[0].set_xlabel(r"FRACFLUX",fontsize = 13)
    ax[0].set_ylabel(r"$\Delta m$ = mag$_{\rm aper}$ - mag$_{\rm DR9}$",fontsize = 15)
    
    ax[0].set_xlim([1e-2, 5])
    ax[0].set_ylim([-5,2])
    ax[0].set_xscale("log")
    
    plt.savefig("/global/homes/v/virajvm/DESI2_LOWZ/quenched_fracs_nbs/paper_plots/fracflux_delta_mag.pdf",bbox_inches="tight", dpi = 300)
    plt.close()



def measure_bias_scatter(quant_1, quant_2):
    '''
    Meausure the median of quant_1 - quant_2 and the scatter in this difference. We restrict ourselves to objects
    '''

    quant_1f = quant_1[~np.isnan(quant_1) & ~np.isnan(quant_2) ]
    quant_2f = quant_2[~np.isnan(quant_2) & ~np.isnan(quant_2) ]

    med_val = np.median(quant_1f - quant_2f)
    scatters = quant_1f - quant_2f - med_val

    sigma =  median_abs_deviation(scatters, scale='normal')

    print(med_val, sigma)
    return med_val, sigma

def make_stellar_mass_comparison_plot():
    '''
    This function makes the stellar mass comparison plot
    '''

    clean_cat = Table.read("/pscratch/sd/v/virajvm/catalog_dr1_dwarfs/desi_y1_dwarf_clean_catalog_v2.fits")

    print(len(clean_cat))
    clean_cat = clean_cat[clean_cat["LOGM_SAGA"] < 9.5]
    print(len(clean_cat))
    
    ##match the clean cat with other catalogs
    gswlc_cat = ascii.read("/pscratch/sd/v/virajvm/desi2_lowz_data/catalogs/GSWLC-X2.dat")
    iron = SkyCoord(np.array(clean_cat["RA"])*u.degree, np.array(clean_cat["DEC"])*u.degree  )
    gswlc = SkyCoord(np.array(gswlc_cat["RA"])*u.degree, np.array(gswlc_cat["DEC"])*u.degree  )
    idx, d2d, _ = iron.match_to_catalog_sky(gswlc)
    clean_cat_gswlc_match = clean_cat[d2d.arcsec < 1]
    gswlc_match = gswlc_cat[idx][d2d.arcsec < 1]

    ##these are stellar masses from Hu Zhou XMPG paper. They also use CIGALE here and no AGN is used
    hu_cat= Table.read("/global/cfs/cdirs/desi/users/dscholte/data_to_share/sample_catalog_viraj_29052024.fits")
    iron = SkyCoord(np.array(clean_cat["RA"])*u.degree, np.array(clean_cat["DEC"])*u.degree  )
    hu = SkyCoord(np.array(hu_cat["RA"])*u.degree, np.array(hu_cat["DEC"])*u.degree  )
    idx, d2d, _ = iron.match_to_catalog_sky(hu)
    clean_cat_hu_match = clean_cat[d2d.arcsec < 1]
    hu_match = hu_cat[idx][d2d.arcsec < 1]

    ###FASTSPECFIT
    print("Reading fastspecfit!")
    iron_vac = fits.open("/global/cfs/cdirs/desi/public/dr1/vac/dr1/fastspecfit/iron/v2.1/catalogs/fastspec-iron.fits")
    fspec_mstar = iron_vac[1].data["LOGMSTAR"]
    fspec_ra = iron_vac[2].data["RA"]
    fspec_dec = iron_vac[2].data["DEC"]
    catalog = SkyCoord(ra= fspec_ra* u.degree, dec= fspec_dec*u.degree )
    c = SkyCoord(ra=np.array(clean_cat["RA"])*u.degree, dec=np.array(clean_cat["DEC"])*u.degree )
    idx, d2d, d3d = c.match_to_catalog_sky(catalog)
    fspec_mstar_f = fspec_mstar[idx][d2d.arcsec < 1]
    clean_cat_fspec_match = clean_cat[d2d.arcsec < 1]
    print("Finished matching fastspecfit!")

    ## loading the cosmos 2020 catalog
    cos2020_data = np.load("/pscratch/sd/v/virajvm/desi2_lowz_data/catalogs/cosmos2020_data.npy")
    iron = SkyCoord(np.array(clean_cat["RA"])*u.degree, np.array(clean_cat["DEC"])*u.degree  )
    cos = SkyCoord( cos2020_data[0]*u.degree, cos2020_data[1]*u.degree  )
    idx, d2d, _ = iron.match_to_catalog_sky(cos)
    clean_cat_cos_match = clean_cat[d2d.arcsec < 1]
    cos2020_mstar = cos2020_data[2][idx][d2d.arcsec < 1]

    ##make the plot

    ax = make_subplots(ncol = 5,nrow = 1,col_spacing = 0.25)

    title_size = 14

    xmstar = "LOGM_SAGA"
    cmap = "BuPu"
    
    vmin = 1
    vmax = 1000
    
    # ax[0].set_title(r"CIGALE (no AGN)",fontsize = title_size )
    # ax[0].hist2d(clean_cat[xmstar][cigale_mask],clean_cat["LOGM_CIGALE"][cigale_mask],range= ( (6,9.5),(6,9.5)),bins=50,norm=LogNorm(vmin=vmin,vmax=vmax),cmap=cmap )

    xpos = 7.4
    ypos = 6.67
    fsize = 14

    ax_id = 0
    ax[ax_id].set_title(r"CIGALE (no AGN)",fontsize = title_size )
    h, xedges, yedges, im=ax[ax_id].hist2d(clean_cat_hu_match[xmstar],hu_match["LOGMSTAR_HU"],range= ( (6,9.5),(6,9.5)),bins= 50,norm=LogNorm(vmin=vmin,vmax=vmax) ,cmap=cmap)

    bias, scatter = measure_bias_scatter(clean_cat_hu_match[xmstar].data,hu_match["LOGMSTAR_HU"])
    
    ax[ax_id].text( xpos,ypos,rf"b = {bias:.2f}, $\sigma$ = {scatter:.2f}",fontsize = fsize)

    
     # Create a colorbar
    cbar = plt.colorbar(im, ax=ax[ax_id], orientation='horizontal', pad=0.05)
    cbar.ax.set_position([
        0.295,   # Left position
        0.62,  # Top position
        ax[ax_id].get_position().width * 0.1,  # Width (40% of plot width)
        0.02  # Height (thin bar)
    ])
    
    ax_id = 1
    ax[ax_id].set_title(r"COSMOS2020",fontsize = title_size )
    ax[ax_id].scatter(clean_cat_cos_match[xmstar],cos2020_mstar,color = "purple",s=10,marker="s")

    bias, scatter = measure_bias_scatter(clean_cat_cos_match[xmstar].data,cos2020_mstar) 

    ax[ax_id].text( xpos,ypos,rf"b = {bias:.2f}, $\sigma$ = {scatter:.2f}",fontsize = fsize)

    ax_id = 2
    ax[ax_id].set_title(r"GSWLC",fontsize = title_size )
    h, xedges, yedges, im=  ax[ax_id].hist2d(clean_cat_gswlc_match[xmstar],gswlc_match["LOGMSTAR"],range= ( (6,9.5),(6,9.5)),bins= 50,norm=LogNorm(vmin=1, vmax=50) ,cmap=cmap)

    bias, scatter = measure_bias_scatter(clean_cat_gswlc_match[xmstar].data,gswlc_match["LOGMSTAR"].data)
    ax[ax_id].text( xpos,ypos,rf"b = {bias:.2f}, $\sigma$ = {scatter:.2f}",fontsize = fsize)
    
     # Create a colorbar
    cbar = plt.colorbar(im, ax=ax[ax_id], orientation='horizontal', pad=0.05)
    cbar.ax.set_position([
        0.795,   # Left position
        0.62,  # Top position
        ax[ax_id].get_position().width * 0.1,  # Width (40% of plot width)
        0.02  # Height (thin bar)
    ])    
    
    #######
    ax_id = 3
    ax[ax_id].set_title(r"Fastspecfit",fontsize = title_size )
    h, xedges, yedges, im =  ax[ax_id].hist2d(clean_cat_fspec_match[xmstar],fspec_mstar_f,range= ( (6,9.5),(6,9.5)),bins= 50,norm=LogNorm(vmin=1, vmax=1000) ,cmap=cmap)

    bias, scatter = measure_bias_scatter(clean_cat_fspec_match[xmstar].data,fspec_mstar_f)
    ax[ax_id].text( xpos,ypos,rf"b = {bias:.2f}, $\sigma$ = {scatter:.2f}",fontsize = fsize)

     # Create a colorbar
    cbar = plt.colorbar(im, ax=ax[ax_id], orientation='horizontal', pad=0.05)
    cbar.ax.set_position([
        1.045,   # Left position
        0.62,  # Top position
        ax[ax_id].get_position().width * 0.1,  # Width (40% of plot width)
        0.02  # Height (thin bar)
    ])

    #######
    ax_id = 4
    ax[ax_id].set_title(r"gr-based, de Los Reyes+(2024)",fontsize = title_size )
    h, xedges, yedges, im =  ax[ax_id].hist2d(clean_cat[xmstar], clean_cat["LOGM_M24"] ,range= ( (6,9.5),(6,9.5)),bins= 50,norm=LogNorm(vmin=1, vmax=1000) ,cmap=cmap)

    bias,scatter = measure_bias_scatter(clean_cat[xmstar],clean_cat["LOGM_M24"]) 
    ax[ax_id].text( xpos,ypos,rf"b = {bias:.2f}, $\sigma$ = {scatter:.2f}",fontsize = fsize)
    

     # Create a colorbar
    cbar = plt.colorbar(im, ax=ax[ax_id], orientation='horizontal', pad=0.05)
    cbar.ax.set_position([
        1.045,   # Left position
        0.62,  # Top position
        ax[ax_id].get_position().width * 0.1,  # Width (40% of plot width)
        0.02  # Height (thin bar)
    ])

    for i,axi in enumerate(ax):
        axi.set_xlim([6.5,9.5])
        axi.set_ylim([6.5,9.5])
        axi.plot([6,11],[6,11],color = "k",lw = 1)
        axi.set_xlabel(r"gr-based $\log_{10}(M_{\star})$",size= 16)
        ax[0].set_ylabel(r"$\log_{10}(M_{\star})$",size= 16)
        ax[0].grid(ls = ":",color = "lightgrey",alpha = 0.5)
    
        if i != 0:
            axi.set_yticklabels([])

    plt.savefig("/global/homes/v/virajvm/DESI2_LOWZ/quenched_fracs_nbs/paper_plots/stellar_mass_comp.pdf",bbox_inches="tight",dpi=300)
    plt.close()

    return
    


def get_density_map(nside_val, ras, decs):    
    hpix = hp.ang2pix(nside_val, ras, decs, nest=True,lonlat=True)
    
    #count how many objects corresponding to each pixel cell
    hpix_idx, hpix_counts = np.unique(hpix,return_counts=True)

    # reate a full map initialized with zeros
    density_map = np.zeros(hp.nside2npix(nside_val))

    #Get area of one pixel (in deg sq.)
    pix_area_deg2 = hp.nside2pixarea(nside_val, degrees=True)

    #Fill in the density (number per deg sq.)
    density_map[hpix_idx] = hpix_counts / pix_area_deg2

    return density_map


def plot_carview(catalog, sample,cmap=None):
    ra_min, ra_max = 180-15, 180+15
    dec_min, dec_max = -5,3
    nsides = 256
    max_val = 40

    catalog_bgsb = catalog[ catalog["SAMPLE"] == sample]
    density_map_zoom = get_density_map(nsides, catalog_bgsb["RA"].data, catalog_bgsb["DEC"].data)    
    print(np.min(density_map_zoom), np.max(density_map_zoom))
    hp.cartview(
    density_map_zoom,
    lonra=[ra_min, ra_max],   # RA range in degrees, e.g. [100, 160]
    latra=[dec_min, dec_max], # Dec range in degrees, e.g. [-10, 10]
    nest=True,
    cmap=cmap,
    min=0, max=max_val,
    title=None,
    notext=False,
    cbar=False)
    
    plt.savefig(f"/global/homes/v/virajvm/DESI2_LOWZ/quenched_fracs_nbs/paper_plots/zoomin_density_{sample}.png",
            bbox_inches="tight",dpi = 300)
    plt.close()
    return

    
def make_sky_density_plot():
    '''
    In this function, we make a plot showing the on sky density of DESI targets with another plot zooming in on a densely observed region and showing density of each sub-sample!
    '''

    catalog = Table.read("/pscratch/sd/v/virajvm/catalog_dr1_dwarfs/desi_y1_dwarf_clean_catalog_v2.fits")

    print(len(catalog))
    catalog = catalog[catalog["LOGM_SAGA"] < 9]
    print(len(catalog))
    
    density_map_64 = get_density_map(256, catalog["RA"].data, catalog["DEC"].data)
    
    cmap = matplotlib.colormaps['YlGn'].copy()
    cmap.set_bad(color='white')

    projview(
        density_map_64, min=0,max = 40,rot = (120, 0, 0), graticule=True, graticule_labels=True, projection_type="mollweide",
        nest=True,cmap = cmap,
        rot_graticule=False,width = 7,
        custom_xtick_labels=[r"$240^{\circ}$",r"$180^{\circ}$",r"$120^{\circ}$", r"$60^{\circ}$",r"$0^{\circ}$"],
        title = r"DESI DR1 Dwarf Galaxy Density",
        unit=r"Galaxy Density (deg$^{-2}$)",cbar_ticks=[10,20,30,40])
    plt.savefig("/global/homes/v/virajvm/DESI2_LOWZ/quenched_fracs_nbs/paper_plots/dwarf_galaxy_density.png",bbox_inches="tight",dpi = 300)
    plt.close()


    ## now let us focus on specific sub-samples
    plot_carview(catalog, "BGS_BRIGHT",cmap=cmap)
    plot_carview(catalog, "BGS_FAINT",cmap=cmap)
    plot_carview(catalog, "LOWZ",cmap=cmap)
    plot_carview(catalog, "ELG",cmap=cmap)
    
    return
        

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
    # use a good colormap and don't interpolate the pixels
    mpl.rc('image', cmap='viridis', interpolation='none', origin='lower')

    # bgsb_shreds = Table.read("/pscratch/sd/v/virajvm/catalog_dr1_dwarfs/iron_photometry/iron_BGS_BRIGHT_shreds_catalog_w_aper_mags_no_pz.fits")
    # bgsf_shreds = Table.read("/pscratch/sd/v/virajvm/catalog_dr1_dwarfs/iron_photometry/iron_BGS_FAINT_shreds_catalog_w_aper_mags_w_pz.fits")
    # elg_shreds = Table.read("/pscratch/sd/v/virajvm/catalog_dr1_dwarfs/iron_photometry/iron_ELG_shreds_catalog_w_aper_mags_w_pz.fits")

    # make_shred_frac_plot()

    # get_delta_mag_fracflux_plot()


    # make_stellar_mass_comparison_plot()

    make_sky_density_plot()
    

    



    

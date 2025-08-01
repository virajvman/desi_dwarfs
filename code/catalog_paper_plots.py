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
from desi_lowz_funcs import get_remove_flag, _n_or_more_lt, make_subplots, _n_or_more_lt, get_stellar_mass, r_kcorr
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
import matplotlib.colors as mcolors
from astropy.cosmology import Planck18


sample_colors = {"BGS_BRIGHT" : "#882255", "BGS_FAINT": "#CC6677", "LOWZ":"#DDCC77", "ELG": "#88CCEE" }


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
    data_main = Table.read("/pscratch/sd/v/virajvm/catalog_dr1_dwarfs/desi_y1_dwarf_shreds_catalog_v4.fits")
    
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
    # ax_hist.axvline(0.4, color = "k", ls = "--", lw = 1, alpha = 0.5)
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

    ax_plot.axvline(0.5, color = "k", ls = "--", lw = 1,alpha = 0.4)
    
    ax_plot.text(0.35,0.22,r"$\frac{N( > p_{\rm CNN}, \text{Not Fragment}  ) }{N( > p_{\rm CNN}  )}$",fontsize = 15,
                 color = color_good)
    
    ax_plot.text(0.075,0.86,r"$\frac{N( > p_{\rm CNN}, \text{ Fragment}  ) }{N( \text{Fragment}  )}$",fontsize = 15,
                 color = color_frag)
    
    plt.savefig("/global/homes/v/virajvm/DESI2_LOWZ/quenched_fracs_nbs/paper_plots/pcnn_threshold.pdf",bbox_inches="tight")
    
    plt.close()

    return





def make_shred_frac_plot():
    '''
    This function makes the primary clean and shred catalogs we work with!
    '''

    bgsb_list = Table.read("/pscratch/sd/v/virajvm/catalog_dr1_dwarfs/iron_bgs_bright_filter_zsucc_zrr02_allfracflux.fits")
    bgsf_list = Table.read("/pscratch/sd/v/virajvm/catalog_dr1_dwarfs/iron_bgs_faint_filter_zsucc_zrr03_allfracflux.fits")
    elg_list = Table.read("/pscratch/sd/v/virajvm/catalog_dr1_dwarfs/iron_elg_filter_zsucc_zrr05_allfracflux.fits")
    lowz_list = Table.read("/pscratch/sd/v/virajvm/catalog_dr1_dwarfs/iron_lowz_filter_zsucc_zrr03.fits")

    dwarf_mask_bgsb = (bgsb_list["LOGM_SAGA"] < 9.25) 
    dwarf_mask_bgsf = (bgsf_list["LOGM_SAGA"] < 9.25) 
    dwarf_mask_lowz = (lowz_list["LOGM_SAGA"] < 9.25) 
    dwarf_mask_elg = (elg_list["LOGM_SAGA"] < 9.25)   

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
    zgrid = np.arange(0.00,0.125,0.0075)
    #the stellar mass bins of 0.5 dex
    mstar_grid = np.arange(5.5, 9.5,0.5)
    magr_grid = np.arange(17.5, 23,0.5)

        
    zcens = 0.5*(zgrid[1:] + zgrid[:-1])
    ms_cens = 0.5*(mstar_grid[1:] + mstar_grid[:-1])
    mr_cens = 0.5*(magr_grid[1:] + magr_grid[:-1])
    


    
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
            return np.nan

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
    
    # shred_cat = Table.read("/pscratch/sd/v/virajvm/catalog_dr1_dwarfs/desi_y1_dwarf_shreds_catalog_v3.fits")
    
    # print(f"In shredded cat, BGS_BRIGHT={len(shred_cat[shred_cat['SAMPLE'] == 'BGS_BRIGHT'  ])}, BGS_FAINT={len(shred_cat[shred_cat['SAMPLE'] == 'BGS_FAINT'  ])}, ELG={len(shred_cat[shred_cat['SAMPLE'] == 'ELG'  ])}, LOWZ={len(shred_cat[shred_cat['SAMPLE'] == 'LOWZ'  ])}")
    
    # print(f"In plotting catalog, BGS_BRIGHT={len(bgsb_dwarfs[shred_bgsb_mask])}, BGS_FAINT={len(bgsf_dwarfs[shred_bgsf_mask])}, ELG={len(elg_dwarfs[shred_elg_mask])}, LOWZ={len(lowz_dwarfs[shred_lowz_mask])} ")

    # #let us confirm that they are in the same order!
    # print(  np.max(  np.abs( shred_cat[shred_cat['SAMPLE'] == 'BGS_BRIGHT']["TARGETID"].data - bgsb_dwarfs[shred_bgsb_mask]["TARGETID"].data ) ))
    # print(  np.max(  np.abs( shred_cat[shred_cat['SAMPLE'] == 'BGS_FAINT']["TARGETID"].data - bgsf_dwarfs[shred_bgsf_mask]["TARGETID"].data ) )) 
    # print(  np.max(  np.abs( shred_cat[shred_cat['SAMPLE'] == 'ELG']["TARGETID"].data - elg_dwarfs[shred_elg_mask]["TARGETID"].data ) )) 
    # print(  np.max(  np.abs( shred_cat[shred_cat['SAMPLE'] == 'LOWZ']["TARGETID"].data - lowz_dwarfs[shred_lowz_mask]["TARGETID"].data ) )) 
    
    # # #we need the mask to include the pcnn cut, let us just initialize an array of zeros
    # bgsb_dwarfs_pcnn = np.zeros(len(bgsb_dwarfs))
    # bgsf_dwarfs_pcnn = np.zeros(len(bgsf_dwarfs))
    # elg_dwarfs_pcnn = np.zeros(len(elg_dwarfs))
    # lowz_dwarfs_pcnn = np.zeros(len(lowz_dwarfs))

    # #now we fill this with pcnn values for just shredded objects
    # bgsb_dwarfs_pcnn[ shred_bgsb_mask ] = shred_cat[shred_cat['SAMPLE'] == 'BGS_BRIGHT']["PCNN_FRAGMENT"].data
    # bgsf_dwarfs_pcnn[ shred_bgsf_mask ] = shred_cat[shred_cat['SAMPLE'] == 'BGS_FAINT']["PCNN_FRAGMENT"].data
    # elg_dwarfs_pcnn[ shred_elg_mask ] = shred_cat[shred_cat['SAMPLE'] == 'ELG']["PCNN_FRAGMENT"].data
    # lowz_dwarfs_pcnn[ shred_lowz_mask ] = shred_cat[shred_cat['SAMPLE'] == 'LOWZ']["PCNN_FRAGMENT"].data

    # PCNN_CUT = 0.4

    # # #and now we can use this in the mask!
    # print( len(bgsb_dwarfs[shred_bgsb_mask]), len(bgsb_dwarfs[shred_bgsb_mask & (bgsb_dwarfs_pcnn >= PCNN_CUT)] )  )
    # print( len(bgsf_dwarfs[shred_bgsf_mask]), len(bgsf_dwarfs[shred_bgsf_mask&(bgsf_dwarfs_pcnn >= PCNN_CUT)]  ) )
    # print( len(elg_dwarfs[shred_elg_mask]), len(elg_dwarfs[shred_elg_mask&(elg_dwarfs_pcnn >= PCNN_CUT)]   ) )
    # print( len(lowz_dwarfs[shred_lowz_mask]), len(lowz_dwarfs[shred_lowz_mask&(lowz_dwarfs_pcnn >= PCNN_CUT)] ) )
    

    shred_frac_bgsb_ms = []
    shred_frac_bgsf_ms = []
    shred_frac_elg_ms = []
    shred_frac_lowz_ms = []

    for i in trange(len(mstar_grid)-1):
        mlow = mstar_grid[i]
        mhi = mstar_grid[i+1]

        shred_frac_bgsb_ms.append(   get_shred_frac(bgsb_dwarfs, shred_bgsb_mask ,mlow, mhi, rel_col = "LOGM_SAGA" )  ) 
        shred_frac_bgsf_ms.append(   get_shred_frac(bgsf_dwarfs, shred_bgsf_mask, mlow, mhi, rel_col = "LOGM_SAGA")  ) 
        shred_frac_elg_ms.append(   get_shred_frac(elg_dwarfs, shred_elg_mask , mlow, mhi, rel_col = "LOGM_SAGA")  ) 
        shred_frac_lowz_ms.append(   get_shred_frac(lowz_dwarfs, shred_lowz_mask, mlow, mhi, rel_col = "LOGM_SAGA")  ) 


    shred_frac_bgsb_magr = []
    shred_frac_bgsf_magr = []
    shred_frac_elg_magr = []
    shred_frac_lowz_magr = []

    for i in trange(len(magr_grid)-1):
        rlow = magr_grid[i]
        rhi = magr_grid[i+1]

        shred_frac_bgsb_magr.append(   get_shred_frac(bgsb_dwarfs, shred_bgsb_mask ,rlow, rhi, rel_col = "MAG_R" )  ) 
        shred_frac_bgsf_magr.append(   get_shred_frac(bgsf_dwarfs, shred_bgsf_mask, rlow, rhi, rel_col = "MAG_R")  ) 
        shred_frac_elg_magr.append(   get_shred_frac(elg_dwarfs, shred_elg_mask , rlow, rhi, rel_col = "MAG_R"  ) ) 
        shred_frac_lowz_magr.append(   get_shred_frac(lowz_dwarfs, shred_lowz_mask, rlow, rhi, rel_col = "MAG_R")  ) 



    from desi_lowz_funcs import make_subplots

    fig,ax = make_subplots(ncol = 3, nrow = 1,return_fig=True,col_spacing=1.25)

    ax[0].plot(zcens, shred_frac_bgsb_1,label = "BGS Bright",lw = 3,color = sample_colors["BGS_BRIGHT"],ls = "-")
    ax[0].plot(zcens, shred_frac_bgsf_1,label = "BGS Faint",lw = 3,color = sample_colors["BGS_FAINT"],ls = "-")
    ax[0].plot(zcens, shred_frac_lowz_1,label = "LOWZ",lw = 3,color = sample_colors["LOWZ"],ls = "-")
    ax[0].plot(zcens, shred_frac_elg_1,label = "ELG",lw = 3,color = sample_colors["ELG"],ls = "-")

    ls = "-"
    lw = 3
    
    ax[1].plot(ms_cens, shred_frac_bgsb_ms,ls= ls,lw = lw,color = sample_colors["BGS_BRIGHT"])
    ax[1].plot(ms_cens, shred_frac_bgsf_ms,ls= ls,lw = lw,color = sample_colors["BGS_FAINT"])
    ax[1].plot(ms_cens, shred_frac_lowz_ms,ls= ls,lw = lw,color = sample_colors["LOWZ"])
    ax[1].plot(ms_cens, shred_frac_elg_ms,ls= ls,lw = lw,color = sample_colors["ELG"])

    ax[2].plot(mr_cens, shred_frac_bgsb_magr,ls= ls,lw = lw,label = "BGS Bright",color = sample_colors["BGS_BRIGHT"])
    ax[2].plot(mr_cens, shred_frac_bgsf_magr,ls= ls,lw = lw,label = "BGS Faint",color = sample_colors["BGS_FAINT"])
    ax[2].plot(mr_cens, shred_frac_lowz_magr,ls= ls,lw = lw,label = "LOWZ",color = sample_colors["LOWZ"])
    ax[2].plot(mr_cens, shred_frac_elg_magr,ls= ls,lw = lw,label = "ELG",color = sample_colors["ELG"])


    ax[2].legend(fontsize = 11,ncol = 2)
    
    ax[0].set_xlim([0,0.1])
    ax[1].set_xlim([6,9])
    ax[2].set_xlim([18,22.25])
    
    for axi in ax:
        axi.set_ylim([1e-2,1])
        axi.set_yscale("log")
        
    ax[0].set_xlabel(r"Redshift",fontsize = 15)
    ax[1].set_xlabel(r"LogM$^{\rm DR9}_{\star}$",fontsize = 15)
    ax[2].set_xlabel(r"r-band magnitude",fontsize = 15)
    
    
    ax[0].set_ylabel(r"Likely Fragment Fraction",fontsize = 14)
    ax[1].set_ylabel(r"Likely Fragment Fraction",fontsize = 14)
    ax[2].set_ylabel(r"Likely Fragment Fraction",fontsize = 14)
    
    
    # plt.grid(ls=":",color = "lightgrey",alpha = 0.5)
    plt.savefig("/global/homes/v/virajvm/DESI2_LOWZ/quenched_fracs_nbs/paper_plots/frac_shreds.pdf", bbox_inches="tight")
    plt.close()
        
    return 

    


def get_offsets(file_path = "/pscratch/sd/v/virajvm/catalog_dr1_dwarfs/iron_photometry/iron_BGS_BRIGHT_shreds_catalog_w_aper_mags.fits"):
    data = Table.read(file_path)
    
    all_fracfluxs = np.concatenate( (data["FRACFLUX_G"],data["FRACFLUX_R"],data["FRACFLUX_Z"]  ) ) 
    all_offsets = np.concatenate( (data["MAG_G_APERTURE_COG"].data - data["MAG_G"].data ,data["MAG_R_APERTURE_COG"].data - data["MAG_R"].data ,data["MAG_Z_APERTURE_COG"].data - data["MAG_Z"].data   ) )

    aper_mags = np.concatenate( (data["MAG_G_APERTURE_COG"].data,data["MAG_R_APERTURE_COG"].data,data["MAG_Z_APERTURE_COG"].data) )
    
    all_fracfluxs = all_fracfluxs[~np.isnan(all_offsets)]
    all_offsets = all_offsets[~np.isnan(all_offsets)]
    aper_mags = all_offsets[~np.isnan(all_offsets)]
    
    return all_fracfluxs, np.array(all_offsets), aper_mags


    
def get_delta_mag_fracflux_plot(resample_bins=False):
    '''
    Plot of Delta_mag vs. fracflux for both clean and shredded sources!
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

    #to avoid any boundary effects, we want to resample some of these such that in each fracflux bin, we have an equal number of sources?
    
    bins = 50
    
    if resample_bins == True:
        # Parameters
        samples_per_bin = 2000         # How many samples to draw from each ff bin (can be min bin count to avoid oversampling)
        
        # Step 1: Bin ff_all
        ff_bins = np.logspace(-2, np.log10(5), bins)
        ff_indices = np.digitize(ff_all, ff_bins) - 1  # bin indices from 0 to n_bins-1
        
        # Step 2: Resample to flatten ff distribution
        resampled_ff = []
        resampled_dm = []
        
        for i in range(bins):
            in_bin = np.where(ff_indices == i)[0]
            n_in_bin = len(in_bin)
            if n_in_bin == 0:
                continue
            n_sample = min(samples_per_bin, n_in_bin)
            chosen = np.random.choice(in_bin, size=n_sample, replace=False)
            resampled_ff.append(ff_all[chosen])
            resampled_dm.append(dm_all[chosen])

        resampled_ff = np.concatenate(resampled_ff)
        resampled_dm = np.concatenate(resampled_dm)

                
    ax = make_subplots(ncol =1, nrow = 1)
    
    ax[0].hlines(y = 0,xmin=1e-2, xmax = 5,ls = "-",color = "k",lw =1 )
    
    h, xedges, yedges, im = ax[0].hist2d(ff_all, dm_all, range = ( (1e-2, 5,), (-5, 2) ),
                 bins = [ np.logspace(-2, np.log10(5), bins), np.linspace(-5,2,bins) ],norm=LogNorm(vmin=10,vmax = 1000),
                cmap = "BuPu")
    
    ax[0].vlines(x = 0.2, ymin = -5, ymax = 2, color= "grey", ls = "--",lw = 1)
    ax[0].set_xlabel(r"FRACFLUX",fontsize = 13)
    ax[0].set_ylabel(r"$\Delta m$ = mag$_{\rm aper}$ - mag$_{\rm DR9}$",fontsize = 15)
    
    ax[0].set_xlim([1e-2, 5])
    ax[0].set_ylim([-5,2])
    ax[0].set_xscale("log")

    #insert a colorbar
    colbar_x, colbar_y = 0.165, 0.65
    
    cbar = plt.colorbar(im, ax=ax[0], orientation='horizontal', pad=0.05)
    cbar.ax.set_position([
    colbar_x,   # Left position
    colbar_y,  # Top position
    ax[0].get_position().width * 0.065,  # Width (40% of plot width)
    0.015  # Height (thin bar)
])
    
    plt.savefig("/global/homes/v/virajvm/DESI2_LOWZ/quenched_fracs_nbs/paper_plots/fracflux_delta_mag.pdf",bbox_inches="tight", dpi = 300)
    plt.close()


    ##also make a 1d histogram of just the clean sources
    ax = make_subplots(ncol =1, nrow = 1)
    ax[0].hist(dm_all_c, range = (-1.5,1.5), bins = 50,density=True)
    ax[0].set_xlim([-1.5,1.5])
    plt.savefig("/global/homes/v/virajvm/DESI2_LOWZ/quenched_fracs_nbs/paper_plots/delta_mag_clean_1d.pdf",bbox_inches="tight", dpi = 300)
    plt.show()

    np.save("/global/homes/v/virajvm/DESI2_LOWZ/quenched_fracs_nbs/delta_mag_cleans.npy", dm_all_c)
    

    if resample_bins:
        #remake the same plot but now with the resampled bins!!
                            
        ax = make_subplots(ncol =1, nrow = 1)
        
        ax[0].hlines(y = 0,xmin=1e-2, xmax = 5,ls = "-",color = "k",lw =1 )

        h, xedges, yedges, im = ax[0].hist2d(resampled_ff, resampled_dm, range = ( (1e-2, 5,), (-5, 2) ),
                     bins = [ np.logspace(-2, np.log10(5), bins), np.linspace(-5,2,bins) ],norm=LogNorm(vmin=10,vmax = 1000),
                    cmap = "BuPu")
        
        ax[0].vlines(x = 0.2, ymin = -5, ymax = 2, color= "grey", ls = "--",lw = 1)
        ax[0].set_xlabel(r"FRACFLUX",fontsize = 13)
        ax[0].set_ylabel(r"$\Delta m$ = mag$_{\rm aper}$ - mag$_{\rm DR9}$",fontsize = 15)
        
        ax[0].set_xlim([1e-2, 5])
        ax[0].set_ylim([-5,2])
        ax[0].set_xscale("log")

        #insert a colorbar
        cbar = plt.colorbar(im, ax=ax[0], orientation='horizontal', pad=0.05)
        cbar.ax.set_position([
        colbar_x,   # Left position
        colbar_y,  # Top position
        ax[0].get_position().width * 0.065,  # Width (40% of plot width)
        0.015  # Height (thin bar)
    ])
        plt.savefig("/global/homes/v/virajvm/DESI2_LOWZ/quenched_fracs_nbs/paper_plots/resampled_fracflux_delta_mag.pdf",bbox_inches="tight", dpi = 300)
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

    # catalog = Table.read("/pscratch/sd/v/virajvm/catalog_dr1_dwarfs/desi_y1_dwarf_clean_catalog_v2.fits")
    catalog = Table.read("/pscratch/sd/v/virajvm/catalog_dr1_dwarfs/desi_y1_dwarf_combine_catalog.fits")

    ##let us compute the target sample specific target densities

    area = 100*10

    area_mask = (catalog["RA"] < 230) & (catalog["RA"] > 130) & (catalog["DEC"] < 5) & (catalog["DEC"] > -5)
    
    bgsb_tgts = len( catalog[(catalog["SAMPLE"] == "BGS_BRIGHT") & area_mask ] )
    bgsf_tgts = len( catalog[(catalog["SAMPLE"] == "BGS_FAINT") & area_mask] )
    lowz_tgts = len( catalog[(catalog["SAMPLE"] == "LOWZ") & area_mask] )
    elg_tgts = len( catalog[(catalog["SAMPLE"] == "ELG") & area_mask] )

    print(f"BGS Bright density = {bgsb_tgts/area:.2f}")
    print(f"BGS Faint density = {bgsf_tgts/area:.2f}")
    print(f"LOWZ density = {lowz_tgts/area:.2f}")
    print(f"ELG density = {elg_tgts/area:.2f}")
    
    print(len(catalog))
    
    density_map_64 = get_density_map(64, catalog["RA"].data, catalog["DEC"].data)
    
    cmap = matplotlib.colormaps['Greys'].copy()
    cmap.set_bad(color='white')

    fig = plt.figure(figsize=(8, 4))
    
    ax = projview(
        density_map_64, min=0,max = 60,rot = (120, 0, 0), graticule=True, graticule_labels=True, projection_type="mollweide",
        nest=True,cmap = cmap,
        rot_graticule=False,width = 7,
        custom_xtick_labels=[r"$240^{\circ}$",r"$180^{\circ}$",r"$120^{\circ}$", r"$60^{\circ}$",r"$0^{\circ}$"],
        title = r"DESI Extragalactic Dwarf Galaxy Density",
        unit=r"Galaxy Density (deg$^{-2}$)",cbar_ticks=[0,25,50])


    ##adding the rectangle 

    # Define rectangle corners in RA/Dec
    ra1, ra2 = 130 + 120, 230 + 120  # degrees
    dec1, dec2 = -5, 5  # degrees
    
    # Convert RA from [0,360] -> [-180,180] and then to radians
    def ra_to_mollweide_radians(ra_deg):
        ra_wrapped = ((ra_deg + 180) % 360) - 180  # wrap into [-180, 180]
        return np.deg2rad(ra_wrapped)
    
    # Convert Dec to radians directly
    def dec_to_radians(dec_deg):
        return np.deg2rad(dec_deg)
    
    # Get rectangle edges
    ra_edges = [ra1, ra2, ra2, ra1, ra1]
    dec_edges = [dec1, dec1, dec2, dec2, dec1]
    
    x = ra_to_mollweide_radians(np.array(ra_edges))
    y = dec_to_radians(np.array(dec_edges))
    
    # Plot on the current axes (projview uses gca)
    ax = plt.gca()
    ax.plot(x, y, color='r', lw=1.5,ls = "--")

    yref = -20
    shift = 12
    
    x = ra_to_mollweide_radians(np.array([130+120]))
    y = dec_to_radians(np.array([yref]))
    ax.text(x,y, fr"BGS Bright: {bgsb_tgts/area:.0f} deg$^{{-2}}$",fontsize = 10,color = "firebrick")

    x = ra_to_mollweide_radians(np.array([130+120]))
    y = dec_to_radians(np.array([yref - shift]))
    ax.text(x,y, fr"BGS Faint: {bgsf_tgts/area:.0f} deg$^{{-2}}$",fontsize = 10,color = "firebrick")

    x = ra_to_mollweide_radians(np.array([130+120]))
    y = dec_to_radians(np.array([yref - shift*2]))
    ax.text(x,y, fr"LOWZ: {lowz_tgts/area:.0f} deg$^{{-2}}$",fontsize = 10,color = "firebrick")

    x = ra_to_mollweide_radians(np.array([130+120]))
    y = dec_to_radians(np.array([yref - shift*3]))
    ax.text(x,y, fr"ELG: {elg_tgts/area:.0f} deg$^{{-2}}$",fontsize = 10,color = "firebrick")

    ####################
    
    plt.savefig("/global/homes/v/virajvm/DESI2_LOWZ/quenched_fracs_nbs/paper_plots/dwarf_galaxy_density.pdf",bbox_inches="tight")
    plt.close()

    


    ## now let us focus on specific sub-samples
    # plot_carview(catalog, "BGS_BRIGHT",cmap=cmap)
    # plot_carview(catalog, "BGS_FAINT",cmap=cmap)
    # plot_carview(catalog, "LOWZ",cmap=cmap)
    # plot_carview(catalog, "ELG",cmap=cmap)
    
    return


def make_bar_pie(ax, tot_cat, col, bins =  np.arange(6, 9.75,0.125)):
    bin_centers = 0.5 * (bins[1:] + bins[:-1])

    # Unique sample names
    samples = ["BGS_BRIGHT","BGS_FAINT","LOWZ","ELG"]
    samp_colors = ["#882255", "#CC6677", "#DDCC77", "#88CCEE" ]
    
    # Initialize a 2D array: shape (n_samples, n_bins)
    fraction_per_bin = np.zeros((len(samples), len(bins) - 1))
    
    # Loop through samples and compute histogram per sample
    for i, s in enumerate(samples):
        sample_mask = (tot_cat["SAMPLE"] == s)
        hist_sample, _ = np.histogram(tot_cat[col][sample_mask], bins=bins)
    
        hist_all, _ = np.histogram(tot_cat[col], bins=bins)
        # Avoid division by zero
        with np.errstate(divide='ignore', invalid='ignore'):
            fraction_per_bin[i] = np.where(hist_all > 0, hist_sample / hist_all, 0)
    

    cumulative = np.zeros(len(bin_centers))
    
    for i, (s, color) in enumerate(zip(samples, samp_colors)):
        upper = cumulative + fraction_per_bin[i]
        ax.fill_between(bin_centers, cumulative, upper, label=str(s), color=color, alpha=1)
        cumulative = upper    

    return


def make_cmap(base_color):
    # Create a colormap from blue to white
    cmap = mcolors.LinearSegmentedColormap.from_list("custom_cmap", ["white", base_color])
    return cmap


def mstar_to_mag(zred,mstar,ave_col = 0.2):
    '''
    Converting mstar to magnitude assuming some average color. This is using the saga color
    '''
    #convert the zred to the luminosity distance 
    d = Planck18.luminosity_distance(zred)
    d_in_pc = d.value * 1e6
    
    kr = r_kcorr(ave_col,zred)
    
    M_r = -1*(mstar - 1.254 - 1.098*ave_col) / 0.4
    
    rmag = M_r - 5 + 5*np.log10(d_in_pc) + kr
    
    return rmag
    
def make_mstar_zred_contour(ave_col = 0.2):
    zred_i = np.linspace(0.001, 0.3, 300)
    mstar_i = np.linspace(6, 9.25, 300)

    X, Y = np.meshgrid(zred_i, mstar_i)
    Z = mstar_to_mag(X, Y, ave_col)

    return X,Y,Z


    
def make_summary_stats():
    '''
    Plots of fraction as a fraction of stellar mass, redshift and magnitude and other summary plots. 

    These are the parts that are the bar pie share plots
    
    '''

    sample_colors = {"BGS_BRIGHT" : "#882255", "BGS_FAINT": "#CC6677", "LOWZ":"#DDCC77", "ELG": "#88CCEE" }
    all_samp_colors = [ sample_colors["BGS_BRIGHT"],sample_colors["BGS_FAINT"],sample_colors["LOWZ"],sample_colors["ELG"],  ]
    samples = ["BGS_BRIGHT","BGS_FAINT","LOWZ","ELG"]
    
    tot_cat = Table.read("/pscratch/sd/v/virajvm/catalog_dr1_dwarfs/desi_y1_dwarf_combine_catalog.fits")
    
    fig,ax = make_subplots(nrow=1,ncol=3,return_fig=True,col_spacing = 0.4,row_spacing = 0.4)

    # Compute horizontal positions for each word
    # We'll spread them evenly around center
    # n_words = len(samples)
    # spacing = 0.07  # tweak this to change spacing
    # center = 0.5
    # start = center - spacing * (n_words - 1) / 2
    # positions = [start + i * spacing for i in range(n_words)]
    
    # # Add each word as a separate fig.text element
    # for word, color, xpos in zip(samples, all_samp_colors, positions):
    #     fig.text(xpos, 0.5, word, color=color,
    #              fontsize=15, ha='center', va='bottom')

    make_bar_pie(ax[0], tot_cat, "LOGM_SAGA", bins =  np.arange(6-0.125/2, 9.5,0.125))

    ax[0].set_xlim(6, 9.25)
    ax[0].set_ylim(0, 1)
    ax[0].set_xlabel(r"$\log_{10}M_{\star}$",fontsize=15)
    # ax.set_title("Fractional Sample Composition vs Stellar Mass")
    
    make_bar_pie(ax[1], tot_cat, "Z", bins =  np.arange(0-0.025/2, 0.4,0.025) )
    ax[1].set_xlim(0.0, 0.3)
    ax[1].set_ylim(0, 1)
    ax[1].set_xlabel("Redshift",fontsize=15)


    make_bar_pie(ax[2], tot_cat, "MAG_R", bins =  np.arange(17,23.5,0.1) )
    ax[2].set_xlim(18, 23)
    ax[2].set_ylim(0, 1)
    ax[2].set_xlabel(r"$r$-band magnitude",fontsize=15)

    ax[0].set_ylabel("Fracion",fontsize = 15)
    for i in range(1,3):
        ax[i].set_yticklabels([])
    
    fig.savefig("/global/homes/v/virajvm/DESI2_LOWZ/quenched_fracs_nbs/paper_plots/dwarf_summaries.pdf",bbox_inches="tight")
    plt.close()



    ## 1d stellar mass distribution and redshift distribution
    
    fig,ax = make_subplots(ncol=1,nrow=2,return_fig=True,col_spacing = 0.4, row_spacing = 0.8)
    
    #first row will be stellar mass and the second row will be redshift
    
    text_titles = ["BGS Bright", "BGS Faint", "LOWZ", "ELG"]
    
    mask_i = (tot_cat["SAMPLE"] == "BGS_BRIGHT") # | (tot_cat["SAMPLE"] == "BGS_FAINT" )
    
    ax[1].hist( tot_cat["LOGM_SAGA"][mask_i], bins = np.arange(6,10,0.5), 
               color = sample_colors[samples[0]], histtype = "step", lw = 2, zorder = 1,alpha = 1 )
    ax[1].hist( tot_cat["LOGM_SAGA"][mask_i], bins = np.arange(6,10,0.5), 
               color = sample_colors[samples[0]], histtype = "stepfilled", lw = 2, zorder = 1,alpha = 0.35,
              label = "BGS Bright")
    
    
    mask_i = (tot_cat["SAMPLE"] == "BGS_FAINT" )
    ax[1].hist( tot_cat["LOGM_SAGA"][mask_i], bins = np.arange(6,10,0.5), 
               color = sample_colors[samples[1]], histtype = "step", lw = 2, zorder = 1,label = "BGS Faint" )
    
    mask_i = (tot_cat["SAMPLE"] == "LOWZ" )
    ax[1].hist( tot_cat["LOGM_SAGA"][mask_i], bins = np.arange(6,10,0.5), 
               color = sample_colors[samples[2]], histtype = "step", lw = 2, zorder = 1,label = "LOWZ" )
    
    mask_i = (tot_cat["SAMPLE"] == "ELG" )
    ax[1].hist( tot_cat["LOGM_SAGA"][mask_i], bins = np.arange(6,10,0.5), 
               color = sample_colors[samples[3]], histtype = "step", lw = 2, zorder = 1,label = "ELG" )
    
    
    ax[1].set_yscale("log")
    ax[1].set_xlim([6,9.25])
    ax[1].set_ylim([1,5e5])
    ax[1].legend(frameon=False,fontsize = 11.5,loc="upper left")
    ax[1].set_xlabel(r"$\log_{10}M_{\star}$",fontsize = 15)
    ax[1].set_ylabel(r"Number",fontsize = 15)
    
    
    binw = 0.0075
    
    
    text_titles = ["BGS Bright", "BGS Faint", "LOWZ", "ELG"]
    
    mask_i = (tot_cat["SAMPLE"] == "BGS_BRIGHT")
    ax[0].hist(tot_cat["Z"][mask_i], bins = np.arange(0,0.45,binw),
               color = sample_colors[samples[0]], histtype = "step", lw = 2, zorder = 0,alpha = 1,density=True )
    
    ax[0].hist( tot_cat["Z"][mask_i], bins = np.arange(0,0.45,binw),
               color = sample_colors[samples[0]], histtype = "stepfilled", lw = 2, zorder = 0,alpha = 0.35,
              label = "BGS Bright",density=True)
    
    
    mask_i = (tot_cat["SAMPLE"] == "BGS_FAINT" )
    ax[0].hist( tot_cat["Z"][mask_i], bins = np.arange(0,0.45,binw),
               color = sample_colors[samples[1]], histtype = "step", lw = 2, zorder = 1,label = "BGS Faint",density=True )
    
    mask_i = (tot_cat["SAMPLE"] == "LOWZ" )
    ax[0].hist( tot_cat["Z"][mask_i], bins = np.arange(0,0.45,binw),
               color = sample_colors[samples[2]], histtype = "step", lw = 2, zorder = 2,density=True )
    ax[0].hist( tot_cat["Z"][mask_i], bins = np.arange(0,0.45,binw),
               color = sample_colors[samples[2]], histtype = "stepfilled", lw = 2, zorder = 2,label = "LOWZ",density=True,alpha=0.55 )
    
    
    
    mask_i = (tot_cat["SAMPLE"] == "ELG" )
    ax[0].hist( tot_cat["Z"][mask_i], bins = np.arange(0,0.45,binw),
               color = sample_colors[samples[3]], histtype = "step", lw = 2, zorder = 3,label = "ELG",density=True )
    
    
    ax[0].set_ylim([0,17])
    ax[0].set_xlim([0,0.4])
    ax[0].set_xlabel(r"Redshift",fontsize = 15)
    ax[0].set_ylabel(r"$n(z)$",fontsize = 15)
    ax[0].legend(frameon=False,fontsize = 11.5,loc = "upper right")
    plt.savefig("/global/homes/v/virajvm/DESI2_LOWZ/quenched_fracs_nbs/paper_plots/zred_mstar_1d.pdf",bbox_inches="tight")
    plt.close()
        
    
    ### 2d redshift vs. stellar mass distribution 

    zgrid = np.arange(0.001,0.31,0.001)
    gr_col = 0.2
    mstar_195= get_stellar_mass(gr_col,19.5,zgrid)
    mstar_20= get_stellar_mass(gr_col,20,zgrid)
    mstar_21= get_stellar_mass(gr_col,21,zgrid)
    mstar_23= get_stellar_mass(gr_col,23,zgrid)

    mstar_grids = [ mstar_195, mstar_20, mstar_21, mstar_23 ]
    labels = [r"$r \sim 19.5$", r"$r \sim 20.175$", r"$r \sim 21.15$", r"$r \sim 23$" ]
    rmag_lims = [19.5, 20.175, 21.15, 23]
    
    # X,Y,Z = make_mstar_zred_contour(ave_col = 0.2)

    def fmt(x):
        return rf"r = %d"%x
    
    fig,ax = make_subplots(ncol=4,nrow=1,return_fig=True,col_spacing = 0.4, row_spacing = 0.8)

    for i in range(1,4):
        ax[i].set_yticklabels([])

    for i in range(4):
        mask_i = (tot_cat["SAMPLE"] == samples[i] )
        
        cmap_i = make_cmap(sample_colors[samples[i]] )
        
        hist = ax[i].hist2d( tot_cat["Z"][mask_i], tot_cat["LOGM_SAGA"][mask_i], range= ( (0,0.3), (6,9.25) ) , bins = 50, norm=LogNorm(vmin=1,vmax=400),cmap = cmap_i)
        ax[i].set_xlabel("Redshift",fontsize = 15)

        ##assuming an average color, of lets say g-r = 0.3, can I obtain a line for r ~ 19.5, 21, and 23
        ax[i].plot(zgrid, mstar_grids[i], color = "k",lw = 1.5)
        zind = 35
        ax[i].text(zgrid[zind]+0.0075,mstar_grids[i][zind] - 0.05,labels[i],color = "k",fontsize = 12,rotation =55)

        # CS = ax[i].contour(X, Y, Z,levels = [rmag_lims[i]],colors = ["k"])
        # ax[i].clabel(CS, CS.levels, inline=True, fmt=fmt, fontsize=12)
        
        if i == 0:
            mstar_17= get_stellar_mass(gr_col,17.7,zgrid)
            ax[i].text(0.019,8.75,"SDSS",color = "lightgrey",fontsize = 10,rotation =60)
            ax[i].plot(zgrid, mstar_17, color = "lightgrey",lw = 1,ls = "--")
            
    
        ax[i].set_title(f"{text_titles[i]}", fontsize = 18)
            
        ax[0].set_ylabel(r"$\log_{10}M_{\star}$",fontsize = 15)

        ##add a colorbar:
        # Force figure to calculate axis positions
        fig.canvas.draw()
    
        # Get subplot position in figure coordinates
        p = ax[i].get_position().get_points().flatten()
    
        # Set colorbar size and position below the subplot
        cbar_height = 0.02
        cbar_padding = 0.06
    
        cax = fig.add_axes([
            p[0]+0.525*(p[2]-p[0]),                  # x0
            p[1] + cbar_padding,   # y0: a bit below the axis
            0.4*(p[2] - p[0]),           # width
            cbar_height            # height
        ])
    
        cbarticks = [1,1e1, 1e2, 500]
        cbar = plt.colorbar(hist[3], cax=cax, orientation='horizontal',
                            ticklocation='bottom', extend='both', ticks=cbarticks)
        cbar.ax.tick_params(labelsize=11)

    
    #include a vertical bar indicating the typical error !
    # 
    plt.savefig("/global/homes/v/virajvm/DESI2_LOWZ/quenched_fracs_nbs/paper_plots/zred_mstar_2d.pdf",bbox_inches="tight")
    plt.close()

    return
    

def halpha_flux_to_lumi(zreds, ha_flux):
    '''
    Function that converts redshift and observed Halpha flux into Halpha luminosity!
    '''
    from astropy.cosmology import Planck18
    lumi_dist_in_cm = Planck18.luminosity_distance(zreds).to(u.cm).value
    ha_lumi = ha_flux * 1e-17 * 4 * np.pi * (lumi_dist_in_cm)**2
    ##this is in units of ergs/s
    return ha_lumi

def halpha_lumi_plot():

    #load in the full catalog
    temp = Table.read("/pscratch/sd/v/virajvm/catalog_dr1_dwarfs/desi_y1_dwarf_combine_catalog.fits")
    #selecting for 5 sigma detection of lines!
    
    temp = temp[ temp["HALPHA_FLUX"].data * np.sqrt(temp["HALPHA_FLUX_IVAR"].data) > 5 ]
    temp["HALPHA_LUMI"] = halpha_flux_to_lumi(temp["Z"].data, temp["HALPHA_FLUX"].data )

    
    fig,ax = make_subplots(ncol = 2, nrow = 1,return_fig=True,col_spacing = 0.05)

    elg_mask = (temp["SAMPLE"] == "ELG")
    bgsb_mask = (temp["SAMPLE"] == "BGS_FAINT") | (temp["SAMPLE"] == "BGS_BRIGHT") | (temp["SAMPLE"] == "LOWZ") 
      
    #we will just show one stellar mass bin
    bins = np.arange(36,42.5,0.5)
    lw = 3
    alpha = 0.75

    lows = [6,7.5]
    his = [7.5,9]

    for i in range(2):

        low_mstar = lows[i]
        hi_mstar = his[i]

        mstar_mask = (temp["LOGM_SAGA"] > low_mstar) & (temp["LOGM_SAGA"] < hi_mstar)
            
        ax[i].set_title(r"$10^{{{}}} < M_{{\ast}} < 10^{{{}}}$".format(low_mstar, hi_mstar), fontsize=15)
        
        ax[i].hist( np.log10(temp["HALPHA_LUMI"][mstar_mask & bgsb_mask]), range = (36,42), bins = bins,density=True,
                  histtype = "stepfilled", color = sample_colors["BGS_BRIGHT"],lw = lw,alpha = alpha)
        
        ax[i].hist( np.log10(temp["HALPHA_LUMI"][mstar_mask & elg_mask]), range = (36,42), bins = bins,density=True,
                  histtype = "stepfilled", color = sample_colors["ELG"],lw = lw,alpha=alpha)
    
        ax[i].set_xlabel(r"$L_{H_{\alpha}}$ (ergs/s)",fontsize = 15)


    ax[0].text()

    
    ax[0].set_ylabel(r"Density",fontsize = 15)
    ax[1].set_yticklabels([])

    for axi in ax:
        axi.set_xlim([36,42])

    plt.savefig("/global/homes/v/virajvm/DESI2_LOWZ/quenched_fracs_nbs/paper_plots/halpha_mstar.png",bbox_inches="tight")
    plt.close()




from sklearn.mixture import GaussianMixture
from scipy.stats import gaussian_kde

def get_gmm(ax,gr_cols, rz_cols,
           contour_cmap = "viridis",hist_cmap = "Reds",
            hist_alpha = 1, cont_alpha = 1,contour_col = "r"):
    
    data = np.vstack( [gr_cols, rz_cols] )
    
    # Create a grid for evaluation
    xmin, xmax = -0.5, 2
    ymin, ymax = -0.5, 1.5
    X, Y = np.meshgrid(np.linspace(xmin, xmax, 200), np.linspace(ymin, ymax, 200))
    positions = np.vstack([X.ravel(), Y.ravel()])
    
    # Fit GMM with 3 components (change as needed)
    gmm = GaussianMixture(n_components=10, covariance_type='full')
    gmm.fit(data.T)
    
    # Evaluate the density on the grid
    Z_gmm = np.exp(gmm.score_samples(positions.T)).reshape(X.shape)
    
    ax.hist2d(gr_cols, rz_cols,bins=100,norm=LogNorm(),cmap = hist_cmap,alpha=hist_alpha )

    ## now compute the 2d quantiels

    Z_gmm = Z_gmm/np.sum(Z_gmm)        
    # plot contours if contour levels are specified in clevs 
    lvls = []
    
    clevs = [0.383,0.68,0.954,0.987]
    for cld in clevs:  
        sig = opt.brentq( conf_interval, 0., 1., args=(Z_gmm,cld) )   
        lvls.append(sig)

    if contour_cmap is None:
        ax.contour(X, Y, Z_gmm, levels=sorted(lvls), colors = contour_col,alpha = cont_alpha)
    else:
        ax.contour(X, Y, Z_gmm, levels=sorted(lvls), cmap=contour_cmap,alpha = cont_alpha)

    ax.set_xlim([-0.5,2])
    ax.set_ylim([-0.5,1.5])

    return gmm

def make_color_contour_plot():
    '''
    Plot that makes the g-r vs. r-z color contours for this distribution
    '''
    
    iron_data = Table.read("/pscratch/sd/v/virajvm/catalog/Iron_bgs_bright_all_phot_final_filter.fits")

    #redshift grid
    zgrid = np.arange(0.001, 0.525,0.025)
    
    
    fig,ax = make_subplots(ncol = 4, nrow = 1,return_fig=True,col_spacing=0.6)
    
    fsize = 18
    for j,i in enumerate([0,2,4,6]):
        
        iron_i = iron_data[ (iron_data["Z"]  > zgrid[i]) & (iron_data["Z"] < zgrid[i+1]) & (iron_data["DELTACHI2"] > 40)]
        gr_cols_i = iron_i["MAG_G"] - iron_i["MAG_R"] 
        rz_cols_i = iron_i["MAG_R"] - iron_i["MAG_Z"] 
    
        gmm_s = get_gmm(ax[j],gr_cols_i, rz_cols_i,contour_cmap = None,hist_cmap = "Greys",
                hist_alpha = 1, cont_alpha = 0.625,contour_col="darkorange")
    
        ax[j].set_xlabel(r"g-r",fontsize = fsize)
        ax[j].set_xticks([-0.5,0,0.5,1,1.5,2])
    
        #add the text inside
        ax[j].set_title(r"$%.3f < z < %.3f$"%(zgrid[i], zgrid[i+1]),color = "k",size=fsize-2)
        
        if j != 0:
            ax[j].set_yticklabels([])
    
        # plot_2d_dist(gr_cols_i,rz_cols_i, 100, 100, 
        #         cmin=1.e-4, cmax=1.0, smooth=2,clevs=[0.383,0.68,0.954,0.987], ax=ax2[j])
    
    ax[0].set_ylabel("r-z",fontsize = fsize)
    
    plt.savefig("paper_plots/gr_rz_zred_dists.pdf",bbox_inches="tight")
    plt.show(fig)

    return


def fraction_remain_dwarf_after_aper():
    '''
    Function where we make a sample specific plot where we plot galaxies that have a significant shift in LOGM > 0.5 and fraction that do not remain dwarf at all.
    '''
    
    photo_path = "/pscratch/sd/v/virajvm/catalog_dr1_dwarfs/iron_photometry/" 
    # all_samps = ["BGS_BRIGHT","BGS_FAINT","LOWZ","ELG"]

    bgsb_shreds = Table.read(photo_path + "iron_BGS_BRIGHT_shreds_catalog_w_aper_mags.fits")
    bgsf_shreds = Table.read(photo_path + "iron_BGS_FAINT_shreds_catalog_w_aper_mags.fits")
    lowz_shreds = Table.read(photo_path + "iron_LOWZ_shreds_catalog_w_aper_mags.fits")
    elg_shreds = Table.read(photo_path + "iron_ELG_shreds_catalog_w_aper_mags.fits")

    print(len(bgsb_shreds))
    print(len(bgsf_shreds))
    print(len(lowz_shreds))
    print(len(elg_shreds))
    
    mstar_grid = np.arange(6, 9.5,0.5)

    ms_cens = 0.5*(mstar_grid[1:] + mstar_grid[:-1])

    def get_shred_frac(tot_cat, subset_mask, low, hi, rel_col="Z"):
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

    #nol = no longer
    nolo_dwarf_frac_bgsb = []
    nolo_dwarf_frac_bgsf = []
    nolo_dwarf_frac_elg = []
    nolo_dwarf_frac_lowz = []

    bigshf_dwarf_frac_bgsb = []
    bigshf_dwarf_frac_bgsf = []
    bigshf_dwarf_frac_lowz = []
    bigshf_dwarf_frac_elg = []
    
    
    for i in trange(len(mstar_grid)-1):
        mlow = mstar_grid[i]
        mhi = mstar_grid[i+1]


        bigshf_bgsb_mask = (bgsb_shreds["LOGM_SAGA_APERTURE_COG"].data - bgsb_shreds["LOGM_SAGA"].data > 0.5)
        bigshf_bgsf_mask = (bgsf_shreds["LOGM_SAGA_APERTURE_COG"].data - bgsf_shreds["LOGM_SAGA"].data > 0.5) 
        bigshf_lowz_mask = (lowz_shreds["LOGM_SAGA_APERTURE_COG"].data - lowz_shreds["LOGM_SAGA"].data > 0.5) 
        bigshf_elg_mask = (elg_shreds["LOGM_SAGA_APERTURE_COG"].data - elg_shreds["LOGM_SAGA"].data > 0.5) 
        
        nolo_bgsb_mask = (bgsb_shreds["LOGM_SAGA_APERTURE_COG"].data > 9.25) 
        nolo_bgsf_mask = (bgsf_shreds["LOGM_SAGA_APERTURE_COG"].data > 9.25) 
        nolo_lowz_mask = (lowz_shreds["LOGM_SAGA_APERTURE_COG"].data > 9.25) 
        nolo_elg_mask = (elg_shreds["LOGM_SAGA_APERTURE_COG"].data > 9.25) 
        

        nolo_dwarf_frac_bgsb.append(   get_shred_frac(bgsb_shreds, nolo_bgsb_mask.data ,mlow, mhi, rel_col = "LOGM_SAGA" )  ) 
        nolo_dwarf_frac_bgsf.append(   get_shred_frac(bgsf_shreds, nolo_bgsf_mask.data, mlow, mhi, rel_col = "LOGM_SAGA")  ) 
        nolo_dwarf_frac_elg.append(   get_shred_frac(elg_shreds, nolo_elg_mask.data, mlow, mhi, rel_col = "LOGM_SAGA")  ) 
        nolo_dwarf_frac_lowz.append(   get_shred_frac(lowz_shreds, nolo_lowz_mask.data, mlow, mhi, rel_col = "LOGM_SAGA")  ) 

        bigshf_dwarf_frac_bgsb.append(   get_shred_frac(bgsb_shreds, bigshf_bgsb_mask.data ,mlow, mhi, rel_col = "LOGM_SAGA" )  ) 
        bigshf_dwarf_frac_bgsf.append(   get_shred_frac(bgsf_shreds, bigshf_bgsf_mask.data, mlow, mhi, rel_col = "LOGM_SAGA")  ) 
        bigshf_dwarf_frac_elg.append(   get_shred_frac(elg_shreds, bigshf_elg_mask.data, mlow, mhi, rel_col = "LOGM_SAGA")  ) 
        bigshf_dwarf_frac_lowz.append(   get_shred_frac(lowz_shreds, bigshf_lowz_mask.data, mlow, mhi, rel_col = "LOGM_SAGA")  ) 


    ##make teh plot now!!
    ax = make_subplots(ncol=1,nrow=1)

    lw = 4

    ax[0].set_title("Fraction of Likely Fragment Galaxies \n Remaining Dwarfs After Aperture Photometry",fontsize = 12)

    # ls = "--"
    # ax[0].plot(ms_cens, bigshf_dwarf_frac_bgsb,ls= ls,lw = lw,color = sample_colors["BGS_BRIGHT"],alpha = 1)
    # ax[0].plot(ms_cens, bigshf_dwarf_frac_bgsf,ls= ls,lw = lw,color = sample_colors["BGS_FAINT"],alpha = 1)
    # ax[0].plot(ms_cens, bigshf_dwarf_frac_lowz,ls= ls,lw = lw,color = sample_colors["LOWZ"],alpha = 1)
    # ax[0].plot(ms_cens, bigshf_dwarf_frac_elg,ls= ls,lw = lw,color = sample_colors["ELG"],alpha = 1)

    ls = "-"    
    ax[0].plot(ms_cens, nolo_dwarf_frac_bgsb,ls= ls,lw = lw,color = sample_colors["BGS_BRIGHT"],alpha = 1)
    ax[0].plot(ms_cens, nolo_dwarf_frac_bgsf,ls= ls,lw = lw,color = sample_colors["BGS_FAINT"],alpha = 1)
    ax[0].plot(ms_cens, nolo_dwarf_frac_lowz,ls= ls,lw = lw,color = sample_colors["LOWZ"],alpha = 1)
    ax[0].plot(ms_cens, nolo_dwarf_frac_elg,ls= ls,lw = lw,color = sample_colors["ELG"],alpha = 1)

    fs = 13
    ax[0].text(6.1, 0.925, r"BGS Bright",color = sample_colors["BGS_BRIGHT"],fontsize = fs,weight="bold")
    ax[0].text(7.35, 0.925, r"BGS Faint",color = sample_colors["BGS_FAINT"],fontsize = fs,weight="bold")
    ax[0].text(6.1, 0.825, r"LOWZ",color = sample_colors["LOWZ"],fontsize = fs,weight="bold")
    ax[0].text(6.85, 0.825, r"ELG",color = sample_colors["ELG"],fontsize = fs,weight="bold")
    
    ax[0].set_ylim([-0.01,1.01])
    ax[0].set_xlim([6,9])
    ax[0].set_xlabel(r"LogM$^{\rm DR9}_{\star}$",fontsize = 15)
    ax[0].set_ylabel("Fraction",fontsize = 15)

    plt.savefig("/global/homes/v/virajvm/DESI2_LOWZ/quenched_fracs_nbs/paper_plots/remain_dwarf_shred_stats.pdf",bbox_inches="tight")
    plt.close()

    return


def scarlet_aper_comp():
    '''
    Plot for comparing the ~100 scarlet models with aperture magnitudes for nearby objects!!
    '''
    
    data = Table.read("/pscratch/sd/v/virajvm/catalog_dr1_dwarfs/desi_y1_dwarf_shreds_catalog_filter.fits")

    #filter for objects that would be good to do a scarlet model for
    data_scarlet = data[(data["Z"] < 0.01) & (data["SAMPLE"] != "ELG") & (data["LOGM_SAGA_APERTURE_COG"] < 9) & (data["MASKBITS"]==0) & (data["STARFDIST"] > 2) & (data["SGA_D26_NORM_DIST"] > 4) & (data["is_south"] == 1)  ]
    
    print(f"Number of galaxies for scarlet model = {len(data_scarlet)}")
    
    
    all_cog_mags = []
    all_scar_mags = []
    
    for index in trange(len(data_scarlet)):
        if index in [7, 22, 23, 25, 30, 45, 63, 66, 77, 82, 85, 93, 94, 95, 97, 98, 108, 119, 123, 126, 127, 128, 129, 130]:
            pass
        else:
            try:
                save_folder = data_scarlet["FILE_PATH"].data[index]
    
                # If save_folder is a byte string, decode it
                if isinstance(save_folder, bytes):
                    save_folder = save_folder.decode("utf-8")
                    
                scar_mags = np.load( save_folder + "/scarlet_mags.npy")
                all_scar_mags.append(scar_mags)
                
                
                cog_mags = []
                for b in "GRZ":
                    cog_mags.append( data_scarlet[f"MAG_{b}_APERTURE_COG"].data[index] )
                    
                all_cog_mags.append(cog_mags)
    
                if np.max(np.abs( np.array(cog_mags) - np.array(scar_mags) ) ) > 1:
                    print( index )
        
    
            except:
                pass

    all_cog_mags = np.concatenate(all_cog_mags)
    all_scar_mags = np.concatenate(all_scar_mags)

    ax = make_subplots(ncol = 1, nrow = 1)

    ax[0].scatter(all_cog_mags, all_cog_mags - all_scar_mags,color = "grey",alpha = 0.6)
    ax[0].axhline(y = 0,color = "k")
    ax[0].set_ylim([-1.5,1.5])
    ax[0].set_xlim([14,20])
    ax[0].set_xlabel(r"mag$_{\rm aper}$",fontsize = 15)
    ax[0].set_ylabel(r"mag$_{\rm aper}$ - mag$_{\rm scarlet}$",fontsize = 15)
    
    dms = all_cog_mags - all_scar_mags
    #removing the 4 large outliers so that they do not bias std value
    dms_clean = dms[np.abs(dms) < 1]
        
    ax[0].text(14.4,1.2, f"bias = ${np.median(dms):.2f}$",fontsize = 15)
    ax[0].text(14.4,0.95, rf"$\sigma$ = {np.std( dms_clean - np.median(dms_clean ) ):.2f}",fontsize = 15)

    import matplotlib.patches as patches
    x,y = 19.55,1.325
    box_size = 0.25
    rect = patches.Rectangle((x - 0.125, y - 0.125), 2*box_size, box_size,
                             linewidth=1, edgecolor='firebrick', facecolor='none',ls = "--")
    ax[0].add_patch(rect)
    ax[0].text(x-0.4,y-0.05,"2",color = "firebrick",fontsize = 13)
    
    x,y = 15.2,-1.2
    box_size = 0.25
    rect = patches.Rectangle((x , y ), 6*box_size, box_size,
                             linewidth=1, edgecolor='firebrick', facecolor='none',ls = "--")
    ax[0].add_patch(rect)
    
    ax[0].text(x-0.25,y+0.075,"1",color = "firebrick",fontsize = 13)
    
    plt.savefig("/global/homes/v/virajvm/DESI2_LOWZ/quenched_fracs_nbs/paper_plots/scarlet_aper_compare.pdf",bbox_inches="tight")
    plt.close()


def make_mur_mstar_plot(tot_cat):

    from desi_lowz_funcs import plot_2d_dist

    bgsb_mask = (tot_cat["SAMPLE"] == "BGS_BRIGHT")
    bgsf_mask = (tot_cat["SAMPLE"] == "BGS_FAINT")
    lowz_mask = (tot_cat["SAMPLE"] == "LOWZ")


    cmap_bgsb = make_cmap(sample_colors["BGS_BRIGHT"] )


    fig,ax = plt.subplots(1,1,figsize = (4,4))
    plot_2d_dist(tot_cat[bgsb_mask]["LOGM_SAGA"], tot_cat[bgsb_mask]["MU_R"], 25, 25, 
                    cmin=1.e-4, cmax=1.0, smooth=2, clevs=[0,0.68,0.95,0.997],ax=ax, bounds= [ 5.75,9.25,17.75,27 ],
                color = cmap_bgsb,filled=True, label = "BGS Bright")
    
    plot_2d_dist(tot_cat[bgsf_mask]["LOGM_SAGA"], tot_cat[bgsf_mask]["MU_R"], 25, 25, 
                    cmin=1.e-4, cmax=1.0, smooth=2, clevs=[0.68,0.95,0.997],ax=ax, bounds= [ 5.75,9.25,17.75,27 ],
                color = sample_colors["BGS_FAINT"],label = "BGS Faint")
    
    
    plot_2d_dist(tot_cat[lowz_mask]["LOGM_SAGA"], tot_cat[lowz_mask]["MU_R"], 25, 25, 
                    cmin=1.e-4, cmax=1.0, smooth=2, clevs=[0.68,0.95,0.997],ax=ax, bounds= [ 5.75,9.25,17.75,27 ],
                color = sample_colors["LOWZ"], label = "LOWZ")
    
    ax.set_ylim([18,26])
    ax.set_xlim([6,9])
    ax.set_xlabel(r"LogM$_{\rm star}$",fontsize = 15)
    ax.set_ylabel(r"$\mu_r$ (mag/arcsec$^2$)",fontsize = 15)
    ax.legend(frameon=False,fontsize = 13, loc = "lower left")
    plt.savefig("/global/homes/v/virajvm/DESI2_LOWZ/quenched_fracs_nbs/paper_plots/mstar_mur.pdf",bbox_inches="tight")

    return


def get_elg_zred_dist():
    '''
    Function that plots the total ELG redshift distribution
    '''

    zred_elgs = np.load("/pscratch/sd/v/virajvm/catalog_dr1_dwarfs/elg_all_redshifts.npy")
    
    plt.figure(figsize = (4,4))
    plt.title(r"ELG redshift distribution",fontsize = 15)
    plt.hist(zred_elgs,density=True,bins=100, color = "#88CCEE")
    plt.xlim([0,1.6])
    plt.fill_betweenx(y = [0,2], x1=0.8, x2 = 1.6,color = "grey",alpha = 0.4,edgecolor = "none" )
    plt.ylim([0,1.75])
    plt.ylabel(r"$n(z)$",fontsize = 15)
    plt.xlabel(r"Redshift",fontsize = 15)
    plt.savefig("/global/homes/v/virajvm/DESI2_LOWZ/quenched_fracs_nbs/paper_plots/elg_zred_dist.png",bbox_inches="tight",dpi=300)
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
    # use a good colormap and don't interpolate the pixels
    mpl.rc('image', cmap='viridis', interpolation='none', origin='lower')

    # bgsb_shreds = Table.read("/pscratch/sd/v/virajvm/catalog_dr1_dwarfs/iron_photometry/iron_BGS_BRIGHT_shreds_catalog_w_aper_mags_no_pz.fits")
    # bgsf_shreds = Table.read("/pscratch/sd/v/virajvm/catalog_dr1_dwarfs/iron_photometry/iron_BGS_FAINT_shreds_catalog_w_aper_mags_w_pz.fits")
    # elg_shreds = Table.read("/pscratch/sd/v/virajvm/catalog_dr1_dwarfs/iron_photometry/iron_ELG_shreds_catalog_w_aper_mags_w_pz.fits")

    # make_shred_frac_plot()

    make_pcnn_completeness()
    
    # make_img_s_pcnn_panels()

    # make_pcnn_completeness()

    # scarlet_aper_comp()

    # make_summary_stats()

    
    # get_delta_mag_fracflux_plot(resample_bins=False)

    # make_summary_stats()

    # fraction_remain_dwarf_after_aper()

    # halpha_lumi_plot()
    
    # make_stellar_mass_comparison_plot()

    # make_sky_density_plot()
    # 
    
    # tot_cat = Table.read("/pscratch/sd/v/virajvm/catalog_dr1_dwarfs/desi_y1_dwarf_combine_catalog.fits")


    # make_mur_mstar_plot(tot_cat)

    



    

import glob
from astropy.io import fits
from astropy.wcs import WCS
from desi_lowz_funcs import make_subplots

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


if __name__ == '__main__':
    bgsb_shreds = Table.read("/pscratch/sd/v/virajvm/catalog_dr1_dwarfs/iron_photometry/iron_BGS_BRIGHT_shreds_catalog_w_aper_mags_no_pz.fits")
    bgsf_shreds = Table.read("/pscratch/sd/v/virajvm/catalog_dr1_dwarfs/iron_photometry/iron_BGS_FAINT_shreds_catalog_w_aper_mags_w_pz.fits")
    elg_shreds = Table.read("/pscratch/sd/v/virajvm/catalog_dr1_dwarfs/iron_photometry/iron_ELG_shreds_catalog_w_aper_mags_w_pz.fits")



    

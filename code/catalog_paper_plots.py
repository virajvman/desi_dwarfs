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
import cmasher as cmr


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

    from mpl_toolkits.axes_grid1.inset_locator import inset_axes

    clean_cat = Table.read("/pscratch/sd/v/virajvm/catalog_dr1_dwarfs/desi_y1_dwarf_clean_catalog_v4.fits")[:50000]

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
    # cos2020_data = np.load("/pscratch/sd/v/virajvm/desi2_lowz_data/catalogs/cosmos2020_data.npy")
    # iron = SkyCoord(np.array(clean_cat["RA"])*u.degree, np.array(clean_cat["DEC"])*u.degree  )
    # cos = SkyCoord( cos2020_data[0]*u.degree, cos2020_data[1]*u.degree  )
    # idx, d2d, _ = iron.match_to_catalog_sky(cos)
    # clean_cat_cos_match = clean_cat[d2d.arcsec < 1]
    # cos2020_mstar = cos2020_data[2][idx][d2d.arcsec < 1]

    # ##make the plot

    ax = make_subplots(ncol = 4,nrow = 1,col_spacing = 0.25)

    title_size = 14

    xmstar = "LOGM_SAGA"
    cmap = cmr.dusk_r
    
    vmin = 1
    vmax = 1000
    
    # ax[0].set_title(r"CIGALE (no AGN)",fontsize = title_size )
    # ax[0].hist2d(clean_cat[xmstar][cigale_mask],clean_cat["LOGM_CIGALE"][cigale_mask],range= ( (6,9.5),(6,9.5)),bins=50,norm=LogNorm(vmin=vmin,vmax=vmax),cmap=cmap )

    xpos = 7.4
    ypos = 6.67
    fsize = 14

    ax_id = 0
    ax[ax_id].set_title(r"CIGALE (no AGN)",fontsize = title_size )
    # h, xedges, yedges, im=ax[ax_id].hist2d(clean_cat_hu_match[xmstar],hu_match["LOGMSTAR_HU"],range= ( (6,9.5),(6,9.5)),bins= 50,norm=LogNorm(vmin=vmin,vmax=vmax) ,cmap=cmap, rasterized=True)

    # bias, scatter = measure_bias_scatter(clean_cat_hu_match[xmstar].data,hu_match["LOGMSTAR_HU"])
    
    ax[ax_id].text( xpos,ypos,rf"b = {bias:.2f}, $\sigma$ = {scatter:.2f}",fontsize = fsize)

    
     # Create a colorbar
    cbar = plt.colorbar(im, ax=ax[ax_id], orientation='horizontal', pad=0.05)
    cbar.ax.set_position([
        0.295,   # Left position
        0.62,  # Top position
        ax[ax_id].get_position().width * 0.1,  # Width (40% of plot width)
        0.02  # Height (thin bar)
    ])
    
    # ax_id = 1
    # ax[ax_id].set_title(r"COSMOS2020",fontsize = title_size )
    # ax[ax_id].scatter(clean_cat_cos_match[xmstar],cos2020_mstar,color = "purple",s=10,marker="s")

    # bias, scatter = measure_bias_scatter(clean_cat_cos_match[xmstar].data,cos2020_mstar) 

    # ax[ax_id].text( xpos,ypos,rf"b = {bias:.2f}, $\sigma$ = {scatter:.2f}",fontsize = fsize)

    ax_id = 1
    ax[ax_id].set_title(r"GSWLC",fontsize = title_size )
    h, xedges, yedges, im=  ax[ax_id].hist2d(clean_cat_gswlc_match[xmstar],gswlc_match["LOGMSTAR"],range= ( (6,9.5),(6,9.5)),bins= 50,norm=LogNorm(vmin=1, vmax=50) ,cmap=cmap, rasterized=True)

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
    ax_id = 2
    ax[ax_id].set_title(r"Fastspecfit",fontsize = title_size )
    h, xedges, yedges, im =  ax[ax_id].hist2d(clean_cat_fspec_match[xmstar],fspec_mstar_f,range= ( (6,9.5),(6,9.5)),bins= 50,norm=LogNorm(vmin=1, vmax=1000) ,cmap=cmap, rasterized=True)

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
    ax_id = 3
    ax[ax_id].set_title(r"gr-based, de Los Reyes+(2024)",fontsize = title_size )
    h, xedges, yedges, im =  ax[ax_id].hist2d(clean_cat[xmstar], clean_cat["LOGM_M24"] ,range= ( (6,9.5),(6,9.5)),bins= 50,norm=LogNorm(vmin=1, vmax=1000) ,cmap=cmap, rasterized=True)

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
        axi.set_xlim([6.5,9.25])
        axi.set_ylim([6.5,9.25])
        axi.plot([6,11],[6,11],color = "k",lw = 1)
        axi.set_xlabel(r"gr-based $\log_{10}(M_{\bigstar})$",size= 16)
        ax[0].set_ylabel(r"$\log_{10}(M_{\bigstar})$",size= 16)
        ax[0].grid(ls = ":",color = "lightgrey",alpha = 0.5)
    
        if i != 0:
            axi.set_yticklabels([])

    plt.savefig("/global/homes/v/virajvm/DESI2_LOWZ/quenched_fracs_nbs/paper_plots/stellar_mass_comp.pdf",bbox_inches="tight")
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
    cmap.set_under(alpha=0)
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

    # make_pcnn_completeness()
    
    # make_img_s_pcnn_panels()

    # make_pcnn_completeness()

    # scarlet_aper_comp()

    # make_summary_stats()

    
    # get_delta_mag_fracflux_plot(resample_bins=False)

    # make_summary_stats()

    # fraction_remain_dwarf_after_aper()

    # halpha_lumi_plot()
    
    make_stellar_mass_comparison_plot()

    # make_sky_density_plot()
    # 
    
    # tot_cat = Table.read("/pscratch/sd/v/virajvm/catalog_dr1_dwarfs/desi_y1_dwarf_combine_catalog.fits")


    # make_mur_mstar_plot(tot_cat)

    



    

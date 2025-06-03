import numpy as np
from astropy.io import fits
from astropy.table import Table, vstack
from astropy.convolution import convolve, Gaussian1DKernel
import astropy.units as u
import matplotlib.pyplot as plt
from desi_lowz_funcs import process_img, download_few_spectra
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.ticker import MaxNLocator
import numpy as np
import matplotlib as mpl


def get_spectra(cat):
    '''
    We will download the spectra is if it is not already saved
    '''

    tgid = cat["TARGETID"][0]
    
    waves, fluxs, ivars = download_few_spectra(cat,ncores=1)

    return waves, fluxs, ivars


def make_spectra_panel(tgids_interest, file_name, tot_cat,wave_min,wave_max=3500, wave_max = 9200):

    fig, ax = plt.subplots(len(tgids_interest), 1, figsize=(15, int(5*len(tgids_interest)) ),sharex=True)
    plt.subplots_adjust(hspace = 0.1)
    
    for i in range(len(blue_tgids)):
    
        temp = tot_cat[tot_cat["TARGETID"] ==  tgids_interest[i]]
    
        waves, fluxs, ivars = download_few_spectra(temp,ncores=1)
    
        # Your image processing
        img_data = fits.open(temp["IMAGE_PATH"][0])[0].data
        rgb_img = process_img(img_data, cutout_size=64, org_size=350)
    
        zred = temp["Z"][0]
        
        # Main spectrum plotting
        ax[i].plot(waves['brz']/(1+zred), fluxs['brz'][0], color='grey', alpha=0.25, lw=1)
        ax[i].plot(waves['brz']/(1+zred), np.sqrt(1/ivars['brz'][0]), color="darkorange", alpha=0.25)
        ax[i].plot(waves['brz'][5:-5]/(1+zred), convolve(fluxs['brz'][0], Gaussian1DKernel(3))[5:-5], color='k', lw=1.25)
        
        ax[i].set_xlim([wave_min, wave_max])
        ax[i].tick_params(axis='both', labelsize=17)
        ax[i].set_ylim([0, np.median(fluxs['brz'][0]) * 10])
        if i == 2:
            ax[i].set_xlabel('Rest-Frame Wavelength [$\\AA$]', fontsize=17)
            
        ax[i].set_ylabel('$F_{\\lambda}$ [$10^{-17}$ erg s$^{-1}$ cm$^{-2}$ $\\AA^{-1}$]', fontsize=17)
        ax[i].yaxis.set_major_locator(MaxNLocator(integer=True))
        # Add inset axes in top-right
        inset_ax = inset_axes(ax[i], width=1.75, height=1.75, loc='upper right', borderpad=2)
        inset_ax.imshow(rgb_img)
        inset_ax.set_title(f"{temp['TARGETID'][0]}",fontsize = 12)
        inset_ax.axis('off')  # Hide axis ticks and frame
        inset_ax.text(0.5, 0.95,"(%.3f,%.3f, z=%.3f)"%(temp["RA"][0],temp["DEC"][0], temp["Z"][0]) ,color = "white",fontsize = 9.25,
                          transform=inset_ax.transAxes, ha = "center", verticalalignment='top')
                   
    
    plt.savefig(f"/global/homes/v/virajvm/DESI2_LOWZ/quenched_fracs_nbs/paper_plots/{file_name}.pdf",bbox_inches="tight")
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

    supernova_tgids = [39628414184849968,39627896712597936, 39627702956722996 ]
    blue_tgids = [39627844418013010, 39633034554639391, 39627994867699225]
    
    tot_cat = Table.read("/pscratch/sd/v/virajvm/catalog_dr1_dwarfs/desi_y1_dwarf_combine_catalog.fits")

    make_spectra_panel(blue_tgids, "very_blue_egs.pdf", tot_cat,wave_max=3400, wave_max = 9200)
    make_spectra_panel(supernova_tgids, "supernova_egs.pdf", tot_cat,wave_max=3500, wave_max = 9200)
    


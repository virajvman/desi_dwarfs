import numpy as np
from astropy.io import fits
from astropy.table import Table, vstack
from astropy.convolution import convolve, Gaussian1DKernel
import astropy.units as u
import matplotlib.pyplot as plt
from desi_lowz_funcs import process_img, download_few_spectra, save_jpg
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.ticker import MaxNLocator
import numpy as np
import matplotlib as mpl
import os
import requests
import matplotlib.image as mpimg


def get_spectra(cat):
    '''
    We will download the spectra is if it is not already saved
    '''

    tgid = cat["TARGETID"][0]
    
    waves, fluxs, ivars = download_few_spectra(cat,ncores=1)

    return waves, fluxs, ivars


def make_spectra_panel(tgids_interest, file_name, tot_cat,wave_min=3500, wave_max = 9200, save_folder = "/pscratch/sd/v/virajvm/catalog_dr1_dwarfs/iron_spectra/example_spec/"):

    fig, ax = plt.subplots(len(tgids_interest), 1, figsize=(15, int(5*len(tgids_interest)) ),sharex=True)
    plt.subplots_adjust(hspace = 0.1)

    print(f"Total number of TGIDS = {len(tgids_interest)}")

    image_size = 64

    session = requests.Session()
    
    for i in range(len(tgids_interest)):
    
        temp = tot_cat[tot_cat["TARGETID"] ==  tgids_interest[i]]
    
        waves, fluxs, ivars = download_few_spectra(temp,ncores=1)

        ##
        ra_i = temp["RA"].data[0]
        dec_i = temp["DEC"].data[0]
        tgid_i = temp["TARGETID"].data[0]

        img_path = save_folder + f"/img_{tgid_i}.jpg"
        #check if img exists first
        if os.path.exists(img_path):
            img = mpimg.imread(img_path)
        else:
            save_jpg(ra_i,dec_i,img_path,session, size=image_size)
            
            img = mpimg.imread(img_path)
                
        ###
    
        zred = temp["Z"][0]

        #save all this in a file

        np.savez(
            save_folder + f"spec_{tgids_interest[i]}.npz",
            wave=waves['brz'],
            flux = fluxs['brz'][0],
            ivar = ivars['brz'][0],
            image = img, 
            zred = zred,
            mag_r = temp["MAG_R"][0],
            dchi2 = temp["DELTACHI2"][0],
            tgid = tgids_interest[i],
            logm = temp["LOGM_SAGA"][0])
        
        # Main spectrum plotting
        ax[i].plot(waves['brz']/(1+zred), fluxs['brz'][0], color='grey', alpha=0.25, lw=1)
        ax[i].plot(waves['brz']/(1+zred), np.sqrt(1/ivars['brz'][0]), color="darkorange", alpha=0.125)
        
        ax[i].plot(waves['brz'][5:-5]/(1+zred), convolve(fluxs['brz'][0], Gaussian1DKernel(5))[5:-5], color='k', lw=1.25)

        # sigma = 5
        # kernel_size = int(8 * sigma + 1)  # cover full kernel
        # from scipy.signal import gaussian
        # g = gaussian(kernel_size, sigma)
        # g /= np.sum(g)  # normalize kernel
        # g2 = g**2       # square of kernel for variance propagation
    
        # smoothed_var = convolve(1/ivars['brz'][0][5:-5], g2, mode='same')

        # ax[i].plot(waves['brz'][5:-5]/(1+zred), np.sqrt(smoothed_var), color='darkorange', lw=1.25)
        
        
        ax[i].set_xlim([wave_min, wave_max])
        ax[i].tick_params(axis='both', labelsize=17)
        ax[i].set_ylim([0, np.median(fluxs['brz'][0]) * 10])
        if i == len(tgids_interest) - 1:
            ax[i].set_xlabel('Rest-Frame Wavelength [$\\AA$]', fontsize=17)
            
        ax[i].set_ylabel('$F_{\\lambda}$ [$10^{-17}$ erg s$^{-1}$ cm$^{-2}$ $\\AA^{-1}$]', fontsize=17)
        ax[i].yaxis.set_major_locator(MaxNLocator(integer=True))
        # Add inset axes in top-right
        inset_ax = inset_axes(ax[i], width=1.75, height=1.75, loc='upper right', borderpad=2)
        inset_ax.imshow(img)
        inset_ax.set_title(f"{temp['TARGETID'][0]}",fontsize = 12)
        inset_ax.axis('off')  # Hide axis ticks and frame
        # inset_ax.text(0.5, 0.95,"(%.3f,%.3f, z=%.3f)"%(temp["RA"][0],temp["DEC"][0], temp["Z"][0]) ,color = "white",fontsize = 9.25,
        #                   transform=inset_ax.transAxes, ha = "center", verticalalignment='top')

        inset_ax.text(0.5, 0.95,"(mag$_{r}$=%.1f, z=%.3f)"%(temp["MAG_R"][0], temp["Z"][0]) ,color = "white",fontsize = 9.25,
                          transform=inset_ax.transAxes, ha = "center", verticalalignment='top')
                   
    
    plt.savefig(f"/global/homes/v/virajvm/DESI2_LOWZ/quenched_fracs_nbs/paper_plots/{file_name}.pdf",bbox_inches="tight")
    plt.close()


EMISSION_LINES = {
    r'[OII]': 3727.09,
    r'H$\gamma$': 4340.47,
    r'H$\beta$': 4861.33,
    r'[OIII]': 4958.91,
    r'[OIII]': 5006.84,
    r'H$\alpha$': 6562.80,
    r'[SII]': 6722.5} #choosing average SII value


import matplotlib.transforms as mtransforms

def add_floating_emission_lines(
    ax,
    line_dict,
    fontsize=13,
    color='k',
    lw=1.5,
):
    """
    Add floating emission-line labels with short vertical ticks below them.

    y values are in *axes coordinates* (0–1), x is in data coordinates.
    """

    trans = mtransforms.blended_transform_factory(
        ax.transData, ax.transAxes
    )

    xmin, xmax = ax.get_xlim()

    for label, wave in line_dict.items():

        text_y=0.92-0.04
        tick_y0=0.86-0.04
        tick_y1=0.90-0.04
    
        if xmin <= wave <= xmax:
            if wave == 4861.33 or wave == 6722.5:
                text_y -= 0.14
                tick_y0 -= 0.14
                tick_y1 -= 0.14
    
            # label
            ax.text(
                wave,
                text_y,
                label,
                transform=trans,
                ha='center',
                va='bottom',
                fontsize=fontsize,
                color=color,
                clip_on=False,
                zorder=5,
            )

            # floating tick
            ax.plot(
                [wave, wave],
                [tick_y0, tick_y1],
                transform=trans,
                color=color,
                lw=lw,
                solid_capstyle='butt',
                zorder=5,
            )

    
def make_spectra_panel_sample(tgids_interest, file_name, tot_cat,wave_min=3500, wave_max = 9200, save_folder = "/pscratch/sd/v/virajvm/catalog_dr1_dwarfs/iron_spectra/example_spec/",lims=None):

    fig, ax = plt.subplots(2, 2, figsize=(20, int(3*2) ),sharex=True)
    plt.subplots_adjust(hspace = 0.1,wspace=0.1)


    fig.suptitle(r"$10^7 < M_{\bigstar} / M_{\odot} < 10^{7.5}$ galaxies in DESI between $19\lesssim r \lesssim 22$",fontsize = 21)

    
    ax = [ ax[0,0], ax[1,0], ax[0,1], ax[1,1] ]

    sample_colors = ["#882255", "#CC6677", "#DDCC77", "#88CCEE" ]

    print(f"Total number of TGIDS = {len(tgids_interest)}")

    image_size = 48

    session = requests.Session()
    
    for i in range(len(tgids_interest)):
    
        temp = tot_cat[tot_cat["TARGETID"] ==  tgids_interest[i]]
    
        waves, fluxs, ivars = download_few_spectra(temp,ncores=1)

        ##
        ra_i = temp["RA"].data[0]
        dec_i = temp["DEC"].data[0]
        tgid_i = temp["TARGETID"].data[0]

        img_path = save_folder + f"/img_{tgid_i}.jpg"
        #check if img exists first
        # if os.path.exists(img_path):
        #     img = mpimg.imread(img_path)
        # else:
        save_jpg(ra_i,dec_i,img_path,session, size=image_size)
        
        img = mpimg.imread(img_path)
                
        ###
    
        zred = temp["Z"][0]

        #save all this in a file

        np.savez(
            save_folder + f"spec_{tgids_interest[i]}.npz",
            wave=waves['brz'],
            flux = fluxs['brz'][0],
            ivar = ivars['brz'][0],
            image = img, 
            zred = zred,
            mag_r = temp["MAG_R"][0],
            dchi2 = temp["DELTACHI2"][0],
            tgid = tgids_interest[i],
            logm = temp["LOGM_SAGA"][0])
        
        # Main spectrum plotting
        ax[i].plot(waves['brz']/(1+zred), fluxs['brz'][0], color='grey', alpha=0.15, lw=1)
        
        ax[i].plot(waves['brz'][5:-5]/(1+zred), convolve(fluxs['brz'][0], Gaussian1DKernel(5))[5:-5], color=sample_colors[i], lw=1.5)
        
        ax[i].set_xlim([wave_min, wave_max])
        ax[i].tick_params(axis='both', labelsize=17)
        ax[i].set_ylim([lims[i][0], lims[i][1] ])

        if i == 0:
            ax[i].set_yticks([0,6,13]) 
        elif i == 1:
            ax[i].set_yticks([0,2,4]) 
        elif  i == 2:
            ax[i].set_yticks([0,5,10]) 
        else:
            ax[i].set_yticks([0,11,23]) 
            
        ax[-1].set_xlabel('Rest-Frame Wavelength [$\\AA$]', fontsize=19)
        ax[1].set_xlabel('Rest-Frame Wavelength [$\\AA$]', fontsize=19)
        

        if i < 2:
            ax[i].set_ylabel('$F_{\\lambda}$', fontsize=19)
            
        # ax[i].yaxis.set_major_locator(MaxNLocator(integer=True))
        # Add inset axes in top-right
        inset_ax = inset_axes(ax[i], 
                              width=1.55, 
                              height=1.55, 
                              loc='upper right', 
                              bbox_to_anchor=(0.965, 0.855),  # (x, y)
                            bbox_transform=ax[i].transAxes,
                              borderpad=0)
        
        inset_ax.imshow(img)
        inset_ax.set_title(f"{temp['TARGETID'][0]}",fontsize = 13)
        inset_ax.axis('off')  # Hide axis ticks and frame

        # 

        inset_ax.text(0.5, 0.95,"z=%.3f"%(temp["Z"][0]) ,color = "white",fontsize = 19,
                          transform=inset_ax.transAxes, ha = "center", verticalalignment='top')

        inset_ax.text(0.5, 0.2,"r=%.1f"%(temp["MAG_R"][0]) ,color = "white",fontsize = 19,
                          transform=inset_ax.transAxes, ha = "center", verticalalignment='top')


        if i == 2 or i == 0:
            add_floating_emission_lines(ax[i], EMISSION_LINES)
                   
    plt.tight_layout()
    plt.savefig(f"/global/homes/v/virajvm/DESI2_LOWZ/quenched_fracs_nbs/paper_plots/{file_name}.png",bbox_inches="tight",dpi=300)
    plt.show()
    


    
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
    eg_tgids =  [39627427021851828, 39627491345697115, 2705980336898048, 39627555304643301, 39627391634506945] #  [  39627322709513192, 39627345413284126, 2705974209019904, 39627357518039557 ]
    
    tot_cat = Table.read("/pscratch/sd/v/virajvm/catalog_dr1_dwarfs/desi_y1_dwarf_combine_catalog.fits")

    # make_spectra_panel(blue_tgids, "very_blue_egs.pdf", tot_cat,wave_max=3400, wave_max = 9200)
    # make_spectra_panel(supernova_tgids, "supernova_egs.pdf", tot_cat,wave_max=3500, wave_max = 9200)
    make_spectra_panel(eg_tgids, "dwarf_egs.pdf", tot_cat,wave_min=3400, wave_max = 9200)
    
    


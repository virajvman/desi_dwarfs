## This is script that generates mock spectra and forward model through desi pipeline

from scipy.interpolate import UnivariateSpline
import sys
sys.path.append('/global/u1/v/virajvm/DESI2_LOWZ/feasiBGS')
from feasibgs import forwardmodel as FM 
from tqdm import trange
import numpy as np
from astropy.cosmology import Planck18
from astropy import units as u
from scipy.interpolate import interp1d
from astropy.convolution import convolve, Gaussian1DKernel
from astropy.io import fits
from astropy.table import Table
import sys
import os
import pandas as pd
import matplotlib.pyplot as plt
import speclite.filters



def add_scores(file_path,cts = 100):
    with fits.open(file_path, mode='update') as hdulist:
        # open a random hdu
        table_hdu = fits.open("/pscratch/sd/v/virajvm/lowz_mock_spectra/temp_lowz_object_1.fits")["SCORES"]
        # Get the data and slice to only include the first two rows
        original_data = table_hdu.data
        new_data = original_data[:cts]
        new_table_hdu = fits.BinTableHDU(data=new_data, header=table_hdu.header)
        hdulist.append(new_table_hdu)
        hdulist.flush()  # This writes the updates to the file
        
    return



def apply_flux(wave, flux, redshift, rfib_mag,decam=None,verbose=True):
    '''
    Convert spectral luminosity to spectral flux
    wave, flux : input spectrum quants in luminosity
    redshift : to redshift wavelenght and dim luminosity 
    rfib_mag : the fiber mag to which we want to bring the overall spectra
    
    Once scaling is done, r-mag is also computed
    '''
    
    wave_new = wave * (1 + redshift)

    #compute the fiducial magnitude so we know how to scale the flux
    wlen = wave_new * u.Angstrom
    flux_new = flux * u.erg / (u.cm**2 * u.s * u.Angstrom)

    ## compute the magnitude then!
    mags_decam = decam.get_ab_magnitudes(flux_new, wlen)
    rmags_decam = float(mags_decam["decamDR1noatm-r"])
    gmags_decam = float(mags_decam["decamDR1noatm-g"])
    
    #then compute the flux scaling factor by comparing the rfib_mag
    flux_factor = 10**(0.4*(rmags_decam - rfib_mag))

    flux_new2 = flux_new * flux_factor
    
    # interpolate spectra to same wavelength grid as DESI 
    wave_i = np.arange(3000, 1.2e4, 0.2)
    interp_flux = interp1d(wave_new, flux_new2) 
    flux_i = interp_flux(wave_i)
    

    
    
    #this interpolation has been confirmed to be working by visual inspection
    return wave_i, flux_i, float(rmags_decam), float(gmags_decam), float(flux_factor)


def generate_mock_spectra(emission=False,low_emi=False,plot=True,summ_file_name=None,decam=None):
    '''
    Function that mock observes a bunch of FSPS spectra. Note that each redrock file we store max of 1500 spectra in it so for efficiency reasons, we will have one 1500 set of spectra correspond to only mock spectra with various different fiber mags. So we will have XX different files correspond to the XX different model files we have.
        
             ## compute some initial value of Halpha flux
         ## if there is no emission, we will set the ha_flux = 0 to be consistent with FASTSPECFIT notation
         # spl = UnivariateSpline(wave_i,flux_i,s=0)
         # ha_val_temp = spl.integral(6556,6575) - spl.integral(6512,6512 + (6575 - 6556))
         
    '''

    #load all the model spectra info we have 
    if emission == True:
        summ_df = Table.read("/pscratch/sd/v/virajvm/fsps_model_spectra_V2/fsps_emi_model_summary_v2.csv")
        
#         summ_df = Table.read("fsps_model_spectra/fsps_emi_model_summary.csv")
#         ## for the emi model I am only using model with age >=4
#         summ_df = summ_df[ (summ_df["age"] >= 4) & (summ_df["tau"] == 2) ]

    if emission == False:
        summ_df = Table.read("fsps_model_spectra/fsps_cont_model_summary.csv")

    print(len(summ_df))
    
    ## strategy is to load the model files one by one and then do 1500 random initializations of it ..
    max_index = len(summ_df)
    
    #the list of file names that we want to run gpu-fied redrock on
    all_file_names = []
    
    for index_i in trange(max_index):
        #read the properties of that model
        logZ_p = summ_df["logZ"][index_i]
        logU_p = summ_df["logU"][index_i]
        tau_p = summ_df["tau"][index_i]
        age_p = summ_df["age"][index_i]
        halpha_ew_p = summ_df["HALPHA_EW"][index_i]
        
        ## load the model data
        if emission == True:
            data = np.loadtxt("/pscratch/sd/v/virajvm/fsps_model_spectra_V2/spectra_emi_mlogz_%.1f_mlogu_%.1f_tau_%d_age_%.2f.txt"%(-1*logZ_p, -1*logU_p, tau_p, age_p))

        if emission == False:
            data = np.loadtxt("fsps_model_spectra/spectra_cont_mlogz_%.1f_mlogu_%.1f_tau_%d_age_%.2f.txt"%(-1*logZ_p, -1*logU_p, tau_p, age_p))

        ## get the wavelength and flux data for this model
        wave_i = data[:,0]
        flux_i = data[:,1]
           
        ## for this one file/spectra, generate 1500 random fiber mag realizations
        rfib_mags = np.random.uniform(low = 19, high= 23.5, size = 1500)
            #also put them at random redshifts
        rnd_zred = np.random.uniform(low = 0.001, high = 0.15,size=1500)

        ## we will loop over each of these objects to get the magnitudes the fluxs            
        gr_vals = []
        waves_vals = []
        fluxs_vals = []
        halpha_ews = []
        flux_factor_vals = []
        model_index = []

        for j in range(1500):
            if j % 500 == 0:
                print(j)
                
            wave_ij, flux_ij, rmag_ij, gmag_ij, flux_factor_ij = apply_flux(wave_i, flux_i, rnd_zred[j], rfib_mags[j], decam=decam)
            waves_vals.append(wave_ij)
            halpha_ews.append(halpha_ew_p)
    
            gr_vals.append(rmag_ij - gmag_ij)
            
            fluxs_vals.append(flux_ij)
            flux_factor_vals.append(flux_factor_ij)
            model_index.append(index_i)

        waves_vals = np.array(waves_vals)
        fluxs_vals = np.array(fluxs_vals)

        ## save the summary file for all these results ..
        dict_summary = {"Z" : rnd_zred, "MAG_R" : rfib_mags, "g-r":gr_vals, "flux_factor": flux_factor_vals, "model_index" : model_index, "HALPHA_EW" : halpha_ews}

        df_summary = pd.DataFrame(dict_summary)

        if emission==True:
            string = "emi"
        if emission==False:
            string="cont"

        df_summary.to_csv("/pscratch/sd/v/virajvm/fsps_model_spec_summs_V2/summary_%s_model_%d.csv"%(string,index_i) )

        ## then generate the mock observations ... 

        fdesi = FM.fakeDESIspec()

        name_180 = "/pscratch/sd/v/virajvm/fsps_dwarf_spectra_V2/spec_obj_180s_%s_model_%d.fits"%(string,index_i)
        name_900 = "/pscratch/sd/v/virajvm/fsps_dwarf_spectra_V2/spec_obj_900s_%s_model_%d.fits"%(string,index_i)
        name_3600 = "/pscratch/sd/v/virajvm/fsps_dwarf_spectra_V2/spec_obj_3600s_%s_model_%d.fits"%(string,index_i)
        
        all_file_names.append(name_180)
        all_file_names.append(name_900)
        all_file_names.append(name_3600)
        
        
        #doing both BGS BRIGHT AND LOWZ type observing
        #the flux needs to be in units of 1e-17 it seems 
        specdata = fdesi.simExposure(wave = waves_vals[0], flux = np.array(fluxs_vals)*1e17, airmass = 1.0, 
                                         exptime=180,seeing = 1.1,filename = name_180)

        specdata = fdesi.simExposure(wave = waves_vals[0], flux = np.array(fluxs_vals)*1e17, airmass = 1.0, 
                                     exptime=900,seeing = 1.1,filename = name_900)

        specdata = fdesi.simExposure(wave = waves_vals[0], flux = np.array(fluxs_vals)*1e17, airmass = 1.0, 
                                     exptime=3600,seeing = 1.1,filename = name_3600)


        ## and then add a mock SCORES table there
        add_scores(name_180, cts = 1500)
        add_scores(name_900, cts = 1500)
        add_scores(name_3600, cts = 1500)

    return all_file_names
        
if __name__ == '__main__':
    
    # decam = speclite.filters.load_filters('decam2014-*')
    decam = speclite.filters.load_filters('decamDR1noatm-*')
    
    all_file_names = generate_mock_spectra(emission=True,plot=False,summ_file_name="summary_emi",decam=decam)
    
    #create a list of files to feed into gpu-fied redrock
    
    print(len(all_file_names))
    output_file = "/pscratch/sd/v/virajvm/fsps_dwarf_spectra_V2/fsps_emi_sims_list_V2.ascii"
    with open(output_file, "w") as f:
        for name in all_file_names:
            f.write(name + "\n")

    # for i in trange(25):
        # generate_mock_spectra(emission=True,low_emi=True,plot=False,summ_file_name="summary_lowemi",file_num=i,loop_num = 50,decam=decam)
        # generate_mock_spectra(emission=False,plot=False,summ_file_name="summary_cont",file_num=i,loop_num = 25,decam=decam) 
            
    
     # if plot and _ % 10 == 0:
        #     plt.figure(figsize = (4,4))
        #     ## also do some plotting for reference
        #     plt.plot(wave_i,flux_i)
        #     xmin=6500
        #     xmax = 6650
        #     plt.xlim([xmin,xmax])
        #     plt.vlines(x = 6556, ymax=0.5*np.max(flux_i), ymin = 2*np.min(flux_i),color = "k")
        #     plt.vlines(x = 6575,ymax=0.5*np.max(flux_i), ymin = 2*np.min(flux_i),color = "k")
        #     grid = np.linspace(xmin, xmax, 1000)
        #     plt.plot(grid, spl(grid), ls = "dotted",color = "r")
        #     plt.yscale("log")
        #     plt.show()

#         if _ < 3 and plot:
#             plt.figure(figsize = (4,4))
#             plt.plot(wave_i,flux_i)
#             xmin=6500
#             xmax = 6650
#             plt.xlim([xmin,xmax])

#             plt.vlines(x = 6556, ymax=0.5*np.max(flux_i), ymin = 2*np.min(flux_i),color = "k")
#             plt.vlines(x = 6575,ymax=0.5*np.max(flux_i), ymin = 2*np.min(flux_i),color = "k")

#             plt.vlines(x = 6512, ymax=0.5*np.max(flux_i), ymin = 2*np.min(flux_i),color = "r")
#             plt.vlines(x = 6512 + (6575 - 6556),ymax=0.5*np.max(flux_i), ymin = 2*np.min(flux_i),color = "r")

#             spl = UnivariateSpline(wave_i,flux_i,s=0)
#             grid = np.linspace(xmin, xmax, 1000)
#             plt.plot(grid, spl(grid), ls = "dotted",color = "r")
#             plt.yscale("log")
#             plt.show()

#             ## now do an intengral
#             print(index, ha_flux_i)
#             print(index, spl.integral(6556,6575) - spl.integral(6512,6512 + (6575 - 6556)))
#             print("--"*5)

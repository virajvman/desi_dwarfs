## This is script that generates mock spectra and forward model through desi pipeline

from scipy.interpolate import UnivariateSpline
import sys
sys.path.append('/global/u1/v/virajvm/DESI2_LOWZ/feasiBGS')
from feasibgs import forwardmodel as FM 
from tqdm import trange
from tqdm import tqdm
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
    
    
Lsolar = 3.826e33 #erg/s


def get_spectrum(ssp_obj,age,zred=0.001,Lgal = 2e7,use_redshift=False):
    '''
    Scale the flux in the spectrum to account for redshift luminosity distance and galaxy luminosity 
    '''
    # ssp_obj.params["zred"] = zred
    # it seems like the redshift parameter is not changing the flux at all??
    # the flux is always returned in solar luminosities, but that should still impact the spectrum amplitude 
    # it also not redshifting the wavelenght.
    ## so the zred parameter does nothing to the get_spectrum function ...
    
    wave, flux = ssp_obj.get_spectrum(tage = age, peraa = True)
    wave_2 = wave[(wave > 2000) & (wave < 2e4)]
    flux_2 = flux[(wave > 2000) & (wave < 2e4)]
    

    if use_redshift == False:
        #no correction for redshift and stuff
        return wave_2, flux_2 * Lsolar * Lgal 
    
    else:
        #computing luminosity distance stuff
        dist_cm = Planck18.luminosity_distance(zred).to(u.cm).value
        flux_factor = 1/(4*np.pi*dist_cm**2)
    
        return wave_2*(1+zred), flux_2 * Lsolar * Lgal * flux_factor 


# import DESI related modules - 
from desimodel.footprint import radec2pix      # For getting healpix values
import desispec.io                             # Input/Output functions related to DESI spectra
from desispec import coaddition                # Functions related to coadding the spectra
from astropy.convolution import convolve, Gaussian1DKernel
from astropy.table import QTable

def get_spectra(tgid_interest,hpx,plot=True,save=False,save_path=None):
    '''
    Function that gets the spectra for the single object.
    
    The following need to be run before this function
    
    # import DESI related modules - 
    from desimodel.footprint import radec2pix      # For getting healpix values
    import desispec.io                             # Input/Output functions related to DESI spectra
    from desispec import coaddition                # Functions related to coadding the spectra
    from astropy.convolution import convolve, Gaussian1DKernel

    specprod = 'iron'    # Internal name for the EDR
    specprod_dir = '/global/cfs/cdirs/desi/spectro/redux/iron/'

    zpix_cat = Table.read(f'{specprod_dir}/zcatalog/zall-pix-{specprod}.fits', hdu="ZCATALOG")  
    healpix_dir = f'{specprod_dir}/healpix'
    
    '''
    # Let us explore the target directory
    # Note that the target directory is different for the different spectra.
    # We first explore the primary spectra and look at the other spectra later.
    healpix_dir = "/global/cfs/cdirs/desi/spectro/redux/iron/healpix"
    survey = "main"
    program = "bright"
    
    tgt_dir = f'{healpix_dir}/{survey}/{program}/{hpx//100}/{hpx}'
    
    coadd_filename = f'coadd-{survey}-{program}-{hpx}.fits'    
    
    h_coadd = fits.open(f'{tgt_dir}/{coadd_filename}')
    
    coadd_obj = desispec.io.read_spectra(f'{tgt_dir}/{coadd_filename}')
    coadd_tgts = coadd_obj.target_ids().data
    #coadd the spec now 
    
    # Selecting the particular spectra of the targetid
    row = (coadd_tgts == tgid_interest)
    coadd_spec = coadd_obj[row]
    
    spec_combined = coaddition.coadd_cameras(coadd_spec)    
    
    ## remove teh nan fluxes
    
    fluxs = spec_combined.flux['brz'][0]
    fluxs_err = 1/np.sqrt(spec_combined.ivar['brz'][0])
    gmask = (fluxs > fluxs_err)
    # gmask = np.ones_like(fluxs).astype(bool)
    # masking the anomalous low snr pixels here
                                      
    if plot==True:
        plt.figure(figsize = (7, 7))
        # Plot the combined spectrum in maroon
        plt.plot(spec_combined.wave['brz'][gmask], spec_combined.flux['brz'][0][gmask], color = 'grey', alpha = 0.5)
        plt.plot(spec_combined.wave['brz'][gmask], 1/np.sqrt(spec_combined.ivar['brz'][0][gmask]),color = "darkorange",alpha = 1)
        # Over-plotting smoothed spectra 
        plt.plot(spec_combined.wave['brz'][gmask], convolve(spec_combined.flux['brz'][0][gmask], Gaussian1DKernel(1)) , color = 'k', lw = 1.25)

        plt.xlim([4000 - 200, 4000 + 300])
        waves= spec_combined.wave['brz']
        
        wmask = (waves > 6563 - 200) & (waves < 6563 + 300)
#                 
        # plt.ylim([-0.2,20])
        plt.yscale("log")
        
        plt.xlabel('Rest-Frame Wavelength [$\AA$]',fontsize = 22)
        plt.ylabel('$F_{\lambda}$ [$10^{-17} erg\ s^{-1}\ cm^{-2}\ \AA^{-1}$]',fontsize = 22)
        plt.xticks(fontsize = 15)
        plt.yticks(fontsize = 12)
       
        if save:
            plt.savefig(save_path + "plot_%d.png"%tgid_interest)
            plt.close()
        else:
            plt.show()
            
      
    wave_f = spec_combined.wave['brz'][gmask]
    flux_f = convolve(spec_combined.flux['brz'][0][gmask], Gaussian1DKernel(1))
    
    ## the gaussian smoothing helps with pixels that are bad??
    
    flux_f_nosmooth = spec_combined.flux['brz'][0][gmask]
    ## interpolate spectra to same wavelength grid as DESI 
    wave_if = np.arange(3000, 11500, 0.2)
    interp_flux = interp1d(wave_f, flux_f, fill_value = ( flux_f[0], flux_f[-1] ), bounds_error =False ) 
    interp_flux_nosmooth = interp1d(wave_f, flux_f_nosmooth, fill_value = ( flux_f_nosmooth[0], flux_f_nosmooth[-1] ), bounds_error =False ) 
    
    flux_if = interp_flux(wave_if)
    flux_if_nosmooth = interp_flux_nosmooth(wave_if)
    
    if save:
        spec_tab = QTable([wave_if, flux_if],names=('wave', 'flux') )
        spec_tab_nosmooth = QTable([wave_if, flux_if_nosmooth],names=('wave', 'flux') )
        
        spec_tab.write(save_path + "spectra_%d.fits"%tgid_interest,overwrite=True)
        spec_tab_nosmooth.write(save_path + "spectra_nosmooth_%d.fits"%tgid_interest,overwrite=True)
        

    return wave_if, flux_if 


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

def reobs_spectra(model_table, waves, all_fluxs, plot=True,iter_num = 30):
    '''
    Function where we reobserve some very high SNR spectra in DESI after scaling down their fluxes between different fiber mags 
    
    model_table contains all the relevant info regarding the spectra like original redshift, fiber magnitude etc. 
    
    
    old code below on how to do the scalings ... 
     # do we need random in fiber mag or random in halpha flux space??

    #first going by random halpha way
    # ha_vals = np.random.uniform(low = 1, high = 40, size=len(model_table))
    # ini_ha_vals = np.array(model_table["HALPHA_FLUX"])
    # flux_scaling = ha_vals/ini_ha_vals
    # #using this flux factor get the new fiber mag
    # new_rfibmags = np.array(model_table["rfib_mags_decam_noatm"]) - 2.5*np.log10(flux_scaling)
    
        #choose random halpha whereever 19 < new_rfibmags < 23.5, if not that then use random rfib
    final_new_rfib_mags = new_rfibmags
    final_new_rfib_mags[(new_rfibmags < 19)|(new_rfibmags > 23.5)] = rfib_mags[(new_rfibmags < 19)|(new_rfibmags > 23.5)]

    final_flux_factors = flux_scaling
    final_flux_factors[(new_rfibmags < 19)|(new_rfibmags > 23.5)] = flux_factors[(new_rfibmags < 19)|(new_rfibmags > 23.5)]
    
    # model_table["rfib_mags_decam_noatm_new"] = final_new_rfib_mags
    
    
    '''

    fdesi = FM.fakeDESIspec()
    
    all_file_names = []
    
    for num in trange(iter_num):
        #generate random array of fiber mags
        #then go by random rfib way
        rfib_mags = np.random.uniform(low = 19, high= 23.5, size = len(model_table))
        
        #walking through the choice of fiber mag here. 
        #We want to use the FIBERMAG for purposes of modeling the completeness. So it should be same definition as our real data 
        #So preferentially the one we have in our photometric catalogs as easier
        #Given that, we will be treating the photometric catalogs FIBERMAG vals as the truth
        #Another idea is to treat this is to compute the effective fiber band magnitude 
        #we will save both values for future ease
        
        ini_fibmags = np.array(model_table["FIBERMAG_R"])
        # find the flux scaling factors ...
        flux_factors = 10**(0.4*(ini_fibmags - rfib_mags))
    
        #save this new table
        #this is the fiber mag if using FIBERMAG_R as truth
        model_table["rfib_decam_noatm_FIBERMAG_BASIS"] = rfib_mags
        
        #this is the fibermag if using rfib_mags_decam_noatm as truth
        #essentially this is saying given the flux factor, what is the new effective rfib mag 
        model_table["rfib_decam_noatm_NOCORR_BASIS"] = model_table["rfib_decam_noatm_nocorr"] - 2.5*np.log10(flux_factors)
        
        model_table["flux_scaling_factor"] = flux_factors
        
        #save this so we can use it later
        model_table.write("/pscratch/sd/v/virajvm/fsps_dwarf_spectra/bgs_bright_highemi_cat_%d.fits"%num,overwrite=True)
        
        name_180 = "/pscratch/sd/v/virajvm/fsps_dwarf_spectra/spec_reobs_v4_180s_%d.fits"%(num)
        name_900 = "/pscratch/sd/v/virajvm/fsps_dwarf_spectra/spec_reobs_v4_900s_%d.fits"%(num)

        #doing both BGS BRIGHT AND LOWZ type observing
        #the flux needs to be in units of 1e-17 it seems 
        all_file_names.append(name_180)
        all_file_names.append(name_900)
        
        model_fluxs =  np.array(all_fluxs) * flux_factors[:,np.newaxis]
        
        specdata = fdesi.simExposure(wave = waves, flux = model_fluxs, airmass = 1.0, 
                                         exptime=180,seeing = 1.1,filename = name_180)

        specdata = fdesi.simExposure(wave = waves, flux = model_fluxs, airmass = 1.0, 
                                     exptime=900,seeing = 1.1,filename = name_900)

        ## and then add a mock SCORES table there
        add_scores(name_180, cts = len(all_fluxs))
        add_scores(name_900, cts = len(all_fluxs))

    ## show an example of mock simulation
#     if plot == True:
#         index = 4
#         print( rfib_mags[index] )
#         fig = plt.figure(figsize=(12,5))
#         sub = fig.add_subplot(111)
#         sub.plot(specdata.wave["b"], specdata.flux["b"][index],lw = 2)
#         sub.plot(specdata.wave["r"], specdata.flux["r"][index],lw = 2)
#         sub.plot(specdata.wave["z"], specdata.flux["z"][index],lw = 2)

#         sub.plot(waves, model_fluxs[index], c='k', ls='--',lw = 1)

#         sub.set_xlabel('observed-frame wavelength [$A$]', fontsize=20)
#         sub.set_xlim(3e3, 1e4)
#         sub.set_ylabel('flux [$10^{-17} erg/s/cm^2/A$]', fontsize=20)
#         sub.set_ylim(0, 0.8*np.max(specdata.flux["r"][index]))

#         plt.show()


#         fig = plt.figure(figsize=(12,5))
#         sub = fig.add_subplot(111)
#         sub.plot(specdata.wave["r"], specdata.flux["r"][index],color = "lightgrey")
#         sub.plot(specdata.wave["r"], 1/np.sqrt(specdata.ivar["r"][index]),color = "firebrick")

#         smooth_flux = convolve(specdata.flux["r"][index], Gaussian1DKernel(3))

#         sub.plot(specdata.wave["r"], smooth_flux,color = "dimgrey")

#         sub.plot(waves, model_fluxs[index], c='k', ls='--',lw = 1)

#         sub.set_xlabel('observed-frame wavelength [$A$]', fontsize=20)
#         sub.set_xlim(6000, 7000)
#         sub.set_ylabel('flux [$10^{-17} erg/s/cm^2/A$]', fontsize=20)
#         sub.set_ylim(0, 0.8*np.max(specdata.flux["r"][index]))

#         plt.show()
        
    return all_file_names
    
    

def reobs_same_spectra(model_table, waves, all_fluxs):
    '''
    Function where we reobserve high SNR spectra as is and compare new Dchi2 values etc. to see how well our simulations
    recover reality etc.
    '''

    ## then generate the mock observations ... 
    #there are around a 1000 objects in total
    #we want to scale their fiber mags to some random fiber magnitude between 19 - 24
    # and set that whole pipeline up

    fdesi = FM.fakeDESIspec()
    
    # name_180 = "/pscratch/sd/v/virajvm/fsps_dwarf_spectra/spec_reobs_literal_180s.fits"
    name_900 = "/pscratch/sd/v/virajvm/fsps_dwarf_spectra/spec_reobs_literal_900s.fits"

    #doing both BGS BRIGHT AND LOWZ type observing
    #the flux needs to be in units of 1e-17 it seems 
    
    #just to have them all in the same method
    flux_factors = np.ones(len(all_fluxs))
    model_fluxs =  np.array(all_fluxs)  * flux_factors[:,np.newaxis]

    # specdata = fdesi.simExposure(wave = waves, flux = model_fluxs, airmass = 1.0, 
    #                                  exptime=180,seeing = 1.1,filename = name_180)
    
    specdata = fdesi.simExposure(wave = waves, flux = model_fluxs, airmass = 1.0, 
                                     exptime=180,seeing = 1.1,filename = name_180)

    # specdata = fdesi.simExposure(wave = waves, flux = model_fluxs, airmass = 1.0, 
                                 # exptime=900,seeing = 1.1,filename = name_900)

    ## and then add a mock SCORES table there
    add_scores(name_180, cts = len(all_fluxs))
    # add_scores(name_900, cts = len(all_fluxs))
    
    return
        
if __name__ == '__main__':
    
    #load the data table
    model_table = Table.read("/pscratch/sd/v/virajvm/catalog/bgs_bright_highemi_catalog_v4.fits")
    
#     ## save the spectra and compute the fiber magnitude 
    decam1_noatm = speclite.filters.load_filters('decamDR1noatm-*')
    decam = speclite.filters.load_filters('decam2014-*')

    all_highemi_rfibs = []
    all_highemi_rfibs_noatm = []
    all_highemi_rfibs_noatm_nocorr = []
    
    for i in trange(len(model_table)):

        #if already saved then do not worry!
        try:
            tgid_interest = model_table["TARGETID"][i]
            #read the smoothed data as that is what we are returning
            temp_data = Table.read("/pscratch/sd/v/virajvm/bgs_bright_highemi_spectra/spectra_%d.fits"%tgid_interest)
            wave_i, flux_i = np.array(temp_data["wave"]), np.array(temp_data["flux"])
            
            if np.min(wave_i) > 3300:
                # we need to regenerate this spectra so it is interpolated over 3000  - 11k range
                wave_i, flux_i = get_spectra(tgid_interest,model_table["HEALPIX"][i], plot=False, save=True, save_path = "/pscratch/sd/v/virajvm/bgs_bright_highemi_spectra/")
            
        except:
            wave_i, flux_i = get_spectra(model_table["TARGETID"][i],model_table["HEALPIX"][i], plot=False, save=True, save_path = "/pscratch/sd/v/virajvm/bgs_bright_highemi_spectra/")

        ## compute the magnitude of this and see if it checks out ...
        wlen =  wave_i  * u.Angstrom
        flux = flux_i * model_table["MEAN_PSF_TO_FIBER_SPECFLUX"][i] * 1e-17 * u.erg / (u.cm**2 * u.s * u.Angstrom)
        flux_nocorr = flux_i * 1e-17 * u.erg / (u.cm**2 * u.s * u.Angstrom)
        
        # print(np.min(wlen), np.max(wlen),model_table["TARGETID"][i] )
        mags_decam = decam.get_ab_magnitudes(flux, wlen)
        mags_decam1_noatm = decam1_noatm.get_ab_magnitudes(flux, wlen)
        mags_decam1_noatm_nocorr = decam1_noatm.get_ab_magnitudes(flux_nocorr, wlen)
    
        mags_decam_r = mags_decam["decam2014-r"]
        mags_decam_r_noatm = mags_decam1_noatm["decamDR1noatm-r"]
        mags_decam_r_noatm_nocorr = mags_decam1_noatm_nocorr["decamDR1noatm-r"]
        
        #we also compute a fiber magnitude without any correction to have some sort of "effective fiber magnitude" in our analysis
        
        all_highemi_rfibs.append(mags_decam_r)
        all_highemi_rfibs_noatm.append(mags_decam_r_noatm)
        all_highemi_rfibs_noatm_nocorr.append(mags_decam_r_noatm_nocorr)
        
    
    #add the fiber magnitudes to the table 
    model_table["rfib_decam"] = np.concatenate(all_highemi_rfibs)
    model_table["rfib_decam_noatm"] = np.concatenate(all_highemi_rfibs_noatm)
    model_table["rfib_decam_noatm_nocorr"] = np.concatenate(all_highemi_rfibs_noatm_nocorr)
    

    #then save this fits file 
    model_table.write("/pscratch/sd/v/virajvm/catalog/bgs_bright_highemi_catalog_v4.fits",overwrite=True)
        
    ## load all the waves and spectra
    all_tgids = model_table["TARGETID"]

    all_waves = []
    all_fluxs = []
    for ti in tqdm(all_tgids):
        #lpoad the file, choose the smoothed or not smoothed flux here
        path_i = "/pscratch/sd/v/virajvm/bgs_bright_highemi_spectra/spectra_%d.fits"%ti
        # path_i = "/pscratch/sd/v/virajvm/bgs_bright_highemi_spectra/spectra_nosmooth_%d.fits"%ti
        
        tab_i = Table.read(path_i)
        wave_i = tab_i["wave"]
        flux_i = tab_i["flux"]
        
        all_waves.append(wave_i)
        all_fluxs.append(flux_i)
                
    #use this if we want flux scaling to different fainter fiber mags
    all_file_names = reobs_spectra(model_table, all_waves[0], all_fluxs, plot=True,iter_num = 100)
    
    ##use this if no flux scaling, and just reobserve as is
    # reobs_same_spectra(model_table, all_waves[0], all_fluxs)
            
    print(len(all_file_names))
    output_file = "/pscratch/sd/v/virajvm/list_coadds_reobs_sims.ascii"
    with open(output_file, "w") as f:
        for name in all_file_names:
            f.write(name + "\n")
    
        
    #####################################################
    #####################################################
    #####################################################
    #####################################################
    #### run the experiment using fsps spectra
    #####################################################
    #####################################################
    #####################################################
    #####################################################
    
#     import fsps
#     import sys
#     sys.path.append('/global/u1/v/virajvm/DESI2_LOWZ/feasiBGS')
#     from feasibgs import forwardmodel as FM 
    
#     ssp_neb = fsps.StellarPopulation(zcontinuous = 1, add_neb_emission = 1, 
#                                  smooth_velocity = True,sigma_smooth = 50, dust_type = 2, dust2= 0.2, sfh=1)

#     ## setting the parameters now 
#     ssp_neb.params['logzsol'] = -1.5
#     # # set parameters for nebular model
#     ssp_neb.params['gas_logz'] = -1.5 # gas metallicity
#     ssp_neb.params['gas_logu'] = -2.5 # ionization parameter
#     ssp_neb.params["tau"] = 2

#     redshift = 0.001

#     wave_i, flux_i = get_spectrum(ssp_neb,2,zred=redshift,Lgal = 2e7,use_redshift=False)
#     #this means that the units are in luminosity and not flux and we can multiply the redshift factor later in mock observing
#     # the units are in ergs/s/A
#     # wave is in A
    
#     ## do the full redshift range from z = 0 to z0.3
    
#     for d in trange(30):

#         zgrid = np.random.uniform(low = 0.001, high = 0.25, size = 1000)

#         wave_if = np.arange(3000, 1.2e4, 0.2)
#         wlen = wave_if * u.Angstrom

#         all_fluxs = []

#         for i,zi in enumerate(zgrid):
#             # interpolate spectra to same wavelength grid as DESI 
#             interp_flux = interp1d(wave_i * (1+zi), flux_i) 
#             flux_if = interp_flux(wave_if)

#             #compute the magnitude now 
#             flux_unis = flux_if * 10**(-55) * u.erg / (u.cm**2 * u.s * u.Angstrom)

#             mags_decam = decam.get_ab_magnitudes(flux_unis, wlen)

#             rmags_decam = mags_decam["decam2014-r"]

#             flux_factor = 10**(0.4*(rmags_decam - 19))
#             flux_iff = flux_if * flux_factor * 10**(-55)

#             if i < 1:
#                 temp = decam.get_ab_magnitudes(flux_unis * flux_factor,  wlen)
#                 print(temp["decam2014-r"])

#             all_fluxs.append(flux_iff)

#         ## let us mock observe this ...
#         fdesi = FM.fakeDESIspec()

#         file_str = "/pscratch/sd/v/virajvm/temp19_%d.fits"%d
#         d_str = "/pscratch/sd/v/virajvm/temp19_zbest_%d.fits"%d
#         o_str = "/pscratch/sd/v/virajvm/temp19_out_%d.fits"%d

#         specdata = fdesi.simExposure(wave = wave_if, flux = np.array(all_fluxs)*1e17, airmass = 1.0, 
#                                          exptime=180,seeing = 1.1,filename = file_str)

#         ## let us add scores 
#         add_scores(file_str, cts = len(all_fluxs) )

#         ## run redrock            
#         os.system("rrdesi -i %s  -d %s -o %s"%(file_str, d_str, o_str))

#         ## save the true redshifts

#         temp_tab = Table.read(o_str)
#         temp_tab["Z_TRUE"] = zgrid
#         temp_tab.write(o_str,overwrite=True)
        
        
#     ## then combine all these temp tabs together
#     from astropy.table import vstack
#     tab_tot = Table.read("/pscratch/sd/v/virajvm/temp19_out_0.fits")
#     for di in trange(1,30):
#         temp_i = Table.read("/pscratch/sd/v/virajvm/temp19_out_%d.fits"%di)

#         #add this to above
#         tab_tot = vstack([tab_tot, temp_i])
        
#     tab_tot.write("/pscratch/sd/v/virajvm/temp19_out_total.fits",overwrite=True)
    
        
    


from desitarget.targetmask import desi_mask, bgs_mask
# import some helpful python packages 
import os
import numpy as np
from tqdm import trange
from astropy.io import fits
from astropy.table import Table, vstack
from astropy.convolution import convolve, Gaussian1DKernel
import astropy.units as u
import astropy.coordinates as coord
import matplotlib
import matplotlib.pyplot as plt
from astropy.table import Column
from tqdm import trange
import pandas as pd
import fitsio
from astropy.table import Table, vstack
from astropy import units as u
from astropy.coordinates import SkyCoord
import sys
import multiprocessing as mp
from tqdm import tqdm

from easyquery import Query, QueryMaker
reduce_compare = QueryMaker.reduce_compare

from desi_lowz_funcs import add_sweeps_column


def get_fibmag(vac_data_bgs):
    vac_fib_fluxr = vac_data_bgs["FIBERFLUX_R"]
    vac_fibrmag_bgs = 22.5 - 2.5*np.log10(vac_fib_fluxr)
    return vac_fibrmag_bgs
    
    
def get_mags(vac_data_bgs, band="R"):
    vac_fluxr = vac_data_bgs["FLUX_" + band]
    vac_fluxr_err = np.sqrt(1/vac_data_bgs["FLUX_IVAR_" + band])
    # vac_mwr = vac_data_bgs["MW_TRANSMISSION_" + band]
    
    ## apparently the fastspecfit catalogs are already corrected for MW extinction 
    # https://fastspecfit.readthedocs.io/en/latest/fastspec.html
    #the data model here says that the flux are already extinction corrected!!

    #we have the magntudes
    vac_rmag_bgs = 22.5 - 2.5*np.log10(vac_fluxr)
    ### this formula for rmag uncertainties is from https://sites.astro.caltech.edu/~george/ay122/Ay122a_Photometry1.pdf pg 34
    ### it is an approximation that is valid in small error regime.
    vac_rmag_err_bgs = 1.087*(vac_fluxr_err/vac_fluxr)
    
    return vac_rmag_bgs, vac_rmag_err_bgs

vacdir = '/global/cfs/cdirs/desi/public/dr1/vac/dr1/lsdr9-photometry/iron/v1.1/observed-targets'
# vacdir = '/global/cfs/cdirs/desi/public/edr/vac/edr/lsdr9-photometry/fuji/v2.0/observed-targets'

def read_tractorphot(zcat, verbose=False):
    from glob import glob
    from desimodel.footprint import radec2pix

    tractorphotfiles = glob(os.path.join(vacdir, 'tractorphot', 'tractorphot-nside4-hp???-iron.fits'))
    
    hdr = fitsio.read_header(tractorphotfiles[0], 'TRACTORPHOT')
    tractorphot_nside = hdr['FILENSID']

    tot_len = len(zcat)
    tot_sofar = 0
    
    # pixels = radec2pix(tractorphot_nside, zcat['TARGET_RA'], zcat['TARGET_DEC'])
    pixels = radec2pix(tractorphot_nside, zcat['RA'], zcat['DEC'])
    
    phot = []
    for pixel in set(pixels):
        J = pixel == pixels
        photfile = os.path.join(vacdir, 'tractorphot', 'tractorphot-nside4-hp{:03d}-iron.fits'.format(pixel))
        targetids = fitsio.read(photfile, columns='TARGETID')
        K = np.where(np.isin(targetids, zcat['TARGETID'][J]))[0]
        
        tot_sofar += len(K)
        
        if verbose:
            # print('Reading photometry for {} objects from {}'.format(len(K), photfile))
            print("%.2f percent done!"%(100*tot_sofar/tot_len))
         
        _phot = fitsio.read(photfile, rows=K)
        phot.append(Table(_phot))
    phot = vstack(phot)

    # Is there a better way to sort here??
    # srt = np.hstack([np.where(tid == phot['TARGETID'])[0] for tid in trange(zcat['TARGETID'])]) 
    # phot = phot[srt]
    ## I will just match by RA, Dec later
    
    return phot


def cross_match_with_cigale(elg_cat):
    cigale_cat = Table.read("/global/cfs/cdirs/desi/science/gqp/stellar_masses_sed_cigale_agn/iron/v1.2/IronPhysProp_v1.2.fits")

    idx, d2d, d3d = match_c_to_catalog(c_cat = elg_cat, catalog_cat =cigale_cat, c_ra = "RA", c_dec = "DEC", catalog_ra = "RA", catalog_dec = "DEC")

    cigale_elg = cigale_cat[idx]
    cigale_elg = cigale_elg[ d2d.arcsec < 1]
    print(len(cigale_elg))

    elg_cat_f = elg_cat[ d2d.arcsec < 1]
    print(len(elg_cat_f), len(elg_cat))

    elg_cat_f["LOGM_CIGALE"] = cigale_elg["LOGM"]
    elg_cat_f["LOGM_ERR_CIGALE"] = cigale_elg["LOGM_ERR"]
    elg_cat_f["AGNFRAC_CIGALE"] = cigale_elg["AGNFRAC"]
    elg_cat_f["LOGSFR_CIGALE"] = cigale_elg["LOGSFR"]
    elg_cat_f["LOGSFR_ERR_CIGALE"] = cigale_elg["LOGSFR_ERR"]
    elg_cat_f["FLAGINFRARED_CIGALE"] = cigale_elg["FLAGINFRARED"]
    
    return elg_cat_f



def get_final_catalogs(vac_data_lowz, zpix_lowz, plot=False):
    '''
    In this function, we cross-match the inputted fastspec catalog with the tractor catalogs to obtain other photometric information
    
    The final catalog is returned and then can be saved
    '''
    
    print(len(vac_data_lowz["TARGETID"]), len(np.unique(vac_data_lowz["TARGETID"])))
    
    vac_gmag_lowz, vac_gmag_err_lowz = get_mags(vac_data_lowz, band="G")
    vac_rmag_lowz, vac_rmag_err_lowz = get_mags(vac_data_lowz, band="R")
    vac_zmag_lowz, vac_zmag_err_lowz = get_mags(vac_data_lowz, band="Z")
    vac_w1mag_lowz, vac_w1mag_err_lowz = get_mags(vac_data_lowz, band="W1")
    vac_w2mag_lowz, vac_w2mag_err_lowz = get_mags(vac_data_lowz, band="W2")

    vac_fibrmag_lowz = get_fibmag(vac_data_lowz)
    
    #convert the fits rec file to an Astropy Table
    vac_data_lowz_table = Table(vac_data_lowz)

    vac_data_lowz_table["MAG_G"] = vac_gmag_lowz
    vac_data_lowz_table["MAG_R"] = vac_rmag_lowz
    vac_data_lowz_table["MAG_Z"] = vac_zmag_lowz
    vac_data_lowz_table["MAG_W1"] = vac_w1mag_lowz
    vac_data_lowz_table["MAG_W2"] = vac_w2mag_lowz

    vac_data_lowz_table["MAG_G_ERR"] = vac_gmag_err_lowz
    vac_data_lowz_table["MAG_R_ERR"] = vac_rmag_err_lowz
    vac_data_lowz_table["MAG_Z_ERR"] = vac_zmag_err_lowz
    vac_data_lowz_table["MAG_W1_ERR"] = vac_w1mag_err_lowz
    vac_data_lowz_table["MAG_W2_ERR"] = vac_w2mag_err_lowz
    
    #matching the two catalogs now so we have a 1-1 comparison
    from astropy.coordinates import SkyCoord
    import astropy.units as u
    c = SkyCoord(ra=np.array(zpix_lowz["RA"])*u.degree, dec=np.array(zpix_lowz["DEC"])*u.degree)
    catalog = SkyCoord(ra=np.array(vac_data_lowz_table["RA"])*u.degree, dec=np.array(vac_data_lowz_table["DEC"])*u.degree)
    idx, d2d, d3d = c.match_to_catalog_sky(catalog)
    
    print("Maximum distance between two catalogs in arcsec = ",np.max(d2d.arcsec))
    print("What fraction of objects we find photometry for in tractor catalog = ",len(zpix_lowz)/len(vac_data_lowz_table) )
    
    #only working with the matched catalog now 
    vac_data_lowz_table = vac_data_lowz_table[idx]
    
    if plot:
        #we make a comparison plot in the photometry here!!
        zpix_rmags = 22.5 - 2.5*np.log10(zpix_lowz["FLUX_R"]/zpix_lowz["MW_TRANSMISSION_R"])
        plt.figure(figsize = (5,5))
        plt.scatter(zpix_rmags, vac_data_lowz_table["MAG_R"])
        plt.xlim([18,23])
        plt.ylim([18,23])
        plt.plot([18,23],[18,23],color = "k",lw = 1)
        plt.show()
        
        
    #computing some other useful quantities
    e_abs = np.hypot(zpix_lowz["SHAPE_E1"], zpix_lowz["SHAPE_E2"])
    all_ba = (1 - e_abs) / (1 + e_abs)
    all_phi = np.rad2deg(np.arctan2(zpix_lowz["SHAPE_E2"],zpix_lowz["SHAPE_E1"]) * 0.5)

    all_ns = zpix_lowz["SERSIC"].data
    all_ns_ivar = zpix_lowz["SERSIC_IVAR"].data
    
    ### we need to add SHAPE_R parameters ... 
    shaper_lowz = np.array(zpix_lowz["SHAPE_R"])
    shaper_err_lowz = np.sqrt(1/np.array(zpix_lowz["SHAPE_R_IVAR"]))

    magr_lowz = np.array(vac_data_lowz_table["MAG_R"])
    magr_err_lowz = np.array(vac_data_lowz_table["MAG_R_ERR"])

    mu_r = magr_lowz + 2.5*np.log10(2*np.pi*(shaper_lowz)**2)
    mu_r_err = np.sqrt( magr_err_lowz**2 + (2.171*(shaper_err_lowz/shaper_lowz))**2 )

    ## from the tractor catalog get other useful things

    vac_data_lowz_table["OBJID"] = zpix_lowz["OBJID"]
    vac_data_lowz_table["BRICKNAME"] = zpix_lowz["BRICKNAME"]

    
    for bi in "GRZ":
        vac_data_lowz_table[f"SIGMA_{bi}"] = zpix_lowz[f"FLUX_{bi}"] * np.sqrt(zpix_lowz[f"FLUX_IVAR_{bi}"])
        vac_data_lowz_table[f"FRACFLUX_{bi}"] = zpix_lowz[f"FRACFLUX_{bi}"]
        vac_data_lowz_table[f"RCHISQ_{bi}"] = zpix_lowz[f"RCHISQ_{bi}"]
        vac_data_lowz_table[f"SIGMA_GOOD_{bi}"] = np.where(vac_data_lowz_table[f"RCHISQ_{bi}"] < 100, vac_data_lowz_table[f"SIGMA_{bi}"], 0.0)
    
    vac_data_lowz_table["SHAPE_R"] = shaper_lowz
    vac_data_lowz_table["SHAPE_R_ERR"] = shaper_err_lowz

    vac_data_lowz_table["MU_R"] = mu_r
    vac_data_lowz_table["MU_R_ERR"] = mu_r_err

    vac_data_lowz_table["SERSIC"] = all_ns
    vac_data_lowz_table["SERSIC_IVAR"] = all_ns_ivar

    vac_data_lowz_table["BA"] = all_ba
    vac_data_lowz_table["TYPE"] = zpix_lowz["TYPE"]

    vac_data_lowz_table["PHI"] = all_phi
    vac_data_lowz_table["FIBERMAG_R"] = 22.5 - 2.5*np.log10(vac_data_lowz_table["FIBERFLUX_R"])

    vac_data_lowz_table["MASKBITS"] = zpix_lowz["MASKBITS"]

    
    return vac_data_lowz_table, zpix_lowz
    


def is_sga_shred(input_data):
    '''
    This function returns bool indicating if it is an SGA shred or not
    '''
    ## find the closet SGA galaxy in the sky
    ## first find all similar galaxies in redshift

    multiplier=2.0

    index = input_data["index"]
    source_ra = input_data["ra"]
    source_dec = input_data["dec"]
    source_redshift = input_data["redshift"]
    siena20_f = input_data["siena_cat"]

    # siena20_f = siena20[ (np.abs(source_redshift - siena20["Z_LEDA"]) * c_light < 1000 ) ]
    
    #the columns we need are RA/DEC_MOMENT , D26, BA, PA, Z_LEDA and nothing else!
        
    if len(siena20_f) > 0:
        #find nearby ones in sky
        c = SkyCoord(ra= source_ra * u.degree, dec= source_dec *u.degree )
        catalog = SkyCoord(ra=siena20_f["RA_MOMENT"].data*u.degree, dec=siena20_f["DEC_MOMENT"].data*u.degree )
        d2d = c.separation(catalog)
        siena20_closet = siena20_f[ np.argmin(d2d.arcsec) ]

        #I need to determine if it is within twice the radius of this galaxy ... 
        norm_dist = calc_normalized_dist(source_ra, source_dec, siena20_closet["RA_MOMENT"], siena20_closet["DEC_MOMENT"],siena20_closet["D26"]*60*0.5, 
                             cen_ba=siena20_closet["BA"], cen_phi=siena20_closet["PA"], multiplier=multiplier)
        
        # if norm_dist <= 1:
        return index,norm_dist
        # else:
        #     return False
    else:
        return index,-99

        
if __name__ == '__main__':
     
    rootdir = '/global/u1/v/virajvm/'
    sys.path.append(os.path.join(rootdir, 'DESI2_LOWZ'))
    from desi_lowz_funcs import save_table, get_useful_cat_colms, _n_or_more_gt, _n_or_more_lt, get_remove_flag
    from desi_lowz_funcs import match_c_to_catalog, get_stellar_mass, get_stellar_mass_mia, calc_normalized_dist
    from desi_lowz_funcs import get_sweep_filename, save_table, is_target_in_south

    c_light = 299792 #km/s
    
    iron_vac = fits.open("/global/cfs/cdirs/desi/public/dr1/vac/dr1/fastspecfit/iron/v2.1/catalogs/fastspec-iron.fits")
    vac_data_good = iron_vac[2].data
    vac_data_other = iron_vac[1].data

    save_folder = "/pscratch/sd/v/virajvm/catalog_dr1_dwarfs"
    #apply only maskbit+sigma cleaning or also fracflux cleaning?
    apply_only_maskbit = True
    #should I filter for successful redshifts?
    apply_zsucc_cut = True
    #should I apply some redshift cut to make sure I am dealing with potential dwarf objects only?
    apply_zred_cut = True
    cross_match_w_cigale = True
    get_color_mstar = True
    #we will remove objects that are within twice the half-light radius of SGA galaxies


    
    zred_cuts = [0.2, 0.3, 0.3, 0.5]
    #either BGS_BRIGHT, ELG or BGS_FAINT
    gal_types = ["BGS_BRIGHT", "BGS_FAINT", "LOWZ", "ELG"]
    
    save_filenames = ["iron_bgs_bright_filter_zsucc_zrr02_allfracflux.fits","iron_bgs_faint_filter_zsucc_zrr03_allfracflux.fits","iron_lowz_filter_zsucc_zrr03.fits" "iron_elg_filter_zsucc_zrr05_allfracflux.fits"]
    
    comment = ""
    # "This is the BGS Bright catalog with Z_RR < 0.2 with successful redshifts! Also includes more columns from CIGALE and color-based stellar mass. No cuts on fracflux and rchisq are made."

    #looping over all the sub-samples!
    for i,gal_type in enumerate(gal_types):
        zred_cut = zred_cuts[i]
        save_filename = save_filenames[i]
        
        if gal_type == "LOWZ":
            #these catalogs are already cleaned by design so no need for more filtering
            vac_data_cat_f = Table.read(save_folder + "/iron_lowz_dark_all_phot_final_fixed.fits")
            ## also I have confirmed that the mag_grz and flux_grz are corrected for MW extinction
            
            zpix_cat = read_tractorphot(vac_data_cat_f, verbose=True)
    
            #if we are doing lowz, we should match with the low-z photometry target catalog I have
            vac_data_cat_f, zpix_cat = get_final_catalogs(vac_data_cat_f, zpix_cat, plot=False)
    
            print("Maximum RA difference is =", np.max(np.abs(zpix_cat["RA"] - vac_data_cat_f["RA"] ) ) )
            
            ##The magnitudes/fluxes in this catalog are already extinction corrected
            ##to confirm, this check the read_ls_lowz_catalogs.py script to see how the lowz target list files are constructed
            ## you can also look at the construct dwarf galaxy notebook to see that even the FLUX columns are extinction corrected
            ## as the flux columns were not corrected in the original target catalog :)
            
        else:
            if gal_type == "BGS_BRIGHT":
                vac_data_bgs_tgid = vac_data_good["BGS_TARGET"]
                gal_mask = (vac_data_bgs_tgid & bgs_mask["BGS_BRIGHT"]) != 0
            
            if gal_type == "BGS_FAINT":
                vac_data_bgs_tgid = vac_data_good["BGS_TARGET"]
                gal_mask = (vac_data_bgs_tgid & bgs_mask["BGS_FAINT"]) !=  0
        
            if gal_type == "ELG":
                desi_tgt = vac_data_good["DESI_TARGET"]
                gal_mask = (desi_tgt & desi_mask["ELG"]) != 0
               
            ## EXTRACTING THE DATA 
            vac_data_cat = vac_data_good[gal_mask]
            
            allmask_grz = [f"ALLMASK_{b}" for b in "GRZ"]
            sigma_grz = [f"SIGMA_GOOD_{b}" for b in "GRZ"]
            sigma_wise = [f"SIGMA_GOOD_W{b}" for b in range(1, 5)]
            fracflux_grz = [f"FRACFLUX_{b}" for b in "GRZ"]
            rchisq_grz = [f"RCHISQ_{b}" for b in "GRZ"]
            fracmasked_grz = [f"FRACMASKED_{b}" for b in "GRZ"]
        
            #number of bands we need 5 sigma detection in
            nsigma_bands = 2
            if gal_type == "ELG":
                nsigma_bands = 3
        
            ##apply the filtering now
            if apply_only_maskbit:
                remove_queries = [
                "(MASKBITS >> 1) % 2 > 0",  # 1
                "(MASKBITS >> 5) % 2 > 0",  # 2
                "(MASKBITS >> 6) % 2 > 0",  # 3
                "(MASKBITS >> 7) % 2 > 0",  # 4
                "(MASKBITS >> 12) % 2 > 0",  # 5
                "(MASKBITS >> 13) % 2 > 0",  # 6
                _n_or_more_lt(sigma_grz, nsigma_bands, 5),  # 7
                ]
            else:
                remove_queries = [
                "(MASKBITS >> 1) % 2 > 0",  # 1
                "(MASKBITS >> 5) % 2 > 0",  # 2
                "(MASKBITS >> 6) % 2 > 0",  # 3
                "(MASKBITS >> 7) % 2 > 0",  # 4
                "(MASKBITS >> 12) % 2 > 0",  # 5
                "(MASKBITS >> 13) % 2 > 0",  # 6
                _n_or_more_lt(sigma_grz, nsigma_bands, 5),  # 7
                Query(_n_or_more_gt(fracflux_grz, 3, 0.7)),  # 8
                ]
                
                # "FRACFLUX_G > 0.7",
                # "FRACFLUX_R > 0.7",
                # "FRACFLUX_Z > 0.7"
                
            ## read tractor phot
            zpix_cat = read_tractorphot(vac_data_cat, verbose=True)
            zpix_cat = get_useful_cat_colms(zpix_cat)
        
            #do some of the processing on catalo
            vac_data_cat, zpix_cat = get_final_catalogs(vac_data_cat, zpix_cat, plot=False)
        
            print("Maximum RA difference is =", np.max(np.abs(zpix_cat["RA"] - vac_data_cat["RA"] ) ) )
            
            mask = get_remove_flag(zpix_cat, remove_queries) == 0
            mask2 = (zpix_cat["is_galaxy"] == True)
            tot_mask = mask&mask2
            ### this is the mask that will keep good objects and remove bad
            print("Fraction that pass cleaning cuts: ", np.sum(tot_mask)/len(tot_mask) )
            
            vac_data_cat_f = vac_data_cat[tot_mask]
        
        
        print(len(vac_data_cat_f))
     
        if apply_zsucc_cut:
            ## note for ELGs, the spectroscopic cleaning cut relies on OII doublet SNR as well
            if gal_type == "ELG":
    
                #we need to match the above targetids to our target ids 
                vac_data_cat_f = vac_data_cat_f[ (vac_data_cat_f["Z"] < 1) & (vac_data_cat_f["ZWARN"] == 0)  & (vac_data_cat_f["SPECTYPE"]== "GALAXY") ]
                
                vac_data_tgids = vac_data_cat_f["TARGETID"].data
    
                oii_flux = -99 * np.ones( len(vac_data_cat_f)  )
                oii_flux_ivar = -99* np.ones( len(vac_data_cat_f)  )
                z_hpx = -99* np.ones( len(vac_data_cat_f)  )
                tgid_2 = -99* np.ones( len(vac_data_cat_f)  )
                
                all_elg_data = Table.read( "/pscratch/sd/v/virajvm/catalog/all_elg_healpix_catalogs.fits")
                    
                ## then we cross-match 
                idx, d2d,_ = match_c_to_catalog(c_cat = vac_data_cat_f, catalog_cat = all_elg_data, c_ra = "RA", c_dec = "DEC" )
    
                all_elg_data_f = all_elg_data[idx]
                all_elg_data_f = all_elg_data_f[d2d.arcsec < 1]
    
                oii_flux[d2d.arcsec < 1] = all_elg_data_f["OII_FLUX"].data
                oii_flux_ivar[d2d.arcsec < 1] = all_elg_data_f["OII_FLUX_IVAR"].data
                z_hpx[d2d.arcsec < 1] = all_elg_data_f["Z"].data
                tgid_2[d2d.arcsec < 1] = all_elg_data_f["TARGETID"].data
                
                ## using this apply the filter cut!
                vac_data_cat_f["OII_FLUX"] = oii_flux
                vac_data_cat_f["OII_FLUX_IVAR"] = oii_flux_ivar
                vac_data_cat_f["Z_HPX"] = z_hpx
                vac_data_cat_f["TARGETID_2"] = tgid_2
                
                vac_data_cat_f["OII_SNR"] = vac_data_cat_f["OII_FLUX"] * np.sqrt(vac_data_cat_f["OII_FLUX_IVAR"])
                vac_data_cat_f["ZSUCC"] = (( 0.2 * np.log10(vac_data_cat_f["DELTACHI2"]) + np.log10(vac_data_cat_f["OII_SNR"]) ) > 0.9 )  
    
                save_table(vac_data_cat_f,  save_folder + "/" + save_filename,comment=comment)
                
                #applying the cleaning cuts!
                vac_data_cat_f = vac_data_cat_f[(vac_data_cat_f["ZSUCC"] == 1)]
    
                
            else:
                ## if not ELGs, but BGS etc.
                good_mask =(vac_data_cat_f["ZWARN"] == 0) & (vac_data_cat_f["SPECTYPE"] == "GALAXY") & (vac_data_cat_f["DELTACHI2"] > 40) & (vac_data_cat_f["Z"] < 0.5)
                vac_data_cat_f = vac_data_cat_f[good_mask]
    
        if apply_zred_cut:
            vac_data_cat_f = vac_data_cat_f[ (vac_data_cat_f["Z"] < zred_cut)]
        
        if cross_match_w_cigale:
            vac_data_cat_f = cross_match_with_cigale(vac_data_cat_f)
    
        if get_color_mstar:
            ## these color based prescriptions only work for Z < 0.5 galaxies though
            gr_colors = vac_data_cat_f["MAG_G"] - vac_data_cat_f["MAG_R"]
            
            zred_mask = (vac_data_cat_f["Z"] < 0.5)
    
            #uses r band magnitude
            mstars_SAGA = get_stellar_mass(gr_colors[zred_mask].data, vac_data_cat_f["MAG_R"][zred_mask].data ,vac_data_cat_f["Z"][zred_mask].data )
            #uses g band magnitude
            mstars_M24 = get_stellar_mass_mia(gr_colors[zred_mask].data, vac_data_cat_f["MAG_G"][zred_mask].data ,vac_data_cat_f["Z"][zred_mask].data)
    
            vac_data_cat_f["LOGM_SAGA"] = -99*np.ones(len(vac_data_cat_f))
            vac_data_cat_f["LOGM_M24"] = -99*np.ones(len(vac_data_cat_f))
    
            #add the stellar masses
            vac_data_cat_f["LOGM_M24"][zred_mask] = mstars_M24
            vac_data_cat_f["LOGM_SAGA"][zred_mask] = mstars_SAGA
    
    
        ## for the final catalog, add info on the sweep file!
        vac_data_cat_f = add_sweeps_column(vac_data_cat_f)
        
        # print(len(vac_data_cat_f))
        # print(len(vac_data_cat_f[ (vac_data_cat_f["FRACFLUX_G"] > 0.35) & (vac_data_cat_f["FRACFLUX_R"] > 0.35) & (vac_data_cat_f["FRACFLUX_Z"] > 0.35) ]))
    
        ## save this catalog 
        save_table(vac_data_cat_f,  save_folder + "/" + save_filename,comment=comment)


 
#### OLD CODE


##Read all the ELG hpx files
# hpi_0 = all_healpix[0]
# elg_data_0 = Table.read(f"/global/cfs/cdirs/desi/spectro/redux/iron/healpix/main/dark/{hpi_0//100}/{hpi_0}/emline-main-dark-{hpi_0}.fits".format(hpi_0) )
# elg_data_0 = elg_data_0["TARGET_RA", "TARGET_DEC" , "TARGETID", "OII_FLUX", "OII_FLUX_IVAR","Z"]
# all_elg_data = [ elg_data_0 ]

# for i in trange(1,len(all_healpix)):
#     try:
#         hpi = all_healpix[i]
#         #read the relevant file
#         elg_data_i = Table.read(f"/global/cfs/cdirs/desi/spectro/redux/iron/healpix/main/dark/{hpi//100}/{hpi}/emline-main-dark-{hpi}.fits".format(hpi) )
#         elg_data_i = elg_data_i["TARGET_RA", "TARGET_DEC" , "TARGETID", "OII_FLUX", "OII_FLUX_IVAR","Z"]
#         all_elg_data.append(elg_data_i)
#     except:
#         print(hpi, "this healpix not found")

# ## now stack all
# all_elg_data = vstack(all_elg_data)
#we can save this for future reference
# save_table(all_elg_data, "/pscratch/sd/v/virajvm/catalog/all_elg_healpix_catalogs.fits")



            ## According to Ashley Ross, the LSS catalogs contain the tile based redshift which would be slightly different than the healpix
            ##based redshift. The below file would give us the tile based redshift. 
      
    
# elg_lss = fits.open("/global/cfs/cdirs/desi/survey/catalogs/dr1/LSS.dr1/iron/LSScats/v1.5/ELG_LOPnotqso_full.dat.fits")
# elg_lss_data = elg_lss[1].data
# elg_lss_tgids = elg_lss_data["TARGETID"].data
# print(len(vac_data_tgids), len(elg_lss_data))

# Find the common values and their indices in arr1
# common_values, idx1, _ = np.intersect1d(vac_data_tgids, elg_lss_tgids, return_indices=True)

# # Find the indices of these values in arr2
# match_inds = np.nonzero(np.isin(elg_lss_tgids, common_values))[0]

# # arr1 is now filtered to only contain values in arr2
# vac_data_cat_f = vac_data_cat_f[idx1]

# print(len(match_inds), len(vac_data_cat_f))

# vac_data_cat_f["OII_FLUX"] = elg_lss_data["OII_FLUX"][match_inds]
# vac_data_cat_f["OII_FLUX_IVAR"] = elg_lss_data["OII_FLUX_IVAR"][match_inds]
# vac_data_cat_f["WEIGHT_ZFAIL"] = elg_lss_data["WEIGHT_ZFAIL"][match_inds]
# vac_data_cat_f["mod_success_rate"] = elg_lss_data["mod_success_rate"][match_inds]
# vac_data_cat_f["TARGETID_2"] = elg_lss_data["TARGETID"][match_inds]
# vac_data_cat_f["DELTACHI2_2"] = elg_lss_data["DELTACHI2"][match_inds]
# vac_data_cat_f["Z_not4clus"] = elg_lss_data["Z_not4clus"][match_inds]

# vac_data_cat_f =  vac_data_cat_f[vac_data_cat_f["OII_FLUX"] > 0]
# vac_data_cat_f["OII_SNR"] = vac_data_cat_f["OII_FLUX"] * np.sqrt(vac_data_cat_f["OII_FLUX_IVAR"])
# vac_data_cat_f["ZSUCC"] = (( 0.2 * np.log10(vac_data_cat_f["DELTACHI2"]) + np.log10(vac_data_cat_f["OII_SNR"]) ) > 0.9 )

###healpix based redshift!!
# all_healpix = np.unique(  vac_data_cat_f["HEALPIX"].data )


'''
This is the script to construct the main dwarf galaxy samples from the Iron fastspecfit catalog! Some basic cleaning cuts are applied
'''

from desitarget.targetmask import desi_mask, bgs_mask
# import some helpful python packages 
import os
import numpy as np
from tqdm import trange
from astropy.io import fits
from astropy.cosmology import Planck18
from astropy.table import Table, vstack, Column
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
from desitarget import targetmask
from desitarget.sv1 import sv1_targetmask
from desitarget.sv2 import sv2_targetmask
from desitarget.sv3 import sv3_targetmask


from easyquery import Query, QueryMaker
reduce_compare = QueryMaker.reduce_compare

from desi_lowz_funcs import add_sweeps_column, match_c_to_catalog, calc_normalized_dist, DVcalculator_list, get_stellar_mass

c_light = 299792 #in km/s


def get_sv_bgs_mask(catalog, bgs_class = "BGS_BRIGHT"):
    '''
    Get the mask to select BGS objects in SV. Note that this function is supposed to be run on the zpix catalog
    and not the fastspecfit catalog
    '''
    
    #then specifically BGS BRIGHT
    sv1_class_mask = (catalog["SV1_BGS_TARGET"] & sv1_targetmask.bgs_mask[bgs_class] != 0)
    sv2_class_mask = (catalog["SV2_BGS_TARGET"] & sv2_targetmask.bgs_mask[bgs_class] != 0)
    sv3_class_mask = (catalog["SV3_BGS_TARGET"] & sv3_targetmask.bgs_mask[bgs_class] != 0)

    
    return (sv1_class_mask) | (sv2_class_mask) | (sv3_class_mask)



def get_sv_elg_mask(catalog):
    '''
    Get the mask to select ELG objects in SV. Note that this function is supposed to be run on the zpix catalog
    and not the fastspecfit catalog
    '''
    sv1_desi_mask = sv1_targetmask.desi_mask
    sv2_desi_mask = sv2_targetmask.desi_mask
    sv3_desi_mask = sv3_targetmask.desi_mask
    
    sv1_elg_mask = (catalog["SV1_DESI_TARGET"] & sv1_desi_mask['ELG'] != 0)
    sv2_elg_mask = (catalog["SV2_DESI_TARGET"] & sv2_desi_mask['ELG'] != 0)
    sv3_elg_mask = (catalog["SV3_DESI_TARGET"] & sv3_desi_mask['ELG'] != 0)
    
    return (sv1_elg_mask) | (sv2_elg_mask) | (sv3_elg_mask)


def get_fibmag(vac_data_bgs):
    vac_fib_fluxr = vac_data_bgs["FIBERFLUX_R"]
    vac_fibrmag_bgs = 22.5 - 2.5*np.log10(vac_fib_fluxr)
    return vac_fibrmag_bgs
    
    
def get_mags(catalog, band="R",correct_mw = False):
    '''
    If using fastspecfit catalogs, note they are already corrected for MW extinction 
    https://fastspecfit.readthedocs.io/en/latest/fastspec.html
    the data model here says that the flux are already extinction corrected!!
    '''
    vac_fluxr = catalog["FLUX_" + band]
    vac_fluxr_err = np.sqrt(1/catalog["FLUX_IVAR_" + band])
    
    if correct_mw:
        vac_mwr = catalog["MW_TRANSMISSION_" + band]
        #correcting the flux for MW extinction
        vac_fluxr = vac_fluxr/vac_mwr
        #propogating the error
        vac_fluxr_err = vac_fluxr_err / vac_mwr
    else:
        pass
        
    vac_rmag = 22.5 - 2.5*np.log10(vac_fluxr)
    
    #we have the magntudes
    ### this formula for rmag uncertainties is from https://sites.astro.caltech.edu/~george/ay122/Ay122a_Photometry1.pdf pg 34
    ### it is an approximation that is valid in small error regime.
    vac_rmag_err = 1.087*(vac_fluxr_err/vac_fluxr)
    
    return vac_rmag, vac_rmag_err

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



def cross_match_with_cigale(dwarf_cat):
    '''
    Cross-matches dwarf_cat with the CIGALE VAC, keeping all dwarf_cat rows
    and adding CIGALE info where available (within 1 arcsec).
    https://data.desi.lbl.gov/doc/releases/dr1/vac/stellar-mass-emline/

    
    cigale_cat = fits.open("/global/cfs/cdirs/desi/public/dr1/vac/dr1/stellar-mass-emline/v1.0/dr1_galaxy_stellarmass_lineinfo_v1.0.fits")
    cigale_data = cigale_cat[1].data
    
    cg_keys = [
        "TARGET_RA", "TARGET_DEC", "PROGRAM", "ZCAT_PRIMARY",
        "HALPHA_FLUX", "HALPHA_FLUXERR", "HALPHA_EW", "HALPHA_EWERR",
        "MASS_CG", "MASSERR_CG", "Z"]
    
    cigale_dict = {  }
    for keyi in cg_keys:
        print(keyi)
        cigale_dict[keyi] = cigale_data[keyi]
    
    cigale_table = Table(cigale_dict)
    
    cigale_table = cigale_table[cigale_table["Z"] < 0.5]
    
    print(len(cigale_table)) -> 7768484
    
    cigale_table.write("/pscratch/sd/v/virajvm/catalog_dr1_dwarfs/cigale_vac_emlines_info.fits",overwrite=True)
    '''

    cigale_cat = Table.read("/pscratch/sd/v/virajvm/catalog_dr1_dwarfs/cigale_vac_emlines_info.fits")
    
    # Perform coordinate match
    idx, d2d, d3d = match_c_to_catalog(
        c_cat=dwarf_cat,
        catalog_cat=cigale_cat,
        c_ra="RA",
        c_dec="DEC",
        catalog_ra="TARGET_RA",
        catalog_dec="TARGET_DEC"
    )

    # Mask for good matches within 1 arcsec
    match_mask = (d2d.arcsec < 1)

    # Define columns to transfer
    cg_keys = ["HALPHA_FLUX", "HALPHA_FLUXERR", "HALPHA_EW", "HALPHA_EWERR",
        "MASS_CG", "MASSERR_CG"
    ]

    # Add new columns initialized to NaN or appropriate fill values
    for key in cg_keys:
        if "_CG" in key:
            flag = ""
        else:
            flag = "_CG"
            
        dtype = cigale_cat[key].dtype
        if np.issubdtype(dtype, np.floating):
            dwarf_cat[key + flag] = np.full(len(dwarf_cat), np.nan)
        else:
            dwarf_cat[key + flag] = np.full(len(dwarf_cat), -999, dtype=dtype)  # fill int/str with -999

    # Fill in only matched rows
    matched_rows = np.where(match_mask)[0]
    matched_idx_in_cigale = idx[match_mask]

    for key in cg_keys:
        if "_CG" in key:
            flag = ""
        else:
            flag = "_CG"
            
        dwarf_cat[key + flag][matched_rows] = cigale_cat[key][matched_idx_in_cigale]

    print(f"Total matches within 1 arcsec: {len(matched_rows)} / {len(dwarf_cat)}")
    print(f"Fraction of matches with CIGALE = {len(matched_rows) / len(dwarf_cat):.3f}")

    return dwarf_cat


def cross_match_with_fastspec(catalog,sample_name):
    '''
    We cross-match here to make sure the photometric information is correct
    '''

    iron_vac = fits.open("/global/cfs/cdirs/desi/public/dr1/vac/dr1/fastspecfit/iron/v2.1/catalogs/fastspec-iron.fits")
    vac_data_good = iron_vac[2].data
    vac_data_good = Table(vac_data_good)

    idx,d2d,_ = match_c_to_catalog(c_cat = catalog, catalog_cat = vac_data_good)
    #the maximum frational different in flux_r

    vac_data_good = vac_data_good[idx]

    max_ind = np.argmax(np.abs(catalog["FLUX_R"] - vac_data_good["FLUX_R"]))

    max_diff = np.abs(catalog["FLUX_R"][max_ind] - vac_data_good["FLUX_R"][max_ind])/catalog["FLUX_R"][max_ind]
    
    print(f"{sample_name}: The maximum fractional difference in FLUX_R between fastspec and constructed catalog is = {max_diff}")

    #and we make a comparison plot
    plt.figure(figsize = (3,3))
    plt.scatter(catalog["FLUX_R"], vac_data_good["FLUX_R"],s=5)
    plt.plot([1,1e4],[1,1e4],color = "k",lw =0.5)
    plt.yscale("log")
    plt.xscale("log")
    plt.ylabel("Fastspecfit FLUX R")
    plt.xlabel("Constructed Catalog FLUX R")
    plt.xlim([1,1e4])
    plt.ylim([1,1e4])
    plt.savefig(f"/global/homes/v/virajvm/DESI2_LOWZ/quenched_fracs_nbs/paper_plots/construct_catalog_{sample_name}_fastspec_compare.png",
                bbox_inches="tight")
    plt.close()



def get_final_catalogs(zpix_cat, zpix_trac, sample_name):
    '''
    In this function, we combine the photometric information from the zpix_trac into the redshift catalog    
    '''

    #matching the two catalogs now so we have a 1-1 comparison    
    c = SkyCoord(ra=np.array(zpix_trac["RA"])*u.degree, dec=np.array(zpix_trac["DEC"])*u.degree)
    catalog = SkyCoord(ra=np.array(zpix_cat["RA"])*u.degree, dec=np.array(zpix_cat["DEC"])*u.degree)
    idx, d2d, d3d = c.match_to_catalog_sky(catalog)

    org_num = len(zpix_cat)
    
    print(f"{sample_name}: Maximum distance between two catalogs in arcsec = ",np.max(d2d.arcsec))
    print(f"{sample_name}: What fraction of objects we find photometry for in tractor catalog = ",len(zpix_trac)/len(zpix_cat) )

    #only working with the matched catalog now 
    zpix_cat = zpix_cat[idx]
    zpix_cat = zpix_cat[ d2d.arcsec < 1 ]
    zpix_trac = zpix_trac[d2d.arcsec < 1]

    print("What fraction of objects have close matches = ",len(zpix_cat)/org_num )
    
    #adding the extinction corrected magnitudes and flux to the catalog
    for bi in "GRZ":

        flux_bi = zpix_trac[f"FLUX_{bi}"]
        flux_err_bi = np.sqrt(1/zpix_trac[f"FLUX_IVAR_{bi}"])
        mw_ext = zpix_trac[f"MW_TRANSMISSION_{bi}"]
    
        #corect for
        flux_corr_bi = flux_bi / mw_ext
        flux_err_corr_bi = flux_err_bi / mw_ext

        mag_bi = 22.5 - 2.5*np.log10( flux_corr_bi )
        #error approximation
        mag_err_bi = 1.087*(flux_err_corr_bi/flux_corr_bi)
        
        zpix_cat[f"FLUX_{bi}"] =  flux_corr_bi
        zpix_cat[f"FLUX_IVAR_{bi}"] =  1/flux_err_corr_bi**2

        zpix_cat[f"MAG_{bi}"] = mag_bi
        zpix_cat[f"MAG_{bi}_ERR"] = mag_err_bi

    ##adding the not extinction corrected fiber magnitude to the catalog
    zpix_cat["FIBERMAG_R"] = 22.5 - 2.5*np.log10(zpix_cat["FIBERFLUX_R"])

    
    #confirm that the two fiber flux match!
    rfib_mag_org =  22.5 - 2.5*np.log10(zpix_trac["FIBERFLUX_R"])
    plt.figure(figsize = (4,4))
    plt.scatter(rfib_mag_org, zpix_cat["FIBERMAG_R"],s=0.1)
    plt.xlim([17,24])
    plt.ylim([17,24])
    plt.xlabel("Tractor rfib-mag")
    plt.ylabel("zpix rfib-mag")
    plt.savefig(f"/global/homes/v/virajvm/DESI2_LOWZ/quenched_fracs_nbs/paper_plots/construct_catalog_{sample_name}_rfib_compare.png",
                bbox_inches="tight")
    plt.close()
    
    #computing some other useful quantities
    e_abs = np.hypot(zpix_trac["SHAPE_E1"], zpix_trac["SHAPE_E2"])
    all_ba = (1 - e_abs) / (1 + e_abs)
    all_phi = np.rad2deg(np.arctan2(zpix_trac["SHAPE_E2"],zpix_trac["SHAPE_E1"]) * 0.5)

    all_ns = zpix_trac["SERSIC"].data
    all_ns_ivar = zpix_trac["SERSIC_IVAR"].data
    
    ### we need to add SHAPE_R parameters ... 
    shaper_lowz = np.array(zpix_trac["SHAPE_R"])
    shaper_err_lowz = np.sqrt(1/np.array(zpix_trac["SHAPE_R_IVAR"]))

    magr_lowz = np.array(zpix_cat["MAG_R"])
    magr_err_lowz = np.array(zpix_cat["MAG_R_ERR"])

    mu_r = magr_lowz + 2.5*np.log10(2*np.pi*(shaper_lowz)**2)
    mu_r_err = np.sqrt( magr_err_lowz**2 + (2.171*(shaper_err_lowz/shaper_lowz))**2 )

    ## from the tractor catalog get other useful things

    zpix_cat["OBJID"] = zpix_trac["OBJID"]
    zpix_cat["BRICKNAME"] = zpix_trac["BRICKNAME"]

    for bi in "GRZ":
        zpix_cat[f"SIGMA_{bi}"] = zpix_trac[f"FLUX_{bi}"] * np.sqrt(zpix_trac[f"FLUX_IVAR_{bi}"])
        zpix_cat[f"FRACFLUX_{bi}"] = zpix_trac[f"FRACFLUX_{bi}"]
        zpix_cat[f"RCHISQ_{bi}"] = zpix_trac[f"RCHISQ_{bi}"]
        #if the RChisq > 100, make sigma_good = 0. This is to account for bad model fits
        zpix_cat[f"SIGMA_GOOD_{bi}"] = np.where(zpix_cat[f"RCHISQ_{bi}"] < 100, zpix_cat[f"SIGMA_{bi}"], 0.0)
    
    zpix_cat["SHAPE_R"] = shaper_lowz
    zpix_cat["SHAPE_R_ERR"] = shaper_err_lowz

    zpix_cat["MU_R"] = mu_r
    zpix_cat["MU_R_ERR"] = mu_r_err

    zpix_cat["SERSIC"] = all_ns
    zpix_cat["SERSIC_IVAR"] = all_ns_ivar

    zpix_cat["BA"] = all_ba
    zpix_cat["TYPE"] = zpix_trac["TYPE"]

    zpix_cat["PHI"] = all_phi
    zpix_cat["FIBERMAG_R"] = 22.5 - 2.5*np.log10(zpix_cat["FIBERFLUX_R"])

    zpix_cat["MASKBITS"] = zpix_trac["MASKBITS"]

    zpix_cat["NOBS_G"] = zpix_trac["NOBS_G"]
    zpix_cat["NOBS_R"] = zpix_trac["NOBS_R"] 
    zpix_cat["NOBS_Z"] = zpix_trac["NOBS_Z"] 

    zpix_cat["MW_TRANSMISSION_G"] = zpix_trac["MW_TRANSMISSION_G"]
    zpix_cat["MW_TRANSMISSION_R"] = zpix_trac["MW_TRANSMISSION_R"] 
    zpix_cat["MW_TRANSMISSION_Z"] = zpix_trac["MW_TRANSMISSION_Z"] 
    
    return zpix_cat, zpix_trac

# def get_zpix_catalog(specprod = 'iron'):
#     '''
#     Function that returns the zpix catalog with some useful info
#     '''
#     # Open the fits file
    
#     all_cols = ["TARGETID", "TARGET_RA","TARGET_DEC", "ZERR", "OBSCONDITIONS", "COADD_NUMEXP", "MIN_MJD", "MAX_MJD", "MEAN_MJD", "ZCAT_PRIMARY", "DESINAME"]
    
#     temp = {}

#     specprod_dir = f'/global/cfs/cdirs/desi/spectro/redux/{specprod}/'
    
#     with fits.open(f'{specprod_dir}zcatalog/v1/zall-pix-{specprod}.fits') as hdul:
#         # Select the specific HDU, e.g., HDU index 1 or by name
#         hdu = hdul[1]  # or hdul['HDU_NAME']
    
#         # Access the data (as a FITS_rec object)
#         data = hdu.data
    
#         for coli in all_cols:
#             temp[coli] = data[coli]

#     return Table(temp)
        


def bright_star_filter(cat):
    '''
    This function adds information on nearby bright stars that can be used for filtering downstream. Function taken from John Moustakas with some modifications
    '''
    
    # add the Gaia mask bits
    print(f'Adding Gaia bright-star masking bits.')

    cat['STARFDIST'] = np.zeros(len(cat), 'f4') + 99.
    cat['STARDIST_DEG'] = np.zeros(len(cat), 'f4') + 99.
    cat['STARMAG'] = np.zeros(len(cat), 'f4') + 99.
    cat['STAR_RADIUS_ARCSEC'] = np.zeros(len(cat), 'f2') + 99.
    cat['STAR_RA'] = np.zeros(len(cat), 'f4') + 99.
    cat['STAR_DEC'] = np.zeros(len(cat), 'f4') + 99.
    
    # gaiafile = os.path.join(sga_dir(), 'gaia', 'gaia-mask-dr3-galb9.fits')
    gaiafile = "/global/cfs/cdirs/desicollab/users/ioannis/SGA/2025/gaia/gaia-mask-dr3-galb9.fits"
    
    gaia = Table(fitsio.read(gaiafile, columns=['ra', 'dec', 'radius', 'mask_mag', 'isbright', 'ismedium']))
    print(f'Read {len(gaia):,d} Gaia stars from {gaiafile}')
    I = gaia['radius'] > 0.
    print(f'Trimmed to {np.sum(I):,d}/{len(gaia):,d} stars with radius>0')
    gaia = gaia[I]

    dmag = 1.
    bright = np.min(np.floor(gaia['mask_mag']))
    faint = np.max(np.ceil(gaia['mask_mag']))
    magbins = np.arange(bright, faint, dmag)

    for mag in magbins:
        # find all Gaia stars in this magnitude bin
        I = np.where((gaia['mask_mag'] >= mag) * (gaia['mask_mag'] < mag+dmag))[0]
    
        # search within 2 times the largest masking radius to be efficient searching!
        maxradius = 2. * np.max(gaia['radius'][I]) # [degrees]
        print(f'Found {len(I):,d} Gaia stars in magnitude bin {mag:.0f} to ' + \
              f'{mag+dmag:.0f} with max radius {maxradius:.4f} degrees.')

        #this is similar to the skycoord match function
        #for every source it is finding the closest star with the search radius being twice the max radius of any star being search
        # m1, m2, sep = match_radec(cat['RA'], cat['DEC'], gaia['ra'][I], gaia['dec'][I], maxradius, nearest=True)
         # m1: indices into the "ra1,dec1" arrays of matching points.
        # m2: same, but for "ra2,dec2".
        # sep: distance, in degrees, between the matching points.

        idx, d2d, _ = match_c_to_catalog(c_cat = cat, catalog_cat = gaia[I], c_ra = "RA", c_dec = "DEC", catalog_ra = "ra", catalog_dec = "dec")
        # idx are indices into catalog that are the closest objects to each of the coordinates in c,

        #this is the matched gaia catalog in this magnitude bin that we will work it
        gaia_match = gaia[I][idx]

        ##for each source, we need to compute the stardist, starfdist, and starmag
        ##if the source is beyond twice the largest masking radius in this bin, then we do not want to update it

        #these are the seperations in arcsecs
        all_stardists = d2d.deg
        #these are the separations normalized to the star radius
        all_starfdists = d2d.deg / gaia_match['radius'].data
        #these are the stellar mags
        all_starmag = gaia_match['mask_mag'].data

        #getting the radius of the star in arcsecs
        all_star_radii = gaia_match['radius'].data * 3600
        all_star_ras = gaia_match['ra'].data 
        all_star_decs = gaia_match['dec'].data 
        

        #however, we will only update the locations where the all_stardists < maxradius
        update_mask = (all_stardists < maxradius)
        #also only update if the new match is *closer* than previous one
        closer_match = (all_starfdists < cat["STARFDIST"])
        update_mask &= closer_match

        if np.sum(update_mask) > 0:
            cat["STARDIST_DEG"][update_mask] = all_stardists[update_mask] 
            cat["STARFDIST"][update_mask] = all_starfdists[update_mask]
            cat["STARMAG"][update_mask] = all_starmag[update_mask]
            cat["STAR_RADIUS_ARCSEC"][update_mask] =  all_star_radii[update_mask]
            cat["STAR_RA"][update_mask] =  all_star_ras[update_mask]
            cat["STAR_DEC"][update_mask] =  all_star_decs[update_mask]
            
        # if len(m1) > 0:
        #     zero = np.where(sep == 0.)[0]
        #     if len(zero) > 0:
        #         cat['STARDIST_DEG'][m1[zero]] = 0.
        #         cat['STARFDIST'][m1[zero]] = 0.
        #         cat['STARMAG'][m1[zero]] = gaia['mask_mag'][I[m2[zero]]]
    
        #     # separations can be identically zero
        #     pos = np.where(sep > 0.)[0]
        #     if len(pos) > 0:
        #         # distance to the nearest star (in this mag bin)
        #         # relative to the mask radius of that star (given its
        #         # mag), capped at a factor of 2; values <1 mean the
        #         # object is within the star's masking radius
        #         fdist = sep[pos] / gaia['radius'][I[m2[pos]]].value
        #         # only store the smallest value
        #         J = np.where((fdist < 2.) * (fdist < cat['STARFDIST'][m1[pos]]))[0]
        #         if len(J) > 0:
        #             cat['STARDIST'][m1[pos[J]]] = sep[pos[J]] # [degrees]
        #             cat['STARFDIST'][m1[pos[J]]] = fdist[J]
        #             cat['STARMAG'][m1[pos[J]]] = gaia['mask_mag'][I[m2[pos[J]]]]
    
    return cat

def produce_elg_catalog(zpix_elg):
    '''
    This function produces the ELG catalog with the useful OII_FLUX values
    '''

    all_healpix = np.unique( zpix_elg["HEALPIX"].data )
    
    all_elg_data = []

    for survey in ["main","sv1","sv2","sv3"]:
    
        for i in trange(len(all_healpix)):
            hpi = all_healpix[i]
            #read the relevant file
            file_path = f"/global/cfs/cdirs/desi/spectro/redux/iron/healpix/{survey}/dark/{hpi//100}/{hpi}/emline-{survey}-dark-{hpi}.fits".format(hpi)
            if os.path.exists(file_path):
                elg_data_i = Table.read(file_path )
                elg_data_i = elg_data_i["TARGET_RA", "TARGET_DEC" , "TARGETID", "OII_FLUX", "OII_FLUX_IVAR","Z"]
                all_elg_data.append(elg_data_i)
            else:
                print(hpi, f"this healpix not found in {survey}")
    
    ## now stack all
    all_elg_data = vstack(all_elg_data)
    # we can save this for future reference
    save_table(all_elg_data, "/pscratch/sd/v/virajvm/catalog/all_elg_healpix_catalogs.fits")

    return



## the below are functions from Elise that she used to combine the north and south catalogs 

def remove_south_lowz(data):
    '''
    This is applied to the north data set
    '''
    c = SkyCoord(ra=data['RA'].data*u.degree, dec=data['DEC'].data*u.degree, frame='icrs')
    cg = c.galactic 
    mask1 = cg.b.value > 0     
    mask2 = data['DEC'] > 32.375 
    mask3 = np.abs(cg.b.value) > 15. 
    return data[mask1&mask2&mask3]

def clean_south_lowz(data):
    '''
    This is applied to the southern data set
    '''
    c = SkyCoord(ra=data['RA'].data*u.degree, dec=data['DEC'].data*u.degree, frame='icrs')
    cg = c.galactic 
    # mask1 = data['DEC'] > -20 #this was originally -18
    mask2 = np.abs(cg.b.value) > 15 
    # mask1 = mask1&mask2
    # mask1 = mask1&mask2
    mask2 = data['DEC'] < 32.375 
    mask3 = cg.b.value < 0 
    mask2 = mask2 | mask3 
    # return data[mask1&mask2]
    return data[mask2]
    
def get_lowz_catalogs(zpix):
    '''
    This function returns the LOWZ redshift and photometry catalogs
    '''

    print("Getting the LOWZ catalogs!")
    
    save_zpix_path =  "/pscratch/sd/v/virajvm/catalog/lowz_dark_zpix_iron.fits"
    save_tracphot_path = "/pscratch/sd/v/virajvm/catalog/lowz_dark_tracphot_iron.fits"

    # if os.path.exists(save_zpix_path) and os.path.exists(save_tracphot_path):
    #     zpix_lowz_f = Table.read(save_zpix_path)
    #     zpix_trac_lowz_f = Table.read( save_tracphot_path )
        
    # else:
    #get the redshift catalog for LOWZ
    zpix = zpix[zpix["PROGRAM"] == "dark"]

    #get the lowz survey mask
    lowz_main_mask = (zpix["SCND_TARGET"] == 2**15) | (zpix["SCND_TARGET"] == 2**16) | (zpix["SCND_TARGET"] == 2**17)
    lowz_sv_mask = np.zeros(len(zpix)).astype(bool)
    for svi in range(1,4):
        svi_mask = (zpix[f"SV{svi}_SCND_TARGET"] == 2**15) | (zpix[f"SV{svi}_SCND_TARGET"] == 2**16) | (zpix[f"SV{svi}_SCND_TARGET"] == 2**17)
        lowz_sv_mask |= svi_mask

    lowz_gal_mask = lowz_main_mask | lowz_sv_mask

    zpix_lowz = zpix[lowz_gal_mask]

    print("Reading the LOWZ photometry catalogs!")
    print(f"LOWZ: Number of objects = {len(zpix_lowz)}")

    #get the total target catalogs for lowz!
    north_targs = Table.read("/pscratch/sd/v/virajvm/target/dr9_north_lowz_targets_no_rfib_cut.fits")
    south_targs = Table.read("/pscratch/sd/v/virajvm/target/dr9_south_lowz_targets_no_rfib_cut_dec20.fits")
    
    elise_north_tgts = remove_south_lowz(north_targs)
    elise_south_tgts = clean_south_lowz(south_targs)
    
    total_targs = vstack([elise_north_tgts, elise_south_tgts])

    idx,d2d,d3d = match_c_to_catalog(c_cat = zpix_lowz, catalog_cat = total_targs)

    #due to a mistake in how the target lists were constructed, 
    #some targets needed to be explicitly matched with a catalog 

    zpix_lowz_nomatch = zpix_lowz[(d2d.arcsec > 1)]
    zpix_lowz_match = zpix_lowz[d2d.arcsec <= 1]

    print(f"In initial matching, {len(zpix_lowz_match)} objects at <1'' and {len(zpix_lowz_nomatch)} objects at >1'' ")

    total_targs_match = total_targs[idx]
    total_targs_match = total_targs_match[d2d.arcsec <= 1]

    ##the ones that do not match, we match with the other catalog
    ##this will yield some matches because there are a specific choice made in how the north and south overlapping regions need to be handled
    idx,d2d,d3d = match_c_to_catalog(c_cat = zpix_lowz_nomatch, catalog_cat = south_targs)
    
    stargs_nomatch = south_targs[idx]
    stargs_nomatch = stargs_nomatch[d2d.arcsec < 1]
    zpix_lowz_nomatch = zpix_lowz_nomatch[d2d.arcsec < 1]
    
    print(f"After matching with south catalog, {len(zpix_lowz_nomatch)} matches found at <1''")

    #construct the final catalogs!!
    zpix_lowz_f = vstack([zpix_lowz_match, zpix_lowz_nomatch])
    zpix_trac_lowz_f = vstack([total_targs_match, stargs_nomatch])

    ##let us save these
    save_table(zpix_lowz_f, "/pscratch/sd/v/virajvm/catalog/lowz_dark_zpix_iron.fits")
    save_table(zpix_trac_lowz_f, "/pscratch/sd/v/virajvm/catalog/lowz_dark_tracphot_iron.fits")

    print(f"LOWZ: Number of objects after matching = {len(zpix_lowz_f)}")
    print(f"LOWZ: Number of trac objects after matching = {len(zpix_trac_lowz_f)}")
    
    return zpix_lowz_f, zpix_trac_lowz_f




def get_corrected_distances(final_cat_i, chunk_size=500,calc_type = "NAM"):
    '''
    Function that computes the expected distances for objects given their RA, DEC and observed line of sight velocities!
    We will only do this for objects within 2850 km/s line of sight velocities.

    More info here: http://edd.ifa.hawaii.edu/
    '''
    ra_vals = final_cat_i["RA"].data
    dec_vals = final_cat_i["DEC"].data
    zred_vals = final_cat_i["Z"].data

    all_fiducial_distances = []
    all_extra_distances = []

    n_total = len(ra_vals)
    n_chunks = (n_total + chunk_size - 1) // chunk_size  # ceil division

    for i in trange(n_chunks):
        start = i * chunk_size
        end = min((i + 1) * chunk_size, n_total)

        print(f"Start:End = {start}:{end}")

        ra_chunk = ra_vals[start:end]
        dec_chunk = dec_vals[start:end]
        z_chunk = zred_vals[start:end]

        outputs = DVcalculator_list(ra_chunk, dec_chunk, system='equatorial',
                                    parameter='velocity', values=z_chunk * c_light,
                                    calculator=calc_type)
        results = outputs["results"]

        for res in results:
            if calc_type == "CF3":
                dvals = res["adjusted"]["distance"]
            else:
                dvals = res["distance"]
            
            if len(dvals) > 1:
                sorted_dvals = np.sort(np.array(dvals))
                # mid_idx = len(sorted_dvals) // 2
                fiducial = np.mean(sorted_dvals) #[mid_idx]
                extras = sorted_dvals
            elif len(dvals) == 1:
                fiducial = dvals[0]
                extras = []
            else:
                fiducial = -99
                extras = []
                #if no distances are found, we will just use the redshift distrance!

            all_fiducial_distances.append(fiducial)
            all_extra_distances.append(extras)

    return all_fiducial_distances, all_extra_distances



def format_extra_dists_as_string(extra_dists):
    return [",".join(f"{d:.2f}" for d in dlist) for dlist in extra_dists]
    
def process_sga_matches(desi_sga_cat):
    '''
    In this function, we sub-select the objects that are dwarf galaxies and in SGA.
    '''

    #load in the siena cat
    siena_cat = Table.read("/global/cfs/cdirs/cosmo/data/sga/2020/SGA-2020.fits")

    #Match SGA to Siena catalog
    idx, d2d, _ = match_c_to_catalog(c_cat=desi_sga_cat, catalog_cat=siena_cat)
    siena_matched = siena_cat[idx]
    print("Total matched with SGA:", len(siena_matched))
    
    #we want to add the relevant columns to the DESI catalog actually!
    desi_sga_cat["SGA_RA_MOMENT"] = siena_matched["RA_MOMENT"]
    desi_sga_cat["SGA_DEC_MOMENT"] = siena_matched["DEC_MOMENT"]
    desi_sga_cat["SGA_SMA_SB26"] = siena_matched["SMA_SB26"]
    desi_sga_cat["SGA_SMA_SB25"] = siena_matched["SMA_SB25"]
    desi_sga_cat["SGA_BA"] = siena_matched["BA"]
    desi_sga_cat["SGA_PA"] = siena_matched["PA"]
    desi_sga_cat["SGA_R_COG_MAG"] = siena_matched["R_COG_PARAMS_MTOT"]
    desi_sga_cat["SGA_G_COG_MAG"] = siena_matched["G_COG_PARAMS_MTOT"]
    desi_sga_cat["SGA_Z_COG_MAG"] = siena_matched["Z_COG_PARAMS_MTOT"]
    desi_sga_cat["SGA_ZRED_LEDA"] = siena_matched["Z_LEDA"]
    desi_sga_cat["SGA_ID"] = siena_matched["SGA_ID"]

    
    #Compute normalized distances
    #the semi-major axis is needed in arcsec and so not muptipled by 60 here
    all_norm_dist = calc_normalized_dist(
        desi_sga_cat["RA"].data,
        desi_sga_cat["DEC"].data,
        desi_sga_cat["SGA_RA_MOMENT"].data,
        desi_sga_cat["SGA_DEC_MOMENT"].data,
        desi_sga_cat["SGA_SMA_SB26"].data,
        cen_ba=desi_sga_cat["SGA_BA"].data,
        cen_phi=desi_sga_cat["SGA_PA"].data,
        multiplier=2,
    )
    print("All normalized distances:", len(all_norm_dist))
    
    #Build masks

    #we want to make sure the SGA catalog always has the photometry, but can relax this afterwards
    good_photo_mask = (
        (desi_sga_cat["SGA_G_COG_MAG"] > 0) &
        (desi_sga_cat["SGA_R_COG_MAG"] > 0)
    )
    
    #this is just to make sure
    inside_galaxy_mask = (all_norm_dist <= 1)
    final_mask =  good_photo_mask & inside_galaxy_mask 
    
    #Apply final mask to both catalogs
    desi_matched_clean = desi_sga_cat[final_mask]
    print("After photo & dist cuts:", len(desi_matched_clean))
    
    #get the flow corrected distances, only for objeccts with velo < 2850 km/s
    #everything above that we will assume it is the same as teh redshift 

    nam_cutoff = 2850/c_light #in km/s
    siena_nam = desi_matched_clean[(desi_matched_clean["Z"] <= nam_cutoff)]
    #get distances for NAM
    fidu_dists,extra_dists = get_corrected_distances(siena_nam, chunk_size=500,calc_type = "NAM")
    siena_nam["DIST_MPC"] = fidu_dists
    extra_dists_str = format_extra_dists_as_string(extra_dists)
    siena_nam["OTHER_DIST_MPC"] = extra_dists_str

    #working with far away catalog
    siena_far = desi_matched_clean[(desi_matched_clean["Z"] > nam_cutoff)]
    siena_far["DIST_MPC"] = Planck18.luminosity_distance(siena_far["Z"].data).value
    # Create a list of empty lists
    blank_extra_dists = [[] for _ in range(len(siena_far))]
    blank_extra_dists_str = format_extra_dists_as_string(blank_extra_dists)
    siena_far["OTHER_DIST_MPC"] = blank_extra_dists_str

    #now we stack these two!!
    desi_cat_f = vstack([siena_far, siena_nam])

    #Compute stellar mass to select only dwarf galaxies!
    gmag = desi_cat_f["SGA_G_COG_MAG"].data
    rmag = desi_cat_f["SGA_R_COG_MAG"].data
    zreds = desi_cat_f["Z"].data
    g_r = gmag - rmag
    
    # logm = get_stellar_mass(np.array(g_r), np.array(rmag), np.array(zreds), input_zred=False)
    logm = get_stellar_mass(g_r, rmag, desi_cat_f["DIST_MPC"].data, input_zred=False)
    
    desi_cat_f["SGA_GR"] = g_r
    desi_cat_f["SGA_LOGM_SAGA"] = logm
    
    #Apply dwarf cut
    dwarf_mask = desi_cat_f["SGA_LOGM_SAGA"] < 9.25
    desi_siena_dwarfs = desi_cat_f[dwarf_mask]
    print("Final dwarf sample:", len(desi_siena_dwarfs))


    #let us call all these to be in one sample!
    desi_siena_dwarfs["SAMPLE"] = ["SGA"]*len(desi_siena_dwarfs)
    desi_siena_dwarfs["PCNN_FRAGMENT"] = np.ones(len(desi_siena_dwarfs))
    
    return desi_siena_dwarfs


def get_image_size(zred,return_arcmin=False):
    """
    Function that returns the box size needed in pixels to obtain to do photometry
    
    Mimics the boundary:
    - For zred > 0.0125, return 1.53 arcmin
    - For zred <= 0.0125, return the value on the line connecting (0, 8) to (0.0125, 1.53)
    """
    zred = np.asarray(zred)  # Handles scalar or array input

    # Define slope and intercept for the line from (0,8) to (0.0125,1.53)
    x0, y0 = 0.0, 8.0
    x1, y1 = 0.0125, 1.53
    slope = (y1 - y0) / (x1 - x0)
    intercept = y0  # Since x0 = 0

    # Calculate the boundary value
    boundary_val = slope * zred + intercept

    # Apply the cutoff at zred = 0.0125
    img_size_arcmin = np.where(zred > 0.0125, 1.53, boundary_val)
    if return_arcmin:
        return img_size_arcmin
    else:
        img_size_pix = img_size_arcmin*60/0.262
        return img_size_pix.astype(int)


    
if __name__ == '__main__':
     
    rootdir = '/global/u1/v/virajvm/'
    sys.path.append(os.path.join(rootdir, 'DESI2_LOWZ'))
    from desi_lowz_funcs import save_table, get_useful_cat_colms, _n_or_more_gt, _n_or_more_lt, get_remove_flag
    from desi_lowz_funcs import match_c_to_catalog, get_stellar_mass, get_stellar_mass_mia, calc_normalized_dist
    from desi_lowz_funcs import get_sweep_filename, save_table, is_target_in_south

    load_zpix_iron=False
    process_sga = True

    zred_cuts = { "BGS_BRIGHT" : 0.2, "BGS_FAINT": 0.3, "LOWZ": 0.3, "ELG":0.5 }
    #either BGS_BRIGHT, ELG or BGS_FAINT
    gal_types = ["BGS_BRIGHT", "BGS_FAINT", "ELG","LOWZ"]

    save_folder = "/pscratch/sd/v/virajvm/catalog_dr1_dwarfs"

    save_filenames = {"BGS_BRIGHT":  "iron_bgs_bright_filter_zsucc_zrr02_allfracflux.fits", 
                      "BGS_FAINT": "iron_bgs_faint_filter_zsucc_zrr03_allfracflux.fits",
                       "LOWZ":  "iron_lowz_filter_zsucc_zrr03.fits" ,
                       "ELG": "iron_elg_filter_zsucc_zrr05_allfracflux.fits"}
    

    if load_zpix_iron:
        zpix_iron = Table.read("/global/cfs/cdirs/desi/spectro/redux/iron/zcatalog/v1/zall-pix-iron.fits")
    
        #we remove these columns as we will be adding columns from the zpix_trac later and so hope to avoid confusion
        cols_to_remove = [
        "FLUX_G", "FLUX_R", "FLUX_Z", "FLUX_W1", "FLUX_W2",
        "FLUX_IVAR_G", "FLUX_IVAR_R", "FLUX_IVAR_Z", "FLUX_IVAR_W1", "FLUX_IVAR_W2",
        "FIBERFLUX_G", "FIBERFLUX_Z",
        "FIBERTOTFLUX_G", "FIBERTOTFLUX_R", "FIBERTOTFLUX_Z",
        "SERSIC", "SHAPE_R", "SHAPE_E1", "SHAPE_E2"]
    
        zpix_iron.remove_columns(cols_to_remove)
    
        ##rename columns
        zpix_iron.rename_column('TARGET_RA', 'RA')
        zpix_iron.rename_column('TARGET_DEC', 'DEC')
    
        ##apply a basic cut of spectype = "GALAXY", z > 0.001 and ZCAT_PRIMARY = 1, and maximum redshift
        basic_cleaning = (zpix_iron["SPECTYPE"] == "GALAXY") & (zpix_iron["ZCAT_PRIMARY"] == 1) & (zpix_iron["COADD_FIBERSTATUS"] == 0) & (zpix_iron["Z"] < 0.5) & (zpix_iron["Z"] > 0.001)
    
        #for the near
    
        print(f"Fraction remaining after basic cleaning cut = { np.sum(basic_cleaning)/len(zpix_iron) }")
    
        zpix_iron = zpix_iron[basic_cleaning]
    
        #select for unique targets
        _, uni_tgids = np.unique( zpix_iron["TARGETID"].data, return_index=True )
        zpix_iron = zpix_iron[uni_tgids]
    
    
        #apply only maskbit+sigma cleaning or also fracflux cleaning?
        apply_only_maskbit = True
        #should I filter for successful redshifts?
        apply_zsucc_cut = True
        #should I apply some redshift cut to make sure I am dealing with potential dwarf objects only?
        apply_zred_cut = True
        cross_match_w_cigale = True
        get_color_mstar = True
        #we will remove objects that are within twice the half-light radius of SGA galaxies
    
        
        #looping over all the sub-samples!
        for i,gal_type in enumerate(gal_types):
            zred_cut = zred_cuts[gal_type]
            save_filename = save_filenames[gal_type]
            
            if gal_type == "LOWZ":
                #read in the lowz redshift and tractor phot catalogs
                zpix_sub_cat,zpix_trac = get_lowz_catalogs(zpix_iron)
    
                print(f"{gal_type} 0: Number of zpix objects = {len(zpix_sub_cat)}")
                print(f"{gal_type} 0: Number of trac objects = {len(zpix_trac)}")
                
                    
                print("Maximum RA difference is =", np.max(np.abs(zpix_sub_cat["RA"] - zpix_trac["RA"] ) ) )
    
                rmags = 22.5 - 2.5*np.log10( zpix_trac["FLUX_R"]/zpix_trac["MW_TRANSMISSION_R"] )
                
                print(f"LOWZ max and min MAG_R = {np.max(rmags), np.min(rmags) }")
                
                # combine these catalogs now, note that in zpix_trac, the flux values are not extinction corrected
                zpix_sub_cat_f, zpix_trac = get_final_catalogs(zpix_sub_cat, zpix_trac, gal_type)
    
                print(f"{gal_type} 1: Number of objects = {len(zpix_sub_cat_f)}")
                print(f"{gal_type} 1: Number of trac objects = {len(zpix_trac)}")
            
                print(f'LOWZ max and min MAG_R = {np.max(zpix_sub_cat_f["MAG_R"]), np.min(zpix_sub_cat_f["MAG_R"]) }')
    
                #to understand how the target catalogs in get_lowz_catalogs() are constructed, look at read_ls_lowz_catalogs.py script
    
                
            else:
                if gal_type == "BGS_BRIGHT":
                    iron_bgs_tgid = zpix_iron["BGS_TARGET"]
                    
                    bgsb_main_mask = ( (iron_bgs_tgid & bgs_mask["BGS_BRIGHT"]) != 0 )
    
                    bgsb_sv_mask = get_sv_bgs_mask(zpix_iron, bgs_class = "BGS_BRIGHT")
    
                    print(f"Number of BGS Bright objects in Main survey = {np.sum(bgsb_main_mask)}")
                    print(f"Number of BGS Bright objects in SV survey = {np.sum(bgsb_sv_mask)}")
                    
                    gal_mask = bgsb_main_mask | bgsb_sv_mask
                    
                    # https://github.com/astro-datalab/notebooks-latest/blob/master/03_ScienceExamples/DESI/01_Intro_to_DESI_EDR.ipynb
                    # https://github.com/Ragadeepika-Pucha/DESI_Functions/blob/main/py/desi_bits.py
                    # code to select dwarf galaxy from the different subsets
                
                if gal_type == "BGS_FAINT":
                    iron_bgs_tgid = zpix_iron["BGS_TARGET"]
                    
                    bgsf_main_mask = ( (iron_bgs_tgid & bgs_mask["BGS_FAINT"]) != 0 )
    
                    bgsf_sv_mask = get_sv_bgs_mask(zpix_iron, bgs_class = "BGS_FAINT")
    
                    print(f"Number of BGS Faint objects in Main survey = {np.sum(bgsf_main_mask)}")
                    print(f"Number of BGS Faint objects in SV survey = {np.sum(bgsf_sv_mask)}")
                    
                    gal_mask = bgsf_main_mask | bgsf_sv_mask
      
    
                if gal_type == "ELG":
                    desi_tgt = zpix_iron["DESI_TARGET"]
                    elg_main_mask = (desi_tgt & desi_mask["ELG"]) != 0
                    elg_sv_mask = get_sv_elg_mask(zpix_iron)
                    gal_mask = elg_main_mask | elg_sv_mask
    
    
                ##note that the below cleaning and matching with tractorphot stuff has already been done for LOWZ sample and hence is only being done on BGSB,BGSF and ELG samples
                
                ##select the catalog for that sample
                zpix_sub_cat = zpix_iron[gal_mask]
                print(f"Number selected in {gal_type} with just bitmask selection = {len(zpix_sub_cat)}")
    
    
                allmask_grz = [f"ALLMASK_{b}" for b in "GRZ"]
                sigma_grz = [f"SIGMA_GOOD_{b}" for b in "GRZ"]
                sigma_wise = [f"SIGMA_GOOD_W{b}" for b in range(1, 5)]
                fracflux_grz = [f"FRACFLUX_{b}" for b in "GRZ"]
                rchisq_grz = [f"RCHISQ_{b}" for b in "GRZ"]
                fracmasked_grz = [f"FRACMASKED_{b}" for b in "GRZ"]
            
                #number of bands we need 5 sigma detection in
                nsigma_bands = 2
                if gal_type == "ELG":
                    nsigma_bands = 2
            
                ##filter by maskbits
                remove_queries = [
                "(MASKBITS >> 1) % 2 > 0",  # 1
                "(MASKBITS >> 5) % 2 > 0",  # 2
                "(MASKBITS >> 6) % 2 > 0",  # 3
                "(MASKBITS >> 7) % 2 > 0",  # 4
                "(MASKBITS >> 13) % 2 > 0",  # 6
                _n_or_more_lt(sigma_grz, nsigma_bands, 5),  # 7
                ]
    
                #For now, we will not apply any SGA cleaning, we will do that in the end stage, because the SGA catalog does have dwarf galaxies.
                # "(MASKBITS >> 12) % 2 > 0",  # 5
    
                ## read tractor phot
                zpix_trac = read_tractorphot(zpix_sub_cat, verbose=True)
                zpix_trac = get_useful_cat_colms(zpix_trac)
            
                #combine some catalogs
                zpix_sub_cat, zpix_trac = get_final_catalogs(zpix_sub_cat, zpix_trac, gal_type)
    
                #cross match this with fastspec to make sure the extinction has been done correctly
                # cross_match_with_fastspec(zpix_sub_cat,gal_type)
            
                print(f"{gal_type}: Maximum RA difference is =", np.max(np.abs(zpix_trac["RA"] - zpix_sub_cat["RA"] ) ) )
                
                mask = get_remove_flag(zpix_trac, remove_queries) == 0
                mask2 = (zpix_trac["is_galaxy"] == True)
                tot_mask = mask&mask2
                ### this is the mask that will keep good objects and remove bad
                print(f"{gal_type}: Fraction that pass cleaning cuts: ", np.sum(tot_mask)/len(tot_mask) )
            
                zpix_sub_cat_f = zpix_sub_cat[tot_mask]
    
    
            ##APPLYIGNG REDSHIFT SUCCESS CRITERION!
            if apply_zsucc_cut:
                print("Applying redshift success cut.")
                ## note for ELGs, the spectroscopic cleaning cut relies on OII doublet SNR as well
                if gal_type == "ELG":
                    elg_hpx_cat_path = "/pscratch/sd/v/virajvm/catalog/all_elg_healpix_catalogs.fits"
                    if os.path.exists(elg_hpx_cat_path):
                        pass
                    else:
                        produce_elg_catalog(zpix_sub_cat)
    
                    zpix_sub_cat_f = zpix_sub_cat_f[zpix_sub_cat_f["ZWARN"] == 0]
    
                    zpix_tgids = zpix_sub_cat_f["TARGETID"].data
        
                    oii_flux = -99 * np.ones( len(zpix_tgids)  )
                    oii_flux_ivar = -99* np.ones( len(zpix_tgids)  )
                    z_hpx = -99* np.ones( len(zpix_tgids)  )
                    tgid_2 = -99* np.ones( len(zpix_tgids)  )
    
                    #read in the elg hpx catalogs which have the OII flux information
                    all_elg_data = Table.read( "/pscratch/sd/v/virajvm/catalog/all_elg_healpix_catalogs.fits")
                        
                    ## then we cross-match 
                    idx, d2d,_ = match_c_to_catalog(c_cat = zpix_sub_cat_f, catalog_cat = all_elg_data, catalog_ra = "TARGET_RA",catalog_dec= "TARGET_DEC")
        
                    all_elg_data_f = all_elg_data[idx]
                    all_elg_data_f = all_elg_data_f[d2d.arcsec < 1]
        
                    oii_flux[d2d.arcsec < 1] = all_elg_data_f["OII_FLUX"].data
                    oii_flux_ivar[d2d.arcsec < 1] = all_elg_data_f["OII_FLUX_IVAR"].data
                    z_hpx[d2d.arcsec < 1] = all_elg_data_f["Z"].data
                    tgid_2[d2d.arcsec < 1] = all_elg_data_f["TARGETID"].data
                    
                    ## using this apply the filter cut!
                    zpix_sub_cat_f["OII_FLUX"] = oii_flux
                    zpix_sub_cat_f["OII_FLUX_IVAR"] = oii_flux_ivar
                    zpix_sub_cat_f["Z_HPX"] = z_hpx
                    zpix_sub_cat_f["TARGETID_2"] = tgid_2
                    
                    zpix_sub_cat_f["OII_SNR"] = zpix_sub_cat_f["OII_FLUX"] * np.sqrt(zpix_sub_cat_f["OII_FLUX_IVAR"])
                    zpix_sub_cat_f["ZSUCC"] = (( 0.2 * np.log10(zpix_sub_cat_f["DELTACHI2"]) + np.log10(zpix_sub_cat_f["OII_SNR"]) ) > 0.9 )  
                
                    #applying the redshift cleaning cuts!
                    zpix_sub_cat_f = zpix_sub_cat_f[(zpix_sub_cat_f["ZSUCC"] == 1) & (zpix_sub_cat_f["DELTACHI2"] > 25)]
    
                    print(f"{gal_type}: Fraction of galaxies with robust redshifts = { len(zpix_sub_cat_f)/len(oii_flux) }")
    
                    #save the entire ELG redshift catalog. Only un-comment if redshift cut is not being applied
                    # np.save( save_folder + "/elg_all_redshifts.npy", np.array(zpix_sub_cat_f["Z"]) )
    
                else:
                    ## if not ELGs, but BGS, LOWZ etc.
                    good_mask =(zpix_sub_cat_f["ZWARN"] == 0) & (zpix_sub_cat_f["DELTACHI2"] > 40)
                    zpix_sub_cat_f = zpix_sub_cat_f[good_mask]
    
                    print(f"{gal_type}: Fraction of galaxies with robust redshifts = { np.sum(good_mask)/len(good_mask) }")
    
            if apply_zred_cut:
                zpix_sub_cat_f = zpix_sub_cat_f[ (zpix_sub_cat_f["Z"] < zred_cut)]
            
            if cross_match_w_cigale:
                print("Crossmatching with CIGALE VAC")
                
                zpix_sub_cat_f = cross_match_with_cigale(zpix_sub_cat_f)
        
            if get_color_mstar:
                print("Getting optical color-based stellar masses.")
                
                ## these color based prescriptions only work for Z < 0.5 galaxies though
                gr_colors = zpix_sub_cat_f["MAG_G"] - zpix_sub_cat_f["MAG_R"]
                
                zred_mask = (zpix_sub_cat_f["Z"] < 0.5)
        
                #uses r band magnitude
                mstars_SAGA = get_stellar_mass(gr_colors[zred_mask].data, zpix_sub_cat_f["MAG_R"][zred_mask].data ,zpix_sub_cat_f["Z"][zred_mask].data )
                #uses g band magnitude
                mstars_M24 = get_stellar_mass_mia(gr_colors[zred_mask].data, zpix_sub_cat_f["MAG_G"][zred_mask].data ,zpix_sub_cat_f["Z"][zred_mask].data)
        
                zpix_sub_cat_f["LOGM_SAGA"] = -99*np.ones(len(zpix_sub_cat_f))
                zpix_sub_cat_f["LOGM_M24"] = -99*np.ones(len(zpix_sub_cat_f))
        
                #add the stellar masses
                zpix_sub_cat_f["LOGM_M24"][zred_mask] = mstars_M24
                zpix_sub_cat_f["LOGM_SAGA"][zred_mask] = mstars_SAGA
        
            ## for the final catalog, add info on the sweep file!
            zpix_sub_cat_f = add_sweeps_column(zpix_sub_cat_f)
    
            ##add information on nearby bright stars
            zpix_sub_cat_f = bright_star_filter(zpix_sub_cat_f)
    
            #we filter to make sure at least one observation in each filter for every source!!
            nobs_mask = (zpix_sub_cat_f["NOBS_G"] > 0) & (zpix_sub_cat_f["NOBS_R"] > 0) & (zpix_sub_cat_f["NOBS_Z"] > 0)
            zpix_sub_cat_f = zpix_sub_cat_f[nobs_mask]
            print(f"Fraction remaining after NOBS cut = { np.sum(nobs_mask)/len(nobs_mask) }")
    

            #at the very end apply the SGA filtering cut to split the catalog.
            #One will be where MASKBIT is removed, and we will be dealing with the fragmentation in the normal typical way
            #the second one is where the sources have that MASKBIT and we will be choosing which objects are dwarfs and which are not.
    
            sga_queries = ["(MASKBITS >> 12) % 2 > 0"]
            sga_mask = get_remove_flag(zpix_sub_cat_f, sga_queries) == 0
    
            zpix_sub_cat_w_sga = zpix_sub_cat_f[~sga_mask]
            zpix_sub_cat_no_sga = zpix_sub_cat_f[sga_mask]
    
            print(f"{gal_type}: Objects in with SGA catalog = {len(zpix_sub_cat_w_sga)}")
            print(f"{gal_type}: Objects not in SGA catalog = {len(zpix_sub_cat_no_sga)}")
            print(f"{gal_type}: Fraction removed by SGA maskbit cut = {len(zpix_sub_cat_w_sga)/len(zpix_sub_cat_f)  }")
            
            ## save this catalog. This is catalog with the MASBIT cut applied!
            save_table(zpix_sub_cat_no_sga,  save_folder + "/" + save_filename,comment="")
    
            #save the with SGA catalog too
            save_table(zpix_sub_cat_w_sga,  save_folder + "/" + save_filename.replace(".fits","_W_SGA.fits"),comment="")
    


    ##add the image size pix column

    for sample_i in gal_types:
        file_1 = save_folder + "/" + save_filenames[sample_i]
        file_2 = save_folder + "/" + save_filenames[sample_i].replace(".fits","_W_SGA.fits")

        print(file_1)
        print(file_2)
        zpix_cat_1 = Table.read(file_1)
        zpix_cat_2 = Table.read(file_2)
        
        ##add column detailing info on the cutout size needed
        zpix_cat_1["IMAGE_SIZE_PIX"] = get_image_size(zpix_cat_1["Z"].data,return_arcmin=False)
        zpix_cat_2["IMAGE_SIZE_PIX"] = get_image_size(zpix_cat_2["Z"].data,return_arcmin=False)
        
        #save the file now!
        save_table(zpix_cat_1,  file_1,comment="")
        save_table(zpix_cat_2,  file_2,comment="")
        
    if process_sga:
        ##by construction, the LOWZ and ELG samples do not overlap with the SGA catalog

        bgsb_cat = Table.read("/pscratch/sd/v/virajvm/catalog_dr1_dwarfs/iron_bgs_bright_filter_zsucc_zrr02_allfracflux_W_SGA.fits")
        bgsf_cat = Table.read("/pscratch/sd/v/virajvm/catalog_dr1_dwarfs/iron_bgs_faint_filter_zsucc_zrr03_allfracflux_W_SGA.fits")

        bgsb_cat["SAMPLE_DESI"] = ["BGS_BRIGHT"] * len(bgsb_cat)
        bgsf_cat["SAMPLE_DESI"] = ["BGS_FAINT"] * len(bgsf_cat)
    
        print("Number of sources overlapping with SGA pixels: ", len(bgsb_cat), len(bgsf_cat))
        print("Note that all of these are not dwarf galaxies!")

        desi_sga_cat = vstack([bgsb_cat, bgsf_cat])

        desi_sga_dwarfs = process_sga_matches(desi_sga_cat)


        ##39628526738999876 -> this object we want to remove as it is not a dwarf galaxy and has missing imaging 
        ## and thus the cutout viewer tool breaks there

        desi_sga_dwarfs = desi_sga_dwarfs[desi_sga_dwarfs["TARGETID"] != 39628526738999876]

        save_table(desi_sga_dwarfs,  save_folder + "/iron_sga_matched_dwarfs.fits",comment = "")

        

        

# ## According to Ashley Ross, the LSS catalogs contain the tile based redshift which would be slightly different than the healpix
# ##based redshift. The below file would give us the tile based redshift. 

# # elg_lss = fits.open("/global/cfs/cdirs/desi/survey/catalogs/dr1/LSS.dr1/iron/LSScats/v1.5/ELG_LOPnotqso_full.dat.fits")
# # elg_lss_data = elg_lss[1].data
# # elg_lss_tgids = elg_lss_data["TARGETID"].data
# # print(len(vac_data_tgids), len(elg_lss_data))

# # Find the common values and their indices in arr1
# # common_values, idx1, _ = np.intersect1d(vac_data_tgids, elg_lss_tgids, return_indices=True)

# # # Find the indices of these values in arr2
# # match_inds = np.nonzero(np.isin(elg_lss_tgids, common_values))[0]

# # # arr1 is now filtered to only contain values in arr2
# # vac_data_cat_f = vac_data_cat_f[idx1]

# # print(len(match_inds), len(vac_data_cat_f))

# # vac_data_cat_f["OII_FLUX"] = elg_lss_data["OII_FLUX"][match_inds]
# # vac_data_cat_f["OII_FLUX_IVAR"] = elg_lss_data["OII_FLUX_IVAR"][match_inds]
# # vac_data_cat_f["WEIGHT_ZFAIL"] = elg_lss_data["WEIGHT_ZFAIL"][match_inds]
# # vac_data_cat_f["mod_success_rate"] = elg_lss_data["mod_success_rate"][match_inds]
# # vac_data_cat_f["TARGETID_2"] = elg_lss_data["TARGETID"][match_inds]
# # vac_data_cat_f["DELTACHI2_2"] = elg_lss_data["DELTACHI2"][match_inds]
# # vac_data_cat_f["Z_not4clus"] = elg_lss_data["Z_not4clus"][match_inds]

# vac_data_cat_f =  vac_data_cat_f[vac_data_cat_f["OII_FLUX"] > 0]
# vac_data_cat_f["OII_SNR"] = vac_data_cat_f["OII_FLUX"] * np.sqrt(vac_data_cat_f["OII_FLUX_IVAR"])
# vac_data_cat_f["ZSUCC"] = (( 0.2 * np.log10(vac_data_cat_f["DELTACHI2"]) + np.log10(vac_data_cat_f["OII_SNR"]) ) > 0.9 )

###healpix based redshift!!
# all_healpix = np.unique(  vac_data_cat_f["HEALPIX"].data )


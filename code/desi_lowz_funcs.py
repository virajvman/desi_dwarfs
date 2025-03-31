## this contains useful functions for DESI LOWZ analysis 
import numpy as np
from tqdm import trange, tqdm
import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.io import fits
#path where SIENA Galaxy Atlas is stored
siena_path = "/global/cfs/cdirs/cosmo/data/sga/2020/SGA-2020.fits"
from astropy.table import Table
from astropy.cosmology import Planck18    
import os, pdb
import fitsio
from astropy.table import Table, vstack, hstack
from easyquery import Query, QueryMaker
import random
import multiprocessing as mp

    
def sdss_rgb(imgs, bands, scales=None,m = 0.02):
    rgbscales = {'u': (2,1.5), #1.0,
                 'g': (2,2.5),
                 'r': (1,1.5),
                 'i': (0,1.0),
                 'z': (0,0.4), #0.3
                 }
    
    if scales is not None:
        rgbscales.update(scales)

    I = 0
    for img,band in zip(imgs, bands):
        plane,scale = rgbscales[band]
        img = np.maximum(0, img * scale + m)
        I = I + img
    I /= len(bands)
        
    # b,g,r = [rimg * rgbscales[b] for rimg,b in zip(imgs, bands)]
    # r = np.maximum(0, r + m)
    # g = np.maximum(0, g + m)
    # b = np.maximum(0, b + m)
    # I = (r+g+b)/3.
    Q = 20
    fI = np.arcsinh(Q * I) / np.sqrt(Q)
    I += (I == 0.) * 1e-6
    H,W = I.shape
    rgb = np.zeros((H,W,3), np.float32)
    for img,band in zip(imgs, bands):
        plane,scale = rgbscales[band]
        rgb[:,:,plane] = (img * scale + m) * fI / I

    # R = fI * r / I
    # G = fI * g / I
    # B = fI * b / I
    # # maxrgb = reduce(np.maximum, [R,G,B])
    # # J = (maxrgb > 1.)
    # # R[J] = R[J]/maxrgb[J]
    # # G[J] = G[J]/maxrgb[J]
    # # B[J] = B[J]/maxrgb[J]
    # rgb = np.dstack((R,G,B))
    rgb = np.clip(rgb, 0, 1)
    
    return rgb


def get_random_markers(n):
    markers = ["o", "s", "D", "^", "v", "<", ">", "p", "*", "H", "+", "x", "|", "_"]
    
    # If n is larger than the number of unique markers, allow repetition
    return random.choices(markers, k=n) if n > len(markers) else random.sample(markers, n)

def get_sweep_filename(ra, dec):
    '''
    example file name sweep-140p000-150p005-pz.fits
    '''
    
    ra_min = int(10 * (ra // 10))
    ra_max = int(ra_min + 10)
    dec_min = int(5 * (dec // 5))
    dec_max = int(dec_min + 5)
    
    ra_min_str = f"{ra_min:03d}"
    ra_max_str = f"{ra_max:03d}"

    # Add 'p' for positive Dec and 'm' for negative Dec
    dec_min_str = f"{'p' if dec_min >= 0 else 'm'}{abs(dec_min):03d}"
    dec_max_str = f"{'p' if dec_max >= 0 else 'm'}{abs(dec_max):03d}"

    return f"sweep-{ra_min_str}{dec_min_str}-{ra_max_str}{dec_max_str}-pz.fits"

    
def calc_normalized_dist(obj_ra, obj_dec, cen_ra, cen_dec, cen_r, cen_ba=None, cen_phi=None, multiplier=2.0):
    """
    Function from YYMao on whether a given point is within an ellipse or not
    obj_ra, obj_dec, cen_ra, cen_dec in degrees
    cen_r is the semi-major axis in arcseconds, set multiplier as needed
    """
    a = cen_r / 3600.0 * multiplier
    cos_dec = np.cos(np.deg2rad((obj_dec + cen_dec) * 0.5))
    dx = np.rad2deg(np.arcsin(np.sin(np.deg2rad(obj_ra - cen_ra)))) * cos_dec
    dy = obj_dec - cen_dec

    # if cen_ba is None:
    #     with np.errstate(divide="ignore"):
    #         return np.hypot(dx, dy) / a

    b = a * cen_ba
    theta = np.deg2rad(90 - cen_phi)
    sin_t = np.sin(theta)
    cos_t = np.cos(theta)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.hypot((dx * cos_t + dy * sin_t) / a, (-dx * sin_t + dy * cos_t) / b)



def is_target_in_south(ras,decs):
    '''
    This is a function that takes in RA,DEC and decides if it is in south or north catalog
    '''
    # south_targs_mask = ((south_targs["GALB"] < 0) | ((south_targs["GALB"] > 0) & (south_targs["DEC"] < 32.375) ))
    # north_targs_mask = (north_targs["GALB"] > 0) & (north_targs["DEC"] > 32.375) & (np.abs(north_targs["GALB"]) > 15)

    c_cord = SkyCoord(ras * u.degree, decs * u.degree, frame='icrs')
    
    galbs = c_cord.galactic.b.value

    is_south = ( (galbs < 0) | ( (galbs > 0) & (decs < 32.375) )  )

    return is_south


def print_stage(line2print, ch='-',end_space=True):
    '''
    Function that prints lines for organizational purposes in the code outputs.

    Parameters:
    -----------
    line2print: str, the message to be printed
    ch : str, the boundary dividing character of the message
    '''
    nl = len(line2print)
    print(ch*nl)
    print(line2print)
    print(ch*nl)
    if end_space == True:
        print(' ')

def check_path_existence(all_paths=None):
    '''
    Creates directories if they do not exist

    Parameters:
    --------------
    all_paths: list, directory list to loop over
    '''
    for pi in all_paths:
        if not os.path.exists(pi):
            print_stage('The path {:s} did not exist. It has now been created.'.format(pi),ch="-")
            os.makedirs(pi)
    return


def match_c_to_catalog(c_cat = None, catalog_cat = None, c_ra = "TARGET_RA", c_dec = "TARGET_DEC", catalog_ra = "TARGET_RA", catalog_dec = "TARGET_DEC"):
    '''
    Function that matches two catalogs and returns the idx, d2d, d3d
    '''
    c = SkyCoord(ra= c_cat[c_ra].data* u.degree, dec= c_cat[c_dec].data*u.degree )
    catalog = SkyCoord(ra=catalog_cat[catalog_ra].data*u.degree, dec=catalog_cat[catalog_dec].data*u.degree )
    idx, d2d, d3d = c.match_to_catalog_sky(catalog)
    return idx, d2d, d3d




def print_radecs(object_list, num=100, ra="TARGET_RA", dec="TARGET_DEC"):
    num = np.minimum( num, len(object_list) )
    
    for i in range(num):
        print(object_list[ra][i], object_list[dec][i]  )
        
    return

def find_objects_nearby(object_list, find_ra, find_dec, deg_rad = 1e-3,ra="TARGET_RA", dec="TARGET_DEC"):
    
    mask = (np.abs(object_list[ra] - find_ra) < deg_rad) & (np.abs(object_list[dec] - find_dec) < deg_rad)
    return object_list[mask]
    

def apply_lowz_mask(table):
    '''
    This function filters table for LOW-Z targets by looking at SCND_TARGET 
    '''
    
    lowz_mask = (table["SCND_TARGET"] == 2**15) | (table["SCND_TARGET"] == 2**16) | (table["SCND_TARGET"] == 2**17)
    
    return table[lowz_mask]


def get_contours(quant_x, quant_y,bins,sigs=False):
    '''
    Want to get contours for the x vs. y relation. This will return a dict with the corresponding contours
    '''

    med = []
    c1sL = []
    c1sH = []
    c2sL = []
    c2sH = []

    for i in range(len(bins)-1):
        temp_y = quant_y[ (quant_x > bins[i]) & (quant_x < bins[i+1]) ]

        med.append(np.nanmedian(temp_y) )
        if sigs:
            c1sL.append(np.nanpercentile(temp_y,16) )  
            c1sH.append(np.nanpercentile(temp_y,84) )  
            c2sL.append(np.nanpercentile(temp_y,2.5) )  
            c2sH.append(np.nanpercentile(temp_y,97.5) ) 
        
    #write this into dictionary
    
    temp_dict = {"bin_cents":0.5*(bins[1:] + bins[:-1])  , "median" : med, 
                "sig1_low" : c1sL, "sig1_high" : c1sH, "sig2_low":c2sL , "sig2_high" : c2sH }
    
    return temp_dict



def get_legacy_survey_link(table,prefix="TARGET_",desi=True):
    '''
    Give this function a table and it will return a table with all the relevant legacy survey links to explore
    '''
    
    links = []
    
    
    for i in range(len(table)):
        ra = table[prefix+"RA"][i]
        dec = table[prefix+"DEC"][i]
        template = "https://www.legacysurvey.org/viewer-desi?ra=%.8f&dec=%.8f&layer=ls-dr10-grz&zoom=14"%(ra,dec)
        links.append(template)
        
    table["LS_links"] = links
    
    return table



def g_kcorr(gr,z):
    '''
    This function returns the k correction for SDSSgr band 
    
    According to the Chilingarian et al. 2010 from which this k-correction is based, 
    we can only apply this on z<0.5    
    '''
    
    # it is power of z * power of gr
    coeff_10 = -0.900332 * (z**1) * (gr**0)
    coeff_11 = 3.97338  * (z**1) * (gr**1)
    coeff_12 = 0.774394  * (z**1) * (gr**2)
    coeff_13 = -1.09389  * (z**1) * (gr**3)
    
    coeff_20 = 3.65877  * (z**2) * (gr**0)
    coeff_21 = -8.04213 * (z**2) * (gr**1)
    coeff_22 = 11.0321 * (z**2) * (gr**2)
    coeff_23 = 0.781176 * (z**2) * (gr**3)
    
    coeff_30 = -16.7457 * (z**3) * (gr**0)
    coeff_31 = -31.1241 * (z**3) * (gr**1)
    coeff_32 = -17.5553 * (z**3) * (gr**2)
    
    coeff_40 = 87.3565 * (z**4) * (gr**0)
    coeff_41 = 71.5801 * (z**4) * (gr**1)
    
    coeff_50 = -123.671 * (z**5) * (gr**0)

    kg =  (coeff_10 + coeff_11 + coeff_12 + coeff_13) + (coeff_20 + coeff_21 + coeff_22 + coeff_23) + (coeff_30 + coeff_31 + coeff_32) + (coeff_40 + coeff_41) + (coeff_50)
    
    return kg


def r_kcorr(gr,z):
    '''
    This function returns the k correction for SDSS r band 
    
    According to the Chilingarian et al. 2010 from which this k-correction is based, 
    we can only apply this on z<0.5
    
    '''
    
    
    # it is power of z * power of gr
    coeff_10 = -1.61294 * (z**1) * (gr**0)
    coeff_11 = 3.81378  * (z**1) * (gr**1)
    coeff_12 = -3.56114  * (z**1) * (gr**2)
    coeff_13 = 2.47133  * (z**1) * (gr**3)
    
    coeff_20 = 9.13285  * (z**2) * (gr**0)
    coeff_21 = 9.85141 * (z**2) * (gr**1)
    coeff_22 = -5.1432 * (z**2) * (gr**2)
    coeff_23 = -7.02213 * (z**2) * (gr**3)
    
    coeff_30 = -81.8341 * (z**3) * (gr**0)
    coeff_31 = -30.3631 * (z**3) * (gr**1)
    coeff_32 = 38.5052 * (z**3) * (gr**2)
    
    coeff_40 = 250.732 * (z**4) * (gr**0)
    coeff_41 = -25.0159 * (z**4) * (gr**1)
    
    coeff_50 = -215.377 * (z**5) * (gr**0)

    kr =  (coeff_10 + coeff_11 + coeff_12 + coeff_13) + (coeff_20 + coeff_21 + coeff_22 + coeff_23) + (coeff_30 + coeff_31 + coeff_32) + (coeff_40 + coeff_41) + (coeff_50)
    
    return kr

def get_stellar_mass_mia( gr_col, gmag, zred):
    '''
    we use the 2 color prescriptions here. One from SAGA and the updated one from Mia's paper.
    Note that Mia's formula is only valid for Mstar < 1e10 though which is okay for us!
    '''
    #use redshift to convert to absolute magnitude
    from astropy.cosmology import Planck18

    #convert the zred to the luminosity distance 
    d = Planck18.luminosity_distance(zred)
    d_in_pc = d.value * 1e6
    #the k correction
    kg = g_kcorr(gr_col,zred)
    Mg = gmag + 5 - 5*np.log10(d_in_pc) - kg
    
    log_mstar = (1.433 * gr_col) + 0.00153 * (Mg**2) - (0.335 * Mg) + 2.072
    
    return log_mstar

def get_stellar_mass(gr,rmag,zred):
    '''
    Computes the stellar mass of object using the SAGA 2 conversion
    
    It is given by Log10(Mstar) = 1.254 + 1.098*(g-r) - 0.4 M_r
    
    We would need to get the absolute r band mag above. We will also have to a K correction to the absolute magnitude .. 
    
    '''
    from astropy.cosmology import Planck18

    #convert the zred to the luminosity distance 
    d = Planck18.luminosity_distance(zred)
    d_in_pc = d.value * 1e6
    
    #M = m + 5 - 5*log10(d/pc) - Kcor
    kr = r_kcorr(gr,zred)
    M_r = rmag + 5 - 5*np.log10(d_in_pc) - kr
    
    log_star = 1.254 + 1.098*gr - 0.4*M_r
    
    return log_star

def get_mag_from_stellar_mass(gr,mstar,zred):
    '''
    Computes the stellar mass of object using the SAGA 2 conversion
    
    It is given by Log10(Mstar) = 1.254 + 1.098*(g-r) - 0.4 M_r
    
    We would need to get the absolute r band mag above. We will also have to a K correction to the absolute magnitude .. 
    
    '''
    from astropy.cosmology import Planck18

    #convert the zred to the luminosity distance 
    d = Planck18.luminosity_distance(zred)
    d_in_pc = d.value * 1e6
    
    # log_star = 1.254 + 1.098*gr - 0.4*M_r
    M_r = -1*(mstar - 1.254 - 1.098*gr)/0.4
    
    #M = m + 5 - 5*log10(d/pc) - Kcor
    kr = r_kcorr(gr,zred)
    # M_r = rmag + 5 - 5*np.log10(d_in_pc) - kr
    rmag = M_r - 5 + 5*np.log10(d_in_pc) + kr
    
    return rmag

## compute the haversine formula 


def get_redshift_success(zred_catalog,min_deltachi2 = 40,spectype_filt = True,min_zred = 0.001):
    '''
    This function filters out the catalog for redshift success objects
    
    and also chooses only extragalactic objects
    
    '''
    zwarn = zred_catalog["ZWARN"]
    deltachi2 = zred_catalog["DELTACHI2"]
    zred = zred_catalog["Z"]
    
    
    if spectype_filt:
        spectype = zred_catalog["SPECTYPE"]
        return zred_catalog[ (zwarn == 0) & (deltachi2 > min_deltachi2) & (spectype == "GALAXY") & (zred > min_zred) ]
    
    
    return zred_catalog[ (zwarn == 0) & (deltachi2 > min_deltachi2) ]

    
    
def save_table(table_data, save_path, comment = ""):
    '''
    An example of a comment can be 

    comment = "This contains LOW-Z dark time data from Iron (all as in including redshift failures as well. The phot_final means that we cross matched these observations to z<0.03 target catalogs to get accurate photometry."
    
    This comment will be saved in one of the hdus so I can later recall notes on this dataset
    
    '''
    # Create primary HDU with your table
    primary_hdu = fits.PrimaryHDU()
    table_hdu = fits.table_to_hdu(table_data)

    # Create a new HDU with comment
    comment_hdu = fits.ImageHDU()
    comment_hdu.header['COMMENT'] = comment

    # Create HDU list and add the HDUs
    hdul = fits.HDUList([primary_hdu, table_hdu, comment_hdu])

    # Save the HDU list to a FITS file
    # fits_file = "data/catalogs/Iron_lowz_dark_all_phot_final.fits"
    hdul.writeto(save_path, overwrite=True)

    return
    

def haversine_np(ra1, dec1, ra2, dec2):
    """
    Calculate the great circle distance between two points
    on the earth (specified in decimal degrees)
    
    All args must be of equal length.    
    
    I have confirmed that the output of this function is the same as SkyCoord separation!
    
    """
    ra1, dec1, ra2, dec2 = map(np.radians, [ra1, dec1, ra2, dec2])
    
    dra = ra2 - ra1
    ddec = dec2 - dec1
    
    a = np.sin(ddec/2.0)**2 + np.cos(dec1) * np.cos(dec2) * np.sin(dra/2.0)**2
    
    dist = 2 * np.arcsin(np.sqrt(a))
    return dist


def filter_out_siena(ra_list,dec_list,Nmax = 500,print_distance=False,which_rad = "SMA"):
    '''
    This function takes in an array of ra and dec values and sees which ones falls 
    within a large galaxy in the SIENA Galaxy Atlas 
    '''
    #read the SIENA Galaxy Atlas
    siena20 = Table.read(siena_path,hdu = "ELLIPSE")
    
    ## the D26 column is in units of arcmins and is the major axis diameter measured at the 𝜇=26 mag arcsec−2 r-band isophote
    ## below is what SAGA does, Yao recommended me to do this 
    ## np.minimum(sga_this["SMA_MOMENT"] * 1.3333, sga_this["D26"] * 30.0)

    s20_d26 = np.array(siena20["D26"])*30 #converting to semi-major axis in arc seconds
    s20_ra = np.array(siena20["RA_MOMENT"])
    s20_dec = np.array(siena20["DEC_MOMENT"])
    s20_sma = np.array(siena20["SMA_MOMENT"])*1.33
    s20_ba = np.array(siena20["BA"])
    s20_phi = np.array(siena20["PA"]) # the position angle
    
    centers = np.column_stack((s20_ra, s20_dec))
    
    #this is the angular radius of big galaxy we will use
    if which_rad == "SMA":
        s20_best_rad = s20_sma
    if which_rad == "D26":
        s20_best_rad = s20_d26
    if which_rad == "smallest":
        s20_best_rad = np.minimum(s20_sma, s20_d26)
    if which_rad == "biggest":
        s20_best_rad = np.maximum(s20_sma, s20_d26)
    

    #the semi-major and semi-minor axes
    a = s20_best_rad / 3600 #converting to degrees
    b = a * s20_ba
      
    '''
    I will first identify what is closet SGA galaxy to each point under consideration
    '''
    ##To avoid memroy issues I will batch this into 500 galaxies and compute it 
    # Number of splits (equal lengths)
    num_splits = int(np.ceil(len(ra_list)/Nmax))

    # Split the array
    ra_lists = np.array_split(ra_list, num_splits)
    dec_lists = np.array_split(dec_list, num_splits)
    
    all_min_inds = []
    
    for i in trange(len(ra_lists)):
        
        ra_list_i = ra_lists[i]
        dec_list_i = dec_lists[i]
        
        ra_i = ra_list_i[:,np.newaxis]
        dec_i = dec_list_i[:,np.newaxis]

        all_dists = haversine_np(ra_i, dec_i, s20_ra, s20_dec)
    
        min_inds = np.argmin(all_dists,axis = 1)
        
        all_min_inds.append(min_inds)
        
    min_inds = np.concatenate(all_min_inds)
    
    a_f = a[min_inds]
    b_f = b[min_inds]
    s20_ra_f = s20_ra[min_inds]
    s20_dec_f = s20_dec[min_inds]
    s20_phi_f = s20_phi[min_inds]
    
    #then we do calculations like normal
    print("The maximum RA separation between SGA galaxy and galaxy of interest is %.4f deg."%np.max(np.abs(ra_list - s20_ra_f)) )

    cos_dec = np.cos(np.deg2rad((dec_list + s20_dec_f) * 0.5))
    #the below fails if the two galaxies are very far away (e.g. 180 separation)
    #however, as we are computing galaxies between two already close galaxies using sin or cos does not matter
    dx = np.rad2deg(np.arcsin(np.sin(np.deg2rad(ra_list - s20_ra_f)))) * cos_dec 
    
    dy = dec_list - s20_dec_f
        
    #computing position angle stuff
    theta = np.deg2rad(90 - s20_phi_f)
    sin_t = np.sin(theta)
    cos_t = np.cos(theta)
        
    distances = np.hypot( (dx*cos_t + dy*sin_t)/a_f , (-dx*sin_t + dy*cos_t)/b_f  )
    
    # Check if squared distance is less than the square of the radius for each circle
    within_circles = distances <= 1
    
    if print_distance:
        print(distances)
    
    indices_overlap = np.where(within_circles)

    print("We found %d galaxies that are overlapped by SIENA Atlas members!"%len(ra_list[indices_overlap]))
    
    ## we need to create a mask of galaxies that are good!1
    
    good_mask = np.ones(len(ra_list), dtype=bool)
    good_mask[indices_overlap] = False
    
    return good_mask, indices_overlap

def calc_normalized_dist_broadcast(ra_list, dec_list, redshift_list, s20_ra, s20_dec, s20_zred, s20_d26, s20_ba, s20_phi,verbose):
    '''
    Function that computes the distance normalized to D26 units in a vectorized manner. This function can be passed to a multiprocessor as well
    '''
    c_light = 299792 #km/s

    #expanding the ra,dec arrays for broad-casting
    ra_i = ra_list[:,np.newaxis]
    dec_i = dec_list[:,np.newaxis]
    redshift_i = redshift_list[:, np.newaxis]

    redshift_mask = (np.abs(redshift_i - s20_zred) * c_light < 1000 )  # Shape: (N, M)

    if verbose:
        print(np.shape(redshift_mask))

    all_norm_dist = calc_normalized_dist(ra_i, dec_i, s20_ra, s20_dec, s20_d26, 
                        cen_ba=s20_ba, cen_phi=s20_phi, multiplier=1)

    ##for each ra,dec in ra_i,dec_i lists, we have computed the distances to all the SGA galaxies. 
    ##for each, we need to filter them by their redshifts. 

    # Mask out distances for galaxies that are outside the redshift range
    all_norm_dist[~redshift_mask] = np.inf 

    nearest_inds = np.argmin(all_norm_dist,axis = 1)
    nearest_norm_dist = np.min( all_norm_dist, axis = 1 )
    #can I just use the nearest_inds thing?
    if verbose:
        print(np.shape(nearest_inds), np.shape(nearest_norm_dist))

    # If all values in a row were masked 
    # that is, if there are no sga galaxies in the relevant redshift range
    no_match_mask = np.all(~redshift_mask, axis=1)

    if verbose:
        print(np.shape(no_match_mask))

    nearest_inds[no_match_mask] = np.nan
    #where there does not exist a valid match, we assign np.nan value
    nearest_norm_dist[no_match_mask] = np.nan

    return nearest_inds, nearest_norm_dist


def get_sga_norm_dists(ra_list,dec_list, redshift_list, Nmax = 500,run_parr = False,ncores = 128,siena_path=None,verbose=True):
    '''
    This function takes in an array of ra,dec and redshift values and see which ones are potentially shreds of a larger SGA galaxy.
    It uses the the calc_normalized_dist function to compute the distance of source from SGA in units of its elliptical aperture

    This function shall return a list of matched SGA id and associated angular norm dist (if relevant) for each object in list

    ra_list: list of all the RAs in the catalog
    dec_list: list pf all the DECs in the catalog
    redshift_list: list of all the redshifts in the catalog
    Nmax : the number of chunks to split the ra_lists into for parallelizing

    '''

    #read the SIENA Galaxy Atlas
    siena20 = Table.read(siena_path,hdu = "ELLIPSE")
    
    ## the D26 column is in units of arcmins and is the major axis diameter measured at the 𝜇=26 mag arcsec−2 r-band isophote
    ## below is what SAGA does, Yao recommended me to do this 
    ## np.minimum(sga_this["SMA_MOMENT"] * 1.3333, sga_this["D26"] * 30.0)

    s20_d26 = np.array(siena20["D26"])*30 #converting to semi-major axis in arc seconds
    s20_ra = np.array(siena20["RA_MOMENT"])
    s20_dec = np.array(siena20["DEC_MOMENT"])
    s20_zred = np.array(siena20["Z_LEDA"])

    s20_sgaid = np.array(siena20["SGA_ID"])

    # s20_sma = np.array(siena20["SMA_MOMENT"])*1.33
    s20_ba = np.array(siena20["BA"])
    s20_phi = np.array(siena20["PA"]) # the position angle
    
    '''
    Goal is to identify nearest SGA galaxy with same redshift 
    '''
    ##To avoid memroy issues I will batch this into 500 galaxies and compute it 
    # Number of splits (equal lengths)
    num_splits = int(np.ceil(len(ra_list)/Nmax))

    # Split the array
    ra_chunk_lists = np.array_split(ra_list, num_splits)
    dec_chunk_lists = np.array_split(dec_list, num_splits)
    redshift_chunk_lists = np.array_split(redshift_list, num_splits)

    if verbose:
        print(np.shape(ra_chunk_lists))

    if run_parr:
        all_input_tuples = []
        ##looping over all the chunks. Every chunk will run in parallel.
        for i in trange(len(ra_chunk_lists)):
            all_input_tuples.append(  ( ra_chunk_lists[i], dec_chunk_lists[i], redshift_chunk_lists[i], s20_ra, s20_dec, s20_zred, s20_d26, s20_ba, s20_phi, verbose )  )


        with mp.Pool(processes=ncores) as pool:
                results = list(tqdm(pool.starmap(calc_normalized_dist_broadcast, all_input_tuples), total = len(all_input_tuples)  ))

        results = np.array(results)
        results = np.concatenate(results,axis=1)
        ##I might need to concatenate these arrays

        all_nearest_inds = results[0]
        all_nearest_dists = results[1]

    else:
        all_nearest_inds = []
        all_nearest_dists = []    
        ##for each object in the catalog, I need to compute the normalized distance to the nearest SGA galaxy

        for i in trange(len(ra_chunk_lists)):

            temp_i, temp_d = calc_normalized_dist_broadcast(ra_chunk_lists[i], dec_chunk_lists[i], redshift_chunk_lists[i], s20_ra, s20_dec, s20_zred, s20_d26, s20_ba, s20_phi, verbose)

            all_nearest_inds.append(temp_i)
            all_nearest_dists.append(temp_d)

        all_nearest_inds = np.concatenate(all_nearest_inds)
        all_nearest_dists = np.concatenate(all_nearest_dists)


    matched_sga_ids = np.ones( len(ra_list) ) * np.nan

    matched_sga_ids[ all_nearest_inds != np.nan ] = s20_sgaid[ all_nearest_inds[all_nearest_inds!=np.nan ] ]

    return matched_sga_ids, all_nearest_dists

    
    

def read_tractorphot_iron(zcat, specprod='iron', verbose=False):
    """Gather Tractor photometry for an input catalog. Note that this function
    hasn't been tested very extensively and will likely fail if there are duplicate
    TARGETIDs in the input redshift catalog.

    """
    from glob import glob
    from desitarget import geomask
    from desimodel.footprint import radec2pix
    
    dr = 'dr1'
    specprod = 'iron'
    lsdr9_version = 'v1.1'
    vacdir = f'/global/cfs/cdirs/desi/public/{dr}/vac/{dr}/lsdr9-photometry/{specprod}/{lsdr9_version}/observed-targets'
    ##this is for iron catalog
    
    
    tractorphotfiles = glob(os.path.join(vacdir, 'tractorphot', f'tractorphot-nside4-hp???-{specprod}.fits'))

    hdr = fitsio.read_header(tractorphotfiles[0], 'TRACTORPHOT')
    tractorphot_nside = hdr['FILENSID']

    pixels = radec2pix(tractorphot_nside, zcat['TARGET_RA'], zcat['TARGET_DEC'])
    phot = []
    
    tot_count = 0
    
    for pixel in sorted(set(pixels)):
        J = pixel == pixels
        photfile = os.path.join(vacdir, 'tractorphot', f'tractorphot-nside4-hp{pixel:03d}-{specprod}.fits')
        if os.path.isfile(photfile):
            targetids = fitsio.read(photfile, columns='TARGETID')
            K = np.where(np.isin(targetids, zcat['TARGETID'][J]))[0]
            tot_count += 100*len(K)/len(zcat)
            if verbose:
                print(f'Finished gathering Tractor photometry for %.2f percent objects!'%tot_count)
                
            _phot = fitsio.read(photfile, rows=K)
            phot.append(Table(_phot))

    if len(phot) > 0:
        phot = vstack(phot)
    else:
        phot = Table()

    # check for objects with missing Tractor photometry
    if len(phot) > 0:
        match = np.where(np.isin(zcat['TARGETID'], phot['TARGETID']))[0]
        miss = np.where(np.logical_not(np.isin(zcat['TARGETID'], phot['TARGETID'])))[0]

        srt = geomask.match_to(phot['TARGETID'], zcat['TARGETID'][match])
        phot = phot[srt]
        assert(np.all(phot['TARGETID'] == zcat['TARGETID'][match]))
    else:
        miss = np.arange(len(zcat))
        
    if len(miss) > 0:
        print(f'Missing Tractor photometry for {len(miss):,d} object(s).')
        zcat_missing = zcat[miss]
    else:
        zcat_missing = Table()
            
    return phot, zcat_missing


def get_tgids_fastspec(tgids_list, columns):
    '''
    Given a list of targetids, get the corresponding fastspecfit columns. 
    '''
    #VAC data upload
    iron_vac = fits.open("/global/cfs/cdirs/desi/public/dr1/vac/dr1/fastspecfit/iron/v2.1/catalogs/fastspec-iron.fits")
    vac_data_2 = iron_vac[1].data

    temp = {}
    for ci in columns:
        temp[ci] = vac_data_2[ci]

    mask = np.isin(temp['TARGETID'], np.array(tgids_list) )

    temp_f = {}
    for ci in columns:
        temp_f[ci] = temp[ci][mask]
    
    return temp_f, mask

        
    


def read_vac_line_info(gal_cat,columns,which_vac = "lines",coord_name = ""):
    '''
    In this function, I feed in a catalog, and based on RA,DEC,TARGETID, the VAC info on lines is obtained.
    
    gal_cat is the catalog we want to match
    columns is the list of columns we will be reading, appending and then returning 
    
    if which_vac == lines, then we get all the emissio line fastspec fit stuff
    if which_vac == normal, then we get the normal catalog
    
    '''
    #VAC data upload
    iron_vac = fits.open("/global/cfs/cdirs/desi/public/dr1/vac/dr1/fastspecfit/iron/v2.1/catalogs/fastspec-iron.fits")
    vac_data = iron_vac[2].data
    vac_data_2 = iron_vac[1].data
    
    
    
    #we match our catalog to the 
    catalog = SkyCoord(ra= np.array(vac_data["RA"])* u.degree, dec= np.array(vac_data["DEC"])*u.degree )
    c = SkyCoord(ra=np.array(gal_cat[coord_name + "RA"])*u.degree, dec=np.array(gal_cat[coord_name + "DEC"])*u.degree )
    idx, d2d, d3d = c.match_to_catalog_sky(catalog)
    
    print("The maximum distance in arcsec is = ", np.max(d2d.arcsec))
    
    if which_vac == "lines":
        vac_data_2_cat = Table(vac_data_2[idx])
    if which_vac == "normal":
        vac_data_2_cat = Table(vac_data[idx])
    

    ##then we read the columns
    vac_data_2_cat = vac_data_2_cat[columns]
    
    #then we horizontal stack these with gal_cat
    gal_cat = hstack([gal_cat, vac_data_2_cat])
    
    
    
    return gal_cat


def _fill_not_finite(arr, fill_value=99.0):
    return np.where(np.isfinite(arr), arr, fill_value)


def get_useful_cat_colms(catalog):
    '''
    This functions takes in photometric catalog and computes useful columns that will be needed in filtering stuff
    '''
    
    # Do galaxy/star separation
    catalog["is_galaxy"] = QueryMaker.not_equal("TYPE", "PSF").mask(catalog)
    
    catalog["is_galaxy"] |= QueryMaker.equal("REF_CAT", "L3").mask(catalog)
    

    # Rename/add columns

    for BAND in ("G", "R", "Z"):
        catalog[f"SIGMA_{BAND}"] = catalog[f"FLUX_{BAND}"] * np.sqrt(catalog[f"FLUX_IVAR_{BAND}"])
        catalog[f"SIGMA_GOOD_{BAND}"] = np.where(catalog[f"RCHISQ_{BAND}"] < 100, catalog[f"SIGMA_{BAND}"], 0.0)

    
    # Recalculate mag and err to fix old bugs in the raw catalogs
    ##these are MW transmission corrected magnitudes
    const = 2.5 / np.log(10)
    for band in "gr":
        BAND = band.upper()
        with np.errstate(divide="ignore", invalid="ignore"):
            catalog[f"{band}_mag"] = _fill_not_finite(
                22.5 - const * np.log(catalog[f"FLUX_{BAND}"] / catalog[f"MW_TRANSMISSION_{BAND}"])
            )
            
    ## now that we have added all the useful columns let us return this catalo
    return catalog



from mpl_toolkits.axes_grid1 import Divider, Size
import matplotlib
import matplotlib.pyplot as plt

def make_subplots(ncol = 3, nrow = 1, row_spacing = 1.1,col_spacing=1.1, label_font_size = 17,plot_size = 3,direction = "horizontal",return_fig=False):
    '''
    This function is my plotting function that returns me all the axes
    '''

    tot_len = int(12 + 5*col_spacing)
    tot_height = 6 * nrow
        
    fig = plt.figure(figsize=(tot_len, tot_height))
    
    h = []
    for i in range(ncol):
        h.append(Size.Fixed(col_spacing))
        h.append( Size.Fixed(plot_size))

    #then we end
    h.append(Size.Fixed(col_spacing))

    v = []
    for j in range(nrow):
        v.append(Size.Fixed(row_spacing))
        v.append(Size.Fixed(plot_size))
                     
    v.append(Size.Fixed(row_spacing))
    #this used to be 0.5 at start and end
    
    divider = Divider(fig, (0, 0, 1, 1), h, v, aspect=False)
    
    all_axes = []

    for i in range(nrow):
        for j in range(ncol):
            axi = fig.add_axes(
            divider.get_position(),
            axes_locator=divider.new_locator(nx=2*j + 1, ny=2*i + 1))

            all_axes.append(axi)

    if return_fig:
        return fig,all_axes
    else:
        return all_axes
    
def get_mags(vac_data_bgs, band="R"):
    vac_fluxr = vac_data_bgs["FLUX_" + band]
    vac_fluxr_err = np.sqrt(1/vac_data_bgs["FLUX_IVAR_" + band])
    vac_mwr = vac_data_bgs["MW_TRANSMISSION_" + band]
    
    #we have the magnitudes
    vac_rmag_bgs = 22.5 - 2.5*np.log10(vac_fluxr/vac_mwr)
    ### this formula for rmag uncertainties is from https://sites.astro.caltech.edu/~george/ay122/Ay122a_Photometry1.pdf pg 34
    ### it is an approximation that is valid in small error regime.
    vac_rmag_err_bgs = 1.087*(vac_fluxr_err/vac_fluxr)
    
    return vac_rmag_bgs, vac_rmag_err_bgs

def get_fibmag(vac_data_bgs):
    vac_fib_fluxr = vac_data_bgs["FIBERFLUX_R"]
    vac_fibrmag_bgs = 22.5 - 2.5*np.log10(vac_fib_fluxr)
    return vac_fibrmag_bgs


from easyquery import Query, QueryMaker

def _n_or_more_gt(cols, n, cut):
    def _n_or_more_gt_this(*arrays, n=n, cut=cut):
        return np.count_nonzero((np.stack(arrays) > cut), axis=0) >= n

    return Query((_n_or_more_gt_this,) + tuple(cols))

def _n_or_more_lt(cols, n, cut):
    def _n_or_more_lt_this(*arrays, n=n, cut=cut):
        return np.count_nonzero((np.stack(arrays) < cut), axis=0) >= n

    return Query((_n_or_more_lt_this,) + tuple(cols))

def _sigma_cut(bands, n, cut):
    return Query((_n_or_more_lt(n, cut), *(f"SIGMA_{b}" for b in bands)))


def get_remove_flag(catalog, remove_queries):
    """
    get remove flag by remove queries. remove_queries can be a list or dict.
    """

    try:
        iter_queries = iter(remove_queries.items())
    except AttributeError:
        iter_queries = enumerate(remove_queries)

    remove_flag = np.zeros(len(catalog), dtype=np.int)
    for i, remove_query in iter_queries:
        remove_flag[Query(remove_query).mask(catalog)] += 1 << i
    return remove_flag



def plot_mwd(RA,Dec,org=0,title='Mollweide projection', projection='mollweide'):
    ''' RA, Dec are arrays of the same length.
    RA takes values in [0,360), Dec in [-90,90],
    which represent angles in degrees.
    org is the origin of the plot, 0 or a multiple of 30 degrees in [0,360).
    title is the title of the figure.
    projection is the kind of projection: 'mollweide', 'aitoff', 'hammer', 'lambert'
    '''
    x = np.remainder(RA+360-org,360) # shift RA values
    ind = x>180
    x[ind] -=360    # scale conversion to [-180, 180]
    x=-x    # reverse the scale: East to the left
    tick_labels = np.array([150, 120, 90, 60, 30, 0, 330, 300, 270, 240, 210])
    tick_labels = np.remainder(tick_labels+360+org,360)
    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_subplot(111, projection=projection)
    ax.scatter(np.radians(x),np.radians(Dec),s=0.1,color = "grey",label = "DESI Y1 (Iron)")  # convert degrees to radians
    
    ax.set_xticklabels(tick_labels)     # we add the scale on the x axis
    ax.set_title(title)
    ax.title.set_fontsize(15)
    ax.set_xlabel("RA [deg]")
    ax.xaxis.label.set_fontsize(13)
    ax.set_ylabel("Dec [deg]")
    ax.yaxis.label.set_fontsize(13)
    ax.grid(True,ls = ":", color = "lightgrey")
    lgnd = ax.legend(frameon=False)
    lgnd.legendHandles[0]._sizes = [30]
    plt.show()
    
    
def get_isdesi(ra, dec):
    '''
    D
    '''
    hdu = fits.open('/global/cscratch1/sd/raichoor/desits/des_hpmask.fits')
    nside, nest = hdu[1].header['HPXNSIDE'], hdu[1].header['HPXNEST']
    hppix = hp.ang2pix(nside,(90.-dec)*np.pi/180.,ra*np.pi/180.,nest=nest)
    isdes = np.zeros(len(ra),dtype=bool)
    isdes[np.in1d(hppix, hdu[1].data['hppix'])] = True
    return isdes

# maybe this works for desi footprint
#this was found in some previous slack thread 
# from desimodel.footprint import is_point_in_desi
# # get the tiles
# t = Table.read("/global/cfs/cdirs/desi/survey/ops/surveyops/trunk/ops/tiles-main.ecsv")
# sel = (t["PROGRAM"] == "DARK") & (t["IN_DESI"])
# t = t[sel]
# len(t)
# # dummy ras, decs
# ras, decs = np.array([0, 0]), np.array([0, -40])
# is_point_in_desi(t, ras, decs)

def get_radec_mw(ra, dec, org):
    # convert radec for mollwide
    ra          = np.remainder(ra+360-org, 360) # shift ra values
    ra[ra>180] -= 360    # scale conversion to [-180, 180]
    ra          =- ra    # reverse the scale: East to the left
    
    return np.radians(ra),np.radians(dec)
    

def add_sweeps_column(data,save_path=None):
    all_sweeps = []
    all_ras = data["RA"]
    all_decs = data["DEC"]
    for i in trange(len(data)):
        all_sweeps.append( get_sweep_filename(  all_ras[i], all_decs[i]) )

    is_souths = is_target_in_south(all_ras,all_decs)  

    # data.remove_column("which_cat")
    data["SWEEP"] = all_sweeps
    data["is_south"] = is_souths.astype(int)
    #this is something that says north or south
    #save this now 
    if save_path is not None:
        save_table(data, save_path)

    return data

def get_galex_source(source_ra, source_dec, rgb_stuff, wcs, save_path):
    
    '''
    This function queries the GALEX catalog to find the nearby source
    '''

    query = """select g.ra, g.dec, g.mag_fuv, g.mag_nuv, g.FLUX95_RADIUS_NUV, g.FLUX95_RADIUS_FUV, g.semiminor, g.semimajor, g.posang
    from fGetNearbyObjEq(%s, %s,0.2) nb
    inner join gcat as g on nb.casjobsid = g.casjobsid
    where g.mag_nuv > 0 and g.mag_nuv > 0
    """%(source_ra,source_dec)
    
    # link where all the column names are: https://archive.stsci.edu/prepds/gcat/gcat_dataproducts.html
    # user is your MAST Casjobs username
    # pwd is your Casjobs password
    # These can also come from the CASJOBS_USERID and CASJOBS_PW environment variables,
    # in which case you do not need the username or password parameters.
    # Create a Casjobs account at <https://mastweb.stsci.edu/ps1casjobs/CreateAccount.aspx>
    #   if you do not already have one.
    
    user = "virajmanwadkar"
    pwd = "dwarfs"
    
    jobs = mastcasjobs.MastCasJobs(username=user, password=pwd, context="GALEX_Catalogs")
    results = jobs.quick(query, task_name="python cone search")

    save_table(results, save_path + "/galex_source_match.fits")


    tempx,tempy,_ = wcs.all_world2pix(source_ra, source_dec,0,1)

    fig,ax = plt.subplots(1,1,figsize = (7,7))
    ax.imshow(rgb_stuff,origin="lower")


    #results contains all the nearby GALEX sources
    if len(results) == 0:
        ##what happens if no GALEX sources are found???
        #make the aperture center the location of our fiber 
        aper_xcen, aper_ycen,_ = wcs.all_world2pix(source_ra, source_dec,0,1)
        aper_loc = np.array([int(aper_xcen), int(aper_ycen)])
        #the aperture radius is set by the size of the main segment    
    else:
        ## let us visualize this query 
        all_locs = wcs.all_world2pix(results["ra"].data, results["dec"].data,0,1)
        

        for i in range(len(all_locs[0])):
            # Define circle parameters
            galex_center = (all_locs[0][i], all_locs[1][i])  # (x, y) center of the circle
            semi_major = float(results["semimajor"][i])    # Semi-major axis length
            semi_minor = float(results["semiminor"][i])    # Semi-minor axis length
            angle = 90 + float(results["posang"][i])         # Position angle in degrees (counterclockwise from the x-axis)
        
            ellipse = patches.Ellipse(galex_center, 2*semi_major, 2*semi_minor, angle=angle, 
                                       edgecolor='hotpink', facecolor='none', linewidth=2,linestyle ="--")
            ax.add_patch(ellipse)
        
        ## for the aperture itself, choose the closet GALEX source 
        ##if there are more than 1 sources in the field, find closest one
        if len(results) > 1:
            #find difference between input position and all GALEX sources
            ref_coord = SkyCoord(ra=source_ra * u.deg, dec=source_dec * u.deg)
            catalog_coords = SkyCoord(ra=results["ra"] * u.deg, dec=results["dec"] * u.deg)
        
            # Compute separations
            separations = ref_coord.separation(catalog_coords).arcsec
            aper_loc = [  all_locs[0][np.argmin(separations)]  , all_locs[1][np.argmin(separations)]  ]
            # aper_rad = scale*float(results["FLUX95_RADIUS_NUV"][np.argmin(separations)])
        else:
            aper_loc = all_locs    

        ##saving the GRZ image plus galex source overlayed
    plt.savefig(save_path + "/galex_source_overlay_rgb.png")
    plt.close()

    # save the location of the aperture in the file
    np.save( save_path + "/aperture_cen_coord.npy", aper_loc )
        
    return


def fetch_psf(ra, dec, session,timeout=30):
    """
    Returns PSFs in dictionary with keys 'g', 'r', and 'z'.

    Sometimes the the psf can have a=north in it and stuff.... 
    use try-except clause in addition to for-loop to try to figure this out
    """
    url_prefix = 'https://www.legacysurvey.org/viewer/'
    all_layers = ["ls-dr9", "ls-dr9-north"]

    for i in range(2):
        try:
            # url = url_prefix + f'coadd-psf/?ra={ra}&dec={dec}&layer=ls-dr9&bands=grz'
            url = url_prefix + f'coadd-psf/?ra={ra}&dec={dec}&layer={all_layers[i]}&bands=grz'
            
            # session = requests.Session()
            print(url)
            resp = session.get(url,timeout=timeout)
            resp.raise_for_status()  # Raise error for bad status codes
            # resp.raise_for_status()  # Raise error for bad responses
            hdulist = fits.open(BytesIO(resp.content))
            psf = {'grz'[i]: hdulist[i].data for i in range(3)}

            return psf
        except:
            print("URL failed, trying another if not already ... ")
            
    return None


def save_subimage(ra, dec, sbimg_path, session, size = 350, timeout = 30):
    """
    Returns coadds in dictionary with keys 'g', 'r', and 'z'.
    """
    url_prefix = 'https://www.legacysurvey.org/viewer/'
    url = url_prefix + f'cutout.fits?ra={ra}&dec={dec}&'
    
    url += 'layer=ls-dr9&size=%d&pixscale=0.262&subimage'%size
    print(url)
    try:
        resp = session.get(url, timeout=timeout)
        resp.raise_for_status()  # Raise error for bad status codes
        # Save the FITS file
        with open(sbimg_path, "wb") as f:
            f.write(resp.content)
    except:
        print("getting sub image data failed!")
        
    return
    

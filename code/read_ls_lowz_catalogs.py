'''
In this script, we query the DR9 photometric catalogs, apply the LOW-Z cuts and then save them
so I can cross match them with the observed catalogs to get the true photometry
'''

from glob import glob
import numpy as np
from os import path
import pandas as pd
from glob import glob
from easyquery import Query, QueryMaker
from astropy.io import fits
from astropy.table import join
from astropy.table import Table, vstack, hstack
from tqdm import trange, tqdm
import astropy.units as u
from astropy.coordinates import SkyCoord
import matplotlib.pyplot as plt
from astropy.cosmology import Planck18 
import multiprocessing as mp
from desi_lowz_funcs import calc_normalized_dist_broadcast, calc_normalized_dist

#load sga catalog for matching 
#this is the most up-to-date version of this catalog 
siena_path="/global/cfs/cdirs/cosmo/data/sga/2020/SGA-2020.fits"
sga_catalog = fits.open(siena_path)
sga_catalog = sga_catalog[1].data

#convert flux to magnitude in band
def convert_mag(band,transmission):
    const = 2.5 / np.log(10)
    d = _fill_not_finite(22.5 - const * np.log(band /transmission))
    return d

#convert flux error to magnitude error in band
def convert_err(band, band_err,transmission):
    sigma = band*np.sqrt(band_err)
    const =  2.5 / np.log(10)
    d = _fill_not_finite(const / np.abs(sigma))
    return d

    
def filter_nearby_object(catalog, host, radius_deg=1.001, remove_coord=True):
    if catalog is not None:
        catalog = build2.filter_nearby_object(catalog, host, radius_deg, remove_coord)
        if len(catalog):
            return catalog

        
def _fill_not_finite(arr, fill_value=99.0):
    return np.where(np.isfinite(arr), arr, fill_value)

def _ivar2err(ivar):
    with np.errstate(divide="ignore"):
        return 1.0 / np.sqrt(ivar)

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

    remove_flag = np.zeros(len(catalog), dtype=np.int64)
    for i, remove_query in iter_queries:
        remove_flag[Query(remove_query).mask(catalog)] += 1 << i
    return remove_flag


def flux_cut(catalog):
    '''
    # Remove objects fainter than r = 23 (unless g or z < 22.5)
    # flux = 10 ** ((22.5 - mag) / 2.5)
    catalog = (
        Query("FLUX_R >= MW_TRANSMISSION_R * 0.63")  # r <= 23
        | Query("FLUX_G >= MW_TRANSMISSION_G")  # g <= 22.5
        | Query("FLUX_Z >= MW_TRANSMISSION_Z")  # z <= 22.5
    ).filter(catalog)
    '''
    
    return (
        (catalog["FLUX_R"] >= catalog["MW_TRANSMISSION_R"] * 0.63) |
        (catalog["FLUX_G"] >= catalog["MW_TRANSMISSION_G"]) |
        (catalog["FLUX_Z"] >= catalog["MW_TRANSMISSION_Z"])
    )

    
def get_data(file_ind, sweep_file,survey = "north"):
    '''
    This function loads in the sweep file and obtains catalog that satisfies LOW-Z cuts.

    Note that even though the magnitudes are extinction corrected, the flux values are not

    '''

    catalog = Table.read(sweep_file, format="fits", memmap=True)

    print(f"Reading {sweep_file}")

    # Remove duplicated Gaia entries
    # catalog = QueryMaker.not_equal("TYPE", "DUP").filter(catalog)
    catalog = catalog[catalog["TYPE"] != "DUP"]

    catalog = catalog[ flux_cut(catalog) ]

    # Do galaxy/star separation
    catalog["is_galaxy"] = QueryMaker.not_equal("TYPE", "PSF").mask(catalog)

    # Bright (r < 17) stars that are misclassified as galaxies
    flux_limit = 10 ** ((22.5 - 17) / 2.5)
    pmra_sig = np.abs(catalog["PMRA"] * np.sqrt(catalog["PMRA_IVAR"])) >= 2
    pmdec_sig = np.abs(catalog["PMDEC"] * np.sqrt(catalog["PMDEC_IVAR"])) >= 2

    bright_stars = (
        (catalog["SHAPE_R"] < 1) &
        (catalog["FLUX_R"] >= catalog["MW_TRANSMISSION_R"] * flux_limit) &
        (pmra_sig | pmdec_sig)
    )

    # bright_stars = Query(
    #     "SHAPE_R < 1",
    #     "FLUX_R >= MW_TRANSMISSION_R * {}".format(flux_limit),
    #     (Query("abs(PMRA * sqrt(PMRA_IVAR)) >= 2") | Query("abs(PMDEC * sqrt(PMDEC_IVAR)) >= 2")),
    # ).mask(catalog)

    # Fix galaxy/star separation with bright_stars and SGA masks
    catalog["is_galaxy"] &= ~bright_stars

    # Rename/add columns
    catalog["radius"] = catalog["SHAPE_R"]
    catalog["radius_err"] = _fill_not_finite(_ivar2err(catalog["SHAPE_R_IVAR"]), 9999.0)
    e_abs = np.hypot(catalog["SHAPE_E1"], catalog["SHAPE_E2"])
    catalog["ba"] = (1 - e_abs) / (1 + e_abs)
    catalog["phi"] = np.rad2deg(np.arctan2(catalog["SHAPE_E2"], catalog["SHAPE_E1"]) * 0.5)
    del e_abs

    for BAND in ("G", "R", "Z", "W1", "W2", "W3", "W4"):
        catalog[f"SIGMA_{BAND}"] = catalog[f"FLUX_{BAND}"] * np.sqrt(catalog[f"FLUX_IVAR_{BAND}"])
        catalog[f"SIGMA_GOOD_{BAND}"] = np.where(catalog[f"RCHISQ_{BAND}"] < 100, catalog[f"SIGMA_{BAND}"], 0.0)

    # Recalculate mag and err to fix old bugs in the raw catalogs
    ##these are MW transmission corrected magnitudes
    const = 2.5 / np.log(10)
    for band in "grz":
        BAND = band.upper()
        with np.errstate(divide="ignore", invalid="ignore"):
            catalog[f"{band}_mag"] = _fill_not_finite(
                22.5 - const * np.log(catalog[f"FLUX_{BAND}"] / catalog[f"MW_TRANSMISSION_{BAND}"])
            )
            catalog[f"{band}_err"] = _fill_not_finite(const / np.abs(catalog[f"SIGMA_{BAND}"]))

            
    for band in "r":
        BAND = band.upper()
        with np.errstate(divide="ignore", invalid="ignore"):
            catalog[f"{band}_fib_mag"] = _fill_not_finite(
                22.5 - const * np.log(catalog[f"FIBERFLUX_{BAND}"] / catalog[f"MW_TRANSMISSION_{BAND}"])
            )
            
     

    allmask_grz = [f"ALLMASK_{b}" for b in "GRZ"]
    sigma_grz = [f"SIGMA_GOOD_{b}" for b in "GRZ"]
    sigma_wise = [f"SIGMA_GOOD_W{b}" for b in range(1, 5)]
    fracflux_grz = [f"FRACFLUX_{b}" for b in "GRZ"]
    rchisq_grz = [f"RCHISQ_{b}" for b in "GRZ"]

    fracmasked_grz = [f"FRACMASKED_{b}" for b in "GRZ"]

    to_remove= None
    
    #need to confirm that the below cleaning cuts are accurate

    remove_queries = [
        "(MASKBITS >> 1) % 2 > 0",  # 1
        "(MASKBITS >> 5) % 2 > 0",  # 2
        "(MASKBITS >> 6) % 2 > 0",  # 3
        "(MASKBITS >> 7) % 2 > 0",  # 4
        "(MASKBITS >> 12) % 2 > 0",  # 5
        "(MASKBITS >> 13) % 2 > 0",  # 6
        _n_or_more_lt(sigma_grz, 2, 5),  # 7
        Query(_n_or_more_gt(fracflux_grz, 2, 0.35)),  # 8
        Query(_n_or_more_gt(rchisq_grz, 2, 2)),  # 9
        "g_mag - r_mag <= -0.1",  # 10        
    ]
        
    mask = get_remove_flag(catalog, remove_queries) == 0
    catalog = catalog[mask]

    #Remove SGA objects
    catalog["is_galaxy"] |= (catalog["REF_CAT"] == "L3")
    # catalog["is_galaxy"] |= QueryMaker.equal("REF_CAT", "L3").mask(catalog)

    # apply the bright galaxy cuts here     
    
    chisq_mask = catalog['RCHISQ_R'] < 20
    r_mask = catalog["r_mag"]  < 16

    objects2 = catalog[chisq_mask & r_mask]

    print(f"Number before bright galaxy cleaning = {len(catalog)}")

    if len(objects2) != 0:
        nearest_inds, nearest_dists = calc_normalized_dist_broadcast(
                                        ra_list = np.array(catalog["RA"]),
                                        dec_list = np.array(catalog["DEC"]),
                                        redshift_list = np.zeros_like(catalog["RA"]),  # or z=0 if you're not using redshift
                                        s20_ra = np.array(objects2["RA"]),
                                        s20_dec = np.array(objects2["DEC"]),
                                        s20_zred = np.zeros_like(objects2["RA"]),  # again, redshift is unused here?
                                        s20_d26 = np.array(objects2["radius"]),
                                        s20_ba = np.array(objects2["ba"]),
                                        s20_phi = np.array(objects2["phi"]),
                                        verbose = False,
                                        use_redshift = False)

        mask = nearest_dists > 4.0  # Keep only objects farther than 4 × R
        catalog = catalog[mask]


    print(f"Number after bright galaxy cleaning = {len(catalog)}")

    #only consider objects above certain magnitude range
    ##applying the basic magnitudes cuts
    #no fiber mag cut
    # mask1 = catalog['r_fib_mag'] < 23.5

    if survey == "north":
        catalog["r_mag"] = catalog["r_mag"] + 0.04
    if survey == "south":
        catalog["r_mag"] = catalog["r_mag"]
        
    mask2 = catalog['r_mag'] < 21.15   
    mask3 = catalog['r_mag'] > 19   
    
    mask = mask2&mask3 #&mask1
    catalog = catalog[mask]
    
    #calculate surface brightness (mao et al. 2020) 
    catalog["mu_r"] = catalog["r_mag"] + 2.5*np.log10(2*np.pi*catalog["radius"]*catalog["radius"])
    catalog["mu_r_err"] = np.hypot(catalog["radius_err"], (5/np.log(10)) * catalog["radius_err"]/catalog["radius"])
    
    mask1 = ((catalog['g_mag'] - catalog['r_mag'])-np.sqrt(catalog['g_err']**2+catalog['r_err']**2)+0.06*(catalog['r_mag']-14) < .99)
    
    mur_vals = catalog['mu_r'] + catalog['mu_r_err'] - 0.7*(catalog['r_mag']-14)
    mask2 = (mur_vals > 16.8)
    mask = mask1 & mask2
    
    catalog = catalog[mask]
    
    ## now we apply the SGA masks

    if len(catalog) > 0:
        
        #get the ra, dec ranges of this sweep file
        #we extend the range by a bit to give some leeway for edge objects
        ra_min = np.min(catalog["RA"]) - 0.1
        ra_max = np.max(catalog["RA"]) + 0.1
        dec_min = np.min(catalog["DEC"]) - 0.1
        dec_max = np.min(catalog["DEC"]) + 0.1
        
        #these objects are SGA catalog objects that are within this sweep file!
        objects = sga_catalog[(sga_catalog["RA"] < ra_max) & (sga_catalog["RA"] > ra_min) & (sga_catalog["DEC"] < dec_max) & (sga_catalog["DEC"] > dec_min) ]
    
        print(f"Number of SGA objects = {len(objects)}")
        
        if(len(objects)!= 0):
            for obj_n in range(len(objects)):
                
                ##we need to use the best radius here. According to Yao, it is 
                objects_rad = np.minimum(objects["SMA_MOMENT"][obj_n] * 1.3333, objects["D26"][obj_n] * 30.0)
               #this is diameter and hence we multiply 60/2 = 30
                
                dist = calc_normalized_dist(catalog["RA"], catalog["DEC"], objects["RA_LEDA"][obj_n], objects["DEC_LEDA"][obj_n],
                                            objects_rad, cen_ba=objects['BA'][obj_n], cen_phi=objects['PA'][obj_n])   
                                
                #above D26 is being converted into arcseconds ... 
                #the calc_normalized distance gets distance in units of half-light radius maybe
                #in the original targeting paper the radius was D25 and 1.5 times. 
                
                nearby_obj_mask = dist > 1.
                catalog = catalog[nearby_obj_mask]
    
                del nearby_obj_mask


    print(f"Total number of LOWZ targets in sweep = {len(catalog)}")
    ##save this catalog and we will combine it later!!
    catalog.write(f"/pscratch/sd/v/virajvm/target/{survey}/file_{file_ind}.fits",overwrite=True)
    
    return


def above_dec(file_name,dec_limit = -20):
    '''
    This function finds the sweep files that lie above a certain dec limit 
    '''
    
    # print(file_name) 
    
    dec1 = file_name[-9:-5]
    if "m" in dec1:
        dec1_val = -1*int(dec1[1:])
    if "p" in dec1:
        dec1_val = int(dec1[1:])
        
    dec2 = file_name[-17:-13]
    
    if "m" in dec2:
        dec2_val = -1*int(dec2[1:])
    if "p" in dec2:
        dec2_val = int(dec2[1:])
    
    #dec range is 
    dec_min =  np.minimum(dec1_val, dec2_val)
    dec_max = np.maximum(dec1_val, dec2_val)
    
    # print(dec_min, dec_max)
    
    if dec_max < dec_limit:
        return False
    else:
        return True


# Define at top-level
def get_data_helper(args):
    ind, sweep_file, survey = args
    return get_data(ind, sweep_file, survey=survey)

def run_main(file_dir, survey, save_path,n_proc = 16):
    '''
    This is the function that makes these files:

    north_targs = Table.read("/pscratch/sd/v/virajvm/target/dr9_north_lowz_targets_no_rfib_cut.fits")
    south_targs = Table.read("/pscratch/sd/v/virajvm/target/dr9_south_lowz_targets_no_rfib_cut_dec20.fits")

    '''
    ##get the sweep files 
    sweep_files = glob(file_dir + '/sweep-*.fits')

    print(f"Original number of sweep files to read = {len(sweep_files)}")

    if survey == "south":
        sweep_files_f = []
        for si in sweep_files:
            temp = above_dec(si,dec_limit = -20)
            if temp:
                sweep_files_f.append(si)

        sweep_files = sweep_files_f

    print(f"Number of sweep files to read = {len(sweep_files)}")
    
    sweep_args = [(ind, sf, survey) for ind,sf in enumerate(sweep_files)]

    # Start multiprocessing pool
    with mp.Pool(processes=n_proc) as pool:
        list(tqdm(pool.imap(get_data_helper, sweep_args), total=len(sweep_args)))

    #we then loop through all the save files, read them and then 
    all_cat_files = glob(f"/pscratch/sd/v/virajvm/target/{survey}/*.fits")

    print(f"Found {len(all_cat_files)} sweep files!")

    results = []
    for ci in all_cat_files:
        results.append( Table.read(ci) )
        
    # Stack all valid catalogs
    cat_table = vstack(results)

    print(f"Total number of LOWZ targets in {survey} = {len(cat_table)}")

    # Save the final table
    cat_table.write(save_path, format="fits", overwrite=True)

if __name__ == '__main__':    

    from astropy.units import UnitsWarning
    import warnings
    warnings.filterwarnings('ignore', category=UnitsWarning)

    south_file_dir = '/global/cfs/cdirs/cosmo/data/legacysurvey/dr9/south/sweep/9.0'
    north_file_dir = '/global/cfs/cdirs/cosmo/data/legacysurvey/dr9/north/sweep/9.0'

    save_north_file = "/pscratch/sd/v/virajvm/target/dr9_north_lowz_targets_no_rfib_cut.fits"
    save_south_file = "/pscratch/sd/v/virajvm/target/dr9_south_lowz_targets_no_rfib_cut_dec20.fits"

    run_main(south_file_dir, "south", save_south_file,n_proc = 8)
    run_main(north_file_dir, "north", save_north_file,n_proc = 8)
    
    #for north or south do we do the 0.04 mag offset?
    #NORTH is BASS, south is decals
    #rbass - rdecals = 0.04
    #if we see some 18.99 objects in the targeting
    #we will add 0.04 to all bass magnitudes
    
   
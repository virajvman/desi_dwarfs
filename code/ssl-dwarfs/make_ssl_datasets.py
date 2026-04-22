import os
import h5py
import numpy as np
from astropy.table import Table
from astropy.io import fits
from tqdm import trange
from tqdm import tqdm
import multiprocessing as mp
from astropy.table import vstack, unique
from astropy.wcs import WCS
import glob
import sys

from ssl_h5_catalog_sync import (
    matching_image_paths,
    next_data_chunk_index,
    run_prune_and_need_image_report,
    scan_h5_targetid_union,
)

rootdir = '/global/u1/v/virajvm/'
sys.path.append(os.path.join(rootdir, 'DESI2_LOWZ/desi_dwarfs/code'))

from desi_lowz_funcs import process_img_shift


def read_fits(path):
    '''
    Function that reads the image fits file and crops to center 152x152 pixels
    '''
    try:
        with fits.open(path, memmap=False) as img:
            img_data = img[0].data
            size = 152
            start = (img_data.shape[1] - size) // 2  # assumes square images
            end = start + size
            img_data = img_data[:, start:end, start:end]
            return img_data
    except Exception as e:
        print(f"Failed reading {path}: {e}")
        return np.zeros((3,152,152))

def process_shred_imgs(tot_cat, i):
    '''
    Function that returns the img_data for the shredded data after properly re-centering the galaxy!
    '''

    with fits.open(tot_cat["IMAGE_PATH"][i], memmap=False) as hdul:
        org_img = hdul[0].data
    
    wcs = WCS(fits.getheader( tot_cat["IMAGE_PATH"][i] ))
    
    #need to recenter the image
    #compute the shift between target_pos and 
    ra_cen_i = tot_cat["RA"][i]
    dec_cen_i = tot_cat["DEC"][i]
    ra_tgt_i = tot_cat["RA_TARGET"][i]
    dec_tgt_i = tot_cat["DEC_TARGET"][i]
    
    x_cen, y_cen,_ =   wcs.all_world2pix(ra_cen_i, dec_cen_i, 0,1)
    x_tgt, y_tgt,_ = wcs.all_world2pix(ra_tgt_i, dec_tgt_i,0, 1)
    
    x_shift = int(np.round(x_cen - x_tgt))
    y_shift = int(np.round(y_cen - y_tgt))

    # _, reconst_img  = process_img_shift(reconst_img, cutout_size=152, org_size=np.shape(reconst_img)[1], 
    #                                     return_shift=False, x_shift=x_shift, y_shift=y_shift)

    _, org_img  = process_img_shift(org_img, cutout_size=152, org_size=np.shape(org_img)[1], 
                                        return_shift=False, x_shift=x_shift, y_shift=y_shift)

    return org_img



def make_dataset_clean_chunk(h5_file_path, catalog, image_data_list, count_low, count_hi):
    '''
    Writing the .h5 file for a single chunk, removing zero images
    '''

    pixel = 152
    batch_size = 500

    # --- Slice chunk ---
    cat_chunk = catalog[count_low:count_hi]
    img_chunk = image_data_list  # already corresponds to this chunk

    if len(img_chunk) != len(cat_chunk):
        raise ValueError(f"Sizes not maching for {h5_file_path}")

    # --- Identify non-zero images ---
    # shape: (N,)
    valid_mask = np.array([
        np.any(img != 0) for img in img_chunk
    ])

    n_valid = np.sum(valid_mask)
    print(f"Total in chunk = {len(img_chunk)}, valid images = {n_valid}")

    if n_valid == 0:
        print("WARNING: no valid images in this chunk, skipping write.")
        return

    # --- Filter everything ---
    img_chunk = img_chunk[valid_mask]

    mag_g  = cat_chunk['MAG_G'][valid_mask]
    mag_r  = cat_chunk['MAG_R'][valid_mask]
    mag_z  = cat_chunk['MAG_Z'][valid_mask]
    logm   = cat_chunk['LOG_MSTAR_M24'][valid_mask]
    star_d = cat_chunk['STARDIST_DEG'][valid_mask]
    zred   = cat_chunk['Z'][valid_mask]
    tgid   = cat_chunk['TARGETID'][valid_mask]

    # --- Write HDF5 ---
    print(f"Writing to HDF5: {h5_file_path}")
    with h5py.File(h5_file_path, 'w') as f:

        images = f.create_dataset(
            'images',
            shape=(n_valid, 3, pixel, pixel),
            dtype=np.float32,
            chunks=(1, 3, pixel, pixel),
            compression="gzip",
            compression_opts=4
        )

        f.create_dataset('mag_g', data=mag_g.astype(np.float32))
        f.create_dataset('mag_r', data=mag_r.astype(np.float32))
        f.create_dataset('mag_z', data=mag_z.astype(np.float32))
        f.create_dataset('mstar', data=logm.astype(np.float32))
        f.create_dataset('star_dist', data=star_d.astype(np.float32))
        f.create_dataset('redshift', data=zred.astype(np.float32))
        f.create_dataset('targetid', data=tgid.astype(np.int64))

        # batch write images only (metadata already written)
        for i in trange(0, n_valid, batch_size, desc="Writing batches"):
            j = min(i + batch_size, n_valid)
            images[i:j] = img_chunk[i:j]

    print(f"Wrote {n_valid} clean objects to {h5_file_path}")




def make_dataset_shred_chunk(h5_file_path, all_imgs=None, all_tgids=None, all_zreds=None, all_gmags=None, all_rmags=None, all_zmags=None, all_mstar=None, all_stardist=None, img_type=None):
    """
    Write a chunk of data to an HDF5 file.
    """
    pixel = 152
    tot_count = len(all_tgids)

    print(f"Total Count in this chunk = {tot_count}")
    print(f"Writing to HDF5: {h5_file_path}")

    with h5py.File(h5_file_path, 'w') as f:

        # Create datasets
        images = f.create_dataset(
            'images', (tot_count, 3, pixel, pixel),
            dtype='float32',
            chunks=(1, 3, pixel, pixel),
            compression="gzip", compression_opts=4,
        )
        
        d_mag_g = f.create_dataset('mag_g', (tot_count,), dtype='float32')
        d_mag_r = f.create_dataset('mag_r', (tot_count,), dtype='float32')
        d_mag_z = f.create_dataset('mag_z', (tot_count,), dtype='float32')
        
        d_mstar = f.create_dataset('mstar', (tot_count,), dtype='float32')
        d_stardist = f.create_dataset('star_dist', (tot_count,), dtype='float32')
        d_zred  = f.create_dataset('redshift', (tot_count,), dtype='float32')
        
        d_tgid  = f.create_dataset('targetid', (tot_count,), dtype='int64')

        # Correct: assign data
        images[:] = all_imgs
        
        d_mag_g[:]  = all_gmags
        d_mag_r[:]  = all_rmags
        d_mag_z[:]  = all_zmags
        
        d_mstar[:]  = all_mstar
        d_zred[:]  = all_zreds  
        d_stardist[:]   = all_stardist
        
        d_tgid[:]   = all_tgids

    print(all_tgids[:3])

    # Optional sanity check
    with h5py.File(h5_file_path, 'r') as f:
        print("Datasets in file:")
        for key in f.keys():
            print(f"{key}: shape={f[key].shape}, dtype={f[key].dtype}")
        print("\nSample values:")
        print("targetid[0] =", f['targetid'][0])
        print("images[0] shape =", f['images'][0].shape)



if __name__ == '__main__':

    # Image cutout generation differs for TRACTOR_OG vs updated photometry (shreds): cutout center differs.
    # Sync workflow: optional prune H5s vs catalog, write ssl_need_image_download.fits, fetch cutouts,
    # then append_new_h5_chunks to add only missing TARGETIDs without overwriting existing data_chunk_*.h5.

    h5_dir = "/pscratch/sd/v/virajvm/catalog_dr1_dwarfs/ssl_shred_data/h5_datasets"
    h5_glob = os.path.join(h5_dir, "data_chunk_*.h5")
    # Prune: drop H5 rows not in the current (filtered) catalog. Export FITS of catalog rows that are
    # missing from H5 and whose cutout file is not on disk.
    prune_h5_against_catalog = False
    write_need_image_download_fits = False
    need_image_download_fits = "/pscratch/sd/v/virajvm/catalog_dr1_dwarfs/ssl_shred_data/ssl_need_image_download.fits"
    prune_dry_run = False
    # Append-only: new data_chunk_N.h5 with N = 1+max existing index; never overwrites. Mutually
    # exclusive with full create_from_scratch below; use when cutouts for new objects exist.
    append_new_h5_chunks = True
    # Full (re)build from chunk 0 — ignored if append_new_h5_chunks is True
    create_clean_chunks = False
    create_shred_chunks = False

    filename = "/pscratch/sd/v/virajvm/desi_dwarf_catalogs/dr1/v1.0/desi_dr1_dwarf_catalog.fits"

    # load the MAIN extension directly as an Astropy Table
    data_cat = Table.read(filename, hdu="MAIN")

    print(f"Size of total catalog = {len(data_cat)}")

    #as that is the regime the data is trained on! We also only want unique dwarfs so that we do not just match to same dwarf!
    data_cat = data_cat[(data_cat["MAG_Z"] < 20) & (data_cat["DWARF_PRIMARY"] == 1)]

    print(f"Size of catalog at z<20 = {len(data_cat)}")

    #split by clean and not clean!!
    data_cat_og = data_cat[data_cat["PHOTOMETRY_UPDATED"] == False]
    data_cat_shreds = data_cat[data_cat["PHOTOMETRY_UPDATED"] == True]

    print(f"Size of OG catalog at z<20 = {len(data_cat_og)}")
    print(f"Size of shredded catalog at z<20 = {len(data_cat_shreds)}")
    
    #now we need the image paths of these objects!!
    temp_clean = Table.read("/pscratch/sd/v/virajvm/catalog_dr1_dwarfs/desi_y1_dwarf_clean_catalog_v5.fits")["TARGETID","IMAGE_PATH","STARDIST_DEG"]
    temp_shred = Table.read("/pscratch/sd/v/virajvm/catalog_dr1_dwarfs/desi_y1_dwarf_shreds_catalog_v5.fits")["TARGETID","IMAGE_PATH","STARDIST_DEG"]
    temp_sga = Table.read("/pscratch/sd/v/virajvm/catalog_dr1_dwarfs/iron_desi_SGA_matched_dwarfs_REPROCESS_V2.fits")["TARGETID","IMAGE_PATH","STARDIST_DEG"]
    
    # Combine and remove duplicate TARGETIDs
    tot_temp = vstack([temp_clean, temp_shred, temp_sga])
    tot_temp = unique(tot_temp, keys="TARGETID")

    if prune_h5_against_catalog or write_need_image_download_fits:
        run_prune_and_need_image_report(
            h5_glob,
            data_cat,
            tot_temp,
            need_image_fits_path=need_image_download_fits
            if (write_need_image_download_fits and need_image_download_fits)
            else "",
            do_prune=prune_h5_against_catalog,
            write_need_fits=write_need_image_download_fits,
            dry_run=prune_dry_run,
        )

    if append_new_h5_chunks:
        if create_clean_chunks or create_shred_chunks:
            print(
                "append_new_h5_chunks: skipping full create_clean_chunks / create_shred_chunks "
                "from scratch; only appending rows missing from existing H5s."
            )
        in_h5 = scan_h5_targetid_union(h5_glob)
        in_arr = (
            np.fromiter(in_h5, dtype=np.int64)
            if in_h5
            else np.array([], dtype=np.int64)
        )
        chunk_num = next_data_chunk_index(h5_glob)
        # --- append clean (PHOTOMETRY_UPDATED == False) ---
        data_cat_og_m, _ = matching_image_paths(data_cat_og, tot_temp)
        to_og = data_cat_og_m[
            ~np.isin(data_cat_og_m["TARGETID"], in_arr)
        ]
        if len(to_og):
            on_disk = np.array(
                [os.path.isfile(p) for p in to_og["IMAGE_PATH"]], dtype=bool
            )
            to_og = to_og[on_disk]
        if len(to_og):
            print(f"Appending {len(to_og)} clean (OG) rows not yet in H5 (images on disk).")
            chunk_size = 2000
            for count_low in trange(0, len(to_og), chunk_size):
                count_hi = min(count_low + chunk_size, len(to_og))
                print(count_low, count_hi)
                sub = to_og[count_low:count_hi]
                image_paths = sub["IMAGE_PATH"]
                h5_file_path = os.path.join(
                    h5_dir, f"data_chunk_{chunk_num}.h5"
                )
                if os.path.isfile(h5_file_path):
                    raise FileExistsError(
                        f"refuse to overwrite in append mode: {h5_file_path}"
                    )
                num_workers = min(8, mp.cpu_count())
                with mp.Pool(processes=num_workers) as pool:
                    img_data_list = list(
                        tqdm(
                            pool.imap(read_fits, image_paths),
                            total=len(image_paths),
                        )
                    )
                print(np.shape(img_data_list))
                img_data_list = np.array(img_data_list, dtype=np.float32)
                make_dataset_clean_chunk(
                    h5_file_path,
                    sub,
                    img_data_list,
                    count_low=0,
                    count_hi=len(sub),
                )
                chunk_num += 1
        in_h5 = scan_h5_targetid_union(h5_glob)
        in_arr = (
            np.fromiter(in_h5, dtype=np.int64)
            if in_h5
            else np.array([], dtype=np.int64)
        )
        # --- append shreds (after clean so chunk indices continue) ---
        to_sh, _ = matching_image_paths(data_cat_shreds, tot_temp)
        to_sh = to_sh[~np.isin(to_sh["TARGETID"], in_arr)]
        if len(to_sh):
            on_disk = np.array(
                [os.path.isfile(p) for p in to_sh["IMAGE_PATH"]], dtype=bool
            )
            to_sh = to_sh[on_disk]
        if len(to_sh):
            print(f"Appending {len(to_sh)} shred rows not yet in H5 (images on disk).")
            all_org_imgs = []
            all_gmag = []
            all_rmag = []
            all_zmag = []
            all_mstar = []
            all_stardist = []
            all_zred = []
            all_tgid = []
            for i in trange(len(to_sh)):
                org_img = process_shred_imgs(to_sh, i)
                if np.shape(org_img) != (3, 152, 152):
                    print(np.shape(org_img))
                    print("--")
                else:
                    all_org_imgs.append(org_img.astype(np.float32))
                    all_tgid.append(to_sh["TARGETID"][i])
                    all_zred.append(to_sh["Z"][i])
                    all_mstar.append(to_sh["LOG_MSTAR_M24"][i])
                    all_stardist.append(to_sh["STARDIST_DEG"][i])
                    all_gmag.append(to_sh["MAG_G"][i])
                    all_rmag.append(to_sh["MAG_R"][i])
                    all_zmag.append(to_sh["MAG_Z"][i])
            chunk_size = 2000
            for i in trange(0, len(all_org_imgs), chunk_size):
                h5_org_path = os.path.join(
                    h5_dir, f"data_chunk_{chunk_num}.h5"
                )
                if os.path.isfile(h5_org_path):
                    raise FileExistsError(
                        f"refuse to overwrite in append mode: {h5_org_path}"
                    )
                assert len(all_org_imgs[i : i + chunk_size]) == len(
                    all_tgid[i : i + chunk_size]
                )
                make_dataset_shred_chunk(
                    h5_org_path,
                    all_imgs=all_org_imgs[i : i + chunk_size],
                    all_tgids=all_tgid[i : i + chunk_size],
                    all_zreds=all_zred[i : i + chunk_size],
                    all_gmags=all_gmag[i : i + chunk_size],
                    all_rmags=all_rmag[i : i + chunk_size],
                    all_zmags=all_zmag[i : i + chunk_size],
                    all_mstar=all_mstar[i : i + chunk_size],
                    all_stardist=all_stardist[i : i + chunk_size],
                    img_type="org",
                )
                print(
                    f"Wrote chunk {chunk_num} with indices [{i}:{i + chunk_size}]"
                )
                chunk_num += 1
    elif create_clean_chunks:
        print("Creating chunks for the clean catalog subset!!")
        data_cat_og, _ = matching_image_paths(data_cat_og, tot_temp)
        chunk_size = 2000
        chunk_num = 0
        for count_low in trange(0, len(data_cat_og), chunk_size):
            count_hi = min(count_low + chunk_size, len(data_cat_og))
            print(count_low, count_hi)
            image_paths = data_cat_og["IMAGE_PATH"][count_low:count_hi]
            num_workers = min(8, mp.cpu_count())
            with mp.Pool(processes=num_workers) as pool:
                img_data_list = list(
                    tqdm(
                        pool.imap(read_fits, image_paths), total=len(image_paths)
                    )
                )
            print(np.shape(img_data_list))
            img_data_list = np.array(img_data_list)
            img_data_list = img_data_list.astype(np.float32)
            h5_file_path = os.path.join(h5_dir, f"data_chunk_{chunk_num}.h5")
            make_dataset_clean_chunk(
                h5_file_path,
                data_cat_og,
                img_data_list,
                count_low=count_low,
                count_hi=count_hi,
            )
            chunk_num += 1
    else:
        chunk_num = next_data_chunk_index(h5_glob)
        n_existing = len(glob.glob(h5_glob))
        print(
            f"Skip full clean build. Next data_chunk index = {chunk_num} "
            f"({n_existing} files match {h5_glob})."
        )
    if create_shred_chunks and not append_new_h5_chunks:
        bgsb_shred = Table.read(
            "/pscratch/sd/v/virajvm/catalog_dr1_dwarfs/iron_photometry/iron_BGS_BRIGHT_shreds_catalog_w_aper_mags.fits"
        )["TARGETID", "IMAGE_PATH", "FILE_PATH", "STARDIST_DEG"]
        bgsf_shred = Table.read(
            "/pscratch/sd/v/virajvm/catalog_dr1_dwarfs/iron_photometry/iron_BGS_FAINT_shreds_catalog_w_aper_mags.fits"
        )["TARGETID", "IMAGE_PATH", "FILE_PATH", "STARDIST_DEG"]
        lowz_shred = Table.read(
            "/pscratch/sd/v/virajvm/catalog_dr1_dwarfs/iron_photometry/iron_LOWZ_shreds_catalog_w_aper_mags.fits"
        )["TARGETID", "IMAGE_PATH", "FILE_PATH", "STARDIST_DEG"]
        elg_shred = Table.read(
            "/pscratch/sd/v/virajvm/catalog_dr1_dwarfs/iron_photometry/iron_ELG_shreds_catalog_w_aper_mags.fits"
        )["TARGETID", "IMAGE_PATH", "FILE_PATH", "STARDIST_DEG"]
        sga_shred = Table.read(
            "/pscratch/sd/v/virajvm/catalog_dr1_dwarfs/iron_photometry/iron_SGA_sga_catalog_w_aper_mags.fits"
        )["TARGETID", "IMAGE_PATH", "FILE_PATH", "STARDIST_DEG"]
        tot_shred_temp = vstack(
            [bgsb_shred, bgsf_shred, lowz_shred, elg_shred, sga_shred]
        )
        tot_shred_temp = unique(tot_shred_temp, keys="TARGETID")
        data_cat_shreds, _ = matching_image_paths(data_cat_shreds, tot_temp)
        all_org_imgs = []
        all_gmag = []
        all_rmag = []
        all_zmag = []
        all_mstar = []
        all_stardist = []
        all_zred = []
        all_tgid = []
        print(data_cat_shreds.keys())
        for i in trange(len(data_cat_shreds)):
            org_img = process_shred_imgs(data_cat_shreds, i)
            if np.shape(org_img) != (3, 152, 152):
                print(np.shape(org_img))
                print("--")
            else:
                all_org_imgs.append(org_img.astype(np.float32))
                all_tgid.append(data_cat_shreds["TARGETID"][i])
                all_zred.append(data_cat_shreds["Z"][i])
                all_mstar.append(data_cat_shreds["LOG_MSTAR_M24"][i])
                all_stardist.append(data_cat_shreds["STARDIST_DEG"][i])
                all_gmag.append(data_cat_shreds["MAG_G"][i])
                all_rmag.append(data_cat_shreds["MAG_R"][i])
                all_zmag.append(data_cat_shreds["MAG_Z"][i])
        chunk_size = 2000
        for i in trange(0, len(all_org_imgs), chunk_size):
            h5_org_path = os.path.join(h5_dir, f"data_chunk_{chunk_num}.h5")
            assert len(all_org_imgs[i : i + chunk_size]) == len(
                all_tgid[i : i + chunk_size]
            )
            make_dataset_shred_chunk(
                h5_org_path,
                all_imgs=all_org_imgs[i : i + chunk_size],
                all_tgids=all_tgid[i : i + chunk_size],
                all_zreds=all_zred[i : i + chunk_size],
                all_gmags=all_gmag[i : i + chunk_size],
                all_rmags=all_rmag[i : i + chunk_size],
                all_zmags=all_zmag[i : i + chunk_size],
                all_mstar=all_mstar[i : i + chunk_size],
                all_stardist=all_stardist[i : i + chunk_size],
                img_type="org",
            )
            print(
                f"Wrote chunk {chunk_num} with indices [{i}:{i + chunk_size}]"
            )
            chunk_num += 1

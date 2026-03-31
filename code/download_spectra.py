'''
python3 desi_dwarfs/code/download_spectra.py -random -nchunks 50 -save_name desi_y1_dwarf_combine > download_spectra.log 2>&1

python3 download_spectra.py -nchunks 10 -append_sample OTHER -existing_h5 /pscratch/sd/v/virajvm/catalog_dr1_dwarfs/iron_spectra/spectra_files/data/desi_dr1_dwarf_catalog_spectra.h5


'''

import numpy as np
import matplotlib.pyplot as plt
# Define normalization
import os
import random
import argparse
from astropy.table import Table
from tqdm import trange
from desi_lowz_funcs import print_stage, check_path_existence, parse_tgids
import desispec.io
from desispec import coaddition  
import h5py

def argument_parser():
    '''
    Function that parses the arguments passed while running a script
    '''
    result = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # this is the catalog file we are using to load the spectra!
    # result.add_argument('-catalog', dest='catalog', type=str, default = "") 
    result.add_argument('-min', dest='min', type=int,default = 0)
    result.add_argument('-max', dest='max', type=int,default = 500000) 
    result.add_argument('-ncores', dest='ncores', type=int,default = 60) 
    result.add_argument('-tgids',dest="tgids_list", type=parse_tgids) 
    result.add_argument('-random', dest='random',  action='store_true') 
    result.add_argument('-nchunks',dest='nchunks', type=int,default = 1)
    result.add_argument('-save_name',dest='save_name', type = str, default = "spectra")
    result.add_argument('-append_sample', dest='append_sample', type=str, default=None,
                        help='If set, only download spectra for this SAMPLE value and append to existing HDF5')
    result.add_argument('-existing_h5', dest='existing_h5', type=str, default=None,
                        help='Path to existing HDF5 file to append to (required with -append_sample)')
    result.add_argument('-overwrite', dest='overwrite', action='store_true',
                        help='Re-download all spectra and overwrite the output h5 file (skips incremental sync)')

    return result


def download_and_append_sample_spectra(catalog_path, existing_h5_path, sample_name, ncores, nchunks):
    """
    Download spectra for a specific SAMPLE subset of the catalog and append
    them to an existing consolidated HDF5 spectra file.

    Parameters
    ----------
    catalog_path : str
        Path to the multi-extension FITS catalog (must have a MAIN HDU).
    existing_h5_path : str
        Path to the existing HDF5 file containing TARGETID, Z, WAVE, FLUX, FLUX_IVAR.
    sample_name : str
        Value of the SAMPLE column to filter by (e.g., "OTHER").
    ncores : int
        Number of parallel processes for spectra download.
    nchunks : int
        Number of chunks to split the download into.
    """
    print_stage(f"Append mode: downloading spectra for SAMPLE = '{sample_name}'")

    data_cat = Table.read(catalog_path, hdu="MAIN")
    data_cat = data_cat[data_cat["SAMPLE"] == sample_name]
    print(f"Objects with SAMPLE='{sample_name}' in catalog: {len(data_cat)}")

    if len(data_cat) == 0:
        print("No objects to download. Exiting append mode.")
        return

    with h5py.File(existing_h5_path, "r") as f:
        existing_tgids = set(f["TARGETID"][:])

    new_mask = ~np.isin(data_cat["TARGETID"].data, list(existing_tgids))
    n_already = np.sum(~new_mask)
    data_cat = data_cat[new_mask]

    print(f"Already present in HDF5 (skipped): {n_already}")
    print(f"New objects to download: {len(data_cat)}")

    if len(data_cat) == 0:
        print("All objects already present. Nothing to append.")
        return

    temp = data_cat["TARGETID", "SURVEY", "PROGRAM", "HEALPIX", "Z"]

    all_ks = np.arange(len(temp))
    all_ks_chunks = np.array_split(all_ks, nchunks)

    new_targetids = []
    new_zreds = []
    new_fluxs = []
    new_ivars = []

    print_stage(f"Downloading {len(temp)} spectra in {nchunks} chunks")

    for chunk_i in trange(nchunks):
        all_ks_i = all_ks_chunks[chunk_i]
        if len(all_ks_i) == 0:
            continue

        print_stage(f"Chunk {chunk_i}/{nchunks}: {len(all_ks_i)} objects")

        temp_chunk_i = temp[all_ks_i]

        data_spec = desispec.io.spectra.read_spectra_parallel(
            temp_chunk_i, nproc=ncores, prefix='coadd',
            rdspec_kwargs={"skip_hdus": ["EXP_FIBERMAP", "SCORES", "EXTRA_CATALOG", "MASK", "RESOLUTION"]},
            specprod="iron", match_order=True,
        )

        spec_combined = coaddition.coadd_cameras(data_spec)

        new_targetids.append(temp_chunk_i["TARGETID"].data)
        new_zreds.append(temp_chunk_i["Z"].data)
        new_fluxs.append(spec_combined.flux["brz"])
        new_ivars.append(spec_combined.ivar["brz"])

    new_targetids = np.concatenate(new_targetids)
    new_zreds = np.concatenate(new_zreds)
    new_fluxs = np.concatenate(new_fluxs)
    new_ivars = np.concatenate(new_ivars)

    print_stage(f"Downloaded {len(new_targetids)} new spectra. Appending to {existing_h5_path}")

    with h5py.File(existing_h5_path, "r") as f:
        old_tgids = f["TARGETID"][:]
        old_z = f["Z"][:]
        old_wave = f["WAVE"][:]
        old_flux = f["FLUX"][:]
        old_ivar = f["FLUX_IVAR"][:]

    combined_tgids = np.concatenate([old_tgids, new_targetids])
    combined_z = np.concatenate([old_z, new_zreds])
    combined_flux = np.concatenate([old_flux, new_fluxs])
    combined_ivar = np.concatenate([old_ivar, new_ivars])

    print(f"Before append: {len(old_tgids)} spectra")
    print(f"After append:  {len(combined_tgids)} spectra")

    with h5py.File(existing_h5_path, "w") as f:
        f.create_dataset("TARGETID", data=combined_tgids, dtype='i8')
        f.create_dataset("Z", data=combined_z, dtype='f4')
        f.create_dataset("WAVE", data=old_wave, dtype='f4')
        f.create_dataset("FLUX", data=combined_flux, dtype='f4')
        f.create_dataset("FLUX_IVAR", data=combined_ivar, dtype='f4')

    print_stage(f"Successfully appended {len(new_targetids)} spectra to {existing_h5_path}")


def general_download(cat, ncores, nchunks):
    """
    Download coadded spectra for every row in *cat* and return the results
    in memory (no disk I/O).

    Parameters
    ----------
    cat : astropy.table.Table
        Must contain columns TARGETID, SURVEY, PROGRAM, HEALPIX, Z.
    ncores : int
        Number of parallel processes for ``read_spectra_parallel``.
    nchunks : int
        Split the download into this many sequential chunks to limit memory.

    Returns
    -------
    dict with keys TARGETID, Z, WAVE, FLUX, FLUX_IVAR (numpy arrays).
    """
    temp = cat["TARGETID", "SURVEY", "PROGRAM", "HEALPIX", "Z"]

    all_ks = np.arange(len(temp))
    all_ks_chunks = np.array_split(all_ks, nchunks)

    collected_tgids = []
    collected_zreds = []
    collected_fluxs = []
    collected_ivars = []
    shared_wave = None

    print_stage(f"Downloading {len(temp)} spectra in {nchunks} chunk(s)")

    for chunk_i in trange(nchunks):
        all_ks_i = all_ks_chunks[chunk_i]
        if len(all_ks_i) == 0:
            continue

        print_stage(f"Chunk {chunk_i}/{nchunks}: {len(all_ks_i)} objects")
        temp_chunk_i = temp[all_ks_i]

        data_spec = desispec.io.spectra.read_spectra_parallel(
            temp_chunk_i, nproc=ncores, prefix='coadd',
            rdspec_kwargs={"skip_hdus": ["EXP_FIBERMAP", "SCORES", "EXTRA_CATALOG", "MASK", "RESOLUTION"]},
            specprod="iron", match_order=True,
        )

        spec_combined = coaddition.coadd_cameras(data_spec)

        collected_tgids.append(temp_chunk_i["TARGETID"].data)
        collected_zreds.append(temp_chunk_i["Z"].data)
        collected_fluxs.append(spec_combined.flux["brz"])
        collected_ivars.append(spec_combined.ivar["brz"])
        if shared_wave is None:
            shared_wave = spec_combined.wave["brz"]

    return {
        "TARGETID": np.concatenate(collected_tgids),
        "Z": np.concatenate(collected_zreds),
        "WAVE": shared_wave,
        "FLUX": np.concatenate(collected_fluxs),
        "FLUX_IVAR": np.concatenate(collected_ivars),
    }


def save_h5(data_dict, save_path):
    """
    Write a spectra dict (TARGETID, Z, WAVE, FLUX, FLUX_IVAR) to an HDF5 file.
    Overwrites the file if it already exists.
    """
    print_stage(f"Saving {len(data_dict['TARGETID'])} spectra to {save_path}")
    with h5py.File(save_path, "w") as f:
        f.create_dataset("TARGETID",  data=data_dict["TARGETID"],  dtype='i8')
        f.create_dataset("Z",         data=data_dict["Z"],         dtype='f4')
        f.create_dataset("WAVE",      data=data_dict["WAVE"],      dtype='f4')
        f.create_dataset("FLUX",      data=data_dict["FLUX"],      dtype='f4')
        f.create_dataset("FLUX_IVAR", data=data_dict["FLUX_IVAR"], dtype='f4')
    print_stage(f"Saved to {save_path}")


if __name__ == '__main__':

    rng = np.random.default_rng(42)

    # read in command line arguments
    args = argument_parser().parse_args()

    #sample_str could also be multiple samples together!
    # cat_path = args.catalog
    min_ind = args.min
    max_ind = args.max
    ncores = args.ncores
    tgids_list = args.tgids_list
    random = args.random
    nchunks = args.nchunks
    save_name = args.save_name

    filename = "/pscratch/sd/v/virajvm/desi_dwarf_catalogs/dr1/v1.0/desi_dr1_dwarf_catalog.fits"

    if args.append_sample is not None:
        ##########################################################
        # APPEND MODE: download spectra for a single SAMPLE value
        # and append to an existing HDF5 file.
        ##########################################################
        if args.existing_h5 is None:
            raise ValueError("-existing_h5 must be provided when using -append_sample")

        download_and_append_sample_spectra(
            catalog_path=filename,
            existing_h5_path=args.existing_h5,
            sample_name=args.append_sample,
            ncores=ncores,
            nchunks=nchunks,
        )

    else:
        ##########################################################
        # SYNC (default) / OVERWRITE mode
        ##########################################################

        print_stage("Loading the DESI catalogs")

        data_cat = Table.read(filename, hdu="MAIN")
        print(f"Total objects in catalog: {len(data_cat)}")

        file_template = "/pscratch/sd/v/virajvm/catalog_dr1_dwarfs/iron_spectra/spectra_files/data/" + save_name
        h5_path = file_template + ".h5"

        overwrite = args.overwrite

        if overwrite or not os.path.exists(h5_path):
            ##########################################################
            # OVERWRITE / FIRST-RUN: download everything
            ##########################################################
            if not os.path.exists(h5_path):
                print_stage("No existing h5 file found — downloading all spectra")
            else:
                print_stage("Overwrite flag set — re-downloading all spectra")

            if tgids_list is not None:
                print("List of targetids to process:", tgids_list)
                data_cat = data_cat[np.isin(data_cat['TARGETID'], np.array(tgids_list))]
                print("Number of targetids to process =", len(data_cat))

            max_ind = np.minimum(max_ind, len(data_cat))
            data_cat = data_cat[min_ind:max_ind]

            print(f"Total number of spectra to download = {len(data_cat)}")

            result = general_download(data_cat, ncores, nchunks)
            save_h5(result, h5_path)

        else:
            ##########################################################
            # SYNC MODE (default): only download new, remove stale
            ##########################################################
            print_stage("Sync mode: comparing catalog to existing h5 file")

            catalog_tgids = set(data_cat["TARGETID"].data)

            with h5py.File(h5_path, "r") as f:
                h5_tgids_arr = f["TARGETID"][:]
            h5_tgids = set(h5_tgids_arr)

            new_tgids = catalog_tgids - h5_tgids
            keep_tgids = h5_tgids & catalog_tgids
            removed_tgids = h5_tgids - catalog_tgids

            print(f"  Catalog TARGETIDs : {len(catalog_tgids)}")
            print(f"  Existing in h5    : {len(h5_tgids)}")
            print(f"  To keep (already) : {len(keep_tgids)}")
            print(f"  To download (new) : {len(new_tgids)}")
            print(f"  To remove (stale) : {len(removed_tgids)}")

            if len(new_tgids) == 0 and len(removed_tgids) == 0:
                print_stage("h5 file already matches catalog — nothing to do")

            else:
                # --- Download spectra for new TARGETIDs ---
                new_data = None
                if len(new_tgids) > 0:
                    new_cat = data_cat[np.isin(data_cat["TARGETID"].data, list(new_tgids))]
                    print_stage(f"Downloading {len(new_cat)} new spectra")
                    new_data = general_download(new_cat, ncores, nchunks)

                # --- Load existing h5 and retain only keep_tgids ---
                with h5py.File(h5_path, "r") as f:
                    existing_tgids = f["TARGETID"][:]
                    keep_mask = np.isin(existing_tgids, list(keep_tgids))
                    kept = {
                        "TARGETID": f["TARGETID"][:][keep_mask],
                        "Z":        f["Z"][:][keep_mask],
                        "WAVE":     f["WAVE"][:],
                        "FLUX":     f["FLUX"][:][keep_mask],
                        "FLUX_IVAR": f["FLUX_IVAR"][:][keep_mask],
                    }

                # --- Merge kept + new ---
                if new_data is not None and len(kept["TARGETID"]) > 0:
                    merged = {
                        "TARGETID":  np.concatenate([kept["TARGETID"],  new_data["TARGETID"]]),
                        "Z":         np.concatenate([kept["Z"],         new_data["Z"]]),
                        "WAVE":      kept["WAVE"],
                        "FLUX":      np.concatenate([kept["FLUX"],      new_data["FLUX"]]),
                        "FLUX_IVAR": np.concatenate([kept["FLUX_IVAR"], new_data["FLUX_IVAR"]]),
                    }
                elif new_data is not None:
                    merged = new_data
                else:
                    merged = kept

                print(f"Final spectra count: {len(merged['TARGETID'])}")
                save_h5(merged, h5_path)
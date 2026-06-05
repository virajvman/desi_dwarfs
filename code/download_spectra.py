'''
# First run (resumable): if this times out or crashes, re-run the SAME command
# WITHOUT --overwrite to resume via sync mode (only the missing TARGETIDs are
# downloaded).
python3 desi_dwarfs/code/download_spectra.py -nchunks 370 -ncores 64 \
    -save_name desi_dr1_dwarf_catalog_spectra --overwrite

# Resume after a timeout (sync mode fills in whatever is missing):
python3 desi_dwarfs/code/download_spectra.py -nchunks 370 -ncores 64 \
    -save_name desi_dr1_dwarf_catalog_spectra

# Append a single SAMPLE to an existing consolidated file:
python3 download_spectra.py -nchunks 10 -append_sample OTHER \
    -existing_h5 /pscratch/sd/v/virajvm/catalog_dr1_dwarfs/iron_spectra/spectra_files/data/desi_dr1_dwarf_catalog_spectra.h5
'''

import os
import argparse

import h5py
import numpy as np
from astropy.table import Table
from desiutil.dust import dust_transmission
from tqdm import trange

import desispec.io
from desispec import coaddition
from desi_lowz_funcs import print_stage, check_path_existence, parse_tgids


def apply_mw_extinction_correction(flux, ivar, wave, ebv, rv=3.1):
    """
    Deredden coadd flux using DESI-standard dust_transmission.

    Uses Fitzpatrick (1999) with R_V=3.1 and SF11 1.029 E(B-V) scaling
    applied internally by desiutil.dust.dust_transmission.

    Parameters
    ----------
    flux, ivar : ndarray, shape (nspec, nwave)
    wave : ndarray, shape (nwave,)
    ebv : ndarray, shape (nspec,) — raw SFD98 E(B-V) from fibermap
    rv : float
        Total-to-selective extinction ratio (default 3.1).

    Returns
    -------
    flux_dered, ivar_dered : ndarray, float32
    """
    ebv = np.asarray(ebv, dtype=np.float64)
    trans = dust_transmission(wave, ebv[:, None], Rv=rv)
    flux_dered = flux / trans
    ivar_dered = ivar * trans**2
    return flux_dered.astype(np.float32), ivar_dered.astype(np.float32)


def argument_parser():
    """Parse command-line arguments."""
    result = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    result.add_argument('-min', dest='min', type=int, default=0)
    result.add_argument('-max', dest='max', type=int, default=500000)
    result.add_argument('-ncores', dest='ncores', type=int, default=60)
    result.add_argument('-tgids', dest="tgids_list", type=parse_tgids)
    result.add_argument('-random', dest='random', action='store_true')
    result.add_argument('-nchunks', dest='nchunks', type=int, default=1)
    result.add_argument('-save_name', dest='save_name', type=str, default="spectra")
    result.add_argument('-append_sample', dest='append_sample', type=str, default=None,
                        help='If set, only download spectra for this SAMPLE value and append to existing HDF5')
    result.add_argument('-existing_h5', dest='existing_h5', type=str, default=None,
                        help='Path to existing HDF5 file to append to (required with -append_sample)')
    result.add_argument('-overwrite', dest='overwrite', action='store_true',
                        help='Re-download all spectra and overwrite the output h5 file (skips incremental sync)')
    result.add_argument('-checkpoint_every', dest='checkpoint_every', type=int, default=25,
                        help='In first-run/overwrite mode, write an atomic checkpoint every N chunks so a '
                             'timeout/crash is resumable (re-run WITHOUT --overwrite to resume). '
                             'Set very large to disable.')
    return result


def _sort_targets_by_file(temp):
    """
    Sort a targets table so that all targets sharing a coadd file are
    contiguous (group by HEALPIX, SURVEY, PROGRAM).

    This matters because the outer chunking below splits by row index: if the
    catalog is not file-sorted, a single coadd file's targets scatter across
    many chunks and read_spectra_parallel re-opens that file once per chunk it
    appears in -- redundant I/O of order (avg targets-per-file). Sorting makes
    each chunk file-contiguous so every file is read once (bar a few that
    straddle a chunk boundary). read_spectra_parallel already groups by file
    internally, so this only helps because of the outer chunk loop.
    """
    temp = temp.group_by(["HEALPIX", "SURVEY", "PROGRAM"])
    nfiles = len(temp.groups)
    print(f"  {len(temp)} targets across {nfiles} unique coadd files "
          f"(avg {len(temp)/max(nfiles, 1):.1f} targets/file)")
    return temp


def _download_chunk(temp_chunk_i, ncores):
    """
    Download + camera-coadd + MW-deredden one chunk of targets.

    Returns (targetids, zreds, ebv, wave, flux, ivar) row-matched to
    temp_chunk_i. Flux/ivar are MW-extinction-corrected (dereddened) in the
    observed frame.
    """
    data_spec = desispec.io.spectra.read_spectra_parallel(
        temp_chunk_i, nproc=ncores, prefix='coadd',
        rdspec_kwargs={"skip_hdus": ["EXP_FIBERMAP", "SCORES", "EXTRA_CATALOG", "MASK", "RESOLUTION"]},
        specprod="iron", match_order=True,
    )

    # match_order=True returns spectra row-matched to the input target order, so
    # EBV (from the downloaded fibermap) lines up with TARGETID/Z (from the
    # catalog). Assert it rather than trust it: a silent misalignment would
    # deredden with the wrong E(B-V) and mislabel TARGETIDs.
    assert np.array_equal(
        np.asarray(data_spec.fibermap["TARGETID"]),
        np.asarray(temp_chunk_i["TARGETID"]),
    ), "fibermap/catalog TARGETID mismatch within chunk"

    spec_combined = coaddition.coadd_cameras(data_spec)

    wave = spec_combined.wave["brz"]
    flux = spec_combined.flux["brz"]
    ivar = spec_combined.ivar["brz"]
    ebv = np.asarray(data_spec.fibermap["EBV"])
    flux, ivar = apply_mw_extinction_correction(flux, ivar, wave, ebv)

    return (
        np.asarray(temp_chunk_i["TARGETID"].data),
        np.asarray(temp_chunk_i["Z"].data),
        ebv,
        wave,
        flux,
        ivar,
    )


def download_and_append_sample_spectra(catalog_path, existing_h5_path, sample_name, ncores, nchunks):
    """
    Download spectra for a specific SAMPLE subset of the catalog and append
    them to an existing consolidated HDF5 spectra file.

    Parameters
    ----------
    catalog_path : str
        Path to the multi-extension FITS catalog (must have a MAIN HDU).
    existing_h5_path : str
        Path to the existing HDF5 file containing TARGETID, Z, WAVE, FLUX, FLUX_IVAR, EBV.
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
    temp = _sort_targets_by_file(temp)

    all_ks = np.arange(len(temp))
    all_ks_chunks = np.array_split(all_ks, nchunks)

    new_targetids = []
    new_zreds = []
    new_ebv = []
    new_fluxs = []
    new_ivars = []

    print_stage(f"Downloading {len(temp)} spectra in {nchunks} chunks")

    for chunk_i in trange(nchunks):
        all_ks_i = all_ks_chunks[chunk_i]
        if len(all_ks_i) == 0:
            continue

        temp_chunk_i = temp[all_ks_i]
        tgids, zreds, ebv, wave, flux, ivar = _download_chunk(temp_chunk_i, ncores)

        new_targetids.append(tgids)
        new_zreds.append(zreds)
        new_ebv.append(ebv)
        new_fluxs.append(flux)
        new_ivars.append(ivar)

    new_targetids = np.concatenate(new_targetids)
    new_zreds = np.concatenate(new_zreds)
    new_ebv = np.concatenate(new_ebv)
    new_fluxs = np.concatenate(new_fluxs)
    new_ivars = np.concatenate(new_ivars)

    print_stage(f"Downloaded {len(new_targetids)} new spectra. Appending to {existing_h5_path}")

    with h5py.File(existing_h5_path, "r") as f:
        old_tgids = f["TARGETID"][:]
        old_z = f["Z"][:]
        old_ebv = f["EBV"][:]
        old_wave = f["WAVE"][:]
        old_flux = f["FLUX"][:]
        old_ivar = f["FLUX_IVAR"][:]

    combined = {
        "TARGETID": np.concatenate([old_tgids, new_targetids]),
        "Z": np.concatenate([old_z, new_zreds]),
        "EBV": np.concatenate([old_ebv, new_ebv]),
        "WAVE": old_wave,
        "FLUX": np.concatenate([old_flux, new_fluxs]),
        "FLUX_IVAR": np.concatenate([old_ivar, new_ivars]),
    }

    print(f"Before append: {len(old_tgids)} spectra")
    print(f"After append:  {len(combined['TARGETID'])} spectra")

    save_h5(combined, existing_h5_path)
    print_stage(f"Successfully appended {len(new_targetids)} spectra to {existing_h5_path}")


def general_download(cat, ncores, nchunks, checkpoint_path=None, checkpoint_every=25):
    """
    Download coadded, MW-dereddened spectra for every row in *cat*.

    Parameters
    ----------
    cat : astropy.table.Table
        Must contain columns TARGETID, SURVEY, PROGRAM, HEALPIX, Z.
    ncores : int
        Number of parallel processes for ``read_spectra_parallel``.
    nchunks : int
        Split the download into this many sequential chunks.
    checkpoint_path : str, optional
        If set, write an atomic snapshot of everything-collected-so-far to this
        path every ``checkpoint_every`` chunks, so a timeout/crash leaves a
        resumable file. Resume by re-running in sync mode (no --overwrite).
    checkpoint_every : int
        Checkpoint cadence in chunks.

    Returns
    -------
    dict with keys TARGETID, Z, EBV, WAVE, FLUX, FLUX_IVAR (numpy arrays).
    Flux and ivar are MW-extinction-corrected (dereddened).
    """
    temp = cat["TARGETID", "SURVEY", "PROGRAM", "HEALPIX", "Z"]
    temp = _sort_targets_by_file(temp)

    all_ks = np.arange(len(temp))
    all_ks_chunks = np.array_split(all_ks, nchunks)

    collected_tgids = []
    collected_zreds = []
    collected_ebv = []
    collected_fluxs = []
    collected_ivars = []
    shared_wave = None

    print_stage(f"Downloading {len(temp)} spectra in {nchunks} chunk(s)")

    def _assemble():
        return {
            "TARGETID": np.concatenate(collected_tgids),
            "Z": np.concatenate(collected_zreds),
            "EBV": np.concatenate(collected_ebv),
            "WAVE": shared_wave,
            "FLUX": np.concatenate(collected_fluxs),
            "FLUX_IVAR": np.concatenate(collected_ivars),
        }

    for chunk_i in trange(nchunks):
        all_ks_i = all_ks_chunks[chunk_i]
        if len(all_ks_i) == 0:
            continue

        temp_chunk_i = temp[all_ks_i]
        tgids, zreds, ebv, wave, flux, ivar = _download_chunk(temp_chunk_i, ncores)

        collected_tgids.append(tgids)
        collected_zreds.append(zreds)
        collected_ebv.append(ebv)
        collected_fluxs.append(flux)
        collected_ivars.append(ivar)
        if shared_wave is None:
            shared_wave = wave

        # Periodic atomic checkpoint so a timeout/crash leaves a resumable file.
        if (checkpoint_path is not None) and ((chunk_i + 1) % checkpoint_every == 0):
            save_h5(_assemble(), checkpoint_path)
            n_so_far = sum(len(t) for t in collected_tgids)
            print_stage(f"Checkpoint written after {chunk_i + 1}/{nchunks} chunks "
                        f"({n_so_far} spectra)")

    return _assemble()


def save_h5(data_dict, save_path):
    """
    Write a spectra dict (TARGETID, Z, EBV, WAVE, FLUX, FLUX_IVAR) to HDF5.

    Writes to a temporary file and atomically renames it into place, so a
    crash mid-write can never corrupt an existing good file.
    """
    print_stage(f"Saving {len(data_dict['TARGETID'])} spectra to {save_path}")
    tmp_path = f"{save_path}.tmp.{os.getpid()}"
    with h5py.File(tmp_path, "w") as f:
        f.create_dataset("TARGETID", data=data_dict["TARGETID"], dtype='i8')
        f.create_dataset("Z", data=data_dict["Z"], dtype='f4')
        f.create_dataset("EBV", data=data_dict["EBV"], dtype='f4')
        f.create_dataset("WAVE", data=data_dict["WAVE"], dtype='f4')
        f.create_dataset("FLUX", data=data_dict["FLUX"], dtype='f4')
        f.create_dataset("FLUX_IVAR", data=data_dict["FLUX_IVAR"], dtype='f4')
    os.replace(tmp_path, save_path)  # atomic on the same filesystem
    print_stage(f"Saved to {save_path}")


if __name__ == '__main__':

    rng = np.random.default_rng(42)

    args = argument_parser().parse_args()

    min_ind = args.min
    max_ind = args.max
    ncores = args.ncores
    tgids_list = args.tgids_list
    random = args.random
    nchunks = args.nchunks
    save_name = args.save_name
    checkpoint_every = args.checkpoint_every

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

        print(h5_path)

        if overwrite or not os.path.exists(h5_path):
            ##########################################################
            # OVERWRITE / FIRST-RUN: download everything
            #
            # Resumable: periodic atomic checkpoints are written to h5_path.
            # If this is interrupted, re-run the SAME command WITHOUT
            # --overwrite to resume via sync mode (downloads only what is
            # missing relative to the last checkpoint).
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

            result = general_download(data_cat, ncores, nchunks,
                                      checkpoint_path=h5_path,
                                      checkpoint_every=checkpoint_every)
            save_h5(result, h5_path)

        else:
            ##########################################################
            # SYNC MODE (default): only download new, remove stale.
            # This is also the resume path after an interrupted first run.
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
                new_data = None
                if len(new_tgids) > 0:
                    new_cat = data_cat[np.isin(data_cat["TARGETID"].data, list(new_tgids))]
                    print_stage(f"Downloading {len(new_cat)} new spectra")
                    new_data = general_download(new_cat, ncores, nchunks)

                with h5py.File(h5_path, "r") as f:
                    existing_tgids = f["TARGETID"][:]
                    keep_mask = np.isin(existing_tgids, list(keep_tgids))
                    kept = {
                        "TARGETID": f["TARGETID"][:][keep_mask],
                        "Z": f["Z"][:][keep_mask],
                        "EBV": f["EBV"][:][keep_mask],
                        "WAVE": f["WAVE"][:],
                        "FLUX": f["FLUX"][:][keep_mask],
                        "FLUX_IVAR": f["FLUX_IVAR"][:][keep_mask],
                    }

                if new_data is not None and len(kept["TARGETID"]) > 0:
                    merged = {
                        "TARGETID": np.concatenate([kept["TARGETID"], new_data["TARGETID"]]),
                        "Z": np.concatenate([kept["Z"], new_data["Z"]]),
                        "EBV": np.concatenate([kept["EBV"], new_data["EBV"]]),
                        "WAVE": kept["WAVE"],
                        "FLUX": np.concatenate([kept["FLUX"], new_data["FLUX"]]),
                        "FLUX_IVAR": np.concatenate([kept["FLUX_IVAR"], new_data["FLUX_IVAR"]]),
                    }
                elif new_data is not None:
                    merged = new_data
                else:
                    merged = kept

                print(f"Final spectra count: {len(merged['TARGETID'])}")
                save_h5(merged, h5_path)
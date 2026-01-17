import glob
import h5py
from pathlib import Path


data_path_temp  = "/pscratch/sd/v/virajvm/catalog_dr1_dwarfs/ssl_shred_data/h5_datasets/data_chunk_*.h5"

input_files = glob.glob(data_path_temp)

total_count = 0
pixel = None

for fp in input_files:
    with h5py.File(fp, "r") as f:
        n = f["images"].shape[0]
        total_count += n
        if pixel is None:
            pixel = f["images"].shape[-1]


# create output file
out_path = "/pscratch/sd/v/virajvm/catalog_dr1_dwarfs/ssl_shred_data/desi_dr1_dwarf_catalog_images.h5"

with h5py.File(out_path, "w") as fout:

    images_out = fout.create_dataset(
        "images",
        (total_count, 3, pixel, pixel),
        dtype="float32",
        chunks=(1, 3, pixel, pixel),
        compression="gzip",
        compression_opts=4,
    )

    targetid_out = fout.create_dataset(
        "targetid",
        (total_count,),
        dtype="int64",
        chunks=(1024,),
    )

    # copy data
    idx = 0
    for fp in input_files:
        with h5py.File(fp, "r") as fin:
            imgs = fin["images"]
            tgids = fin["targetid"]

            n = imgs.shape[0]

            images_out[idx:idx+n] = imgs[:]       # images
            targetid_out[idx:idx+n] = tgids[:]    # targetid

            idx += n

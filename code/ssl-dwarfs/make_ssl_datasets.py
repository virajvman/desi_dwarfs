import os
import h5py
import numpy as np
from astropy.table import Table
from astropy.io import fits
from tqdm import trange
from tqdm import tqdm
import multiprocessing as mp

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
    except:
        print(path)
        return np.zeros( (3,152,152) )



def make_dataset_chunk(h5_file_path, catalog, image_data_list, count_low, count_hi):
    '''
    Writing the .h5 file for a single chunk
    '''

    str_dt = h5py.string_dtype(encoding='utf-8')

    pixel = 152
    #the batch size is how we are saving data into the h5 file
    batch_size = 1000

    tot_count = len(catalog[count_low : count_hi])

    print(f"Total Count in this chunk ={tot_count}")
    
    # Open or create HDF5 file
    print(f"Writing to HDF5: {h5_file_path}")
    with h5py.File(h5_file_path, 'w') as f:
        
        # Create datasets with chunking and compression
        images = f.create_dataset(
            'images', (tot_count, 3, pixel, pixel),
            maxshape=(None, 3, pixel, pixel),
            dtype=np.float32,
            chunks=(1, 3, pixel, pixel),
            compression="gzip", compression_opts=4
        )

        
        ra    = f.create_dataset('ra',    (tot_count,), maxshape=(None,), dtype=np.float64)
        dec   = f.create_dataset('dec',   (tot_count,), maxshape=(None,), dtype=np.float64)
        mag_g = f.create_dataset('mag_g', (tot_count,), maxshape=(None,), dtype=np.float32)
        mag_r = f.create_dataset('mag_r', (tot_count,), maxshape=(None,), dtype=np.float32)
        mag_z = f.create_dataset('mag_z', (tot_count,), maxshape=(None,), dtype=np.float32)
        zred  = f.create_dataset('redshift', (tot_count,), maxshape=(None,), dtype=np.float32)
        tgid  = f.create_dataset('targetid', (tot_count,), maxshape=(None,), dtype=np.int)
        image_path  = f.create_dataset('image_path', (tot_count,), maxshape=(None,), dtype=str_dt)
        file_path  = f.create_dataset('file_path', (tot_count,), maxshape=(None,), dtype=str_dt)
        
        
        #should also save the image file path!
        
    
        # Write in batches
        for i in trange(count_low, count_hi, batch_size, desc="Writing batches"):
            j_end = min(i + batch_size, count_hi)

            rel_i = i - count_low
            rel_j_end = j_end - count_low

            # images[rel_i:rel_j_end] = img_data_list[i:j_end]
            #we are feeding an image data list that only corresponds to this specific chunk and thus we index.
            images[rel_i:rel_j_end] = img_data_list[rel_i : rel_j_end]
            ra[rel_i:rel_j_end]     = catalog['RA'][i:j_end]
            dec[rel_i:rel_j_end]    = catalog['DEC'][i:j_end]
            mag_g[rel_i:rel_j_end]  = catalog['MAG_G'][i:j_end]
            mag_r[rel_i:rel_j_end]  = catalog['MAG_R'][i:j_end]
            mag_z[rel_i:rel_j_end]  = catalog['MAG_Z'][i:j_end]
            zred[rel_i:rel_j_end]   = catalog['Z'][i:j_end]
            tgid[rel_i:rel_j_end]   = catalog['TARGETID'][i:j_end]
            image_path[rel_i:rel_j_end]   = catalog['IMAGE_PATH'][i:j_end]
            file_path[rel_i:rel_j_end]   = catalog['FILE_PATH'][i:j_end]
            
    print("Done writing HDF5 dataset.")

    ##check that it is save correctly!
    with h5py.File(h5_file_path, 'r') as f:
        print("Datasets in file:")
        for key in f.keys():
            print(f"{key}: shape = {f[key].shape}, dtype = {f[key].dtype}")
        
        # Optionally, look at a sample value
        print("\nSample values:")
        print("ra[0] =", f['ra'][0])
        print("targetid[0] =", f['targetid'][0])
        print("images[0] shape =", f['images'][0].shape)

if __name__ == '__main__':

    #the entire dwarf galaxy catlaog
    catalog_path = "/pscratch/sd/v/virajvm/catalog_dr1_dwarfs/desi_y1_dwarf_combine_catalog.fits"
    data = Table.read(catalog_path)

    print(f"Size of catalog w/ELG = {len(data)}")
    
    data = data[(data["SAMPLE"] != "ELG") & (data["IMAGE_PATH"] != "") ]

    print(f"Size of catalog wo/ELG = {len(data)}")
    
    chunk_size = 20000
    
    chunk_num = 0
    for count_low in trange(0, len(data),chunk_size):

        ##read all the images for this specific chunk! 
        image_paths = data["IMAGE_PATH"][count_low : count_low + chunk_size]
        
        num_workers = 128
        with mp.Pool(processes=num_workers) as pool:
            # Wrap with tqdm for progress tracking
            img_data_list = list(tqdm(pool.imap(read_fits, image_paths), total=len(image_paths)) )
            

        print(np.shape(img_data_list))
         
        img_data_list = np.array(img_data_list)
        img_data_list = img_data_list.astype(np.float32)

        print(count_low, count_low + chunk_size)
        
        h5_file_path = f"/pscratch/sd/v/virajvm/ssl-legacysurvey-dwarfs/h5_datasets/data_chunk_{chunk_num}.h5"
        
        make_dataset_chunk(h5_file_path, data, img_data_list, count_low = count_low, count_hi = count_low + chunk_size)
        
        chunk_num += 1
    


    



    


    

"""
Code adapted from https://github.com/georgestein/ssl-legacysurvey/blob/main/scripts/similarity_search_nxn.py

1. Loads chunked 2048-dim `.npy` representation files into one big array.
2. Loads a full-array of magnitudes aligned with the representation array
3. Normalizes all vectors for cosine similarity.
4. For each galaxy (parallelized with joblib threading):
   a. Filters catalog to ±delta_mag of its magnitude.
   b. Builds a small FAISS inner-product index on that subset.
   c. Queries top-k neighbors and maps back to full catalog indices.
5. Saves two outputs of shape (N, k):
   - all_similarity_indices.npy
   - all_similarity_scores.npy

conda activate
conda activate ssl-pl
/global/u1/v/virajvm/miniforge3/envs/ssl-pl/bin/python similarity_search.py
"""

import os
import glob
import numpy as np
import faiss
from joblib import Parallel, delayed


def load_all_representations(): #rep_dir, file_pattern="represent*.npy"
    """Load & stack all chunked .npy rep files into one float32 array."""
    # rep_files = sorted(glob.glob(os.path.join(rep_dir, file_pattern)))
    # reps_list = []
    # for f in rep_files:
    #     reps = np.load(f).astype(np.float32)
    #     reps_list.append(reps)

    # reps_list = np.vstack(reps_list)
    
    # print(f"Shape of representation array = {reps_list}")

    reps_list = np.load("/pscratch/sd/v/virajvm/ssl-legacysurvey-dwarfs/representations/total_representation_arr.npy")
    
    return reps_list

def load_all_magnitudes(mag_file):
    """Load the full-array of magnitudes aligned with representations."""
    return np.load(mag_file).astype(np.float32)


def query_one(i, reps, mags, tgids, k, delta_mag, res=None, device=0):

    if i < 10:
        print(i)
    elif i % 500 == 0:
        print(i)
    else:
        pass
    
    mag_i = mags[i]
    mask = np.abs(mags - mag_i) <= delta_mag
    candidate_idxs = np.nonzero(mask)[0]

    if candidate_idxs.size <= k:
        sims = reps[candidate_idxs] @ reps[i]
        order = np.argsort(sims)[::-1]
        sel = candidate_idxs[order]
        sc = sims[order]
        # pad if needed
        if sel.size < k:
            pad = k - sel.size
            sel = np.pad(sel, (0,pad), constant_values=-1)
            sc  = np.pad(sc,  (0,pad), constant_values=np.nan)
        return sel, sc

    sub_reps = reps[candidate_idxs]

    # Normalize subset if reps are not guaranteed normalized here
    # faiss.normalize_L2(sub_reps)

    index_cpu = faiss.IndexFlatIP(sub_reps.shape[1])

    if res is not None:
        index = faiss.index_cpu_to_gpu(res, device, index_cpu)
    else:
        index = index_cpu  # fallback CPU index

    index.add(sub_reps)

    # Query vector must be normalized (if needed)
    # q = reps[i:i+1].copy()
    # faiss.normalize_L2(q)

    D, I = index.search(reps[i:i+1], k)
    full_inds = candidate_idxs[I[0]]
    sims = D[0]

    full_tgids = tgids[full_inds]

    return full_inds, full_tgids, sims


def run_similarity_by_mag_bins(reps, mags, tgids, k, delta_mag, output_dir, mag_bin_width=0.25, device=0, n_jobs = 8,use_parallel=True):

    #there are bins in which I am doing the chunked similarity search
    bins = np.arange(mags.min(), mags.max() + mag_bin_width, mag_bin_width)
    # bins = np.arange(16, 21.75 + mag_bin_width, mag_bin_width)

    print(f"Found this targetid = {tgids[tgids == 39627631053769626]}")
    
    print(f"Mag Max = {mags.max()}, Mag Min = {mags.min()}")
    print(f"Mag Bins = {bins}")
    
    results_inds = []
    results_tgids = []
    results_sims = []

    faiss.normalize_L2(reps)

    for i in range(len(bins)-1): 
        print("==="*10)
        
        print(f"[{i+1}/{len(bins)}")

        mag_center = 0.5*(bins[i] + bins[i+1])
        print(f"Galaxies whose similarity scores are being computed = {[bins[i], bins[i+1]]}")
        print(f"Processing mag slice {mag_center:.2f} ± {mag_bin_width:.2f}")

        # 1. build index only once
        idxs_in_bin = np.where(np.abs(mags - mag_center) <= delta_mag)[0]
        if len(idxs_in_bin) == 0:
            continue

        reps_bin = reps[idxs_in_bin]
        tgids_bin = tgids[idxs_in_bin]
        print(f"Galaxies in this mag bin over which index is  built = {len(tgids_bin)}")

        print(f"Found this targetid = {tgids_bin[tgids_bin == 39627631053769626]}")
        
        index = faiss.IndexFlatIP(reps_bin.shape[1])
        if device is not None:
            print("Using GPUs!")
            res = faiss.StandardGpuResources()
            index = faiss.index_cpu_to_gpu(res, device, index)
        index.add(reps_bin)

        # 2. Galaxies to query in this bin (strict inner part)
        query_mask = (mags >= bins[i]) & (mags < bins[i+1])
        query_idxs = np.where(query_mask)[0]

        print(f"Galaxies for which similarity is being computed = {len(query_idxs)}")
        
        def query_one_in_bin(qidx):
            qvec = reps[qidx:qidx+1]
            D, I = index.search(qvec, k)
            selected_global_inds = idxs_in_bin[I[0]]
            return selected_global_inds, tgids[selected_global_inds], D[0]

        if use_parallel:
            results = Parallel(n_jobs=n_jobs, backend="threading", verbose=5)(
                delayed(query_one_in_bin)(qidx) for qidx in query_idxs
            )
            
            for inds, tgs, sims in results:
                results_inds.append(inds)
                results_tgids.append(tgs)
                results_sims.append(sims)
        else:
            for qidx in query_idxs:
                inds, tgs, sims = query_one_in_bin(qidx)
                
                results_inds.append(inds)
                results_tgids.append(tgs)
                results_sims.append(sims)

        ##temporary saving results 
        results_inds_i = np.stack(results_inds)
        results_tgids_i = np.stack(results_tgids)
        results_sims_i = np.stack(results_sims)
            
        os.makedirs(output_dir, exist_ok=True)
        np.save(os.path.join(output_dir, f"all_similarity_indices_till_bin_{i}.npy"), results_inds_i,)
        
        results_tgids_i = results_tgids_i.astype(np.int64)
        
        np.save(os.path.join(output_dir, f"all_similarity_targetids_till_bin_{i}.npy"), results_tgids_i)
        np.save(os.path.join(output_dir, f"all_similarity_scores_till_bin_{i}.npy"), results_sims_i)

    # Stack results
    print("Saving results...")
    
    results_inds = np.stack(results_inds)
    results_tgids = np.stack(results_tgids)
    results_sims = np.stack(results_sims)

    os.makedirs(output_dir, exist_ok=True)
    np.save(os.path.join(output_dir, "all_similarity_indices_total.npy"), results_inds)
    results_tgids = results_tgids.astype(np.int64)
    np.save(os.path.join(output_dir, "all_similarity_targetids_total.npy"), results_tgids)
    np.save(os.path.join(output_dir, "all_similarity_scores_total.npy"), results_sims)

    print("Done.")

def main(mag_file, tgid_file, output_dir,
         k=100, delta_mag=0.5, n_jobs=8, use_parallel=True):
    
    # 1) Load data
    print("Loading representations…")
    reps = load_all_representations()

    print(f"Represenation Shape = {reps.shape}")
    
    print("Loading magnitudes…")
    mags = load_all_magnitudes(mag_file)
    print(f"Mags Shape = {mags.shape}")

    print("Loading targetids…")
    tgids = np.load(tgid_file).astype(np.int64)
    print(f"Tgids Shape = {tgids.shape}")
    
    # 2) Normalize all for cosine similarity
    # print("Normalizing representations for cosine similarity…")
    # faiss.normalize_L2(reps)

    # 3) Parallel per-galaxy querying
    print(f"Querying top-{k} neighbors for {reps.shape[0]} galaxies…")

    run_similarity_by_mag_bins(reps, mags, tgids, k, delta_mag, output_dir, mag_bin_width=0.25, device=0, n_jobs = n_jobs,use_parallel=use_parallel)
    
    
if __name__ == "__main__":
    # === User parameters: edit these paths & values ===
    mag_file    = "/pscratch/sd/v/virajvm/ssl-legacysurvey-dwarfs/representations/total_rmags_arr.npy"
    tgid_file    = "/pscratch/sd/v/virajvm/ssl-legacysurvey-dwarfs/representations/total_targetids_arr.npy"
    output_dir  = "/pscratch/sd/v/virajvm/ssl-legacysurvey-dwarfs/similarity_search_magb"
    k           = 5
    delta_mag   = 0.5
    n_jobs      = 2
    use_parallel = False

    main(mag_file, tgid_file, output_dir, k, delta_mag, n_jobs, use_parallel)


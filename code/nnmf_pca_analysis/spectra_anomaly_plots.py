import os
import sys
import numpy as np
import h5py
import torch
import torch.nn as nn
import torch.nn.functional as F

class PCA(nn.Module):
    '''
    This is taken from https://github.com/gngdb/pytorch-pca/blob/main/pca.py    
    '''
    def __init__(self, n_components):
        super().__init__()
        self.n_components = n_components

    @staticmethod
    def _svd_flip(u, v, u_based_decision=True):
        """
        Adjusts the signs of the singular vectors from the SVD decomposition for
        deterministic output.

        This method ensures that the output remains consistent across different
        runs.

        Args:
            u (torch.Tensor): Left singular vectors tensor.
            v (torch.Tensor): Right singular vectors tensor.
            u_based_decision (bool, optional): If True, uses the left singular
              vectors to determine the sign flipping. Defaults to True.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Adjusted left and right singular
              vectors tensors.
        """
        if u_based_decision:
            max_abs_cols = torch.argmax(torch.abs(u), dim=0)
            signs = torch.sign(u[max_abs_cols, range(u.shape[1])])
        else:
            max_abs_rows = torch.argmax(torch.abs(v), dim=1)
            signs = torch.sign(v[range(v.shape[0]), max_abs_rows])
        u *= signs
        v *= signs[:, None]
        return u, v

    @torch.no_grad()
    def fit(self, X):
        n, d = X.size()
        if self.n_components is not None:
            d = min(self.n_components, d)
        self.register_buffer("mean_", X.mean(0, keepdim=True))
        Z = X - self.mean_ # center
        U, S, Vh = torch.linalg.svd(Z, full_matrices=False)
        Vt = Vh
        U, Vt = self._svd_flip(U, Vt)
        self.register_buffer("components_", Vt[:d])
        return self

    def forward(self, X):
        return self.transform(X)

    def transform(self, X):
        assert hasattr(self, "components_"), "PCA must be fit before use."
        return torch.matmul(X - self.mean_, self.components_.t())

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

    def inverse_transform(self, Y):
        assert hasattr(self, "components_"), "PCA must be fit before use."
        return torch.matmul(Y, self.components_) + self.mean_


if __name__ == '__main__':

    compute_norm_resis = False

    on_gpu_node=True
    #if we are not on a gpu node, the below are just effectivelt False    
    run_pca = True
    run_umap = True

    flag = "NEW" #this is OG or NEW
    
    save_path = f"/pscratch/sd/v/virajvm/catalog_dr1_dwarfs/iron_spectra/spectra_files/desi_dr1_dwarf_catalog_nnmf_{flag}.h5"
    with h5py.File(save_path, "r") as f:
        wave_rest = f["WAVE_REST"][:]
        flux_scale = f["FLUX_NORM"][:]
        flux_ivar_scale = f["FLUX_IVAR_NORM"][:] 
        nnmf_coeffs = f["NNMF_COEFFS"][:]

    print(np.shape(wave_rest))
    print(np.shape(flux_scale))
    print(np.shape(nnmf_coeffs))

    ## load the nnmf templates !
    nnmf_temps = np.load(f"/pscratch/sd/v/virajvm/catalog_dr1_dwarfs/iron_spectra/nnmf_templates/templates_dwarfs_{flag}.npy")
    print(nnmf_temps.shape)

    if compute_norm_resis:
        print("Computing NNMF residuals!")
        all_inputs = []
        for i in range(flux_scale.shape[1]):
            all_inputs.append(  (flux_scale[:,i], flux_ivar_scale[:,i], nnmf_coeffs[i] )   )
        print(all_inputs[0][0].shape, all_inputs[0][1].shape, all_inputs[0][2].shape)
        from spectra_nnmf_resid import parallel_residual
        all_norm_resis = parallel_residual(all_inputs,  n_processes=128)
        np.save( f"/pscratch/sd/v/virajvm/catalog_dr1_dwarfs/iron_spectra/spectra_files/norm_residuals_dwarfs_{flag}.npy", all_norm_resis )
    else:
        print("Loading NNMF residuals!")
        
        all_norm_resis = np.load(f"/pscratch/sd/v/virajvm/catalog_dr1_dwarfs/iron_spectra/spectra_files/norm_residuals_dwarfs_{flag}.npy"  )
    
    print(f"all_norm_resis shape = {all_norm_resis.shape}")

    if on_gpu_node:
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"device={device}")
        
        if run_pca:
            print("Running PCA on residuals!")
        
            # Load residuals and move to GPU
            X_full = torch.tensor(all_norm_resis, dtype=torch.float32, device=device)
        
            # Randomly split into half for fitting PCA
            n_total = X_full.shape[0]
            n_subset = n_total
            perm = torch.randperm(n_total, device=device)
            subset_idx = perm[:n_subset]
            X_subset = X_full[subset_idx]
        
            print(f"Fitting PCA on subset of {n_subset} spectra out of {n_total}")
        
            # Fit PCA on subset
            pca = PCA(n_components=20).fit(X_subset)
        
            # Extract PCA components
            templates_pca_arr = pca.components_.cpu().numpy()  # (20, 3980)
        
            # Transform entire dataset
            t_arr = pca.transform(X_full).cpu().numpy()  # (Ngal, 20)
        
            # Save components and transformed coefficients
            np.save(f"/pscratch/sd/v/virajvm/catalog_dr1_dwarfs/iron_spectra/spectra_files/pca_components_{flag}.npy",templates_pca_arr)
            np.save(f"/pscratch/sd/v/virajvm/catalog_dr1_dwarfs/iron_spectra/spectra_files/pca_transform_{flag}.npy",t_arr)
        
            print("PCA finished on subset and applied to full dataset!")
        
        else:
            print("Reading PCA results!")
        
            # Load precomputed components and transformed data
            templates_pca_arr = np.load(f"/pscratch/sd/v/virajvm/catalog_dr1_dwarfs/iron_spectra/spectra_files/pca_components_{flag}.npy")
            t_arr = np.load(f"/pscratch/sd/v/virajvm/catalog_dr1_dwarfs/iron_spectra/spectra_files/pca_transform_{flag}.npy")
            
        print(np.shape(t_arr))

        print("Saving the data in h5 files!!")
        
        # Existing file
        h5_in = f"/pscratch/sd/v/virajvm/catalog_dr1_dwarfs/iron_spectra/spectra_files/desi_dr1_dwarf_catalog_nnmf_{flag}.h5"
        
        # New file with PCA_COEFFS added
        h5_out = f"/pscratch/sd/v/virajvm/catalog_dr1_dwarfs/iron_spectra/spectra_files/desi_dr1_dwarf_catalog_nnmf_pca_{flag}.h5"
        
        # PCA coefficients from your PCA fit
        # Make sure this is shape (N_spectra, n_components)
        pca_coeffs = t_arr  

        print(pca_coeffs.shape)
        print(nnmf_coeffs.shape)

        with h5py.File(h5_in, "r") as f_in, h5py.File(h5_out, "w") as f_out:
            # Copy all datasets
            for key in f_in.keys():
                f_in.copy(key, f_out)
            
            # Add PCA_COEFFS dataset
            f_out.create_dataset("PCA_COEFFS", data=pca_coeffs, dtype='f4')
        
        print(f"Saved new HDF5 file with PCA_COEFFS at {h5_out}")
      
        if run_umap:
            print("Running UMAP!")
            
            import umap.umap_ as umap
            from sklearn.preprocessing import StandardScaler
            
            all_spec_feats = np.concatenate( [nnmf_coeffs, t_arr], axis = 1 )
            print(all_spec_feats.shape)
            reducer = umap.UMAP(metric='cosine', random_state=42)
            scaled_t_arr = StandardScaler().fit_transform(all_spec_feats)
            print(scaled_t_arr[0])
            embedding = reducer.fit_transform(scaled_t_arr)
            np.save(f"/pscratch/sd/v/virajvm/catalog_dr1_dwarfs/iron_spectra/spectra_files/desi_dwarfs_umap_nnmf_and_pca_{flag}.npy", embedding)
        else:
            pass


        
    
    
     
    

    



 





































    

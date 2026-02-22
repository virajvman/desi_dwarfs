# NNMF + PCA Spectral Analysis Pipeline

This folder contains the scripts used to decompose DESI dwarf galaxy spectra using Non-Negative Matrix Factorization (NNMF), compute residuals, run PCA on those residuals, and generate UMAP embeddings for anomaly detection.

## Pipeline Overview

```
Raw DESI spectra
      │
      ▼
 nnmf_analysis.py          De-redshift → normalize → train NNMF templates → fit all spectra
      │
      │  outputs: NNMF coefficients, normalized flux/ivar (HDF5)
      ▼
 spectra_nnmf_resid.py     Reconstruct each spectrum from NNMF fit, compute noise-normalized residuals
      │
      │  outputs: residual array (.npy)
      ▼
 spectra_anomaly_plots.py   Run PCA on residuals → combine NNMF + PCA coefficients → UMAP
      │
      │  outputs: PCA components, PCA coefficients, UMAP embedding (.npy / HDF5)
      ▼
 Downstream plotting / anomaly identification
```

## Scripts

### `nnmf_analysis.py`

Handles the full NNMF fitting pipeline: loading DESI spectra, de-redshifting them to rest frame, normalizing with a single-template fit, training 10 NNMF templates on a random training subset (using GPU via CuPy), and fitting the templates to all spectra with non-negative least squares. Results are saved to HDF5.

**Key functions:**

| Function | Description |
|---|---|
| `get_wave(wavemin, wavemax, dloglam)` | Returns a logarithmically-spaced wavelength grid. Default range 3600–10000 Å. |
| `_deredshift_one_spectrum(args)` | De-redshifts and resamples a single spectrum to a rest-frame wavelength grid using `desispec.interpolation.resample_flux`. |
| `deredshift_resample_desi_spectra(all_waves, all_fluxs, all_ivar, all_zreds, ...)` | Parallelized wrapper that de-redshifts an array of spectra onto a common rest-frame grid using multiprocessing. Returns `(wave_out, fluxes, ivars)`. |

The `__main__` block runs the end-to-end pipeline:
1. Load or create de-redshifted spectra (saved to HDF5)
2. Normalize all spectra using a single NNMF template fit (coefficient = scaling factor)
3. Iteratively train 10 NNMF templates on a training subset (GPU, `nearly_nmf`)
4. Fit templates to all spectra via `scipy.optimize.nnls`
5. Save target IDs, redshifts, normalized flux/ivar, NNMF coefficients, residual norms, and validation flags to HDF5

**Dependencies:** `cupy`, `desispec`, `nearly_nmf`, `scipy`, `h5py`, `joblib`, `multiprocessing`

---

### `spectra_nnmf_resid.py`

Helper module (not run standalone) that computes noise-normalized residuals between each observed spectrum and its NNMF reconstruction. Imported by `spectra_anomaly_plots.py`.

**Key functions:**

| Function | Description |
|---|---|
| `init_session(templates)` | Sets a global `nnmf_temps` variable for multiprocessing workers so templates are shared across processes. |
| `get_nnmf_fit(coeffs_i, nnmf_temps)` | Reconstructs a single spectrum as `coeffs_i @ nnmf_temps.T`. |
| `construct_residuals(args)` | Computes `(flux - NNMF_fit) / noise` for one spectrum, where noise = `sqrt(1/ivar)`. |
| `parallel_residual(inputs, n_processes)` | Loads the NNMF templates from disk, then maps `construct_residuals` over all spectra in parallel using `multiprocessing.Pool`. Returns the full residual array. |

**Dependencies:** `numpy`, `multiprocessing`

---

### `spectra_anomaly_plots.py`

Orchestrates the residual computation, PCA dimensionality reduction, and UMAP embedding. Loads the NNMF results from HDF5, computes (or loads) residuals via `spectra_nnmf_resid`, runs PCA on GPU with PyTorch, and optionally produces a UMAP embedding from the combined NNMF + PCA coefficient vectors.

**Key class / functions:**

| Name | Description |
|---|---|
| `PCA` (class, `nn.Module`) | PyTorch-based PCA using SVD. Methods: `fit(X)`, `transform(X)`, `fit_transform(X)`, `inverse_transform(Y)`. Runs on GPU when available. |

The `__main__` block runs:
1. Load normalized flux, ivar, and NNMF coefficients from the NNMF HDF5 output
2. Compute (or load) noise-normalized residuals via `spectra_nnmf_resid.parallel_residual`
3. Fit PCA (20 components) on the residual array and project all spectra
4. Save PCA components and coefficients; copy existing HDF5 and append `PCA_COEFFS`
5. Concatenate NNMF + PCA coefficients, StandardScale, and run UMAP (cosine metric)

**Dependencies:** `torch`, `h5py`, `numpy`, `umap-learn`, `scikit-learn`, `spectra_nnmf_resid`

'''
Central configuration for the spectroscopic incompleteness pipeline.

All NERSC paths, sample cuts, wavelength grids, sampling ranges, and model
hyperparameters live here so every stage script reads the same values.

See README.md in this folder for the statistical framework and the
load-bearing conditions behind these choices.
'''

import numpy as np

# =============================================================================
# NERSC data locations
# =============================================================================
BASE_DIR = "/pscratch/sd/v/virajvm/specz_incompleteness"

REFERENCE_DIR = f"{BASE_DIR}/reference_sample"
BASES_DIR = f"{BASE_DIR}/bases"
MOCK_DIR = f"{BASE_DIR}/mock_spectra"
REDROCK_DIR = f"{BASE_DIR}/redrock"
TRAINING_DIR = f"{BASE_DIR}/training"
MODELS_DIR = f"{BASE_DIR}/models"
VALIDATION_DIR = f"{BASE_DIR}/validation"
CALIB_DIR = f"{BASE_DIR}/efftime_calibration"

# Consolidated reference-sample spectra (built by build_reference_sample.py)
REFERENCE_CATALOG = f"{REFERENCE_DIR}/reference_sample_catalog.fits"
REFERENCE_SPECTRA_H5 = f"{REFERENCE_DIR}/reference_sample_spectra.h5"
REFERENCE_EFFTIME_NPZ = f"{REFERENCE_DIR}/reference_efftime_distribution.npz"

# NNMF bases + bootstrap pool (built by train_nnmf_bases.py)
GEN_BASIS_NPY = f"{BASES_DIR}/W_gen_templates.npy"
FEAT_BASIS_NPY = f"{BASES_DIR}/W_feat_templates.npy"
POOL_H5 = f"{BASES_DIR}/bootstrap_coefficient_pool.h5"

# Trained models (train_completeness_model.py)
PURITY_MODEL_JOBLIB = f"{MODELS_DIR}/purity_mlp.joblib"
COMPLETENESS_MODEL_JOBLIB = f"{MODELS_DIR}/completeness_mlp.joblib"

# Input catalogs used to select the reference sample. These are the existing
# consolidated Iron VAC catalogs with Halpha info + FIBERMAG + TSNR2 columns.
IRON_BGS_BRIGHT_VAC = "/pscratch/sd/v/virajvm/catalog/iron_bgs_bright_vac_halpha.fits"
IRON_LOWZ_DARK_VAC = "/pscratch/sd/v/virajvm/catalog/iron_lowz_dark_vac_halpha.fits"

# Real dwarf catalog + its consolidated spectra (apply_weights.py inputs)
DWARF_CATALOG = "/pscratch/sd/v/virajvm/catalog_dr1_dwarfs/desi_y1_dwarf_clean_catalog_v2.fits"
DWARF_SPECTRA_H5 = ("/pscratch/sd/v/virajvm/catalog_dr1_dwarfs/iron_spectra/"
                    "spectra_files/desi_dr1_dwarf_catalog_spectra.h5")

# SV3 repeat/deep truth table for external validation (ChangHoon Hahn)
SV3_TRUTH_FILE = ("/global/cfs/cdirs/desi/users/chahah/bgs-cmxsv/sv-paper/"
                  "sv3.bgs_exps.efftime160_200.zsuccess.fuji.fits")

# Template FITS file holding a SCORES HDU copied into every mock coadd file
# (redrock's readers expect one; feasibgs does not write it).
SCORES_TEMPLATE_FITS = "/pscratch/sd/v/virajvm/lowz_mock_spectra/temp_lowz_object_1.fits"

# feasiBGS local checkout (imported via sys.path, not installed)
FEASIBGS_PATH = "/global/u1/v/virajvm/DESI2_LOWZ/feasiBGS"

# =============================================================================
# Redrock pinning
# =============================================================================
# The mocks MUST be processed with the same redrock version + galaxy templates
# as the specprod being modeled (iron / DR1), NOT `desi_environment.sh main`.
# The 23.1 software release is the iron-era stack (matches old run_rrdesi.sh
# usage). TODO(viraj): confirm exact iron production tag against the iron run
# documentation before the production redrock run.
DESI_ENVIRONMENT_VERSION = "23.1"
SPECPROD = "iron"

# =============================================================================
# Wavelength grids
# =============================================================================
# Rest-frame log-lambda grid for both NNMF bases. Extended beyond the usual
# [3600, 9000] so that a generated rest-frame spectrum covers the full DESI
# camera range (3600-9824 A observed) at EVERY z in [Z_MIN, Z_MAX]:
#   blue: 3600 / (1 + 0.15) ~ 3130 A ; red: 9824 / (1 + 0.001) ~ 9814 A.
# No single galaxy covers this whole range, but nearly-NMF is ivar-weighted so
# pixels outside a galaxy's coverage (ivar=0) simply don't constrain it there.
REST_WAVEMIN = 3100.0
REST_WAVEMAX = 9850.0
REST_DLOGLAM = 1e-4

# Observed-frame grid on which generated mock spectra are handed to the
# forward-model backend (covers the DESI cameras with margin).
OBS_WAVEMIN = 3566.0
OBS_WAVEMAX = 9852.0
OBS_DWAVE = 0.5

# =============================================================================
# Reference-sample selection (NNMF training set)
# =============================================================================
# Broad, high-SNR, robust-redshift galaxy sample at z < 0.15 (dwarfs AND
# non-dwarfs) drawn from BGS Bright + LOWZ Dark.
REF_Z_MIN = 0.001
REF_Z_MAX = 0.15
REF_DELTACHI2_MIN = 100.0      # robust-redshift requirement
REF_ZWARN_ALLOWED = (0,)       # ZWARN == 0 only for the training sample
REF_SPECTYPE = "GALAXY"

# Bright-pool restriction for the BOOTSTRAP POOL (subset of the reference
# sample used to draw coefficient vectors). At these fiber mags even passive
# galaxies achieve robust redshifts, so the within-cell spectral-type mixture
# is nearly selection-free. (Critical fix for quenched-dwarf coverage.)
POOL_RFIB_MAX = 20.0

# =============================================================================
# NNMF bases
# =============================================================================
N_TEMPLATES_GEN = 10   # rich basis for generating realistic spectra
N_TEMPLATES_FEAT = 3   # coarse basis for NN spectral-type features
NNMF_N_ITER_WARM = 50
NNMF_N_ITER_JOINT = 1000
NNMF_SEED = 42

# =============================================================================
# Emission-line windows (VACUUM wavelengths, Angstrom; DESI convention)
# =============================================================================
LINE_CENTERS = [3727.09, 3729.88,   # [OII]
                4862.68,            # Hbeta
                4960.30, 5008.24,   # [OIII]
                6564.61,            # Halpha
                6718.29, 6732.68]   # [SII]
LINE_HALFWIDTH = 10.0    # A, rest frame
LINE_SIDEBAND_OFFSET = 25.0   # A, offset of continuum sidebands from line center
LINE_SIDEBAND_HALFWIDTH = 10.0

# =============================================================================
# Mock generation
# =============================================================================
Z_MIN = 0.001
Z_MAX = 0.15
RFIB_MIN = 19.0
RFIB_MAX = 23.5

# Stratified oversampling of the bootstrap pool: pool galaxies are binned by
# their emission-line score (specz_utils.emission_line_score) into N_STRATA
# strata (quantiles of the >0 scores, plus a dedicated lowest/passive stratum),
# and mock draws sample strata UNIFORMLY. Legal because the NN conditions on
# spectral type: only coverage matters, not proportions.
N_STRATA = 5

N_MOCKS_TOTAL = 1_000_000
CHUNK_SIZE = 1000              # spectra per mock coadd file (redrock-friendly)

# Effective-exposure-time sampling. Empirical distribution built from the
# reference catalogs' TSNR2 columns and saved to REFERENCE_EFFTIME_NPZ.
# EFFTIME conversions (Guy et al. 2023):
TSNR2_BGS_TO_EFFTIME = 0.1400    # EFFTIME_BRIGHT = 0.1400 * TSNR2_BGS
TSNR2_LRG_TO_EFFTIME = 12.15     # EFFTIME_DARK   = 12.15  * TSNR2_LRG
# Fraction of mock draws taking BGS-Bright-like vs LOWZ-Dark-like efftimes.
EFFTIME_MIX_BGS = 0.5

# Global calibration factor mapping sampled EFFTIME to the `exptime` argument
# of feasibgs simExposure (fit by calibrate_efftime.py; 1.0 until calibrated).
# Slides showed nominal-exptime sims underpredict Dchi2 vs real BGS.
EXPTIME_CALIBRATION_FACTOR = 1.0

# simExposure observing conditions
SIM_AIRMASS = 1.0
SIM_SEEING = 1.1

# speclite filter used for ALL fiber magnitudes (mocks and real data alike).
FIBER_FILTER = "decamDR1noatm-r"
# Softening parameter (nanomaggies) for asinh fiber magnitudes, so the feature
# stays defined when a noisy faint spectrum integrates to <= 0 flux.
ASINH_SOFTENING_NMGY = 0.1

# =============================================================================
# Labels and catalog selection
# =============================================================================
# Catastrophic-redshift threshold: |z_RR - z_true| / (1 + z_true) > this value
# (~1000 km/s). Standardized EVERYWHERE, including the SV3 validation (old
# code mixed 0.003 and 0.0033).
CATASTROPHIC_THRESHOLD = 0.0033

# "Enters the dwarf catalog" selection applied to mocks for the PURITY model
# training set. MUST mirror the real catalog selection in
# code/construct_dwarf_galaxy_catalogs.py (basic_cleaning + ZWARN cut).
# Single source of truth: specz_utils.in_catalog_selection().
CATALOG_Z_MIN = 0.001
CATALOG_Z_MAX = 0.15
CATALOG_ZWARN_ALLOWED = (0,)
CATALOG_SPECTYPE = "GALAXY"

# =============================================================================
# Neural network
# =============================================================================
# Features (order matters; assembled by specz_utils.assemble_features):
#   RFIB_ASINH     - asinh synthetic fiber mag (decamDR1noatm-r on the coadd)
#   LOG_DELTACHI2  - log10(DELTACHI2) from redrock
#   Z_RR           - redrock redshift (measured quantity; resolves
#                    z-dependent failure modes + decouples from mock z-prior)
#   SHAPE_FRAC_i   - W_feat coefficients rescaled to unit sum (i < N_TEMPLATES_FEAT;
#                    last fraction dropped as redundant)
#   FEAT_REDCHI2   - reduced chi2 of the W_feat fit (flags wrong-z / weird spectra)
#   FEAT_OVERLAP   - fraction of W_feat rest grid covered after de-redshifting
#                    at z_RR (small overlap screams wrong redshift)
#
# NOTE (flagged assumption): effective exposure time is NOT a feature; Dchi2
# is the depth/SNR proxy. Confirm later that adding TSNR2/efftime does not
# materially improve the model.
FEATURE_NAMES = (["RFIB_ASINH", "LOG_DELTACHI2", "Z_RR"]
                 + [f"SHAPE_FRAC_{i}" for i in range(N_TEMPLATES_FEAT - 1)]
                 + ["FEAT_REDCHI2", "FEAT_OVERLAP"])

MLP_HIDDEN_LAYERS = (200, 200, 200)
MLP_ALPHA = 1e-4
MLP_BATCH_SIZE = 256
MLP_LEARNING_RATES = np.logspace(-2.5, -4, 10)
MLP_SEED = 20230329
TRAIN_FRACTION = 0.8

# =============================================================================
# Efftime calibration milestone
# =============================================================================
CALIB_N_SPECTRA = 300          # real high-SNR spectra to re-observe
CALIB_SCALE_GRID = [0.8, 1.0, 1.2, 1.4, 1.6]   # exptime scale factors to try

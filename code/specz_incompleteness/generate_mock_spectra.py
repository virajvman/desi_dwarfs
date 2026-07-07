'''
Stage 3a: Generate noise-free mock galaxy spectra from the NNMF bases.

Each mock is a stratified-bootstrap draw of a W_gen coefficient vector from
the bright pool (strata sampled UNIFORMLY across emission strength -- legal
because the NN conditions on spectral type: only coverage matters, not
proportions), reconstructed as flux_rest = W_gen @ c, redshifted to
z ~ U(Z_MIN, Z_MAX), and rescaled so its decamDR1noatm-r fiber magnitude
equals rfib ~ U(RFIB_MIN, RFIB_MAX).

This module is imported by mock_observe.py (generation + observation happen
in-memory per chunk; only the noisy coadd FITS + truth tables hit disk).
Chunks are deterministic: chunk k uses rng seed NNMF_SEED + k, so any chunk
can be regenerated independently (SLURM array friendly).
'''

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import h5py
import numpy as np
from astropy.table import Table

import config
import specz_utils


class MockGenerator:
    """Holds the basis + pool in memory and generates chunks on demand."""

    def __init__(self):
        self.W_gen = np.load(config.GEN_BASIS_NPY)
        self.wave_rest = specz_utils.get_rest_wave()
        self.wave_obs = specz_utils.get_obs_wave()
        with h5py.File(config.POOL_H5, "r") as f:
            self.pool_coeffs = f["COEFFS"][:]
            self.pool_tgids = f["TARGETID"][:]
            self.pool_strata = f["STRATUM"][:]
            self.pool_scores = f["EMISSION_SCORE"][:]
        assert self.W_gen.shape[1] == self.pool_coeffs.shape[1]
        print(f"MockGenerator: {self.pool_coeffs.shape[0]} pool vectors, "
              f"{self.W_gen.shape[1]} generation templates")

    def generate_chunk(self, chunk_id, n_spec=config.CHUNK_SIZE):
        """Generate one chunk of noise-free observed-frame spectra.

        Returns
        -------
        wave_obs : (n_pix,) observed-frame grid
        flux_obs : (n_spec, n_pix) f_lambda in DESI 1e-17 erg/s/cm^2/A units
        efftime : effective exposure time [s] shared by this chunk
        truth : astropy Table with per-mock truth bookkeeping
        """
        rng = np.random.default_rng(config.NNMF_SEED + chunk_id)

        pool_idx = specz_utils.stratified_bootstrap_indices(
            self.pool_strata, n_spec, rng)
        z_true = rng.uniform(config.Z_MIN, config.Z_MAX, n_spec)
        rfib_target = rng.uniform(config.RFIB_MIN, config.RFIB_MAX, n_spec)
        # one efftime per chunk (simExposure takes a scalar exptime); the
        # ~n_chunks distinct draws sample the real depth distribution
        efftime = float(specz_utils.sample_efftime(1, rng)[0])

        coeffs = self.pool_coeffs[pool_idx]                 # (n_spec, n_temp)
        flux_rest = coeffs @ self.W_gen.T                   # (n_spec, n_pix_rest)

        flux_obs = np.empty((n_spec, len(self.wave_obs)), dtype="f8")
        for i in range(n_spec):
            wave_shift = self.wave_rest * (1.0 + z_true[i])
            flux_obs[i] = np.interp(self.wave_obs, wave_shift,
                                    flux_rest[i] / (1.0 + z_true[i]))

        # rescale each spectrum to its target fiber magnitude
        maggies_now = specz_utils.synthetic_fiber_maggies(self.wave_obs, flux_obs)
        maggies_target = specz_utils.pogson_mag_to_maggies(rfib_target)
        factor = np.where(maggies_now > 0, maggies_target / maggies_now, 0.0)
        flux_obs *= factor[:, None]

        truth = Table({
            "MOCKID": chunk_id * config.CHUNK_SIZE + np.arange(n_spec),
            "CHUNK": np.full(n_spec, chunk_id, dtype="i4"),
            "POOL_INDEX": pool_idx.astype("i4"),
            "POOL_TARGETID": self.pool_tgids[pool_idx],
            "STRATUM": self.pool_strata[pool_idx],
            "EMISSION_SCORE": self.pool_scores[pool_idx],
            "Z_TRUE": z_true,
            "RFIB_TARGET": rfib_target,
            "EFFTIME": np.full(n_spec, efftime),
        })
        return self.wave_obs, flux_obs, efftime, truth


def truth_path(chunk_id):
    return f"{config.MOCK_DIR}/truth_chunk_{chunk_id:05d}.fits"


def coadd_path(chunk_id):
    return f"{config.MOCK_DIR}/mock_coadd_chunk_{chunk_id:05d}.fits"


def redrock_path(chunk_id):
    # wrap_rrdesi writes each output with the same basename as its input
    # into the -o directory (matches the old fsps_emi_rr_output_V2 layout).
    return f"{config.REDROCK_DIR}/{os.path.basename(coadd_path(chunk_id))}"

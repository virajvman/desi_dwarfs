'''
Stage 3b: Mock-observe generated spectra -> redrock-ready coadd files.

Wraps the forward-model noise backend behind a small interface so feasibgs
can later be swapped for desisim without touching the rest of the pipeline
(Julien Guy caveat: the feasibgs noise model is optimistic; the efftime
calibration milestone in calibrate_efftime.py is the gate, desisim the
fallback).

Generation (Stage 3a) happens in-memory per chunk; only the noisy coadd FITS
+ truth tables + redrock input file lists are written.

Usage (inside DESI env on a CPU node):
    python mock_observe.py --start 0 --end 250
SLURM array jobs split the chunk range across nodes.
'''

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

import config
import specz_utils
from generate_mock_spectra import MockGenerator, truth_path, coadd_path


class ObserveBackend:
    """Interface: turn noise-free observed-frame spectra into a redrock-ready
    coadd FITS file at the given effective exposure time."""

    def observe(self, wave, flux_1e17, efftime, filename):
        raise NotImplementedError


class FeasiBGSBackend(ObserveBackend):
    """feasibgs fakeDESIspec.simExposure + the SCORES-HDU hack.

    The `exptime` passed to simExposure is efftime scaled by the global
    calibration factor fit in calibrate_efftime.py (simExposure's nominal
    dark-sky exptime is not 1:1 with real EFFTIME_SPEC).
    """

    def __init__(self):
        sys.path.append(config.FEASIBGS_PATH)
        from feasibgs import forwardmodel as FM
        self.fdesi = FM.fakeDESIspec()

    def observe(self, wave, flux_1e17, efftime, filename):
        exptime = efftime * config.EXPTIME_CALIBRATION_FACTOR
        self.fdesi.simExposure(wave=wave, flux=np.asarray(flux_1e17),
                               airmass=config.SIM_AIRMASS,
                               exptime=exptime, seeing=config.SIM_SEEING,
                               filename=filename)
        specz_utils.add_scores_hdu(filename, n_rows=len(flux_1e17))
        return filename


BACKENDS = {"feasibgs": FeasiBGSBackend}


def run_chunks(start, end, backend_name="feasibgs", overwrite=False):
    specz_utils.check_dirs()
    gen = MockGenerator()
    backend = BACKENDS[backend_name]()

    file_list = []
    for chunk_id in range(start, end):
        out_coadd = coadd_path(chunk_id)
        out_truth = truth_path(chunk_id)
        if os.path.exists(out_coadd) and os.path.exists(out_truth) and not overwrite:
            print(f"chunk {chunk_id}: exists, skipping")
            file_list.append(out_coadd)
            continue

        wave, flux, efftime, truth = gen.generate_chunk(chunk_id)
        backend.observe(wave, flux, efftime, out_coadd)
        truth.write(out_truth, overwrite=True)
        file_list.append(out_coadd)
        print(f"chunk {chunk_id}: wrote {out_coadd} (efftime={efftime:.0f}s)")

    list_path = f"{config.MOCK_DIR}/redrock_input_list_{start:05d}_{end:05d}.ascii"
    with open(list_path, "w") as f:
        f.write("\n".join(file_list) + "\n")
    print(f"Wrote redrock input list -> {list_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--start", type=int, required=True)
    p.add_argument("--end", type=int, required=True,
                   help="exclusive; total chunks = N_MOCKS_TOTAL / CHUNK_SIZE")
    p.add_argument("--backend", default="feasibgs", choices=list(BACKENDS))
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()
    run_chunks(args.start, args.end, args.backend, args.overwrite)

# PSF prepass — quick reference

Builds the empirical Legacy Surveys coadd PSF per object into per-brick HDF5
shards (`code/psf_store.py`), the store `scarlet_photo`'s fitter reads from.
Sibling of the image-cutout job (`job_scripts/image_cutouts/general/`).

For the full design rationale (why a separate prepass, how `build_coadd_psf`
reproduces the viewer's `coadd-psf` endpoint, standardization details) see
`code/scarlet_photo/DESIGN.md` §6. This file is just how to run it.

## Files

- `many_psfs_general.py` — the actual builder (MPI + per-rank multiprocessing,
  runs inside the `dstndstn/cutouts` container).
- `psfs_cnn_general.sh` — shell wrapper computing the MPI task layout, called
  by the sbatch script.
- `get_psfs_general.sbatch` — **production job.** Loops over one or more input
  catalogs (see below) against the same PSF store.
- `validate_psf_debug.sbatch` — **the validation gate below**, as a
  `qos=debug` job for fast turnaround.

## Manual validation gate — run this BEFORE every production batch

The container build (`build_coadd_psf`, CFS-only, no network) is supposed to
exactly reproduce the legacysurvey.org viewer's own `coadd-psf` URL endpoint
(the same computation, just run locally instead of over HTTP). `--validate-url`
proves that on a small sample by fetching both and comparing them directly —
**URL is treated as ground truth**; the container build is what's being
checked.

**Why this matters:** the container path re-implements the viewer's endpoint
(CCD selection, per-CCD PsfEx rendering, inverse-variance coadd weighting) by
calling the *installed* `imagine`/`legacypipe`/`tractor` libraries directly,
not by hitting the real endpoint — so it's the kind of thing that can silently
drift out of sync with the viewer (a library version bump, a subtly different
CCD cut, a rendering-center convention that only matters at the sub-pixel
level) without ever raising an exception. The fit downstream has no other way
to notice a systematically-wrong PSF; deconvolving against the wrong empirical
PSF doesn't error, it just quietly biases every model. This gate is the only
check between "the code runs" and "the code is right."

**Run it:**

```bash
sbatch validate_psf_debug.sbatch
```

(or manually: `shifter --image dstndstn/cutouts:dvsro3 python3 many_psfs_general.py
--catalog-path <cat.fits> --outdir-data <psf-store-dir> --validate-url 30
--validate-outdir <panel-dir> --nompi` — needs network for the URL fetch,
unlike the production job, and does **not** write to the PSF store regardless
of what `--outdir-data` points at.)

**What it produces**, in `--validate-outdir`:
- `psf_val_<TARGETID>.png` — one panel per validated object: container | URL |
  residual, per band, log-stretched.
- `psf_validation_metrics.csv` — per-object, per-band: `max_abs_frac`
  (max|Δ|/peak), `flux_ratio` (container/URL), `centroid_shift_px`.
- A printed summary (median/max across the sample) against these tolerances:

| metric | tolerance |
|---|---|
| `max\|Δ\|/peak` | < 1e-3 |
| flux ratio | within 0.5% of 1.0 |
| centroid shift | < 0.05 px |

**This is a manual gate, not an automated assertion** — deliberately not
wired into the production job (that would reintroduce a network dependency
into an otherwise offline batch). Eyeball the printed summary *and* a few
panels together before submitting `get_psfs_general.sbatch`. If a
**systematic** gap appears (most objects exceed tolerance, not just a couple
of edge cases), diagnose the root cause (survey-ccds/DR version mismatch,
`ccd_cuts` drift between the installed libraries and the live viewer) rather
than loosening the thresholds — the thresholds encode "this PSF is accurate
enough to deconvolve against," not an arbitrary pass bar.

## Production run

```bash
sbatch get_psfs_general.sbatch
```

Edit `CATALOG_PATHS` (an array) at the top of the script to add/remove input
catalogs — the store is unified across sample classes (keyed by
`BRICKNAME`/`TARGETID` only), so running e.g. both a BGS_BRIGHT and an SGA
catalog through the same job just works; any `TARGETID` overlap between them
costs nothing extra thanks to the manifest-based incremental skip (see
`psf_store.py`). **Never run two production PSF-writing jobs against the
store concurrently** — sequential catalogs within one job submission are
fine, but two separate concurrent `sbatch` submissions racing the same store
are not.

Incremental: objects already in the PSF manifest, or tombstoned in
`permanently_failed.csv`, are skipped automatically — a job that times out
partway through can simply be resubmitted.

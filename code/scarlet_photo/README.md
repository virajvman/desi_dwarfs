# scarlet_photo — quick reference

SCARLET whole-image-plane photometry / reconstruction pipeline for DESI dwarfs.
Per object: model the full grz image plane, group components into an initial
dwarf reconstruction, and emit a VI bundle fragment plus benchmark magnitudes.
Then consolidate the per-object fragments into a per-brick store.

**Status:** stages 1 + 2 + 3 BUILT (stage 3 = the VI tool, `recon_vi_scarlet`,
a separate sibling package with its own README) and locally validated
end-to-end. Stage 4 (post-VI model store) is not built yet.

For the *why* behind every design choice and the full bundle contract, see
`DESIGN.md`. This file is just how to run it.

## Input catalog columns

`--input-catalog` is any FITS table with one row per object to fit — a
sub-selected or hand-modified table works fine as long as these columns are
present. Grounded directly in what `inputs.py::load_object` reads (nothing
guessed):

**Required** (missing/invalid → the object hard-fails with `InputError`, or
the row can't even be read):
- `TARGETID` (int), `RA`, `DEC` (float), `BRICKNAME` (str) — cutout/PSF store keys.
- `FILE_PATH` (str) — per-object dir; must contain `source_cat_f.fits` (or
  `source_cat_f_more.fits`).
- `IMAGE_SIZE_PIX` (int) — the cutout size to request. Still required even
  when overriding with `--fixed-box-size` (see below) — it's read as the
  *native* size, used to correctly shift the per-object pixel-position
  columns/files into the smaller working frame.

**Strongly recommended** (soft default via a missing-column fallback — no
crash, but silently degrades correctness):
- `Z` — selects the per-z-bin GMM color-color bin for grouping
  (`grouping_rule='or'`/`'gmm'`). Missing → defaults to `NaN` → silently
  always uses the lowest (z≈0) GMM bin for that object, regardless of its
  real redshift.
- `APER_PARAMS_ISOLATE` (`[semi_major_σ, b/a, θ_rad]`) + `APER_CEN_XY_PIX_ISOLATE`
  (`[x_pix, y_pix]`) — needed for the R4.25 aperture magnitude only. Missing
  or non-finite → the R4 mag comes back `NaN` (gracefully undefined); the
  TOTAL magnitude and the fit itself are unaffected. `APER_CEN_XY_PIX_ISOLATE`
  is a native-frame pixel position — `--fixed-box-size` shifts it
  automatically using `IMAGE_SIZE_PIX`, no extra action needed.

**Optional / informational only** (loaded but never read by any fit logic —
safe to omit entirely): `MASKBITS`, `is_south`, `LOGM_M24_FIDU_CORR`,
`COG_MAG_{G,R,Z}_ISOLATE` (the last is handy to keep anyway — it's what you
compare `SCARLET_MAG_*_TOTAL` against for the benchmark, just done externally
by whoever reads the output, not consumed inside the fit).

Per-object `FILE_PATH` directory (referenced by `source_cat_f.fits` above):
`segment_map_v2.npy`, `star_mask.npy`, `noise_per_band_rms.npy`,
`fiber_pix_pos.npy` are all consumed if present, with a documented graceful
fallback if any are missing (see `DESIGN.md` §5) — none of them are hard
requirements.

## Run it

Both commands are run from `code/` (or anywhere on `sys.path`); the modules
self-bootstrap the package path.

### Stage 1 — FIT (Python with scarlet importable)

```bash
# smoke test first: a few objects, with the diagnostic plot
python -m scarlet_photo.driver --input-catalog <filtered_w_aper_mags>.fits \
    --fixed-box-size 350 --limit 5 --save-plots --ncores 1

# full run
python -m scarlet_photo.driver --input-catalog <cat>.fits --fixed-box-size 350 --ncores 128
```

`--fixed-box-size 350` matches the locked, locally-validated config (see
`DESIGN.md`) — the production catalog's native `IMAGE_SIZE_PIX` varies and is
always ≥350, so this fits everything at the validated size regardless.

Outputs: per-object `{FILE_PATH}/scarlet_vi_bundle.h5` (the VI fragment) +
`<cat>_scarlet_mags.fits` (the benchmark magnitudes).

### Stage 2 — CONSOLIDATE (numpy/h5py only)

```bash
python -m scarlet_photo.consolidate --input-catalog <cat>.fits --ncores 64
```

Output: `.../dwarf_cutouts/scarlet_bundles/{brick[:3]}/{brick}.h5` +
`scarlet_bundle_manifest.csv`.

## Environment

- **Stage 1** needs only that `import scarlet` works. There is *no* dedicated
  "scarlet env" — scarlet was a local source install (`pip install -e .` from
  `galaxy_prior_proj/scarlet`) into the default Python, which is why the
  prototype `scarlet_photo.py` just does a bare `import scarlet`. Run it in
  whatever Python has your locally-installed scarlet on its path (the same
  interpreter used for the prototype). Sanity check:

```bash
python -c "import scarlet; print(scarlet.__file__)"
```

- **Stage 2** is numpy/h5py/astropy only, so it can run in that same Python, or
  in the cutouts container `dstndstn/cutouts:dvsro3` (the image used by
  `job_scripts/image_cutouts/psf/`):

```bash
shifter --image=dstndstn/cutouts:dvsro3 python3 -m scarlet_photo.consolidate --input-catalog <cat>.fits --ncores 64
```

## Flags

**`driver.py` (Stage 1):**

- `--input-catalog` (required) — pre-filtered `*_w_aper_mags` FITS; selection is done upstream.
- `--output` — output mags FITS (default `<input>_scarlet_mags.fits`).
- `--ncores` — worker processes (default 128).
- `--overwrite` — re-fit objects whose fragment already exists.
- `--tgids` — restrict to these TARGETIDs.
- `--limit` — fit only the first N rows (testing).
- `--save-plots` — write the per-object diagnostic panel (see below).
- `--cutouts-dir`, `--psfs-dir` — override store locations (default: the
  canonical NERSC paths hardcoded in `cutout_store.py`/`psf_store.py`; no
  flags needed on NERSC).
- `--fixed-box-size N` — fit every object at a uniform `N`px box regardless of
  its own `IMAGE_SIZE_PIX` (must be ≤ every row's native size — the store only
  ever crops down). Purely in-memory at load time; no files written. Locked
  default for the real run: `350`.
- `--model-psf-sigma` — override the model/target PSF sigma.
- `--detection-method {chi2,ivar_weighted,sum}` — detection coadd.
- `--detect-scale N` — wavelet scale index for footprint peaks (config default `1`).
- `--fit-lsb` / `--no-fit-lsb` — fit a global LSB StarletSource (config default: **off**).
- `--lsb-monotonic` / `--no-lsb-monotonic` — monotonic vs L0-thresholded LSB (moot with `--no-fit-lsb`).
- `--star-shift-free` / `--no-star-shift-free` — free sub-pixel star shift, init at Gaia (config default: **on**).
- `--grouping-rule {or,bluebox,gmm}` — initial membership rule (config default: **`gmm`**).
- `--gmm-model-dir DIR` — load GMM pickles from a local dir instead of
  `aperture_photo`'s hardcoded NERSC pscratch path. Not needed on NERSC unless
  that path has been purged (pscratch is not permanent storage).

Incremental skip: re-running resumes (objects with a fragment are skipped); use
`--overwrite` to redo them.

**`consolidate.py` (Stage 2):**

- `--input-catalog` (required) — the catalogue of objects that were fit.
- `--bundle-dir` — override the bundle store dir.
- `--ncores` — worker processes (default 64).
- `--overwrite` — re-pack objects already in a shard.
- `--limit` — process only the first N rows (testing).
- `--tgids` — restrict to these TARGETIDs.

Skip: objects already in a shard are skipped; the manifest is rebuilt from the
shards on every run.

## Diagnostic plot + config

- `--save-plots` writes `{FILE_PATH}/scarlet_diagnostic.pdf`:
  **data | full scarlet model | residual | initial dwarf reconstruction**.
- Every other tunable (model PSF sigma, detection params, convergence, grouping
  cuts, R4 aperture scale, ...) lives in `config.py` (`ScarletConfig`). See
  `config.py` for the knobs and `DESIGN.md` for the reasoning.

# scarlet_photo — quick reference

SCARLET whole-image-plane photometry / reconstruction pipeline for DESI dwarfs.
Per object: model the full grz image plane, group components into an initial
dwarf reconstruction, and emit a VI bundle fragment plus benchmark magnitudes.
Then consolidate the per-object fragments into a per-brick store.

**Status:** stages 1 + 2 BUILT, awaiting NERSC validation. Stage 3 (VI tool,
`recon_vi_scarlet`) and stage 4 (post-VI model store) are not built yet.

For the *why* behind every design choice and the full bundle contract, see
`DESIGN.md`. This file is just how to run it.

## Run it

Both commands are run from `code/` (or anywhere on `sys.path`); the modules
self-bootstrap the package path.

### Stage 1 — FIT (Python with scarlet importable)

```bash
# smoke test first: a few objects, with the diagnostic plot
python -m scarlet_photo.driver --input-catalog <filtered_w_aper_mags>.fits --limit 5 --save-plots --ncores 1

# full run
python -m scarlet_photo.driver --input-catalog <cat>.fits --ncores 128
```

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
- `--cutouts-dir`, `--psfs-dir` — override store locations.
- `--model-psf-sigma` — override the model/target PSF sigma.
- `--detection-method {chi2,ivar_weighted,sum}` — detection coadd.
- `--grouping-rule {or,bluebox,gmm}` — initial membership rule.

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

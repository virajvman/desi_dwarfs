# Handoff: Incremental FastSpecFit for ~8k Missing Dwarf Objects

**Repo:** `desi_dwarfs`
**Date:** 2026-06-12
**Context:** Custom FastSpecFit run on ~450k dwarf galaxies; ~8k objects in the final dwarf catalog MAIN extension lack rows in the merged fastspec catalog (the catalog changed slightly after the production run). Goal: fit only those missing objects without re-running the full ~450k production job, and without touching the canonical 450k outputs.

---

## 1. Problem statement

### What the user wanted

- Run FastSpecFit on ~8k objects missing from the existing merged catalog.
- Combine results with the existing `fastspec-iron-dr1-dwarfs.fits` catalog.
- Avoid wasting compute re-fitting the full ~450k sample.
- Avoid modifying the expensive, canonical 450k per-healpix tree.

### Source of truth for "missing"

- **Dwarf catalog:** `desi_dr1_dwarf_catalog.fits`, HDU `MAIN` (consolidated final catalog).
- **Existing fastspec:** `fastspec-iron-dr1-dwarfs.fits`, HDU `3` (FASTSPEC extension).
- A TARGETID is "missing" if it appears in dwarf MAIN but not in the merged fastspec FASTSPEC HDU.
- This mirrors `report_fastspec_coverage()` in `code/consolidate_photometry.py`.

### Why the obvious "resume" doesn't work

`mpi-fastspecfit` (fastspecfit 3.4.3) does **not** resume at TARGETID granularity. Without `--overwrite`, it skips when the **per-healpix output file** already exists ([`fastspecfit/py/fastspecfit/mpi.py:347-348`](../fastspecfit/py/fastspecfit/mpi.py#L347-L348)):

```python
for ii, outfile in enumerate(outfiles):
    if os.path.isfile(outfile) and not overwrite:
        todo[ii] = False
```

So pointing a fit at the canonical tree with `--samplefile` of the missing IDs would **skip every healpix that already exists** (which is all of them, for this run) and fit nothing.

Worse, with `--samplefile`, mpi-fastspecfit fits **only** the passed TARGETIDs per healpix and writes **one FITS per healpix containing only those IDs** ([`mpi.py:756-762`](../fastspecfit/py/fastspecfit/mpi.py#L756-L762)). So re-running an existing healpix with only the missing IDs (after deleting its file) would **clobber the already-fit neighbors** in that healpix.

### The fix: fit into a SEPARATE scratch tree

Rather than deleting/clobbering files in the canonical tree, the incremental fit writes to a **fresh scratch `--outdir-data`** (`fastspecfit_incremental_run/`). Because that tree starts empty:

- No healpix is ever skipped (nothing exists there yet) — every requested healpix is fit.
- The canonical 450k tree is **never read or written** by the fit.
- We fit **only the missing TARGETIDs** — no neighbor re-fitting, no surgical deletes, no wasted compute.
- No disk/catalog divergence: the canonical tree is untouched, so a future full `--merge` is unaffected.

The scratch per-healpix files contain exactly the missing IDs for each healpix; combine harvests them all.

> **Note on gzip:** fastspec output is always gzipped (`.fits.gz`) in 3.4.3 ([`mpi.py:282-283`](../fastspecfit/py/fastspecfit/mpi.py#L282-L283)). The existing `--merge` only "worked" against `.fits` paths because cfitsio transparently falls back to `.fits.gz`. The incremental combine does its own `os.path.isfile` check, so it explicitly resolves both suffixes (`_resolve_existing`).

---

## 2. Solution overview

A three-step incremental workflow:

```
MAIN catalog TARGETIDs
        +
Merged fastspec catalog (HDU 3)
        ↓
prepare_incremental_fastspec_sample.py
        ↓
Incremental sample FITS (missing IDs only) + manifest
        ↓
mpi-fastspecfit --outdir-data=<SCRATCH>  (NO --overwrite)
        ↓
Per-healpix outputs in the SCRATCH tree
        ↓
combine_incremental_fastspec.py (vstack missing rows only)
        ↓
fastspec-iron-dr1-dwarfs-v2.fits
```

**Design choices (confirmed with user):**

- Coverage check: merged catalog HDU 3 (not a per-healpix scan on disk).
- Fit destination: **separate scratch tree** — canonical 450k outputs are never modified.
- Run sample: **missing TARGETIDs only** — no neighbor re-fitting.
- Combine strategy: **vstack incremental** rows onto the existing merged catalog (do not re-merge all 450k per-healpix files).

---

## 3. Files implemented

| File | Purpose |
|---|---|
| `job_scripts/fastspec/run_custom_fastspec_job.sh` | Header comment documenting healpix-level skip + pointer to incremental script |
| `code/prepare_incremental_fastspec_sample.py` | Diff catalogs, write missing-only sample + manifest (scratch paths) |
| `job_scripts/fastspec/run_incremental_fastspec_job.sh` | SLURM driver: prep → fit into scratch tree → combine |
| `code/combine_incremental_fastspec.py` | Vstack new rows onto existing merged multi-HDU catalog |
| `job_scripts/fastspec/combine_incremental_fastspec_cat.sh` | Standalone combine-only SLURM wrapper |

---

## 4. Step 1: `prepare_incremental_fastspec_sample.py`

### Inputs (CLI, with Perlmutter defaults)

| Argument | Default |
|---|---|
| `--dwarf-catalog` | `/pscratch/sd/v/virajvm/desi_dwarf_catalogs/dr1/v1.0/desi_dr1_dwarf_catalog.fits` |
| `--dwarf-hdu` | `MAIN` |
| `--fastspec-merged` | `.../fastspecfit_custom_run/iron/catalogs/fastspec-iron-dr1-dwarfs.fits` |
| `--fastspec-hdu` | `3` |
| `--outdir-data` | `.../fastspecfit_custom_run/` (canonical; **read-only**, informational only) |
| `--incremental-outdir` | `.../fastspecfit_incremental_run/` (scratch; fit writes here, combine reads here) |
| `--specprod` | `iron` |
| `--out-sample` | `.../catalog_dr1_dwarfs/desi_dr1_dwarfs_fastspec_incremental.fits` |
| `--manifest` | `<out-sample>.manifest.json` |
| `--dry-run` | Write manifest only; skip sample FITS |

### Logic

1. Read dwarf MAIN; require `SURVEY`, `PROGRAM`, `HEALPIX`, `TARGETID`.
2. Read `TARGETID` from merged fastspec HDU 3 → `have_tids`.
3. `missing = dwarf[~isin(TARGETID, have_tids)]`.
4. Run sample = **missing TARGETIDs only** (dedup by TARGETID).
5. For each unique healpix in `missing`, record the **scratch** per-healpix path the fit will write (`<incremental_outdir>/<specprod>/healpix/<survey>/<program>/<hpx//100>/<hpx>/fastspec-<survey>-<program>-<hpx>.fits.gz`), the missing count, and `in_canonical` (whether the canonical tree already covers it — informational).
6. Write manifest JSON.
7. Print grep-friendly summary: `INCREMENTAL FASTSPEC PREP`.

The scratch path layout mirrors `fastspecfit.mpi.findfiles`/`plan` (`outdir_data/specprod/healpix/...`).

### Manifest fields (for verification)

- `n_missing_targetids` — should be ~8k
- `n_run_sample_rows` — equals `n_missing_targetids` (missing only)
- `n_healpix` — unique healpix to fit
- `n_healpix_new_region` — healpix **not** present in the canonical 450k tree (informational; if > 0, those are genuinely new regions)
- `missing_targetids` — full list
- `healpix` — per-healpix metadata (survey/program/healpix, scratch `outfile`, `n_missing_targets`, `in_canonical`)
- `incremental_outdir` / `outdir_data` — scratch and canonical roots

### Edge case: 0 missing

Writes manifest with `n_missing_targetids: 0` and exits 0. Job script reads manifest and exits without fitting.

---

## 5. Step 2: `run_incremental_fastspec_job.sh`

### SLURM allocation

- 2 nodes (vs 10 for full 450k run), 2 hr, `mp=16`.
- Same fastspecfit 3.4.3 module, templates, emlines, constraints as production.

### Workflow

1. **Prep:** `prepare_incremental_fastspec_sample.py` (honors `DRY_RUN=1`).
2. **Guard:** Read `n_missing_targetids` from manifest; exit if 0.
3. **Warm-up:** Numba cache warm-up (same as production, `/tmp`).
4. **Fit:** `mpi-fastspecfit --samplefile=<incremental sample> --outdir-data=<SCRATCH>` — **no `--overwrite`**, writes only into the scratch tree.
5. **Combine:** `combine_incremental_fastspec.py` → writes `fastspec-iron-dr1-dwarfs-v2.fits`.

No delete step: the canonical tree is never modified.

### Knobs

- `DRY_RUN=1` — prep only; inspect manifest before spending compute.

---

## 6. Step 3: `combine_incremental_fastspec.py`

### Why not `mpi-fastspecfit --merge`?

We only want the ~8k new rows appended. So: read the scratch per-healpix files (from the manifest `healpix` list), filter to `missing_targetids` only, vstack onto the existing merged catalog.

### Logic

1. Load manifest; get `missing_targetids` and scratch healpix file paths.
2. Read existing merged catalog: `METADATA`, `SPECPHOT`, `FASTSPEC` + primary header.
3. Assert no overlap between `missing_targetids` and existing merged TARGETIDs.
4. For each scratch per-healpix output, resolve `.fits`/`.fits.gz` (`_resolve_existing`) and read via `fastspecfit.mpi.read_to_merge_one`.
5. Keep rows where `TARGETID ∈ missing_targetids`.
6. Assert all missing TARGETIDs found in the scratch outputs.
7. `vstack` each HDU: `existing + new_rows` (METADATA/SPECPHOT/FASTSPEC stay row-aligned).
8. Assert no duplicate TARGETIDs; assert `len(out) == len(existing) + n_missing`.
9. Write via `fastspecfit.io.write_fastspecfit` to `--out-merged` (default: `fastspec-iron-dr1-dwarfs-v2.fits`), same single-file multi-HDU layout as the input (HDU 3 = FASTSPEC preserved).
10. Optional `--replace-original` copies v2 over the canonical merged path.

### Output

- Default: `fastspec-iron-dr1-dwarfs-v2.fits` (does **not** overwrite canonical file unless `--replace-original`).
- User should review v2, then `mv` or use `--replace-original`.

---

## 7. Verification checklist for reviewing agent

### Code review

- [x] Prep scratch path layout matches `fastspecfit.mpi.findfiles`/`plan`.
- [x] Run sample = missing TARGETIDs only (no neighbor expansion).
- [x] Fit writes to scratch `--outdir-data`; canonical tree never deleted or written.
- [x] Production/incremental fit does **not** use `--overwrite`.
- [x] Combine resolves `.fits`/`.fits.gz` (`_resolve_existing`) — no silent drop of gzipped outputs.
- [x] Combine vstack keeps METADATA/SPECPHOT/FASTSPEC row-aligned.
- [x] Combine filters to `missing_targetids` only.
- [x] Duplicate TARGETID checks before and after vstack.
- [x] 0-missing edge case: manifest written, job exits cleanly.

### Dry-run on Perlmutter (before fit)

```bash
DRY_RUN=1 sbatch job_scripts/fastspec/run_incremental_fastspec_job.sh
cat /pscratch/sd/v/virajvm/catalog_dr1_dwarfs/desi_dr1_dwarfs_fastspec_incremental.fits.manifest.json
```

Check:

- `n_missing_targetids` ≈ 8000
- `n_run_sample_rows` == `n_missing_targetids`
- `n_healpix_new_region` — expected 0 for this run (all missing objects are in already-processed regions); if > 0, those are genuinely new sky regions, which is fine with the scratch-tree design.

### Post-run verification

```python
from consolidate_photometry import report_fastspec_coverage

report_fastspec_coverage(
    "/pscratch/sd/v/virajvm/desi_dwarf_catalogs/dr1/v1.0/desi_dr1_dwarf_catalog.fits",
    "/pscratch/sd/v/virajvm/desi_dwarf_catalogs/fastspecfit_custom_run/iron/catalogs/fastspec-iron-dr1-dwarfs-v2.fits",
    fastspec_hdu=3,
)
# Expect: 0 missing TARGETIDs
```

### Row-count sanity

- `len(v2 METADATA) == len(v1 METADATA) + n_missing_targetids`
- No duplicate TARGETIDs in v2

---

## 8. Known risks / limitations

1. **Schema drift:** Incremental run must use the same fastspecfit version (3.4.3), templates, emlines, constraints as the original 450k run, or vstack column mismatch / inconsistent fits.
2. **Dwarf MAIN must have `SURVEY`, `PROGRAM`, `HEALPIX`:** Required for the mpi-fastspecfit samplefile contract.
3. **Canonical catalog promotion is manual:** v2 is written alongside v1; user must `mv` or `--replace-original` after review.
4. **Scratch per-healpix outputs are not in the canonical tree:** Brand-new (new-region) objects' per-healpix files live only in `fastspecfit_incremental_run/`. The merged catalog is the source of truth and re-diffs against it on the next run, so this is intentional — but a future *full* `--merge` of the canonical tree would not include those new-region objects.
5. **Downstream not in scope:** `consolidate_photometry.py` / `add_nebular_props.py` re-run after promoting the merged catalog is the user's responsibility.

---

## 9. Related existing code (patterns mirrored)

| Existing | Pattern used |
|---|---|
| `code/download_spectra.py` | Incremental sync / resume without `--overwrite` |
| `code/tractor_model.py` | Filter catalog against existing output catalog by TARGETID |
| `code/consolidate_photometry.py::report_fastspec_coverage` | MAIN vs fastspec diff |
| `job_scripts/fastspec/combine_custom_fastspec_cat.sh` | Full merge via `mpi-fastspecfit --merge` (not used for incremental) |
| `job_scripts/fastspec/run_custom_fastspec_job.sh` | Production fit knobs (templates, constraints, mp, nodes) |

---

## 10. Key paths (Perlmutter)

| Item | Path |
|---|---|
| Dwarf catalog MAIN | `/pscratch/sd/v/virajvm/desi_dwarf_catalogs/dr1/v1.0/desi_dr1_dwarf_catalog.fits` |
| Merged fastspec (input) | `/pscratch/sd/v/virajvm/desi_dwarf_catalogs/fastspecfit_custom_run/iron/catalogs/fastspec-iron-dr1-dwarfs.fits` |
| Merged fastspec (output) | `.../fastspec-iron-dr1-dwarfs-v2.fits` |
| Canonical per-healpix tree | `/pscratch/sd/v/virajvm/desi_dwarf_catalogs/fastspecfit_custom_run/iron/healpix/...` (read-only) |
| **Scratch incremental tree** | `/pscratch/sd/v/virajvm/desi_dwarf_catalogs/fastspecfit_incremental_run/iron/healpix/...` (fit writes here) |
| Incremental sample | `/pscratch/sd/v/virajvm/catalog_dr1_dwarfs/desi_dr1_dwarfs_fastspec_incremental.fits` |
| Manifest | `.../desi_dr1_dwarfs_fastspec_incremental.fits.manifest.json` |

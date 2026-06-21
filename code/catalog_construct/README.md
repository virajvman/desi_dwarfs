# catalog_construct

Standalone, reusable catalog-construction utilities (not wired into the main
dwarf pipeline). Together these build a cleaned **BGS_BRIGHT + BGS_FAINT + LOW_Z**
catalog from the DESI Y5 **matterhorn** `zall` zcatalog (v2).

Pipeline (two stages):

```
matterhorn zall  ──①select──▶  *_raw.fits  ──②crossmatch──▶  *_clean.fits
 (base+imaging)                (targeting +                 (+ FRACFLUX cut,
                                redshift cuts,               + dereddened mags)
                                raw mags)
```

## ① `select_matterhorn_bgs_lowz.py`

Selects **BGS_BRIGHT + BGS_FAINT + LOW_Z** (spare-fiber secondary) targets across
**main + SV1/2/3**, and writes a FITS catalog with per-sample boolean flags
(`IS_BGS_BRIGHT`, `IS_BGS_FAINT`, `IS_LOWZ`).

- Reads only the **base** (`ZCATALOG`) and **imaging** (`ZCATALOG_IMAGING`)
  files — the row-matched, ~50 GB `-extra` file is never opened.
- Targeting masks imported **by name** from `desitarget` (main + sv1/2/3); LOW_Z
  = all `LOW_Z_TIER0–3`.
- Redshift cut: `GOOD_SPEC & ZWARN_BEST==0 & DELTACHI2_BEST>40`
  (= the official `GOOD_Z_BGS` for BGS; same galaxy criterion applied to LOW_Z,
  which has no precomputed good-z flag). `GOOD_SPEC` is recomputed from base
  columns to avoid reading `-extra`.
- Science cuts: `SPECTYPE_BEST=='GALAXY'`, `0.001 < Z_BEST < 0.2`,
  `ZCAT_PRIMARY==True` (all toggleable; `--zmax` bumps the redshift range).
- Magnitudes are **raw** `22.5 - 2.5·log10(FLUX)`.
- Carries `RELEASE/BRICKID/BRICK_OBJID/BRICKNAME/PHOTSYS/EBV` so stage ② is a
  deterministic key join needing no re-read.

```bash
python select_matterhorn_bgs_lowz.py --help
```

Run on an interactive/compute node (one pass over the ~30 GB base file).

## ② `crossmatch_tractorphot.py`

There is **no matterhorn `tractorphot` VAC**, so this gathers Tractor photometry
on the fly with `desispec.io.photo.gather_tractorphot` (matching on
`RELEASE+BRICKID+BRICK_OBJID`, 1″ positional fallback). It adds
`FRACFLUX_{G,R,Z}`, `FRACMASKED_*`, `FRACIN_*`, `RCHISQ_*`, `NOBS_*`,
`MW_TRANSMISSION_{G,R,Z}`, dereddened `MAG_{G,R,Z}_DERED`, and the flags
`TRACTORPHOT_MATCH` / `FRACFLUX_PASS`.

- **FRACFLUX cut:** keep only objects with `FRACFLUX_G/R/Z` *all* `< 0.35`
  (`--fracflux-max`). Unmatched objects get `FRACFLUX=NaN` → fail.
- **Dereddening:** `MAG_X_DERED = 22.5 - 2.5·log10(FLUX_X / MW_TRANSMISSION_X)`,
  using the Tractor `MW_TRANSMISSION` (identical to the iron pipeline's).
- By default writes only matched & FRACFLUX-passing rows; `--keep-all` writes
  every row with the flag columns instead.
- **Cost** ∝ number of unique bricks the sample touches (one Tractor file read
  each). Parallelized over bricks with `--nproc`; the unique-brick count is
  printed up front.

```bash
python crossmatch_tractorphot.py --nproc 128          # default
python crossmatch_tractorphot.py --nproc 1            # serial (small samples)
python crossmatch_tractorphot.py --keep-all           # flag, don't drop
```

## Running the full chain on Perlmutter

Job scripts live in `../../job_scripts/catalog_construct/`. From a **login node**:

```bash
./submit_matterhorn_catalog.sh
```

This submits both stages at once as an `afterok` dependency chain (stage ② waits
for stage ① to succeed), mirroring `submit_make_cat.sh`. Re-running just stage ②
(e.g. after changing the FRACFLUX threshold) never repeats the 30 GB base read.

Output (default): `/pscratch/sd/v/virajvm/matterhorn/matterhorn_pix_bgs_lowz_clean.fits`.

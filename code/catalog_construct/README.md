# catalog_construct

Standalone, reusable catalog-construction utilities (not wired into the main
dwarf pipeline). Together these build a cleaned **BGS_BRIGHT + BGS_FAINT + LOW_Z**
catalog from the DESI Y5 **matterhorn** `zall` zcatalog (v2).

Pipeline (two stages):

```
matterhorn zall ──①select──▶ *_raw.fits ──②crossmatch──┬─▶ *_phot.fits  (all + photometry)
 (base+imaging)              (targeting,               └─▶ *_clean.fits (FRACFLUX cut)
                             redshift cuts)
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

Attaches the photometric columns the zcatalog lacks (`FRACFLUX_*`,
`MW_TRANSMISSION_*`, `FRACMASKED_*`, `FRACIN_*`, `RCHISQ_*`, `NOBS_*`,
`FIBERFLUX_*`), recomputes raw + dereddened `MAG_{G,R,Z}[_DERED]`, and writes
**two** catalogs. Photometry comes from two sources by target class, recorded in
`PHOT_SOURCE`:

- **BGS** (BGS_BRIGHT/FAINT, incl. objects that are *also* LOW_Z) →
  `desispec.io.photo.gather_tractorphot` against the DR9 Tractor catalogs
  (matched on `RELEASE+BRICKID+BRICK_OBJID`, 1″ positional fallback). There is
  no matterhorn `tractorphot` VAC, so this is gathered on the fly.
- **LOW_Z-only** (IS_LOWZ & not BGS) → positional 1″ match to Elise's DR9 LOW_Z
  target catalogs (north + south, using the `remove_south_lowz`/`clean_south_lowz`
  footprint logic + south-only fallback). **All** photometry is taken from Elise.
  LOW_Z-only objects that miss Elise fall back to `gather_tractorphot`
  (`PHOT_SOURCE='tractorphot_fallback'`).

Adds `PHOT_SOURCE` (`tractorphot`/`lowz_target`/`tractorphot_fallback`/`none`),
`PHOT_MATCH`, `FRACFLUX_PASS`, and `LS_ID` (dedup key for shared sources).

**Outputs (both written):**
- `*_phot.fits` — every z<0.2 selected object with photometry attached (no cut).
- `*_clean.fits` — matched objects passing `FRACFLUX_G/R/Z` *all* `< 0.35`.

- **Dereddening:** `MAG_X_DERED = 22.5 - 2.5·log10(FLUX_X / MW_TRANSMISSION_X)`.
- **Cost** ∝ unique bricks the BGS/fallback set touches; parallelized over bricks
  with `--nproc`, with a progress + ETA log line.

```bash
python crossmatch_tractorphot.py --nproc 128          # default
python crossmatch_tractorphot.py --nproc 1            # serial (small samples)
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

## loa dwarf catalog (`select_loa_dwarfs.py`)

The **loa** counterpart of stage ①, for the loa production (still **healpix**, and
its zcatalog is in the **iron/DR1 format** — one `zall-pix-loa.fits` file, ext
`ZCATALOG`, redshift cols `Z/ZWARN/DELTACHI2/SPECTYPE` with no `_BEST` suffix, and
`FLUX_G/R/Z` in the *same* file). Same BGS_BRIGHT/BGS_FAINT/LOW_Z selection and
redshift/science cuts as the matterhorn selector, plus:

- **Dereddens** g/r/z from `EBV`+`PHOTSYS` (the zcatalog has no `MW_TRANSMISSION_*`)
  via `desiutil.dust.mwdust_transmission(..., match_legacy_surveys=True)`.
- **Keeps only dwarfs**: `LOGM_M24 = get_stellar_mass_mia` (de los Reyes+2024
  Eq.13) from the dereddened g−r, g and redrock z; cut `LOGM_M24 < 9.25`
  (`--logmstar-max`). No k-correction beyond the estimator, no nebular correction.
- Carries the same join keys, so **stage ② (`crossmatch_tractorphot.py`) is reused
  unchanged** to attach `FRACFLUX_*` (from the DR9 sweeps — loa has no public
  lsdr9-photometry VAC) and apply the FRACFLUX cut.

```bash
python select_loa_dwarfs.py --help
```

Job scripts live in `../../job_scripts/catalog_construct/`. From a **login node**:

```bash
./submit_loa_catalog.sh          # run_select_loa.sh --afterok--> run_crossmatch_loa.sh
```

Outputs (default, `/pscratch/sd/v/virajvm/loa/`): `loa_pix_bgs_lowz_dwarfs.fits`
(stage ①), then `loa_pix_bgs_lowz_dwarfs_phot.fits` (all dwarfs + photometry) and
`loa_pix_bgs_lowz_dwarfs_clean.fits` (FRACFLUX-cut) from stage ②.

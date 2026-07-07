# Spectroscopic Incompleteness Model

Pipeline that builds per-galaxy **purity** and **completeness** weights for the
DESI dwarf galaxy catalog by forward-modeling realistic mock spectra through
the DESI observing + redrock pipeline and training neural-network classifiers
on the outcomes.

Old exploratory code (notebooks, FSPS-based sims, slide deck) is archived in
`old_reference/` — kept for provenance, superseded by this pipeline.

## The statistical framework

We want to correct the observed `z_RR < 0.15` catalog for two effects:
galaxies **in** the catalog with wrong redshifts (impurity) and true low-z
galaxies **missing** from their proper place (incompleteness). Each catalog
galaxy gets

```
W_spec = p / c
```

- `p` = **purity weight** = P(redshift correct | measured features, entered
  the catalog). Trained on mocks passing the exact catalog selection.
  Acts as a soft per-object down-weight of interlopers.
- `c` = **completeness** = P(redshift correct | measured features), trained
  on the **full** true-z<0.15 mock population. `1/c` up-weights each correct
  galaxy for its lost siblings.

### Why measured features are legal (and one load-bearing condition)

All features (`Dchi2`, fiber mag, `z_RR`, spectral shape fit at `z_RR`) are
*measured* quantities — available identically on mocks and real data, even
when the redshift is wrong. The estimator `sum(p_i / c_i)` over the catalog
recovers the true population because, in any bin of measured features, the
expected number of *correct* objects is `n * c`, each carrying `1/c`; wrong-z
objects contribute ~0 through `p`. This resolves the "true vs measured EW"
worry from the original slides.

**Load-bearing condition:** `c` must be estimated over the FULL true-z<0.15
mock population — including mocks with `z_RR > 0.15`, `ZWARN != 0`, and wildly
wrong `z_RR` — with features measured at whatever redshift redrock returned.
Training completeness only on "recovered" mocks silently breaks the estimator.
`measure_features.py` + `train_completeness_model.py` enforce this.

### Scope

`W_spec` corrects **redshift-measurement** incompleteness only. Targeting
incompleteness and fiber-assignment incompleteness are separate multiplicative
selection terms and are NOT included here.

## Pipeline stages

| # | Script | Where | What |
|---|--------|-------|------|
| 1 | `build_reference_sample.py` | CPU node | Broad hi-SNR robust-z BGS Bright + LOWZ Dark z<0.15 sample; consolidated spectra HDF5; empirical efftime distributions; bright-pool flag. |
| 2 | `train_nnmf_bases.py` | GPU node | Train `W_gen` (10 templates, generation) and `W_feat` (2–3 templates, NN features); bootstrap coefficient pool + strata; **pool-coverage diagnostic (inspect before generating!)**. |
| 3a | `generate_mock_spectra.py` | (library) | Stratified-bootstrap coefficient draws → `W_gen @ c`; redshift + fiber-mag scaling. Deterministic per chunk. |
| 3b | `mock_observe.py` | CPU node | `ObserveBackend` interface; feasibgs `simExposure` + SCORES hack; writes redrock-ready coadds + input lists. |
| 3c | `calibrate_efftime.py` | CPU + redrock | **Early milestone/gate**: fit the global exptime scale factor; verify NNMF-reconstruction smoothness doesn't bias Dchi2. Failure ⇒ desisim backend. |
| 4 | `run_redrock.sbatch` | GPU nodes | `wrap_rrdesi` on the mock lists, **pinned to the iron-era stack (23.1)**. |
| 5 | `measure_features.py` | CPU node | Features at `z_RR` for ALL mocks (partial-overlap NNLS fit); labels; training table. |
| 6 | `train_completeness_model.py` | login/local | Purity + completeness MLPs (sklearn, warm-start LR schedule); held-out split saved with the models. |
| 7 | `apply_weights.py` | CPU node | Same features on the real catalog spectra → `W_spec` table. |
| 8 | `validation.py` | login/local | Recovery maps, reliability diagrams, closed-loop estimator test, SV3 anchor, distribution matching. |

Job scripts: `job_scripts/specz_incompleteness/`. All paths/cuts/knobs:
`config.py`. Shared machinery: `specz_utils.py`.

### Run order

```
sbatch build_reference_sample.sbatch
sbatch train_nnmf_bases.sbatch            # then INSPECT pool_coverage_diagnostic.pdf
python calibrate_efftime.py prepare       # on a CPU node w/ DESI env
sbatch run_redrock.sbatch <calib list>    # then: python calibrate_efftime.py analyze
#  -> set EXPTIME_CALIBRATION_FACTOR in config.py
sbatch generate_and_observe.sbatch
sbatch run_redrock.sbatch <each mock list>
sbatch measure_features.sbatch            # then: python measure_features.py --consolidate
python train_completeness_model.py
python validation.py
python apply_weights.py                   # on a CPU node w/ DESI env
```

## Key design decisions

- **Generation = bootstrap, not a density model.** Whole coefficient vectors
  are resampled from real galaxies (preserves inter-template correlations,
  guaranteed physical).
- **Two bases.** Rich `W_gen` for realism in generation; coarse `W_feat` for
  NN features — redshift success cares about first-order type (strong lines
  vs continuum), not detailed line ratios.
- **Quenched-dwarf coverage** (the science-critical corner):
  1. bootstrap pool restricted to bright fiber mags (`POOL_RFIB_MAX`), where
     even passive galaxies get robust redshifts → within-cell type mixture is
     nearly selection-free;
  2. strata sampled **uniformly** in emission strength (legal because the NN
     conditions on type: only coverage matters, not proportions).
  Bootstrap diversity is set by the number of *distinct* pool vectors per
  stratum, not by the number of mock draws — hence the mandatory diagnostic.
- **Synthetic fiber mag everywhere.** The rfib feature is the asinh mag of the
  coadded spectrum through `decamDR1noatm-r`, computed identically for mocks
  and real data. Photometric `FIBERMAG_R` never enters the model (avoids the
  fiber-loss/aperture covariate shift for extended dwarfs).
- **`z_RR` is a feature.** (a) Real z-dependent failure modes (Hα enters the
  OH forest by z≳0.12; camera-gap crossings; line-confusion degeneracies);
  (b) conditioning on it removes the need for the mock z-prior to match the
  real n(z) — uniform z sampling then only needs coverage.
- **Efftime is simulated realistically but is NOT a feature** — `Dchi2` is
  the depth/SNR proxy (flagged assumption, below).
- **Flux-conserving resampling** for all rest-frame work (`use_invvar=False`
  equivalent): ivar-weighted resampling suppresses emission-line cores.
- **Catastrophic threshold** `|dz|/(1+z) > 0.0033` (~1000 km/s) everywhere,
  including the SV3 validation (old code mixed 0.003/0.0033).
- **Catalog selection single source of truth**:
  `specz_utils.in_catalog_selection()` mirrors
  `construct_dwarf_galaxy_catalogs.py`. If the catalog cuts change, change it
  there and retrain the purity model.

## Flagged assumptions (to confirm)

1. **Dchi2-only depth proxy.** Re-train with efftime/TSNR2 added as a feature
   and compare AUC + calibration; if it matters, add it.
2. **feasibgs noise realism** (Julien Guy caveat: no noise correlations,
   cosmic rays, bias-level effects). Gate = `calibrate_efftime.py`; fallback =
   desisim backend behind `ObserveBackend`.
3. **2–3 feature templates suffice.** Check with
   `code/nnmf_pca_analysis/scan_nnmf_ntemplates.py` on the rest-frame file +
   recovery-map quality.
4. **Reconstruction smoothness** doesn't bias Dchi2 — tested by the
   real-vs-reconstruction comparison inside `calibrate_efftime.py`.
5. **Redrock version pinning**: confirm the exact iron production tag
   (currently assuming the 23.1 software release).
6. **Phase-2 (z>0.15 contamination) is small** (SV3: 17 wrong-z with
   zTrue<0.15 vs 1 with zTrue>0.15). When built, note `W_gen` (rest
   3100–9850 Å) cannot generate z≳0.15 spectra observed to 3600 Å — needs an
   extended basis or re-observed real high-z spectra behind the same
   `ObserveBackend`/generator interfaces.
7. **Completeness floor** (`apply_weights.COMPLETENESS_FLOOR = 0.02`): objects
   at the floor live in uncorrectable parameter space (both methods fail where
   completeness → 0); they are flagged `AT_COMPLETENESS_FLOOR` — decide per
   science case whether to keep or cut them.

# Answers to plan.md Section 7 + data-flow note

## 0. Data-flow clarification (do not change downstream error logic)

Two independent error channels:

1. **Detection gate (measurement S/N):** `line_snr_mask(t[[0]], ...)` in `stack_direct_metallicity.py` uses FastSpecFit per-line `*_FLUX_IVAR` from `--nmonte=100`, driven by the input spectrum `ivar`. After the ivar fix this is a true measurement-detection criterion.

2. **Reported uncertainty (population spread):** `compute_direct_metallicities(t, ...)` fits every FITS row. `res[0]` → central values; `res[1:]` → bootstrap-row fits. `_fill_param_errors` / `boot_spread` take 16/50/84 percentiles of bootstrap-row posterior medians as `{PARAM}_ERR_LO/_ERR_HI`. Mean-stack posterior interval is stored separately as `{PARAM}_MEANFIT_LO/HI` and is **not** combined with bootstrap error.

**No double-counting:** nmonte/measurement-ivar → gating + informational `_MEANFIT_*`; bootstrap rows → reported `_ERR_*`.

### Expected side effect

Switching row-0 `ivar` from `1/boot_std²` to propagated measurement ivar raises per-line S/N at emission lines. **More bins will pass `DETECTED_7LINE`** than before. This is correct and intended.

---

## 1. FastSpecFit invocation, output parsing, downstream error path

- **Invocation:** [`job_scripts/fastspec/run_stack_fastspec_haew_5pct.sh`](../job_scripts/fastspec/run_stack_fastspec_haew_5pct.sh) — CLI `stackfit` on each `stack_ALL_mstar_*.fits`, `fastspecfit/3.4.2`, `--nmonte=100`. Fits each FITS row independently. **No script change needed.**

- **Parsing:** [`stack_direct_metallicity.py`](stack_direct_metallicity.py) reads HDU 3, gates on row 0, fits all rows, reduces `res[1:]` via `boot_spread`. Was dormant with 1-row FITS; active with 1+50-row stacks.

- **No new ratio-aggregation script** — downstream analysis stays in `stack_direct_metallicity.py`.

---

## 2. Stack FITS schema / multi-row support

- [`write_stacked_spectra`](../stacking_analysis/stack_explore.py) accepts 2D flux/ivar; HDUs: `FLUX`, `IVAR`, `WAVE`, `STACKINFO`.
- **5pct (was):** 1 row, `ivar = 1/boot_std²` (bug).
- **5pct (now):** 1 + 50 rows; row 0 = `central_flux` + `central_ivar`; rows 1–50 = `real_flux[k]` + `real_ivar[k]`; `IS_MEAN` flag on `STACKINFO`.
- **3bin reference:** same multi-row pattern in `stack_mstar_haew_3bin.py` (still uses old ivar until separately ported).

---

## 3. Existing bootstrap machinery

- Core: `bootstrap_stack` in [`stack_explore.py`](../stacking_analysis/stack_explore.py).
- **Correct:** `norm_ivar = ivar * H²` in `normalize_by_line_catalog` (line 296).
- **Fixed:** `use_ivars` now used via `coadd_mean_with_propagated_ivar` inside each bootstrap draw.
- **Fixed:** bootstrap draws `n_valid` galaxies with replacement (was `min(5000, n_valid)`).
- **Fixed:** `N_BOOTSTRAP = N_BOOT_SAVE = 50` in 5pct (was 200 → subsample 50).

---

## 4. `bootstrap_stack` return signature

**New return:** `(central_flux, boot_std, real_flux, real_ivar, central_ivar)`

| Quantity | Role |
|----------|------|
| `central_flux` | Mean over bootstrap realizations (unchanged flux convention) |
| `central_ivar` | Mean propagated measurement ivar (Scholte step v) → **fed to FastSpecFit row 0** |
| `real_flux`, `real_ivar` | `(50, N_wave)` per-realization coadds → **FITS rows 1–50** |
| `boot_std` | Diagnostic only; **never** fed to FastSpecFit |

Callers in `stack_mstar_haew_3bin.py`, `stack_mstar_haew.py`, `stack_mstar_elg_vs_noelg.py` updated to unpack 5-tuple; their FITS writers still use `1/boot_std²` until separately ported.

---

## 5. Emission-line column names

Verified in [`data_model.py`](../data_model.py) and `fastspecfit/3.4.2`:

| Line | Gaussian | Boxcar |
|------|----------|--------|
| [O III] 5007 | `OIII_5007_FLUX` | `OIII_5007_BOXFLUX` |
| Hβ | `HBETA_FLUX` | `HBETA_BOXFLUX` |
| Hα | `HALPHA_FLUX` | `HALPHA_BOXFLUX` |
| [S II] | `SII_6716_FLUX`, `SII_6731_FLUX` | `SII_6716_BOXFLUX`, `SII_6731_BOXFLUX` |

Production downstream: `--line-flux-type BOXFLUX`.

---

## 6. Direct-method per-row cost

`compute_direct_metallicities` in [`pn_functions.py`](pn_functions.py) uses **joblib `Parallel(n_jobs=...)` across rows** when `n_jobs > 1`. With `N_JOBS=128` on Perlmutter, 51 rows per bin fit in one parallel batch (~one UltraNest fit wall-time per bin, not 51× serial). No fallback to fewer UltraNest bootstrap fits is needed given current `n_jobs` setting.

---

## 7. Bootstrap reliability gate

Added `MIN_N_BOOT_FIT = 30`, `BOOT_ERR_RELIABLE`, `N_BOOT_TOTAL` in `stack_direct_metallicity.py`. Auroral-line (`OIII_4363`) survivor bias can leave `boot_ok` with too few successful fits; bins below the threshold are flagged but still reported.

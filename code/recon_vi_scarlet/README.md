# recon_vi_scarlet — SCARLET component VI & grouping fine-tune GUI

A local, offline single-page app for finetuning the **component grouping** of
the `scarlet_photo` fits (stage 3 of the SCARLET pipeline). Sibling of
`recon_vi` (which edits Tractor-source reconstructions); `recon_vi` itself is
never touched.

The SCARLET fitter models the whole image plane as a sum of components and
proposes an *initial* dwarf membership (color cuts + GMM). This tool shows two
**complementary live model panels** — *galaxy* and *not-galaxy* — whose sum is
always the full SCARLET model, and lets you **move** components between them:

```
galaxy      = science_cube   + Σ(added patches) − Σ(removed patches)
not-galaxy  = nondwarf_cube  − Σ(added patches) + Σ(removed patches)
```

Toggling is a pure flux-space add/subtract in the browser (no scarlet at
inspection time). The app records a per-object verdict + comment + a **bad-fit
flag** and autosaves a **replayable decision CSV** whose deltas are relative to
the fitter's deterministic `initial_membership` (component IDs are
deterministic from their seeds, so decisions survive a re-fit). Post-VI
magnitudes (pure sums of the model-frame component fluxes over the final
member set) are computed server-side at save time and baked into the CSV.

---

## Workflow at a glance

```
NERSC (or local test)                    Local (your laptop)
─────────────────────                    ───────────────────
scarlet_photo.driver      (stage 1: fits + fragments)
scarlet_photo.consolidate (stage 2: per-brick bundle store)
recon_vi_scarlet.build_bundle ──► bundle.h5 ──scp──► recon_vi_scarlet.server
(packs a hand-picked VI catalog                        │
 out of the bundle store)                              ▼
                                          http://127.0.0.1:8001/
                                          decisions.csv (autosaved)
                                                       │
                        recon_vi_scarlet.apply_decisions
                        (stage 4a: replay decisions → final galaxy cubes)
                                                       │
                                                       ▼
                                          scarlet_final_cubes.h5
                                                       │
                        recon_vi_scarlet.make_crops
                        (stage 4b: light-centered crop → ML tensor)
                                                       │
                                                       ▼
                                          scarlet_crops128.h5
                                                       │
    vi_cat.fits ──►         recon_vi_scarlet.add_psfsize
    (TARGETID +             (stage 4c: attach native psfsize_g/r/z + brickname)
     PSFSIZE_* +                                       │
     BRICKNAME)                                        ▼
                       scarlet_crops128.h5  (+ /psfsize_g/r/z + /brickname)
                                                       │
                              (→ galaxy_prior_proj: PSF homogenization + repack)
```

## Quick reference — copy-paste commands

Set the paths once, then run the steps in order. These are the actual local
paths for the current run; edit them if the files move. (Per-step detail is in
the numbered sections below.)

```bash
# --- set once (local paths for this run) ---
CODE=/Users/virajmanwadkar/Desktop/Stanford/Research/DESI/desi_code/desi_dwarfs/code
DATA=/Users/virajmanwadkar/Desktop/Stanford/Research/galaxy_prior_proj/data
CAT=/Users/virajmanwadkar/Downloads/datasets/vi_cat.fits
cd "$CODE"

# --- stage 3: VI GUI (open http://127.0.0.1:8001/) ---
python -m recon_vi_scarlet.server \
    --bundle "$DATA/bundle_partial.h5" \
    --out    "$DATA/scarlet_decisions.csv" \
    --inspector viraj

# --- stage 4a: decisions -> final galaxy cubes (accept-only) ---
python -m recon_vi_scarlet.apply_decisions \
    --bundle    "$DATA/bundle_partial.h5" \
    --decisions "$DATA/scarlet_decisions.csv" \
    --out       "$DATA/scarlet_final_cubes.h5"

# --- stage 4b: light-centered 128x128 crops (one per object, no augmentation) ---
python -m recon_vi_scarlet.make_crops \
    --in  "$DATA/scarlet_final_cubes.h5" \
    --out "$DATA/scarlet_crops128.h5"

# --- stage 4c: attach native psfsize_g/r/z from the catalog (in place) ---
python -m recon_vi_scarlet.add_psfsize \
    --crops   "$DATA/scarlet_crops128.h5" \
    --catalog "$CAT"
```

> **Stage 1–2 (bundle build)** run on NERSC and feed `bundle_partial.h5`; see
> §1 below for `build_bundle`. The chain above is the repeatable local part.


And then once the scarlet_crop128.h5 file is shopped to NERSC, we can homogenize the psf (if we are traning that model variant)

```bash
cd galaxy_cutouts
python homogenize_scarlet_crops.py \
    --input  /pscratch/sd/v/virajvm/galaxy_prior_data/scarlet_crops128.h5 \
    --output /pscratch/sd/v/virajvm/galaxy_prior_data/scarlet_crops128.homog.h5 \
    --target-fwhm-file /pscratch/sd/v/virajvm/galaxy_cutouts/run1/target_fwhm.json
```

## 1. Build the bundle

`build_bundle.py` copies the per-object groups **verbatim** out of the stage-2
bundle store (`scarlet_bundles/{brick[:3]}/{brick}.h5`) into a single HDF5
keyed by TARGETID, plus a top-level `/index` for the sidebar. Requires only
numpy/h5py/astropy. Catalog needs `TARGETID` + `BRICKNAME`; row order = VI
order. Missing objects are skipped (logged to `<out>_skipped.csv`), never fatal.

```bash
cd .../desi_dwarfs/code
python -m recon_vi_scarlet.build_bundle \
    -vi_catalog PICKED.fits \
    -bundle_dir /global/cfs/cdirs/desi/users/virajvm/dwarf_cutouts/scarlet_bundles \
    -out bundle.h5            # [-float16] [-limit N]
```

`-float16` halves the four display cubes; component patches always stay
float32 (they are summed in the compositor).

## 2. Run the GUI locally

```bash
cd .../desi_dwarfs/code
python -m recon_vi_scarlet.server \
    --bundle /Users/virajmanwadkar/Desktop/Stanford/Research/galaxy_prior_proj/data/bundle_partial.h5 \
    --out    /Users/virajmanwadkar/Desktop/Stanford/Research/galaxy_prior_proj/data/scarlet_decisions.csv \
    --inspector viraj
# open http://127.0.0.1:8001/   (port 8001 so recon_vi on 8000 can coexist)
# --bundle must be the bundle .h5 file (from build_bundle.py), NOT a directory.
```

On launch it reads any existing `decisions.csv`, marks decided objects, and
opens at the first object without a verdict. Reopening a decided object
restores its toggles, bad-fit flag, comment, and verdict, fully editable.

## Using the app

Four panels, locked to the same FOV / zoom / pan:

| Panel | Shows | Markers |
|---|---|---|
| **Input cutout** | the raw grz data | ALL components |
| **Galaxy model** | live: the current member set, rendered observed-frame | current members only |
| **Not-galaxy model** | live: everything else; the two panels always sum to the full model | non-members only |
| **Residual** | static: data − full model (fit quality; LSB pathologies show up here) | none |

**Component overlay:** small circles at component centers, **diamonds** for
Gaia stars, and a **cyan crosshair** at the DESI target position (the target
is *not* a component — the fitter has no guaranteed target seed; the crosshair
is orientation only). Each model panel shows **only its own population**, so a
crowded field splits into two sparse clickable sets. The full-frame **LSB
component has no marker**: use the **`LSB:`** chip (hotkey **`l`**) to move it
between panels. Visual grammar (same as recon_vi):

- **Line style:** solid = in galaxy · **dashed = in not-galaxy**.
- **Color:** green = in galaxy · red = in not-galaxy · **amber = you changed
  it this session** · yellow = pending selection.
- **＋/－ badge** on every component you've moved (＋ to galaxy, － to
  not-galaxy).
- **Hover** traces the component's **actual flux outline** (isophote at 8% of
  the patch peak, computed client-side and cached) and shows
  `comp_id · type · mag g/r/z · g−r · r−z · P_gmm · state`.

**Interaction (works on the input AND both model panels):** click components
to build a selection (repeated clicks cycle stacked components), then
**→ Galaxy** / **→ Not-galaxy** move them all; **double-click** a component to
move it instantly (undo-able); **Undo** / **Reset to baseline** as in recon_vi.
Scroll = zoom, drag = pan (all panels synced). Hold **`O`** to hide all
overlays for a clean look.

**Bad fit** (**`b`**): flags *the SCARLET model itself* as wrong (e.g. a
degenerate LSB spectrum) — a refit candidate, independent of the verdict. You
can still salvage the membership and accept, or reject; the flag keeps the
object queryable either way.

**Verdict (auto-advances):** **Accept** / **Unsure** / **Reject** save the row
and move on — the only thing that marks an object *inspected*. Prev/Next, the
jump box, the sidebar, and the **accepted-only filter** (**`f`**) navigate
without committing a verdict. Edits and the comment autosave.

## Output: `decisions.csv` (replayable)

One row per object; deltas are relative to the fitter's `initial_membership`
baseline, so `final members = (initial − removed) ∪ added` reproduces the
fine-tuned grouping exactly later (stage 4 consumes this).

| column | meaning |
|---|---|
| `TARGETID`, `BRICKNAME` | identity |
| `removed_comp_ids` | `;`-joined comp_ids moved galaxy → not-galaxy |
| `added_comp_ids` | `;`-joined comp_ids moved not-galaxy → galaxy |
| `n_components_changed` | int |
| `lsb_in_galaxy` | whether the starlet LSB component ends up in the galaxy |
| `bad_fit` | the SCARLET model itself is wrong — refit candidate |
| `verdict` | `accept` / `unsure` / `remove` (blank until decided) |
| `inspected` | true once a verdict is set |
| `mag_g/r/z` | post-VI mags: 22.5 − 2.5 log₁₀ Σ(member model-frame fluxes) |
| `comment` | free text |
| `timestamp`, `inspector` | ISO-8601 UTC; from `--inspector` |

The CSV is rewritten atomically (temp + `os.replace`) on every save.

## 3. Apply decisions → final galaxy cubes (stage 4a)

`apply_decisions.py` replays `decisions.csv` against the `bundle.h5` and writes the
**primary product**: the final finetuned galaxy MODEL cube per object — grz,
PSF-convolved (observed frame, *not* deconvolved), background-free. Pure
numpy/h5py (+ astropy for the WCS); no scarlet, no NERSC.

```bash
cd .../desi_dwarfs/code
python -m recon_vi_scarlet.apply_decisions \
    --bundle bundle.h5 \
    --decisions decisions.csv \
    --out scarlet_final_cubes.h5 \
    [--include-unsure] [--include-undecided] [--float16]
```

Each cube is the flux-space **sum of the VI-selected component patches**, with the
final membership reconstructed exactly as the GUI computed it:

```
member = (initial_membership and cid not in removed) or (cid in added)
```

Component patches are stored float32 in the bundle (only the *display* cubes are
float16), so the sum is a clean float32 cube independent of the float16 baseline.

**By default only `verdict=='accept'` objects are written.** `--include-unsure`
adds the `unsure` rows; `--include-undecided` adds objects with no verdict
(applying their edits if any, else the automatic initial membership).

Output layout (one group per object, keyed by TARGETID):

| item | meaning |
|---|---|
| `/{TARGETID}/galaxy_cube` | `(3, S, S)` float32 — the final model cube (grz), at the **full box_size** cutout frame |
| group attrs | `TARGETID`, `BRICKNAME`, `box_size`, `gal_ra/dec`, `gal_xpix/ypix`, `pixscale`, `wcs_header` (TAN, reconstructed from the anchor), `verdict`, `bad_fit`, `lsb_in_galaxy`, `comment`, `n_members_final`, `member_comp_ids`, `removed_comp_ids`/`added_comp_ids`, `mag_g/r/z` (post-VI, not MW-corrected), `inspector`, `decision_timestamp`, `created` |
| `/index` | compound table: `targetid, brickname, box_size, n_members_final, mag_g/r/z, verdict` |
| top-level attrs | `n_objects`, `source_bundle`, `source_decisions`, `n_by_verdict`, `created` |

The reconstructed TAN WCS runs through the exact `(gal_ra,gal_dec) ↔ 0-based
(gal_xpix,gal_ypix)` anchor at the Legacy Surveys 0.262″/px, N-up/E-left scale —
accurate to well under a pixel across the field. Warnings are printed for empty
member sets and for requested decisions that had no bundle group.

## 4. Make light-centered crops (stage 4b)

`make_crops.py` turns the final cubes into a stacked ML tensor: **one crop per
object** (no augmentation), recentered on the light-weighted centroid. It only
READS the input. Pure numpy/h5py (+ astropy for pixel→sky).

```bash
cd .../desi_dwarfs/code
python -m recon_vi_scarlet.make_crops \
    --in  scarlet_final_cubes.h5 \
    --out scarlet_crops128.h5 \
    [--crop-size 128] [--window 0]
```

Per object: compute the flux-weighted centroid of the grz-summed cube (negatives
clipped to 0; `--window R` for an iterative windowed centroid seeded at the
target), crop a fixed `crop-size` box (default 128) with the origin snapped to the
nearest int and **clamped** to stay on-frame (no zero-padding, `clamped` flag
recorded).

The light center is recorded **both** as sky coordinates and as pixel coordinates
**in the original box_size cutout frame**, so you can re-download a matching cutout
from the real data at exactly the light-weighted center.

Output — one row per object:

| item | meaning |
|---|---|
| `/images` | `(N, 3, size, size)` float32 — `N = n_objects`, one crop each |
| `/targetid` | `(N,)` int64 — source TARGETID (join key) |
| `/index` | compound: `targetid`, `light_center_ra`, `light_center_dec` (deg), `light_center_xpix`, `light_center_ypix` (0-based px, original frame), `crop_origin_x/y` (box top-left, original frame), `clamped`, `mag_g/r/z` |
| top-level attrs | `n_crops`, `n_objects`, `crop_size`, `recenter`, `source_file`, `created` |

`light_center_ra/dec` come from pushing `(light_center_xpix, light_center_ypix)`
through each object's stored `wcs_header`; they are `NaN` if the object had no
usable WCS.

## 5. Attach native PSF sizes + brickname (stage 4c)

`add_psfsize.py` joins a FITS catalog (with a `TARGETID` column) onto the crops h5
by TARGETID and writes per-object arrays: the **native per-band PSF FWHM**
`psfsize_g/r/z` (arcsec, three float32 datasets) and the **Legacy `brickname`**
(string). The psfsizes are what the galaxy_prior_proj homogenization step reads to
build each object's Gaussian matching kernel; `brickname` together with the
light-center RA/Dec in `/index` lets you pull the matching real-data cutout. Both
satisfy the `psfsize_g/r/z` + `brickname` parts of `repack_scarlet_clean.py`'s
input contract. Needs numpy/h5py/astropy.

```bash
cd .../desi_dwarfs/code
python -m recon_vi_scarlet.add_psfsize \
    --crops scarlet_crops128.h5 \
    --catalog /path/vi_cat.fits \
    [--hdu 1] [--targetid-col TARGETID] \
    [--psfsize-cols PSFSIZE_G PSFSIZE_R PSFSIZE_Z] \
    [--brickname-col BRICKNAME] [--strict]
```

- **Per-object, not attrs.** Each galaxy has its own seeing / brick, so
  `psfsize_g/r/z` + `brickname` are length-`N` datasets, **row-aligned to the
  top-level `/targetid`** — row `i` is the value for the galaxy at `targetid[i]`.
- **In place + idempotent.** The datasets are written into the crops h5 directly
  (existing ones overwritten), so re-running is safe. Also sets attrs
  `psf_homogenized=False` (this is the native/PSF-variation variant),
  `psfsize_source_catalog`, `psfsize_units="arcsec_fwhm"`, `psfsize_added`.
- **Join policy (default, graceful):** a crop targetid absent from the catalog →
  `psfsize` `NaN` (homogenization treats a non-finite PSF as un-homogenizable and
  skips it) and `brickname` `""`; duplicate catalog TARGETID → first occurrence;
  counts reported. `--strict` aborts on either instead.
- **Columns** default to the Legacy standard `TARGETID` + `PSFSIZE_G/R/Z` +
  `BRICKNAME` (case-insensitive); override with `--targetid-col` /
  `--psfsize-cols` / `--brickname-col`. PSF values are taken as FWHM in arcsec
  (no conversion); FITS byte-strings are decoded to UTF-8.

## Notes & limitations

- **Toggleable universe = SCARLET components only.** You can't split a
  component or edit pixels; if the decomposition itself is wrong, flag
  **bad fit** and refit.
- Post-VI mags are **not MW-corrected** and are model-frame flux sums (they
  differ from the stage-1 `SCARLET_MAG_*_TOTAL` only by the membership edits
  and the MW transmission factor).
- No infill: a scarlet model has no masked holes (recon_vi's infill machinery
  was dropped, not hidden).

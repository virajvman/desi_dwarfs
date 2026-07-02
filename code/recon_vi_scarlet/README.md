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
python -m recon_vi_scarlet.server --bundle bundle.h5 --out decisions.csv
# open http://127.0.0.1:8001/   (port 8001 so recon_vi on 8000 can coexist)
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

## Notes & limitations

- **Toggleable universe = SCARLET components only.** You can't split a
  component or edit pixels; if the decomposition itself is wrong, flag
  **bad fit** and refit.
- Post-VI mags are **not MW-corrected** and are model-frame flux sums (they
  differ from the stage-1 `SCARLET_MAG_*_TOTAL` only by the membership edits
  and the MW transmission factor).
- No infill: a scarlet model has no masked holes (recon_vi's infill machinery
  was dropped, not hidden).

# recon_vi — reconstructed-cube visual inspection & fine-tuning GUI

A local, offline single-page app for visually inspecting the reconstructed grz
cubes of DESI dwarf galaxies and **fine-tuning** each reconstruction by toggling
individual Tractor sources on/off.

The reconstruction you edit is the **real masked science cube**
(`final_reconstruct_galaxy_subfunction.npy` — background+blend-subtracted data
with the cog mask applied, exactly what the pipeline produces). Fine-tuning is a
pure-numpy edit of that real image:

```
fine_tuned = science_cube − Σ(removed source models) + Σ(added source models)
```

so **no Tractor / legacypipe runs at inspection time** — toggling a source is an
instant flux-space add/subtract in the browser. The app records a per-object
verdict + comment and autosaves a **replayable decision CSV**.

> z < 0.005 objects have no per-source models (`recon_variant == "no_isolate"`).
> They are shipped **view-only**: input cutout + the stored no-isolate cube,
> with source toggling disabled (you can still flag them with a verdict/comment).

---

## Workflow at a glance

```
NERSC                                    Local (your laptop)
─────                                    ───────────────────
build_bundle.py  ──►  bundle.h5  ──scp──►  server.py  ──►  http://127.0.0.1:8000/
(packs catalog +                            (Flask; serves      ▲
 per-object artifacts)                       one object/req)    │ all compositing
                                            decisions.csv ◄─────┘ client-side
```

---

## 1. Build the bundle on NERSC

`build_bundle.py` reads a **hand-picked VI catalog** (a row-subset you carve out
of the `*_w_aper_mags*.fits` catalogs) and packs everything the GUI needs into a
single HDF5 file.

**Required catalog columns:** `TARGETID`, `BRICKNAME`, `FILE_PATH`,
`APER_RADEC_CEN_ISOLATE`, `APER_RADEC_CEN_NO_ISOLATE`.
(Grid size `S` and `recon_variant` are read from the files on disk, so no
`IMAGE_SIZE_PIX` or `Z` column is needed.)

```bash
cd /global/u1/v/virajvm/DESI2_LOWZ/desi_dwarfs/code
source /global/cfs/cdirs/desi/software/desi_environment.sh main   # numpy/h5py/astropy

python recon_vi/build_bundle.py \
    -vi_catalog /path/to/PICKED.fits \
    -out /pscratch/sd/v/virajvm/recon_vi/bundle.h5 \
    -ncores 16
```

Useful flags: `-float16` (store the two display cubes as float16 to ~halve the
bundle; model patches stay float32), `-rel_thresh` (patch-crop threshold,
default `1e-4`), `-limit N` (debug: first N rows only).

**Caveat — run while pscratch is warm.** Each row's `FILE_PATH` folder must still
contain `tractor_models/`, `parent_galaxy_sources.fits`,
`parent_galaxy_sources_isolate_FINAL.fits`, `final_reconstruct_galaxy_subfunction.npy`
(and for z<0.005 the no-isolate cube). These live on **purgeable pscratch** — run
the export before they are purged.

**Missing files → that object is skipped** (the run never aborts) and logged to a
sidecar `bundle_skipped.csv` next to the output, with a final count printed.

A few-thousand-object bundle is ~1–4 GB → one `scp`:

```bash
scp perlmutter:/pscratch/sd/v/virajvm/recon_vi/bundle.h5 ./
```

## 2. Run the GUI locally

One dependency beyond the scientific stack:

```bash
pip install flask          # numpy + h5py are already present

cd /Users/.../desi_dwarfs/code
python -m recon_vi.server --bundle bundle.h5 --out decisions.csv \
    [--inspector yourname] [--port 8000]
```

(Equivalently `python recon_vi/server.py --bundle ...` — the module is
self-contained.) Then open **http://127.0.0.1:8000/**.

On launch it reads any existing `decisions.csv`, marks decided objects, and opens
at the **first object without a verdict**. Reopening a decided object restores its
toggles, comment, and verdict, fully editable.

---

## Using the app

Three panels, locked to the same FOV / zoom / pan:

| Panel | Shows |
|---|---|
| **Input cutout** | the raw sky data — **hosts the clickable source overlay** |
| **Current reconstruction** | the science cube as the pipeline made it (the "before") |
| **Fine-tuned** | live: `science_cube − removed + added`, updates on every edit |

**Source overlay (input panel):** Tractor ellipses (`shape_r/e1/e2`); small
circles for PSF / star-like sources. Three orthogonal visual channels (so
membership never collides with anything, and edits pop):

- **Line style = in/out of cube:** solid = in the fiducial cube; **dashed =
  subtracted/excluded** (the sources the pipeline removed — find these to add back).
- **Color = state:** green = kept · **red = subtracted** · **amber = *you*
  changed it this session** · cyan = target · yellow = pending selection.
- **＋/－ badge** on every source you've edited (`＋` added back, `－` removed),
  so undo candidates are obvious at a glance.
- The **target** (`separation < 1″`) is cyan + thick and needs **Shift-click** to
  select (guarded against accidental toggling).

Source *type* (PSF/REX/EXP/DEV/SER) is intentionally **not** color-coded (that
would collide with the membership colors); it shows on the hover tooltip
alongside `objid · mag_r · separation · in cube/subtracted · edited`.

**Interaction (select-then-act, multi-select):**

1. **Click** an ellipse to add/remove it from the pending selection (click again
   to deselect; stacked sources cycle on repeated clicks). Hover shows
   `objid · type · mag_r · separation · ON/OFF`.
2. **Remove from cube** / **Add to cube** apply to all selected sources; the
   live panel recomputes and the selection clears.
3. **Undo** reverts the last Add/Remove; **Reset to baseline** snaps back to the
   isolate-FINAL set.
4. Scroll = zoom, drag = pan (all panels synced). Hold **`O`** to flash the
   overlay onto the reconstruction panels too.

**Verdict (mouse-only, auto-advances):** **Accept** / **Unsure** / **Reject**.
Pressing one saves the row and moves to the next object — this is the only thing
that marks an object *inspected*. (Reject is written as `remove` in the CSV.)

Plain **Prev/Next**, the **jump** box (index *or* TARGETID), and the **sidebar**
(✓ accept · ? unsure · ✗ reject · ✎ edited-not-decided) navigate **without**
committing a verdict, so you can browse/revisit. Edits and the comment box
autosave on navigation-away and on comment blur.

---

## Output: `decisions.csv` (replayable)

One row per object; deltas are relative to the deterministic **isolate-FINAL
baseline** captured in the bundle, so `science_cube − removed + added` reproduces
the fine-tuned cube exactly later.

| column | meaning |
|---|---|
| `TARGETID`, `BRICKNAME` | identity |
| `removed_objids` | `;`-joined `source_objid_new` that were ON in the baseline and turned OFF (model subtracted) |
| `added_objids` | `;`-joined objids that were OFF in the baseline and turned ON (model added) |
| `n_sources_changed` | int |
| `verdict` | `accept` / `unsure` / `remove` (blank until decided) |
| `inspected` | true once a verdict is set |
| `comment` | free text |
| `toggle_disabled` | true for z<0.005 view-only objects |
| `timestamp` | ISO-8601 UTC |
| `inspector` | from `--inspector` |

The CSV is rewritten atomically (temp + `os.replace`) on every save.

---

## Notes & limitations

- **Toggleable universe = parent-segment sources only.** Only sources in
  `parent_galaxy_sources.fits` have saved per-source models, so only they are
  shown/toggleable. You cannot model-subtract an arbitrary foreground star (no
  individual model) or add a source Tractor never detected. (No pixel-mask
  editing — out of scope by design.)
- **Adding an excluded source** drops its *Tractor model* into a region that is
  currently masked (zeroed) in the science cube, so a re-added source appears
  "modely" while everything around it is real data. Removing a source is the
  clean case (real data minus its model). This is the expected consequence of
  editing in model space without unmasking.
- **Fonts:** JetBrains Mono (OFL) is vendored in `static/fonts/` (no CDN). Change
  the whole look via the single `--font-family` / `--font-size` CSS variables at
  the top of `static/style.css`; drop in a user-licensed woff2 and point the
  variable at it.

## Files

```
recon_vi/
├── build_bundle.py   # NERSC export → one HDF5 bundle (+ bundle_skipped.csv)
├── server.py         # local Flask backend (read-only bundle, atomic CSV writes)
├── static/
│   ├── index.html
│   ├── style.css     # dark theme; --font-family / --font-size knobs
│   ├── rgb.js        # faithful sdss_rgb() port + float16 decode
│   ├── app.js        # navigation, overlay, flux-space compositing, autosave
│   └── fonts/        # JetBrains Mono (OFL) woff2
└── README.md
```

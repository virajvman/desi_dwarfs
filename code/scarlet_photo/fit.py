"""
fit.py -- the 3-stage whole-image-plane SCARLET fit.

Choreography (LOCKED 2026-06-30; see DESIGN.md). It is a TRUE warm start: the
same component objects are carried forward across stages (no copying), and
``.fixed`` flags are set explicitly at the top of every stage via `_set_freeze`
(name-based, so it is robust to whether a source has a 'center' parameter).

  Stage 1  extended sources only (wavelet seeds), stars masked in the weights.
           init_all_sources(set_spectra=True) does the ONE full LLSQ spectrum
           solve; then fit.
  Stage 2  (if cfg.fit_lsb) append the global LSB StarletSource; spectra are
           PRESERVED (not re-solved): fix every existing spectrum, run
           set_spectra_to_match so it initialises ONLY the LSB spectrum against
           the residual, then restore the fit-time freeze. Either way: freeze +
           weight-mask the off-main-blob sources (segment 1); keep main-blob
           sources (segment 2) free so they co-adapt with the LSB (if any) --
           this refinement under the off-blob mask still runs with fit_lsb=False.
  Stage 3  unmask everything (full invvar). Add forced Gaia PointSources with
           positions frozen (or shift-free if cfg.star_shift_free). Extended
           spectra stay FROZEN at their stage-1/2 solved values (2026-07-02
           simplification -- this stage now only redistributes morphology
           now that star/off-blob pixels are unmasked, not re-solve total
           flux); morphology thawed, positions frozen. LSB spectrum frozen
           if present (coeffs stay free). Init only the new star spectra
           (others held), then fit.

`run_fit` returns a FitResult: the fitted components (stable order
[extended..., stars..., lsb]), per-component metadata, the final full-weight
observation and model frame (for rendering/measurement), and per-stage logs.
"""

from dataclasses import dataclass, field

import numpy as np

from .config import ScarletConfig, BANDS
from .detect import apply_circular_mask


class FitError(Exception):
    """Raised when the fit cannot proceed (e.g. no components to fit)."""


@dataclass
class FitResult:
    model_frame: object
    observation: object              # final full-weight observation (for render/measure)
    components: list                 # [extended..., stars..., lsb]
    comp_meta: list                  # aligned dicts: type/is_star/center_yx/ref_id/on_main_blob
    lsb_index: int                    # None if cfg.fit_lsb is False (no LSB fit)
    stage_logs: list                 # [(stage, n_iter, logL), ...]
    band_covered: np.ndarray
    n_extended: int = 0
    n_stars: int = 0


def _set_freeze(comp, spectrum=None, morphology=None, center=None):
    """Set ``.fixed`` on a component's parameters by NAME (robust to which
    parameters exist). A missing 'center' parameter simply means the position is
    already fixed, so freezing/freeing it is a harmless no-op."""
    for p in comp.parameters:
        nm = getattr(p, "name", "")
        if spectrum is not None and nm == "spectrum":
            p.fixed = bool(spectrum)
        elif morphology is not None and nm in ("image", "coeffs", "morphology"):
            p.fixed = bool(morphology)
        elif center is not None and nm in ("center", "shift"):
            # PointSource position is 'center'; ExtendedSource position is 'shift'
            # (ImageMorphology). Match both so "freeze position" is always honored.
            p.fixed = bool(center)


def _build_observation(scarlet, data, obs_psf, weights, bands, model_frame):
    return scarlet.Observation(
        data, psf=obs_psf, weights=weights, channels=bands
    ).match(model_frame)


def run_fit(inp, seeds, cfg=None):
    """Run the 3-stage fit for one object. `inp` is an ObjectInputs, `seeds` a
    Seeds. Returns a FitResult. Raises FitError if there is nothing to fit."""
    import scarlet
    from scarlet.initialization import init_all_sources, set_spectra_to_match

    if cfg is None:
        cfg = ScarletConfig()

    S = inp.box_size
    bands = list(BANDS)
    data = np.ascontiguousarray(inp.data, dtype=np.float32)
    seg = inp.segment_map_v2

    # ---- model frame + empirical observation PSF --------------------------
    model_psf = scarlet.GaussianPSF(sigma=(cfg.model_psf_sigma,) * 3)
    model_frame = scarlet.Frame(data.shape, psf=model_psf, channels=bands)

    psf_cube = np.array(inp.psf, dtype=np.float32, copy=True)
    covered = np.asarray(inp.psf_bands_covered, dtype=bool)
    if not covered.all():
        ref = int(np.where(covered)[0][0])
        for b in range(3):
            if not covered[b]:
                psf_cube[b] = inp.psf[ref]
    obs_psf = scarlet.ImagePSF(psf_cube)

    # base weights: invvar, sanitised; uncovered PSF bands -> 0
    base_w = np.nan_to_num(inp.invvar, nan=0.0).astype(np.float32)
    base_w[base_w < 0] = 0.0
    for b in range(3):
        if not covered[b]:
            base_w[b] = 0.0

    def star_masked(weights):
        w = weights.copy()
        for (cy, cx), rad in zip(seeds.star_centers, seeds.star_radii):
            apply_circular_mask(w, cy, cx, rad)
        return w

    stage_logs = []

    # ===== STAGE 1: extended only, stars masked =====
    w1 = star_masked(base_w)
    obs1 = _build_observation(scarlet, data, obs_psf, w1, bands, model_frame)

    centers = list(seeds.extended_centers)
    sources = []
    kept_centers = []
    on_main = []
    if centers:
        srcs, skipped = init_all_sources(
            model_frame, centers, obs1,
            max_components=cfg.init_max_components, min_snr=cfg.init_min_snr,
            thresh=cfg.init_thresh, fallback=True, silent=True, set_spectra=True)
        skip = set(int(i) for i in (skipped or []))
        kept_centers = [c for i, c in enumerate(centers) if i not in skip]
        sources = list(srcs)
        # init_all_sources returns sources for non-skipped centers, in order
        kept_centers = kept_centers[:len(sources)]
        for c in kept_centers:
            yy = min(max(int(round(c[0])), 0), S - 1)
            xx = min(max(int(round(c[1])), 0), S - 1)
            on_main.append(bool(seg[yy, xx] == 2))

    if sources:
        blend1 = scarlet.Blend(sources, obs1)
        it1, logL1 = blend1.fit(cfg.max_iter_stage1, e_rel=cfg.e_rel, min_iter=cfg.min_iter)
        stage_logs.append((1, int(it1), float(logL1)))
    else:
        stage_logs.append((1, 0, float("nan")))

    # ===== STAGE 2: (optionally) add global LSB; freeze + mask off-main-blob =====
    w2 = base_w.copy()
    w2[:, seg == 1] = 0.0
    w2 = star_masked(w2)
    obs2 = _build_observation(scarlet, data, obs_psf, w2, bands, model_frame)

    lsb = None
    if cfg.fit_lsb:
        lsb = scarlet.source.StarletSource(
            model_frame, starlet_thresh=cfg.starlet_thresh,
            monotonic=cfg.lsb_monotonic)

        # targeted spectrum init: hold every existing spectrum so the LLSQ solve
        # initialises ONLY the LSB spectrum (against the residual)
        for src in sources:
            _set_freeze(src, spectrum=True)
        comps2 = sources + [lsb]
        set_spectra_to_match(comps2, obs2)
    else:
        comps2 = list(sources)

    # fit-time freeze: off-main-blob fully frozen; main-blob free; LSB (if any) free
    for src, om in zip(sources, on_main):
        frozen = not om
        _set_freeze(src, spectrum=frozen, morphology=frozen, center=frozen)

    if comps2:
        blend2 = scarlet.Blend(comps2, obs2)
        it2, logL2 = blend2.fit(cfg.max_iter_stage2, e_rel=cfg.e_rel, min_iter=cfg.min_iter)
        stage_logs.append((2, int(it2), float(logL2)))
    else:
        stage_logs.append((2, 0, float("nan")))

    # ===== STAGE 3: unmask all; add forced PointSource stars =====
    obs3 = _build_observation(scarlet, data, obs_psf, base_w, bands, model_frame)

    star_sources = []
    for (cy, cx) in seeds.star_centers:
        star_sources.append(scarlet.PointSource(model_frame, (cy, cx), obs3))

    # init only the new star spectra: hold every existing spectrum + LSB spectrum
    for src in sources:
        _set_freeze(src, spectrum=True)
    if lsb is not None:
        _set_freeze(lsb, spectrum=True)
    comps3 = sources + star_sources + ([lsb] if lsb is not None else [])
    set_spectra_to_match(comps3, obs3)

    # fit-time freeze: extended spectra FROZEN at their stage-1/2 solved
    # values (2026-07-02 simplification, see module docstring); morphology
    # thawed; positions frozen. LSB spectrum frozen (coeffs free) if present.
    # star flux free + position frozen (or shift-free).
    for src in sources:
        _set_freeze(src, spectrum=True, morphology=False, center=True)
    if lsb is not None:
        _set_freeze(lsb, spectrum=True)
    for ps in star_sources:
        # center=True pins the star at the Gaia catalog position; with
        # star_shift_free the sub-pixel shift stays free (initialised at Gaia)
        # to absorb catalog/coadd astrometry offsets that otherwise leave
        # dipole/ring residuals.
        _set_freeze(ps, spectrum=False, center=not cfg.star_shift_free)

    blend3 = scarlet.Blend(comps3, obs3)
    it3, logL3 = blend3.fit(cfg.max_iter_stage3, e_rel=cfg.e_rel, min_iter=cfg.min_iter)
    stage_logs.append((3, int(it3), float(logL3)))

    if not comps3:
        raise FitError("no components to fit for TARGETID {}".format(inp.targetid))

    # ---- assemble component metadata (aligned with comps3 order) ----------
    comp_meta = []
    for src, om, c in zip(sources, on_main, kept_centers):
        comp_meta.append({
            "type": "extended", "is_star": False,
            "center_yx": (float(c[0]), float(c[1])),
            "ref_id": None, "on_main_blob": bool(om),
        })
    for (cy, cx), rid in zip(seeds.star_centers, seeds.star_ref_ids):
        comp_meta.append({
            "type": "point", "is_star": True,
            "center_yx": (float(cy), float(cx)),
            "ref_id": int(rid), "on_main_blob": False,
        })
    lsb_index = None
    if lsb is not None:
        comp_meta.append({
            "type": "starlet_lsb", "is_star": False,
            "center_yx": None, "ref_id": None, "on_main_blob": False,
        })
        lsb_index = len(comps3) - 1

    return FitResult(
        model_frame=model_frame, observation=obs3,
        components=comps3, comp_meta=comp_meta, lsb_index=lsb_index,
        stage_logs=stage_logs, band_covered=covered,
        n_extended=len(sources), n_stars=len(star_sources),
    )

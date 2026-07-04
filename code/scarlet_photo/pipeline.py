"""
pipeline.py -- per-object orchestration for stage 1 (FIT).

`run_scarlet_object(row, cfg)` ties the whole fit together:
    inputs -> detect/seed -> 3-stage fit -> render + measure -> grouping ->
    cubes -> photometry -> write VI bundle fragment -> (optional) diagnostic plot

and returns a typed `ScarletResult` (fixing the prototype's inconsistent
None-vs-NaN-tuple returns). All rendering happens ONCE here (render_full) and is
reused by photometry, the cubes, and the fragment.
"""

import os
import sys
from dataclasses import dataclass, field

import numpy as np

_CODE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _CODE_DIR not in sys.path:
    sys.path.insert(0, _CODE_DIR)

from .config import ScarletConfig
from . import inputs as _inputs
from . import detect as _detect
from . import fit as _fit
from . import grouping as _grouping
from . import photometry as _photometry
from . import fragment as _fragment


@dataclass
class ScarletResult:
    targetid: int
    status: str                       # 'ok' | 'input_failed' | 'detection_failed' | 'fit_failed' | 'error'
    total_mag: np.ndarray = None      # (3,) g,r,z MW-corrected
    total_mag_err: np.ndarray = None
    r4_mag: np.ndarray = None
    r4_mag_err: np.ndarray = None
    mw_transmission: np.ndarray = None
    n_components: int = 0
    n_extended: int = 0
    n_stars: int = 0
    n_members: int = 0
    stage_logs: list = field(default_factory=list)
    fragment_path: str = None
    plot_path: str = None
    invariant_max_frac: float = float("nan")
    warnings: list = field(default_factory=list)
    error_msg: str = None


def fragment_path_for(row, cfg=None):
    if cfg is None:
        cfg = ScarletConfig()
    return os.path.join(str(row["FILE_PATH"]), cfg.fragment_name)


def fragment_exists(row, cfg=None):
    """Stage-1 skip signal: the per-object fragment is already written."""
    return os.path.exists(fragment_path_for(row, cfg))


def harvest_result(row, cfg=None):
    """Rebuild a ScarletResult (benchmark mags + counts) from an EXISTING
    fragment, without refitting. Runs the same measure_photometry the live fit
    ran, on the same stored quantities (per-component model-frame fluxes,
    initial membership, rendered science_cube), so the mags are identical to
    what the driver would have written at fit time.

    Exists because the driver only writes the mags FITS at the END of a run: a
    walltime kill loses every completed object's mags row even though their
    fragments are safely on disk (atomic writes). `driver.py --harvest-mags`
    uses this to regenerate the full mags catalog from fragments alone.
    Never raises -- failures reported via ScarletResult.status."""
    import h5py

    if cfg is None:
        cfg = ScarletConfig()
    targetid = int(row["TARGETID"])

    path = fragment_path_for(row, cfg)
    if not os.path.exists(path):
        return ScarletResult(targetid=targetid, status="no_fragment",
                             error_msg="fragment missing: {}".format(path))
    try:
        inp = _inputs.load_object(row, cfg)
        with h5py.File(path, "r") as f:
            comps = f["components"]
            keys = list(comps.keys())
            comp_flux = np.array(
                [[comps[k].attrs["flux_g"], comps[k].attrs["flux_r"],
                  comps[k].attrs["flux_z"]] for k in keys], dtype=np.float64)
            membership = np.array(
                [bool(comps[k].attrs["initial_membership"]) for k in keys])
            dwarf_obs = np.asarray(f["science_cube"][:], dtype=np.float64)
            res = ScarletResult(targetid=targetid, status="ok")
            res.n_components = int(f.attrs["n_components"])
            res.n_members = int(f.attrs["n_members"])
            res.n_extended = int(f.attrs.get("n_extended", 0))
            res.n_stars = int(f.attrs.get("n_stars", 0))
            res.invariant_max_frac = float(
                f.attrs.get("invariant_max_frac", float("nan")))
        phot = _photometry.measure_photometry(inp, comp_flux, membership,
                                              dwarf_obs, cfg)
        res.total_mag = phot.total_mag
        res.total_mag_err = phot.total_mag_err
        res.r4_mag = phot.r4_mag
        res.r4_mag_err = phot.r4_mag_err
        res.mw_transmission = phot.mw_transmission
        return res
    except Exception as exc:                                # noqa: BLE001
        return ScarletResult(targetid=targetid, status="error",
                             error_msg=repr(exc))


def run_scarlet_object(row, cfg=None):
    """Run the full stage-1 fit for one catalogue row. Never raises -- failures
    are reported via ScarletResult.status."""
    import scarlet

    if cfg is None:
        cfg = ScarletConfig()
    targetid = int(row["TARGETID"])

    # ---- inputs ----
    try:
        inp = _inputs.load_object(row, cfg)
    except _inputs.InputError as exc:
        return ScarletResult(targetid=targetid, status="input_failed", error_msg=str(exc))
    except Exception as exc:                                # noqa: BLE001
        return ScarletResult(targetid=targetid, status="input_failed", error_msg=repr(exc))

    res = ScarletResult(targetid=targetid, status="ok", warnings=list(inp.warnings))

    # ---- seeds (detection) ----
    try:
        seeds = _detect.build_seeds(inp, cfg)
    except Exception as exc:                                # noqa: BLE001
        res.status = "detection_failed"
        res.error_msg = repr(exc)
        return res

    # ---- 3-stage fit ----
    try:
        fitr = _fit.run_fit(inp, seeds, cfg)
    except _fit.FitError as exc:
        res.status = "fit_failed"
        res.error_msg = str(exc)
        return res
    except Exception as exc:                                # noqa: BLE001
        res.status = "fit_failed"
        res.error_msg = repr(exc)
        return res

    try:
        # ---- render every component ONCE (observed frame) + model-frame flux ----
        comps = fitr.components
        N = len(comps)
        S = inp.box_size
        comp_flux = np.full((N, 3), np.nan)
        render_full = []
        for i, c in enumerate(comps):
            try:
                comp_flux[i] = np.asarray(scarlet.measure.flux(c), dtype=np.float64)[:3]
            except Exception:                               # noqa: BLE001
                comp_flux[i] = np.nan
            try:
                m = c.get_model(frame=fitr.model_frame)
                render_full.append(np.asarray(fitr.observation.render(m), dtype=np.float64))
            except Exception:                               # noqa: BLE001
                render_full.append(np.zeros((3, S, S)))

        # an uncovered PSF band carries no real flux -> mark it explicitly NaN
        # (rather than the ~0 floor set_spectra_to_match leaves), so colours and
        # mags for that band are NaN by provenance, not a misleading zero.
        covered = np.asarray(fitr.band_covered, dtype=bool)
        if not covered.all():
            comp_flux[:, ~covered] = np.nan

        # ---- grouping (initial membership) ----
        grp = _grouping.group_components(fitr, comp_flux, inp.z, cfg)
        membership = np.asarray(grp.membership, dtype=bool)

        # ---- cubes ----
        full_model = np.zeros((3, S, S))
        for rf in render_full:
            full_model += rf
        dwarf_obs = np.zeros((3, S, S))
        for i in np.where(membership)[0]:
            dwarf_obs += render_full[i]
        nondwarf_obs = full_model - dwarf_obs
        residual = np.asarray(inp.data, dtype=np.float64) - full_model

        # ---- photometry ----
        phot = _photometry.measure_photometry(inp, comp_flux, membership, dwarf_obs, cfg)

        # ---- write fragment ----
        frag = _fragment.write_fragment(
            inp, fitr, comp_flux, grp,
            render_full, dwarf_obs, nondwarf_obs, residual, cfg)

        # ---- optional diagnostic plot ----
        plot_path = None
        if cfg.save_plots:
            from . import plots as _plots
            plot_path = _plots.save_diagnostic_panel(
                os.path.join(inp.file_path, "scarlet_diagnostic.pdf"),
                inp.data, full_model, residual, dwarf_obs,
                title="TARGETID {}".format(targetid))

        res.total_mag = phot.total_mag
        res.total_mag_err = phot.total_mag_err
        res.r4_mag = phot.r4_mag
        res.r4_mag_err = phot.r4_mag_err
        res.mw_transmission = phot.mw_transmission
        res.n_components = frag.n_components
        res.n_extended = fitr.n_extended
        res.n_stars = fitr.n_stars
        res.n_members = frag.n_members
        res.stage_logs = fitr.stage_logs
        res.fragment_path = frag.path
        res.plot_path = plot_path
        res.invariant_max_frac = frag.invariant_max_frac
        return res
    except Exception as exc:                                # noqa: BLE001
        res.status = "error"
        res.error_msg = repr(exc)
        return res

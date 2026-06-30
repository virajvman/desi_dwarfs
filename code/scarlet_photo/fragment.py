"""
fragment.py -- write the per-object VI bundle FRAGMENT.

Stage 1 emits one HDF5 fragment per object at {FILE_PATH}/{fragment_name}
(default scarlet_vi_bundle.h5). Stage 2 (consolidate.py) packs these into the
per-brick bundle store the VI tool reads.

The schema mirrors recon_vi (model-agnostic flux-space compositing), extended
for SCARLET's two complementary model panels:

  datasets (3, S, S) float32:
    input_cutout    real grz data (reference panel)
    science_cube    GALAXY baseline      = sum of initial_membership=True patches
    nondwarf_cube   NOT-GALAXY baseline  = sum of initial_membership=False patches
    residual_cube   data - full model    (optional fit-quality panel)
  group /components/{id}:
    dataset         bbox-cropped observed-frame (3, h, w) float32 patch
    attrs           comp_id, type, is_star, initial_membership, xpix, ypix,
                    ra, dec, bbox (y0,x0), flux_g/r/z (MODEL-frame), gr, rz, gmm_prob

Invariant (exact, since render is linear):
    science_cube + nondwarf_cube + residual_cube == input_cutout

Component IDs are deterministic from the seed (Gaia ref_id for stars, encoded
seed pixel for wavelet sources, -1 for the LSB) so a decision CSV's add/remove
deltas survive a re-fit.
"""

import os
import time
from dataclasses import dataclass

import numpy as np

from .config import ScarletConfig

LSB_COMPONENT_ID = -1


@dataclass
class FragmentSummary:
    path: str
    n_components: int
    n_members: int
    invariant_max_frac: float        # max|input-(sci+nondwarf+resid)|/peak (should be ~0)


def _crop_patch(model, rel_thresh):
    """Crop a (3, S, S) flux model to the bbox of its non-negligible pixels.
    Returns (patch (3,h,w) float32, y0, x0); all-zero -> (3,1,1) zeros at (0,0).
    (Identical contract to recon_vi/build_bundle._crop_patch.)"""
    band_sum = np.abs(model).sum(axis=0)
    peak = float(band_sum.max()) if band_sum.size else 0.0
    if peak <= 0.0:
        return np.zeros((3, 1, 1), dtype=np.float32), 0, 0
    keep = band_sum > (rel_thresh * peak)
    ys, xs = np.where(keep)
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    return np.ascontiguousarray(model[:, y0:y1, x0:x1], dtype=np.float32), y0, x0


def _component_id(meta):
    """Deterministic-from-seed integer ID (stable across re-fits)."""
    if meta["type"] == "starlet_lsb":
        return LSB_COMPONENT_ID
    if meta["is_star"] and meta["ref_id"] is not None:
        return int(meta["ref_id"])                          # Gaia ref_id (huge, distinct)
    cy, cx = meta["center_yx"]
    return int(round(cy)) * 100000 + int(round(cx))         # encoded seed pixel (< 1e10)


def _radec_for(meta, inp):
    if meta["center_yx"] is None:                           # LSB: anchor at galaxy centre
        return inp.fiber_x, inp.fiber_y, inp.ra, inp.dec
    cy, cx = meta["center_yx"]
    # the stored cutout WCS is 3-D (NAXIS=3 grz cube); reduce to its celestial
    # 2-D subset so the 2-coordinate call is accepted (cf. recon_vi _gal_pixel).
    ra, dec = inp.wcs.celestial.all_pix2world(cx, cy, 0)
    return float(cx), float(cy), float(ra), float(dec)


def write_fragment(inp, fit_result, comp_flux, grouping_result,
                   render_full, science_cube, nondwarf_cube, residual_cube,
                   cfg=None):
    """Assemble and atomically write the per-object fragment. Returns a
    FragmentSummary."""
    import h5py

    if cfg is None:
        cfg = ScarletConfig()

    meta = fit_result.comp_meta
    membership = np.asarray(grouping_result.membership, dtype=bool)
    comp_flux = np.asarray(comp_flux, dtype=np.float64)
    S = inp.box_size

    # invariant check (exact by construction; record the residual anyway)
    recombined = science_cube + nondwarf_cube + residual_cube
    peak = float(np.abs(inp.data).max()) or 1.0
    inv_max = float(np.abs(inp.data - recombined).max() / peak)

    nrms = (inp.noise_per_band_rms if inp.noise_per_band_rms is not None
            else np.full(3, np.nan))

    final = os.path.join(inp.file_path, cfg.fragment_name)
    tmp = final + ".tmp"
    if os.path.exists(tmp):
        os.remove(tmp)
    created = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    def _f32(a):
        return np.ascontiguousarray(a, dtype=np.float32)

    n_members = int(membership.sum())
    with h5py.File(tmp, "w") as f:
        # top-level cubes
        for name, cube in (("input_cutout", inp.data),
                           ("science_cube", science_cube),
                           ("nondwarf_cube", nondwarf_cube),
                           ("residual_cube", residual_cube)):
            f.create_dataset(name, data=_f32(cube), compression="gzip", compression_opts=4)

        # top-level attrs
        a = f.attrs
        a["TARGETID"] = int(inp.targetid)
        a["BRICKNAME"] = str(inp.brickname)
        a["box_size"] = int(S)
        a["gal_xpix"] = float(inp.fiber_x)
        a["gal_ypix"] = float(inp.fiber_y)
        a["gal_ra"] = float(inp.ra)
        a["gal_dec"] = float(inp.dec)
        a["gr_cut"] = float(grouping_result.gr_cut)
        a["rz_cut"] = float(grouping_result.rz_cut)
        a["gmm_contour_value"] = float(grouping_result.gmm_contour_value)
        a["target_component_id"] = LSB_COMPONENT_ID         # protected (always-in) component
        a["version"] = str(cfg.version)
        a["created"] = created
        a["n_components"] = len(meta)
        a["n_extended"] = int(fit_result.n_extended)
        a["n_stars"] = int(fit_result.n_stars)
        a["n_members"] = n_members
        a["noise_rms_grz"] = np.asarray(nrms, dtype=np.float32)
        a["invariant_max_frac"] = inv_max
        a["stage_logs"] = np.asarray(fit_result.stage_logs, dtype=np.float64)

        # per-component patches
        grp = f.create_group("components")
        for i, m in enumerate(meta):
            cid = _component_id(m)
            patch, y0, x0 = _crop_patch(render_full[i], cfg.rel_thresh_patch)
            xpix, ypix, ra, dec = _radec_for(m, inp)
            ds = grp.create_dataset(str(cid), data=patch,
                                    compression="gzip", compression_opts=4)
            d = ds.attrs
            d["comp_id"] = int(cid)
            d["type"] = str(m["type"])
            d["is_star"] = bool(m["is_star"])
            d["initial_membership"] = bool(membership[i])
            d["bbox"] = np.array([int(y0), int(x0)], dtype=np.int32)
            d["xpix"] = float(xpix)
            d["ypix"] = float(ypix)
            d["ra"] = float(ra)
            d["dec"] = float(dec)
            d["flux_g"] = float(comp_flux[i, 0])
            d["flux_r"] = float(comp_flux[i, 1])
            d["flux_z"] = float(comp_flux[i, 2])
            d["gr"] = float(grouping_result.gr[i])
            d["rz"] = float(grouping_result.rz[i])
            d["gmm_prob"] = float(grouping_result.gmm_prob[i])

    os.replace(tmp, final)
    return FragmentSummary(path=final, n_components=len(meta),
                           n_members=n_members, invariant_max_frac=inv_max)

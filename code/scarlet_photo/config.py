"""
config.py -- single source of truth for every tunable in the SCARLET photometry
pipeline.

The prototype `code/scarlet_photo.py` scattered ~15 magic numbers through a
1400-line function (model-PSF sigma, min_snr, thresh, min_sep, min_area, 15px
star match, mag<23, colour leniency 0.1, floor 0.3, fit iters, e_rel ...). They
all live here now, as one dataclass, so a run is fully described by its config.

Defaults reproduce the agreed v1 design (grill 2026-06-30; see DESIGN.md). This
module imports only the stdlib + numpy-free dataclasses, so it is safe to import
anywhere (including the container).
"""

from dataclasses import dataclass


BANDS = ("g", "r", "z")
PIXSCALE = 0.262  # arcsec / pixel (Legacy Surveys grz coadds)


@dataclass
class ScarletConfig:
    """Every knob for one SCARLET fit. Construct once per run, pass everywhere."""

    # ---- model frame / PSF -------------------------------------------------
    # Narrow target/model PSF the empirical observation PSF is deconvolved TO.
    # Shared across grz on purpose: the model frame is one common-resolution
    # hyperspectral cube (that is what makes the colours meaningful). 0.8 model-
    # frame px ~ 0.5" FWHM -- narrower than grz seeing in all bands, so every
    # band gets real deconvolution and none is accidentally blurred.
    model_psf_sigma: float = 0.8

    # ---- detection (wavelet peaks on a chi-squared coadd) ------------------
    # 'chi2'          : sqrt(sum_b max(SNR_b,0)^2) -- colour-agnostic, detects a
    #                   source significant in ANY band (recommended).
    # 'ivar_weighted' : sum_b(data_b*iv_b)/sqrt(sum_b iv_b) -- matched filter for
    #                   a flat-spectrum source.
    # 'sum'           : plain g+r+z (the prototype's choice; kept for parity).
    detection_method: str = "chi2"
    wavelet_scales: int = 3            # starlet_transform scales
    wavelet_K: float = 3.0            # K*sigma threshold for multiresolution support
    detect_scale: int = 1            # wavelet scale index used for footprint peaks
    min_separation: float = 7.0        # get_footprints min peak separation (px)
    min_area: int = 10                 # get_footprints min footprint area (px)
    dedup_radius_px: float = 3.0       # drop a wavelet peak within this of a star seed

    # ---- extended-source initialisation (init_all_sources) ----------------
    init_min_snr: float = 50.0
    init_thresh: float = 1.0
    init_max_components: int = 1

    # ---- global LSB StarletSource ------------------------------------------
    starlet_thresh: float = 5e-3       # sparsity threshold; higher => more conservative LSB

    # ---- stars --------------------------------------------------------------
    # Gaia star selection: ref_cat=='G2' AND (type=='PSF' OR proper-motion
    # significant at >pm_sigma). 2 sigma matches the aperture pipeline (the
    # prototype used 3 here -- we unify on 2).
    pm_sigma: float = 2.0

    # ---- fit / convergence --------------------------------------------------
    # e_rel is a STOPPING threshold, not a precision cap. 1e-4 is scarlet's own
    # quickstart value; the prototype's 1e-8/1e-10 never trigger and just burn a
    # fixed 200 iters. max_iter is the real ceiling.
    e_rel: float = 1e-4
    min_iter: int = 10
    max_iter_stage1: int = 300
    max_iter_stage2: int = 200
    max_iter_stage3: int = 200

    # ---- grouping (initial dwarf membership; the VI default) ---------------
    # 'or'      : (blue-box) OR (GMM contour)  -- inclusive, the agreed default.
    # 'bluebox' : LSB-anchored colour box only.
    # 'gmm'     : GMM contour only.
    grouping_rule: str = "or"
    col_lenient: float = 0.1           # widen the blue box by this in g-r and r-z
    color_floor: float = 0.3           # floor on the LSB anchor colour
    gr_cut_fallback: float = 0.2       # used when the LSB colour is NaN
    rz_cut_fallback: float = 0.09
    gmm_contour: str = "98.7"          # confidence-level key for the GMM contour
    gmm_conf_levels: tuple = (0.38, 0.68, 0.86, 0.954, 0.987)

    # ---- photometry ---------------------------------------------------------
    r4_aperture_scale: float = 4.25    # R4 ellipse = semi_major * this (matches aperture pipe)

    # ---- bundle / output ----------------------------------------------------
    rel_thresh_patch: float = 1e-4     # threshold for cropping a component patch (recon_vi parity)
    save_plots: bool = False           # save the data|model|residual|recon diagnostic panel
    fragment_name: str = "scarlet_vi_bundle.h5"   # per-object fragment filename in FILE_PATH

    # ---- store locations (None -> module defaults) -------------------------
    cutouts_dir: str = None
    psfs_dir: str = None

    # ---- version tag carried into outputs ----------------------------------
    version: str = "scarlet_v1"

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
    # K*sigma threshold for the wavelet multiresolution support. THE effective
    # component-count knob (2026-07-02 six-object grid, mean seeds/object:
    # K=3 -> ~82, K=5 -> ~28 at detect_scale=2; detect_scale alone did nothing).
    # Raised 3->5 to curb shredding for the VI.
    wavelet_K: float = 5.0
    # wavelet scale index used for footprint peaks. Tried 2 (coarser, merges
    # small clumps) on 2026-07-02 but it made no measurable difference (the
    # wavelet_K threshold above is the effective component-count knob) and a
    # no-LSB retest at detect_scale=1 didn't regress anything -- reverted to
    # 1 (the prototype's original value) as the standing local default.
    detect_scale: int = 1
    min_separation: float = 7.0        # get_footprints min peak separation (px)
    min_area: int = 10                 # get_footprints min footprint area (px)
    dedup_radius_px: float = 3.0       # drop a wavelet peak within this of a star seed

    # ---- extended-source initialisation (init_all_sources) ----------------
    init_min_snr: float = 50.0
    init_thresh: float = 1.0
    init_max_components: int = 1

    # ---- global LSB StarletSource ------------------------------------------
    # Whether to fit a global LSB StarletSource at all. False skips creating it
    # in stage 2 entirely (stage 2 still runs, letting main-blob sources
    # co-adapt under the off-blob weight mask, just with no diffuse component
    # added); any genuinely diffuse light then goes to the residual or gets
    # absorbed by the extended components instead. DEFAULT FALSE since
    # 2026-07-02 (Viraj): with monotonic=True the untargeted LSB's
    # MonotonicMaskConstraint center is the FIXED geometric center of the
    # WHOLE cutout box -- not the target RA/Dec, not adaptive -- so it was
    # dropped rather than fixed (see DESIGN.md's deferred "Option B" for the
    # target-anchored alternative, not built). The initial-grouping color
    # anchor that used to come from the LSB's own colour now comes from the
    # GMM contour classifier instead (grouping_rule='gmm' below).
    fit_lsb: bool = False
    # Per-scale L0 threshold on the starlet coefficients (the coarsest scale is
    # never thresholded, so raising this suppresses small-scale texture while
    # keeping the smooth envelope). Scarlet's default is 5e-3; raised 10x
    # 2026-07-02 (Viraj: too much small-scale structure in the LSB).
    starlet_thresh: float = 5e-2
    # Replace the per-scale L0 threshold with scarlet's MonotonicMaskConstraint
    # about the box center (= the target position in our centered cutouts):
    # a strictly radially-decreasing diffuse component. NOTE: when True,
    # starlet_thresh is NOT applied at all (scarlet uses one constraint or the
    # other) -- the threshold above only matters with monotonic off.
    # Default True since 2026-07-02 (Viraj).
    lsb_monotonic: bool = True

    # ---- stars --------------------------------------------------------------
    # Gaia star selection: ref_cat=='G2' AND (type=='PSF' OR proper-motion
    # significant at >pm_sigma). 2 sigma matches the aperture pipeline (the
    # prototype used 3 here -- we unify on 2).
    pm_sigma: float = 2.0
    # Leave the stars' sub-pixel shift free during stage 3 (still initialised
    # at the Gaia position) instead of hard-freezing it, to absorb small
    # catalog/coadd astrometry offsets (removes the dipole star residuals;
    # the remaining symmetric cores are empirical-PSF width mismatch).
    # Default True since 2026-07-02 (Viraj).
    star_shift_free: bool = True

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
    # 'or'      : (blue-box) OR (GMM contour)  -- inclusive.
    # 'bluebox' : LSB-anchored colour box only (needs an LSB; meaningless with
    #             fit_lsb=False, where it always falls back to the literal
    #             gr/rz_cut_fallback constants -- too strict for these dwarfs,
    #             starved several test objects down to 0 members).
    # 'gmm'     : GMM contour only -- the DEFAULT since 2026-07-02 (Viraj): with
    #             fit_lsb=False there is no LSB colour to anchor a blue-box on,
    #             so membership is driven purely by the pre-trained per-z-bin
    #             color-color GMM, same as the eventual NERSC default.
    grouping_rule: str = "gmm"
    col_lenient: float = 0.1           # widen the blue box by this in g-r and r-z
    color_floor: float = 0.3           # floor on the LSB anchor colour
    gr_cut_fallback: float = 0.2       # used when the LSB colour is NaN (bluebox/or rules only)
    rz_cut_fallback: float = 0.09
    gmm_contour: str = "98.7"          # confidence-level key for the GMM contour
    gmm_conf_levels: tuple = (0.38, 0.68, 0.86, 0.954, 0.987)
    # Local override for aperture_photo._GMM_MODEL_DIR (a hardcoded NERSC-only
    # pscratch path). If set, GMM pickles load directly from this directory
    # instead -- aperture_photo.py itself is left untouched either way, so the
    # NERSC production path is unaffected. Copy gmm_model_idx_{0..20}.pkl here
    # for local testing (e.g. ~/Downloads/gmm_color_models).
    gmm_model_dir: str = None

    # ---- fixed working box size override -------------------------------------
    # When set, every object is fit at THIS box size regardless of its catalog
    # IMAGE_SIZE_PIX (e.g. the NERSC production catalog varies per row, always
    # >= 350 -- the size validated locally). inputs.py requests this size from
    # cutout_store (safe: it only ever central-crops down, raising if native <
    # requested) and shifts the native-frame per-object quantities
    # (segment_map_v2.npy, fiber_pix_pos.npy, APER_CEN_XY_PIX_ISOLATE) to
    # match, using the row's own IMAGE_SIZE_PIX as the native-size reference.
    # No files are written or mutated on disk -- purely an in-memory crop at
    # load time. None (default) = current behavior, unchanged.
    fixed_box_size: int = None

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

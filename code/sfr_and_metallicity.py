import numpy as np
from astropy.cosmology import Planck18
from desispec.interpolation import resample_flux

def line_snr_mask(fastspec_cat, line_names=["HALPHA"], snr_val=3):
    """
    Returns a boolean mask selecting objects with line flux SNR > snr_val
    for the specified emission lines.
    """
    mask = np.ones(len(fastspec_cat), dtype=bool)

    for li in line_names:
        flux = fastspec_cat[f"{li}_FLUX"]
        ivar = fastspec_cat[f"{li}_FLUX_IVAR"]
        
        snr = flux * np.sqrt(ivar)
        mask &= (snr > snr_val) & (flux > 1) 

    return mask


def compute_o32(fastspec):
    '''
    Function that computes the O32 = OIII 5007 / OII 3726 index
    '''
    o32 = np.array(fastspec["OIII_5007_FLUX"]) / np.array(fastspec["OII_3726_FLUX"])
    return o32 


def compute_r32(fastspec):
    '''
    Function that computes the R32 = (OIII 4959,5007 + OI 3726) / Hbeta index
    '''
    r32 =  ( fastspec["OIII_5007_FLUX"] + fastspec["OIII_4959_FLUX"] + fastspec["OII_3726_FLUX"] ) / fastspec["HBETA_FLUX"]
    return np.array(r32)

##########################################################
##########################################################
# METALLICITY (strong line but we will add direct method later?)
##########################################################
##########################################################


##########################################################
##########################################################
# STAR FORMATION RATES
##########################################################
##########################################################

# -----------------------------------------------------------------------------
# Physical / calibration constants (SI units — Watts throughout, matching
# Bauer et al. 2013 Eq. 2 which is natively in SI)
# -----------------------------------------------------------------------------

# Kennicutt & Evans (2012), Table 1: log C_Hα = 41.27 [erg/s per M_sun/yr],
# natively Kroupa (2001). We rescale to Chabrier (2003) IMF for consistency
# with Chabrier-based stellar masses.
#
# From Madau & Dickinson (2014):
#     M*_Salp : M*_Kroupa : M*_Chab ≈ 1.00 : 0.66 : 0.61
# so  SFR_Chab = SFR_Kroupa * (0.61 / 0.66) ≈ SFR_Kroupa * 0.924.
# Equivalently, the SFR divisor shifts by that factor:
#     log C_Hα (Chabrier) ≈ 41.27 + log10(0.66/0.61) ≈ 41.30
# In SI (W): 10^41.30 / 10^7 = 10^34.30.
#
#   SFR_Chabrier [M_sun/yr] = L(Hα) [W] / 10^34.30
#
# Note: Kroupa vs Chabrier differ by only ~8% (~0.03 dex), well below typical
# systematics (aperture correction, stochastic IMF sampling in dwarfs, dust),
# so this rescaling is a consistency choice, not a physically important shift.
_KENNICUTT_EVANS_12_HA_W_CHABRIER = 10.0**34.30   # W per (M_sun/yr), Chabrier IMF

_HALPHA_REST_A    = 6564.61   # Hα rest wavelength [Å]
_BALMER_INTRINSIC = 2.86      # Case B Hα/Hβ at T_e=1e4 K, n_e=100 cm^-3
_DUST_EXPONENT    = 2.36      # Bauer+13 Eq. 2 dust-correction exponent
_AB_MAG_ZPT       = 34.10     # Bauer+13 Eq. 2 zeropoint; gives L_nu in [W/Hz]
                              # when applied as 10^(-0.4*(M_r - 34.10))
def calc_SFR_Halpha(
    EW_Halpha,
    EW_Halpha_ivar,
    spec_z,
    spec_z_err,
    Mr,
    Mr_err,
    EWc=0.0,
    BD=3.25,
    BD_err=0.1,
    imf_factor=1.0,
):
    """
    Hα star formation rate from fiber spectroscopy via the Bauer+13 / Hopkins+03
    EW × continuum prescription.

    Uses the Kennicutt & Evans (2012) Hα→SFR calibration, rescaled from its
    native Kroupa IMF to a Chabrier (2003) IMF for consistency with Chabrier-
    based stellar masses (the ~8% / ~0.03 dex offset from Madau & Dickinson
    2014).

    Implements Eq. 2 of Bauer et al. (2013, MNRAS 434, 209):

        L(Hα) [W] = (EW + EWc) * 10^(-0.4*(Mr - 34.10))
                    * 3e18 / (6564.61 * (1+z))^2
                    * (BD / 2.86)^2.36

    where 3e18 is the speed of light in Å/s (for the L_ν → L_λ conversion
    via c/λ^2), and 34.10 is the AB absolute-magnitude zeropoint that gives
    L_ν in [W/Hz]. L(Hα) comes out in Watts, and is then divided by
    10^34.30 W/(M_sun/yr) (Kennicutt & Evans 2012 Kroupa, rescaled to
    Chabrier) to get the SFR.
    
    SFR normalization uses the Kennicutt & Evans (2012) Kroupa-calibrated
    value (log C_Hα = 41.27 [erg/s per M_sun/yr] = 34.27 in SI [W per M_sun/yr]),
    which supersedes Kennicutt (1998) Salpeter.

    FIBER vs. GLOBAL SFR — which one you get depends on Mr
    -----------------------------------------------------
    The formula is the same in both cases; what changes is the r-band
    magnitude you pass in:

    - GLOBAL (aperture-corrected) SFR:
        Pass the galaxy's *total* absolute r-band magnitude (e.g. Tractor
        model magnitude or Petrosian). The fiber EW is then scaled by the
        total continuum luminosity, yielding the estimated whole-galaxy
        L(Hα). This assumes EW(Hα) is spatially uniform across the galaxy
        (equivalently: Hα surface brightness ∝ r-band surface brightness).
        This is the Bauer+13 / SAGA IV convention.

    - FIBER SFR (no aperture correction):
        Pass the absolute magnitude *within the fiber* (i.e. synthesize an
        r-band magnitude from the fiber-spectrum continuum at ~6565 Å, or use
        a fiber-aperture photometric measurement). The result is the SFR
        inside the fiber only. Pair this with a stellar mass measured in the
        same fiber aperture to get a self-consistent fiber sSFR.

    Mixing a fiber EW with a total magnitude gives global SFR; mixing a fiber
    EW with a fiber magnitude gives fiber SFR. Do not mix a fiber SFR with a
    total stellar mass (or vice versa) — the two will be inconsistent.

    Parameters
    ----------
    EW_Halpha : array_like
        Rest-frame Hα emission-line equivalent width [Å], measured in the
        fiber. Should be the *emission-only* EW (continuum-subtracted),
        e.g. FastSpecFit's HALPHA_EW column.
    EW_Halpha_ivar : array_like
        Inverse variance on EW_Halpha [Å^-2].
    spec_z : array_like
        Spectroscopic redshift.
    spec_z_err : array_like
        1-sigma uncertainty on spec_z. For DESI this is typically negligible.
    Mr : array_like
        Absolute r-band AB magnitude (total OR fiber, see above), k-corrected
        to z=0 when possible.

        Notes on photometric system:
        - Bauer+13 specifies SDSS r, Petrosian, AB, k-corrected to z=0.
        - DECam r is a close but not identical filter (Δ ~ 0.02-0.05 mag for
          typical star-forming colors); fine as a drop-in for dwarf SFR
          estimates, worth noting in a paper.
        - k-correction effect is <~0.05 mag for blue galaxies at z < ~0.05,
          usually negligible compared to other systematics.
    Mr_err : array_like
        1-sigma uncertainty on Mr [mag].
    EWc : float, optional
        Constant stellar-absorption correction to add to the emission EW [Å].
        Default 0. Bauer+13 use 2.5 Å for a population-averaged correction.
        Set to 0 if your EW already has stellar absorption subtracted (e.g.
        FastSpecFit outputs).
    BD : float or array_like, optional
        Balmer decrement F(Hα)/F(Hβ) used for dust correction. Default 3.25
        (SAGA IV population average; also see Bauer+13). For per-object
        corrections, pass the measured BD array.
    BD_err : float or array_like, optional
        1-sigma uncertainty on BD. Default 0.1.
    imf_factor : float, optional
        Optional additional multiplicative rescaling of the SFR. Default 1.0
        (Chabrier IMF, baked into the divisor). Pass a non-unity value only
        if you want to convert to a different IMF:
            Salpeter: imf_factor = 1.00 / 0.61 ≈ 1.64
            Kroupa:   imf_factor = 0.66 / 0.61 ≈ 1.08
        Normally leave as 1.0.

    Returns
    -------
    log_SFR : ndarray
        log10(SFR / [M_sun yr^-1]).
    log_SFR_err : ndarray
        1-sigma uncertainty on log_SFR [dex], propagated from EW, Mr, z, and BD.

    Assumptions & caveats
    ---------------------
    - When used with a total magnitude (global SFR), assumes EW(Hα) is spatially
      uniform across the galaxy. This can fail badly for compact starburst
      dwarfs, BCDs, or galaxies with off-center star-forming knots.
    - Case B recombination at T_e = 1e4 K, n_e = 100 cm^-3.
    - Fixed dust attenuation law (Bauer+13 Eq. 2 exponent 2.36).
    - Kennicutt-style calibrations assume a fully-sampled IMF and continuous SF
      over ~5-10 Myr; Hα becomes a noisy SFR tracer for SFR < ~0.01 M_sun/yr
      due to stochastic IMF sampling (see e.g. Lee et al. 2009, da Silva et al.
      2012 / SLUG).
    - Error propagation is first-order (Gaussian) and ignores covariance.

    References
    ----------
    Bauer et al. 2013, MNRAS 434, 209 (Eq. 2)
    Hopkins et al. 2003, ApJ 599, 971 (aperture-correction method)
    Kennicutt & Evans 2012, ARA&A 50, 531 (Table 1; Hα → SFR calibration)
    Madau & Dickinson 2014, ARA&A 52, 415 (IMF conversions, Fig. 4)
    Geha et al. 2024, ApJ 976, 118 (SAGA IV; same prescription applied to dwarfs)
    """
    EW_Halpha = np.asarray(EW_Halpha, dtype=float)
    EW_Halpha_ivar = np.asarray(EW_Halpha_ivar, dtype=float)

    # Guard against zero/negative ivar
    with np.errstate(divide="ignore", invalid="ignore"):
        EW_Halpha_err = np.where(EW_Halpha_ivar > 0,
                                 1.0 / np.sqrt(EW_Halpha_ivar),
                                 np.nan)

    EW_total = EW_Halpha + EWc

    # Bauer+13 Eq. 2, three multiplicative terms, in SI (gives L in Watts):
    #   term1: EW × continuum luminosity L_ν from Mr  [W/Hz × Å]
    #   term2: c/λ_obs^2 (c in Å/s), converts L_ν → L_λ  [Hz/Å]
    #   term3: Balmer-decrement dust correction  [dimensionless]
    term1 = EW_total * 10.0 ** (-0.4 * (Mr - _AB_MAG_ZPT))
    term2 = 3.0e18 / (_HALPHA_REST_A * (1.0 + spec_z)) ** 2
    term3 = (BD / _BALMER_INTRINSIC) ** _DUST_EXPONENT

    L_Halpha = term1 * term2 * term3  # [W]

    # Kennicutt & Evans 2012, Kroupa-native, optionally rescaled to another IMF
    SFR = L_Halpha * imf_factor / _KENNICUTT_EVANS_12_HA_W

    with np.errstate(divide="ignore", invalid="ignore"):
        log_SFR = np.log10(SFR)

    # Fractional error propagation
    with np.errstate(divide="ignore", invalid="ignore"):
        term1_EW_frac = EW_Halpha_err / EW_total
        term1_Mr_frac = 0.4 * np.log(10.0) * Mr_err
        term1_frac = np.hypot(term1_EW_frac, term1_Mr_frac)

        term2_frac = 2.0 * np.asarray(spec_z_err) / (1.0 + spec_z)
        term3_frac = _DUST_EXPONENT * (np.asarray(BD_err) / BD)

        L_frac_err = np.sqrt(term1_frac**2 + term2_frac**2 + term3_frac**2)
        log_SFR_err = L_frac_err / np.log(10.0)

    return log_SFR, log_SFR_err


def get_halpha_sfrs(cat, halpha_ew, halpha_ew_ivar):
    """
    Convenience wrapper: compute aperture-corrected (global) Hα SFRs for a
    catalog with DECam Tractor photometry and DESI spectroscopic redshifts.

    Uses MAG_R (total/model magnitude, so this returns GLOBAL SFRs — see the
    `calc_SFR_Halpha` docstring for how to get fiber SFRs instead) and
    LUMI_DIST_MPC from the input catalog. Redshift and photometric errors
    are treated as zero; this is fine for DESI redshifts but ignores the
    (small) Tractor magnitude errors. A population-average Balmer decrement
    is assumed for every galaxy — if per-object BDs are available, call
    calc_SFR_Halpha directly.

    Aperture-correction caveats apply; in particular for low-redshift and/or
    compact dwarf galaxies the assumption of spatially uniform EW(Hα) can
    bias the inferred global SFR significantly.

    Parameters
    ----------
    cat : Table-like
        Must contain columns MAG_R (DECam r, AB, total/model), Z (spec
        redshift), and LUMI_DIST_MPC.
    halpha_ew, halpha_ew_ivar : array_like
        Rest-frame fiber Hα EW [Å] and inverse variance.

    Returns
    -------
    log_halpha_sfr : ndarray
        log10(global SFR / [M_sun yr^-1]), Kroupa IMF (Kennicutt & Evans 2012).
    """
    absm_r = cat["MAG_R"] + 5.0 - 5.0 * np.log10(1e6 * cat["LUMI_DIST_MPC"])
    zeros = np.zeros_like(np.asarray(cat["Z"]), dtype=float)

    log_halpha_sfr, _ = calc_SFR_Halpha(
        EW_Halpha=halpha_ew,
        EW_Halpha_ivar=halpha_ew_ivar,
        spec_z=cat["Z"],
        spec_z_err=zeros,
        Mr=absm_r,
        Mr_err=zeros,
    )
    return log_halpha_sfr


#### include the old funcs from saga ..

def calc_SFR_Halpha_OLD_DO_NOT_USE(EW_Halpha, EW_Halpha_ivar, spec_z, spec_z_err, Mr, r_err, EWc=0, BD=3.25, BD_err=0.1,_IMF_FACTOR = 0.66):
    """
    Calculate Halpha-based EW SFR
    Bauer+ (2013) https://ui.adsabs.harvard.edu/abs/2013MNRAS.434..209B/abstract
    This function does an apeture correction through the Mr term
    we will set EWc = 0, because fastspecfit already accounts for stellar absorption
    """
    EW_Halpha_err = 1/np.sqrt(EW_Halpha_ivar)
    # Bauer, EQ 2, term1
    term1 = (EW_Halpha + EWc) * 10 ** (-0.4 * (Mr - 34.1))
    # Bauer Eq 2, term2
    term2 = 3e18 / (6564.6 * (1.0 + spec_z)) ** 2
    # Balmer Decrement
    term3 = (BD / 2.86) ** 2.36
    L_Halpha = term1 * term2 * term3
    # EQ 3, Bauer et al above, also account for Salpeter -> Koupa IMF
    # in SAGA, they assume some IMF_FACTOR = 0.66. See equation 2 of SAGA IV paper
    #https://github.com/sagasurvey/saga/blob/master/SAGA/objects/calc_sfr.py
    SFR = (L_Halpha * _IMF_FACTOR) / 1.27e34
    log_Ha_SFR = np.log10(SFR)
    # PROPAGATE ERRORS: EW_err, Mr_err and AV_err
    term1_EW_frac_err = EW_Halpha_err / (EW_Halpha + EWc)
    term1_Mr_frac_err = 0.4 * np.log(10) * r_err
    term1_frac_err = np.hypot(term1_EW_frac_err, term1_Mr_frac_err)
    
    term2_frac_err = 2.0 * spec_z_err / (1.0 + spec_z)
    
    term3_frac_err = 2.36 * (BD_err / BD)
    
    L_Halpha_frac_err = np.sqrt(term1_frac_err ** 2 + term2_frac_err ** 2 + term3_frac_err ** 2)
    #the above is the fractional error
    
    log_Ha_SFR_err  = L_Halpha_frac_err / np.log(10)
    return log_Ha_SFR, log_Ha_SFR_err
    
def get_halpha_sfrs_OLD_DO_NOT_USE(cat, halpha_ew, halpha_ew_ivar):
    '''
    Get approximate halpha based sfrs. Approximate because the aperture corrections for lowest redshift galaxies is difficult
    '''
    absm_r = cat["MAG_R"] + 5 - 5*np.log10(1e6*cat["LUMI_DIST_MPC"] )
    log_halpha_sfr, _  = calc_SFR_Halpha(halpha_ew, halpha_ew_ivar, cat["Z"], 0*cat["Z"].data, absm_r, 0 * cat["Z"].data)
    return log_halpha_sfr

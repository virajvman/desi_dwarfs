import os
import tempfile

import numpy as np
import astropy.io.fits as fits
from astropy.cosmology import Planck18
from desispec.interpolation import resample_flux

from mass_and_photo_corrections import (
    DWARF_CATALOG_SPEC_HDU,
    FASTSPEC_DELTA_MAG_COLS,
    safe_read_table,
)
from desi_lowz_funcs import get_stellar_mass_mia

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

import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import minimize
from tqdm import trange


def line_snr(cat, line_flux):
    '''
    Function to apply SNR cuts on line
    '''
    snr_val = cat[line_flux+ "_FLUX"].data * np.sqrt(cat[line_flux+"_FLUX_IVAR"].data)

    return (snr_val > 3) & (cat[line_flux+ "_FLUX"].data > 0)

    
def return_metallicity_estimates_PG16(R2, R3, N2):
    """
    function estimates the metallicity using the PG16 calibrations
    the calibrations are brolen according to the N2 ratio and whether log10(N2) >= -0.6 (up) or not (down)
    """
    log_OH_R = np.ones(len(N2)) * -99

    is_up = (np.log10(N2) >= -0.6) & (N2 > 0)
    is_down = (np.log10(N2) < -0.6) & (N2 > 0)
    log_OH_R[is_up] = 8.589 + 0.022*np.log10(R3[is_up]/R2[is_up]) +\
                      0.399*np.log10(N2[is_up]) + (-0.137 + 0.164*np.log10(R3[is_up]/R2[is_up]) +\
                                                      0.589*np.log10(N2[is_up])) * np.log10(R2[is_up])
    
    log_OH_R[is_down] = 7.932 + 0.944*np.log10(R3[is_down]/R2[is_down]) +\
                        0.695*np.log10(N2[is_down]) + (0.970 - 0.291*np.log10(R3[is_down]/R2[is_down]) -\
                                                          0.019*np.log10(N2[is_down])) * np.log10(R2[is_down])
    
    return log_OH_R


def get_metallicity_P16(fastspec_cat):

    oii_mask_1 = line_snr(fastspec_cat, "OII_3726")
    oii_mask_2 = line_snr(fastspec_cat, "OII_3729")

    oiii_mask_1 = line_snr(fastspec_cat, "OIII_4959")
    oiii_mask_2 = line_snr(fastspec_cat, "OIII_5007")

    hbeta_mask = line_snr(fastspec_cat, "HBETA")
    halpha_mask = line_snr(fastspec_cat, "HALPHA")

    nii_mask_1 = line_snr(fastspec_cat, "NII_6584")
    nii_mask_2 = line_snr(fastspec_cat, "NII_6548")
    
    tot_mask = oii_mask_1 & oii_mask_2 & oiii_mask_1 & oiii_mask_2 & hbeta_mask & halpha_mask & nii_mask_1 & nii_mask_2

    fastspec_cat = fastspec_cat[tot_mask]
    
    oii_3726_flux = fastspec_cat["OII_3726_FLUX"].data
    oii_3729_flux = fastspec_cat["OII_3729_FLUX"].data

    oiii_4959_flux = fastspec_cat["OIII_4959_FLUX"].data
    oiii_5007_flux = fastspec_cat["OIII_5007_FLUX"].data

    nii_flux = fastspec_cat["NII_6584_FLUX"].data
    
    hbeta_flux = fastspec_cat["HBETA_FLUX"].data
    halpha_flux = fastspec_cat["HALPHA_FLUX"].data

    R3 = (oiii_5007_flux * 1.33)/hbeta_flux
    R2 = (oii_3726_flux + oii_3729_flux) / hbeta_flux
    N2 = nii_flux * 1.33 / hbeta_flux
    
    oh_vals = return_metallicity_estimates_PG16(R2, R3, N2)
    
    return oh_vals, tot_mask



def get_metallicity_P16_tgid(tgid, fastspec_cat):

    fastspec_cat = fastspec_cat[fastspec_cat["TARGETID"] == tgid]
    
    oii_mask_1 = line_snr(fastspec_cat, "OII_3726")
    oii_mask_2 = line_snr(fastspec_cat, "OII_3729")

    oiii_mask_1 = line_snr(fastspec_cat, "OIII_4959")
    oiii_mask_2 = line_snr(fastspec_cat, "OIII_5007")

    hbeta_mask = line_snr(fastspec_cat, "HBETA")
    halpha_mask = line_snr(fastspec_cat, "HALPHA")

    nii_mask_1 = line_snr(fastspec_cat, "NII_6584")
    nii_mask_2 = line_snr(fastspec_cat, "NII_6548")
    
    tot_mask = oii_mask_1 & oii_mask_2 & oiii_mask_1 & oiii_mask_2 & hbeta_mask & halpha_mask & nii_mask_1 & nii_mask_2

    fastspec_cat = fastspec_cat[tot_mask]

    if len(fastspec_cat) == 0:
        return -99

    else:
    
        oii_3726_flux = fastspec_cat["OII_3726_FLUX"].data
        oii_3729_flux = fastspec_cat["OII_3729_FLUX"].data
    
        oiii_4959_flux = fastspec_cat["OIII_4959_FLUX"].data
        oiii_5007_flux = fastspec_cat["OIII_5007_FLUX"].data
    
        nii_flux = fastspec_cat["NII_6584_FLUX"].data
        
        hbeta_flux = fastspec_cat["HBETA_FLUX"].data
        halpha_flux = fastspec_cat["HALPHA_FLUX"].data
    
        R3 = (oiii_5007_flux * 1.33)/hbeta_flux
        R2 = (oii_3726_flux + oii_3729_flux) / hbeta_flux
        N2 = nii_flux * 1.33 / hbeta_flux
        
        oh_vals = return_metallicity_estimates_PG16(R2, R3, N2)
    
        return oh_vals[0]



### metallicity measurement from Scholte+22 
def k_ccm89(lam, Rv=3.1, unit_aa=True):
    lam = np.atleast_1d(lam)
    if unit_aa:
        lam=lam/10000.
    else:
        lam=lam
    xs=1/lam
    def a(x):
        y = x-1.82
        if (x>=0.3) & (x<=1.1):
            return 0.574*x**1.61
        elif (x>1.1) & (x<=3.3):
            return 1 + 0.17699*y - 0.50447*y**2 - 0.02427*y**3 + 0.72085*y**4 + 0.01979*y**5 - 0.77530*y**6 + 0.32999*y**7
        else:
            return np.ones_like(x) * np.nan
    def b(x):
        y = x-1.82
        if (x>=0.3) & (x<=1.1):
            return -0.527*x**1.61
        elif (x>1.1) & (x<=3.3):
            return 1.41338*y + 2.28305*y**2 + 1.07233*y**3 - 5.38434*y**4 - 0.62251*y**5 + 5.30260*y**6 - 2.09002*y**7
        else:
            return np.ones_like(x) * np.nan
    if len(lam)==1:
        return [a(x) + b(x)/Rv for x in xs][0]
    else:
        return [a(x) + b(x)/Rv for x in xs]

def attenuation(lam, bd_obs):
    if len(np.atleast_1d(lam))>1.:
        bd_obs = np.atleast_1d(bd_obs)
    return (2.5*np.log10(1/2.86 * bd_obs)/(k_ccm89(4861) - k_ccm89(6563))) * k_ccm89(lam)
    
def transmission(bd_obs, lam):
    return 10**(-0.4*attenuation(lam, bd_obs))
    


def Z_R23_N2(
    OII3727,
    OII3729,
    hb,
    OIII4959,
    OIII5007,
    ha,
    NII6584,
):
    '''
    Code by Dirk Scholte to compute metallicities
    '''
    BD = ha/hb
    if ha/hb>2.86:
        OII3727 = OII3727 / transmission(BD, 3727)
        OII3729 = OII3729 / transmission(BD, 3729)
        hb = hb / transmission(BD, 4861)
        OIII4959 = OIII4959 / transmission(BD, 4959)
        OIII5007 = OIII5007 / transmission(BD, 5007)
        ha = ha / transmission(BD, 6563)
        NII6584 = NII6584 / transmission(BD, 6584)

    R23 = (OII3727 + OII3729 + OIII4959 + OIII5007) / hb
    N2 = NII6584 / ha

    def R23_Z(Z):
        Z = Z - 8.69
        return 0.515 - 1.474 * Z - 1.392 * Z**2 - 0.274 * Z**3
        

    def N2_Z(Z):
        return (Z-9.12)/0.73

    def residuals(
        Z, R23, N2,
    ):
        if (Z < 6.5) | (Z > 9.4):
            return np.inf
        # numerator = np.array(
        #     [   ((R23_Z(Z) - np.log10(R23))/0.10) ** 2,
        #         ((N2_Z(Z) - np.log10(N2))/0.24) ** 2,
        #     ]
        numerator = np.array(
            [   ((R23_Z(Z) - np.log10(R23))/1) ** 2,
                ((N2_Z(Z) - np.log10(N2))/1) ** 2,
            ]
            
        ).reshape(-1)
        residual = np.sum(numerator)
        return residual
    
    def inv_residuals(Z,
        R23, N2,
    ):
        if (Z<6.5) or (Z>9.4):
            return -np.inf
        else:
            return -residuals(Z, R23, N2)
    def wrapped_residuals(Z,
    ):
        if (Z<6.5) or (Z>9.4):
            return np.inf
        else:
            return residuals(Z, R23, N2)

    Zvals = np.linspace(6.5001, 9.3999, 30)
    residual = [residuals(Z, R23, N2) for Z in Zvals]
    Zstart = Zvals[np.argmin(residual)]

    result = minimize(wrapped_residuals, x0=[Zstart], bounds=[[6.5001, 9.3999]], tol=1e-8)
    return (
        result.x[0],
        wrapped_residuals(result.x[0])
    )


def get_metallicity_S22(fastspec_cat):

    oii_mask_1 = line_snr(fastspec_cat, "OII_3726")
    oii_mask_2 = line_snr(fastspec_cat, "OII_3729")

    oiii_mask_1 = line_snr(fastspec_cat, "OIII_4959")
    oiii_mask_2 = line_snr(fastspec_cat, "OIII_5007")

    hbeta_mask = line_snr(fastspec_cat, "HBETA")
    halpha_mask = line_snr(fastspec_cat, "HALPHA")

    nii_mask = line_snr(fastspec_cat, "NII_6584")
    
    nii_mask_2 = line_snr(fastspec_cat, "NII_6548")

    #apply cuts
    sf_Ka03_mask = (np.log10(fastspec_cat["OIII_5007_FLUX"]/fastspec_cat["HBETA_FLUX"]) <= 0.61*(np.log10(fastspec_cat["NII_6584_FLUX"]/fastspec_cat["HALPHA_FLUX"]) - 0.05)**-1 + 1.3) & (np.log10(fastspec_cat["NII_6584_FLUX"]/fastspec_cat["HALPHA_FLUX"]) < 0.0)
    
    sf_Ke01_mask = (np.log10(fastspec_cat["OIII_5007_FLUX"]/fastspec_cat["HBETA_FLUX"]) <= 0.61*(np.log10(fastspec_cat["NII_6584_FLUX"]/fastspec_cat["HALPHA_FLUX"]) - 0.47)**-1 + 1.19) & (np.log10(fastspec_cat["NII_6584_FLUX"]/fastspec_cat["HALPHA_FLUX"]) < 1.0)

    tot_mask = oii_mask_1 & oii_mask_2 & oiii_mask_1 & oiii_mask_2 & hbeta_mask & halpha_mask & nii_mask & nii_mask_2 & sf_Ke01_mask & sf_Ka03_mask

    print(f"Satisyfing line snr+ratio mask = {np.sum(tot_mask)/len(tot_mask)}")

    fastspec_cat_f = fastspec_cat[tot_mask]

    zmetals = []
    for i in trange(len(fastspec_cat_f)):
    
        zmet_i = Z_R23_N2(fastspec_cat_f["OII_3726_FLUX"].data[i], fastspec_cat_f["OII_3729_FLUX"].data[i],
                 fastspec_cat_f["HBETA_FLUX"].data[i], fastspec_cat_f["OIII_4959_FLUX"].data[i], fastspec_cat_f["OIII_5007_FLUX"].data[i],
                 fastspec_cat_f["HALPHA_FLUX"].data[i], fastspec_cat_f["NII_6584_FLUX"].data[i] )

        zmetals.append(zmet_i[0])

    return np.array(zmetals), tot_mask

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
    SFR = L_Halpha * imf_factor / _KENNICUTT_EVANS_12_HA_W_CHABRIER

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


def _fiber_tot_mw_mags(flux_g, flux_r, mw_g, mw_r):
    """Apparent AB mags from Tractor FIBERTOTFLUX and MW transmission (nanomaggy)."""
    with np.errstate(divide="ignore", invalid="ignore"):
        fg = np.asarray(flux_g, dtype=np.float64) / np.asarray(mw_g, dtype=np.float64)
        fr = np.asarray(flux_r, dtype=np.float64) / np.asarray(mw_r, dtype=np.float64)
    mag_g = np.where(np.isfinite(fg) & (fg > 0), 22.5 - 2.5 * np.log10(fg), np.nan)
    mag_r = np.where(np.isfinite(fr) & (fr > 0), 22.5 - 2.5 * np.log10(fr), np.nan)
    return mag_g, mag_r


def _spec_derived_delta_corrected_mags(fspec_cat, mag_g_fib, mag_r_fib, low_snr):
    """
    Sum SPEC_DERIVED DELTA_MAG_* onto fiber mags (BASS2DECAM already north-masked
    when columns were written). Rows with low_snr or non-finite deltas are NaN.
    """
    n = len(fspec_cat)
    mag_g_corr = np.full(n, np.nan, dtype=np.float64)
    mag_r_corr = np.full(n, np.nan, dtype=np.float64)
    for c in FASTSPEC_DELTA_MAG_COLS:
        if c not in fspec_cat.colnames:
            return mag_g_corr, mag_r_corr
    stacks = np.column_stack(
        [np.asarray(fspec_cat[c].data, dtype=np.float64) for c in FASTSPEC_DELTA_MAG_COLS]
    )
    all_finite = np.all(np.isfinite(stacks), axis=1)
    g_sum = stacks[:, 0] + stacks[:, 2] + stacks[:, 4] + stacks[:, 6]
    r_sum = stacks[:, 1] + stacks[:, 3] + stacks[:, 5] + stacks[:, 7]
    ok = (~low_snr) & all_finite
    mag_g_corr[ok] = mag_g_fib[ok] + g_sum[ok]
    mag_r_corr[ok] = mag_r_fib[ok] + r_sum[ok]
    return mag_g_corr, mag_r_corr


def add_sfr_halpha_to_spec_derived(cat_path, verbose=True):
    """
    Append LOG_SFR_HALPHA, LOG_SFR_HALPHA_ERR, LOG_MSTAR_24_FIBER, and
    LOG_HALPHA_SFR_FIBER to the SPEC_DERIVED HDU.

    Must run after consolidate_associated_fiber_properties so MAIN MAG_R and
    LUMI_DIST_MPC are group-consolidated; HALPHA_EW(_IVAR) remain per-fiber from
    SPEC_DERIVED. TARGETID order must match between MAIN and SPEC_DERIVED.

    Fiber mass and SFR use per-target FIBERTOTFLUX (MAIN), MW-corrected to
    apparent mags, then either summed SPEC_DERIVED DELTA_MAG_* (continuum SNR
    >= 10 in MAG_*_FIBER_NOEMI_ERR) or get_stellar_mass_mia fallback with Z_CMB.
    """
    if verbose:
        print("=" * 60)
        print("Adding LOG_SFR_HALPHA and fiber Mstar/SFR columns to SPEC_DERIVED HDU")
        print("=" * 60)

    main_cat = safe_read_table(cat_path, hdu="MAIN")
    fspec_cat = safe_read_table(cat_path, hdu=DWARF_CATALOG_SPEC_HDU)
    n_main = len(main_cat)
    n_fspec = len(fspec_cat)
    if n_main != n_fspec:
        raise ValueError(
            f"MAIN ({n_main} rows) and {DWARF_CATALOG_SPEC_HDU} ({n_fspec} rows) length mismatch"
        )
    tid_main = np.asarray(main_cat["TARGETID"])
    tid_fspec = np.asarray(fspec_cat["TARGETID"])
    if not np.all(tid_main == tid_fspec):
        raise ValueError(
            f"TARGETID mismatch between MAIN and {DWARF_CATALOG_SPEC_HDU}"
        )

    z = np.asarray(main_cat["Z"].data, dtype=float)
    mag_r = np.asarray(main_cat["MAG_R"].data, dtype=float)
    lumi_dist = np.asarray(main_cat["LUMI_DIST_MPC"].data, dtype=float)
    absm_r = mag_r + 5.0 - 5.0 * np.log10(1e6 * lumi_dist)

    halpha_ew = np.asarray(fspec_cat["HALPHA_EW"].data, dtype=float)
    halpha_ew_ivar = np.asarray(fspec_cat["HALPHA_EW_IVAR"].data, dtype=float)

    zeros = np.zeros_like(z, dtype=float)
    log_sfr, log_sfr_err = calc_SFR_Halpha(
        EW_Halpha=halpha_ew,
        EW_Halpha_ivar=halpha_ew_ivar,
        spec_z=z,
        spec_z_err=zeros,
        Mr=absm_r,
        Mr_err=zeros,
        EWc=0.0,
        BD=3.25,
        BD_err=0.0,
        imf_factor=1.0,
    )
    fspec_cat["LOG_SFR_HALPHA"] = log_sfr
    fspec_cat["LOG_SFR_HALPHA_ERR"] = log_sfr_err

    # --- Fiber-aperture stellar mass and Halpha SFR (per plan) ---
    required_main = (
        "FIBERTOTFLUX_G",
        "FIBERTOTFLUX_R",
        "MW_TRANSMISSION_G",
        "MW_TRANSMISSION_R",
        "Z_CMB",
    )
    missing_main = [c for c in required_main if c not in main_cat.colnames]
    if missing_main:
        raise ValueError(
            f"add_sfr_halpha_to_spec_derived: MAIN missing columns {missing_main} "
            "needed for LOG_MSTAR_24_FIBER / LOG_HALPHA_SFR_FIBER"
        )

    mag_g_fib, mag_r_fib = _fiber_tot_mw_mags(
        main_cat["FIBERTOTFLUX_G"].data,
        main_cat["FIBERTOTFLUX_R"].data,
        main_cat["MW_TRANSMISSION_G"].data,
        main_cat["MW_TRANSMISSION_R"].data,
    )
    z_cmb = np.asarray(main_cat["Z_CMB"].data, dtype=float)

    mag_err_limit = 1.0857 / 10.0
    if (
        "MAG_G_FIBER_NOEMI_ERR" in fspec_cat.colnames
        and "MAG_R_FIBER_NOEMI_ERR" in fspec_cat.colnames
    ):
        g_err = np.asarray(fspec_cat["MAG_G_FIBER_NOEMI_ERR"].data, dtype=float)
        r_err = np.asarray(fspec_cat["MAG_R_FIBER_NOEMI_ERR"].data, dtype=float)
        low_snr = (
            ~np.isfinite(g_err)
            | ~np.isfinite(r_err)
            | (g_err >= mag_err_limit)
            | (r_err >= mag_err_limit)
        )
    else:
        low_snr = np.ones(n_fspec, dtype=bool)
        if verbose:
            print(
                "  WARNING: MAG_G_FIBER_NOEMI_ERR / MAG_R_FIBER_NOEMI_ERR missing; "
                "all rows use low-SNR fallback (no DELTA_MAG) for fiber mass/SFR."
            )

    mag_g_corr, mag_r_corr = _spec_derived_delta_corrected_mags(
        fspec_cat, mag_g_fib, mag_r_fib, low_snr
    )

    z_zero = np.zeros(n_fspec, dtype=float)
    log_m_hi = get_stellar_mass_mia(
        mag_g_corr - mag_r_corr,
        mag_g_corr,
        z_zero,
        d_in_mpc=lumi_dist,
        input_zred=False,
    )
    log_m_lo = get_stellar_mass_mia(
        mag_g_fib - mag_r_fib,
        mag_g_fib,
        z_cmb,
        d_in_mpc=lumi_dist,
        input_zred=False,
    )
    log_m_hi = np.asarray(log_m_hi, dtype=np.float64)
    log_m_lo = np.asarray(log_m_lo, dtype=np.float64)
    log_mstar_fiber = np.where(low_snr, log_m_lo, log_m_hi).astype(np.float32)

    mag_r_sfr = np.where(low_snr, mag_r_fib, mag_r_corr)
    absm_r_fiber = mag_r_sfr + 5.0 - 5.0 * np.log10(1e6 * lumi_dist)
    log_sfr_fiber, _ = calc_SFR_Halpha(
        EW_Halpha=halpha_ew,
        EW_Halpha_ivar=halpha_ew_ivar,
        spec_z=z,
        spec_z_err=zeros,
        Mr=absm_r_fiber,
        Mr_err=zeros,
        EWc=0.0,
        BD=3.25,
        BD_err=0.0,
        imf_factor=1.0,
    )
    fspec_cat["LOG_MSTAR_24_FIBER"] = log_mstar_fiber
    fspec_cat["LOG_HALPHA_SFR_FIBER"] = log_sfr_fiber

    fspec_hdu_new = fits.table_to_hdu(fspec_cat)
    fspec_hdu_new.name = DWARF_CATALOG_SPEC_HDU
    fspec_hdu_new.add_checksum()

    cat_abs = os.path.abspath(cat_path)
    cat_dir = os.path.dirname(cat_abs) or "."
    fd, tmp_path = tempfile.mkstemp(
        suffix=".fits", prefix="log_sfr_halpha_", dir=cat_dir
    )
    os.close(fd)
    try:
        with fits.open(cat_abs, memmap=False) as hdul:
            hdu_names = [hdu.name for hdu in hdul]
            main_idx = hdu_names.index("MAIN")
            fspec_idx = hdu_names.index(DWARF_CATALOG_SPEC_HDU)
            main_tab = safe_read_table(cat_abs, hdu="MAIN")
            main_hdu_preserved = fits.table_to_hdu(main_tab)
            main_hdu_preserved.name = "MAIN"
            main_hdu_preserved.add_checksum()
            new_hdus = []
            for i, hdu in enumerate(hdul):
                if i == main_idx:
                    new_hdus.append(main_hdu_preserved)
                elif i == fspec_idx:
                    new_hdus.append(fspec_hdu_new)
                else:
                    new_hdus.append(hdu.copy())
            new_hdul = fits.HDUList(new_hdus)
            new_hdul[0].add_checksum()
            new_hdul.writeto(tmp_path, overwrite=True)
        os.replace(tmp_path, cat_abs)
    except BaseException:
        if os.path.isfile(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass
        raise

    if verbose:
        print(f"Updated {cat_path}:")
        print(
            "  SPEC_DERIVED HDU: added LOG_SFR_HALPHA, LOG_SFR_HALPHA_ERR, "
            "LOG_MSTAR_24_FIBER, LOG_HALPHA_SFR_FIBER"
        )
        print("=" * 60)


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

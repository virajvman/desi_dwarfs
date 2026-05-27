import os
import tempfile

import numpy as np
import astropy.io.fits as fits
from astropy.table import Table
from astropy.cosmology import Planck18
from desispec.interpolation import resample_flux

from mass_and_photo_corrections import (
    DWARF_CATALOG_SPEC_HDU,
    DWARF_CATALOG_DERIVED_HDU,
    FASTSPEC_DELTA_MAG_COLS,
    NEBCORR_DEFAULT_FOLDER,
    _load_nebcorr_delta_mag_table,
    safe_read_table,
    safe_vstack,
)
from desi_lowz_funcs import get_stellar_mass_mia, r_kcorr
from data_model import spec_derived_hdu_datamodel

def line_snr_mask(fastspec_cat, line_names=["HALPHA"], snr_val=3, min_lines=3):
    """
    Returns a boolean mask selecting objects with line flux SNR > snr_val
    in at least `min_lines` of the specified emission lines.
    """
    # Count how many lines pass the SNR cut for each object
    n_pass = np.zeros(len(fastspec_cat), dtype=int)
    for li in line_names:
        flux = fastspec_cat[f"{li}_FLUX"]
        ivar = fastspec_cat[f"{li}_FLUX_IVAR"]
        
        snr = flux * np.sqrt(ivar)
        n_pass += ((snr > snr_val) & (flux > 0)).astype(int)
    
    return n_pass >= min_lines


def compute_o32(fastspec):
    '''
    Function that computes the O32 = OIII 5007 / OII 3726 index
    '''
    o32 = np.array(fastspec["OIII_5007_FLUX"]) / ( np.array(fastspec["OII_3726_FLUX"]) + np.array(fastspec["OII_3729_FLUX"]) )
    return o32 

def compute_r32(fastspec):
    '''
    Function that computes the R32 = (OIII 4959,5007 + OI 3726) / Hbeta index
    '''
    r32 =  ( fastspec["OIII_5007_FLUX"] + fastspec["OIII_4959_FLUX"] + fastspec["OII_3726_FLUX"] + fastspec["OII_3729_FLUX"] ) / fastspec["HBETA_FLUX"]
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


# Line stems for Z_R23_N2 (OII3726/3729, Hβ, OIII, Hα, NII6584) in FastSpec column names.
_R23_N2_LINE_STEMS = (
    "OII_3726",
    "OII_3729",
    "HBETA",
    "OIII_4959",
    "OIII_5007",
    "HALPHA",
    "NII_6584",
)


def r23_n2_line_snr_mask(fastspec_cat):
    """
    True where all seven emission lines used by Z_R23_N2 pass line_snr (SNR > 3, flux > 0).
    """
    mask = np.ones(len(fastspec_cat), dtype=bool)
    for stem in _R23_N2_LINE_STEMS:
        mask &= line_snr(fastspec_cat, stem)
    return mask


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

    snr_r23_n2 = r23_n2_line_snr_mask(fastspec_cat)
    nii_mask_2 = line_snr(fastspec_cat, "NII_6548")

    #apply cuts
    # sf_Ka03_mask = (np.log10(fastspec_cat["OIII_5007_FLUX"]/fastspec_cat["HBETA_FLUX"]) <= 0.61*(np.log10(fastspec_cat["NII_6584_FLUX"]/fastspec_cat["HALPHA_FLUX"]) - 0.05)**-1 + 1.3) & (np.log10(fastspec_cat["NII_6584_FLUX"]/fastspec_cat["HALPHA_FLUX"]) < 0.0)
    
    # sf_Ke01_mask = (np.log10(fastspec_cat["OIII_5007_FLUX"]/fastspec_cat["HBETA_FLUX"]) <= 0.61*(np.log10(fastspec_cat["NII_6584_FLUX"]/fastspec_cat["HALPHA_FLUX"]) - 0.47)**-1 + 1.19) & (np.log10(fastspec_cat["NII_6584_FLUX"]/fastspec_cat["HALPHA_FLUX"]) < 1.0)

    tot_mask = snr_r23_n2 & nii_mask_2 #& sf_Ke01_mask & sf_Ka03_mask

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
# _KENNICUTT_EVANS_12_HA_W_CHABRIER = 10.0**34.30   # W per (M_sun/yr), Chabrier IMF
_BPASS_LOWZ_12_HA_W_CHABRIER = (3.63 * 10**34)   # W per (M_sun/yr), Chabrier IMF

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
    imf_factor=0.94,
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
    SFR = L_Halpha * imf_factor / _BPASS_LOWZ_12_HA_W_CHABRIER

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


def _spec_derived_delta_corrected_mags(fspec_cat, mag_g_base, mag_r_base, low_snr):
    """
    Sum FASTSPEC DELTA_MAG_* onto arbitrary apparent mags (MAIN totals or
    FIBERTOT fiber mags). BASS2DECAM is already north-masked when columns were
    written. Rows with low_snr or non-finite deltas leave NaN in the corrected
    arrays (caller uses low-SNR Mr path instead).
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
    mag_g_base = np.asarray(mag_g_base, dtype=np.float64)
    mag_r_base = np.asarray(mag_r_base, dtype=np.float64)
    mag_g_corr[ok] = mag_g_base[ok] + g_sum[ok]
    mag_r_corr[ok] = mag_r_base[ok] + r_sum[ok]
    return mag_g_corr, mag_r_corr


def _mr_for_halpha_sfr(mag_r_base, gr_for_kcorr, mag_r_corr_high, low_snr, z_cmb, lumi_dist_mpc):
    """
    Absolute r mag for Bauer/Kennicutt-style Hα SFR: high-SNR uses
    emission-model k-term via mag_r_corr_high; low-SNR uses polynomial r_kcorr.
    """
    d_pc = np.asarray(lumi_dist_mpc, dtype=np.float64) * 1.0e6
    with np.errstate(divide="ignore", invalid="ignore"):
        dist_term = 5.0 - 5.0 * np.log10(d_pc)
    mr_hi = np.asarray(mag_r_corr_high, dtype=np.float64) + dist_term
    z_cmb = np.asarray(z_cmb, dtype=np.float64)
    gr = np.asarray(gr_for_kcorr, dtype=np.float64)
    mag_r_b = np.asarray(mag_r_base, dtype=np.float64)
    kr = r_kcorr(gr, z_cmb)
    mr_lo = mag_r_b + dist_term - kr
    return np.where(low_snr, mr_lo, mr_hi)


def _apply_spec_derived_metadata(tab):
    """Apply units / dtypes / descriptions from spec_derived_hdu_datamodel."""
    for col in tab.colnames:
        meta = spec_derived_hdu_datamodel.get(col)
        if meta is None:
            continue
        desired_dtype = np.dtype(meta["dtype"])
        if tab[col].dtype != desired_dtype:
            tab[col] = tab[col].astype(desired_dtype)
        if meta.get("description"):
            tab[col].description = meta["description"]
        if meta.get("unit") is not None:
            tab[col].unit = meta["unit"]
    return tab


def build_spec_derived_hdu(
    cat_path,
    verbose=True,
    n_jobs=1,
    te_method="ultranest",
    min_num_live_points=400,
    te_line_names=("HALPHA", "HBETA", "HGAMMA",
                   "OIII_4363", "OIII_5007", "OII_3726", "OII_3729"),
    te_snr_val=5,
    te_min_lines=7,
):
    """
    Build / refresh the SPEC_DERIVED HDU (DWARF_CATALOG_DERIVED_HDU) of a
    consolidated dwarf catalog with spectroscopically derived nebular
    properties.

    Output columns:

      Existing block (unchanged from previous version):
        TARGETID, LOG_SFR_HALPHA, LOG_SFR_HALPHA_ERR, LOG_MSTAR_24_FIBER,
        LOG_HALPHA_SFR_FIBER, Z_GAS_R23_N2

      DELTA_MAG block (was previously written to the FASTSPEC HDU by
      add_delta_mag_to_fastspec; now lives here, matched by TARGETID to the
      NEBCORR INT_V2 tables, with BASS2DECAM north-masked):
        DELTA_MAG_G_BASS2DECAM, DELTA_MAG_R_BASS2DECAM,
        DELTA_MAG_G_NEB,        DELTA_MAG_R_NEB,
        DELTA_MAG_G_DECAM2SDSS, DELTA_MAG_R_DECAM2SDSS,
        DELTA_MAG_G_KCORR,      DELTA_MAG_R_KCORR

      Direct-method nebular block (Scholte+2026 inference via
      pn_functions.compute_direct_metallicities), populated only for rows
      passing the te_mask (line_snr_mask on te_line_names, snr_val,
      min_lines); NaN / False / 0 elsewhere:
        TE_NE_OII, TE_T_OIII, TE_AV,
        TE_LOG_O2_ABUND, TE_LOG_O3_ABUND, TE_12_LOG_OH
            (each with _LO / _HI / _ERR siblings)
        TE_N_RATIOS, TE_FIT_SUCCESS

    Reads MAIN, FASTSPEC (DWARF_CATALOG_SPEC_HDU), and TRACTOR. The function
    does NOT modify any existing HDU; it builds a fresh BinTableHDU and either
    replaces an existing SPEC_DERIVED HDU or appends a new one. Existing
    HDUs (including FASTSPEC) are preserved bit-for-bit using a temp-file +
    os.replace pattern.

    Parameters
    ----------
    cat_path : str
        Path to the multi-extension dwarf catalog FITS file.
    verbose : bool
        Print progress.
    n_jobs : int
        Number of parallel workers for the per-row direct-method fits
        (forwarded to compute_direct_metallicities). Default 1 (serial).
        Recommended on NERSC compute nodes: number of allocated cores.
    te_method : {'ultranest', 'mle'}
        Direct-method fitting backend. Default 'ultranest'.
    min_num_live_points : int
        UltraNest min_num_live_points (ignored for 'mle'). Default 400.
    te_line_names : iterable of str
        Emission lines fed to line_snr_mask for the te_mask. Default is the
        seven lines required for n_e, T_e, A_V, O+/H+ and O++/H+:
        HALPHA, HBETA, HGAMMA, OIII_4363, OIII_5007, OII_3726, OII_3729.
    te_snr_val : float
        Per-line SNR threshold for te_mask. Default 5.
    te_min_lines : int
        Minimum number of lines passing the per-line SNR cut for te_mask.
        Default 7 (i.e. all of te_line_names must pass).

    Must run after consolidate_associated_fiber_properties so MAIN MAG_R and
    LUMI_DIST_MPC are group-consolidated; HALPHA_EW(_IVAR) remain per-fiber
    from FASTSPEC. TARGETID order must match between MAIN and FASTSPEC.

    Global and fiber Hα SFR share the same continuum-SNR split from
    MAG_{G,R}_FIBER_NOEMI_ERR (threshold SNR 10). High SNR: sum FASTSPEC
    DELTA_MAG_* (nebular, filter, template k-term) onto MAIN mags (global) or
    FIBERTOT mags (fiber). Low SNR: skip deltas and use Chilingarian r_kcorr
    with MAIN g-r (global) or FIBERTOT g-r (fiber). MAG_R_FIBER_NOEMI_ERR is
    passed as Mr_err on the high-SNR branch only (DELTA_MAG terms exact in
    propagation). Fiber-derived DELTA_MAG values applied to MAIN totals for
    global SFR assumes those corrections represent the whole galaxy.

    Stellar mass LOG_MSTAR_24_FIBER uses the same high/low SNR split as
    before (DELTA_MAG on FIBERTOT vs get_stellar_mass_mia with Z_CMB).

    Z_GAS_R23_N2 is gas metallicity from Z_R23_N2 using FASTSPEC line fluxes;
    per-line SNR > 3 (r23_n2_line_snr_mask) with no BPT cuts; NaN otherwise
    or if the fit fails.

    LOG_SFR_HALPHA, LOG_SFR_HALPHA_ERR, and LOG_HALPHA_SFR_FIBER are only set
    for rows with finite HALPHA_FLUX > 0, HBETA_FLUX > 0, HALPHA_EW > 0,
    HALPHA_EW SNR > 3 (EW × sqrt(EW_IVAR)), HALPHA_FLUX SNR > 3, and
    HBETA_FLUX SNR > 3; otherwise those entries are NaN. This is independent
    of the continuum-SNR split from MAG_*_FIBER_NOEMI_ERR above.

    The Balmer decrement used for the SFR dust correction is computed per
    object as BD = HALPHA_FLUX / HBETA_FLUX on rows passing the SFR mask,
    and floored at the Case-B value 2.86 (values below 2.86 are unphysical).
    Rows that do not pass the mask are filled with the population-average
    BD = 3.25, but their SFRs are subsequently nulled to NaN.

    r_kcorr in desi_lowz_funcs is nominally valid for z < 0.5.
    """
    if verbose:
        print("=" * 60)
        print(
            f"Building {DWARF_CATALOG_DERIVED_HDU} HDU "
            "(LOG_SFR_HALPHA, fiber Mstar/SFR, Z_GAS_R23_N2, "
            "DELTA_MAG_*, direct-method TE_*)"
        )
        print("=" * 60)

    main_cat = safe_read_table(cat_path, hdu="MAIN")
    fspec_cat = safe_read_table(cat_path, hdu=DWARF_CATALOG_SPEC_HDU)
    tractor_cat = safe_read_table(cat_path, hdu="TRACTOR")

    print("Finished reading tables!")

    n_main = len(main_cat)
    n_fspec = len(fspec_cat)
    if n_main != n_fspec:
        raise ValueError(
            f"MAIN ({n_main} rows) and {DWARF_CATALOG_SPEC_HDU} "
            f"({n_fspec} rows) length mismatch"
        )
    tid_main = np.asarray(main_cat["TARGETID"])
    tid_fspec = np.asarray(fspec_cat["TARGETID"])
    if not np.all(tid_main == tid_fspec):
        raise ValueError(
            f"TARGETID mismatch between MAIN and {DWARF_CATALOG_SPEC_HDU}"
        )

    z = np.asarray(main_cat["Z"].data, dtype=float)
    mag_g = np.asarray(main_cat["MAG_G"].data, dtype=float)
    mag_r = np.asarray(main_cat["MAG_R"].data, dtype=float)
    lumi_dist = np.asarray(main_cat["LUMI_DIST_MPC"].data, dtype=float)
    z_cmb = np.asarray(main_cat["Z_CMB"].data, dtype=float)

    halpha_ew = np.asarray(fspec_cat["HALPHA_EW"].data, dtype=float)
    halpha_ew_ivar = np.asarray(fspec_cat["HALPHA_EW_IVAR"].data, dtype=float)
    halpha_flux = np.asarray(fspec_cat["HALPHA_FLUX"].data, dtype=float)
    halpha_flux_ivar = np.asarray(fspec_cat["HALPHA_FLUX_IVAR"].data, dtype=float)
    hbeta_flux = np.asarray(fspec_cat["HBETA_FLUX"].data, dtype=float)
    hbeta_flux_ivar = np.asarray(fspec_cat["HBETA_FLUX_IVAR"].data, dtype=float)
    with np.errstate(invalid="ignore"):
        halpha_ew_snr = halpha_ew * np.sqrt(halpha_ew_ivar)
        halpha_flux_snr = halpha_flux * np.sqrt(halpha_flux_ivar)
        hbeta_flux_snr = hbeta_flux * np.sqrt(hbeta_flux_ivar)
    ok_halpha_for_sfr = (
        np.isfinite(halpha_flux)
        & (halpha_flux > 0)
        & np.isfinite(halpha_ew)
        & (halpha_ew > 0)
        & np.isfinite(halpha_ew_ivar)
        & (halpha_ew_ivar > 0)
        & np.isfinite(halpha_ew_snr)
        & (halpha_ew_snr > 3.0)
        & np.isfinite(halpha_flux_ivar)
        & (halpha_flux_ivar > 0)
        & np.isfinite(halpha_flux_snr)
        & (halpha_flux_snr > 3.0)
        & np.isfinite(hbeta_flux)
        & (hbeta_flux > 0)
        & np.isfinite(hbeta_flux_ivar)
        & (hbeta_flux_ivar > 0)
        & np.isfinite(hbeta_flux_snr)
        & (hbeta_flux_snr > 3.0)
    )

    # Per-object Balmer decrement = HALPHA_FLUX / HBETA_FLUX on the SFR-eligible
    # rows; values below the Case-B floor of 2.86 are unphysical and clipped.
    # Rows that fail the mask get the population-average 3.25 fill so the BD
    # array stays finite for calc_SFR_Halpha; their SFRs are nulled below.
    with np.errstate(divide="ignore", invalid="ignore"):
        bd_raw = np.where(
            ok_halpha_for_sfr & (hbeta_flux > 0),
            halpha_flux / hbeta_flux,
            np.nan,
        )
    n_below_bd = int(np.sum(ok_halpha_for_sfr & np.isfinite(bd_raw) & (bd_raw < 2.86)))
    n_eligible = int(np.sum(ok_halpha_for_sfr))
    print(
        f"  build_spec_derived_hdu: {n_below_bd} / {n_eligible} "
        "SFR-eligible objects had per-object BD < 2.86; clipped to 2.86"
    )
    bd_per_object = np.where(
        ok_halpha_for_sfr,
        np.maximum(bd_raw, 2.86),
        3.25,
    )

    mag_err_limit = 1.0857 / 10.0
    if (
        "MAG_G_FIBER_NOEMI_ERR" in fspec_cat.colnames
        and "MAG_R_FIBER_NOEMI_ERR" in fspec_cat.colnames
    ):
        g_err = np.asarray(fspec_cat["MAG_G_FIBER_NOEMI_ERR"].data, dtype=float)
        r_err_noemi = np.asarray(
            fspec_cat["MAG_R_FIBER_NOEMI_ERR"].data, dtype=float
        )
        low_snr = (
            ~np.isfinite(g_err)
            | ~np.isfinite(r_err_noemi)
            | (g_err >= mag_err_limit)
            | (r_err_noemi >= mag_err_limit)
        )
    else:
        low_snr = np.ones(n_fspec, dtype=bool)
        r_err_noemi = np.zeros(n_fspec, dtype=float)
        if verbose:
            print(
                "  WARNING: MAG_G_FIBER_NOEMI_ERR / MAG_R_FIBER_NOEMI_ERR missing; "
                "all rows use low-SNR fallback (no DELTA_MAG) for Hα SFR / fiber mass."
            )

    mag_g_corr_main, mag_r_corr_main = _spec_derived_delta_corrected_mags(
        fspec_cat, mag_g, mag_r, low_snr
    )

    print("Collected the delta mags!")

    mr_global = _mr_for_halpha_sfr(
        mag_r,
        mag_g - mag_r,
        mag_r_corr_main,
        low_snr,
        z_cmb,
        lumi_dist,
    )
    mr_err = np.where(low_snr, 0.0, r_err_noemi)

    zeros = np.zeros_like(z, dtype=float)

    print("Computing SFR now!")

    log_sfr, log_sfr_err = calc_SFR_Halpha(
        EW_Halpha=halpha_ew,
        EW_Halpha_ivar=halpha_ew_ivar,
        spec_z=z,
        spec_z_err=zeros,
        Mr=mr_global,
        Mr_err=mr_err,
        EWc=0.0,
        BD=bd_per_object,
        BD_err=0.0,
        imf_factor=0.94,
    )
    log_sfr = np.where(ok_halpha_for_sfr, log_sfr, np.nan)
    log_sfr_err = np.where(ok_halpha_for_sfr, log_sfr_err, np.nan)

    # --- Fiber-aperture stellar mass and fiber Hα SFR ---
    mag_g_fib, mag_r_fib = _fiber_tot_mw_mags(
        tractor_cat["FIBERTOTFLUX_G"].data,
        tractor_cat["FIBERTOTFLUX_R"].data,
        tractor_cat["MW_TRANSMISSION_G"].data,
        tractor_cat["MW_TRANSMISSION_R"].data,
    )

    mag_g_corr_fib, mag_r_corr_fib = _spec_derived_delta_corrected_mags(
        fspec_cat, mag_g_fib, mag_r_fib, low_snr
    )

    print("Collected the delta mags for fiber-based!")

    z_zero = np.zeros(n_fspec, dtype=float)
    log_m_hi = get_stellar_mass_mia(
        mag_g_corr_fib - mag_r_corr_fib,
        mag_g_corr_fib,
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

    mr_fiber = _mr_for_halpha_sfr(
        mag_r_fib,
        mag_g_fib - mag_r_fib,
        mag_r_corr_fib,
        low_snr,
        z_cmb,
        lumi_dist,
    )
    log_sfr_fiber, _ = calc_SFR_Halpha(
        EW_Halpha=halpha_ew,
        EW_Halpha_ivar=halpha_ew_ivar,
        spec_z=z,
        spec_z_err=zeros,
        Mr=mr_fiber,
        Mr_err=mr_err,
        EWc=0.0,
        BD=bd_per_object,
        BD_err=0.0,
        imf_factor=0.94,
    )
    log_sfr_fiber = np.where(ok_halpha_for_sfr, log_sfr_fiber, np.nan)

    required_z = [
        f"{stem}_{suffix}"
        for stem in _R23_N2_LINE_STEMS
        for suffix in ("FLUX", "FLUX_IVAR")
    ]
    missing_z = [c for c in required_z if c not in fspec_cat.colnames]
    if missing_z:
        raise ValueError(
            f"build_spec_derived_hdu: {DWARF_CATALOG_SPEC_HDU} missing columns "
            f"{missing_z} needed for Z_GAS_R23_N2"
        )

    z_gas = np.full(n_fspec, np.nan, dtype=np.float64)
    mask_z = r23_n2_line_snr_mask(fspec_cat)
    for i in np.flatnonzero(mask_z):
        try:
            z_i = Z_R23_N2(
                fspec_cat["OII_3726_FLUX"].data[i],
                fspec_cat["OII_3729_FLUX"].data[i],
                fspec_cat["HBETA_FLUX"].data[i],
                fspec_cat["OIII_4959_FLUX"].data[i],
                fspec_cat["OIII_5007_FLUX"].data[i],
                fspec_cat["HALPHA_FLUX"].data[i],
                fspec_cat["NII_6584_FLUX"].data[i],
            )
            z_gas[i] = z_i[0]
        except Exception:
            pass

    derived_tab = Table()
    derived_tab["TARGETID"] = np.asarray(tid_main, dtype=np.int64)
    derived_tab["LOG_SFR_HALPHA"] = log_sfr
    derived_tab["LOG_SFR_HALPHA_ERR"] = log_sfr_err
    derived_tab["LOG_MSTAR_24_FIBER"] = log_mstar_fiber
    derived_tab["LOG_HALPHA_SFR_FIBER"] = log_sfr_fiber
    derived_tab["Z_GAS_R23_N2"] = z_gas

    # ------------------------------------------------------------------
    # Block A: DELTA_MAG_* photometric correction columns from NEBCORR.
    # Previously written to the FASTSPEC HDU by add_delta_mag_to_fastspec;
    # now lives in SPEC_DERIVED. BASS2DECAM rows are zeroed for south
    # (is_south == 1); other deltas copied verbatim. Unmatched TARGETIDs
    # leave NaN.
    # ------------------------------------------------------------------
    if verbose:
        print("Adding DELTA_MAG_* photometric correction columns")

    delta_tab = _load_nebcorr_delta_mag_table(
        save_folder=NEBCORR_DEFAULT_FOLDER, verbose=verbose,
    )
    if delta_tab is None:
        if verbose:
            print(
                "  WARNING: No NEBCORR DELTA_MAG tables found; "
                "DELTA_MAG_* columns filled with NaN."
            )
        for col in FASTSPEC_DELTA_MAG_COLS:
            derived_tab[col] = np.full(n_fspec, np.nan, dtype=np.float64)
    else:
        neb_tids = np.asarray(delta_tab["TARGETID"])
        tid_to_row = {int(t): i for i, t in enumerate(neb_tids)}
        is_south_rows = np.asarray(delta_tab["is_south"], dtype=np.int64)
        north_row = (is_south_rows == 0).astype(np.float64)

        matched_rows = np.array(
            [tid_to_row.get(int(t), -1) for t in tid_main], dtype=np.int64,
        )
        n_matched = int(np.sum(matched_rows >= 0))
        if verbose:
            print(
                f"  Matched {n_matched}/{n_fspec} TARGETIDs to NEBCORR "
                "delta-mag table"
            )

        for col in FASTSPEC_DELTA_MAG_COLS:
            arr = np.full(n_fspec, np.nan, dtype=np.float64)
            src = np.asarray(delta_tab[col], dtype=np.float64)
            for j in range(n_fspec):
                row = matched_rows[j]
                if row < 0:
                    continue
                v = src[row]
                if col in ("DELTA_MAG_G_BASS2DECAM",
                           "DELTA_MAG_R_BASS2DECAM"):
                    v = v * north_row[row]
                arr[j] = v
            derived_tab[col] = arr

    # ------------------------------------------------------------------
    # Block B: direct-method nebular fits via
    # pn_functions.compute_direct_metallicities.
    # Only rows passing the te_mask (line_snr_mask on te_line_names,
    # snr_val=te_snr_val, min_lines=te_min_lines) get fits; all other rows
    # have NaN / False / 0 fills.
    # ------------------------------------------------------------------
    if verbose:
        print("Computing direct-method nebular properties (TE_*)")

    # Lazy import: pn_functions builds PyNeb interpolation grids at module
    # import time (~seconds), so we only pay that cost when this function
    # actually runs.
    from pn_functions import compute_direct_metallicities

    # pn_functions PARAM_NAMES are
    #   ['ne_oii', 'te_oiii', 'Av', 'log_O2_abund', 'log_O3_abund']
    # plus the derived 'twelve_log_OH'. Rename to the SPEC_DERIVED TE_*
    # convention.
    _TE_RENAME = {
        "ne_oii":        "TE_NE_OII",
        "te_oiii":       "TE_T_OIII",
        "Av":            "TE_AV",
        "log_O2_abund":  "TE_LOG_O2_ABUND",
        "log_O3_abund":  "TE_LOG_O3_ABUND",
        "twelve_log_OH": "TE_12_LOG_OH",
    }

    te_mask = line_snr_mask(
        fspec_cat,
        line_names=list(te_line_names),
        snr_val=te_snr_val,
        min_lines=te_min_lines,
    )
    n_te = int(te_mask.sum())
    if verbose:
        print(
            f"  te_mask (>= {te_min_lines} of {len(list(te_line_names))} "
            f"lines @ SNR >= {te_snr_val}): {n_te}/{n_fspec} rows"
        )

    # Pre-fill all TE_* columns with default blank values so row order is
    # preserved with no gaps.
    for new_name in _TE_RENAME.values():
        for suffix in ("", "_LO", "_HI", "_ERR"):
            derived_tab[new_name + suffix] = np.full(
                n_fspec, np.nan, dtype=np.float64
            )
    derived_tab["TE_N_RATIOS"] = np.zeros(n_fspec, dtype=np.int32)
    derived_tab["TE_FIT_SUCCESS"] = np.zeros(n_fspec, dtype=bool)

    if n_te > 0:
        idx_te = np.flatnonzero(te_mask)
        fit_tab = compute_direct_metallicities(
            fspec_cat[idx_te],
            method=te_method,
            n_jobs=n_jobs,
            min_num_live_points=min_num_live_points,
            verbose=verbose,
        )

        # Scatter fit_tab rows back to full-length arrays. fit_tab has one
        # row per row in fspec_cat[idx_te], in the same order, so positional
        # assignment via idx_te is correct.
        for orig_name, new_name in _TE_RENAME.items():
            for src_suffix, dst_suffix in (
                ("", ""), ("_lo", "_LO"), ("_hi", "_HI"), ("_err", "_ERR"),
            ):
                src_col = orig_name + src_suffix
                dst_col = new_name + dst_suffix
                if src_col not in fit_tab.colnames:
                    continue
                arr = np.asarray(derived_tab[dst_col], dtype=np.float64).copy()
                arr[idx_te] = np.asarray(fit_tab[src_col], dtype=np.float64)
                derived_tab[dst_col] = arr

        if "n_ratios" in fit_tab.colnames:
            arr = np.asarray(derived_tab["TE_N_RATIOS"], dtype=np.int32).copy()
            arr[idx_te] = np.asarray(fit_tab["n_ratios"], dtype=np.int32)
            derived_tab["TE_N_RATIOS"] = arr
        if "fit_success" in fit_tab.colnames:
            arr = np.asarray(derived_tab["TE_FIT_SUCCESS"], dtype=bool).copy()
            arr[idx_te] = np.asarray(fit_tab["fit_success"], dtype=bool)
            derived_tab["TE_FIT_SUCCESS"] = arr

        if verbose:
            n_ok = int(np.sum(derived_tab["TE_FIT_SUCCESS"]))
            print(f"  TE_FIT_SUCCESS: {n_ok}/{n_te} fits converged")

    _apply_spec_derived_metadata(derived_tab)

    derived_hdu = fits.table_to_hdu(derived_tab)
    derived_hdu.name = DWARF_CATALOG_DERIVED_HDU
    derived_hdu.add_checksum()

    cat_abs = os.path.abspath(cat_path)
    cat_dir = os.path.dirname(cat_abs) or "."
    fd, tmp_path = tempfile.mkstemp(
        suffix=".fits", prefix="spec_derived_", dir=cat_dir
    )
    os.close(fd)
    try:
        with fits.open(cat_abs, memmap=False) as hdul:
            hdu_names = [hdu.name for hdu in hdul]
            new_hdus = []
            replaced = False
            for i, hdu in enumerate(hdul):
                if hdu_names[i] == DWARF_CATALOG_DERIVED_HDU:
                    new_hdus.append(derived_hdu)
                    replaced = True
                else:
                    new_hdus.append(hdu.copy())
            if not replaced:
                new_hdus.append(derived_hdu)
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
        action = "Replaced" if replaced else "Appended"
        print(f"Updated {cat_path}:")
        print(
            f"  {action} {DWARF_CATALOG_DERIVED_HDU} HDU with TARGETID, "
            "LOG_SFR_HALPHA, LOG_SFR_HALPHA_ERR, LOG_MSTAR_24_FIBER, "
            "LOG_HALPHA_SFR_FIBER, Z_GAS_R23_N2, "
            "DELTA_MAG_{G,R}_{BASS2DECAM,NEB,DECAM2SDSS,KCORR}, "
            "TE_NE_OII, TE_T_OIII, TE_AV, TE_LOG_O2_ABUND, "
            "TE_LOG_O3_ABUND, TE_12_LOG_OH (+_LO/_HI/_ERR), "
            "TE_N_RATIOS, TE_FIT_SUCCESS"
        )
        print("=" * 60)


def add_model_photometry_to_spec_derived(
    cat_path,
    model_phot_dir="/pscratch/sd/v/virajvm/catalog_dr1_dwarfs",
    gal_types=("LOWZ", "BGS_FAINT", "BGS_BRIGHT", "ELG", "OTHER"),
    verbose=True,
):
    """
    Read pre-computed fastspec model photometry from
    model_photometry_diffs_{gal_type}.fits files, cross-match by TARGETID,
    and append 10 model-magnitude columns to the SPEC_DERIVED HDU of the
    multi-extension catalog at *cat_path*.

    Must be run after build_spec_derived_hdu has created the SPEC_DERIVED
    HDU; the SPEC_DERIVED TARGETID order is identical to MAIN/FASTSPEC by
    construction. Existing HDUs are preserved bit-for-bit via a temp-file
    + os.replace swap; only the SPEC_DERIVED HDU is rewritten.

    New SPEC_DERIVED columns:
        MAG_{G,R}_DECAM_MODEL_NOEMI   - DECam model mags, continuum only
        MAG_{G,R}_DECAM_MODEL_WEMI    - DECam model mags, continuum + emission
        MAG_{G,R}_BASS_MODEL_WEMI     - BASS  model mags, continuum + emission
        MAG_{G,R}_SDSS_MODEL_NOEMI    - SDSS  model mags, continuum only
        MAG_{G,R}_SDSS_Z0_MODEL_NOEMI - SDSS  z=0 rest-frame model mags, continuum only
    """
    if verbose:
        print("=" * 60)
        print(
            f"Adding fastspec model photometry columns to "
            f"{DWARF_CATALOG_DERIVED_HDU} HDU"
        )
        print("=" * 60)

    # Cache columns sourced from compute_photometry_catalog now describe the
    # continuum-only model variants (smooth_continuum is no longer added in
    # the photometry pipeline). The MAG_*_SDSS_MODEL_NOEMI / *_SDSS_Z0_MODEL_NOEMI
    # values written here therefore use the continuum-only model template,
    # matching the run_nebular_correction_int_v2 chain semantics.
    _COL_MAP = {
        "g_model_no_emi":   "MAG_G_DECAM_MODEL_NOEMI",
        "r_model_no_emi":   "MAG_R_DECAM_MODEL_NOEMI",
        "g_model_w_emi":    "MAG_G_DECAM_MODEL_WEMI",
        "r_model_w_emi":    "MAG_R_DECAM_MODEL_WEMI",
        "g_bass_w_emi":     "MAG_G_BASS_MODEL_WEMI",
        "r_bass_w_emi":     "MAG_R_BASS_MODEL_WEMI",
        "g_sdss_no_emi":    "MAG_G_SDSS_MODEL_NOEMI",
        "r_sdss_no_emi":    "MAG_R_SDSS_MODEL_NOEMI",
        "g_sdss_z0_no_emi": "MAG_G_SDSS_Z0_MODEL_NOEMI",
        "r_sdss_z0_no_emi": "MAG_R_SDSS_Z0_MODEL_NOEMI",
    }

    tables = []
    for gal_type in gal_types:
        path = os.path.join(model_phot_dir, f"model_photometry_diffs_{gal_type}.fits")
        if not os.path.exists(path):
            print(f"  WARNING: {path} not found, skipping {gal_type}")
            continue
        tab = safe_read_table(path)
        if verbose:
            print(f"  Loaded {len(tab)} rows from {path}")
        tables.append(tab)

    if len(tables) == 0:
        print("  ERROR: No model photometry files found. Aborting.")
        return

    model_phot = safe_vstack(tables)

    # De-duplicate on TARGETID (keep first occurrence)
    _, unique_idx = np.unique(np.asarray(model_phot["TARGETID"]), return_index=True)
    model_phot = model_phot[np.sort(unique_idx)]
    if verbose:
        print(f"  Combined model photometry table: {len(model_phot)} unique TARGETIDs")

    derived_cat = safe_read_table(cat_path, hdu=DWARF_CATALOG_DERIVED_HDU)
    n_objects = len(derived_cat)
    cat_tids = np.asarray(derived_cat["TARGETID"])

    if verbose:
        print(f"  {DWARF_CATALOG_DERIVED_HDU} HDU has {n_objects} rows")

    model_tid_to_row = {int(t): i for i, t in enumerate(model_phot["TARGETID"])}

    for old_col, new_col in _COL_MAP.items():
        arr = np.full(n_objects, np.nan, dtype=np.float64)
        src = np.asarray(model_phot[old_col], dtype=np.float64)
        for j, tid in enumerate(cat_tids):
            row = model_tid_to_row.get(int(tid))
            if row is not None:
                arr[j] = src[row]
        derived_cat[new_col] = arr

    n_matched = int(np.sum(np.isfinite(derived_cat["MAG_G_DECAM_MODEL_NOEMI"])))
    if verbose:
        print(f"  Matched {n_matched}/{n_objects} objects to model photometry")

    _apply_spec_derived_metadata(derived_cat)

    derived_hdu_new = fits.table_to_hdu(derived_cat)
    derived_hdu_new.name = DWARF_CATALOG_DERIVED_HDU
    derived_hdu_new.add_checksum()

    cat_abs = os.path.abspath(cat_path)
    cat_dir = os.path.dirname(cat_abs) or "."
    fd, tmp_path = tempfile.mkstemp(
        suffix=".fits", prefix="model_phot_", dir=cat_dir
    )
    os.close(fd)
    try:
        with fits.open(cat_abs, memmap=False) as hdul:
            hdu_names = [hdu.name for hdu in hdul]
            new_hdus = []
            for i, hdu in enumerate(hdul):
                if hdu_names[i] == DWARF_CATALOG_DERIVED_HDU:
                    new_hdus.append(derived_hdu_new)
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

    new_cols_str = ", ".join(_COL_MAP.values())
    if verbose:
        print(f"Updated {cat_path}:")
        print(f"  {DWARF_CATALOG_DERIVED_HDU} HDU: added {new_cols_str}")
        print("=" * 60)


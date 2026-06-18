"""
The attenuation functions of Cardelli et al. 1989.
Code taken from Dirk Scholte's desimetals 
"""
import numpy as np

# Intrinsic (dust-free) Case B Balmer decrement Halpha/Hbeta assumed when
# converting an OBSERVED decrement into a CCM89 attenuation. Single source of
# truth for that constant across the nebular pipeline: the strong-line
# metallicity dereddening (Z_R23_N2), the Halpha->SFR dust term in
# sfr_and_metallicity.calc_SFR_Halpha, and deredden_flux_mstar below all
# reference it. NOTE: this theoretical intrinsic ratio (2.79) is deliberately
# distinct from the empirical low-mass asymptote (2.86) in model_hahb, which is
# a fit to OBSERVED stacked decrements -- so a dust-free-end dwarf still picks
# up a small, physically real correction.
BALMER_INTRINSIC = 2.79


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
        elif (x>3.3) & (x<=8.):
            if x < 5.9:
                fa = 0.
            else:
                fa = -0.04473*(x-5.9)**2 - 0.009779*(x-5.9)**3
            return 1.752 - 0.316*x - 0.104/((x-4.67)**2 + 0.341) + fa
        else:
            return np.ones_like(x) * np.nan
    def b(x):
        y = x-1.82
        if (x>=0.3) & (x<=1.1):
            return -0.527*x**1.61
        elif (x>1.1) & (x<=3.3):
            return 1.41338*y + 2.28305*y**2 + 1.07233*y**3 - 5.38434*y**4 - 0.62251*y**5 + 5.30260*y**6 - 2.09002*y**7
        elif (x>3.3) & (x<=8.):
            if x < 5.9:
                fb = 0.
            else:
                fb = 0.2130*(x-5.9)**2 + 0.1207*(x-5.9)**3
            return -3.090 + 1.825*x + 1.206/((x-4.62)**2 + 0.263) + fb
        else:
            return np.ones_like(x) * np.nan
    if len(lam)==1:
        return [a(x) + b(x)/Rv for x in xs][0]
    else:
        return [a(x) + b(x)/Rv for x in xs]

def attenuation(lam, bd_obs):
    if len(np.atleast_1d(lam))>1.:
        bd_obs = np.atleast_1d(bd_obs)
    return (2.5*np.log10(1/BALMER_INTRINSIC * bd_obs)/(k_ccm89(4861) - k_ccm89(6563))) * k_ccm89(lam)
   
def transmission(bd_obs, lam):
    return 10**(-0.4*attenuation(lam, bd_obs))

def attenuation_Av(lam, Av):
    Av = np.atleast_1d(Av)
    return Av * k_ccm89(lam) 

def transmission_Av(lam, Av):
    Av = np.atleast_1d(Av)
    return 10**(-0.4*attenuation_Av(lam, Av))


def deredden_flux(flux_observed, wavelength_rest, Av, flux_err_observed=None):
    """
    De-redden an observed line flux to its intrinsic (rest-frame) value
    using a Cardelli+1989 extinction law with R_V = 3.1.

    Parameters
    ----------
    flux_observed : float or array
        Observed (attenuated) line flux, in any flux units.
    wavelength_rest : float
        Rest-frame wavelength of the line in Angstroms.
    Av : float or array
        V-band attenuation in magnitudes. Can be array-like to broadcast
        against flux_observed for catalog-level dereddening.
    flux_err_observed : float or array, optional
        1-sigma uncertainty on the observed flux. If given, returned
        alongside the dereddened flux (scaled by the same factor).

    Returns
    -------
    flux_intrinsic : float or array
    flux_err_intrinsic : float or array  (only if flux_err_observed given)
    """
    t = transmission_Av(wavelength_rest, Av)
    flux_intrinsic = flux_observed / t
    if flux_err_observed is None:
        return flux_intrinsic
    return flux_intrinsic, flux_err_observed / t


def model_hahb(log_mstar, A=10.0, k=1.287, x0=11.521):
    """
    Average OBSERVED Balmer decrement Halpha/Hbeta as a function of stellar mass.

    Logistic fit to the stacked Balmer decrements (Halpha/Hbeta measured on
    spectral stacks binned by stellar mass). Used for average internal nebular
    dust corrections when a reliable per-object decrement is unavailable -- e.g.
    the Halpha->SFR dust term, or dereddening line fluxes for population plots:

        Ha/Hb(logM*) = 2.86 + A / (1 + exp(-k * (log_mstar - x0)))

    The additive 2.86 is the EMPIRICAL low-mass asymptote of the observed stacks
    (the dust-free end of the population), NOT the theoretical intrinsic ratio
    used for dereddening (BALMER_INTRINSIC = 2.79). The two differ on purpose:
    even the lowest-mass stacks carry a little dust, so dereddening a model
    decrement against the 2.79 intrinsic yields a small but physically real
    correction.

    Parameters
    ----------
    log_mstar : float or array_like
        log10(stellar mass / M_sun). Use the GLOBAL (total) stellar mass; the
        relation was fit against total stellar mass.
    A, k, x0 : float
        Logistic amplitude, steepness, and midpoint from the stacked-decrement
        fit. Defaults are the production values.

    Returns
    -------
    Ha/Hb : float or ndarray
        Predicted average observed Balmer decrement (>= 2.86).
    """
    log_mstar = np.asarray(log_mstar, dtype=float)
    return 2.86 + A / (1.0 + np.exp(-k * (log_mstar - x0)))


def deredden_flux_mstar(flux_observed, wavelength_rest, log_mstar,
                        flux_err_observed=None):
    """
    De-redden an observed line flux using the mass-based average Balmer
    decrement model_hahb(log_mstar) and a Cardelli+1989 (R_V=3.1) law.

    Mass-based sibling of deredden_flux (which takes an explicit A_V): the
    attenuation is set by the predicted average decrement at this stellar mass
    rather than a per-object measurement. This is the right tool for population
    line-ratio plots, or for correcting fluxes of objects whose own Hbeta is
    too noisy to trust. The intrinsic reference is BALMER_INTRINSIC (2.79), so a
    dust-free-end dwarf (model decrement ~ 2.86) still receives a small
    correction.

    Parameters
    ----------
    flux_observed : float or array
        Observed (attenuated) line flux, in any flux units.
    wavelength_rest : float
        Rest-frame wavelength of the line in Angstroms (a single line).
    log_mstar : float or array
        log10(stellar mass / M_sun); fed to model_hahb. Broadcasts against
        flux_observed for catalog-level dereddening.
    flux_err_observed : float or array, optional
        1-sigma uncertainty on the observed flux. If given, returned alongside
        the dereddened flux (scaled by the same factor).

    Returns
    -------
    flux_intrinsic : float or array
    flux_err_intrinsic : float or array  (only if flux_err_observed given)
    """
    bd = model_hahb(log_mstar)
    t = transmission(bd, wavelength_rest)
    flux_intrinsic = flux_observed / t
    if flux_err_observed is None:
        return flux_intrinsic
    return flux_intrinsic, flux_err_observed / t
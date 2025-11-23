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
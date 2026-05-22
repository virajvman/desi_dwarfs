"""
The attenuation functions of Cardelli et al. 1989.
Code taken from Dirk Scholte's desimetals 
"""
import numpy as np

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
    return (2.5*np.log10(1/2.86 * bd_obs)/(k_ccm89(4861) - k_ccm89(6563))) * k_ccm89(lam)
   
def transmission(bd_obs, lam):
    return 10**(-0.4*attenuation(lam, bd_obs))

def attenuation_Av(lam, Av):
    Av = np.atleast_1d(Av)
    return Av * k_ccm89(lam) 

def transmission_Av(lam, Av):
    Av = np.atleast_1d(Av)
    return 10**(-0.4*attenuation_Av(lam, Av))
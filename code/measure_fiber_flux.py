'''
In this function, we measure the fiber magnitude at the new centers of the objects?
'''
from photutils import CircularAperture, aperture_photometry
from desi_lowz_funcs import flux_to_mag


def measure_simple_fiberflux(xcen, ycen, image_grz_dict, aperture_diam_arcsec = 1.5, pixscale = 0.262 ):
    '''
    Take the reconstruced image (e.g., real data or tractor model) and simply measure the flux within 1.5'' aperture. In this simple case, we are just implicity assuming the PSF of the imaging. Should be a decent first observation. This quantity will be most similar to FIBERTOT_FLUX as we do not just care about the very specific source DESI is targeting.

    Function returns the fiber magnitudes (not extinction corrected)
    '''

    fiberrad = (aperture_diam_arcsec / pixscale) / 2.0

    center_xy = (xcen, ycen) 

    aper = CircularAperture(center_xy, fiberrad)

    #by default, the method is 'exact'
    phot_g = aperture_photometry(image_grz_dict["g"], aper)
    phot_r = aperture_photometry(image_grz_dict["r"], aper)
    phot_z = aperture_photometry(image_grz_dict["z"], aper)

    flux_g = phot_g["aperture_sum"][0]
    flux_r = phot_r["aperture_sum"][0]
    flux_z = phot_z["aperture_sum"][0]

    #convert to magnitude
    mag_g = flux_to_mag(flux_g)
    mag_r = flux_to_mag(flux_r)
    mag_z = flux_to_mag(flux_z)
    
    return [mag_g, mag_r, mag_z]



    

  
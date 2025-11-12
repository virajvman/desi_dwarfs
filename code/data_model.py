from astropy.table import Table, Column
import numpy as np

# Example datamodel template

#different distance estimates, stellar mass estimates, final best photometry, quality_maskbits
#incluce information on closest angular distance to SGA galaxy
import numpy as np
import astropy.units as u

logM_sun = u.def_unit('log(solMass)', format={'latex': r'\log(M_\odot)'})

# maggy = u.def_unit("maggy", 3631 * u.Jy)
# None = u.def_unit("None", 1e-9 * maggy)


main_datamodel = {
    "TARGETID": {
        "unit": None,
        "description": "DESI TARGET ID",
        "dtype": "int64"
    },
    "SURVEY": {
        "unit": None,
        "description": "Survey name",
        "dtype": "str"
    },
    "PROGRAM": {
        "unit": None,
        "description": "Program name",
        "dtype": "str"
    },
    "HEALPIX": {
        "unit": None,
        "description": "healpix containing this location at NSIDE=64 in the NESTED scheme",
        "dtype": "int32"
    },
    "Z": {
        "unit": None,
        "description": "Redrock Redshift (heliocentric)",
        "blank_value": np.nan,
        "dtype": "float64"
    },
    
    "DELTACHI2": {
        "unit": None,
        "description": "Redrock delta-chi-squared",
        "blank_value": np.nan,
        "dtype": "float64"
    },
    "ZWARN": {
        "unit": None,
        "description": "Redrock zwarning bit",
        "dtype": "int8"
    },
    "Z_CMB": {
        "unit": None,
        "description": "Redrock Redshift (CMB rest frame)",
        "blank_value": np.nan,
        "dtype": "float64"
    },
    "RA": {
        "unit": "deg",
        "description": "Right Ascension of the galaxy. Same as target catalog, except for galaxies that are reprocessed after identified as likely shredded.",
        "blank_value": np.nan,
        "dtype": "float64"
    },
    "DEC": {
        "unit": "deg",
        "description": "Declination of the galaxy. Same as target catalog, except for galaxies that are reprocessed after identified as likely shredded.",
        "blank_value": np.nan,
        "dtype": "float64"
    },
    "RA_TARGET": {
        "unit": "deg",
        "description": "Right Ascension from target catalog",
        "blank_value": np.nan,
        "dtype": "float64"
    },
    "DEC_TARGET": {
        "unit": "deg",
        "description": "Declination from target catalog",
        "blank_value": np.nan,
        "dtype": "float64"
    },
    "DESINAME": {
        "unit": None,
        "description": "DESI object name",
        "dtype": "str"
    },
    "LUMI_DIST_MPC": {
        "unit": "Mpc",
        "description": "Fiducial luminosity distance in Mpc.",
        "blank_value": np.nan,
        "dtype": "float32"
    },
    "LOG_MSTAR_SAGA": {
        "unit": logM_sun,
        "description": "Log stellar mass (in Msol) using the LUMI_DIST_MPC luminosity distance and SAGA gr-based approximation",
        "blank_value": np.nan,
        "dtype": "float32"
    },
    "LOG_MSTAR_M24": {
        "unit": logM_sun,
        "description": "Log stellar mass (in Msol) using the LUMI_DIST_MPC luminosity distance and de los Reyes et al. 2024 gr-based approximation",
        "blank_value": np.nan,
        "dtype": "float32"
    },
    "MAG_G": {
        "unit": u.mag,
        "description": "g-band magnitude (MW extinction corrected). Same as Tractor photometry, except for galaxies that are reprocessed after identifed as likely shredded.",
        "blank_value": np.nan,
        "dtype": "float32"
    },
    "MAG_R": {
        "unit": u.mag,
        "description": "Same as MAG_G but for r-band (MW extinction corrected)",
        "blank_value": np.nan,
        "dtype": "float32"
    },
    "MAG_Z": {
        "unit": u.mag,
        "description": "Same as MAG_G but for z-band (MW extinction corrected)",
        "blank_value": np.nan,
        "dtype": "float32"
    },
     "MAG_G_TARGET": {
        "unit": u.mag,
        "description": "Tractor g-band magnitude of DESI target source (MW extinction corrected). For shredded sources, this is the uncorrected, shredded photometry.",
        "blank_value": np.nan,
        "dtype": "float32"
    },
    "MAG_R_TARGET": {
        "unit": u.mag,
        "description": "Same as MAG_G_TARGET but for r-band (MW extinction corrected)",
        "blank_value": np.nan,
        "dtype": "float32"
    },
    "MAG_Z_TARGET": {
        "unit": u.mag,
        "description": "Same as MAG_G_TARGET but for z-band (MW extinction corrected)",
        "blank_value": np.nan,
        "dtype": "float32"
    },
    "SAMPLE": {
        "unit": None,
        "description": "DESI target class (BGS_BRIGHT, BGS_FAINT, LOWZ, or ELG)",
        "dtype": "str"
    },
    "DWARF_MASKBIT": {
        "unit": None,
        "description": "Bitwise mask to apply various cleaning cuts. See for description of bitmasks here.",
        "dtype": "int32"
    },
    "MAG_TYPE": {
        "unit": None,
        "description": "String indicating what kind of photometry is used for MAG_GRZ columns",
        "dtype": "str"
    }, 
    "PHOTOMETRY_UPDATED": {
        "unit": None,
        "description": "Boolean indicating whether the photometry was updated from its original target Tractor photometry.",
        "dtype": "bool"
    },
     "R50_R": {
        "unit": None,
        "description": "Half-light radius in arc-seconds",
        "blank_value": np.nan,
        "dtype": "float32"
    },
    "SHAPE_PARAMS": {
        "unit": None,
        "description": "Galaxy shape parameters: b/a ratio, position angle",
        "shape": (2,),
        "blank_value": np.nan,
        "dtype": "float32"
    },
    "IN_SGA_2020": {
        "unit": None,
        "description": "Boolean indicating whether targeted source had Tractor MASKBITS=12, that is, in SGA-2020 catalog",
        "dtype": "bool"
    }
}


zcat_datamodel = {
     "TARGETID": {
        "unit": None,
        "description": "DESI TARGET ID",
        "dtype": "int64"
    },
    "CMX_TARGET": {
        "unit": None,
        "description": "Commissioning (CMX) targeting bit",
        "dtype": "int64"
    },
    "DESI_TARGET": {
        "unit": None,
        "description": "DESI targeting bit",
        "dtype": "int64"
    },
    "BGS_TARGET": {
        "unit": None,
        "description": "BGS targeting bit",
        "dtype": "int64"
    },
    "MWS_TARGET": {
        "unit": None,
        "description": "MWS targeting bit",
        "dtype": "int64"
    },
    "SCND_TARGET": {
        "unit": None,
        "description": "Secondary target targeting bit",
        "dtype": "int64"
    },
    "SV1_DESI_TARGET": {
        "unit": None,
        "description": "SV1 DESI targeting bit",
        "dtype": "int64"
    },
    "SV1_BGS_TARGET": {
        "unit": None,
        "description": "SV1 BGS targeting bit",
        "dtype": "int64"
    },
    "SV1_MWS_TARGET": {
        "unit": None,
        "description": "SV1 MWS targeting bit",
        "dtype": "int64"
    },
    "SV2_DESI_TARGET": {
        "unit": None,
        "description": "SV2 DESI targeting bit",
        "dtype": "int64"
    },
    "SV2_BGS_TARGET": {
        "unit": None,
        "description": "SV2 BGS targeting bit",
        "dtype": "int64"
    },
    "SV2_MWS_TARGET": {
        "unit": None,
        "description": "SV2 MWS targeting bit",
        "dtype": "int64"
    },
    "SV3_DESI_TARGET": {
        "unit": None,
        "description": "SV3 DESI targeting bit",
        "dtype": "int64"
    },
    "SV3_BGS_TARGET": {
        "unit": None,
        "description": "SV3 BGS targeting bit",
        "dtype": "int64"
    },
    "SV3_MWS_TARGET": {
        "unit": None,
        "description": "SV3 MWS targeting bit",
        "dtype": "int64"
    },
    "SV1_SCND_TARGET": {
        "unit": None,
        "description": "SV1 secondary targeting bit",
        "dtype": "int64"
    },
    "SV2_SCND_TARGET": {
        "unit": None,
        "description": "SV2 secondary targeting bit",
        "dtype": "int64"
    },
    "SV3_SCND_TARGET": {
        "unit": None,
        "description": "SV3 secondary targeting bit",
        "dtype": "int64"
    },
    "TSNR2_LRG": {
        "unit": None,
        "description": "LRG template (S/N)^2 summed over B,R,Z",
        "blank_value": np.nan,
        "dtype": "float32"
    },
    "CHI2": {
        "unit": None,
        "description": "Best fit Redrock chi squared",
        "blank_value": np.nan,
        "dtype": "float32"
    },

    "OBJTYPE": {
        "unit": None,
        "description": "Object type: TGT, SKY, NON, BAD",
        "blank_value": None,
        "dtype": "str",  # stored as char[3] in FITS
    },

    "OBSCONDITIONS": {
        "unit": None,
        "description": "Flag the target to be observed in graytime",
        "blank_value": None,
        "dtype": "int32",
    },

    "COADD_NUMEXP": {
        "unit": None,
        "description": "Number of exposures in coadd",
        "blank_value": None,
        "dtype": "int16",
    },

    "COADD_EXPTIME": {
        "unit": "s",
        "description": "Summed exposure time for coadd",
        "blank_value": np.nan,
        "dtype": "float32",
    },

    "COADD_NUMTILE": {
        "unit": None,
        "description": "Number of tiles in coadd",
        "blank_value": None,
        "dtype": "int16",
    },

    "MEAN_PSF_TO_FIBER_SPECFLUX": {
        "unit": None,
        "description": "Mean fraction of light from point-like source captured by 1.5 arcsec diameter fiber given atmospheric seeing",
        "blank_value": np.nan,
        "dtype": "float32",
    },

    "MIN_MJD": {
        "unit": "d",
        "description": "Minimum Modified Julian Date when the shutter was open for the first exposure used in the coadded spectrum",
        "blank_value": np.nan,
        "dtype": "float64",
    },

    "MAX_MJD": {
        "unit": "d",
        "description": "Maximum Modified Julian Date when the shutter was open for the last exposure used in the coadded spectrum",
        "blank_value": np.nan,
        "dtype": "float64",
    },

    "MEAN_MJD": {
        "unit": "d",
        "description": "Mean Modified Julian Date over exposures used in the coadded spectrum",
        "blank_value": np.nan,
        "dtype": "float64",
    },

    "ZCAT_NSPEC": {
        "unit": None,
        "description": "Number of times this TARGETID appears in this catalog",
        "blank_value": None,
        "dtype": "int16",
    },

    "ZCAT_PRIMARY": {
        "unit": None,
        "description": "Boolean flag (True/False) for the primary coadded spectrum in zpix zcatalog",
        "blank_value": None,
        "dtype": "bool",
    }

}


#also include the most relevant tractor columns!! This is of the original target catalog

tractor_datamodel = {
    "TARGETID": {
        "unit": None,
        "description": "DESI TARGET ID",
        "dtype": "int64"
    },

    "RELEASE": { 
        "unit": None, 
        "description": "Integer denoting the camera and filter set used, which will be unique for a given processing run of the data.", 
        "blank_value": None, 
        "dtype": "int16"
    },

    "BRICKNAME": {
        "unit": None,
        "description": (
            "Name of the sky brick, encoding RA and Dec (e.g., '1126p222' "
            "for RA=112.6, Dec=+22.2)."
        ),
        "blank_value": None,
        "dtype": "str",
    },

    "BRICKID": {
        "unit": None,
        "description": "Integer ID of the brick [1–662174].",
        "blank_value": None,
        "dtype": "int32",
    },

    "BRICK_OBJID": {
        "unit": None,
        "description": "Catalog object number within this brick. Unique identifier when combined with RELEASE and BRICKID.",
        "blank_value": None,
        "dtype": "int32",
    },

    "EBV": {
        "unit": u.mag,
        "description": "Galactic extinction E(B-V) reddening from SFD98, used to compute the mw_transmission_ columns",
        "blank_value": np.nan,
        "dtype": "float32",
    },

    "FIBERFLUX_R": {
        "unit": None,
        "description": (
            "Predicted r-band flux within a 1.5'' diameter fiber under 1' "
            "Gaussian seeing (not extinction corrected)."
        ),
        "blank_value": np.nan,
        "dtype": "float32",
    },

    "MASKBITS": {
        "unit": None,
        "description": (
            "Tractor Bitwise mask indicating that an object touches a pixel in "
            "the coadd maskbits maps (see DR9 bitmasks documentation)."
        ),
        "blank_value": None,
        "dtype": "int16",
    },

    "REF_ID": {
        "unit": None,
        "description": (
            "Reference catalog source ID (Tyc1*1e6 + Tyc2*10 + Tyc3 for Tycho-2, "
            "‘sourceid’ for Gaia DR2)."
        ),
        "blank_value": None,
        "dtype": "int64",
    },

    "REF_CAT": {
        "unit": None,
        "description": (
            "Reference catalog identifier: 'T2' (Tycho-2), 'G2' (Gaia DR2), "
            "'L3' (SGA), or empty if none."
        ),
        "blank_value": None,
        "dtype": "str",
    },

    "FLUX_G": {
        "unit": None,
        "description": "Total g-band flux corrected for Galactic extinction.",
        "blank_value": np.nan,
        "dtype": "float32",
    },

    "FLUX_IVAR_G": {
        "unit": None,
        "description": "Inverse variance of FLUX_G (extinction corrected).",
        "blank_value": 0.0,
        "dtype": "float32",
    },

    "MAG_G": {
        "unit": u.mag,
        "description": "Extinction-corrected g-band magnitude.",
        "blank_value": np.nan,
        "dtype": "float32",
    },

    "MAG_G_ERR": {
        "unit": u.mag,
        "description": "Uncertainty in g-band magnitude.",
        "blank_value": np.nan,
        "dtype": "float32",
    },

    "FLUX_R": {
        "unit": None,
        "description": "Total r-band flux corrected for Galactic extinction.",
        "blank_value": np.nan,
        "dtype": "float32",
    },

    "FLUX_IVAR_R": {
        "unit": None,
        "description": "Inverse variance of FLUX_R (extinction corrected).",
        "blank_value": 0.0,
        "dtype": "float32",
    },

    "MAG_R": {
        "unit": u.mag,
        "description": "Extinction-corrected r-band magnitude.",
        "blank_value": np.nan,
        "dtype": "float32",
    },

    "MAG_R_ERR": {
        "unit": u.mag,
        "description": "Uncertainty in r-band magnitude.",
        "blank_value": np.nan,
        "dtype": "float32",
    },

    "FLUX_Z": {
        "unit": None,
        "description": "Total z-band flux corrected for Galactic extinction.",
        "blank_value": np.nan,
        "dtype": "float32",
    },

    "FLUX_IVAR_Z": {
        "unit": None,
        "description": "Inverse variance of FLUX_Z (extinction corrected).",
        "blank_value": 0.0,
        "dtype": "float32",
    },

    "MAG_Z": {
        "unit": u.mag,
        "description": "Extinction-corrected z-band magnitude.",
        "blank_value": np.nan,
        "dtype": "float32",
    },

    "MAG_Z_ERR": {
        "unit": u.mag,
        "description": "Uncertainty in z-band magnitude.",
        "blank_value": np.nan,
        "dtype": "float32",
    },

    "FIBERMAG_R": {
        "unit": u.mag,
        "description": (
            "Predicted r-band magnitude within 1.5'' fiber (not extinction corrected)."
        ),
        "blank_value": np.nan,
        "dtype": "float32",
    },

    "OBJID": {
        "unit": None,
        "description": (
            "Object number within the brick (0–N−1), unique within "
            "a given RELEASE and BRICKID."
        ),
        "blank_value": None,
        "dtype": "int32",
    },

    "SIGMA_G": {
        "unit": "arcsec",
        "description": "Gaussian sigma of the object model in g-band.",
        "blank_value": np.nan,
        "dtype": "float32",
    },

    "FRACFLUX_G": {
        "unit": None,
        "description": (
            "Profile-weighted fraction of flux from neighboring sources "
            "divided by total flux in g-band."
        ),
        "blank_value": np.nan,
        "dtype": "float32",
    },

    "RCHISQ_G": {
        "unit": None,
        "description": "Reduced chi-squared of the g-band model fit.",
        "blank_value": np.nan,
        "dtype": "float32",
    },

    "SIGMA_R": {
        "unit": "arcsec",
        "description": "Gaussian sigma of the object model in r-band.",
        "blank_value": np.nan,
        "dtype": "float32",
    },

    "FRACFLUX_R": {
        "unit": None,
        "description": (
            "Profile-weighted fraction of flux from neighboring sources "
            "divided by total flux in r-band."
        ),
        "blank_value": np.nan,
        "dtype": "float32",
    },

    "RCHISQ_R": {
        "unit": None,
        "description": "Reduced chi-squared of the r-band model fit.",
        "blank_value": np.nan,
        "dtype": "float32",
    },

    "SIGMA_Z": {
        "unit": "arcsec",
        "description": "Gaussian sigma of the object model in z-band.",
        "blank_value": np.nan,
        "dtype": "float32",
    },

    "FRACFLUX_Z": {
        "unit": None,
        "description": (
            "Profile-weighted fraction of flux from neighboring sources "
            "divided by total flux in z-band."
        ),
        "blank_value": np.nan,
        "dtype": "float32",
    },

    "RCHISQ_Z": {
        "unit": None,
        "description": "Reduced chi-squared of the z-band model fit.",
        "blank_value": np.nan,
        "dtype": "float32",
    },

    "SHAPE_R": {
        "unit": "arcsec",
        "description": "Half-light radius of the best-fit galaxy model (r-band).",
        "blank_value": np.nan,
        "dtype": "float32",
    },

    "SHAPE_R_ERR": {
        "unit": "arcsec",
        "description": "Uncertainty in the half-light radius (r-band).",
        "blank_value": np.nan,
        "dtype": "float32",
    },

    "MU_R": {
        "unit": "mag/arcsec^2",
        "description": "Surface brightness within the effective radius in r-band.",
        "blank_value": np.nan,
        "dtype": "float32",
    },

    "MU_R_ERR": {
        "unit": "mag/arcsec^2",
        "description": "Uncertainty in the surface brightness (r-band).",
        "blank_value": np.nan,
        "dtype": "float32",
    },

    "SERSIC": {
        "unit": None,
        "description": "Power-law index for the Sersic profile model (type='SER').",
        "blank_value": np.nan,
        "dtype": "float32",
    },

    "SERSIC_IVAR": {
        "unit": None,
        "description": "Inverse variance of the Sersic index parameter.",
        "blank_value": 0.0,
        "dtype": "float32",
    },

    "BA": {
        "unit": None,
        "description": "Axis ratio (b/a) of the best-fit galaxy model.",
        "blank_value": np.nan,
        "dtype": "float32",
    },

    "TYPE": {
        "unit": None,
        "description": "Object type as classified by the Tractor model.",
        "blank_value": None,
        "dtype": "str",
    },

    "PHI": {
        "unit": "deg",
        "description": "Position angle of the major axis",
        "blank_value": np.nan,
        "dtype": "float32",
    },

    "NOBS_G": {
        "unit": None,
        "description": (
            "Number of images contributing to the central pixel in the g-band."
        ),
        "blank_value": 0,
        "dtype": "int16",
    },

    "NOBS_R": {
        "unit": None,
        "description": (
            "Number of images contributing to the central pixel in the r-band."
        ),
        "blank_value": 0,
        "dtype": "int16",
    },

    "NOBS_Z": {
        "unit": None,
        "description": (
            "Number of images contributing to the central pixel in the z-band."
        ),
        "blank_value": 0,
        "dtype": "int16",
    },

    "MW_TRANSMISSION_G": {
        "unit": None,
        "description": "Galactic transmission in g filter in linear units [0, 1]",
        "blank_value": np.nan,
        "dtype": "float32",
    },

    "MW_TRANSMISSION_R": {
        "unit": None,
        "description": "Galactic transmission in r filter in linear units [0, 1]",
        "blank_value": np.nan,
        "dtype": "float32",
    },

    "MW_TRANSMISSION_Z": {
        "unit": None,
        "description": "Galactic transmission in z filter in linear units [0, 1]",
        "blank_value": np.nan,
        "dtype": "float32",
    },

    "SWEEP": {
        "unit": None,
        "description": "Name of the sweep file from which this source was extracted.",
        "blank_value": None,
        "dtype": "str",
    },
}



Nparams = 5

#these are only the columns we hope to retain in the final catalog
photo_datamodel = {
    "TARGETID": {
        "unit": None,
        "description": "DESI TARGET ID",
        "dtype": "int64"
    },
    # --- COG magnitudes ---
    "COG_MAG_G_ISOLATE": {
        "unit": u.mag,
        "description": "COG magnitude in g-band (with isolate mask); MW extinction corrected",
        "blank_value": np.nan,
        "dtype": "float32"
    },
    
    "COG_MAG_R_ISOLATE": {
        "unit": u.mag,
        "description": "COG magnitude in r-band (with isolate mask); MW extinction corrected",
        "blank_value": np.nan,
        "dtype": "float32"
    },
    "COG_MAG_Z_ISOLATE": {
        "unit": u.mag,
        "description": "COG magnitude in z-band (with isolate mask); MW extinction corrected",
        "blank_value": np.nan,
        "dtype": "float32"
    },

    ##

    "COG_MAG_G_NO_ISOLATE": {
        "unit": u.mag,
        "description": "COG magnitude in g-band (without isolate mask); MW extinction corrected",
        "blank_value": np.nan,
        "dtype": "float32"
    },

    "COG_MAG_R_NO_ISOLATE": {
        "unit": u.mag,
        "description": "COG magnitude in r-band (without isolate mask); MW extinction corrected",
        "blank_value": np.nan,
        "dtype": "float32"
    },

    "COG_MAG_Z_NO_ISOLATE": {
        "unit": u.mag,
        "description": "COG magnitude in z-band (without isolate mask); MW extinction corrected",
        "blank_value": np.nan,
        "dtype": "float32"
    },


    # --- Aper R4 ---
    "APER_R4_MAG_G_ISOLATE": {
        "unit": u.mag,
        "description": "R4 aperture magnitude in g-band (with isolate mask); MW extinction corrected",
        "blank_value": np.nan,
        "dtype": "float32"
    },

    "APER_R4_MAG_R_ISOLATE": {
        "unit": u.mag,
        "description": "R4 aperture magnitude in r-band (with isolate mask); MW extinction corrected",
        "blank_value": np.nan,
        "dtype": "float32"
    },

    "APER_R4_MAG_Z_ISOLATE": {
        "unit": u.mag,
        "description": "R4 aperture magnitude in z-band (with isolate mask); MW extinction corrected",
        "blank_value": np.nan,
        "dtype": "float32"
    },

    ##

    "APER_R4_MAG_G_NO_ISOLATE": {
        "unit": u.mag,
        "description": "R4 aperture magnitude in g-band (without isolate mask); MW extinction corrected",
        "blank_value": np.nan,
        "dtype": "float32"
    },

    "APER_R4_MAG_R_NO_ISOLATE": {
        "unit": u.mag,
        "description": "R4 aperture magnitude in r-band (without isolate mask); MW extinction corrected",
        "blank_value": np.nan,
        "dtype": "float32"
    },

    "APER_R4_MAG_Z_NO_ISOLATE": {
        "unit": u.mag,
        "description": "R4 aperture magnitude in z-band (without isolate mask); MW extinction corrected",
        "blank_value": np.nan,
        "dtype": "float32"
    },
    
    
    # --- Tractor ---
    "TRACTOR_BASED_MAG_G_ISOLATE": {
        "unit": u.mag,
        "description": "Tractor based parent magnitude in g-band (with isolate mask); MW extinction corrected",
        "blank_value": np.nan,
        "dtype": "float32"
    },

    "TRACTOR_BASED_MAG_R_ISOLATE": {
        "unit": u.mag,
        "description": "Tractor based parent magnitude in r-band (with isolate mask); MW extinction corrected",
        "blank_value": np.nan,
        "dtype": "float32"
    },

    "TRACTOR_BASED_MAG_Z_ISOLATE": {
        "unit": u.mag,
        "description": "Tractor based parent magnitude in z-band (with isolate mask); MW extinction corrected",
        "blank_value": np.nan,
        "dtype": "float32"
    },

    ##
    
    "TRACTOR_BASED_MAG_G_NO_ISOLATE": {
        "unit": u.mag,
        "description": "Tractor based parent magnitude in g-band (without isolate mask); MW extinction corrected",
        "blank_value": np.nan,
        "dtype": "float32"
    },

    "TRACTOR_BASED_MAG_R_NO_ISOLATE": {
        "unit": u.mag,
        "description": "Tractor based parent magnitude in r-band (without isolate mask); MW extinction corrected",
        "blank_value": np.nan,
        "dtype": "float32"
    },

    "TRACTOR_BASED_MAG_Z_NO_ISOLATE": {
        "unit": u.mag,
        "description": "Tractor based parent magnitude in z-band (without isolate mask); MW extinction corrected",
        "blank_value": np.nan,
        "dtype": "float32"
    },

    # --- Simplest photometry ---
    
    "SIMPLE_PHOTO_MAG_G": {
        "unit": u.mag,
        "description": "Simplest photometry method based magnitude in g-band (with isolate mask); MW extinction corrected",
        "blank_value": np.nan,
        "dtype": "float32"
    },

    "SIMPLE_PHOTO_MAG_R": {
        "unit": u.mag,
        "description": "Simplest photometry method based magnitude in r-band (with isolate mask); MW extinction corrected",
        "blank_value": np.nan,
        "dtype": "float32"
    },

    "SIMPLE_PHOTO_MAG_Z": {
        "unit": u.mag,
        "description": "Simplest photometry method based magnitude in z-band (with isolate mask); MW extinction corrected",
        "blank_value": np.nan,
        "dtype": "float32"
    },


    # --- Aperture properties ---
    "APERFRAC_R4_IN_IMG_ISOLATE": {
        "unit": None,
        "description": "Fraction of R4 aperture inside image (with isolate mask)",
        "blank_value": np.nan,
        "dtype": "float32"
    },
    "APERFRAC_R4_IN_IMG_NO_ISOLATE": {
        "unit": None,
        "description": "Fraction of R4 aperture inside image (without isolate mask)",
        "blank_value": np.nan,
        "dtype": "float32"
        
    },



    # --- Arrays / structured data ---
    "COG_PARAMS_G_ISOLATE": {
        "unit": None,
        "description": "COG fit parameters for g-band (with isolate mask)",
        "shape": (Nparams,),  # replace with actual number of params
        "blank_value": np.nan,
        "dtype": "float32"
    },

    "COG_PARAMS_R_ISOLATE": {
        "unit": None,
        "description": "COG fit parameters for r-band (with isolate mask)",
        "shape": (Nparams,),  # replace with actual number of params
        "blank_value": np.nan,
        "dtype": "float32"
        
    },

    "COG_PARAMS_Z_ISOLATE": {
        "unit": None,
        "description": "COG fit parameters for z-band (with isolate mask)",
        "shape": (Nparams,),  # replace with actual number of params
        "blank_value": np.nan,
        "dtype": "float32"
        
    },

    "COG_PARAMS_G_NO_ISOLATE": {
        "unit": None,
        "description": "COG fit parameters for g-band (without isolate mask)",
        "shape": (Nparams,),  # replace with actual number of params
        "blank_value": np.nan,
        "dtype": "float32"
        
    },

    "COG_PARAMS_R_NO_ISOLATE": {
        "unit": None,
        "description": "COG fit parameters for r-band (without isolate mask)",
        "shape": (Nparams,),  # replace with actual number of params
        "blank_value": np.nan,
        "dtype": "float32"
        
    },

    "COG_PARAMS_Z_NO_ISOLATE": {
        "unit": None,
        "description": "COG fit parameters for z-band (without isolate mask)",
        "shape": (Nparams,),  # replace with actual number of params
        "blank_value": np.nan,
        "dtype": "float32"
        
    },

    "COG_PARAMS_G_ERR_ISOLATE": {
        "unit": None,
        "description": "COG fit parameter errors for g-band (with isolate mask)",
        "shape": (Nparams,),  # replace with actual number of params
        "blank_value": np.nan,
        "dtype": "float32"
        
    },

    "COG_PARAMS_R_ERR_ISOLATE": {
        "unit": None,
        "description": "COG fit parameter errors for r-band (with isolate mask)",
        "shape": (Nparams,),  # replace with actual number of params
        "blank_value": np.nan,
        "dtype": "float32"
        
    },

    "COG_PARAMS_Z_ERR_ISOLATE": {
        "unit": None,
        "description": "COG fit parameter errors for z-band (with isolate mask)",
        "shape": (Nparams,),  # replace with actual number of params
        "blank_value": np.nan,
        "dtype": "float32"
        
    },

    "COG_PARAMS_G_ERR_NO_ISOLATE": {
        "unit": None,
        "description": "COG fit parameter errors for g-band (without isolate mask)",
        "shape": (Nparams,),  # replace with actual number of params
        "blank_value": np.nan,
        "dtype": "float32"
        
    },

    "COG_PARAMS_R_ERR_NO_ISOLATE": {
        "unit": None,
        "description": "COG fit parameter errors for r-band (without isolate mask)",
        "shape": (Nparams,),  # replace with actual number of params
        "blank_value": np.nan,
        "dtype": "float32"
        
    },

    "COG_PARAMS_Z_ERR_NO_ISOLATE": {
        "unit": None,
        "description": "COG fit parameter errors for z-band (without isolate mask)",
        "shape": (Nparams,),  # replace with actual number of params
        "blank_value": np.nan,
        "dtype": "float32"
        
    },

    "COG_MAG_ERR_ISOLATE": {
        "unit": None,
        "description": "COG magnitude errors (with isolate mask)",
        "shape": (3,),  # replace with actual number of params
        "blank_value": np.nan,
        "dtype": "float32"
        
    },

    "COG_MAG_ERR_NO_ISOLATE": {
        "unit": None,
        "description": "COG magnitude errors (without isolate mask)",
        "shape": (3,),  # replace with actual number of params
        "blank_value": np.nan,
        "dtype": "float32"
        
    },


    # --- cog goodness metrics ---

    "COG_SEG_ON_BLOB": {
        "unit": None,
        "description": "Bool indicating whether on the smoothed main blob used in COG analysis",
        "blank_value": False,
        "dtype": "bool"
    },

    "COG_FIT_RESID_ISOLATE": {
        "unit": None,
        "description": "COG fit residuals for the g-band (with isolate mask)",
        "shape": (3,),  # replace with actual number of params
        "blank_value": np.nan,
        "dtype": "float32"
        
    },

    "COG_DECREASE_MAX_LEN_ISOLATE": {
        "unit": None,
        "description": "Maximum consecutive decrease in COG for each band (with isolate mask)",
        "shape": (3,),  # replace with actual number of params
        "blank_value": np.nan,
        "dtype": "float32"
        
    },

    "COG_DECREASE_MAX_MAG_ISOLATE": {
        "unit": None,
        "description": "Magnitude decrease in the maximum consecutive decrease part in COG for each band (with isolate mask)",
        "shape": (3,),  # replace with actual number of params
        "blank_value": np.nan,
        "dtype": "float32"
        
    },

    "COG_FIT_RESID_NO_ISOLATE": {
        "unit": None,
        "description": "COG fit residuals for the g-band (without isolate mask)",
        "shape": (3,),  # replace with actual number of params
        "blank_value": np.nan,
        "dtype": "float32"
        
    },

    "COG_DECREASE_MAX_LEN_NO_ISOLATE": {
        "unit": None,
        "description": "Maximum consecutive decrease in COG for each band (without isolate mask)",
        "shape": (3,),  # replace with actual number of params
        "blank_value": np.nan,
        "dtype": "float32"
        
    },

    "COG_DECREASE_MAX_MAG_NO_ISOLATE": {
        "unit": None,
        "description": "Magnitude decrease in the maximum consecutive decrease part in COG for each band (without isolate mask)",
        "shape": (3,),  # replace with actual number of params
        "blank_value": np.nan,
        "dtype": "float32"
        
    },

    # --- Aperture parameters ---

    "APER_CEN_RADEC_ISOLATE": {
        "unit": None,
        "description": "Aperture centroid RA, Dec (with isolate mask)",
        "shape": (2,),
        "blank_value": np.nan,
        "dtype": "float32"
        
    },

    "APER_CEN_RADEC_NO_ISOLATE": {
        "unit": None,
        "description": "Aperture centroid RA, Dec (without isolate mask)",
        "shape": (2,),
        "blank_value": np.nan,
        "dtype": "float32"
        
    },

    "APER_PARAMS_ISOLATE": {
        "unit": None,
        "description": "Aperture parameters: semi-major axis in pixels, b/a ratio, PA (with isolate mask)",
        "shape": (3,),
        "blank_value": np.nan,
        "dtype": "float32"
        
    },

    "APER_PARAMS_NO_ISOLATE": {
        "unit": None,
        "description": "Aperture parameters: semi-major axis in pixels, b/a ratio, PA (without isolate mask)",
        "shape": (3,),
        "blank_value": np.nan,
        "dtype": "float32"
        
    },


    # --- blob information ---

    "APER_SOURCE_ON_ORG_BLOB": {
        "unit": None,
        "description": "Bool indicating whether DESI source location lies on the unsmoothed detection blob",
        "blank_value": False,
        "dtype": "bool"
        
    },
    
    # --- other imp info ----

    "NEAREST_STAR_NORM_DIST": {
        "unit": None,
        "description": "Distance to nearest star in units of stars masking radius",
        "blank_value": np.nan,
        "dtype": "float32"
    },

    "NEAREST_STAR_MAX_MAG": {
        "unit": None,
        "description": "Magnitude (brightest across BP,RP,G) of the nearest star",
        "blank_value": np.nan,
        "dtype": "float32"
    },

    "NUM_TRACTOR_SOURCES_NO_ISOLATE": {
        "unit": None,
        "description": "Number of Tractor sources part of parent galaxy (without isolate mask)",
        "blank_value": 0,
        "dtype": "int32"
    },

    "NUM_TRACTOR_SOURCES_ISOLATE": {
        "unit": None,
        "description": "Number of Tractor sources part of parent galaxy (with isolate mask)",
        "blank_value": 0,
        "dtype": "int32"
    },

    "APER_R2_MU_R_ELLIPSE_TRACTOR": {
        "unit": None,
        "description": "Surface brightness in r band within R2 ellipse",
        "blank_value": np.nan,
        "dtype": "float32"
    },

    "APER_R2_MU_R_BLOB_TRACTOR": {
        "unit": None,
        "description": "Surface brightness in r band within segmented blob ",
        "blank_value": np.nan,
        "dtype": "float32"
    },

    "APERFRAC_R4_IN_IMG_DATA_NO_ISOLATE": {
        "unit": None,
        "description": "Fraction of R4 aperture on initial parent galaxy reconstruction (g+r+z) inside image",
        "blank_value": np.nan,
        "dtype": "float32"
    },


}



fastspec_hdu_datamodel = {
    "TARGETID": {
        "unit": None,
        "description": "DESI TARGET ID",
        "dtype": "int64"
    },
    "RA_TARGET": {
        "unit": "deg",
        "description": "Right Ascension from target catalog",
        "blank_value": np.nan,
        "dtype": "float64"
    },
    "DEC_TARGET": {
        "unit": "deg",
        "description": "Declination from target catalog",
        "blank_value": np.nan,
        "dtype": "float64"
    },
    "DN4000": {
        "unit": None,
        "description": "Narrow 4000-Å break index (Balogh et al. 1999) measured from the emission-line subtracted spectrum.",
        "dtype": "float32"
    },
    "DN4000_OBS": {
        "unit": None,
        "description": "Narrow 4000-Å break index measured from the observed spectrum.",
        "dtype": "float32"
    },
    "DN4000_IVAR": {
        "unit": None,
        "description": "Inverse variance of DN4000 and DN4000_OBS.",
        "dtype": "float32"
    },
    "DN4000_MODEL": {
        "unit": None,
        "description": "Narrow 4000-Å break index measured from the best-fitting continuum model.",
        "dtype": "float32"
    },
    "SNR_B": {
        "unit": None,
        "description": "Median signal-to-noise ratio per pixel in the b camera.",
        "dtype": "float32"
    },
    "SNR_R": {
        "unit": None,
        "description": "Median signal-to-noise ratio per pixel in the r camera.",
        "dtype": "float32"
    },
    "SNR_Z": {
        "unit": None,
        "description": "Median signal-to-noise ratio per pixel in the z camera.",
        "dtype": "float32"
    },
    "APERCORR": {
        "unit": None,
        "description": "Median aperture correction factor.",
        "dtype": "float32"
    },
    "APERCORR_G": {
        "unit": None,
        "description": "Aperture correction factor measured in the g band.",
        "dtype": "float32"
    },
    "APERCORR_R": {
        "unit": None,
        "description": "Aperture correction factor measured in the r band.",
        "dtype": "float32"
    },
    "APERCORR_Z": {
        "unit": None,
        "description": "Aperture correction factor measured in the z band.",
        "dtype": "float32"
    },

}
    # "DN4000_MODEL_IVAR": {
    #     "unit": None,
    #     "description": "Inverse variance of DN4000_MODEL.",
    #     "dtype": "float32"
    # },
    # "VDISP": {
#         "unit": "km / s",
#         "description": "Stellar velocity dispersion.",
#         "dtype": "float32"
#     },
#     "VDISP_IVAR": {
#         "unit": "s2 / km2",
#         "description": "Inverse variance of VDISP.",
#         "dtype": "float32"
#     },
#     "FOII_3727_CONT": {
#         "unit": "1e-17 erg / (Angstrom cm2 s)",
#         "description": "Continuum flux at 3728.483 Å in the rest-frame.",
#         "dtype": "float32"
#     },
#     "FOII_3727_CONT_IVAR": {
#         "unit": "1e+34 cm4 Angstrom2 s2 / erg2",
#         "description": "Inverse variance in FOII_3727_CONT.",
#         "dtype": "float32"
#     },
#     "FHBETA_CONT": {
#         "unit": "1e-17 erg / (Angstrom cm2 s)",
#         "description": "Continuum flux at 4862.683 in the rest-frame.",
#         "dtype": "float32"
#     },
#     "FHBETA_CONT_IVAR": {
#         "unit": "1e+34 cm4 Angstrom2 s2 / erg2",
#         "description": "Inverse variance in FHBETA_CONT.",
#         "dtype": "float32"
#     },
#     "FOIII_5007_CONT": {
#         "unit": "1e-17 erg / (Angstrom cm2 s)",
#         "description": "Continuum flux at 5008.239 Å in the rest-frame.",
#         "dtype": "float32"
#     },
#     "FOIII_5007_CONT_IVAR": {
#         "unit": "1e+34 cm4 Angstrom2 s2 / erg2",
#         "description": "Inverse variance in FOIII_5007_CONT.",
#         "dtype": "float32"
#     },
#     "FHALPHA_CONT": {
#         "unit": "1e-17 erg / (Angstrom cm2 s)",
#         "description": "Continuum flux at 6564.613 Å in the rest-frame.",
#         "dtype": "float32"
#     },
#     "FHALPHA_CONT_IVAR": {
#         "unit": "1e+34 cm4 Angstrom2 s2 / erg2",
#         "description": "Inverse variance in FHALPHA_CONT.",
#         "dtype": "float32"
#     }
# }

# Emission-line definitions
_emission_lines = [
    "OII_3726", "OII_3729", "OIII_4363", "HEII_4686", "HBETA",
    "OIII_4959", "OIII_5007", "HEI_5876", "NII_6548", "HALPHA",
    "HALPHA_BROAD", "NII_6584", "SII_6716", "SII_6731",
    "SIII_9069", "SIII_9532"
]

# Add flux-related fields for all emission lines
for line in _emission_lines:
    fastspec_hdu_datamodel[f"{line}_FLUX"] = {
        "unit": "1e-17 erg / (cm2 s)",
        "description": f"Gaussian-integrated emission-line flux for {line}.",
        "dtype": "float32"
    }
    fastspec_hdu_datamodel[f"{line}_FLUX_IVAR"] = {
        "unit": "1e+34 cm4 s2 / erg2",
        "description": f"Inverse variance in {line}_FLUX.",
        "dtype": "float32"
    }


# Add additional HALPHA-only measurements
fastspec_hdu_datamodel["HALPHA_BOXFLUX"] = {
    "unit": "1e-17 erg / (cm2 s)",
    "description": "Boxcar-integrated Halpha emission-line flux.",
    "dtype": "float32"
}
fastspec_hdu_datamodel["HALPHA_BOXFLUX_IVAR"] = {
    "unit": "1e+34 cm4 s2 / erg2",
    "description": "Inverse variance in HALPHA_BOXFLUX.",
    "dtype": "float32"
}
fastspec_hdu_datamodel["HALPHA_EW"] = {
    "unit": "Angstrom",
    "description": "Rest-frame equivalent width of Halpha emission line.",
    "dtype": "float32"
}
fastspec_hdu_datamodel["HALPHA_EW_IVAR"] = {
    "unit": "1 / Angstrom2",
    "description": "Inverse variance in HALPHA_EW.",
    "dtype": "float32"
}

# fastspec_hdu_datamodel["HALPHA_SIGMA"] = {
#     "unit": "km / s",
#     "description": "Gaussian emission-line width of Halpha before convolution with the resolution matrix.",
#     "dtype": "float32"
# }
# fastspec_hdu_datamodel["HALPHA_SIGMA_IVAR"] = {
#     "unit": "s2 / km2",
#     "description": "Inverse variance in HALPHA_SIGMA.",
#     "dtype": "float32"
# }




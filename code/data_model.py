from astropy.table import Table, Column
import numpy as np

# Example datamodel template

#different distance estimates, stellar mass estimates, final best photometry, quality_maskbits
#incluce information on closest angular distance to SGA galaxy
import numpy as np
import astropy.units as u

logM_sun = u.def_unit('log(solMass)', format={'latex': r'\log(M_\odot)'})

main_datamodel = {
    "TARGETID": {
        "unit": None,
        "description": "DESI TARGET ID",
        "blank_value": np.nan,
        "dtype": "int64"
    },
    "SURVEY": {
        "unit": None,
        "description": "Survey name",
        "blank_value": "",
        "dtype": "str"
    },
    "PROGRAM": {
        "unit": None,
        "description": "Program name",
        "blank_value": "",
        "dtype": "str"
    },
    "Z": {
        "unit": None,
        "description": "Redrock Redshift",
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
        "blank_value": np.nan,
        "dtype": "int8"
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
        "blank_value": "",
        "dtype": "str"
    },
    "DIST_MPC_FIDU": {
        "unit": "Mpc",
        "description": "Fiducial luminosity distance in Mpc",
        "blank_value": np.nan,
        "dtype": "float32"
    },
    "LOGM_SAGA_FIDU": {
        "unit": logM_sun,
        "description": "Log stellar mass using the fiducial luminosity distance and SAGA gr-based approximation",
        "blank_value": np.nan,
        "dtype": "float32"
    },
    "LOGM_M24_VCMB": {
        "unit": logM_sun,
        "description": "Log stellar mass using the fiducial luminosity distance and de los Reyes et al. 2024 gr-based approximation",
        "blank_value": np.nan,
        "dtype": "float32"
    },
    "MAG_G": {
        "unit": u.mag,
        "description": "g-band magnitude (MW extinction corrected)",
        "blank_value": np.nan,
        "dtype": "float32"
    },
    "MAG_R": {
        "unit": u.mag,
        "description": "r-band magnitude (MW extinction corrected)",
        "blank_value": np.nan,
        "dtype": "float32"
    },
    "MAG_Z": {
        "unit": u.mag,
        "description": "z-band magnitude (MW extinction corrected)",
        "blank_value": np.nan,
        "dtype": "float32"
    },
    "SAMPLE": {
        "unit": None,
        "description": "DESI target class (e.g., BGS_BRIGHT, BGS_FAINT) ",
        "blank_value": "",
        "dtype": "str"
    }
}


zcat_datamodel = {
     "TARGETID": {
        "unit": None,
        "description": "DESI TARGET ID",
        "blank_value": np.nan,
        "dtype": "int64"
    },
    "COADD_FIBERSTATUS": {
        "unit": None,
        "description": "Fiber status bit",
        "blank_value": np.nan,
        "dtype": "int64"
    },
    "CMX_TARGET": {
        "unit": None,
        "description": "Commissioning (CMX) targeting bit",
        "blank_value": np.nan,
        "dtype": "int64"
    },
    "DESI_TARGET": {
        "unit": None,
        "description": "DESI targeting bit",
        "blank_value": np.nan,
        "dtype": "int64"
    },
    "BGS_TARGET": {
        "unit": None,
        "description": "BGS targeting bit",
        "blank_value": np.nan,
        "dtype": "int64"
    },
    "MWS_TARGET": {
        "unit": None,
        "description": "MWS targeting bit",
        "blank_value": np.nan,
        "dtype": "int64"
    },
    "SCND_TARGET": {
        "unit": None,
        "description": "Secondary target targeting bit",
        "blank_value": np.nan,
        "dtype": "int64"
    },
    "SV1_DESI_TARGET": {
        "unit": None,
        "description": "SV1 DESI targeting bit",
        "blank_value": np.nan,
        "dtype": "int64"
    },
    "SV1_BGS_TARGET": {
        "unit": None,
        "description": "SV1 BGS targeting bit",
        "blank_value": np.nan,
        "dtype": "int64"
    },
    "SV1_MWS_TARGET": {
        "unit": None,
        "description": "SV1 MWS targeting bit",
        "blank_value": np.nan,
        "dtype": "int64"
    },
    "SV2_DESI_TARGET": {
        "unit": None,
        "description": "SV2 DESI targeting bit",
        "blank_value": np.nan,
        "dtype": "int64"
    },
    "SV2_BGS_TARGET": {
        "unit": None,
        "description": "SV2 BGS targeting bit",
        "blank_value": np.nan,
        "dtype": "int64"
    },
    "SV2_MWS_TARGET": {
        "unit": None,
        "description": "SV2 MWS targeting bit",
        "blank_value": np.nan,
        "dtype": "int64"
    },
    "SV3_DESI_TARGET": {
        "unit": None,
        "description": "SV3 DESI targeting bit",
        "blank_value": np.nan,
        "dtype": "int64"
    },
    "SV3_BGS_TARGET": {
        "unit": None,
        "description": "SV3 BGS targeting bit",
        "blank_value": np.nan,
        "dtype": "int64"
    },
    "SV3_MWS_TARGET": {
        "unit": None,
        "description": "SV3 MWS targeting bit",
        "blank_value": np.nan,
        "dtype": "int64"
    },
    "SV1_SCND_TARGET": {
        "unit": None,
        "description": "SV1 secondary targeting bit",
        "blank_value": np.nan,
        "dtype": "int64"
    },
    "SV2_SCND_TARGET": {
        "unit": None,
        "description": "SV2 secondary targeting bit",
        "blank_value": np.nan,
        "dtype": "int64"
    },
    "SV3_SCND_TARGET": {
        "unit": None,
        "description": "SV3 secondary targeting bit",
        "blank_value": np.nan,
        "dtype": "int64"
    },
    "TSNR2_LRG": {
        "unit": None,
        "description": "TSNR2 for LRG targets (like TSNR2_BGS)",
        "blank_value": np.nan,
        "dtype": "float32"
    },

    NEED TO ADD MORE COLUMNS HERE!!
}


}


tractor_datamodel = {

#also include the most relevant tractor columns!!

}

Nparams = 5

#these are only the columns we hope to retain in the final catalog
photo_datamodel = {
    # --- COG magnitudes ---
    "COG_MAG_G_ISOLATE": {
        "unit": "mag",
        "description": "COG magnitude in g-band (with isolate mask); MW extinction corrected",
        "blank_value": np.nan,
        "dtype": "float32"
    },
    
    "COG_MAG_R_ISOLATE": {
        "unit": "mag",
        "description": "COG magnitude in r-band (with isolate mask); MW extinction corrected",
        "blank_value": np.nan,
        "dtype": "float32"
    },
    "COG_MAG_Z_ISOLATE": {
        "unit": "mag",
        "description": "COG magnitude in z-band (with isolate mask); MW extinction corrected",
        "blank_value": np.nan,
        "dtype": "float32"
    },

    ##

    "COG_MAG_G_NO_ISOLATE": {
        "unit": "mag",
        "description": "COG magnitude in g-band (without isolate mask); MW extinction corrected",
        "blank_value": np.nan,
        "dtype": "float32"
    },

    "COG_MAG_R_NO_ISOLATE": {
        "unit": "mag",
        "description": "COG magnitude in r-band (without isolate mask); MW extinction corrected",
        "blank_value": np.nan,
        "dtype": "float32"
    },

    "COG_MAG_Z_NO_ISOLATE": {
        "unit": "mag",
        "description": "COG magnitude in z-band (without isolate mask); MW extinction corrected",
        "blank_value": np.nan,
        "dtype": "float32"
    },


    # --- Aper R4 ---
    "APER_R4_MAG_G_ISOLATE": {
        "unit": "mag",
        "description": "R4 aperture magnitude in g-band (with isolate mask); MW extinction corrected",
        "blank_value": np.nan,
        "dtype": "float32"
    },

    "APER_R4_MAG_R_ISOLATE": {
        "unit": "mag",
        "description": "R4 aperture magnitude in r-band (with isolate mask); MW extinction corrected",
        "blank_value": np.nan,
        "dtype": "float32"
    },

    "APER_R4_MAG_Z_ISOLATE": {
        "unit": "mag",
        "description": "R4 aperture magnitude in z-band (with isolate mask); MW extinction corrected",
        "blank_value": np.nan,
        "dtype": "float32"
    },

    ##

    "APER_R4_MAG_G_NO_ISOLATE": {
        "unit": "mag",
        "description": "R4 aperture magnitude in g-band (without isolate mask); MW extinction corrected",
        "blank_value": np.nan,
        "dtype": "float32"
    },

    "APER_R4_MAG_R_NO_ISOLATE": {
        "unit": "mag",
        "description": "R4 aperture magnitude in r-band (without isolate mask); MW extinction corrected",
        "blank_value": np.nan,
        "dtype": "float32"
    },

    "APER_R4_MAG_Z_NO_ISOLATE": {
        "unit": "mag",
        "description": "R4 aperture magnitude in z-band (without isolate mask); MW extinction corrected",
        "blank_value": np.nan,
        "dtype": "float32"
    },
    
    
    # --- Tractor ---
    "TRACTOR_PARENT_MAG_G_ISOLATE": {
        "unit": "mag",
        "description": "Tractor based parent magnitude in g-band (with isolate mask); MW extinction corrected",
        "blank_value": np.nan,
        "dtype": "float32"
    },

    "TRACTOR_PARENT_MAG_R_ISOLATE": {
        "unit": "mag",
        "description": "Tractor based parent magnitude in r-band (with isolate mask); MW extinction corrected",
        "blank_value": np.nan,
        "dtype": "float32"
    },

    "TRACTOR_PARENT_MAG_Z_ISOLATE": {
        "unit": "mag",
        "description": "Tractor based parent magnitude in z-band (with isolate mask); MW extinction corrected",
        "blank_value": np.nan,
        "dtype": "float32"
    },

    ##
    
    "TRACTOR_PARENT_MAG_G_NO_ISOLATE": {
        "unit": "mag",
        "description": "Tractor based parent magnitude in g-band (without isolate mask); MW extinction corrected",
        "blank_value": np.nan,
        "dtype": "float32"
    },

    "TRACTOR_PARENT_MAG_R_NO_ISOLATE": {
        "unit": "mag",
        "description": "Tractor based parent magnitude in r-band (without isolate mask); MW extinction corrected",
        "blank_value": np.nan,
        "dtype": "float32"
    },

    "TRACTOR_PARENT_MAG_Z_NO_ISOLATE": {
        "unit": "mag",
        "description": "Tractor based parent magnitude in z-band (without isolate mask); MW extinction corrected",
        "blank_value": np.nan,
        "dtype": "float32"
    },

    # --- Simplest photometry ---
    
    "SIMPLE_PHOTO_MAG_G": {
        "unit": "mag",
        "description": "Simplest photometry method based magnitude in g-band (with isolate mask); MW extinction corrected",
        "blank_value": np.nan,
        "dtype": "float32"
    },

    "SIMPLE_PHOTO_MAG_R": {
        "unit": "mag",
        "description": "Simplest photometry method based magnitude in r-band (with isolate mask); MW extinction corrected",
        "blank_value": np.nan,
        "dtype": "float32"
    },

    "SIMPLE_PHOTO_MAG_Z": {
        "unit": "mag",
        "description": "Simplest photometry method based magnitude in z-band (with isolate mask); MW extinction corrected",
        "blank_value": np.nan,
        "dtype": "float32"
    },


    # --- Aperture properties ---
    "APER_R4_FRAC_IN_IMG_ISOLATE": {
        "unit": None,
        "description": "Fraction of R4 aperture inside image (with isolate mask)",
        "blank_value": np.nan,
        "dtype": "float32"
    },
    "APER_R4_FRAC_IN_IMG_NO_ISOLATE": {
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

    "COG_COG_DECREASE_MAX_LEN_ISOLATE": {
        "unit": None,
        "description": "Maximum consecutive decrease in COG for each band (with isolate mask)",
        "shape": (3,),  # replace with actual number of params
        "blank_value": np.nan,
        "dtype": "float32"
        
    },

    "COG_COG_DECREASE_MAX_MAG_ISOLATE": {
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

    "COG_COG_DECREASE_MAX_LEN_NO_ISOLATE": {
        "unit": None,
        "description": "Maximum consecutive decrease in COG for each band (without isolate mask)",
        "shape": (3,),  # replace with actual number of params
        "blank_value": np.nan,
        "dtype": "float32"
        
    },

    "COG_COG_DECREASE_MAX_MAG_NO_ISOLATE": {
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
        "blank_value": np.nan,
        "dtype": "int32"
    },

    "NUM_TRACTOR_SOURCES_ISOLATE": {
        "unit": None,
        "description": "Number of Tractor sources part of parent galaxy (with isolate mask)",
        "blank_value": np.nan,
        "dtype": "int32"
    },

    "APER_R2_MU_R_TRACTOR": {
        "unit": None,
        "description": "Surface brightness of ",
        "blank_value": np.nan,
        "dtype": "float32"
    },

    "APER_R4_DATA_FRAC_IN_IMAGE_NO_ISOLATE": {
        "unit": None,
        "description": "Fraction of R4 aperture on initial parent galaxy reconstruction (g+r+z) inside image",
        "blank_value": np.nan,
        "dtype": "float32"
    },


}


spectra_datamodel = {}


imaging_datamodel = {}




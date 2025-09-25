from astropy.table import Table, Column
import numpy as np

# Example datamodel template

main_datamodel = {

#different distance estimates, stellar mass estimates, final best photometry
#incluce information on closest angular distance to SGA galaxy

}

zcat_datamodel = {


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
        "blank_value": False
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


spectra_datamodel = {

}


imaging_datamodel = {


}

































    
}

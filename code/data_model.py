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
        "dtype": "int32"
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
    "DIST_SOURCE": {
        "unit": None,
        "description": "Source of the luminosity distance. One of: NED_ZIND, VIRGO_SBF, VIRGO_EVCC, CF3_NAM, V_CMB.",
        "dtype": "str"
    },
    "LOG_MSTAR_M24": {
        "unit": logM_sun,
        "description": "Log stellar mass (in Msol) using LUMI_DIST_MPC and de los Reyes et al. 2024 gr-based relation, computed from nebular+filter+k-corrected SDSS z=0 continuum-only photometry",
        "blank_value": np.nan,
        "dtype": "float32"
    },
    "LOG_MSTAR_M24_ERR": {
        "unit": u.Unit("dex"),
        "description": "Uncertainty in LOG_MSTAR_M24 propagated from nebular correction errors in emission-subtracted fiber photometry",
        "blank_value": np.nan,
        "dtype": "float64"
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
    "IS_BGS_BRIGHT": {
        "unit": None,
        "description": (
            "Boolean: target belongs to the BGS_BRIGHT target list. Derived from ZCAT "
            "targeting bits (BGS_TARGET main survey OR SV{1,2,3}_BGS_TARGET), evaluated "
            "for every row. Non-exclusive: a target may belong to multiple lists."
        ),
        "dtype": "bool"
    },
    "IS_BGS_FAINT": {
        "unit": None,
        "description": (
            "Boolean: target belongs to the BGS_FAINT target list. Derived from ZCAT "
            "targeting bits (BGS_TARGET main survey OR SV{1,2,3}_BGS_TARGET), evaluated "
            "for every row. Non-exclusive."
        ),
        "dtype": "bool"
    },
    "IS_LOWZ": {
        "unit": None,
        "description": (
            "Boolean: target belongs to the LOWZ secondary target list. Derived from "
            "ZCAT SCND_TARGET (main survey) OR SV{1,2,3}_SCND_TARGET bits 15/16/17, "
            "evaluated for every row. Non-exclusive."
        ),
        "dtype": "bool"
    },
    "IS_ELG": {
        "unit": None,
        "description": (
            "Boolean: target belongs to the ELG target list. Derived from ZCAT "
            "targeting bits (DESI_TARGET main survey OR SV{1,2,3}_DESI_TARGET), "
            "evaluated for every row. Non-exclusive."
        ),
        "dtype": "bool"
    },
    "IS_OTHER": {
        "unit": None,
        "description": (
            "Boolean: object entered the catalog via the QSO/SCND hidden-dwarf "
            "candidate branch (provenance flag, formerly SAMPLE=='OTHER'); mirrors "
            "IN_SGA_2020. Not a targeting-bit membership flag."
        ),
        "dtype": "bool"
    },
    "DWARF_MASKBIT": {
        "unit": None,
        "description": (
            "Bitwise mask to apply various cleaning cuts. Bit 19 = junk spectrum "
            "(SNR_B/R/Z < 0 in at least two of the three arms). Bit 20 = suspect "
            "spectrum (200 A median-smoothed coadd flux < -5 in any ivar>0 pixel; "
            "negative-continuum / normalization failure). Low continuum SNR in "
            "emission-subtracted fiber photometry (mass pipeline) is recorded in "
            "MSTAR_MASKBIT, not here. See bitmask descriptions in README."
        ),
        "dtype": "int32"
    },
    "MSTAR_MASKBIT": {
        "unit": None,
        "description": (
            "Bitwise mask for the LOG_MSTAR_M24 derivation: bit 0 = low continuum SNR "
            "(nebular-subtracted g/r fiber photometry, threshold SNR 10); "
            "bit 1 = M_g,0 < -18.5 after corrections; "
            "bit 2 = unreliable nebular correction (|DELTA_MAG_G_NEB| or "
            "|DELTA_MAG_R_NEB| > 2). Bits 0 and 2 both route the mass onto the "
            "fallback path (aggregate MAG_G/MAG_R + Z_CMB + g_kcorr, no delta-mag chain). "
            "Copied with LOG_MSTAR_M24 from PROPERTY_SOURCE_TARGETID during associated-fiber consolidation."
        ),
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
        "description": "Galaxy shape [b/a, PA]: axis ratio and major-axis position angle in degrees East of North, range [0,180). Folded to one convention across photometry types; raw Tractor PHI in [-90,90] is the same orientation mod 180.",
        "shape": (2,),
        "blank_value": np.nan,
        "dtype": "float32"
    },
    "IN_SGA_2020": {
        "unit": None,
        "description": "Boolean indicating whether targeted source had Tractor MASKBITS=12, that is, in SGA-2020 catalog",
        "dtype": "bool"
    },
    "ASSOCIATED_TARGETIDS": {
        "unit": None,
        "description": (
            "List of associated TARGETIDs. Variable-length per row."
        ),
        "dtype": "object"
    },
    
    "DWARF_PRIMARY_TARGETID": {
        "unit": None,
        "description": (
            "TARGETID selected as the primary fiber for this dwarf galaxy, chosen "
            "as the brightest MAG_R_TARGET among associated fibers."
        ),
        "dtype": "int64"
    },
    
    "DWARF_PRIMARY": {
        "unit": None,
        "description": (
            "Boolean flag indicating whether this row corresponds to the primary "
            "fiber (TARGETID == DWARF_PRIMARY_TARGETID) for the dwarf galaxy."
        ),
        "dtype": "bool"
    },

    "PROPERTY_SOURCE_TARGETID": {
        "unit": None,
        "description": (
            "TARGETID of the associated fiber whose galaxy-level properties "
            "(MAG_G/R/Z, RA, DEC, R50_R, LOG_MSTAR_M24, LOG_MSTAR_M24_ERR, "
            "MSTAR_MASKBIT, etc.) are adopted for this row. "
            "Chosen as the brightest MAG_R among the associated group."
        ),
        "dtype": "int64"
    },
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

    "FIBERTOTFLUX_G": {
        "unit": None,
        "description": (
            "Predicted g-band flux within a fiber of diameter 1.5 arcsec from all sources "
            "at this location in 1 arcsec Gaussian seeing. Not corrected for Galactic extinction"
        ),
        "blank_value": np.nan,
        "dtype": "float32",
    },

    "FIBERTOTFLUX_R": {
        "unit": None,
        "description": (
            "Predicted r-band flux within a fiber of diameter 1.5 arcsec from all sources "
            "at this location in 1 arcsec Gaussian seeing. Not corrected for Galactic extinction"
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
        "description": "Raw Tractor major-axis position angle 0.5*arctan2(SHAPE_E2,SHAPE_E1) in deg, range [-90,90] (DR10 convention). Same orientation as SHAPE_PARAMS PA but differs by 180 deg for PA in [90,180); use SHAPE_PARAMS[:,1] for science PA in [0,180).",
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
        "description": "Aperture parameters: semi-major axis in pixels, b/a ratio, PA in radians (raw photutils pixel-frame orientation; converted to East-of-North deg in SHAPE_PARAMS) (with isolate mask)",
        "shape": (3,),
        "blank_value": np.nan,
        "dtype": "float32"
        
    },

    "APER_PARAMS_NO_ISOLATE": {
        "unit": None,
        "description": "Aperture parameters: semi-major axis in pixels, b/a ratio, PA in radians (raw photutils pixel-frame orientation; converted to East-of-North deg in SHAPE_PARAMS) (without isolate mask)",
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
    "ABSMAG01_SDSS_G": {
        "unit": u.mag,
        "description": "Absolute magnitude in SDSS g-band band-shifted to z=0.1 assuming h=1.0.",
        "dtype": "float32"
    },
    "ABSMAG01_SDSS_R": {
        "unit": u.mag,
        "description": "Absolute magnitude in SDSS r-band band-shifted to z=0.1 assuming h=1.0.",
        "dtype": "float32"
    },
    "ABSMAG01_SDSS_I": {
        "unit": u.mag,
        "description": "Absolute magnitude in SDSS i-band band-shifted to z=0.1 assuming h=1.0.",
        "dtype": "float32"
    },
    "ABSMAG01_SDSS_Z": {
        "unit": u.mag,
        "description": "Absolute magnitude in SDSS z-band band-shifted to z=0.1 assuming h=1.0.",
        "dtype": "float32"
    },
    "ABSMAG01_IVAR_SDSS_G": {
        "unit": 1/u.mag**2,
        "description": "Inverse variance of absolute magnitude in SDSS g-band band-shifted to z=0.1 assuming h=1.0.",
        "dtype": "float32"
    },
    "ABSMAG01_IVAR_SDSS_R": {
        "unit": 1/u.mag**2,
        "description": "Inverse variance of absolute magnitude in SDSS r-band band-shifted to z=0.1 assuming h=1.0.",
        "dtype": "float32"
    },
    "ABSMAG01_IVAR_SDSS_I": {
        "unit": 1/u.mag**2,
        "description": "Inverse variance of absolute magnitude in SDSS i-band band-shifted to z=0.1 assuming h=1.0.",
        "dtype": "float32"
    },
    "ABSMAG01_IVAR_SDSS_Z": {
        "unit": 1/u.mag**2,
        "description": "Inverse variance of absolute magnitude in SDSS z-band band-shifted to z=0.1 assuming h=1.0.",
        "dtype": "float32"
    },
    "LOG_MSTAR": {
     "unit": logM_sun,
     "description": "Stellar mass from fastspecfit assuming h=1.0.",
    "dtype": "float32"}
    
}
                    
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
fastspec_hdu_datamodel["LOGMSTAR_FASTSPEC"] = {
    "unit": None,
    "description": (
        "log10(stellar mass / Msun) from FastSpecFit SED fit (LOGMSTAR column "
        "in the FastSpecFit SPECPHOT HDU); renamed from LOGMSTAR for clarity."
    ),
    "dtype": "float32",
}

fastspec_hdu_datamodel["FLUX_SYNTH_G"] = {
    "unit": "nanomaggy",
    "description": "g-band flux (in the PHOTSYS photometric system) synthesized from the observed spectrum.",
    "dtype": "float32"
}
fastspec_hdu_datamodel["FLUX_SYNTH_R"] = {
    "unit": "nanomaggy",
    "description": "Like FLUX_SYNTH_G but for the r-band.",
    "dtype": "float32"
}
fastspec_hdu_datamodel["FLUX_SYNTH_Z"] = {
    "unit": "nanomaggy",
    "description": "Like FLUX_SYNTH_G but for the z-band.",
    "dtype": "float32"
}
fastspec_hdu_datamodel["FLUX_SYNTH_SPECMODEL_G"] = {
    "unit": "nanomaggy",
    "description": "g-band flux (in the PHOTSYS photometric system) synthesized from the best-fitting spectroscopic model.",
    "dtype": "float32"
}
fastspec_hdu_datamodel["FLUX_SYNTH_SPECMODEL_R"] = {
    "unit": "nanomaggy",
    "description": "Like FLUX_SYNTH_SPECMODEL_G but in the r-band.",
    "dtype": "float32"
}
fastspec_hdu_datamodel["FLUX_SYNTH_SPECMODEL_Z"] = {
    "unit": "nanomaggy",
    "description": "Like FLUX_SYNTH_SPECMODEL_G but in the z-band.",
    "dtype": "float32"
}
fastspec_hdu_datamodel["FLUX_SYNTH_PHOTMODEL_G"] = {
    "unit": "nanomaggy",
    "description": "g-band flux (in the PHOTSYS photometric system) synthesized from the best-fitting photometric continuum model.",
    "dtype": "float32"
}
fastspec_hdu_datamodel["FLUX_SYNTH_PHOTMODEL_R"] = {
    "unit": "nanomaggy",
    "description": "Like FLUX_SYNTH_PHOTMODEL_G but in the r-band.",
    "dtype": "float32"
}
fastspec_hdu_datamodel["FLUX_SYNTH_PHOTMODEL_Z"] = {
    "unit": "nanomaggy",
    "description": "Like FLUX_SYNTH_PHOTMODEL_G but in the z-band.",
    "dtype": "float32"
}

# MAG_*_MODEL_* entries moved to spec_derived_hdu_datamodel (see below).
# They are now written by code/add_nebular_props.py to the SPEC_DERIVED HDU
# via add_model_photometry_to_spec_derived rather than to the FASTSPEC HDU.

# MAG_*_FIBER_NOEMI[_ERR] moved to spec_derived_hdu_datamodel (see above).
# They are still computed by compute_emission_subtracted_photo_errors and written
# into the FASTSPEC HDU during catalog build, but code/add_nebular_props.py then
# relocates them into the SPEC_DERIVED HDU and strips them from FASTSPEC.

# DELTA_MAG_{G,R}_* entries moved to spec_derived_hdu_datamodel (see below).
# They are now written by code/add_nebular_props.py to the SPEC_DERIVED HDU
# rather than to the FASTSPEC HDU.





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


# ---------------------------------------------------------------------------
# SPEC_DERIVED HDU data model
#
# Spectroscopically derived nebular properties appended by
# code/add_nebular_props.py. Row order / TARGETID order matches MAIN and
# FASTSPEC. New nebular properties (e.g. AV from Balmer decrements,
# direct-method metallicity) should be added to this datamodel as they
# are introduced.
# ---------------------------------------------------------------------------

spec_derived_hdu_datamodel = {
    "TARGETID": {
        "unit": None,
        "description": "DESI TARGET ID (matches MAIN.TARGETID and FASTSPEC.TARGETID row-by-row).",
        "dtype": "int64",
    },
    "LOG_SFR_HALPHA": {
        "unit": None,
        "description": (
            "log10(SFR / (Msun/yr)) from Bauer et al. (2013) Eq. 2 L(Halpha) combined with a per-object, "
            "metallicity-dependent Halpha->SFR calibration (sfr_log_cz_BPASS; BPASS, Korhonen Cuestas 2025). "
            "The calibration metallicity is set from MAIN LOG_MSTAR_M24 via stellar_mass_msz "
            "(Kirby+13 mass-metallicity relation with +0.2 dex young-population offset, Zahid et al. 2017), "
            "replacing the previous fixed Chabrier constant. Global aperture-corrected "
            "SFR using MAIN MAG_R and LUMI_DIST_MPC (distance modulus), per-row fiber HALPHA_EW from FASTSPEC. "
            "Internal dust correction uses the mass-based average Balmer decrement model_hahb(LOG_MSTAR_M24) "
            "dereddened against the intrinsic Halpha/Hbeta = 2.79 (Bauer+13 dust exponent 2.36), replacing the "
            "former per-object HALPHA/HBETA. Set where HALPHA_FLUX > 1, HALPHA_EW > 0, HALPHA_EW and HALPHA_FLUX "
            "SNR > 3, and LOG_MSTAR_M24 is finite (Hbeta is no longer required); NaN otherwise."
        ),
        "dtype": "float64",
    },
    "LOG_SFR_HALPHA_ERR": {
        "unit": None,
        "description": (
            "1-sigma uncertainty on LOG_SFR_HALPHA in dex, from first-order error propagation in calc_SFR_Halpha. "
            "The metallicity-dependent calibration constant is treated as exact, so with zero assumed errors on "
            "redshift, absolute r magnitude, and BD this reflects Halpha EW uncertainty only (HALPHA_EW_IVAR)."
        ),
        "dtype": "float64",
    },
    "LOG_MSTAR_24_FIBER": {
        "unit": None,
        "description": (
            "log10(Mstar / Msun) in the fiber aperture only"
        ),
        "dtype": "float32",
    },
    "LOG_HALPHA_SFR_FIBER": {
        "unit": None,
        "description": (
            "log10(SFR / (Msun/yr)) in the fiber aperture only. Same metallicity-dependent BPASS Halpha->SFR "
            "calibration as LOG_SFR_HALPHA (metallicity from MAIN LOG_MSTAR_M24 via stellar_mass_msz: "
            "Kirby+13 with +0.2 dex Zahid et al. 2017 offset). Internal dust correction uses the mass-based "
            "model_hahb(LOG_MSTAR_M24) (GLOBAL stellar mass) dereddened against intrinsic 2.79, as in LOG_SFR_HALPHA."
        ),
        "dtype": "float64",
    },
    "Z_GAS_R23_N2": {
        "unit": None,
        "description": (
            "Gas-phase strong-line metallicity from R23+N2 (see Scholte et al. 2024) "
            "using FASTSPEC Gaussian line fluxes; per-object internal dust correction (CCM89) when "
            "Halpha/Hbeta > 2.79 (intrinsic BALMER_INTRINSIC). "
            "Filled where all seven lines pass line_snr (SNR > 3, flux > 1 in FastSpec units); NaN otherwise."
        ),
        "dtype": "float64",
    },

    # ---------------------------------------------------------------
    # Emission-subtracted fiber photometry (MAG_*_FIBER_NOEMI)
    #
    # Computed by mass_and_photo_corrections.compute_emission_subtracted_photo_errors
    # (DECam g/r measured on the fiber spectrum after subtracting the FastSpecFit
    # emission model). They are written into the FASTSPEC HDU during catalog build,
    # then RELOCATED here by code/add_nebular_props.py: build_spec_derived_hdu copies
    # them into SPEC_DERIVED and strips them from FASTSPEC so FASTSPEC stays a clean
    # FastSpecFit passthrough. The fiber-mass / Halpha-SFR pipeline consumes the
    # _ERR columns to set the continuum-SNR (>= 10) split.
    # ---------------------------------------------------------------
    "MAG_G_FIBER_NOEMI": {
        "unit": "mag",
        "description": "DECam g-band AB magnitude measured from the emission-subtracted DESI fiber spectrum.",
        "dtype": "float64",
        "blank_value": np.nan,
    },
    "MAG_R_FIBER_NOEMI": {
        "unit": "mag",
        "description": "DECam r-band AB magnitude measured from the emission-subtracted DESI fiber spectrum.",
        "dtype": "float64",
        "blank_value": np.nan,
    },
    "MAG_G_FIBER_NOEMI_ERR": {
        "unit": "mag",
        "description": "Uncertainty in MAG_G_FIBER_NOEMI.",
        "dtype": "float64",
        "blank_value": np.nan,
    },
    "MAG_R_FIBER_NOEMI_ERR": {
        "unit": "mag",
        "description": "Uncertainty in MAG_R_FIBER_NOEMI.",
        "dtype": "float64",
        "blank_value": np.nan,
    },

    # ---------------------------------------------------------------
    # DELTA_MAG_* photometric correction columns
    #
    # Previously written to the FASTSPEC HDU by
    # mass_and_photo_corrections.add_delta_mag_to_fastspec; now written to
    # SPEC_DERIVED by code/add_nebular_props.py (matched by TARGETID to the
    # NEBCORR INT_V2 tables). BASS2DECAM columns are zeroed for south
    # (is_south == 1); all other deltas copied verbatim. Unmatched
    # TARGETIDs leave NaN.
    # ---------------------------------------------------------------
    "DELTA_MAG_G_BASS2DECAM": {
        "unit": "mag",
        "description": "Delta magnitude (add to Tractor g) for BASS-to-DECam conversion from fastspecfit models (north only; south typically ~0).",
        "dtype": "float64",
        "blank_value": np.nan,
    },
    "DELTA_MAG_R_BASS2DECAM": {
        "unit": "mag",
        "description": "Same as DELTA_MAG_G_BASS2DECAM for r-band.",
        "dtype": "float64",
        "blank_value": np.nan,
    },
    "DELTA_MAG_G_NEB": {
        "unit": "mag",
        "description": "Delta magnitude (add to working g mag) for nebular emission removal, from fastspecfit template difference.",
        "dtype": "float64",
        "blank_value": np.nan,
    },
    "DELTA_MAG_R_NEB": {
        "unit": "mag",
        "description": "Same as DELTA_MAG_G_NEB for r-band.",
        "dtype": "float64",
        "blank_value": np.nan,
    },
    "DELTA_MAG_G_DECAM2SDSS": {
        "unit": "mag",
        "description": "Delta magnitude (add to working g mag) for DECam-to-SDSS filter conversion from continuum-only model.",
        "dtype": "float64",
        "blank_value": np.nan,
    },
    "DELTA_MAG_R_DECAM2SDSS": {
        "unit": "mag",
        "description": "Same as DELTA_MAG_G_DECAM2SDSS for r-band.",
        "dtype": "float64",
        "blank_value": np.nan,
    },
    "DELTA_MAG_G_KCORR": {
        "unit": "mag",
        "description": "Delta magnitude (add to working g mag) for k-correction SDSS observed-z to z=0 from continuum-only model.",
        "dtype": "float64",
        "blank_value": np.nan,
    },
    "DELTA_MAG_R_KCORR": {
        "unit": "mag",
        "description": "Same as DELTA_MAG_G_KCORR for r-band.",
        "dtype": "float64",
        "blank_value": np.nan,
    },

    # ---------------------------------------------------------------
    # Direct-method nebular properties (TE_*)
    #
    # Produced by pn_functions.compute_direct_metallicities (UltraNest
    # nested sampling, matching Scholte+2026, Table 3). Populated only for
    # rows passing line_snr_mask([HALPHA, HBETA, HGAMMA, OIII_4363,
    # OIII_5007, OII_3726, OII_3729], snr_val=5, min_lines=7, flux > 1);
    # all other rows are NaN / False / 0.
    #
    # Each *_LO / *_HI / *_ERR sibling holds the 16th percentile, 84th
    # percentile and half the 84-16 spread of the posterior.
    # ---------------------------------------------------------------
    "TE_NE_OII": {
        "unit": "cm-3",
        "description": "Direct-method electron density inferred from the [O II] 3729/3726 doublet ratio (posterior median; NaN if te_mask fails or fit failed).",
        "dtype": "float64",
        "blank_value": np.nan,
    },
    "TE_NE_OII_LO": {
        "unit": "cm-3",
        "description": "16th percentile of the TE_NE_OII posterior.",
        "dtype": "float64",
        "blank_value": np.nan,
    },
    "TE_NE_OII_HI": {
        "unit": "cm-3",
        "description": "84th percentile of the TE_NE_OII posterior.",
        "dtype": "float64",
        "blank_value": np.nan,
    },
    "TE_NE_OII_ERR": {
        "unit": "cm-3",
        "description": "Half-width 0.5*(TE_NE_OII_HI - TE_NE_OII_LO) of the TE_NE_OII posterior.",
        "dtype": "float64",
        "blank_value": np.nan,
    },
    "TE_T_OIII": {
        "unit": "K",
        "description": "Direct-method [O III]-zone electron temperature T_high (posterior median) inferred from [O III] 4363/5007 (Scholte+2026).",
        "dtype": "float64",
        "blank_value": np.nan,
    },
    "TE_T_OIII_LO": {
        "unit": "K",
        "description": "16th percentile of the TE_T_OIII posterior.",
        "dtype": "float64",
        "blank_value": np.nan,
    },
    "TE_T_OIII_HI": {
        "unit": "K",
        "description": "84th percentile of the TE_T_OIII posterior.",
        "dtype": "float64",
        "blank_value": np.nan,
    },
    "TE_T_OIII_ERR": {
        "unit": "K",
        "description": "Half-width 0.5*(TE_T_OIII_HI - TE_T_OIII_LO) of the TE_T_OIII posterior.",
        "dtype": "float64",
        "blank_value": np.nan,
    },
    "TE_AV": {
        "unit": "mag",
        "description": "Direct-method V-band attenuation A_V (Cardelli+1989, R_V=3.1) jointly constrained by Halpha/Hbeta and Hgamma/Hbeta in the pn_functions fit.",
        "dtype": "float64",
        "blank_value": np.nan,
    },
    "TE_AV_LO": {
        "unit": "mag",
        "description": "16th percentile of the TE_AV posterior.",
        "dtype": "float64",
        "blank_value": np.nan,
    },
    "TE_AV_HI": {
        "unit": "mag",
        "description": "84th percentile of the TE_AV posterior.",
        "dtype": "float64",
        "blank_value": np.nan,
    },
    "TE_AV_ERR": {
        "unit": "mag",
        "description": "Half-width 0.5*(TE_AV_HI - TE_AV_LO) of the TE_AV posterior.",
        "dtype": "float64",
        "blank_value": np.nan,
    },
    "TE_LOG_O2_ABUND": {
        "unit": None,
        "description": "log10(O+/H+) ionic abundance (posterior median) from the pn_functions direct-method fit.",
        "dtype": "float64",
        "blank_value": np.nan,
    },
    "TE_LOG_O2_ABUND_LO": {
        "unit": None,
        "description": "16th percentile of the TE_LOG_O2_ABUND posterior.",
        "dtype": "float64",
        "blank_value": np.nan,
    },
    "TE_LOG_O2_ABUND_HI": {
        "unit": None,
        "description": "84th percentile of the TE_LOG_O2_ABUND posterior.",
        "dtype": "float64",
        "blank_value": np.nan,
    },
    "TE_LOG_O2_ABUND_ERR": {
        "unit": None,
        "description": "Half-width 0.5*(TE_LOG_O2_ABUND_HI - TE_LOG_O2_ABUND_LO) of the TE_LOG_O2_ABUND posterior.",
        "dtype": "float64",
        "blank_value": np.nan,
    },
    "TE_LOG_O3_ABUND": {
        "unit": None,
        "description": "log10(O++/H+) ionic abundance (posterior median) from the pn_functions direct-method fit.",
        "dtype": "float64",
        "blank_value": np.nan,
    },
    "TE_LOG_O3_ABUND_LO": {
        "unit": None,
        "description": "16th percentile of the TE_LOG_O3_ABUND posterior.",
        "dtype": "float64",
        "blank_value": np.nan,
    },
    "TE_LOG_O3_ABUND_HI": {
        "unit": None,
        "description": "84th percentile of the TE_LOG_O3_ABUND posterior.",
        "dtype": "float64",
        "blank_value": np.nan,
    },
    "TE_LOG_O3_ABUND_ERR": {
        "unit": None,
        "description": "Half-width 0.5*(TE_LOG_O3_ABUND_HI - TE_LOG_O3_ABUND_LO) of the TE_LOG_O3_ABUND posterior.",
        "dtype": "float64",
        "blank_value": np.nan,
    },
    "TE_12_LOG_OH": {
        "unit": None,
        "description": "Direct-method 12 + log10(O/H) = 12 + log10(10^TE_LOG_O2_ABUND + 10^TE_LOG_O3_ABUND), computed per posterior sample so the O+/O++ anti-correlation is preserved.",
        "dtype": "float64",
        "blank_value": np.nan,
    },
    "TE_12_LOG_OH_LO": {
        "unit": None,
        "description": "16th percentile of the TE_12_LOG_OH posterior.",
        "dtype": "float64",
        "blank_value": np.nan,
    },
    "TE_12_LOG_OH_HI": {
        "unit": None,
        "description": "84th percentile of the TE_12_LOG_OH posterior.",
        "dtype": "float64",
        "blank_value": np.nan,
    },
    "TE_12_LOG_OH_ERR": {
        "unit": None,
        "description": "Half-width 0.5*(TE_12_LOG_OH_HI - TE_12_LOG_OH_LO) of the TE_12_LOG_OH posterior.",
        "dtype": "float64",
        "blank_value": np.nan,
    },
    "TE_N_RATIOS": {
        "unit": None,
        "description": "Number of usable emission-line ratios fed to the direct-method fit (out of 6: [O II] 3726/3729, Hbeta/Halpha, Hgamma/Hbeta, [O III] 4363/5007, ([O II] 3726+3729)/Hbeta, [O III] 5007/Hbeta). 0 for rows without an attempted fit.",
        "dtype": "int32",
    },
    "TE_FIT_SUCCESS": {
        "unit": None,
        "description": "True if the direct-method fit converged for this row (te_mask passed AND pn_functions.compute_direct_metallicities returned success). False otherwise.",
        "dtype": "bool",
    },
    "TE_CHI2_AV": {
        "unit": None,
        "description": "Chi-square of the observed Balmer ratios (Hbeta/Halpha, Hgamma/Hbeta) vs the model at the reported posterior-median ne/Te/Av; the extinction goodness-of-fit for the reported values. NaN if no Balmer ratio was usable.",
        "dtype": "float64",
        "blank_value": np.nan,
    },
    "TE_CHI2_AV_ML": {
        "unit": None,
        "description": "Same Balmer-ratio chi-square as TE_CHI2_AV but evaluated at the maximum-likelihood ne/Te/Av (Stage-1 ML point for the two-stage informative-prior fit); the best achievable Balmer fit. NaN if no Balmer ratio was usable.",
        "dtype": "float64",
        "blank_value": np.nan,
    },
    "TE_AV_ML": {
        "unit": "mag",
        "description": "Maximum-likelihood V-band attenuation A_V (Stage-1 ML point for the two-stage informative-prior fit), for reference next to the posterior-median TE_AV. NaN where no fit was attempted.",
        "dtype": "float64",
        "blank_value": np.nan,
    },
    "TE_ESS": {
        "unit": None,
        "description": "UltraNest effective sample size of the posterior (Stage 2 for the two-stage informative-prior fit). NaN where no fit was attempted.",
        "dtype": "float64",
        "blank_value": np.nan,
    },
    "TE_LOGZ": {
        "unit": None,
        "description": "UltraNest log-evidence (ln Z) of the direct-method fit (Stage 2 for the two-stage informative-prior fit). NaN where no fit was attempted.",
        "dtype": "float64",
        "blank_value": np.nan,
    },
    "TE_LOGZERR": {
        "unit": None,
        "description": "1-sigma uncertainty on TE_LOGZ. NaN where no fit was attempted.",
        "dtype": "float64",
        "blank_value": np.nan,
    },

    # ---------------------------------------------------------------
    # MAG_*_MODEL_* fastspec model photometry columns
    #
    # Previously written to the FASTSPEC HDU by
    # consolidate_photometry.add_model_photometry_to_fastspec; now written
    # to SPEC_DERIVED by code/add_nebular_props.py via
    # sfr_and_metallicity.add_model_photometry_to_spec_derived, matched by
    # TARGETID to the pre-computed model_photometry_diffs_{gal_type}.fits
    # tables. Unmatched TARGETIDs leave NaN.
    # ---------------------------------------------------------------
    "MAG_G_DECAM_MODEL_NOEMI": {
        "unit": "mag",
        "description": "DECam g-band AB magnitude of the fastspecfit continuum-only model (no emission lines).",
        "dtype": "float64",
    },
    "MAG_R_DECAM_MODEL_NOEMI": {
        "unit": "mag",
        "description": "DECam r-band AB magnitude of the fastspecfit continuum-only model (no emission lines).",
        "dtype": "float64",
    },
    "MAG_G_DECAM_MODEL_WEMI": {
        "unit": "mag",
        "description": "DECam g-band AB magnitude of the fastspecfit model including emission lines.",
        "dtype": "float64",
    },
    "MAG_R_DECAM_MODEL_WEMI": {
        "unit": "mag",
        "description": "DECam r-band AB magnitude of the fastspecfit model including emission lines.",
        "dtype": "float64",
    },
    "MAG_G_BASS_MODEL_WEMI": {
        "unit": "mag",
        "description": "BASS g-band AB magnitude of the fastspecfit model including emission lines.",
        "dtype": "float64",
    },
    "MAG_R_BASS_MODEL_WEMI": {
        "unit": "mag",
        "description": "BASS r-band AB magnitude of the fastspecfit model including emission lines.",
        "dtype": "float64",
    },
    "MAG_G_SDSS_MODEL_NOEMI": {
        "unit": "mag",
        "description": "SDSS g-band AB magnitude of the fastspecfit continuum-only model (no emission lines) at observed redshift.",
        "dtype": "float64",
    },
    "MAG_R_SDSS_MODEL_NOEMI": {
        "unit": "mag",
        "description": "SDSS r-band AB magnitude of the fastspecfit continuum-only model (no emission lines) at observed redshift.",
        "dtype": "float64",
    },
    "MAG_G_SDSS_Z0_MODEL_NOEMI": {
        "unit": "mag",
        "description": "SDSS g-band AB magnitude of the fastspecfit continuum-only model (no emission lines) k-corrected to z=0.",
        "dtype": "float64",
    },
    "MAG_R_SDSS_Z0_MODEL_NOEMI": {
        "unit": "mag",
        "description": "SDSS r-band AB magnitude of the fastspecfit continuum-only model (no emission lines) k-corrected to z=0.",
        "dtype": "float64",
    },
}




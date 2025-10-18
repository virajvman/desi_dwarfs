

## DESI DR1 Extragalactic Dwarf Galaxy Catalog Code Base

IN PROGRESS!

Contact: Viraj Manwadkar (virajvm@stanford.edu)

### DESI DR1 Extragalactic Dwarf Galaxy Catalog — Data Model


<details>
<summary><strong>Extension: MAIN</strong></summary>

<br>

| Name | Type | Units | Description |
|------|------|-------|-------------|
| TARGETID | int64 |  | DESI TARGET ID |
| SURVEY | str |  | Survey name |
| PROGRAM | str |  | Program name |
| Z | float64 |  | Redrock Redshift |
| DELTACHI2 | float64 |  | Redrock delta-chi-squared |
| ZWARN | int8 |  | Redrock zwarning bit |
| RA | float64 | deg | Right Ascension of galaxy center |
| DEC | float64 | deg | Declination of galaxy center |
| RA_TARGET | float64 | deg | Right Ascension from target catalog |
| DEC_TARGET | float64 | deg | Declination from target catalog |
| DESINAME | str |  | DESI object name |
| LUMI_DIST_MPC | float32 | Mpc | Luminosity distance in Mpc |
| LOG_MSTAR_SAGA | float32 | $\mathrm{\log(M_\odot)}$ | Log stellar mass using the fiducial luminosity distance and SAGA gr-based approximation |
| LOG_MSTAR_M24 | float32 | $\mathrm{\log(M_\odot)}$ | Log stellar mass using the fiducial luminosity distance and de los Reyes et al. 2024 gr-based approximation |
| MAG_G | float32 | $\mathrm{mag}$ | g-band magnitude (MW extinction corrected) |
| MAG_R | float32 | $\mathrm{mag}$ | r-band magnitude (MW extinction corrected) |
| MAG_Z | float32 | $\mathrm{mag}$ | z-band magnitude (MW extinction corrected) |
| MAG_G_TARGET | float32 | $\mathrm{mag}$ | g-band magnitude (MW extinction corrected). For shredded sources, this is the uncorrected, shredded photometry |
| MAG_R_TARGET | float32 | $\mathrm{mag}$ | r-band magnitude (MW extinction corrected) |
| MAG_Z_TARGET | float32 | $\mathrm{mag}$ | z-band magnitude (MW extinction corrected) |
| SAMPLE | str |  | DESI target class (e.g., BGS_BRIGHT, BGS_FAINT)  |
| DWARF_MASKBIT | int32 |  | Bitwise mask to apply various cleaning cuts. See here for description of bitmasks here.  |
| MAG_TYPE | str |  | Photometry MASKBIT  |
| PHOTOMETRY_UPDATED | bool |  | Boolean indicating whether the photometry was updated from its original target Tractor photometry.  |
| SHAPE_PARAMS | str | | Galaxy shape parameters: semi-major axis in arcsec, b/a ratio, position angle (degrees)  |
| IN_SGA_2020 | bool |  | Boolean indicating whether targeted source had Tractor MASKBITS=12, that is, in SGA-2020 catalog  |

</details>



<details>
<summary><strong>Extension: ZCAT</strong></summary>

<br>


| Name | Type | Units | Description |
|------|------|-------|-------------|
| TARGETID | int64 |  | DESI TARGET ID |
| HEALPIX | int32 |  | healpix containing this location at NSIDE=64 in the NESTED scheme |
| CMX_TARGET | int64 |  | Commissioning (CMX) targeting bit |
| DESI_TARGET | int64 |  | DESI targeting bit |
| BGS_TARGET | int64 |  | BGS targeting bit |
| MWS_TARGET | int64 |  | MWS targeting bit |
| SCND_TARGET | int64 |  | Secondary target targeting bit |
| SV1_DESI_TARGET | int64 |  | SV1 DESI targeting bit |
| SV1_BGS_TARGET | int64 |  | SV1 BGS targeting bit |
| SV1_MWS_TARGET | int64 |  | SV1 MWS targeting bit |
| SV2_DESI_TARGET | int64 |  | SV2 DESI targeting bit |
| SV2_BGS_TARGET | int64 |  | SV2 BGS targeting bit |
| SV2_MWS_TARGET | int64 |  | SV2 MWS targeting bit |
| SV3_DESI_TARGET | int64 |  | SV3 DESI targeting bit |
| SV3_BGS_TARGET | int64 |  | SV3 BGS targeting bit |
| SV3_MWS_TARGET | int64 |  | SV3 MWS targeting bit |
| SV1_SCND_TARGET | int64 |  | SV1 secondary targeting bit |
| SV2_SCND_TARGET | int64 |  | SV2 secondary targeting bit |
| SV3_SCND_TARGET | int64 |  | SV3 secondary targeting bit |
| TSNR2_LRG | float32 |  | LRG template (S/N)^2 summed over B,R,Z |
| CHI2 | float32 |  | Best fit Redrock chi squared |
| OBJTYPE | str |  | Object type: TGT, SKY, NON, BAD |
| OBSCONDITIONS | int32 |  | Flag the target to be observed in graytime |
| COADD_NUMEXP | int16 |  | Number of exposures in coadd |
| COADD_EXPTIME | float32 | s | Summed exposure time for coadd |
| COADD_NUMTILE | int16 |  | Number of tiles in coadd |
| MEAN_PSF_TO_FIBER_SPECFLUX | float32 |  | Mean fraction of light from point-like source captured by 1.5 arcsec diameter fiber given atmospheric seeing |
| MIN_MJD | float64 | d | Minimum Modified Julian Date when the shutter was open for the first exposure used in the coadded spectrum |
| MAX_MJD | float64 | d | Maximum Modified Julian Date when the shutter was open for the last exposure used in the coadded spectrum |
| MEAN_MJD | float64 | d | Mean Modified Julian Date over exposures used in the coadded spectrum |
| ZCAT_NSPEC | int16 |  | Number of times this TARGETID appears in this catalog |
| ZCAT_PRIMARY | bool |  | Boolean flag (True/False) for the primary coadded spectrum in zpix zcatalog |


</details>

<details>
<summary><strong>Extension: TRACTOR_CAT</strong></summary>


<br>


| Name | Type | Units | Description |
|------|------|-------|-------------|
| TARGETID | int64 |  | DESI TARGET ID |
| RELEASE | int16 |  | Legacy Surveys data release number. |
| BRICKNAME | str |  | Name of the sky brick, encoding RA and Dec (e.g., '1126p222' for RA=112.6, Dec=+22.2). |
| BRICKID | int32 |  | Integer ID of the brick [1–662174]. |
| BRICK_OBJID | int32 |  | Catalog object number within this brick. Unique identifier when combined with RELEASE and BRICKID. |
| EBV | float32 | mag | Galactic extinction E(B-V) reddening from SFD98, used to compute the mw_transmission_ columns |
| FIBERFLUX_R | float32 | nmgy | Predicted r-band flux within a 1.5″ diameter fiber under 1″ Gaussian seeing (not extinction corrected). |
| MASKBITS | int16 |  | Tractor Bitwise mask indicating that an object touches a pixel in the coadd maskbits maps (see DR9 bitmasks documentation). |
| REF_ID | int64 |  | Reference catalog source ID (Tyc1*1e6 + Tyc2*10 + Tyc3 for Tycho-2, ‘sourceid’ for Gaia DR2). |
| REF_CAT | str |  | Reference catalog identifier: 'T2' (Tycho-2), 'G2' (Gaia DR2), 'L3' (SGA), or empty if none. |
| FLUX_G | float32 | nmgy | Total g-band flux corrected for Galactic extinction. |
| FLUX_IVAR_G | float32 | 1/nmgy^2 | Inverse variance of FLUX_G (extinction corrected). |
| MAG_G | float32 | mag | Extinction-corrected g-band magnitude. |
| MAG_G_ERR | float32 | mag | Uncertainty in g-band magnitude. |
| FLUX_R | float32 | nmgy | Total r-band flux corrected for Galactic extinction. |
| FLUX_IVAR_R | float32 | 1/nmgy^2 | Inverse variance of FLUX_R (extinction corrected). |
| MAG_R | float32 | mag | Extinction-corrected r-band magnitude. |
| MAG_R_ERR | float32 | mag | Uncertainty in r-band magnitude. |
| FLUX_Z | float32 | nmgy | Total z-band flux corrected for Galactic extinction. |
| FLUX_IVAR_Z | float32 | 1/nmgy^2 | Inverse variance of FLUX_Z (extinction corrected). |
| MAG_Z | float32 | mag | Extinction-corrected z-band magnitude. |
| MAG_Z_ERR | float32 | mag | Uncertainty in z-band magnitude. |
| FIBERMAG_R | float32 | mag | Predicted r-band magnitude within 1.5″ fiber (not extinction corrected). |
| OBJID | int32 |  | Object number within the brick (0–N−1), unique within a given RELEASE and BRICKID. |
| SIGMA_G | float32 | arcsec | Gaussian sigma of the object model in g-band. |
| FRACFLUX_G | float32 |  | Profile-weighted fraction of flux from neighboring sources divided by total flux in g-band. |
| RCHISQ_G | float32 |  | Reduced chi-squared of the g-band model fit. |
| SIGMA_R | float32 | arcsec | Gaussian sigma of the object model in r-band. |
| FRACFLUX_R | float32 |  | Profile-weighted fraction of flux from neighboring sources divided by total flux in r-band. |
| RCHISQ_R | float32 |  | Reduced chi-squared of the r-band model fit. |
| SIGMA_Z | float32 | arcsec | Gaussian sigma of the object model in z-band. |
| FRACFLUX_Z | float32 |  | Profile-weighted fraction of flux from neighboring sources divided by total flux in z-band. |
| RCHISQ_Z | float32 |  | Reduced chi-squared of the z-band model fit. |
| SHAPE_R | float32 | arcsec | Half-light radius of the best-fit galaxy model (r-band). |
| SHAPE_R_ERR | float32 | arcsec | Uncertainty in the half-light radius (r-band). |
| MU_R | float32 | mag/arcsec^2 | Surface brightness within the effective radius in r-band. |
| MU_R_ERR | float32 | mag/arcsec^2 | Uncertainty in the surface brightness (r-band). |
| SERSIC | float32 |  | Power-law index for the Sersic profile model (type='SER'). |
| SERSIC_IVAR | float32 |  | Inverse variance of the Sersic index parameter. |
| BA | float32 |  | Axis ratio (b/a) of the best-fit galaxy model. |
| TYPE | str |  | Object type as classified by the Tractor model. |
| PHI | float32 | deg | Position angle of the major axis |
| NOBS_G | int16 |  | Number of images contributing to the central pixel in the g-band. |
| NOBS_R | int16 |  | Number of images contributing to the central pixel in the r-band. |
| NOBS_Z | int16 |  | Number of images contributing to the central pixel in the z-band. |
| MW_TRANSMISSION_G | float32 |  | Galactic transmission in g filter in linear units [0, 1] |
| MW_TRANSMISSION_R | float32 |  | Galactic transmission in r filter in linear units [0, 1] |
| MW_TRANSMISSION_Z | float32 |  | Galactic transmission in z filter in linear units [0, 1] |
| SWEEP | str |  | Name of the sweep file from which this source was extracted. |

</details>

<details>
<summary><strong>Extension: REPROCESS_PHOTO_CAT</strong></summary>

<br>



</details>


<details>
<summary><strong>Extension: SPECTRA_TEMPLATE_CAT</strong></summary>

<br>



</details>


<details>
<summary><strong>Extension: FASTSPEC</strong></summary>

<br>

| Name | Type | Units | Description |
|------|------|-------|-------------|
| TARGETID | int64 |  | DESI TARGET ID |
| DN4000 | float32 |  | Narrow 4000-Å break index (Balogh et al. 1999) measured from the emission-line subtracted spectrum. |
| DN4000_OBS | float32 |  | Narrow 4000-Å break index measured from the observed spectrum. |
| DN4000_IVAR | float32 |  | Inverse variance of DN4000 and DN4000_OBS. |
| DN4000_MODEL | float32 |  | Narrow 4000-Å break index measured from the best-fitting continuum model. |
| DN4000_MODEL_IVAR | float32 |  | Inverse variance of DN4000_MODEL. |
| SNR_B | float32 |  | Median signal-to-noise ratio per pixel in the b camera. |
| SNR_R | float32 |  | Median signal-to-noise ratio per pixel in the r camera. |
| SNR_Z | float32 |  | Median signal-to-noise ratio per pixel in the z camera. |
| APERCORR | float32 |  | Median aperture correction factor. |
| APERCORR_G | float32 |  | Aperture correction factor measured in the g band. |
| APERCORR_R | float32 |  | Aperture correction factor measured in the r band. |
| APERCORR_Z | float32 |  | Aperture correction factor measured in the z band. |
| OII_3726_FLUX | float32 | 1e-17 erg / (cm2 s) | Gaussian-integrated emission-line flux for OII_3726. |
| OII_3726_FLUX_IVAR | float32 | 1e+34 cm4 s2 / erg2 | Inverse variance in OII_3726_FLUX. |
| OII_3729_FLUX | float32 | 1e-17 erg / (cm2 s) | Gaussian-integrated emission-line flux for OII_3729. |
| OII_3729_FLUX_IVAR | float32 | 1e+34 cm4 s2 / erg2 | Inverse variance in OII_3729_FLUX. |
| OIII_4363_FLUX | float32 | 1e-17 erg / (cm2 s) | Gaussian-integrated emission-line flux for OIII_4363. |
| OIII_4363_FLUX_IVAR | float32 | 1e+34 cm4 s2 / erg2 | Inverse variance in OIII_4363_FLUX. |
| HEII_4686_FLUX | float32 | 1e-17 erg / (cm2 s) | Gaussian-integrated emission-line flux for HEII_4686. |
| HEII_4686_FLUX_IVAR | float32 | 1e+34 cm4 s2 / erg2 | Inverse variance in HEII_4686_FLUX. |
| HBETA_FLUX | float32 | 1e-17 erg / (cm2 s) | Gaussian-integrated emission-line flux for HBETA. |
| HBETA_FLUX_IVAR | float32 | 1e+34 cm4 s2 / erg2 | Inverse variance in HBETA_FLUX. |
| OIII_4959_FLUX | float32 | 1e-17 erg / (cm2 s) | Gaussian-integrated emission-line flux for OIII_4959. |
| OIII_4959_FLUX_IVAR | float32 | 1e+34 cm4 s2 / erg2 | Inverse variance in OIII_4959_FLUX. |
| OIII_5007_FLUX | float32 | 1e-17 erg / (cm2 s) | Gaussian-integrated emission-line flux for OIII_5007. |
| OIII_5007_FLUX_IVAR | float32 | 1e+34 cm4 s2 / erg2 | Inverse variance in OIII_5007_FLUX. |
| HEI_5876_FLUX | float32 | 1e-17 erg / (cm2 s) | Gaussian-integrated emission-line flux for HEI_5876. |
| HEI_5876_FLUX_IVAR | float32 | 1e+34 cm4 s2 / erg2 | Inverse variance in HEI_5876_FLUX. |
| NII_6548_FLUX | float32 | 1e-17 erg / (cm2 s) | Gaussian-integrated emission-line flux for NII_6548. |
| NII_6548_FLUX_IVAR | float32 | 1e+34 cm4 s2 / erg2 | Inverse variance in NII_6548_FLUX. |
| HALPHA_FLUX | float32 | 1e-17 erg / (cm2 s) | Gaussian-integrated emission-line flux for HALPHA. |
| HALPHA_FLUX_IVAR | float32 | 1e+34 cm4 s2 / erg2 | Inverse variance in HALPHA_FLUX. |
| HALPHA_BROAD_FLUX | float32 | 1e-17 erg / (cm2 s) | Gaussian-integrated emission-line flux for HALPHA_BROAD. |
| HALPHA_BROAD_FLUX_IVAR | float32 | 1e+34 cm4 s2 / erg2 | Inverse variance in HALPHA_BROAD_FLUX. |
| NII_6584_FLUX | float32 | 1e-17 erg / (cm2 s) | Gaussian-integrated emission-line flux for NII_6584. |
| NII_6584_FLUX_IVAR | float32 | 1e+34 cm4 s2 / erg2 | Inverse variance in NII_6584_FLUX. |
| SII_6716_FLUX | float32 | 1e-17 erg / (cm2 s) | Gaussian-integrated emission-line flux for SII_6716. |
| SII_6716_FLUX_IVAR | float32 | 1e+34 cm4 s2 / erg2 | Inverse variance in SII_6716_FLUX. |
| SII_6731_FLUX | float32 | 1e-17 erg / (cm2 s) | Gaussian-integrated emission-line flux for SII_6731. |
| SII_6731_FLUX_IVAR | float32 | 1e+34 cm4 s2 / erg2 | Inverse variance in SII_6731_FLUX. |
| SIII_9069_FLUX | float32 | 1e-17 erg / (cm2 s) | Gaussian-integrated emission-line flux for SIII_9069. |
| SIII_9069_FLUX_IVAR | float32 | 1e+34 cm4 s2 / erg2 | Inverse variance in SIII_9069_FLUX. |
| SIII_9532_FLUX | float32 | 1e-17 erg / (cm2 s) | Gaussian-integrated emission-line flux for SIII_9532. |
| SIII_9532_FLUX_IVAR | float32 | 1e+34 cm4 s2 / erg2 | Inverse variance in SIII_9532_FLUX. |
| HALPHA_BOXFLUX | float32 | 1e-17 erg / (cm2 s) | Boxcar-integrated Halpha emission-line flux. |
| HALPHA_BOXFLUX_IVAR | float32 | 1e+34 cm4 s2 / erg2 | Inverse variance in HALPHA_BOXFLUX. |
| HALPHA_EW | float32 | Angstrom | Rest-frame equivalent width of Halpha emission line. |
| HALPHA_EW_IVAR | float32 | 1 / Angstrom2 | Inverse variance in HALPHA_EW. |
| HALPHA_SIGMA | float32 | km / s | Gaussian emission-line width of Halpha before convolution with the resolution matrix. |
| HALPHA_SIGMA_IVAR | float32 | s2 / km2 | Inverse variance in HALPHA_SIGMA. |
</details>


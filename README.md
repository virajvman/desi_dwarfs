
# DESI Extragalactic Dwarf Galaxy Catalog

**Contact:** Viraj Manwadkar ([virajvm@stanford.edu](mailto:virajvm@stanford.edu))

The DESI Extragalactic Dwarf Galaxy Catalog provides spectroscopically confirmed dwarf galaxies from the Dark Energy Spectroscopic Instrument (DESI) Data Release 1. The catalog includes reprocessed photometry, spectral measurements, and value-added properties for low-mass galaxies with $\log(M_\star / M_\odot) < 9.25$. The catalog is stored as a multi-extension FITS file with six extensions: **MAIN**, **ZCAT**, **TRACTOR**, **FASTSPEC**, **REPROCESS_PHOTO**, and **SPECTRA_TEMPLATE**.

<p align="center">
  <img src="figs/dwarf_example_panel.jpg" width="90%" alt="Example dwarf galaxies from the DESI catalog">
</p>
<p align="center"><em>Example dwarf galaxies in the DESI DR1 Extragalactic Dwarf Galaxy Catalog.</em></p>

---

### [Interactive Catalog Viewer](https://virajvman.github.io/desidwarfs_webapp/interactive.html)

Explore the DESI Dwarf Galaxy catalog interactively in your browser.

---

<details>
<summary><h3>Catalog Data Model</h3></summary>

<br>

The catalog is a multi-extension FITS file. Each extension is keyed by `TARGETID` and described below.

---

<details>
<summary><strong>Extension 1 &mdash; MAIN</strong></summary>

<br>

Core identifiers, coordinates, redshifts, stellar masses, photometry, and quality flags.

| Name | Type | Units | Description |
| :--- | :--- | :---: | :---------- |
| `TARGETID` | int64 | | DESI TARGET ID |
| `SURVEY` | str | | Survey name |
| `PROGRAM` | str | | Program name |
| `HEALPIX` | int32 | | HEALPix index at NSIDE=64 in the NESTED scheme |
| `Z` | float64 | | Redrock redshift (heliocentric) |
| `DELTACHI2` | float64 | | Redrock delta-chi-squared |
| `ZWARN` | int8 | | Redrock zwarning bit |
| `Z_CMB` | float64 | | Redrock redshift (CMB rest frame) |
| `RA` | float64 | deg | Right Ascension of the galaxy. Same as target catalog, except for galaxies reprocessed after being identified as likely shredded |
| `DEC` | float64 | deg | Declination of the galaxy. Same as target catalog, except for galaxies reprocessed after being identified as likely shredded |
| `RA_TARGET` | float64 | deg | Right Ascension from target catalog |
| `DEC_TARGET` | float64 | deg | Declination from target catalog |
| `DESINAME` | str | | DESI object name |
| `LUMI_DIST_MPC` | float32 | Mpc | Fiducial luminosity distance |
| `LOG_MSTAR_SAGA` | float32 | $\log(M_\odot)$ | Log stellar mass using `LUMI_DIST_MPC` and the SAGA *gr*-based approximation |
| `LOG_MSTAR_M24` | float32 | $\log(M_\odot)$ | Log stellar mass using `LUMI_DIST_MPC` and de los Reyes et al. 2024 *gr*-based approximation |
| `MAG_G` | float32 | mag | *g*-band magnitude (MW extinction corrected). Same as Tractor photometry, except for galaxies reprocessed after being identified as likely shredded |
| `MAG_R` | float32 | mag | Same as `MAG_G` but for *r*-band |
| `MAG_Z` | float32 | mag | Same as `MAG_G` but for *z*-band |
| `MAG_G_TARGET` | float32 | mag | Tractor *g*-band magnitude of DESI target source (MW extinction corrected). For shredded sources, this is the uncorrected, shredded photometry |
| `MAG_R_TARGET` | float32 | mag | Same as `MAG_G_TARGET` but for *r*-band |
| `MAG_Z_TARGET` | float32 | mag | Same as `MAG_G_TARGET` but for *z*-band |
| `SAMPLE` | str | | DESI target class (`BGS_BRIGHT`, `BGS_FAINT`, `LOWZ`, or `ELG`) |
| `DWARF_MASKBIT` | int32 | | Bitwise mask for cleaning cuts. See [bitmask descriptions](#dwarf_maskbit-descriptions) |
| `MAG_TYPE` | str | | Photometry method used for `MAG_G/R/Z`. See [MAG_TYPE descriptions](#mag-type-descriptions) |
| `PHOTOMETRY_UPDATED` | bool | | Whether photometry was updated from original target Tractor photometry |
| `R50_R` | float32 | arcsec | Half-light radius in *r*-band |
| `SHAPE_PARAMS` | float32 (2,) | | Galaxy shape parameters: *b/a* ratio, position angle (degrees) |
| `IN_SGA_2020` | bool | | Whether target source had Tractor `MASKBITS=12` (in SGA-2020 catalog) |
| `ASSOCIATED_TARGETIDS` | object | | List of associated TARGETIDs (variable-length per row) |
| `DWARF_PRIMARY_TARGETID` | int64 | | TARGETID of the primary fiber, chosen as the brightest `MAG_R_TARGET` among associated fibers |
| `DWARF_PRIMARY` | bool | | Whether this row is the primary fiber (`TARGETID == DWARF_PRIMARY_TARGETID`) |

</details>

---

<details>
<summary><strong>Extension 2 &mdash; ZCAT</strong></summary>

<br>

Redshift catalog columns, targeting bits, and observation metadata.

| Name | Type | Units | Description |
| :--- | :--- | :---: | :---------- |
| `TARGETID` | int64 | | DESI TARGET ID |
| `CMX_TARGET` | int64 | | Commissioning (CMX) targeting bit |
| `DESI_TARGET` | int64 | | DESI targeting bit |
| `BGS_TARGET` | int64 | | BGS targeting bit |
| `MWS_TARGET` | int64 | | MWS targeting bit |
| `SCND_TARGET` | int64 | | Secondary target targeting bit |
| `SV1_DESI_TARGET` | int64 | | SV1 DESI targeting bit |
| `SV1_BGS_TARGET` | int64 | | SV1 BGS targeting bit |
| `SV1_MWS_TARGET` | int64 | | SV1 MWS targeting bit |
| `SV2_DESI_TARGET` | int64 | | SV2 DESI targeting bit |
| `SV2_BGS_TARGET` | int64 | | SV2 BGS targeting bit |
| `SV2_MWS_TARGET` | int64 | | SV2 MWS targeting bit |
| `SV3_DESI_TARGET` | int64 | | SV3 DESI targeting bit |
| `SV3_BGS_TARGET` | int64 | | SV3 BGS targeting bit |
| `SV3_MWS_TARGET` | int64 | | SV3 MWS targeting bit |
| `SV1_SCND_TARGET` | int64 | | SV1 secondary targeting bit |
| `SV2_SCND_TARGET` | int64 | | SV2 secondary targeting bit |
| `SV3_SCND_TARGET` | int64 | | SV3 secondary targeting bit |
| `TSNR2_LRG` | float32 | | LRG template (S/N)^2 summed over B, R, Z |
| `CHI2` | float32 | | Best-fit Redrock chi-squared |
| `OBJTYPE` | str | | Object type: TGT, SKY, NON, BAD |
| `OBSCONDITIONS` | int32 | | Flag the target to be observed in graytime |
| `COADD_NUMEXP` | int16 | | Number of exposures in coadd |
| `COADD_EXPTIME` | float32 | s | Summed exposure time for coadd |
| `COADD_NUMTILE` | int16 | | Number of tiles in coadd |
| `MEAN_PSF_TO_FIBER_SPECFLUX` | float32 | | Mean fraction of light from point-like source captured by 1.5 arcsec diameter fiber given atmospheric seeing |
| `MIN_MJD` | float64 | d | Minimum MJD of the first exposure used in the coadded spectrum |
| `MAX_MJD` | float64 | d | Maximum MJD of the last exposure used in the coadded spectrum |
| `MEAN_MJD` | float64 | d | Mean MJD over exposures used in the coadded spectrum |
| `ZCAT_NSPEC` | int16 | | Number of times this TARGETID appears in this catalog |
| `ZCAT_PRIMARY` | bool | | Primary coadded spectrum flag in zpix zcatalog |

</details>

---

<details>
<summary><strong>Extension 3 &mdash; TRACTOR</strong></summary>

<br>

Original Tractor photometry and morphological parameters from the DESI Legacy Surveys DR9.

| Name | Type | Units | Description |
| :--- | :--- | :---: | :---------- |
| `TARGETID` | int64 | | DESI TARGET ID |
| `RELEASE` | int16 | | Integer denoting the camera and filter set used, unique for a given processing run |
| `BRICKNAME` | str | | Name of the sky brick, encoding RA and Dec (e.g., `1126p222`) |
| `BRICKID` | int32 | | Integer ID of the brick [1--662174] |
| `BRICK_OBJID` | int32 | | Catalog object number within this brick. Unique when combined with `RELEASE` and `BRICKID` |
| `EBV` | float32 | mag | Galactic extinction E(B-V) reddening from SFD98 |
| `FIBERFLUX_R` | float32 | nanomaggy | Predicted *r*-band flux within a 1.5 arcsec diameter fiber under 1 arcsec Gaussian seeing (not extinction corrected) |
| `MASKBITS` | int16 | | Tractor bitwise mask from coadd maskbits maps ([DR9 bitmasks](https://www.legacysurvey.org/dr9/bitmasks/)) |
| `REF_ID` | int64 | | Reference catalog source ID (Tycho-2 or Gaia DR2) |
| `REF_CAT` | str | | Reference catalog: `T2` (Tycho-2), `G2` (Gaia DR2), `L3` (SGA), or empty |
| `FLUX_G` | float32 | nanomaggy | Total *g*-band flux (extinction corrected) |
| `FLUX_IVAR_G` | float32 | 1/nanomaggy² | Inverse variance of `FLUX_G` |
| `MAG_G` | float32 | mag | Extinction-corrected *g*-band magnitude |
| `MAG_G_ERR` | float32 | mag | Uncertainty in *g*-band magnitude |
| `FLUX_R` | float32 | nanomaggy | Total *r*-band flux (extinction corrected) |
| `FLUX_IVAR_R` | float32 | 1/nanomaggy² | Inverse variance of `FLUX_R` |
| `MAG_R` | float32 | mag | Extinction-corrected *r*-band magnitude |
| `MAG_R_ERR` | float32 | mag | Uncertainty in *r*-band magnitude |
| `FLUX_Z` | float32 | nanomaggy | Total *z*-band flux (extinction corrected) |
| `FLUX_IVAR_Z` | float32 | 1/nanomaggy² | Inverse variance of `FLUX_Z` |
| `MAG_Z` | float32 | mag | Extinction-corrected *z*-band magnitude |
| `MAG_Z_ERR` | float32 | mag | Uncertainty in *z*-band magnitude |
| `FIBERMAG_R` | float32 | mag | Predicted *r*-band magnitude within 1.5 arcsec fiber (not extinction corrected) |
| `OBJID` | int32 | | Object number within the brick, unique within a given `RELEASE` and `BRICKID` |
| `SIGMA_G` | float32 | arcsec | Gaussian sigma of the object model in *g*-band |
| `FRACFLUX_G` | float32 | | Profile-weighted fraction of flux from neighboring sources in *g*-band |
| `RCHISQ_G` | float32 | | Reduced chi-squared of the *g*-band model fit |
| `SIGMA_R` | float32 | arcsec | Gaussian sigma of the object model in *r*-band |
| `FRACFLUX_R` | float32 | | Profile-weighted fraction of flux from neighboring sources in *r*-band |
| `RCHISQ_R` | float32 | | Reduced chi-squared of the *r*-band model fit |
| `SIGMA_Z` | float32 | arcsec | Gaussian sigma of the object model in *z*-band |
| `FRACFLUX_Z` | float32 | | Profile-weighted fraction of flux from neighboring sources in *z*-band |
| `RCHISQ_Z` | float32 | | Reduced chi-squared of the *z*-band model fit |
| `SHAPE_R` | float32 | arcsec | Half-light radius of the best-fit galaxy model (*r*-band) |
| `SHAPE_R_ERR` | float32 | arcsec | Uncertainty in the half-light radius (*r*-band) |
| `MU_R` | float32 | mag/arcsec² | Surface brightness within the effective radius in *r*-band |
| `MU_R_ERR` | float32 | mag/arcsec² | Uncertainty in the surface brightness (*r*-band) |
| `SERSIC` | float32 | | Sersic profile index (type=`SER`) |
| `SERSIC_IVAR` | float32 | | Inverse variance of the Sersic index |
| `BA` | float32 | | Axis ratio (*b/a*) of the best-fit galaxy model |
| `TYPE` | str | | Tractor morphological type |
| `PHI` | float32 | deg | Position angle of the major axis |
| `NOBS_G` | int16 | | Number of images at the central pixel in *g*-band |
| `NOBS_R` | int16 | | Number of images at the central pixel in *r*-band |
| `NOBS_Z` | int16 | | Number of images at the central pixel in *z*-band |
| `MW_TRANSMISSION_G` | float32 | | Galactic transmission in *g* filter [0, 1] |
| `MW_TRANSMISSION_R` | float32 | | Galactic transmission in *r* filter [0, 1] |
| `MW_TRANSMISSION_Z` | float32 | | Galactic transmission in *z* filter [0, 1] |
| `SWEEP` | str | | Name of the sweep file this source was extracted from |

</details>

---

<details>
<summary><strong>Extension 4 &mdash; FASTSPEC</strong></summary>

<br>

Spectral measurements from FastSpecFit: spectral indices, emission-line fluxes, and *k*-corrected absolute magnitudes.

| Name | Type | Units | Description |
| :--- | :--- | :---: | :---------- |
| `TARGETID` | int64 | | DESI TARGET ID |
| `RA_TARGET` | float64 | deg | Right Ascension from target catalog |
| `DEC_TARGET` | float64 | deg | Declination from target catalog |
| `DN4000` | float32 | | Narrow 4000 Å break index (Balogh et al. 1999) from emission-line subtracted spectrum |
| `DN4000_OBS` | float32 | | Narrow 4000 Å break index from observed spectrum |
| `DN4000_IVAR` | float32 | | Inverse variance of `DN4000` and `DN4000_OBS` |
| `DN4000_MODEL` | float32 | | Narrow 4000 Å break index from best-fitting continuum model |
| `SNR_B` | float32 | | Median S/N per pixel in the *b* camera |
| `SNR_R` | float32 | | Median S/N per pixel in the *r* camera |
| `SNR_Z` | float32 | | Median S/N per pixel in the *z* camera |
| `APERCORR` | float32 | | Median aperture correction factor |
| `APERCORR_G` | float32 | | Aperture correction factor in *g* band |
| `APERCORR_R` | float32 | | Aperture correction factor in *r* band |
| `APERCORR_Z` | float32 | | Aperture correction factor in *z* band |
| `ABSMAG01_SDSS_G` | float32 | mag | Absolute magnitude in SDSS *g*-band, band-shifted to *z*=0.1 (*h*=1.0) |
| `ABSMAG01_SDSS_R` | float32 | mag | Absolute magnitude in SDSS *r*-band, band-shifted to *z*=0.1 (*h*=1.0) |
| `ABSMAG01_SDSS_I` | float32 | mag | Absolute magnitude in SDSS *i*-band, band-shifted to *z*=0.1 (*h*=1.0) |
| `ABSMAG01_SDSS_Z` | float32 | mag | Absolute magnitude in SDSS *z*-band, band-shifted to *z*=0.1 (*h*=1.0) |
| `ABSMAG01_IVAR_SDSS_G` | float32 | 1/mag² | Inverse variance of `ABSMAG01_SDSS_G` |
| `ABSMAG01_IVAR_SDSS_R` | float32 | 1/mag² | Inverse variance of `ABSMAG01_SDSS_R` |
| `ABSMAG01_IVAR_SDSS_I` | float32 | 1/mag² | Inverse variance of `ABSMAG01_SDSS_I` |
| `ABSMAG01_IVAR_SDSS_Z` | float32 | 1/mag² | Inverse variance of `ABSMAG01_SDSS_Z` |
| `OII_3726_FLUX` | float32 | 1e-17 erg/(cm² s) | Gaussian-integrated emission-line flux |
| `OII_3726_FLUX_IVAR` | float32 | 1e+34 cm⁴ s²/erg² | Inverse variance of `OII_3726_FLUX` |
| `OII_3729_FLUX` | float32 | 1e-17 erg/(cm² s) | Gaussian-integrated emission-line flux |
| `OII_3729_FLUX_IVAR` | float32 | 1e+34 cm⁴ s²/erg² | Inverse variance of `OII_3729_FLUX` |
| `OIII_4363_FLUX` | float32 | 1e-17 erg/(cm² s) | Gaussian-integrated emission-line flux |
| `OIII_4363_FLUX_IVAR` | float32 | 1e+34 cm⁴ s²/erg² | Inverse variance of `OIII_4363_FLUX` |
| `HEII_4686_FLUX` | float32 | 1e-17 erg/(cm² s) | Gaussian-integrated emission-line flux |
| `HEII_4686_FLUX_IVAR` | float32 | 1e+34 cm⁴ s²/erg² | Inverse variance of `HEII_4686_FLUX` |
| `HBETA_FLUX` | float32 | 1e-17 erg/(cm² s) | Gaussian-integrated emission-line flux |
| `HBETA_FLUX_IVAR` | float32 | 1e+34 cm⁴ s²/erg² | Inverse variance of `HBETA_FLUX` |
| `OIII_4959_FLUX` | float32 | 1e-17 erg/(cm² s) | Gaussian-integrated emission-line flux |
| `OIII_4959_FLUX_IVAR` | float32 | 1e+34 cm⁴ s²/erg² | Inverse variance of `OIII_4959_FLUX` |
| `OIII_5007_FLUX` | float32 | 1e-17 erg/(cm² s) | Gaussian-integrated emission-line flux |
| `OIII_5007_FLUX_IVAR` | float32 | 1e+34 cm⁴ s²/erg² | Inverse variance of `OIII_5007_FLUX` |
| `HEI_5876_FLUX` | float32 | 1e-17 erg/(cm² s) | Gaussian-integrated emission-line flux |
| `HEI_5876_FLUX_IVAR` | float32 | 1e+34 cm⁴ s²/erg² | Inverse variance of `HEI_5876_FLUX` |
| `NII_6548_FLUX` | float32 | 1e-17 erg/(cm² s) | Gaussian-integrated emission-line flux |
| `NII_6548_FLUX_IVAR` | float32 | 1e+34 cm⁴ s²/erg² | Inverse variance of `NII_6548_FLUX` |
| `HALPHA_FLUX` | float32 | 1e-17 erg/(cm² s) | Gaussian-integrated emission-line flux |
| `HALPHA_FLUX_IVAR` | float32 | 1e+34 cm⁴ s²/erg² | Inverse variance of `HALPHA_FLUX` |
| `HALPHA_BROAD_FLUX` | float32 | 1e-17 erg/(cm² s) | Gaussian-integrated broad emission-line flux |
| `HALPHA_BROAD_FLUX_IVAR` | float32 | 1e+34 cm⁴ s²/erg² | Inverse variance of `HALPHA_BROAD_FLUX` |
| `NII_6584_FLUX` | float32 | 1e-17 erg/(cm² s) | Gaussian-integrated emission-line flux |
| `NII_6584_FLUX_IVAR` | float32 | 1e+34 cm⁴ s²/erg² | Inverse variance of `NII_6584_FLUX` |
| `SII_6716_FLUX` | float32 | 1e-17 erg/(cm² s) | Gaussian-integrated emission-line flux |
| `SII_6716_FLUX_IVAR` | float32 | 1e+34 cm⁴ s²/erg² | Inverse variance of `SII_6716_FLUX` |
| `SII_6731_FLUX` | float32 | 1e-17 erg/(cm² s) | Gaussian-integrated emission-line flux |
| `SII_6731_FLUX_IVAR` | float32 | 1e+34 cm⁴ s²/erg² | Inverse variance of `SII_6731_FLUX` |
| `SIII_9069_FLUX` | float32 | 1e-17 erg/(cm² s) | Gaussian-integrated emission-line flux |
| `SIII_9069_FLUX_IVAR` | float32 | 1e+34 cm⁴ s²/erg² | Inverse variance of `SIII_9069_FLUX` |
| `SIII_9532_FLUX` | float32 | 1e-17 erg/(cm² s) | Gaussian-integrated emission-line flux |
| `SIII_9532_FLUX_IVAR` | float32 | 1e+34 cm⁴ s²/erg² | Inverse variance of `SIII_9532_FLUX` |
| `HALPHA_BOXFLUX` | float32 | 1e-17 erg/(cm² s) | Boxcar-integrated H-alpha emission-line flux |
| `HALPHA_BOXFLUX_IVAR` | float32 | 1e+34 cm⁴ s²/erg² | Inverse variance of `HALPHA_BOXFLUX` |
| `HALPHA_EW` | float32 | Angstrom | Rest-frame equivalent width of H-alpha |
| `HALPHA_EW_IVAR` | float32 | 1/Angstrom² | Inverse variance of `HALPHA_EW` |

</details>

---

<details>
<summary><strong>Extension 5 &mdash; REPROCESS_PHOTO</strong></summary>

<br>

Reprocessed photometry for sources whose photometry was updated. This extension only contains rows for galaxies with `PHOTOMETRY_UPDATED = True` in the MAIN extension.

| Name | Type | Units | Description |
| :--- | :--- | :---: | :---------- |
| `TARGETID` | int64 | | DESI TARGET ID |
| `COG_MAG_G_ISOLATE` | float32 | mag | COG magnitude in *g*-band (with isolate mask); MW extinction corrected |
| `COG_MAG_R_ISOLATE` | float32 | mag | COG magnitude in *r*-band (with isolate mask); MW extinction corrected |
| `COG_MAG_Z_ISOLATE` | float32 | mag | COG magnitude in *z*-band (with isolate mask); MW extinction corrected |
| `COG_MAG_G_NO_ISOLATE` | float32 | mag | COG magnitude in *g*-band (without isolate mask); MW extinction corrected |
| `COG_MAG_R_NO_ISOLATE` | float32 | mag | COG magnitude in *r*-band (without isolate mask); MW extinction corrected |
| `COG_MAG_Z_NO_ISOLATE` | float32 | mag | COG magnitude in *z*-band (without isolate mask); MW extinction corrected |
| `APER_R4_MAG_G_ISOLATE` | float32 | mag | R4 aperture magnitude in *g*-band (with isolate mask); MW extinction corrected |
| `APER_R4_MAG_R_ISOLATE` | float32 | mag | R4 aperture magnitude in *r*-band (with isolate mask); MW extinction corrected |
| `APER_R4_MAG_Z_ISOLATE` | float32 | mag | R4 aperture magnitude in *z*-band (with isolate mask); MW extinction corrected |
| `APER_R4_MAG_G_NO_ISOLATE` | float32 | mag | R4 aperture magnitude in *g*-band (without isolate mask); MW extinction corrected |
| `APER_R4_MAG_R_NO_ISOLATE` | float32 | mag | R4 aperture magnitude in *r*-band (without isolate mask); MW extinction corrected |
| `APER_R4_MAG_Z_NO_ISOLATE` | float32 | mag | R4 aperture magnitude in *z*-band (without isolate mask); MW extinction corrected |
| `TRACTOR_BASED_MAG_G_ISOLATE` | float32 | mag | Tractor-based parent magnitude in *g*-band (with isolate mask); MW extinction corrected |
| `TRACTOR_BASED_MAG_R_ISOLATE` | float32 | mag | Tractor-based parent magnitude in *r*-band (with isolate mask); MW extinction corrected |
| `TRACTOR_BASED_MAG_Z_ISOLATE` | float32 | mag | Tractor-based parent magnitude in *z*-band (with isolate mask); MW extinction corrected |
| `TRACTOR_BASED_MAG_G_NO_ISOLATE` | float32 | mag | Tractor-based parent magnitude in *g*-band (without isolate mask); MW extinction corrected |
| `TRACTOR_BASED_MAG_R_NO_ISOLATE` | float32 | mag | Tractor-based parent magnitude in *r*-band (without isolate mask); MW extinction corrected |
| `TRACTOR_BASED_MAG_Z_NO_ISOLATE` | float32 | mag | Tractor-based parent magnitude in *z*-band (without isolate mask); MW extinction corrected |
| `SIMPLE_PHOTO_MAG_G` | float32 | mag | Simple photometry magnitude in *g*-band (with isolate mask); MW extinction corrected |
| `SIMPLE_PHOTO_MAG_R` | float32 | mag | Simple photometry magnitude in *r*-band (with isolate mask); MW extinction corrected |
| `SIMPLE_PHOTO_MAG_Z` | float32 | mag | Simple photometry magnitude in *z*-band (with isolate mask); MW extinction corrected |
| `APERFRAC_R4_IN_IMG_ISOLATE` | float32 | | Fraction of R4 aperture inside image (with isolate mask) |
| `APERFRAC_R4_IN_IMG_NO_ISOLATE` | float32 | | Fraction of R4 aperture inside image (without isolate mask) |
| `COG_PARAMS_G_ISOLATE` | float32 (5,) | | COG fit parameters for *g*-band (with isolate mask) |
| `COG_PARAMS_R_ISOLATE` | float32 (5,) | | COG fit parameters for *r*-band (with isolate mask) |
| `COG_PARAMS_Z_ISOLATE` | float32 (5,) | | COG fit parameters for *z*-band (with isolate mask) |
| `COG_PARAMS_G_NO_ISOLATE` | float32 (5,) | | COG fit parameters for *g*-band (without isolate mask) |
| `COG_PARAMS_R_NO_ISOLATE` | float32 (5,) | | COG fit parameters for *r*-band (without isolate mask) |
| `COG_PARAMS_Z_NO_ISOLATE` | float32 (5,) | | COG fit parameters for *z*-band (without isolate mask) |
| `COG_PARAMS_G_ERR_ISOLATE` | float32 (5,) | | Errors on COG fit parameters for *g*-band (with isolate mask) |
| `COG_PARAMS_R_ERR_ISOLATE` | float32 (5,) | | Errors on COG fit parameters for *r*-band (with isolate mask) |
| `COG_PARAMS_Z_ERR_ISOLATE` | float32 (5,) | | Errors on COG fit parameters for *z*-band (with isolate mask) |
| `COG_PARAMS_G_ERR_NO_ISOLATE` | float32 (5,) | | Errors on COG fit parameters for *g*-band (without isolate mask) |
| `COG_PARAMS_R_ERR_NO_ISOLATE` | float32 (5,) | | Errors on COG fit parameters for *r*-band (without isolate mask) |
| `COG_PARAMS_Z_ERR_NO_ISOLATE` | float32 (5,) | | Errors on COG fit parameters for *z*-band (without isolate mask) |
| `COG_MAG_ERR_ISOLATE` | float32 (3,) | | COG magnitude errors in *g, r, z* (with isolate mask) |
| `COG_MAG_ERR_NO_ISOLATE` | float32 (3,) | | COG magnitude errors in *g, r, z* (without isolate mask) |
| `COG_SEG_ON_BLOB` | bool | | Whether object lies on the smoothed main blob used in COG analysis |
| `COG_FIT_RESID_ISOLATE` | float32 (3,) | | COG fit residuals for each band (with isolate mask) |
| `COG_DECREASE_MAX_LEN_ISOLATE` | float32 (3,) | | Maximum consecutive decrease length in COG for each band (with isolate mask) |
| `COG_DECREASE_MAX_MAG_ISOLATE` | float32 (3,) | | Magnitude decrease during maximum consecutive COG decrease (with isolate mask) |
| `COG_FIT_RESID_NO_ISOLATE` | float32 (3,) | | COG fit residuals for each band (without isolate mask) |
| `COG_DECREASE_MAX_LEN_NO_ISOLATE` | float32 (3,) | | Maximum consecutive decrease length in COG for each band (without isolate mask) |
| `COG_DECREASE_MAX_MAG_NO_ISOLATE` | float32 (3,) | | Magnitude decrease during maximum consecutive COG decrease (without isolate mask) |
| `APER_CEN_RADEC_ISOLATE` | float32 (2,) | deg | Aperture centroid (RA, Dec) (with isolate mask) |
| `APER_CEN_RADEC_NO_ISOLATE` | float32 (2,) | deg | Aperture centroid (RA, Dec) (without isolate mask) |
| `APER_PARAMS_ISOLATE` | float32 (3,) | pixels, ratio, deg | Aperture parameters: semi-major axis (pixels), *b/a* ratio, position angle (with isolate mask) |
| `APER_PARAMS_NO_ISOLATE` | float32 (3,) | pixels, ratio, deg | Aperture parameters: semi-major axis (pixels), *b/a* ratio, position angle (without isolate mask) |
| `APER_SOURCE_ON_ORG_BLOB` | bool | | Whether DESI source lies on the unsmoothed detection blob |
| `NEAREST_STAR_NORM_DIST` | float32 | | Distance to nearest star in units of the star masking radius |
| `NEAREST_STAR_MAX_MAG` | float32 | mag | Brightest magnitude (across BP, RP, or G) of the nearest star |
| `NUM_TRACTOR_SOURCES_NO_ISOLATE` | int32 | | Number of Tractor sources in parent galaxy (without isolate mask) |
| `NUM_TRACTOR_SOURCES_ISOLATE` | int32 | | Number of Tractor sources in parent galaxy (with isolate mask) |
| `APER_R2_MU_R_ELLIPSE_TRACTOR` | float32 | | *r*-band surface brightness within R2 ellipse (Tractor model) |
| `APER_R2_MU_R_BLOB_TRACTOR` | float32 | | *r*-band surface brightness within segmented blob (Tractor model) |
| `APERFRAC_R4_IN_IMG_DATA_NO_ISOLATE` | float32 | | Fraction of R4 aperture on parent galaxy reconstruction (*g+r+z*) inside image |

</details>

---

<details>
<summary><strong>Extension 6 &mdash; SPECTRA_TEMPLATE</strong></summary>

<br>

Spectral template decomposition coefficients and UMAP coordinates.

| Name | Type | Units | Description |
| :--- | :--- | :---: | :---------- |
| `TARGETID` | int64 | | DESI TARGET ID |
| `NNMF_i` | float32 | | *i*-th NNMF spectral template coefficient (10 columns: `NNMF_0` ... `NNMF_9`) |
| `PCA_i` | float32 | | *i*-th PCA spectral template coefficient (20 columns: `PCA_0` ... `PCA_19`) |
| `SPEC_UMAP_0` | float32 | | Spectra 2D UMAP coordinate (*x*) |
| `SPEC_UMAP_1` | float32 | | Spectra 2D UMAP coordinate (*y*) |
| `NNMF_RESID` | float32 | | Residual norm of NNMF fit to spectra |
| `NNMF_NORM_FACTOR` | float32 | | Normalization factor used before fitting templates |

</details>

</details>

---

<details>
<summary><h3>Image Cutouts</h3></summary>

<br>

We provide 152 x 152 pixel image cutouts for all galaxies in the catalog with *z*-band magnitudes *z* < 20. The image cutouts are stored in an HDF5 (`.h5`) file available at this link. Each image cutout can be matched to a row in the catalog using the `TARGETID` column. Example code demonstrating how to read the HDF5 file and visualize the image cutouts is provided in the tutorials.

For sources identified as shredded, the image cutouts have been recentered on the reconstructed parent galaxy center. The 2048-dimensional representations derived from the self-supervised learning (SSL) model for these image cutouts are available at the same link.

</details>

---

<details>
<summary><h3>Spectra</h3></summary>

<br>

We provide all available DESI spectra for objects in this catalog. The spectra are stored in an HDF5 (`.h5`) file. Each spectrum can be matched to a row in the catalog using the `TARGETID` column. Example code demonstrating how to read the HDF5 file and visualize the spectra is provided in the tutorials.

</details>

---

### Additional Notes

<details>
<summary><strong>DWARF_MASKBIT Descriptions</strong></summary>

<br>

<a name="dwarf_maskbit-descriptions"></a>

Each bit in `DWARF_MASKBIT` corresponds to a quality or cleaning flag. A value of `1 << n` indicates bit `n` is set.

| Bit | Value | Description |
| :-: | ----: | :---------- |
| 0 | 1 | Curve of growth computation failed (NaN values) |
| 1 | 2 | Curve of growth likely not converged (APER R4 - COG > 0.5 mag) |
| 2 | 4 | Large residuals in curve of growth fit |
| 3 | 8 | Curve of growth decreases with increasing aperture |
| 4 | 16 | Large fraction of R4 aperture outside image bounds (>0.25) |
| 5 | 32 | Large fraction of pixels masked within R4 aperture (>0.33) |
| 6 | 64 | Large fraction of pixels in image cutout masked (>0.33) |
| 7 | 128 | Bad $g-r$ or $r-z$ colors (\|color\| > 2) |
| 8 | 256 | Source does not lie on segmented map |
| 9 | 512 | Source is likely shredded ($p_{\rm CNN} > 0.25$) and near bright star |
| 10 | 1024 | Aperture center lies in masked region |
| 11 | 2048 | Large reduced $\chi^2 > 10$ (at least one band) if using original Tractor photometry |
| 12 | 4096 | Source within twice the D26 of an SGA-2020 galaxy at same redshift, but not flagged as SGA-2020 source in Tractor |
| 13 | 8192 | Low signal-to-noise detection (SNR > 5 in only one band or fewer) |
| 14 | 16384 | If `MAG_TYPE = TRACTOR_OG` and `TRACTOR_MASKBITS` has at least one of {2,3,4,8,9} [Tractor bits](https://www.legacysurvey.org/dr9/bitmasks/) flagged |
| 15 | 32768 | Likely incorrect Redrock redshift, based on UMAP analysis |

</details>

<details>
<summary><strong>MAG_TYPE Descriptions</strong></summary>

<br>

<a name="mag-type-descriptions"></a>

| MAG_TYPE | Description |
| :------- | :---------- |
| `TRACTOR_OG` | Original Tractor DR9 photometry |
| `COG` | Remeasured photometry using curve-of-growth approach |
| `SIMPLE` | Remeasured photometry using curve-of-growth approach, but no color-based association criterion is used |
| `TRACTOR_BASED` | Remeasured photometry by summing fluxes of individual, associated Tractor sources |

See Manwadkar et al. 2026a for details.

</details>

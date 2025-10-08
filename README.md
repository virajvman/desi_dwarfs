| Name | Type | Units | Description |
|------|------|-------|-------------|
| TARGETID | int64 |  | DESI TARGET ID |
| SURVEY | str |  | Survey name |
| PROGRAM | str |  | Program name |
| Z | float64 |  | Redrock Redshift |
| DELTACHI2 | float64 |  | Redrock delta-chi-squared |
| ZWARN | int8 |  | Redrock zwarning bit |
| RA_TARGET | float64 | deg | Right Ascension from target catalog |
| DEC_TARGET | float64 | deg | Declination from target catalog |
| DESINAME | str |  | DESI object name |
| DIST_MPC_FIDU | float32 | Mpc | Fiducial luminosity distance in Mpc |
| LOGM_SAGA_FIDU | float32 | $\mathrm{\log(M_\odot)}$ | Log stellar mass using the fiducial luminosity distance and SAGA gr-based approximation |
| LOGM_M24_VCMB | float32 | $\mathrm{\log(M_\odot)}$ | Log stellar mass using the fiducial luminosity distance and de los Reyes et al. 2024 gr-based approximation |
| MAG_G | float32 | $\mathrm{mag}$ | g-band magnitude (MW extinction corrected) |
| MAG_R | float32 | $\mathrm{mag}$ | r-band magnitude (MW extinction corrected) |
| MAG_Z | float32 | $\mathrm{mag}$ | z-band magnitude (MW extinction corrected) |
| SAMPLE | str |  | DESI target class (e.g., BGS_BRIGHT, BGS_FAINT)  |


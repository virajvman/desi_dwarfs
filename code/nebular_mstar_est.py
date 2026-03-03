"""
Test how nebular emission contribution to photometry affects
stellar mass estimates. Stack BGS_BRIGHT dwarf spectra in bins of
HALPHA_EW, then run stackfit to get continuum models and synthetic
photometry for each bin.
"""

import numpy as np
from astropy.table import Table, hstack
from elg_explore import load_spectra, write_stacked_spectra


CATALOG_PATH = "/pscratch/sd/v/virajvm/desi_dwarf_catalogs/dr1/v1.0/desi_dr1_dwarf_catalog.fits"
SPECTRA_PATH = "/pscratch/sd/v/virajvm/catalog_dr1_dwarfs/iron_spectra/spectra_files/desi_y1_dwarf_combine_deredshift_hires.h5"
OUTDIR = "/pscratch/sd/v/virajvm/catalog_dr1_dwarfs/iron_spectra/stack_files/nebular_mstar"
OUTFILE = f"{OUTDIR}/stack_halpha_ew_bins.fits"

N_BINS = 30

def load_bgs_bright_catalog(catalog_path=CATALOG_PATH):
    """Load the dwarf catalog and select clean BGS_BRIGHT galaxies with HALPHA_EW."""
    main_cat = Table.read(catalog_path, hdu="MAIN")
    fspec_cat = Table.read(catalog_path, hdu="FASTSPEC")

    mask = (
        (main_cat["SAMPLE"] == "BGS_BRIGHT") &
        (main_cat["DWARF_MASKBIT"] == 0)
    )

    main_sub = main_cat[mask]
    fspec_sub = fspec_cat[mask]

    catalog = hstack([
        main_sub["TARGETID", "Z", "SAMPLE", "LOG_MSTAR_M24"],
        fspec_sub["HALPHA_EW", "HALPHA_EW_IVAR", "HALPHA_FLUX", "HALPHA_FLUX_IVAR"],
    ])

    ew = np.array(catalog["HALPHA_EW"])
    ew_ivar = np.array(catalog["HALPHA_EW_IVAR"])
    ew_snr = np.abs(ew) * np.sqrt(np.where(ew_ivar > 0, ew_ivar, 0.0))
    valid = np.isfinite(ew) & (ew_ivar > 0) & (ew_snr > 3)
    catalog = catalog[valid]

    print(f"BGS_BRIGHT clean sample with HALPHA_EW SNR > 3: {len(catalog)}")
    return catalog


def define_ew_bins(halpha_ew, n_bins=N_BINS):
    """Split HALPHA_EW into bins using percentiles so each bin has roughly equal counts."""
    percentiles = np.linspace(0, 100, n_bins + 1)
    bin_edges = np.percentile(halpha_ew, percentiles)
    bin_edges[0] = bin_edges[0] - 1e-6
    bin_edges[-1] = bin_edges[-1] + 1e-6

    print(f"HALPHA_EW bin edges ({n_bins} bins):")
    for i in range(n_bins):
        print(f"  Bin {i}: [{bin_edges[i]:.2f}, {bin_edges[i+1]:.2f})")

    return bin_edges


def match_catalog_to_spectra(catalog, spectra_data):
    """Match catalog TARGETIDs to the spectra HDF5 data. Returns matched arrays."""
    cat_tids = np.array(catalog["TARGETID"])
    spec_tids = spectra_data["targetid"]

    spec_id_to_idx = {tid: idx for idx, tid in enumerate(spec_tids)}

    matched_spec_idx = []
    matched_cat_idx = []
    for i, tid in enumerate(cat_tids):
        if tid in spec_id_to_idx:
            matched_spec_idx.append(spec_id_to_idx[tid])
            matched_cat_idx.append(i)

    matched_spec_idx = np.array(matched_spec_idx)
    matched_cat_idx = np.array(matched_cat_idx)

    n_match = len(matched_spec_idx)
    print(f"Matched {n_match}/{len(cat_tids)} catalog entries to spectra")

    fluxes = spectra_data["flux"][matched_spec_idx]
    ivars = spectra_data["flux_ivar"][matched_spec_idx]
    matched_catalog = catalog[matched_cat_idx]

    return fluxes, ivars, matched_catalog


CONT_WINDOW = (5400.0, 5600.0)


def normalize_to_continuum(fluxes, ivars, wave, cont_window=CONT_WINDOW):
    """Normalize each spectrum by its median flux in a line-free continuum window.

    Spectra with non-positive or non-finite continuum levels are flagged invalid.
    Returns normalized flux/ivar and a boolean mask of valid spectra.
    """
    cont_mask = (wave >= cont_window[0]) & (wave <= cont_window[1])
    n_spec = fluxes.shape[0]

    cont_levels = np.zeros(n_spec)
    for i in range(n_spec):
        pixel_vals = fluxes[i, cont_mask]
        finite = np.isfinite(pixel_vals)
        if np.sum(finite) > 0:
            cont_levels[i] = np.median(pixel_vals[finite])
        else:
            cont_levels[i] = np.nan

    valid = np.isfinite(cont_levels) & (cont_levels > 0)

    norm_fluxes = np.zeros_like(fluxes)
    norm_ivars = np.zeros_like(ivars)
    norm_fluxes[valid] = fluxes[valid] / cont_levels[valid, None]
    norm_ivars[valid] = ivars[valid] * cont_levels[valid, None] ** 2

    n_valid = np.sum(valid)
    print(f"    Continuum normalization ({cont_window[0]:.0f}-{cont_window[1]:.0f} A): "
          f"{n_valid}/{n_spec} spectra with valid continuum")

    return norm_fluxes, norm_ivars, valid


def ivar_weighted_stack(fluxes, ivars):
    """Inverse-variance weighted mean stack of spectra.

    Where total ivar is zero, falls back to a simple mean of spectra
    with positive ivar at that pixel.
    """
    weights = ivars.copy()
    weights[~np.isfinite(weights)] = 0.0

    sum_weights = np.sum(weights, axis=0)
    sum_weighted_flux = np.sum(fluxes * weights, axis=0)

    good = sum_weights > 0
    stacked_flux = np.zeros(fluxes.shape[1], dtype=np.float64)
    stacked_ivar = np.zeros(fluxes.shape[1], dtype=np.float64)

    stacked_flux[good] = sum_weighted_flux[good] / sum_weights[good]
    stacked_ivar[good] = sum_weights[good]

    bad = ~good
    if np.any(bad):
        finite_mask = np.isfinite(fluxes[:, bad])
        n_finite = np.sum(finite_mask, axis=0)
        has_data = n_finite > 0
        bad_idx = np.where(bad)[0]
        for j, bj in enumerate(bad_idx):
            if has_data[j]:
                stacked_flux[bj] = np.nanmean(fluxes[:, bj])

    return stacked_flux, stacked_ivar


def stack_in_ew_bins(catalog, fluxes, ivars, wave, bin_edges):
    """Normalize spectra to a common continuum level, then stack per HALPHA_EW bin."""
    n_bins = len(bin_edges) - 1
    n_wave = len(wave)

    print("  Normalizing all spectra by continuum level...")
    norm_fluxes, norm_ivars, cont_valid = normalize_to_continuum(fluxes, ivars, wave)

    all_stacked_flux = np.zeros((n_bins, n_wave), dtype=np.float64)
    all_stacked_ivar = np.zeros((n_bins, n_wave), dtype=np.float64)
    bin_counts = np.zeros(n_bins, dtype=int)
    bin_median_ew = np.zeros(n_bins, dtype=np.float64)

    halpha_ew = np.array(catalog["HALPHA_EW"])

    for i in range(n_bins):
        in_bin = (halpha_ew >= bin_edges[i]) & (halpha_ew < bin_edges[i + 1]) & cont_valid
        n_in_bin = np.sum(in_bin)
        bin_counts[i] = n_in_bin

        if n_in_bin == 0:
            print(f"  Bin {i}: 0 spectra -- skipping")
            continue

        bin_median_ew[i] = np.median(halpha_ew[in_bin])
        sf, si = ivar_weighted_stack(norm_fluxes[in_bin], norm_ivars[in_bin])
        all_stacked_flux[i] = sf
        all_stacked_ivar[i] = si

        print(f"  Bin {i}: {n_in_bin} spectra, median HALPHA_EW = {bin_median_ew[i]:.2f} A")

    return all_stacked_flux, all_stacked_ivar, bin_counts, bin_median_ew


def main():
    import os

    catalog = load_bgs_bright_catalog()

    bin_edges = define_ew_bins(np.array(catalog["HALPHA_EW"]))

    print("\nLoading de-redshifted spectra...")
    spectra_data = load_spectra(SPECTRA_PATH)
    wave = spectra_data["wave_rest"]

    print("\nMatching catalog to spectra...")
    fluxes, ivars, matched_cat = match_catalog_to_spectra(catalog, spectra_data)

    print("\nStacking in HALPHA_EW bins...")
    stacked_flux, stacked_ivar, bin_counts, bin_median_ew = stack_in_ew_bins(
        matched_cat, fluxes, ivars, wave, bin_edges
    )

    os.makedirs(OUTDIR, exist_ok=True)

    n_bins = len(bin_edges) - 1
    stackids = np.arange(n_bins)
    stack_redshift = np.zeros(n_bins)

    ##save the stacked spectra as a numpy array!!
    np.save("/pscratch/sd/v/virajvm/catalog_dr1_dwarfs/iron_spectra/scratch/stacked_flux_ew.npy", stacked_flux)
    np.save("/pscratch/sd/v/virajvm/catalog_dr1_dwarfs/iron_spectra/scratch/stacked_wave_ew.npy", wave)

    table_cols = {
        "NSPEC": bin_counts,
        "HALPHA_EW_LO": bin_edges[:-1],
        "HALPHA_EW_HI": bin_edges[1:],
        "HALPHA_EW_MEDIAN": bin_median_ew,
    }
    table_fmts = {
        "NSPEC": "K",
        "HALPHA_EW_LO": "D",
        "HALPHA_EW_HI": "D",
        "HALPHA_EW_MEDIAN": "D",
    }

    write_stacked_spectra(
        OUTFILE,
        wave,
        stacked_flux,
        stacked_ivar,
        stackids=stackids,
        stack_redshift=stack_redshift,
        table_column_dict=table_cols,
        table_format_dict=table_fmts,
    )

    print(f"\n--- Done! ---")
    print(f"Output: {OUTFILE}")
    print(f"\nTo run fastspecfit on the stacks:")
    print(f"  stackfit {OUTFILE} -o {OUTDIR}/fastspec_stack_halpha_ew_bins.fits --mp 16")


if __name__ == "__main__":
    main()

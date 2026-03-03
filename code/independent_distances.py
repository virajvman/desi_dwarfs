"""
Consolidate distance measurements for the DESI dwarf catalog.

Priority order:
1. NED LVS redshift-independent distances (zIndependent, ziDist > 3, indicator='P')
   → DIST_SOURCE = "NED_ZIND"
2. Mei et al. (2007) SBF distances for Virgo cluster galaxies
   → DIST_SOURCE = "VIRGO_SBF"
3. Kim et al. (2014) Extended Virgo Cluster Catalog members (assigned 16.5 Mpc)
   → DIST_SOURCE = "VIRGO_EVCC"
4. Everything else keeps original LUMI_DIST
   → DIST_SOURCE = "V_CMB"

Usage:
    python consolidate_distances.py
"""

import numpy as np
from astropy.table import Table
from astropy.coordinates import SkyCoord
import astropy.units as u
from astropy.io import ascii as asc
import warnings

# =============================================================================
# Configuration / file paths
# =============================================================================
DESI_CATALOG = "/pscratch/sd/v/virajvm/desi_dwarf_catalogs/dr1/v1.0/desi_dr1_dwarf_catalog.fits"
NED_LVS_CATALOG = "/pscratch/sd/v/virajvm/catalog/NEDLVS_20250602.fits"

# Mei et al. 2007 — SBF distances (table with distance moduli, matched by VCC)
MEI_SBF_FILE = "/pscratch/sd/v/virajvm/catalog/mei_2007_virgo_table1.dat"       # ADJUST PATH
# Cote et al. 2004 — RA/Dec for VCC objects (matched by VCC number to Mei)
COTE_RADEC_FILE = "/pscratch/sd/v/virajvm/catalog/cote_2004_virgo_table1.dat"   # ADJUST PATH

# Kim et al. 2014 — Extended Virgo Cluster Catalog
EVCC_FILE = "/pscratch/sd/v/virajvm/catalog/kim_2014_evcc_catalog.txt"            # ADJUST PATH

OUTPUT_FILE = "/pscratch/sd/v/virajvm/desi_dwarf_catalogs/dr1/v1.0/desi_dr1_dwarf_catalog_distances.fits"

# Default Virgo distance for EVCC members without SBF distances
VIRGO_DEFAULT_DIST_MPC = 16.5


# =============================================================================
# Helper functions
# =============================================================================
def dist_modulus_to_mpc(m_M):
    """Convert distance modulus to distance in Mpc."""
    return 10.0 ** ((m_M - 25.0) / 5.0)


def parse_mei2007(filepath):
    """
    Parse Mei et al. 2007 Table 1 (SBF distances).
    
    Fixed-width format with '|' as separator within some fields.
    Key columns: VCC, (g-z)0, m-M and their errors.
    Some entries have '---' for missing SBF data.
    
    Returns an astropy Table with columns: VCC, dist_mpc
    (only rows with valid distance modulus).
    """
    vcc_list = []
    dist_list = []

    with open(filepath, 'r') as f:
        for line in f:
            line = line.rstrip('\n')
            if len(line) < 40:
                continue
            # Skip comment/header lines
            if line.strip().startswith('#') or line.strip().startswith('-') or line.strip().startswith('Byte'):
                continue

            try:
                # Fixed-width parsing based on byte positions
                # ACSVCS: bytes 1-3, VCC: bytes 5-8
                acsvcs_str = line[0:3].strip()
                vcc_str = line[4:8].strip()

                # m-M: bytes 31-35, but the data uses '|' as delimiter
                # Let's parse more robustly by splitting on whitespace after
                # replacing '|' with space
                cleaned = line.replace('|', ' ')
                parts = cleaned.split()

                if len(parts) < 6:
                    continue

                vcc = int(parts[1])

                # Check if (g-z)0 is '---' meaning no SBF measurement
                gz0_str = parts[2]
                if gz0_str == '---' or gz0_str == '-':
                    continue  # No SBF distance for this galaxy

                # After replacing '|' with space, the layout becomes:
                # parts[0] = ACSVCS
                # parts[1] = VCC
                # parts[2] = (g-z)0 value
                # parts[3] = e_(g-z)0
                # parts[4] = M850
                # parts[5] = e_M850
                # parts[6] = m-M
                # parts[7] = e_m-M
                # parts[8] = BTmag
                # parts[9] = velocity
                # parts[10] = e_velocity
                # ...

                dm_str = parts[6]
                if dm_str == '---' or dm_str == '-':
                    continue

                dm = float(dm_str)
                dist_mpc = dist_modulus_to_mpc(dm)

                vcc_list.append(vcc)
                dist_list.append(dist_mpc)

            except (ValueError, IndexError):
                continue

    mei_tab = Table()
    mei_tab['VCC'] = np.array(vcc_list, dtype=int)
    mei_tab['dist_mpc'] = np.array(dist_list, dtype=float)

    print(f"[Mei et al. 2007] Parsed {len(mei_tab)} galaxies with SBF distances")
    return mei_tab


def parse_cote2004(filepath):
    """
    Parse Cote et al. 2004 Table 1 (RA/Dec for ACSVCS galaxies).
    
    Fixed-width format with RA in HMS and Dec in DMS.
    Returns astropy Table with columns: VCC, RA, DEC (in degrees).
    """
    vcc_list = []
    ra_list = []
    dec_list = []

    with open(filepath, 'r') as f:
        for line in f:
            line = line.rstrip('\n')
            if len(line) < 32:
                continue
            if line.strip().startswith('#') or line.strip().startswith('-') or line.strip().startswith('Byte'):
                continue

            try:
                # Fixed-width columns:
                # Seq: 1-3, VCC: 5-8
                # RAh: 10-11, RAm: 13-14, RAs: 16-20
                # DE-: 22, DEd: 23-24, DEm: 26-27, DEs: 29-32
                seq_str = line[0:3].strip()
                vcc_str = line[4:8].strip()
                rah_str = line[9:11].strip()
                ram_str = line[12:14].strip()
                ras_str = line[15:20].strip()
                de_sign = line[21:22]
                ded_str = line[22:24].strip()
                dem_str = line[25:27].strip()
                des_str = line[28:32].strip()

                vcc = int(vcc_str)
                rah = int(rah_str)
                ram = int(ram_str)
                ras = float(ras_str)
                ded = int(ded_str)
                dem = int(dem_str)
                des = float(des_str)

                # Convert to degrees
                ra_deg = 15.0 * (rah + ram / 60.0 + ras / 3600.0)
                dec_deg = ded + dem / 60.0 + des / 3600.0
                if de_sign == '-':
                    dec_deg = -dec_deg

                vcc_list.append(vcc)
                ra_list.append(ra_deg)
                dec_list.append(dec_deg)

            except (ValueError, IndexError):
                continue

    cote_tab = Table()
    cote_tab['VCC'] = np.array(vcc_list, dtype=int)
    cote_tab['RA'] = np.array(ra_list, dtype=float)
    cote_tab['DEC'] = np.array(dec_list, dtype=float)

    print(f"[Cote et al. 2004] Parsed {len(cote_tab)} galaxies with RA/Dec")
    return cote_tab



def parse_evcc(filepath):
    """
    Parse Kim et al. 2014 Extended Virgo Cluster Catalog.
    
    Fixed-width format. We need RA, Dec, and membership flags.
    Only keep galaxies that are certain members (M) in either the
    infall model (MemIn, Note 2) or the VCC classification (MemVCC, Note 3).
    
    Returns astropy Table with columns: EVCC, RA, DEC, MemIn, MemVCC
    """
    evcc_list = []
    ra_list = []
    dec_list = []
    memin_list = []
    memvcc_list = []
    n_total = 0

    with open(filepath, 'r') as f:
        for line in f:
            line = line.rstrip('\n')
            if len(line) < 92:
                continue
            if line.strip().startswith('#') or line.strip().startswith('-') or line.strip().startswith('Byte'):
                continue

            try:
                # Fixed-width columns from the byte description:
                # EVCC: 1-4, VCC: 6-9
                # RAdeg: 17-24, DEdeg: 26-32
                # MemIn: 74, MemVCC: 76
                evcc_id = int(line[0:4].strip())
                ra_deg = float(line[16:24].strip())
                dec_deg = float(line[25:32].strip())
                mem_in = line[73:74].strip()
                mem_vcc = line[75:76].strip()

                n_total += 1

                # Keep if certain member in EITHER classification:
                #   MemIn  = 'M' (certain member from infall model)
                #   MemVCC = 'M' (certain member from VCC)
                if mem_in == 'M' or mem_vcc == 'M':
                    evcc_list.append(evcc_id)
                    ra_list.append(ra_deg)
                    dec_list.append(dec_deg)
                    memin_list.append(mem_in)
                    memvcc_list.append(mem_vcc)

            except (ValueError, IndexError):
                continue

    evcc_tab = Table()
    evcc_tab['EVCC'] = np.array(evcc_list, dtype=int)
    evcc_tab['RA'] = np.array(ra_list, dtype=float)
    evcc_tab['DEC'] = np.array(dec_list, dtype=float)
    evcc_tab['MemIn'] = memin_list
    evcc_tab['MemVCC'] = memvcc_list

    n_memin = sum(1 for m in memin_list if m == 'M')
    n_memvcc = sum(1 for m in memvcc_list if m == 'M')
    print(f"[Kim et al. 2014 EVCC] Total galaxies in catalog: {n_total}")
    print(f"[Kim et al. 2014 EVCC] Kept {len(evcc_tab)} certain members "
          f"(MemIn=M: {n_memin}, MemVCC=M: {n_memvcc})")
    return evcc_tab



# Speed of light
C_LIGHT_KMS = 2.9979e5   # km/s

# Redshift velocity threshold for CF3_NAM vs V_CMB
V_THRESHOLD_KMS = 2850.0  # km/s

def update_distance_catalog(main_cat_path, keep_lumi_dist_orig=False):
    """
    Consolidate distance measurements for the DESI dwarf catalog.

    Parameters
    ----------
    keep_lumi_dist_orig : bool
        If True, keep the LUMI_DIST_ORIG column in the output catalog.
        If False (default), delete it before returning.

    Returns
    -------
    tot_cat : astropy.table.Table
        The catalog with updated LUMI_DIST and DIST_SOURCE columns.
    """
    # -------------------------------------------------------------------------
    # 1. Load the main DESI dwarf catalog
    # -------------------------------------------------------------------------
    print("=" * 70)
    print("Loading DESI dwarf catalog...")
    tot_cat = Table.read(main_cat_path, hdu="MAIN")
    n_total = len(tot_cat)
    print(f"  Loaded {n_total} galaxies")

    # Find the position of the LUMI_DIST column so we can insert
    # DIST_SOURCE right before it
    lumi_dist_idx = tot_cat.colnames.index('LUMI_DIST_MPC')

    # Initialize DIST_SOURCE column with enough characters for longest label
    dist_source_col = np.full(n_total, 'V_CMB', dtype='U10')
    tot_cat.add_column(dist_source_col, name='DIST_SOURCE', index=lumi_dist_idx)

    # Store original LUMI_DIST for reference
    tot_cat['LUMI_DIST_ORIG'] = tot_cat['LUMI_DIST_MPC'].copy()

    # Build SkyCoord for the main catalog
    cat_coords = SkyCoord(ra=np.array(tot_cat['RA']) * u.deg,
                          dec=np.array(tot_cat['DEC']) * u.deg)

    # Track which galaxies have been assigned a distance from a higher-priority source
    assigned = np.zeros(n_total, dtype=bool)

    # -------------------------------------------------------------------------
    # 2. Cross-match with NED LVS (redshift-independent distances)
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("Step 1: Cross-matching with NED LVS zIndependent distances...")

    ned_cat = Table.read(NED_LVS_CATALOG)
    print(f"  NED LVS full catalog: {len(ned_cat)} entries")

    # Apply filters
    ned_cat = ned_cat[
        (ned_cat["DistMpc_method"] == "zIndependent") &
        (ned_cat["ziDist"] > 3) &
        (ned_cat["ziDist_indicator"] == "P")
    ]
    print(f"  After filtering (zIndependent, ziDist>3, indicator=P): {len(ned_cat)} entries")

    ned_coords = SkyCoord(ra=ned_cat['ra'] * u.deg, dec=ned_cat['dec'] * u.deg)

    # Cross-match: for each galaxy in our catalog, find nearest NED source
    # Uses R50_R as match radius; if multiple NED sources fall within R50_R,
    # match_to_catalog_sky returns the closest one automatically.
    idx_ned, sep_ned, _ = cat_coords.match_to_catalog_sky(ned_coords)

    # Match within R50_R of each galaxy
    r50_all = np.array(tot_cat['R50_R'], dtype=float)  # arcsec
    match_mask_ned = sep_ned.arcsec < r50_all

    n_ned = np.sum(match_mask_ned)
    print(f"  Matched {n_ned} galaxies within their R50_R of a NED LVS source")

    # Update distances
    tot_cat['LUMI_DIST_MPC'][match_mask_ned] = ned_cat['ziDist'][idx_ned[match_mask_ned]]
    tot_cat['DIST_SOURCE'][match_mask_ned] = 'NED_ZIND'
    assigned[match_mask_ned] = True

    # Print some diagnostics
    if n_ned > 0:
        matched_seps = sep_ned[match_mask_ned].arcsec
        print(f"  Separation stats: median={np.median(matched_seps):.2f}\", "
              f"max={np.max(matched_seps):.2f}\"")

    # -------------------------------------------------------------------------
    # 3. Cross-match remaining with Mei et al. 2007 (SBF distances)
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("Step 2: Cross-matching remaining galaxies with Mei et al. 2007 SBF distances...")

    mei_tab = parse_mei2007(MEI_SBF_FILE)
    cote_tab = parse_cote2004(COTE_RADEC_FILE)

    # Merge Mei SBF distances with Cote RA/Dec by VCC number
    # Inner join: only keep VCC numbers present in both tables
    from astropy.table import join
    mei_with_coords = join(mei_tab, cote_tab, keys='VCC', join_type='inner')
    print(f"  Mei+Cote merged table: {len(mei_with_coords)} galaxies with SBF dist + RA/Dec")

    mei_coords = SkyCoord(ra=mei_with_coords['RA'] * u.deg,
                          dec=mei_with_coords['DEC'] * u.deg)

    # For unassigned galaxies, cross-match with Mei catalog
    # Use R50_R (half-light radius) as the match radius for each galaxy
    unassigned_idx = np.where(~assigned)[0]
    cat_coords_remaining = SkyCoord(ra=np.array(tot_cat['RA'][unassigned_idx]) * u.deg,
                                    dec=np.array(tot_cat['DEC'][unassigned_idx]) * u.deg)

    idx_mei, sep_mei, _ = cat_coords_remaining.match_to_catalog_sky(mei_coords)

    # Match within R50_R of each galaxy (converting R50_R from arcsec to angular sep)
    r50_values = np.array(tot_cat['R50_R'][unassigned_idx], dtype=float)  # arcsec
    match_mask_mei = sep_mei.arcsec < r50_values

    n_mei = np.sum(match_mask_mei)
    print(f"  Matched {n_mei} galaxies within their R50_R of a Mei et al. source")

    if n_mei > 0:
        # Map back to original catalog indices
        mei_matched_orig_idx = unassigned_idx[match_mask_mei]
        mei_matched_ref_idx = idx_mei[match_mask_mei]

        tot_cat['LUMI_DIST_MPC'][mei_matched_orig_idx] = mei_with_coords['dist_mpc'][mei_matched_ref_idx]
        tot_cat['DIST_SOURCE'][mei_matched_orig_idx] = 'VIRGO_SBF'
        assigned[mei_matched_orig_idx] = True

        matched_seps_mei = sep_mei[match_mask_mei].arcsec
        print(f"  Separation stats: median={np.median(matched_seps_mei):.2f}\", "
              f"max={np.max(matched_seps_mei):.2f}\"")
        print(f"  Distance range: {np.min(mei_with_coords['dist_mpc'][mei_matched_ref_idx]):.2f} - "
              f"{np.max(mei_with_coords['dist_mpc'][mei_matched_ref_idx]):.2f} Mpc")

    # -------------------------------------------------------------------------
    # 4. Cross-match remaining with EVCC (Kim et al. 2014)
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("Step 3: Cross-matching remaining galaxies with EVCC (Kim et al. 2014)...")

    evcc_tab = parse_evcc(EVCC_FILE)
    evcc_coords = SkyCoord(ra=evcc_tab['RA'] * u.deg, dec=evcc_tab['DEC'] * u.deg)

    # For still-unassigned galaxies
    unassigned_idx2 = np.where(~assigned)[0]
    cat_coords_remaining2 = SkyCoord(ra=np.array(tot_cat['RA'][unassigned_idx2]) * u.deg,
                                     dec=np.array(tot_cat['DEC'][unassigned_idx2]) * u.deg)

    idx_evcc, sep_evcc, _ = cat_coords_remaining2.match_to_catalog_sky(evcc_coords)

    # Match within R50_R of each galaxy
    r50_values2 = np.array(tot_cat['R50_R'][unassigned_idx2], dtype=float)  # arcsec
    match_mask_evcc = sep_evcc.arcsec < r50_values2

    n_evcc = np.sum(match_mask_evcc)
    print(f"  Matched {n_evcc} galaxies within their R50_R of an EVCC member")

    if n_evcc > 0:
        evcc_matched_orig_idx = unassigned_idx2[match_mask_evcc]

        tot_cat['LUMI_DIST_MPC'][evcc_matched_orig_idx] = VIRGO_DEFAULT_DIST_MPC
        tot_cat['DIST_SOURCE'][evcc_matched_orig_idx] = 'VIRGO_EVCC'
        assigned[evcc_matched_orig_idx] = True

        matched_seps_evcc = sep_evcc[match_mask_evcc].arcsec
        print(f"  Separation stats: median={np.median(matched_seps_evcc):.2f}\", "
              f"max={np.max(matched_seps_evcc):.2f}\"")

    # -------------------------------------------------------------------------
    # 5. Assign CF3_NAM vs V_CMB for remaining unmatched galaxies
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("Step 4: Assigning CF3_NAM vs V_CMB for remaining galaxies...")

    # Remaining unassigned galaxies keep their original LUMI_DIST
    # but get DIST_SOURCE based on their redshift:
    #   Z < 2850 km/s / c  →  CF3_NAM
    #   Z >= 2850 km/s / c →  V_CMB
    unassigned_final = ~assigned

    # Compute velocity from redshift: v = Z * c  (Z is dimensionless redshift)
    z_values = np.array(tot_cat['Z'], dtype=float)  # dimensionless redshift
    v_values = z_values * C_LIGHT_KMS  # km/s

    cf3_mask = unassigned_final & (v_values < V_THRESHOLD_KMS)
    vcmb_mask = unassigned_final & (v_values >= V_THRESHOLD_KMS)

    tot_cat['DIST_SOURCE'][cf3_mask] = 'CF3_NAM'
    tot_cat['DIST_SOURCE'][vcmb_mask] = 'V_CMB'

    n_cf3 = np.sum(cf3_mask)
    n_vcmb = np.sum(vcmb_mask)
    print(f"  CF3_NAM (v < {V_THRESHOLD_KMS:.0f} km/s): {n_cf3} galaxies")
    print(f"  V_CMB   (v >= {V_THRESHOLD_KMS:.0f} km/s): {n_vcmb} galaxies")

    # -------------------------------------------------------------------------
    # 6. Summary
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    for source in ['NED_ZIND', 'VIRGO_SBF', 'VIRGO_EVCC', 'CF3_NAM', 'V_CMB']:
        n = np.sum(tot_cat['DIST_SOURCE'] == source)
        if n > 0:
            print(f"  {source:12s}: {n:6d} galaxies")

    n_changed = np.sum(tot_cat['LUMI_DIST_MPC'] != tot_cat['LUMI_DIST_ORIG'])
    print(f"\n  Total galaxies with updated distances: {n_changed}")
    print(f"  Total galaxies unchanged:              {n_total - n_changed}")

    # Optionally remove the backup column
    if not keep_lumi_dist_orig:
        del tot_cat['LUMI_DIST_ORIG']
        print("  Removed LUMI_DIST_ORIG column")
    else:
        print("  Kept LUMI_DIST_ORIG column")

    print("Done!")
    return tot_cat, ned_cat, mei_with_coords, evcc_tab


    

# if __name__ == "__main__":
    # Run the demo to sanity-check the Virgo membership function
    # demo_virgo_membership()

    # Uncomment below to run the full distance consolidation pipeline
    # main()
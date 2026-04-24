'''
These are functions used to flag objects with outlier k corrections
'''

import numpy as np
from scipy.interpolate import interp1d
import pickle

def build_flag_interpolators(bin_cents, sig2_low, sig2_high):
    """
    Build interpolators for the 95% confidence interval bounds.
    Uses linear interpolation within range, constant extrapolation outside.
    
    Parameters
    ----------
    bin_cents : array-like
        Redshift bin centers where percentiles were computed.
    sig2_low : array-like
        Lower 2.5th percentile values at each bin center.
    sig2_high : array-like
        Upper 97.5th percentile values at each bin center.
    
    Returns
    -------
    dict with 'low' and 'high' interp1d functions.
    """
    bin_cents = np.asarray(bin_cents)
    sig2_low = np.asarray(sig2_low)
    sig2_high = np.asarray(sig2_high)
    
    # Sort by z just to be safe
    order = np.argsort(bin_cents)
    bin_cents = bin_cents[order]
    sig2_low = sig2_low[order]
    sig2_high = sig2_high[order]

    #we can smooth this now!
    # from scipy.signal import savgol_filter
    # sig2_low = savgol_filter(sig2_low, window_length=3, polyorder=2)
    # sig2_high = savgol_filter(sig2_high, window_length=3, polyorder=2)
    
    # Constant extrapolation: fill_value uses edge values outside bounds
    low_interp = interp1d(
        bin_cents, sig2_low,
        kind='linear',
        bounds_error=False,
        fill_value=(sig2_low[0], sig2_low[-1])
    )
    high_interp = interp1d(
        bin_cents, sig2_high,
        kind='linear',
        bounds_error=False,
        fill_value=(sig2_high[0], sig2_high[-1])
    )
    
    return {
        'low': low_interp,
        'high': high_interp,
        'z_min': bin_cents[0],
        'z_max': bin_cents[-1],
    }


def save_flag_interpolators(filepath, elg_g, elg_r, bgs_g, bgs_r, 
                             abs_threshold=0.15):
    """
    Save interpolator data to a pickle file.
    We save the raw arrays (not the interp1d objects) for portability,
    and reconstruct interpolators on load.
    """
    data = {
        'abs_threshold': abs_threshold,
        'elg': {
            'g': {
                'bin_cents': np.asarray(elg_g['bin_cents']),
                'sig2_low': np.asarray(elg_g['sig2_low']),
                'sig2_high': np.asarray(elg_g['sig2_high']),
            },
            'r': {
                'bin_cents': np.asarray(elg_r['bin_cents']),
                'sig2_low': np.asarray(elg_r['sig2_low']),
                'sig2_high': np.asarray(elg_r['sig2_high']),
            },
        },
        'bgs': {
            'g': {
                'bin_cents': np.asarray(bgs_g['bin_cents']),
                'sig2_low': np.asarray(bgs_g['sig2_low']),
                'sig2_high': np.asarray(bgs_g['sig2_high']),
            },
            'r': {
                'bin_cents': np.asarray(bgs_r['bin_cents']),
                'sig2_low': np.asarray(bgs_r['sig2_low']),
                'sig2_high': np.asarray(bgs_r['sig2_high']),
            },
        },
    }
    with open(filepath, 'wb') as f:
        pickle.dump(data, f)
    print(f"Saved flag contours to {filepath}")


def load_flag_interpolators(filepath):
    """
    Load and reconstruct interpolators.
    Returns a dict with structure {population: {band: interpolator_dict}}.
    """
    with open(filepath, 'rb') as f:
        data = pickle.load(f)
    
    interpolators = {'abs_threshold': data['abs_threshold']}
    for pop in ['elg', 'bgs']:
        interpolators[pop] = {}
        for band in ['g', 'r']:
            d = data[pop][band]
            interpolators[pop][band] = build_flag_interpolators(
                d['bin_cents'], d['sig2_low'], d['sig2_high']
            )
    return interpolators


def flag_kcorr_outliers(z, delta_k_g, delta_k_r, is_elg, interpolators):
    """
    Flag sources whose K-correction residuals fall outside the 95% 
    confidence interval AND exceed the absolute threshold.
    
    Parameters
    ----------
    z : array-like
        Redshifts.
    delta_k_g, delta_k_r : array-like
        K_you - K_C10 residuals in g and r bands.
    is_elg : array-like of bool
        True for ELG, False for BGS/non-ELG.
    interpolators : dict
        Output of load_flag_interpolators.
    
    Returns
    -------
    flag_g, flag_r : arrays of bool
        True where the source is flagged as unreliable.
    """
    z = np.asarray(z)
    delta_k_g = np.asarray(delta_k_g)
    delta_k_r = np.asarray(delta_k_r)
    is_elg = np.asarray(is_elg, dtype=bool)
    
    abs_thresh = interpolators['abs_threshold']
    
    # Evaluate bounds for each galaxy based on its population
    low_g = np.where(is_elg,
                     interpolators['elg']['g']['low'](z),
                     interpolators['bgs']['g']['low'](z))
    high_g = np.where(is_elg,
                      interpolators['elg']['g']['high'](z),
                      interpolators['bgs']['g']['high'](z))
    low_r = np.where(is_elg,
                     interpolators['elg']['r']['low'](z),
                     interpolators['bgs']['r']['low'](z))
    high_r = np.where(is_elg,
                      interpolators['elg']['r']['high'](z),
                      interpolators['bgs']['r']['high'](z))
    
    # Outside 95% interval
    outside_interval_g = (delta_k_g < low_g) | (delta_k_g > high_g)
    outside_interval_r = (delta_k_r < low_r) | (delta_k_r > high_r)
    
    # Beyond absolute threshold
    beyond_abs_g = np.abs(delta_k_g) > abs_thresh
    beyond_abs_r = np.abs(delta_k_r) > abs_thresh
    
    # Flag: both conditions must be true
    flag_g = outside_interval_g & beyond_abs_g
    flag_r = outside_interval_r & beyond_abs_r
    
    return flag_g, flag_r


# save_flag_interpolators(
# '/global/homes/v/virajvm/DESI2_LOWZ/desi_dwarfs/data/kcorr_flag_contours.pkl',
# elg_g=all_conts_g[0], elg_r=all_conts_r[0],
# bgs_g=all_conts_g[1], bgs_r=all_conts_r[1],
# abs_threshold=0.15,
# )

# # Step 2: In your bitmask pipeline, load and apply
# interp = load_flag_interpolators('/global/homes/v/virajvm/DESI2_LOWZ/desi_dwarfs/data/kcorr_flag_contours.pkl')

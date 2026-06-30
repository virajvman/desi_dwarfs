"""
Catalog-paper reference figure: emission lines moving in and out of the DECam grz
bands as a dwarf is redshifted, illustrating why the ELG n(z) peaks at z ~ 0.1.

The scientific content of the figure is the LINE POSITIONS relative to the filters,
which are exact and set purely by redshift -- the spectrum's continuum is only visual
scaffolding.  We therefore use a simple strong-line FSPS mock SED (not a tuned model
of the real population); the panel shows the SAME rest-frame SED redshifted to two z
that bracket the n(z) peak (z=0.08, lines in g; z=0.12, [OIII] in the g/r gap).

Two-part workflow
-----------------
1. COMPUTE (local; needs FSPS + speclite):
       python elg_oiii_filter_figure.py
   Builds the mock SED, grabs the grz transmission curves, dumps both to a .npz in
   data/, and writes a local preview PNG next to this script.

2. PLOT (anywhere, incl. NERSC; needs only numpy + matplotlib):
       prod = load_products()
       plot_spectra_filters(prod, axes)
   The plotting function takes ONLY the loaded arrays -- no FSPS / speclite -- so the
   identical code renders in the NERSC paper notebook under the paper style.

The .npz is committed to data/ so it travels to NERSC via git.
"""
import os
import numpy as np
import matplotlib.pyplot as plt

# --------------------------------------------------------------------------
# Paths / constants
# --------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
NPZ_PATH = os.path.join(REPO_ROOT, "data", "elg_oiii_filter_products.npz")

FILTER_SET = "decam2014"
EX_Z = (0.08, 0.12)        # bracket the n(z) peak: [OIII] in g vs in the g/r gap

# rest-frame vacuum wavelengths (Angstrom) to annotate
MARK_LINES = {
    "[OII]":      3728.5,
    r"H$\beta$":  4862.7,
    "[OIII]4959": 4960.3,
    "[OIII]5007": 5008.2,
    r"H$\alpha$": 6564.6,
}
BAND_COLORS = {"g": "tab:green", "r": "tab:red", "z": "tab:purple"}


# --------------------------------------------------------------------------
# COMPUTE  (local only -- requires fsps + speclite)
# --------------------------------------------------------------------------
def build_mock_sed(logzsol=-1.0, gas_logz=-1.0, gas_logu=-2.0, tage=0.3):
    """Rest-frame mock dwarf SED (vacuum AA, f_lambda arb. units) from FSPS.

    Young, metal-poor, high-ionization, dust-free constant-SFH population with
    nebular emission -> strong [OIII] complex.  Chosen for clear, strong lines;
    NOT a fit to the real population (see module docstring).
    """
    import fsps
    sp = fsps.StellarPopulation(
        zcontinuous=1, sfh=1, const=1.0, add_neb_emission=True,
        logzsol=logzsol, gas_logz=gas_logz, gas_logu=gas_logu,
        dust_type=0, dust2=0.0)
    wave, flam = sp.get_spectrum(tage=tage, peraa=True)   # rest-frame, Lsun/AA
    return np.asarray(wave, float), np.asarray(flam, float)


def get_filter_curves(filter_set=FILTER_SET):
    """{band: (wavelength, response)} for g/r/z, so plotting needs no speclite."""
    import speclite.filters
    grz = speclite.filters.load_filters(
        f"{filter_set}-g", f"{filter_set}-r", f"{filter_set}-z")
    return {band: (np.asarray(f.wavelength, float), np.asarray(f.response, float))
            for f, band in zip(grz, ["g", "r", "z"])}


def compute_and_save(npz_path=NPZ_PATH):
    """Build the SED + filter curves and dump all arrays to .npz."""
    wave_rest, flam_rest = build_mock_sed()
    keep = (wave_rest > 2500) & (wave_rest < 10000)   # optical window we plot
    filt = get_filter_curves()

    payload = dict(
        wave_rest=wave_rest[keep], flam_rest=flam_rest[keep],
        ex_z=np.asarray(EX_Z, float),
        line_names=np.array(list(MARK_LINES.keys()), dtype=object),
        line_waves=np.asarray(list(MARK_LINES.values()), float),
        filt_g_wave=filt["g"][0], filt_g_resp=filt["g"][1],
        filt_r_wave=filt["r"][0], filt_r_resp=filt["r"][1],
        filt_z_wave=filt["z"][0], filt_z_resp=filt["z"][1],
    )
    os.makedirs(os.path.dirname(npz_path), exist_ok=True)
    np.savez(npz_path, **payload)
    print(f"wrote {npz_path}  ({len(payload)} arrays)")
    return payload


# --------------------------------------------------------------------------
# LOAD + PLOT  (runs anywhere -- numpy + matplotlib only)
# --------------------------------------------------------------------------
def load_products(npz_path=NPZ_PATH):
    """Load the .npz into a plain dict of arrays for the plotting function."""
    d = np.load(npz_path, allow_pickle=True)
    prod = {k: d[k] for k in d.files}
    prod["filters"] = {
        "g": (prod["filt_g_wave"], prod["filt_g_resp"]),
        "r": (prod["filt_r_wave"], prod["filt_r_resp"]),
        "z": (prod["filt_z_wave"], prod["filt_z_resp"]),
    }
    return prod


def _overlay_filters(ax, prod, alpha=0.12, label_bands=True):
    """Draw g/r/z response on a 0..1 axis from stored curves (no speclite)."""
    for band in ["g", "r", "z"]:
        w, resp = prod["filters"][band]
        ax.plot(w, resp, color=BAND_COLORS[band], lw=1.2, zorder=1)
        ax.fill_between(w, resp, color=BAND_COLORS[band], alpha=alpha, lw=0, zorder=0)
        if label_bands:
            ax.text(w[np.argmax(resp)], 1.03, band, color=BAND_COLORS[band],
                    ha="center", va="bottom", fontsize=11)
    ax.set_ylim(0, 1.18)


def plot_spectra_filters(prod, axes=None, xlim=(3600, 9900),
                         rest_norm=(4150.0, 4350.0), ymax=8.0):
    """Stacked panels: the SAME mock SED redshifted to each ex_z, over grz filters.

    `axes` : sequence of len(ex_z) axes (e.g. from make_subplots). If None, makes its
             own figure.  Hα is allowed to run off the top (ymax) so the [OIII]
             complex stays readable.  Returns the axes.
    """
    ex_z = np.atleast_1d(prod["ex_z"])
    if axes is None:
        _, axes = plt.subplots(len(ex_z), 1, figsize=(7.5, 3.0 * len(ex_z)),
                               sharex=True)
    axes = np.atleast_1d(axes)
    w0, f0 = prod["wave_rest"], prod["flam_rest"]

    for ax, z in zip(axes, ex_z):
        wobs = w0 * (1.0 + z)
        fobs = f0 / (1.0 + z)
        sel = (wobs > rest_norm[0] * (1 + z)) & (wobs < rest_norm[1] * (1 + z))
        norm = np.median(fobs[sel]) if np.any(sel) else 1.0

        axf = ax.twinx()                 # filters behind the spectrum
        _overlay_filters(axf, prod)
        axf.set_zorder(ax.get_zorder() - 1)
        ax.patch.set_visible(False)
        axf.set_ylabel("filter response")

        ax.plot(wobs, fobs / norm, color="k", lw=0.7, zorder=5)
        for i, (name, lam0) in enumerate(zip(prod["line_names"], prod["line_waves"])):
            lam = float(lam0) * (1.0 + z)
            ax.axvline(lam, color="0.5", ls=":", lw=0.8, zorder=2)
            # small alternating vertical offset so adjacent labels don't collide
            ytxt = 0.97 if i % 2 == 0 else 0.78
            ax.text(lam, ytxt, str(name), rotation=90, va="top", ha="right",
                    fontsize=7, color="0.35", transform=ax.get_xaxis_transform())

        ax.set_ylabel(r"$f_\lambda$ (cont.-norm.)")
        ax.text(0.015, 0.88, fr"$z={z:.2f}$", transform=ax.transAxes, fontsize=12)
        ax.set_ylim(0, ymax)
        ax.set_xlim(*xlim)              # every panel (make_subplots has no sharex)

    # only the last panel keeps the x-axis label + tick labels
    for ax in axes[:-1]:
        ax.set_xticklabels([])
    axes[-1].set_xlabel(r"observed wavelength [$\AA$]")
    return axes


def make_preview(prod, outdir=_HERE):
    """Local sanity PNG for the spectra panel."""
    fig, axes = plt.subplots(len(np.atleast_1d(prod["ex_z"])), 1,
                             figsize=(7.5, 6.0), sharex=True)
    plot_spectra_filters(prod, axes=axes)
    fig.tight_layout()
    p = os.path.join(outdir, "preview_spectra_filters.png")
    fig.savefig(p, dpi=110)
    print(f"wrote {p}")


if __name__ == "__main__":
    compute_and_save()
    make_preview(load_products())

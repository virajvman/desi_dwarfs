'''
Stage 7: Validate the purity + completeness models.

Five checks (all figures -> config.VALIDATION_DIR):
  1. recovery maps      predicted vs simulated purity/completeness fractions
                        in bins of rfib x Dchi2, split by emission stratum
                        (held-out mocks only; like slides 25-26 but 2D).
  2. reliability        per-object probability calibration curves on the
                        held-out mocks -- this is what the weights actually
                        use, binned maps alone do not test it.
  3. closed loop        apply W_spec = p/c to the held-out mocks passing the
                        catalog selection and check the weighted counts
                        recover the TRUE input population in rfib x stratum
                        bins (end-to-end estimator test).
  4. SV3 anchor         empirical purity vs Dchi2 from the SV3 repeat/deep
                        truth table (catastrophic threshold 0.0033, same as
                        everywhere) compared against the mock-population
                        purity in matching rfib bins.
  5. distributions      simulated vs real log10(Dchi2) vs emission proxy /
                        rfib loci for BGS-Bright-like and LOWZ-Dark-like
                        depths (gate for the feasibgs noise model).

Runs on a login node (needs the training table, models, SV3 file, reference
catalog).
'''

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from astropy.table import Table
from joblib import load

import config
import specz_utils
from measure_features import TRAINING_TABLE

RFIB_BINS = np.arange(19.0, 23.75, 0.25)
DCHI2_BINS = np.arange(0.0, 4.5, 0.25)   # log10
PROB_BINS = np.linspace(0, 1, 21)


def load_heldout(tab, bundle):
    mask = np.isin(np.asarray(tab["MOCKID"]), bundle["test_mockids"])
    return tab[mask]


def binned_fraction(x, y, values, xbins, ybins, min_count=10):
    n, _, _ = np.histogram2d(x, y, bins=[xbins, ybins])
    s, _, _ = np.histogram2d(x, y, bins=[xbins, ybins], weights=values)
    frac = np.where(n >= min_count, s / np.clip(n, 1, None), np.nan)
    return frac


# ----------------------------------------------------------------------------
# 1. Recovery maps
# ----------------------------------------------------------------------------
def recovery_maps(tab, bundle, label):
    sub = load_heldout(tab, bundle)
    if label == "purity":
        sub = sub[np.asarray(sub["IN_CATALOG"], dtype=bool)]
    X = specz_utils.assemble_features(sub)
    pred = specz_utils.predict_proba(bundle["scaler"], bundle["clf"], X)
    y = np.asarray(sub["IS_CORRECT"], dtype=float)
    rfib = np.asarray(sub["RFIB_ASINH"])
    ldchi2 = np.asarray(sub["LOG_DELTACHI2"])
    strata = np.asarray(sub["STRATUM"])

    strata_sets = [("all", np.ones(len(sub), bool)),
                   ("passive (stratum 0)", strata == 0),
                   ("strong emission (top stratum)", strata == strata.max())]

    fig, axes = plt.subplots(len(strata_sets), 3,
                             figsize=(16, 4.5 * len(strata_sets)),
                             layout="constrained")
    for row, (name, m) in enumerate(strata_sets):
        sim = binned_fraction(rfib[m], ldchi2[m], y[m], RFIB_BINS, DCHI2_BINS)
        mod = binned_fraction(rfib[m], ldchi2[m], pred[m], RFIB_BINS, DCHI2_BINS)
        for col, (arr, title) in enumerate([(sim, "simulated"), (mod, "model"),
                                            (mod - sim, "model - sim")]):
            ax = axes[row, col]
            vlim = dict(vmin=0, vmax=1, cmap="Spectral") if col < 2 else \
                dict(vmin=-0.2, vmax=0.2, cmap="coolwarm")
            im = ax.imshow(arr.T, origin="lower", aspect="auto",
                           extent=[RFIB_BINS[0], RFIB_BINS[-1],
                                   DCHI2_BINS[0], DCHI2_BINS[-1]], **vlim)
            fig.colorbar(im, ax=ax)
            ax.set_title(f"{name}: {title}")
            ax.set_xlabel("rfib (asinh)")
            ax.set_ylabel(r"log10 $\Delta\chi^2$")
    fig.suptitle(f"{label} fraction: simulation vs model (held-out)")
    plt.savefig(f"{config.VALIDATION_DIR}/recovery_maps_{label}.pdf",
                bbox_inches="tight")
    plt.close()


# ----------------------------------------------------------------------------
# 2. Reliability (calibration) diagrams
# ----------------------------------------------------------------------------
def reliability(tab, bundle, label, ax):
    sub = load_heldout(tab, bundle)
    if label == "purity":
        sub = sub[np.asarray(sub["IN_CATALOG"], dtype=bool)]
    X = specz_utils.assemble_features(sub)
    pred = specz_utils.predict_proba(bundle["scaler"], bundle["clf"], X)
    y = np.asarray(sub["IS_CORRECT"], dtype=float)

    centers = 0.5 * (PROB_BINS[1:] + PROB_BINS[:-1])
    emp, err = np.full(len(centers), np.nan), np.full(len(centers), np.nan)
    for i in range(len(centers)):
        m = (pred >= PROB_BINS[i]) & (pred < PROB_BINS[i + 1])
        if m.sum() >= 20:
            f = y[m].mean()
            emp[i], err[i] = f, np.sqrt(f * (1 - f) / m.sum())
    ax.errorbar(centers, emp, yerr=err, fmt="o-", label=label)
    ax.plot([0, 1], [0, 1], "k:", lw=1)
    ax.set_xlabel("predicted P(correct)")
    ax.set_ylabel("empirical fraction correct")
    ax.legend()


# ----------------------------------------------------------------------------
# 3. Closed-loop estimator test
# ----------------------------------------------------------------------------
def closed_loop(tab, purity_bundle, completeness_bundle):
    """Weighted mock 'catalog' counts vs true input counts, rfib x stratum."""
    heldout = np.isin(np.asarray(tab["MOCKID"]), completeness_bundle["test_mockids"])
    full = tab[heldout]                       # the true z<0.15 population
    catalog = full[np.asarray(full["IN_CATALOG"], dtype=bool)]

    Xc = specz_utils.assemble_features(catalog)
    p = specz_utils.predict_proba(purity_bundle["scaler"], purity_bundle["clf"], Xc)
    c = specz_utils.predict_proba(completeness_bundle["scaler"],
                                  completeness_bundle["clf"], Xc)
    w = p / np.clip(c, 0.02, None)

    strata = np.unique(np.asarray(full["STRATUM"]))
    fig, axes = plt.subplots(1, len(strata), figsize=(4.5 * len(strata), 4.5),
                             layout="constrained", sharey=False)
    for ax, s in zip(np.atleast_1d(axes), strata):
        # truth binned by TRUE rfib target; catalog binned by measured rfib
        m_true = np.asarray(full["STRATUM"]) == s
        m_cat = np.asarray(catalog["STRATUM"]) == s
        n_true, _ = np.histogram(np.asarray(full["RFIB_TARGET"])[m_true],
                                 bins=RFIB_BINS)
        n_raw, _ = np.histogram(np.asarray(catalog["RFIB_ASINH"])[m_cat],
                                bins=RFIB_BINS)
        n_wgt, _ = np.histogram(np.asarray(catalog["RFIB_ASINH"])[m_cat],
                                bins=RFIB_BINS, weights=w[m_cat])
        cen = 0.5 * (RFIB_BINS[1:] + RFIB_BINS[:-1])
        ax.plot(cen, n_true, "k-", label="true input")
        ax.plot(cen, n_raw, "C3--", label="raw catalog")
        ax.plot(cen, n_wgt, "C0-", label="W_spec-weighted")
        ax.set_title(f"stratum {s}")
        ax.set_xlabel("rfib")
        ax.set_ylabel("counts")
        ax.legend(fontsize=8)
    fig.suptitle("Closed-loop estimator test (held-out mocks)")
    plt.savefig(f"{config.VALIDATION_DIR}/closed_loop_test.pdf",
                bbox_inches="tight")
    plt.close()


# ----------------------------------------------------------------------------
# 4. SV3 external anchor
# ----------------------------------------------------------------------------
def sv3_anchor(tab):
    if not os.path.exists(config.SV3_TRUTH_FILE):
        print("SV3 truth file not found -- skipping anchor check")
        return
    sv3 = Table.read(config.SV3_TRUTH_FILE)
    z_true = np.asarray(sv3["Z_TRUE"], dtype=float)
    z_rr = np.asarray(sv3["Z"], dtype=float)
    dchi2 = np.asarray(sv3["DELTACHI2"], dtype=float)
    low = (z_rr > config.CATALOG_Z_MIN) & (z_rr < config.CATALOG_Z_MAX)
    correct = ~specz_utils.catastrophic_z(z_true, z_rr)

    cat_mask = np.asarray(tab["IN_CATALOG"], dtype=bool)
    mock = tab[cat_mask]

    fig, ax = plt.subplots(figsize=(6, 5), layout="constrained")
    bins = 10 ** np.arange(0.5, 4.5, 0.25)
    cen = np.sqrt(bins[1:] * bins[:-1])
    for data_dchi2, data_ok, name, color in [
            (dchi2[low], correct[low], "SV3 truth", "k"),
            (np.asarray(mock["DELTACHI2"]), np.asarray(mock["IS_CORRECT"], bool),
             "mocks (in catalog)", "C0")]:
        frac = np.full(len(cen), np.nan)
        for i in range(len(cen)):
            m = (data_dchi2 >= bins[i]) & (data_dchi2 < bins[i + 1])
            if m.sum() >= 10:
                frac[i] = data_ok[m].mean()
        ax.plot(cen, frac, "o-", color=color, label=name)
    ax.set_xscale("log")
    ax.set_xlabel(r"$\Delta\chi^2$")
    ax.set_ylabel("purity fraction (z correct)")
    ax.set_title(f"SV3 anchor (threshold {config.CATASTROPHIC_THRESHOLD})")
    ax.legend()
    ax.grid(ls=":", alpha=0.5)
    plt.savefig(f"{config.VALIDATION_DIR}/sv3_anchor.pdf", bbox_inches="tight")
    plt.close()
    print("NOTE: SV3 sample is BGS Bright depth with its own (rfib, type) mix; "
          "compare shapes per rfib bin, not the absolute marginal, before "
          "reading too much into offsets.")


# ----------------------------------------------------------------------------
# 5. Simulated vs real Dchi2-feature distributions
# ----------------------------------------------------------------------------
def distribution_match(tab):
    ref = Table.read(config.REFERENCE_CATALOG)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), layout="constrained")

    axes[0].hexbin(np.asarray(tab["RFIB_ASINH"]),
                   np.asarray(tab["LOG_DELTACHI2"]), gridsize=60, bins="log",
                   extent=[19, 23.5, 0, 4.5], cmap="Blues")
    axes[0].set_title("mocks")
    good = (np.asarray(ref["DELTACHI2"]) > 0)
    axes[1].hexbin(np.asarray(ref["RFIB_SYNTH"])[good],
                   np.log10(np.asarray(ref["DELTACHI2"])[good]), gridsize=60,
                   bins="log", extent=[19, 23.5, 0, 4.5], cmap="Reds")
    axes[1].set_title("real reference sample (hi-SNR selected!)")
    for ax in axes:
        ax.set_xlabel("rfib (synthetic)")
        ax.set_ylabel(r"log10 $\Delta\chi^2$")
    plt.savefig(f"{config.VALIDATION_DIR}/dchi2_rfib_distributions.pdf",
                bbox_inches="tight")
    plt.close()
    print("NOTE: the real panel carries the reference-sample Dchi2>100 "
          "selection; for the definitive noise-model gate use the "
          "calibrate_efftime.py comparison on unselected re-observations.")


if __name__ == "__main__":
    specz_utils.check_dirs()
    tab = Table.read(TRAINING_TABLE)
    purity_bundle = load(config.PURITY_MODEL_JOBLIB)
    completeness_bundle = load(config.COMPLETENESS_MODEL_JOBLIB)

    print("1/5 recovery maps ...")
    recovery_maps(tab, purity_bundle, "purity")
    recovery_maps(tab, completeness_bundle, "completeness")

    print("2/5 reliability diagrams ...")
    fig, ax = plt.subplots(figsize=(6, 6), layout="constrained")
    reliability(tab, purity_bundle, "purity", ax)
    reliability(tab, completeness_bundle, "completeness", ax)
    plt.savefig(f"{config.VALIDATION_DIR}/reliability_diagrams.pdf",
                bbox_inches="tight")
    plt.close()

    print("3/5 closed-loop estimator test ...")
    closed_loop(tab, purity_bundle, completeness_bundle)

    print("4/5 SV3 anchor ...")
    sv3_anchor(tab)

    print("5/5 distribution matching ...")
    distribution_match(tab)

    print(f"\nAll figures in {config.VALIDATION_DIR}")

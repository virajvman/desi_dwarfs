'''
Stage 5: Train the purity and completeness MLP classifiers.

Two models with IDENTICAL features but different training populations:
  purity        mocks passing the exact dwarf-catalog selection (IN_CATALOG),
                i.e. P(correct z | features, entered the catalog) -- the
                per-object down-weight.
  completeness  the FULL z_true < 0.15 mock population including all
                failures (load-bearing condition; see README) -- 1/P is the
                per-object up-weight.

A held-out test split (1 - TRAIN_FRACTION) is reserved and its indices
stored with the model so validation.py can evaluate calibration on data the
model never saw. Outputs: joblib bundles {scaler, clf, feature_names, meta}.

Runs fine on a login node or locally if the training table is synced.
'''

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from astropy.table import Table
from joblib import dump

import config
import specz_utils
from measure_features import TRAINING_TABLE


def train_one(tab, mask, label, out_path, rng):
    sub = tab[mask]
    X = specz_utils.assemble_features(sub)
    y = np.asarray(sub["IS_CORRECT"], dtype=int)

    n = len(y)
    idx = rng.permutation(n)
    n_train = int(config.TRAIN_FRACTION * n)
    tr, te = idx[:n_train], idx[n_train:]

    print(f"\n=== {label} model: {n} objects "
          f"({y.mean():.3f} correct), {n_train} train / {n - n_train} test ===")
    scaler, clf = specz_utils.train_mlp_classifier(X[tr], y[tr])

    # quick held-out summary
    from sklearn.metrics import roc_auc_score, brier_score_loss
    p_test = specz_utils.predict_proba(scaler, clf, X[te])
    print(f"{label}: held-out AUC = {roc_auc_score(y[te], p_test):.4f}, "
          f"Brier = {brier_score_loss(y[te], p_test):.4f}")

    bundle = {
        "scaler": scaler,
        "clf": clf,
        "feature_names": list(config.FEATURE_NAMES),
        "label": label,
        "catastrophic_threshold": config.CATASTROPHIC_THRESHOLD,
        "training_table": TRAINING_TABLE,
        "population_mask": ("IN_CATALOG" if label == "purity"
                            else "ALL z_true < 0.15 mocks"),
        # MOCKIDs of the held-out test set, for validation.py
        "test_mockids": np.asarray(sub["MOCKID"])[te],
    }
    dump(bundle, out_path)
    print(f"saved -> {out_path}")


if __name__ == "__main__":
    specz_utils.check_dirs()
    rng = np.random.default_rng(config.MLP_SEED)

    tab = Table.read(TRAINING_TABLE)
    print(f"Training table: {len(tab)} mocks")

    # completeness: FULL population (all mocks have z_true < 0.15 by design)
    train_one(tab, np.ones(len(tab), dtype=bool), "completeness",
              config.COMPLETENESS_MODEL_JOBLIB, rng)

    # purity: catalog-selected population only
    train_one(tab, np.asarray(tab["IN_CATALOG"], dtype=bool), "purity",
              config.PURITY_MODEL_JOBLIB, rng)

    print("\nNOTE (flagged assumption): efftime/TSNR2 is deliberately NOT a "
          "feature; Dchi2 is the depth proxy. Re-train with it added and "
          "compare AUC/calibration to confirm this holds.")

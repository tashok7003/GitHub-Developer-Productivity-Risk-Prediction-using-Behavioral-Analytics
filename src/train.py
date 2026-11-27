"""Train and evaluate the GitHub Developer Visibility Risk model.

Usage:
    python src/train.py                       # uses dataset/developers_features_engineered.csv
    python src/train.py --data path/to.csv

Outputs:
    models/visibility_risk_model.joblib   final pipeline + metadata
    reports/metrics.json                  CV + test metrics for every model
    reports/*.png                         confusion matrix, feature importance, SHAP summary
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             ConfusionMatrixDisplay, f1_score, roc_auc_score)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from explain import contributions  # noqa: E402
from features import CLASS_NAMES, FEATURES, LEAKY_COLUMNS, SKEWED, engineer, make_target  # noqa: E402

SEED = 42


def load(path: Path) -> tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv(path)
    n0 = len(df)
    if "type" in df:
        df = df[df["type"].eq("User")]          # drop bot accounts
    df = df.drop_duplicates()
    print(f"Loaded {n0:,} rows -> {len(df):,} after removing bots/duplicates")
    y = make_target(df["followers"])
    X = engineer(df)[FEATURES]
    assert not set(X.columns) & set(LEAKY_COLUMNS), "leaky column in features!"
    return X, y


def models() -> dict:
    skew_idx = [FEATURES.index(c) for c in SKEWED]
    linear_prep = ColumnTransformer(
        [("log", FunctionTransformer(np.log1p), skew_idx)], remainder="passthrough"
    )
    return {
        "Baseline (majority class)": DummyClassifier(strategy="most_frequent"),
        "Logistic Regression": Pipeline([
            ("log", linear_prep), ("scale", StandardScaler()),
            ("clf", LogisticRegression(max_iter=2000, random_state=SEED)),
        ]),
        "Random Forest": RandomForestClassifier(
            n_estimators=400, min_samples_leaf=5, n_jobs=-1, random_state=SEED),
        "XGBoost": XGBClassifier(
            n_estimators=400, max_depth=5, learning_rate=0.05, subsample=0.8,
            colsample_bytree=0.8, min_child_weight=5, eval_metric="mlogloss",
            n_jobs=-1, random_state=SEED),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=ROOT / "dataset" / "developers_features_engineered.csv", type=Path)
    args = ap.parse_args()

    (ROOT / "models").mkdir(exist_ok=True)
    reports = ROOT / "reports"
    reports.mkdir(exist_ok=True)

    X, y = load(args.data)
    print("Class balance:", y.value_counts(normalize=True).sort_index().round(3).to_dict())

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)
    cv = StratifiedKFold(5, shuffle=True, random_state=SEED)

    results, fitted = {}, {}
    for name, model in models().items():
        cvr = cross_validate(model, X_tr, y_tr, cv=cv, scoring=["accuracy", "f1_macro"], n_jobs=1)
        model.fit(X_tr, y_tr)
        pred = model.predict(X_te)
        proba = model.predict_proba(X_te)
        results[name] = {
            "cv_accuracy_mean": float(cvr["test_accuracy"].mean()),
            "cv_accuracy_std": float(cvr["test_accuracy"].std()),
            "cv_f1_macro_mean": float(cvr["test_f1_macro"].mean()),
            "train_accuracy": float(model.score(X_tr, y_tr)),
            "test_accuracy": float(accuracy_score(y_te, pred)),
            "test_f1_macro": float(f1_score(y_te, pred, average="macro")),
            "test_roc_auc_ovr": float(roc_auc_score(y_te, proba, multi_class="ovr")),
        }
        fitted[name] = model
        r = results[name]
        print(f"{name:28s} CV acc {r['cv_accuracy_mean']:.3f}±{r['cv_accuracy_std']:.3f} | "
              f"train {r['train_accuracy']:.3f} | test acc {r['test_accuracy']:.3f} | "
              f"F1 {r['test_f1_macro']:.3f} | AUC {r['test_roc_auc_ovr']:.3f}")

    # Pick best non-baseline model by CV macro-F1 (never by test score).
    best_name = max((n for n in results if not n.startswith("Baseline")),
                    key=lambda n: results[n]["cv_f1_macro_mean"])
    best = fitted[best_name]
    print(f"\nSelected model: {best_name}")
    best_pred = best.predict(X_te)
    report = classification_report(y_te, best_pred, target_names=[CLASS_NAMES[i] for i in range(3)])
    print(report)

    # Confusion matrix
    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay(confusion_matrix(y_te, best_pred),
                           display_labels=["High", "Medium", "Low"]).plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title(f"{best_name} - test confusion matrix\n(rows = true visibility risk)")
    fig.tight_layout(); fig.savefig(reports / "confusion_matrix.png", dpi=120); plt.close(fig)

    # Explainability: SHAP-style contributions towards the High-risk class (class 0)
    sample = X_te.sample(min(2000, len(X_te)), random_state=SEED)
    sv = contributions(best, sample)                       # (n, features, classes)
    imp = pd.Series(np.abs(sv[:, :, 0]).mean(0), index=FEATURES).sort_values()
    shap_top = imp.sort_values(ascending=False).round(4).to_dict()
    fig, ax = plt.subplots(figsize=(7, 5))
    imp.plot.barh(ax=ax, color="#4C72B0")
    ax.set_title(f"{best_name}: mean |SHAP| towards HIGH visibility risk")
    ax.set_xlabel("mean |contribution| (log-odds)")
    fig.tight_layout(); fig.savefig(reports / "shap_high_risk_drivers.png", dpi=120); plt.close(fig)
    try:
        import shap
        plt.figure()
        shap.summary_plot(sv[:, :, 0], sample, show=False, max_display=14)
        plt.title("SHAP summary - class: High visibility risk")
        plt.tight_layout(); plt.savefig(reports / "shap_summary_high_risk.png", dpi=120); plt.close()
    except Exception as e:  # beeswarm plot is optional
        print("SHAP summary plot skipped:", e)

    # Refit chosen model on all data for deployment.
    final = models()[best_name].fit(X, y)
    bundle = {
        "model": final,
        "model_name": best_name,
        "features": FEATURES,
        "class_names": CLASS_NAMES,
        "follower_bands": {"High": "0-19", "Medium": "20-99", "Low": "100+"},
        "train_rows": int(len(X)),
        "data_snapshot": "GitHub profiles collected Feb 2026",
        "test_metrics": results[best_name],
    }
    joblib.dump(bundle, ROOT / "models" / "visibility_risk_model.joblib", compress=3)
    json.dump({"selected_model": best_name, "models": results, "shap_mean_abs_high_risk": shap_top,
               "n_rows": int(len(X)), "class_balance": y.value_counts().sort_index().to_dict()},
              open(reports / "metrics.json", "w"), indent=2)
    (reports / "classification_report.txt").write_text(f"{best_name}\n\n{report}")
    print("Saved models/visibility_risk_model.joblib and reports/")


if __name__ == "__main__":
    main()

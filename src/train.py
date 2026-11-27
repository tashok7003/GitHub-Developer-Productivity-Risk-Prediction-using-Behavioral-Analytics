"""Train and evaluate the GitHub developer risk models.

Usage:
    python src/train.py                                            # v2 data, visibility task
    python src/train.py --data dataset/developers_v3.csv           # v3 data, visibility task
    python src/train.py --data dataset/developers_v3.csv --task decline

Outputs (per task):
    models/<task model>.joblib     final pipeline + metadata
    reports[/decline]/metrics.json, classification_report.txt, confusion_matrix.png,
                      shap_high_risk_drivers.png, shap_summary_high_risk.png
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
from features import LEAKY, SKEWED, TASKS, engineer, feature_list, target_for  # noqa: E402

SEED = 42


def load(path: Path, task: str) -> tuple[pd.DataFrame, pd.Series, str]:
    df = pd.read_csv(path)
    n0 = len(df)
    if "type" in df:
        df = df[df["type"].eq("User")]          # drop bot accounts
    df = df.drop_duplicates(subset="developer" if "developer" in df else None)
    print(f"Loaded {n0:,} rows -> {len(df):,} after removing bots/duplicates")
    y, desc = target_for(task, df)
    keep = y.notna()
    if (~keep).any():
        print(f"{(~keep).sum():,} developers not eligible for this task (dropped)")
    df, y = df[keep], y[keep].astype(int)
    eng = engineer(df)
    feats = feature_list(eng, task)
    X = eng[feats]
    leaks = set(feats) & set(LEAKY[task])
    assert not leaks, f"leaky columns in features: {leaks}"
    print(f"Task: {task} | Target: {desc}")
    print(f"{len(feats)} input features: {feats}")
    return X, y, desc


def models(features: list[str]) -> dict:
    skew_idx = [features.index(c) for c in SKEWED if c in features]
    linear_prep = ColumnTransformer(
        [("log", FunctionTransformer(np.log1p), skew_idx)], remainder="passthrough"
    )
    return {
        "Baseline (majority class)": DummyClassifier(strategy="most_frequent"),
        "Logistic Regression": Pipeline([
            ("log", linear_prep), ("scale", StandardScaler()),
            ("clf", LogisticRegression(max_iter=3000, random_state=SEED)),
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
    ap.add_argument("--task", choices=list(TASKS), default="visibility")
    args = ap.parse_args()
    cfg = TASKS[args.task]
    class_names, short = cfg["classes"], cfg["short"]

    (ROOT / "models").mkdir(exist_ok=True)
    reports = ROOT / cfg["report_dir"]
    reports.mkdir(parents=True, exist_ok=True)

    X, y, target = load(args.data, args.task)
    feats = list(X.columns)
    print("Class balance:", {short[k]: v for k, v in y.value_counts(normalize=True).sort_index().round(3).items()})

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)
    cv = StratifiedKFold(5, shuffle=True, random_state=SEED)

    results, fitted = {}, {}
    for name, model in models(feats).items():
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
    report = classification_report(y_te, best_pred, target_names=[class_names[i] for i in range(3)])
    print(report)

    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay(confusion_matrix(y_te, best_pred),
                           display_labels=[short[i] for i in range(3)]).plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title(f"{best_name} - test confusion matrix\n(rows = true class)")
    fig.tight_layout(); fig.savefig(reports / "confusion_matrix.png", dpi=120); plt.close(fig)

    # Explainability: contributions towards the highest-risk class (class 0)
    sample = X_te.sample(min(2000, len(X_te)), random_state=SEED)
    sv = contributions(best, sample)                       # (n, features, classes)
    imp = pd.Series(np.abs(sv[:, :, 0]).mean(0), index=feats).sort_values()
    shap_top = imp.sort_values(ascending=False).round(4).to_dict()
    fig, ax = plt.subplots(figsize=(7, max(4, 0.28 * len(feats))))
    imp.tail(20).plot.barh(ax=ax, color="#4C72B0")
    ax.set_title(f"{best_name}: mean |SHAP| towards '{short[0]}'")
    ax.set_xlabel("mean |contribution|")
    fig.tight_layout(); fig.savefig(reports / "shap_high_risk_drivers.png", dpi=120); plt.close(fig)
    try:
        import shap
        plt.figure()
        shap.summary_plot(sv[:, :, 0], sample, show=False, max_display=15)
        plt.title(f"SHAP summary - class: {short[0]}")
        plt.tight_layout(); plt.savefig(reports / "shap_summary_high_risk.png", dpi=120); plt.close()
    except Exception as e:  # beeswarm plot is optional
        print("SHAP summary plot skipped:", e)

    # Refit chosen model on all data for deployment.
    final = models(feats)[best_name].fit(X, y)
    bundle = {
        "task": args.task,
        "model": final,
        "model_name": best_name,
        "features": feats,
        "target": target,
        "class_names": class_names,
        "class_short": short,
        "needs_graphql": any(c.endswith(("_90d", "_1y")) or c in ("original_repos", "pinned_repos")
                             for c in feats),
        "train_rows": int(len(X)),
        "data_file": args.data.name,
        "test_metrics": results[best_name],
    }
    joblib.dump(bundle, ROOT / "models" / cfg["model_file"], compress=3)
    json.dump({"task": args.task, "selected_model": best_name, "target": target, "features": feats,
               "models": results, "shap_mean_abs_class0": shap_top, "n_rows": int(len(X)),
               "class_balance": {short[k]: int(v) for k, v in y.value_counts().sort_index().items()}},
              open(reports / "metrics.json", "w"), indent=2)
    (reports / "classification_report.txt").write_text(f"{best_name}\n{target}\n\n{report}")
    print(f"Saved models/{cfg['model_file']} and {cfg['report_dir']}/")


if __name__ == "__main__":
    main()

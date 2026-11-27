"""Per-feature contributions for the trained model (works for the linear pipeline and tree models)."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline


def contributions(model, X: pd.DataFrame, background: pd.DataFrame | None = None) -> np.ndarray:
    """Return SHAP-style contributions with shape (n_rows, n_features, n_classes).

    * Logistic Regression pipeline: exact linear SHAP in log-odds space,
      coef * (x_scaled - mean_scaled), where the scaler mean is 0, so coef * x_scaled.
    * Tree models: shap.TreeExplainer.
    """
    if isinstance(model, Pipeline):
        Z = model[:-1].transform(X)                 # log1p + standardised (columns reordered)
        clf = model[-1]
        # ColumnTransformer puts log-transformed columns first, then the remainder.
        order = _transformed_order(model, list(X.columns))
        contrib = Z[:, :, None] * clf.coef_.T[None, :, :]     # (n, f, classes)
        out = np.zeros_like(contrib)
        for new_i, orig_i in enumerate(order):
            out[:, orig_i, :] = contrib[:, new_i, :]
        return out
    import shap
    sv = shap.TreeExplainer(model).shap_values(X)
    return np.stack(sv, axis=-1) if isinstance(sv, list) else sv


def _transformed_order(pipe: Pipeline, cols: list[str]) -> list[int]:
    ct = pipe.named_steps["log"]
    log_idx = list(ct.transformers_[0][2])
    rest = [i for i in range(len(cols)) if i not in log_idx]
    return log_idx + rest

from __future__ import annotations

import numpy as np
import pandas as pd


def population_stability_index(expected: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
    expected = np.asarray(expected, dtype=float)
    actual = np.asarray(actual, dtype=float)

    if expected.size == 0 or actual.size == 0:
        return 0.0
    if np.all(expected == expected[0]) and np.all(actual == actual[0]):
        return 0.0

    quantiles = np.linspace(0, 1, bins + 1)
    edges = np.unique(np.quantile(expected, quantiles))
    if len(edges) < 3:
        return 0.0

    exp_hist, _ = np.histogram(expected, bins=edges)
    act_hist, _ = np.histogram(actual, bins=edges)

    exp_pct = np.clip(exp_hist / max(exp_hist.sum(), 1), 1e-6, None)
    act_pct = np.clip(act_hist / max(act_hist.sum(), 1), 1e-6, None)
    psi = np.sum((act_pct - exp_pct) * np.log(act_pct / exp_pct))
    return float(psi)


def feature_drift_report(train_df: pd.DataFrame, oot_df: pd.DataFrame, numeric_columns: list[str]) -> dict:
    per_feature: dict[str, float] = {}
    for col in numeric_columns:
        train_values = pd.to_numeric(train_df[col], errors="coerce").fillna(0).to_numpy()
        oot_values = pd.to_numeric(oot_df[col], errors="coerce").fillna(0).to_numpy()
        per_feature[col] = population_stability_index(train_values, oot_values)

    avg_psi = float(np.mean(list(per_feature.values()))) if per_feature else 0.0
    max_psi = float(np.max(list(per_feature.values()))) if per_feature else 0.0
    return {
        "feature_psi": per_feature,
        "avg_psi": avg_psi,
        "max_psi": max_psi,
        "drift_flag": bool(max_psi >= 0.2),
    }


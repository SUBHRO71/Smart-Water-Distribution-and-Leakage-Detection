"""Pre-training checks for target, duplicate, and suspicious-feature leakage."""

from typing import Any

import numpy as np
import pandas as pd

PROHIBITED_TOKENS = ("leak", "label", "target", "repair", "event_id", "is_leak")


def audit_feature_leakage(
    features: pd.DataFrame,
    targets: pd.DataFrame,
    allowed_features: list[str],
    suspicious_correlation: float = 0.995,
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Audit an already selected development feature table without fitting a model."""
    missing_allowed = sorted(set(allowed_features) - set(features.columns))
    unexpected = sorted(set(features.columns) - set(allowed_features))
    prohibited = sorted(
        name for name in features.columns if any(token in name.lower() for token in PROHIBITED_TOKENS)
    )
    if not features.index.equals(targets.index):
        raise ValueError("feature and target indexes must match exactly")

    selected = features.loc[:, [name for name in allowed_features if name in features]]
    duplicate_timestamps = int(selected.index.duplicated().sum())
    duplicate_feature_rows = int(selected.duplicated(keep=False).sum())
    rows = []
    for feature_name in selected.columns:
        x = selected[feature_name]
        for target_name in targets.columns:
            y = targets[target_name]
            valid = x.notna() & y.notna()
            correlation = float("nan")
            if valid.sum() >= 3 and x[valid].nunique() > 1 and y[valid].nunique() > 1:
                correlation = float(np.corrcoef(x[valid].astype(float), y[valid].astype(float))[0, 1])
            rows.append(
                {
                    "feature": feature_name,
                    "target": target_name,
                    "pearson_correlation": correlation,
                    "absolute_correlation": abs(correlation) if np.isfinite(correlation) else None,
                    "suspicious": bool(
                        np.isfinite(correlation) and abs(correlation) >= suspicious_correlation
                    ),
                }
            )
    correlations = pd.DataFrame(rows)
    summary = {
        "rows": len(selected),
        "selected_feature_count": len(selected.columns),
        "missing_allowed_features": missing_allowed,
        "unexpected_features_excluded": unexpected,
        "prohibited_feature_names": prohibited,
        "duplicate_timestamps": duplicate_timestamps,
        "duplicate_feature_rows": duplicate_feature_rows,
        "suspicious_correlation_threshold": suspicious_correlation,
        "suspicious_feature_target_pairs": int(correlations["suspicious"].sum()),
        "status": (
            "pass"
            if not missing_allowed and not prohibited and duplicate_timestamps == 0
            else "fail"
        ),
    }
    return summary, correlations

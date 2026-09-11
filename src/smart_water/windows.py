"""LeakDB sequence windows and leakage-safe fold preprocessing."""

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class WindowSet:
    """Model inputs with immutable provenance for every window."""

    x: np.ndarray
    y: np.ndarray
    scenario: np.ndarray
    timestamp: np.ndarray
    features: tuple[str, ...]


def build_scenario_windows(
    frame: pd.DataFrame,
    features: list[str],
    history_steps: int = 24,
    stride: int = 6,
) -> WindowSet:
    """Build causal windows without crossing scenario boundaries."""
    if history_steps < 1 or stride < 1:
        raise ValueError("history_steps and stride must be positive")
    required = {"scenario", "timestamp", "label", *features}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"missing required columns: {missing}")
    if not features:
        raise ValueError("at least one feature is required")

    xs, ys, scenarios, timestamps = [], [], [], []
    for scenario, group in frame.groupby("scenario", sort=True):
        group = group.sort_values("timestamp")
        ts = pd.DatetimeIndex(group["timestamp"])
        if ts.has_duplicates or not ts.is_monotonic_increasing:
            raise ValueError(f"scenario {scenario}: timestamps must be unique and increasing")
        if len(ts) > 1 and not np.all(np.diff(ts.asi8) == 30 * 60 * 1_000_000_000):
            raise ValueError(f"scenario {scenario}: expected complete 30-minute cadence")
        values = group[features].to_numpy(dtype=np.float32)
        labels = group["label"].to_numpy(dtype=np.int8)
        if not np.isfinite(values).all() or not np.isin(labels, [0, 1]).all():
            raise ValueError(f"scenario {scenario}: nonfinite features or nonbinary labels")
        for end in range(history_steps - 1, len(group), stride):
            start = end - history_steps + 1
            xs.append(values[start : end + 1])
            ys.append(labels[end])
            scenarios.append(scenario)
            timestamps.append(ts[end].to_datetime64())

    if not xs:
        raise ValueError("no windows can be constructed")
    return WindowSet(
        x=np.stack(xs),
        y=np.asarray(ys, dtype=np.int8),
        scenario=np.asarray(scenarios),
        timestamp=np.asarray(timestamps),
        features=tuple(features),
    )


def split_development_test(windows: WindowSet, boundary: str = "2017-10-01") -> tuple[np.ndarray, np.ndarray]:
    """Return development and locked-test masks using target timestamps."""
    threshold = np.datetime64(boundary)
    return windows.timestamp < threshold, windows.timestamp >= threshold


def leave_one_scenario_out_masks(
    windows: WindowSet, held_out_scenario: int, boundary: str = "2017-10-01"
) -> tuple[np.ndarray, np.ndarray]:
    """Create one grouped CV fold inside the chronological development period."""
    development, _ = split_development_test(windows, boundary)
    train = development & (windows.scenario != held_out_scenario)
    assessment = development & (windows.scenario == held_out_scenario)
    return train, assessment


def fit_transform_fold(
    windows: WindowSet,
    train_mask: np.ndarray,
    assessment_mask: np.ndarray,
    *,
    apply_smote: bool,
    random_state: int = 42,
    k_neighbors: int = 5,
) -> dict[str, object]:
    """Fit scaling on training rows and optionally SMOTE training windows only."""
    from sklearn.preprocessing import StandardScaler

    if np.any(train_mask & assessment_mask):
        raise ValueError("training and assessment masks overlap")
    x_train = windows.x[train_mask]
    y_train = windows.y[train_mask]
    x_assessment = windows.x[assessment_mask]
    y_assessment = windows.y[assessment_mask].copy()
    if not len(x_train) or not len(x_assessment):
        raise ValueError("training and assessment must both contain windows")

    scaler = StandardScaler()
    feature_count = x_train.shape[-1]
    scaler.fit(x_train.reshape(-1, feature_count))
    x_train_scaled = scaler.transform(x_train.reshape(-1, feature_count)).reshape(x_train.shape)
    x_assessment_scaled = scaler.transform(
        x_assessment.reshape(-1, feature_count)
    ).reshape(x_assessment.shape)

    original_counts = _counts(y_train)
    if apply_smote:
        from imblearn.over_sampling import SMOTE

        sampler = SMOTE(random_state=random_state, k_neighbors=k_neighbors)
        flat, y_train = sampler.fit_resample(
            x_train_scaled.reshape(len(x_train_scaled), -1), y_train
        )
        x_train_scaled = flat.reshape(-1, windows.x.shape[1], feature_count)

    return {
        "x_train": x_train_scaled.astype(np.float32),
        "y_train": y_train.astype(np.int8),
        "x_assessment": x_assessment_scaled.astype(np.float32),
        "y_assessment": y_assessment,
        "scaler": scaler,
        "train_counts_before": original_counts,
        "train_counts_after": _counts(y_train),
        "assessment_counts": _counts(y_assessment),
        "smote_applied": apply_smote,
    }


def _counts(labels: np.ndarray) -> dict[str, int]:
    return {"negative": int((labels == 0).sum()), "positive": int((labels == 1).sum())}

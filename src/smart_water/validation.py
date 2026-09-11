"""Chronological cross-validation construction and fold feasibility checks."""

from typing import Any

import pandas as pd


def expanding_time_folds(
    index: pd.DatetimeIndex,
    n_splits: int = 10,
    embargo_steps: int = 299,
) -> list[dict[str, Any]]:
    """Create equal-width expanding folds with an origin embargo.

    With 288 history steps and 12 future steps, 299 embargoed origins ensure that
    the raw timestamps used by the last training window and first assessment window
    do not overlap.
    """
    if n_splits < 2:
        raise ValueError("n_splits must be at least 2")
    if embargo_steps < 0:
        raise ValueError("embargo_steps cannot be negative")
    if not index.is_monotonic_increasing or index.has_duplicates:
        raise ValueError("index must be unique and increasing")
    block = len(index) // (n_splits + 1)
    if block < 1:
        raise ValueError("not enough samples for requested folds")

    folds = []
    for fold_number in range(1, n_splits + 1):
        assessment_start = fold_number * block
        assessment_end = len(index) if fold_number == n_splits else assessment_start + block
        training_end = assessment_start - embargo_steps
        if training_end <= 0:
            raise ValueError("embargo leaves no training samples")
        folds.append(
            {
                "fold": fold_number,
                "train_positions": (0, training_end),
                "assessment_positions": (assessment_start, assessment_end),
                "train_start": str(index[0]),
                "train_end_inclusive": str(index[training_end - 1]),
                "assessment_start": str(index[assessment_start]),
                "assessment_end_inclusive": str(index[assessment_end - 1]),
                "embargo_steps": embargo_steps,
            }
        )
    return folds


def fold_feasibility(
    index: pd.DatetimeIndex,
    targets: pd.DataFrame,
    events: pd.DataFrame,
    folds: list[dict[str, Any]],
    smote_k_neighbors: int = 5,
) -> pd.DataFrame:
    """Count class and event support without resampling or fitting a model."""
    targets = targets.reindex(index)
    rows = []
    for fold in folds:
        train_a, train_b = fold["train_positions"]
        assess_a, assess_b = fold["assessment_positions"]
        for target in targets.columns:
            area = target.removeprefix("leak_")
            train = targets[target].iloc[train_a:train_b].dropna().astype(int)
            assessment = targets[target].iloc[assess_a:assess_b].dropna().astype(int)
            minority = min(int((train == 0).sum()), int((train == 1).sum()))
            assess_start = index[assess_a]
            assess_end_exclusive = index[assess_b - 1] + pd.Timedelta(5, unit="m")
            assessment_events = events[
                (events["area"] == area)
                & (pd.to_datetime(events["onset"]) < assess_end_exclusive)
                & (pd.to_datetime(events["end_exclusive"]) > assess_start)
            ]
            both_train = train.nunique() == 2
            both_assessment = assessment.nunique() == 2
            rows.append(
                {
                    "fold": fold["fold"],
                    "target": target,
                    "train_start": fold["train_start"],
                    "train_end_inclusive": fold["train_end_inclusive"],
                    "assessment_start": fold["assessment_start"],
                    "assessment_end_inclusive": fold["assessment_end_inclusive"],
                    "train_negative": int((train == 0).sum()),
                    "train_positive": int((train == 1).sum()),
                    "assessment_negative": int((assessment == 0).sum()),
                    "assessment_positive": int((assessment == 1).sum()),
                    "assessment_distinct_events": len(assessment_events),
                    "both_classes_in_training": both_train,
                    "both_classes_in_assessment": both_assessment,
                    "smote_supported": both_train and minority > smote_k_neighbors,
                    "evaluation_supported": both_assessment,
                    "failure_reason": _failure_reason(
                        both_train, both_assessment, minority, smote_k_neighbors
                    ),
                }
            )
    return pd.DataFrame(rows)


def _failure_reason(
    both_train: bool,
    both_assessment: bool,
    minority: int,
    k_neighbors: int,
) -> str:
    reasons = []
    if not both_train:
        reasons.append("single_class_training")
    elif minority <= k_neighbors:
        reasons.append("insufficient_smote_neighbors")
    if not both_assessment:
        reasons.append("single_class_assessment")
    return ";".join(reasons) if reasons else "None"

import pandas as pd
import pytest

from smart_water.validation import expanding_time_folds, fold_feasibility


def test_expanding_folds_are_ordered_and_embargoed():
    index = pd.date_range("2018-01-01", periods=1100, freq="5min")
    folds = expanding_time_folds(index, n_splits=10, embargo_steps=9)
    assert len(folds) == 10
    previous_train_end = -1
    for fold in folds:
        _, train_end = fold["train_positions"]
        assess_start, _ = fold["assessment_positions"]
        assert train_end == assess_start - 9
        assert train_end > previous_train_end
        previous_train_end = train_end


def test_fold_feasibility_flags_single_class_assessment():
    index = pd.date_range("2018-01-01", periods=60, freq="5min")
    targets = pd.DataFrame({"leak_Area_A": [0] * 30 + [1] * 30}, index=index)
    events = pd.DataFrame(
        [{"area": "Area_A", "onset": index[30], "end_exclusive": index[-1]}]
    )
    folds = expanding_time_folds(index, n_splits=2, embargo_steps=1)
    report = fold_feasibility(index, targets, events, folds)
    assert not report.iloc[0]["both_classes_in_training"]
    assert "single_class" in report.iloc[0]["failure_reason"]


def test_invalid_fold_request_is_rejected():
    index = pd.date_range("2018-01-01", periods=10, freq="5min")
    with pytest.raises(ValueError):
        expanding_time_folds(index, n_splits=10, embargo_steps=2)

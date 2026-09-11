import numpy as np
import pandas as pd
import pytest

from smart_water.windows import (
    build_scenario_windows,
    fit_transform_fold,
    leave_one_scenario_out_masks,
    split_development_test,
)


def _frame() -> pd.DataFrame:
    rows = []
    for scenario in (1, 2, 3):
        for step, timestamp in enumerate(pd.date_range("2017-09-30", periods=12, freq="30min")):
            rows.append(
                {
                    "scenario": scenario,
                    "timestamp": timestamp,
                    "label": int(step >= (7 + scenario % 2)),
                    "pressure__node_2": float(step + scenario * 10),
                    "pressure__node_3": float(step * 2 - scenario),
                }
            )
    return pd.DataFrame(rows)


def test_windows_do_not_cross_scenarios_and_split_on_target_time():
    windows = build_scenario_windows(
        _frame(), ["pressure__node_2", "pressure__node_3"], history_steps=4, stride=2
    )
    assert windows.x.shape == (15, 4, 2)
    assert np.all(np.bincount(windows.scenario)[1:] == 5)
    development, test = split_development_test(windows, "2017-09-30 04:00")
    assert not np.any(development & test)
    assert np.all(development | test)


def test_fold_scaling_and_smote_touch_training_only():
    windows = build_scenario_windows(
        _frame(), ["pressure__node_2", "pressure__node_3"], history_steps=2, stride=1
    )
    train = windows.scenario != 3
    assessment = windows.scenario == 3
    held_out_labels = windows.y[assessment].copy()
    result = fit_transform_fold(
        windows, train, assessment, apply_smote=True, random_state=7, k_neighbors=1
    )
    assert result["train_counts_after"]["negative"] == result["train_counts_after"]["positive"]
    assert np.array_equal(result["y_assessment"], held_out_labels)
    assert len(result["y_assessment"]) == int(assessment.sum())
    raw_train_mean = windows.x[train].reshape(-1, 2).mean(axis=0)
    assert np.allclose(result["scaler"].mean_, raw_train_mean)


def test_group_fold_is_development_only_and_scenario_disjoint():
    windows = build_scenario_windows(
        _frame(), ["pressure__node_2", "pressure__node_3"], history_steps=2
    )
    train, assessment = leave_one_scenario_out_masks(
        windows, 2, boundary="2017-09-30 04:00"
    )
    assert set(windows.scenario[train]) == {1, 3}
    assert set(windows.scenario[assessment]) == {2}
    assert np.all(windows.timestamp[train | assessment] < np.datetime64("2017-09-30 04:00"))


def test_overlapping_masks_are_rejected():
    windows = build_scenario_windows(
        _frame(), ["pressure__node_2", "pressure__node_3"], history_steps=2
    )
    mask = windows.scenario == 1
    with pytest.raises(ValueError, match="overlap"):
        fit_transform_fold(windows, mask, mask, apply_smote=False)

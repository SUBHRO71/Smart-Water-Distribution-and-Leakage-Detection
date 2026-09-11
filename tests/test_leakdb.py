import pandas as pd

from smart_water.leakdb import scenario_fold_feasibility


def test_scenario_folds_keep_test_period_and_training_scenarios_separate():
    frames = []
    for scenario in range(1, 11):
        timestamps = pd.date_range("2017-09-29", periods=8, freq="12h")
        labels = [0, 0, 1 if scenario >= 3 else 0, 0, 0, 1 if scenario >= 3 else 0, 0, 0]
        frames.append(
            pd.DataFrame({"scenario": scenario, "timestamp": timestamps, "label": labels})
        )
    frame = pd.concat(frames, ignore_index=True)
    folds, summary = scenario_fold_feasibility(frame, split_time="2017-10-01", smote_k_neighbors=1)
    assert len(folds) == 10
    assert folds["smote_supported"].all()
    assert summary["test_negative"] + summary["test_positive"] == 40

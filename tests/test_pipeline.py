import numpy as np
import pandas as pd
import pytest

from smart_water.baseline import evaluate
from smart_water.data import load_scada, prepare


def write_csv(path, values, column="sensor"):
    pd.DataFrame({"Timestamp": pd.date_range("2018-01-01", periods=len(values), freq="5min"),
                  column: values}).to_csv(path, sep=";", decimal=",", index=False)


def test_decimal_comma_and_unit_conversion_without_label_leakage(tmp_path):
    write_csv(tmp_path / "2018_SCADA_Flows.csv", [10.5, 12.5])
    write_csv(tmp_path / "2018_SCADA_Demands.csv", [1000.0, 2000.0])
    write_csv(tmp_path / "2018_Leakages.csv", [0.0, 3.0])
    out = tmp_path / "processed"
    prepare(tmp_path, out, 2018)
    features = pd.read_csv(out / "2018_features.csv")
    assert features["flows__sensor"].tolist() == [10.5, 12.5]
    assert features["demands__sensor"].tolist() == [1.0, 2.0]
    assert not any("leak" in col for col in features)
    assert pd.read_csv(out / "2018_labels.csv")["is_leak"].tolist() == [0, 1]


@pytest.mark.parametrize("timestamps", [
    ["2018-01-01 00:00", "2018-01-01 00:00"],
    ["2018-01-01 00:05", "2018-01-01 00:00"],
    ["2018-01-01 00:00", "2018-01-01 00:10"],
])
def test_reject_broken_time_grid(tmp_path, timestamps):
    path = tmp_path / "broken.csv"
    pd.DataFrame({"Timestamp": timestamps, "sensor": [1, 2]}).to_csv(path, sep=";", index=False)
    with pytest.raises(ValueError):
        load_scada(path)


def test_reject_nonfinite_readings(tmp_path):
    path = tmp_path / "missing.csv"
    write_csv(path, [1, np.nan])
    with pytest.raises(ValueError, match="Missing"):
        load_scada(path)


def test_reject_misaligned_sensor_files(tmp_path):
    write_csv(tmp_path / "2018_SCADA_Flows.csv", [1, 2])
    write_csv(tmp_path / "2018_SCADA_Pressures.csv", [1, 2, 3])
    with pytest.raises(ValueError, match="identical"):
        prepare(tmp_path, tmp_path / "out", 2018)


def test_seasonal_baseline_uses_past_and_evaluates_only_holdout():
    frame = pd.DataFrame({"flow": [1, 2, 1, 2, 4, 6]},
                         index=pd.date_range("2018-01-01", periods=6, freq="5min"))
    result = evaluate(frame, "flow", "2018-01-01 00:20", period=2)
    assert result["rows"] == 2
    assert result["mae_m3_h"] == 3.5
    assert result["rmse_m3_h"] == pytest.approx(np.sqrt(12.5))

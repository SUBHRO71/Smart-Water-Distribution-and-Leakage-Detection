import pandas as pd
import pytest

from smart_water.leakage_audit import audit_feature_leakage


def test_audit_finds_prohibited_missing_duplicate_and_suspicious_features():
    index = pd.date_range("2018-01-01", periods=6, freq="5min")
    targets = pd.DataFrame({"leak_A": [0, 0, 0, 1, 1, 1]}, index=index)
    features = pd.DataFrame(
        {"pressure": [0, 0, 0, 1, 1, 1], "leak_flow": [0, 0, 0, 2, 2, 2]},
        index=index,
    )
    summary, correlations = audit_feature_leakage(
        features, targets, ["pressure", "missing_sensor", "leak_flow"]
    )
    assert summary["status"] == "fail"
    assert summary["prohibited_feature_names"] == ["leak_flow"]
    assert summary["missing_allowed_features"] == ["missing_sensor"]
    assert summary["duplicate_feature_rows"] == 6
    assert correlations.loc[correlations["feature"] == "pressure", "suspicious"].iloc[0]


def test_audit_requires_exact_index_alignment():
    x = pd.DataFrame({"pressure": [1, 2]}, index=pd.date_range("2018-01-01", periods=2))
    y = pd.DataFrame({"leak": [0, 1]}, index=pd.date_range("2019-01-01", periods=2))
    with pytest.raises(ValueError, match="indexes"):
        audit_feature_leakage(x, y, ["pressure"])

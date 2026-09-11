"""Rolling one-step seasonal persistence baseline; no fitted parameters."""

import numpy as np
import pandas as pd


def evaluate(frame: pd.DataFrame, column: str, start: str, period: int = 288) -> dict:
    if period < 1:
        raise ValueError("period must be positive")
    actual = frame[column]
    prediction = actual.shift(period)
    mask = (frame.index >= pd.Timestamp(start)) & prediction.notna() & actual.notna()
    if not mask.any():
        raise ValueError("No evaluation rows after warmup and split boundary")
    errors = (actual[mask] - prediction[mask]).to_numpy()
    if not np.isfinite(errors).all():
        raise ValueError("Nonfinite values in evaluation")
    return {"model": "seasonal_persistence", "target": column, "period_steps": period,
            "protocol": "rolling one-step; previous observed values available at each origin",
            "evaluation_start": str(frame.index[mask][0]), "rows": int(mask.sum()),
            "mae_m3_h": float(np.abs(errors).mean()),
            "rmse_m3_h": float(np.sqrt(np.square(errors).mean()))}

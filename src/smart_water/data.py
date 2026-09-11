"""Verified downloads and strict parsing for the original BattLeDIM release."""

import hashlib
import json
import shutil
from pathlib import Path
from urllib.parse import quote
from urllib.request import urlopen

import numpy as np
import pandas as pd

RECORD = "https://zenodo.org/api/records/4017659"
FILES_BASE = "https://zenodo.org/records/4017659/files/"


def download(output: Path, year: int = 2018, full: bool = False) -> list[Path]:
    """Download flow + metadata by default; full adds the other sensor and label files.

    Labels are downloaded separately from features. The real network and configuration
    are deliberately excluded because they expose hidden simulation information.
    """
    if year not in (2018, 2019):
        raise ValueError("year must be 2018 or 2019")
    output.mkdir(parents=True, exist_ok=True)
    names = ["README.txt", "L-TOWN.inp", f"{year}_SCADA_Flows.csv"]
    if full:
        names += [f"{year}_SCADA_{kind}.csv" for kind in ("Demands", "Pressures", "Levels")]
        names += [f"{year}_Leakages.csv"]
    with urlopen(RECORD, timeout=60) as response:
        metadata = json.load(response)
    files = {item["key"]: item for item in metadata["files"]}
    downloaded = []
    for name in names:
        item = files[name]
        algorithm, expected = item["checksum"].split(":", 1)
        if algorithm != "md5":
            raise ValueError(f"Unexpected upstream checksum: {algorithm}")
        destination = output / name
        if destination.exists() and file_md5(destination) == expected:
            downloaded.append(destination)
            continue
        temporary = destination.with_suffix(destination.suffix + ".part")
        try:
            with (
                urlopen(FILES_BASE + quote(name) + "?download=1", timeout=120) as response,
                temporary.open("wb") as handle,
            ):
                shutil.copyfileobj(response, handle)
            if file_md5(temporary) != expected:
                raise ValueError(f"Checksum mismatch: {name}")
            temporary.replace(destination)
        finally:
            temporary.unlink(missing_ok=True)
        downloaded.append(destination)
    (output / f"manifest_{year}.json").write_text(
        json.dumps({"record": RECORD, "license": metadata["metadata"].get("license"),
                    "files": [{"name": p.name, "checksum": files[p.name]["checksum"]}
                              for p in downloaded]}, indent=2), encoding="utf-8"
    )
    return downloaded


def file_md5(path: Path) -> str:
    # MD5 is used solely to match the publisher's file-integrity metadata.
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "md5").hexdigest()


def load_scada(path: Path) -> pd.DataFrame:
    """Parse semicolon CSV with decimal commas; refuse ambiguous or broken time axes."""
    frame = pd.read_csv(path, sep=";", decimal=",")
    if "Timestamp" not in frame or len(frame.columns) < 2 or frame.empty:
        raise ValueError("Expected nonempty BattLeDIM CSV with Timestamp and sensor columns")
    frame["Timestamp"] = pd.to_datetime(frame["Timestamp"], errors="raise")
    frame = frame.set_index("Timestamp")
    if frame.index.hasnans or frame.index.has_duplicates or not frame.index.is_monotonic_increasing:
        raise ValueError("Timestamps must be present, unique and increasing")
    if len(frame) > 1 and not (np.diff(frame.index.to_numpy()) == np.timedelta64(5, "m")).all():
        raise ValueError("Expected a complete 5-minute grid; audit missing intervals first")
    frame = frame.apply(pd.to_numeric, errors="raise").astype(float)
    if not np.isfinite(frame.to_numpy()).all():
        raise ValueError("Missing or infinite readings: audit and impute causally before use")
    return frame


def prepare(raw: Path, output: Path, year: int) -> dict:
    """Build aligned features and optional binary leak labels without mixing them."""
    groups = {}
    for kind in ("Flows", "Pressures", "Demands", "Levels"):
        path = raw / f"{year}_SCADA_{kind}.csv"
        if path.exists():
            frame = load_scada(path)
            if kind == "Demands":
                frame = frame / 1000  # L/h -> m3/h, matching the flow sensors.
            groups[kind] = frame.add_prefix(kind.lower() + "__")
    if "Flows" not in groups:
        raise ValueError("Flow CSV is required; run download first")
    index = groups["Flows"].index
    if not all(frame.index.equals(index) for frame in groups.values()):
        raise ValueError("Sensor files must have identical timestamp grids")
    if not (index.year == year).all():
        raise ValueError("Sensor timestamps do not match the requested year")
    features = pd.concat(groups.values(), axis=1)
    angle = 2 * np.pi * (index.hour * 60 + index.minute) / 1440
    features["hour_sin"], features["hour_cos"] = np.sin(angle), np.cos(angle)
    features["day_of_week"] = index.dayofweek
    label_path = raw / f"{year}_Leakages.csv"
    labels = None
    if label_path.exists():
        leaks = load_scada(label_path)
        if not leaks.index.equals(index):
            raise ValueError("Leak labels must match sensor timestamps exactly")
        if (leaks < 0).any().any():
            raise ValueError("Leak flow cannot be negative")
        labels = pd.DataFrame({"is_leak": (leaks > 0).any(axis=1).astype(int)}, index=index)
    output.mkdir(parents=True, exist_ok=True)
    features.to_csv(output / f"{year}_features.csv", index_label="timestamp")
    destination = output / f"{year}_labels.csv"
    if labels is not None:
        labels.to_csv(destination, index_label="timestamp")
    else:
        destination.unlink(missing_ok=True)  # Never leave stale labels after a sensor-only run.
    summary = {"year": year, "rows": len(features), "feature_count": len(features.columns),
               "groups": list(groups), "labels_available": labels is not None,
               "flow_and_demand_units": "m3/h", "pressure_and_level_units": "m",
               "start": str(index[0]), "end": str(index[-1]),
               "timezone": "unspecified by source; timestamps retained as provided"}
    (output / f"{year}_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary

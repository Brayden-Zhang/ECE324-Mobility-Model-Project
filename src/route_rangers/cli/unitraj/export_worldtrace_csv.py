import argparse
import csv
from pathlib import Path

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(
        description="Export WorldTrace pickle/parquet to a simple CSV"
    )
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    return parser.parse_args()


def _normalize_lon_lat(points):
    arr = np.asarray(points, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[1] < 2:
        raise ValueError("trajectory must be a list of [lat, lon] or [lon, lat]")
    lat = arr[:, 0]
    lon = arr[:, 1]
    if np.any(np.abs(lat) > 90) and np.any(np.abs(lon) <= 90):
        lat, lon = lon, lat
    return lat, lon


def load_records(path: str):
    import pandas as pd

    if path.endswith(".parquet"):
        df = pd.read_parquet(path)
    else:
        df = pd.read_pickle(path)
    return df.to_dict(orient="records")


def main():
    args = parse_args()
    records = load_records(args.input)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)

    with open(out, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["traj_id", "idx", "timestamp", "lat", "lon"])
        for tid, rec in enumerate(records):
            traj = rec.get("trajectory") or rec.get("traj") or rec.get("points")
            if traj is None:
                continue
            times = rec.get("time") or rec.get("times") or rec.get("timestamp")
            lat, lon = _normalize_lon_lat(traj)
            if times is None:
                times = list(range(len(lat)))
            if len(times) != len(lat):
                n = min(len(times), len(lat))
                times = times[:n]
                lat = lat[:n]
                lon = lon[:n]
            for i, (t, la, lo) in enumerate(zip(times, lat, lon)):
                writer.writerow([tid, i, t, float(la), float(lo)])

    print(f"saved CSV: {out}")


if __name__ == "__main__":
    main()

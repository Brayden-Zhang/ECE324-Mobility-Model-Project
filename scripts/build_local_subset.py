#!/usr/bin/env python3
import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(description="Filter a trajectory dataset to a bounding box.")
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--north", type=float, required=True)
    parser.add_argument("--south", type=float, required=True)
    parser.add_argument("--east", type=float, required=True)
    parser.add_argument("--west", type=float, required=True)
    parser.add_argument("--min_points", type=int, default=10)
    parser.add_argument("--max_rows", type=int, default=0)
    return parser.parse_args()


def normalize_lon_lat(points: np.ndarray) -> np.ndarray:
    arr = np.asarray(points, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[1] < 2:
        return arr
    lat = arr[:, 0]
    lon = arr[:, 1]
    if np.any(np.abs(lat) > 90) and np.any(np.abs(lon) <= 90):
        arr = arr[:, ::-1]
    return arr


def in_bbox(arr: np.ndarray, north: float, south: float, east: float, west: float) -> bool:
    if arr.size == 0:
        return False
    lat = arr[:, 0]
    lon = arr[:, 1]
    return np.any((lat >= south) & (lat <= north) & (lon >= west) & (lon <= east))


def main():
    args = parse_args()
    inp = Path(args.input)
    out = Path(args.output)
    if not inp.exists():
        raise FileNotFoundError(f"input not found: {inp}")

    df = pd.read_pickle(inp)
    keep_rows = []
    for _, row in df.iterrows():
        traj = None
        for key in ("trajectory", "traj", "points"):
            if key in row and row[key] is not None:
                traj = row[key]
                break
        if traj is None:
            continue
        arr = normalize_lon_lat(np.asarray(traj))
        if arr.shape[0] < args.min_points:
            continue
        if in_bbox(arr, args.north, args.south, args.east, args.west):
            keep_rows.append(row)
            if args.max_rows > 0 and len(keep_rows) >= args.max_rows:
                break

    if not keep_rows:
        raise RuntimeError("no trajectories matched bbox")

    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(keep_rows).to_pickle(out)
    print(f"saved {len(keep_rows)} rows to {out}")


if __name__ == "__main__":
    main()

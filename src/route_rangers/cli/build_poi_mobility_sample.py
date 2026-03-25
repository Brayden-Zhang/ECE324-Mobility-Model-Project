#!/usr/bin/env python3
"""Build a POI mobility sample from trajectory data, enriched with OSM context."""

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from route_rangers.cli import run_benchmarks as rb

try:
    import h3
except Exception as exc:  # pragma: no cover
    raise RuntimeError("h3 is required to build poi mobility sample") from exc


OSM_TAG_NAMES = ("amenity", "shop", "leisure", "highway")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build POI mobility sample from trajectory records"
    )
    parser.add_argument("--input", type=str, default="data/worldtrace_sample.pkl")
    parser.add_argument("--output", type=str, default="data/poi_mobility_sample.pkl")
    parser.add_argument("--osm_context", type=str, default="data/osm_context.json")
    parser.add_argument("--poi_res", type=int, default=9)
    parser.add_argument("--user_res", type=int, default=4)
    parser.add_argument("--city_res_candidates", type=str, default="4,3,2,1,0")
    parser.add_argument("--target_min_city_records", type=int, default=100)
    parser.add_argument("--target_city_count", type=int, default=2)
    parser.add_argument("--min_poi_len", type=int, default=2)
    parser.add_argument("--max_rows", type=int, default=0)
    return parser.parse_args()


def _h3_cell(lat: float, lon: float, res: int) -> str:
    if hasattr(h3, "latlng_to_cell"):
        return str(h3.latlng_to_cell(lat, lon, res))
    return str(h3.geo_to_h3(lat, lon, res))


def _h3_parent(cell: str, res: int) -> str:
    if hasattr(h3, "cell_to_parent"):
        return str(h3.cell_to_parent(cell, res))
    return str(h3.h3_to_parent(cell, res))


def _h3_resolution(cell: str) -> int:
    if hasattr(h3, "get_resolution"):
        return int(h3.get_resolution(cell))
    return int(h3.h3_get_resolution(cell))


def _iter_points(record: dict) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    traj = record.get("trajectory")
    if traj is None:
        traj = record.get("traj")
    if traj is None:
        traj = record.get("points")
    if traj is None:
        return None

    lat, lon = rb.normalize_lon_lat(traj)
    valid = np.isfinite(lat) & np.isfinite(lon)
    if valid.sum() < 2:
        return None
    lat = lat[valid]
    lon = lon[valid]

    times = record.get("time")
    if times is None:
        times = record.get("times")
    if times is None:
        times = record.get("timestamp")
    if times is None:
        ts = np.arange(lat.shape[0], dtype=np.float32)
    else:
        ts = rb.parse_times(times)
        if ts.shape[0] != valid.shape[0]:
            n = min(ts.shape[0], lat.shape[0])
            lat = lat[:n]
            lon = lon[:n]
            ts = ts[:n]
        else:
            ts = ts[valid]
    if lat.shape[0] < 2:
        return None
    return lat.astype(np.float32), lon.astype(np.float32), ts.astype(np.float32)


def _dedup_consecutive(tokens: Sequence[str]) -> List[str]:
    out: List[str] = []
    prev = None
    for t in tokens:
        if t != prev:
            out.append(t)
            prev = t
    return out


def _load_osm_context(path: Path) -> Tuple[Dict[str, List[float]], Optional[int]]:
    if not path.exists():
        return {}, None
    data = json.loads(path.read_text())
    if not data:
        return {}, None
    sample_key = next(iter(data.keys()))
    return data, _h3_resolution(sample_key)


def _dominant_osm_tag(vec: Sequence[float]) -> str:
    if not vec:
        return "none"
    arr = np.asarray(vec, dtype=np.float32)
    if arr.size < 4:
        return "none"
    first4 = arr[:4]
    if float(np.max(first4)) <= 0.0:
        return "none"
    return OSM_TAG_NAMES[int(np.argmax(first4))]


def _select_city_res(
    start_cells_by_res: Dict[int, List[str]],
    candidates: Iterable[int],
    min_records: int,
    min_cities: int,
) -> int:
    selected = None
    for res in candidates:
        cells = start_cells_by_res.get(res, [])
        if not cells:
            continue
        counts = Counter(cells)
        eligible = [k for k, v in counts.items() if v >= min_records]
        if len(eligible) >= min_cities:
            return res
        selected = res if selected is None else min(selected, res)
    return selected if selected is not None else 2


def main():
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)
    osm_path = Path(args.osm_context)

    if not input_path.exists():
        raise FileNotFoundError(f"input not found: {input_path}")

    df = pd.read_pickle(input_path)
    records = df.to_dict(orient="records")

    osm_context, osm_res = _load_osm_context(osm_path)

    city_res_candidates = [
        int(x.strip())
        for x in args.city_res_candidates.split(",")
        if x.strip()
    ]

    parsed = []
    start_cells_by_res: Dict[int, List[str]] = {r: [] for r in city_res_candidates}

    for rec in records:
        out = _iter_points(rec)
        if out is None:
            continue
        lat, lon, ts = out

        poi_cells = [_h3_cell(float(a), float(b), args.poi_res) for a, b in zip(lat, lon)]
        poi_cells = _dedup_consecutive(poi_cells)
        if len(poi_cells) < args.min_poi_len:
            continue

        for r in city_res_candidates:
            start_cells_by_res.setdefault(r, []).append(_h3_cell(float(lat[0]), float(lon[0]), r))

        parsed.append((rec, lat, lon, ts, poi_cells))
        if args.max_rows > 0 and len(parsed) >= args.max_rows:
            break

    if not parsed:
        raise RuntimeError("no valid records found for POI sample")

    city_res = _select_city_res(
        start_cells_by_res,
        candidates=city_res_candidates,
        min_records=args.target_min_city_records,
        min_cities=args.target_city_count,
    )

    out_rows = []
    for rec, lat, lon, ts, poi_cells in parsed:
        poi_seq = []
        for cell in poi_cells:
            tag = "none"
            if osm_context and osm_res is not None:
                parent = cell
                if _h3_resolution(cell) > osm_res:
                    parent = _h3_parent(cell, osm_res)
                vec = osm_context.get(parent)
                if vec is not None:
                    tag = _dominant_osm_tag(vec)
            poi_seq.append(f"{cell}|{tag}")

        city_cell = _h3_cell(float(lat[0]), float(lon[0]), city_res)
        user_o = _h3_cell(float(lat[0]), float(lon[0]), args.user_res)
        user_d = _h3_cell(float(lat[-1]), float(lon[-1]), args.user_res)
        user_id = f"u_{user_o}_{user_d}"

        row = dict(rec)
        row["trajectory"] = np.stack([lat, lon], axis=1).tolist()
        row["time"] = ts.tolist()
        row["poi_sequence"] = poi_seq
        row["city_name"] = f"city_{city_cell}"
        row["user_id"] = user_id
        out_rows.append(row)

    out_df = pd.DataFrame(out_rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_pickle(output_path)

    city_counts = Counter(out_df["city_name"].tolist())
    top = city_counts.most_common(10)
    eligible = sum(1 for _, v in city_counts.items() if v >= args.target_min_city_records)

    print(f"saved {len(out_df)} rows to {output_path}")
    print(f"city_res={city_res} unique_cities={len(city_counts)} eligible_cities={eligible}")
    print(f"top_cities={top}")


if __name__ == "__main__":
    main()

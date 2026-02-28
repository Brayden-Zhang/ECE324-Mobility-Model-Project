import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Iterable, Optional

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.hmt_dataset import WorldTraceZipIterableDataset, prepare_record  # noqa: E402

try:
    import h3
except Exception:  # optional
    h3 = None


def parse_args():
    parser = argparse.ArgumentParser(description="Build H3 vocab from a local WorldTrace pickle/parquet")
    parser.add_argument("--local_data", type=str, default="")
    parser.add_argument("--data_mode", type=str, default="local", choices=["local", "hf_zip"])
    parser.add_argument("--hf_name", type=str, default="OpenTrace/WorldTrace")
    parser.add_argument("--worldtrace_file", type=str, default="Trajectory.zip")
    parser.add_argument("--worldtrace_local_path", type=str, default="")
    parser.add_argument("--res0", type=int, default=9)
    parser.add_argument("--res1", type=int, default=7)
    parser.add_argument("--res2", type=int, default=5)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--max_records", type=int, default=0)
    parser.add_argument("--max_cells_l0", type=int, default=0)
    parser.add_argument("--max_cells_l1", type=int, default=0)
    parser.add_argument("--max_cells_l2", type=int, default=0)
    return parser.parse_args()


def _h3_cell(lat: float, lon: float, res: int):
    if h3 is None:
        raise RuntimeError("h3 is not installed; install h3 to build vocab")
    if hasattr(h3, "latlng_to_cell"):
        return h3.latlng_to_cell(lat, lon, res)
    return h3.geo_to_h3(lat, lon, res)


def load_records(path: str) -> Iterable[dict]:
    import pandas as pd

    if path.endswith(".parquet"):
        df = pd.read_parquet(path)
    else:
        df = pd.read_pickle(path)
    return df.to_dict(orient="records")


def iter_records(args) -> Iterable[dict]:
    if args.data_mode == "hf_zip":
        ds = WorldTraceZipIterableDataset(
            hf_name=args.hf_name,
            filename=args.worldtrace_file,
            max_len=200,
            mask_ratio=0.0,
            shuffle_buffer=1,
            take=args.max_records if args.max_records > 0 else None,
            local_path=args.worldtrace_local_path,
            seed=42,
        )
        for rec in ds._iter_raw_records():  # noqa: SLF001 (intentional internal use)
            yield rec
    else:
        if not args.local_data:
            raise ValueError("local_data is required for data_mode=local")
        for rec in load_records(args.local_data):
            yield rec


def main():
    args = parse_args()
    counter_l0 = Counter()
    counter_l1 = Counter()
    counter_l2 = Counter()

    seen = 0
    for rec in iter_records(args):
        prepared = prepare_record(rec)
        if prepared is None:
            continue
        lat = prepared["lat"]
        lon = prepared["lon"]
        for la, lo in zip(lat, lon):
            c0 = _h3_cell(float(la), float(lo), args.res0)
            c1 = _h3_cell(float(la), float(lo), args.res1)
            c2 = _h3_cell(float(la), float(lo), args.res2)
            counter_l0[str(c0)] += 1
            counter_l1[str(c1)] += 1
            counter_l2[str(c2)] += 1
        seen += 1
        if args.max_records > 0 and seen >= args.max_records:
            break

    def top_cells(counter: Counter, max_cells: int) -> list:
        if max_cells and max_cells > 0:
            return [cell for cell, _ in counter.most_common(max_cells)]
        return list(counter.keys())

    cells_l0 = top_cells(counter_l0, args.max_cells_l0)
    cells_l1 = top_cells(counter_l1, args.max_cells_l1)
    cells_l2 = top_cells(counter_l2, args.max_cells_l2)

    payload = {
        "res0": args.res0,
        "res1": args.res1,
        "res2": args.res2,
        "cells_l0": sorted(cells_l0),
        "cells_l1": sorted(cells_l1),
        "cells_l2": sorted(cells_l2),
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(payload, f)
    print(
        f"saved h3 vocab to {out} (l0={len(payload['cells_l0'])}, l1={len(payload['cells_l1'])}, l2={len(payload['cells_l2'])})"
    )


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
import argparse
import json
import re
import zipfile
from collections import defaultdict
from datetime import date
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import numpy as np
import pandas as pd


DATE_RE = re.compile(r"(\d{4}-\d{2}-\d{2})")


def parse_date(text: str) -> Optional[date]:
    if not text:
        return None
    m = DATE_RE.search(text)
    if not m:
        return None
    y, mo, d = (int(x) for x in m.group(1).split("-"))
    return date(y, mo, d)


def iter_zip_paths(input_dir: Path) -> Iterable[Path]:
    for p in sorted(input_dir.glob("*.zip")):
        yield p


def load_csv_from_zip(
    zf: zipfile.ZipFile, name: str, usecols: Optional[list]
) -> pd.DataFrame:
    with zf.open(name) as f:
        if usecols:
            return pd.read_csv(f, usecols=usecols)
        return pd.read_csv(f)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Prepare Movement Distribution data into month-level tensors."
    )
    parser.add_argument(
        "--input_dir",
        required=True,
        help="Directory with Movement Distribution zip files",
    )
    parser.add_argument("--output", required=True, help="Output .npz path")
    parser.add_argument(
        "--max_zips", type=int, default=0, help="Limit number of zip files (0 = all)"
    )
    parser.add_argument(
        "--max_csv_per_zip", type=int, default=0, help="Limit CSVs per zip (0 = all)"
    )
    parser.add_argument("--aggregate", choices=["monthly", "daily"], default="monthly")
    parser.add_argument(
        "--latest_months", type=int, default=0, help="Keep only latest N months"
    )
    parser.add_argument(
        "--keep_columns", action="store_true", help="Store per-region metadata arrays"
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        raise FileNotFoundError(f"input_dir not found: {input_dir}")

    # Accumulators: (region_id, time_key, category) -> [sum, count]
    sums = defaultdict(float)
    counts = defaultdict(int)
    region_meta: Dict[str, dict] = {}
    categories = set()
    time_keys = set()

    zip_paths = list(iter_zip_paths(input_dir))
    if args.max_zips > 0:
        zip_paths = zip_paths[: args.max_zips]

    for zp in zip_paths:
        with zipfile.ZipFile(zp, "r") as zf:
            names = [
                n
                for n in zf.namelist()
                if n.lower().endswith(".csv") and "macosx" not in n.lower()
            ]
            names = sorted(names)
            if args.max_csv_per_zip > 0:
                names = names[: args.max_csv_per_zip]
            for name in names:
                # Load only required columns when possible.
                expected_cols = [
                    "gadm_id",
                    "gadm_name",
                    "country",
                    "polygon_level",
                    "home_to_ping_distance_category",
                    "distance_category_ping_fraction",
                    "ds",
                ]
                try:
                    df = load_csv_from_zip(zf, name, usecols=expected_cols)
                except ValueError:
                    df = load_csv_from_zip(zf, name, usecols=None)
                if df.empty:
                    continue

                # expected columns
                cols = {c.lower(): c for c in df.columns}
                id_col = cols.get("gadm_id")
                name_col = cols.get("gadm_name")
                country_col = cols.get("country")
                level_col = cols.get("polygon_level")
                cat_col = cols.get("home_to_ping_distance_category")
                frac_col = cols.get("distance_category_ping_fraction")
                date_col = cols.get("ds")
                if not all([id_col, cat_col, frac_col]):
                    continue

                # derive date -> month
                ds_val = None
                if date_col and date_col in df.columns:
                    ds_val = str(df[date_col].iloc[0])
                if not ds_val:
                    ds_val = name
                d = parse_date(ds_val)
                if d is None:
                    continue
                if args.aggregate == "monthly":
                    time_key = f"{d.year:04d}-{d.month:02d}"
                else:
                    time_key = f"{d.year:04d}-{d.month:02d}-{d.day:02d}"
                time_keys.add(time_key)

                # Track region metadata once per file.
                if id_col in df.columns:
                    meta_cols = [
                        c for c in [id_col, name_col, country_col, level_col] if c
                    ]
                    meta_df = df[meta_cols].drop_duplicates(subset=[id_col])
                    for _, row in meta_df.iterrows():
                        rid = str(row[id_col])
                        if rid in region_meta:
                            continue
                        region_meta[rid] = {
                            "gadm_name": str(row[name_col]) if name_col else "",
                            "country": str(row[country_col]) if country_col else "",
                            "polygon_level": int(row[level_col]) if level_col else -1,
                        }

                grouped = (
                    df.groupby([id_col, cat_col], sort=False)[frac_col]
                    .agg(["sum", "count"])
                    .reset_index()
                )
                for _, row in grouped.iterrows():
                    rid = str(row[id_col])
                    cat = str(row[cat_col])
                    categories.add(cat)
                    key = (rid, time_key, cat)
                    sums[key] += float(row["sum"])
                    counts[key] += int(row["count"])

    regions = sorted(region_meta.keys())
    categories = sorted(categories)
    time_keys = sorted(time_keys)
    if args.latest_months and args.latest_months > 0:
        time_keys = time_keys[-args.latest_months :]

    region_index = {r: i for i, r in enumerate(regions)}
    cat_index = {c: i for i, c in enumerate(categories)}
    time_index = {t: i for i, t in enumerate(time_keys)}

    num_records = len(regions) * len(time_keys)
    dist = np.zeros((num_records, len(categories)), dtype=np.float32)
    region_idx = np.zeros((num_records,), dtype=np.int32)
    time_idx = np.zeros((num_records,), dtype=np.int32)

    k = 0
    for r in regions:
        r_i = region_index[r]
        for t in time_keys:
            region_idx[k] = r_i
            time_idx[k] = time_index[t]
            for c in categories:
                key = (r, t, c)
                if key in sums:
                    dist[k, cat_index[c]] = sums[key] / max(1, counts[key])
            k += 1

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        dist=dist,
        region_idx=region_idx,
        time_idx=time_idx,
        regions=np.array(regions, dtype=object),
        time_keys=np.array(time_keys, dtype=object),
        categories=np.array(categories, dtype=object),
    )

    if args.keep_columns:
        meta_path = out_path.with_suffix(".meta.json")
        meta_path.write_text(json.dumps(region_meta, indent=2))

    print(
        f"saved {out_path} records={num_records} "
        f"regions={len(regions)} "
        f"times={len(time_keys)} "
        f"categories={len(categories)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

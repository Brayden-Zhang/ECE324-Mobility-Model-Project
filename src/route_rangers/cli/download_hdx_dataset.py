#!/usr/bin/env python3
import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Iterable, Optional, Tuple
from urllib.parse import urlencode, urlparse
from urllib.request import Request, urlopen
from datetime import date


API_BASE = "https://data.humdata.org/api/3/action"


def _http_json(url: str) -> dict:
    req = Request(url, headers={"User-Agent": "trajfm-hdx/1.0"})
    with urlopen(req) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _sanitize_filename(name: str) -> str:
    name = name.strip().replace(" ", "_")
    name = re.sub(r"[^A-Za-z0-9._-]+", "-", name)
    return name[:180] if len(name) > 180 else name


def _ext(name: str) -> str:
    ext = os.path.splitext(name)[1].lower()
    if ext and len(ext) <= 6:
        return ext
    return ""


def _infer_filename(resource: dict) -> str:
    name = resource.get("name") or ""
    url = resource.get("url") or ""
    url_base = os.path.basename(urlparse(url).path) if url else ""
    if name:
        if _ext(name):
            return _sanitize_filename(name)
        if url_base and _ext(url_base):
            return _sanitize_filename(f"{name}{_ext(url_base)}")
        return _sanitize_filename(name)
    if url_base:
        return _sanitize_filename(url_base)
    return _sanitize_filename(resource.get("id", "resource"))


def _iter_resources(
    resources: Iterable[dict], formats: Optional[set]
) -> Iterable[dict]:
    for res in resources:
        fmt = (res.get("format") or "").lower()
        if formats and fmt not in formats:
            continue
        yield res


def _extract_dates(text: str) -> list:
    if not text:
        return []
    matches = re.findall(r"(\d{4}-\d{2}-\d{2})", text)
    out = []
    for m in matches:
        try:
            y, mo, d = (int(x) for x in m.split("-"))
            out.append(date(y, mo, d))
        except Exception:
            continue
    return out


def _resource_date(resource: dict) -> Optional[Tuple[date, date]]:
    parts = []
    for key in ("name", "url", "description"):
        val = resource.get(key) or ""
        parts.extend(_extract_dates(val))
    if not parts:
        return None
    parts = sorted(parts)
    return parts[0], parts[-1]


def _download(url: str, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    req = Request(url, headers={"User-Agent": "trajfm-hdx/1.0"})
    with urlopen(req) as resp, open(out_path, "wb") as f:
        while True:
            chunk = resp.read(1024 * 1024)
            if not chunk:
                break
            f.write(chunk)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Download HDX dataset resources via CKAN API"
    )
    parser.add_argument(
        "--dataset",
        required=True,
        help="HDX dataset slug or id, e.g., movement-distribution",
    )
    parser.add_argument(
        "--output_dir",
        default="",
        help="Output directory (default: data/hdx/<dataset>)",
    )
    parser.add_argument(
        "--download",
        action="store_true",
        help="Download resources instead of listing only",
    )
    parser.add_argument(
        "--formats",
        default="",
        help=(
            "Comma-separated list of formats to download "
            "(e.g., csv,geojson,zip). Empty means all."
        ),
    )
    parser.add_argument(
        "--max_resources",
        type=int,
        default=0,
        help="Limit downloads/listing to N resources",
    )
    parser.add_argument(
        "--latest_months",
        type=int,
        default=0,
        help=(
            "If >0, select latest N unique months based on dates "
            "in resource names/urls."
        ),
    )
    args = parser.parse_args()

    out_dir = (
        Path(args.output_dir)
        if args.output_dir
        else Path("data") / "hdx" / args.dataset
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    params = {"id": args.dataset}
    url = f"{API_BASE}/package_show?{urlencode(params)}"
    payload = _http_json(url)
    if not payload.get("success"):
        print(f"ERROR: CKAN request failed for {args.dataset}", file=sys.stderr)
        return 1

    result = payload.get("result", {})
    meta_path = out_dir / "package_show.json"
    meta_path.write_text(json.dumps(result, indent=2))
    resources = result.get("resources", [])

    fmt_filter = None
    if args.formats:
        fmt_filter = {f.strip().lower() for f in args.formats.split(",") if f.strip()}

    selected = list(_iter_resources(resources, fmt_filter))
    if args.latest_months and args.latest_months > 0:
        dated = []
        for res in selected:
            dr = _resource_date(res)
            if dr is None:
                continue
            start, end = dr
            month_key = f"{start.year:04d}-{start.month:02d}"
            dated.append((end, month_key, res))
        dated.sort(key=lambda x: x[0])
        # keep latest N unique months
        chosen = []
        seen_months = set()
        for end, month_key, res in reversed(dated):
            if month_key in seen_months:
                continue
            chosen.append(res)
            seen_months.add(month_key)
            if len(chosen) >= args.latest_months:
                break
        selected = list(reversed(chosen))

    count = 0
    for res in selected:
        count += 1
        if args.max_resources and count > args.max_resources:
            break
        url = res.get("url") or ""
        fmt = (res.get("format") or "").lower()
        name = _infer_filename(res)
        if fmt and not _ext(name):
            name = f"{name}.{fmt}"
        out_path = out_dir / name
        if args.download:
            print(f"downloading {name} -> {out_path}")
            _download(url, out_path)
        else:
            print(f"{name}\t{fmt}\t{url}")

    print(f"saved metadata to {meta_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

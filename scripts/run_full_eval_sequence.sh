#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

VENV_PYTHON="${VENV_PYTHON:-.venv/bin/python}"
PYTHONPATH_VALUE="${PYTHONPATH_VALUE:-src}"

CHECKPOINT="${CHECKPOINT:-checkpoints/hmt_nohash_full_step_10000.pt}"
LOCAL_DATA="${LOCAL_DATA:-data/samples/worldtrace_sample.pkl}"
SAMPLE_LIMIT="${SAMPLE_LIMIT:-2000}"

STAMP="$(date +%Y%m%d_%H%M%S)"

ROBUST_JSON="cache/unitraj_eval_regression_robust_${STAMP}.json"
BENCH_JSON="cache/benchmark_${STAMP}.json"
CHANGE_JSON="cache/change_detection_${STAMP}.json"
REPORT_PREFIX="foundation_downstream_report_${STAMP}"

echo "[1/5] robust unitraj regression eval"
PYTHONPATH="$PYTHONPATH_VALUE" "$VENV_PYTHON" -m route_rangers.cli.run_unitraj_eval \
	--checkpoint "$CHECKPOINT" \
	--local_data "$LOCAL_DATA" \
	--split_mode all \
	--task both \
	--use-regression \
	--coord_noise_std_m 30 \
	--input_drop_ratio 0.2 \
	--sample_limit "$SAMPLE_LIMIT" \
	--output "$ROBUST_JSON"

echo "[2/5] benchmark core"
PYTHONPATH="$PYTHONPATH_VALUE" "$VENV_PYTHON" -m route_rangers.cli.run_benchmarks \
	--checkpoint "$CHECKPOINT" \
	--local_data "$LOCAL_DATA" \
	--split_mode random \
	--sample_limit "$SAMPLE_LIMIT" \
	--output "$BENCH_JSON"

echo "[3/5] change detection"
PYTHONPATH="$PYTHONPATH_VALUE" "$VENV_PYTHON" -m route_rangers.cli.run_change_detection \
	--checkpoint "$CHECKPOINT" \
	--local_data "$LOCAL_DATA" \
	--sample_limit "$SAMPLE_LIMIT" \
	--output "$CHANGE_JSON"

echo "[4/5] regenerate consolidated report"
PYTHONPATH="$PYTHONPATH_VALUE" "$VENV_PYTHON" -m route_rangers.cli.generate_foundation_report \
	--cache_dir cache \
	--output_dir reports \
	--filename_prefix "$REPORT_PREFIX"

echo "[5/5] compute win/tie/loss on comparable metrics"
PYTHONPATH="$PYTHONPATH_VALUE" "$VENV_PYTHON" - <<'PY'
import csv
from pathlib import Path

reports = sorted(Path("reports").glob("foundation_downstream_report_*.csv"), key=lambda p: p.stat().st_mtime)
if not reports:
		raise SystemExit("No report CSV found in reports/")
report = reports[-1]
rows = list(csv.DictReader(report.open()))
comp = [r for r in rows if r["status"] == "ok" and r["baseline_value"] not in ("", None)]
win = tie = lose = 0
for r in comp:
		d = float(r["delta"]) if r["delta"] else 0.0
		if d > 1e-12:
				win += 1
		elif d < -1e-12:
				lose += 1
		else:
				tie += 1

print(f"report={report}")
print(f"comparable={len(comp)} win={win} tie={tie} lose={lose}")
if lose:
		print("losses:")
		for r in comp:
				d = float(r["delta"]) if r["delta"] else 0.0
				if d < -1e-12:
						print(f"  {r['task']}::{r['metric']} delta={d:.6f}")
PY

echo "done"
echo "robust_eval=$ROBUST_JSON"
echo "benchmark=$BENCH_JSON"
echo "change_detection=$CHANGE_JSON"
echo "report_prefix=$REPORT_PREFIX"

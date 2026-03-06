import argparse
import json
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Summarize UniTraj-style eval JSON outputs"
    )
    parser.add_argument("files", nargs="+", help="eval JSON files")
    return parser.parse_args()


def _get(payload, path, default=0.0):
    cur = payload
    for key in path:
        if key not in cur:
            return default
        cur = cur[key]
    return cur


def main():
    args = parse_args()
    print("file,split,task,split_set,mae_m,rmse_m,n")
    for f in args.files:
        p = Path(f)
        with open(p, "r") as fp:
            payload = json.load(fp)
        splits = payload.get("splits", {})
        for split_name, split_payload in splits.items():
            for task in ("recovery", "prediction"):
                if task not in split_payload:
                    continue
                for split_set in ("train", "val", "test"):
                    mae = _get(split_payload, [task, split_set, "mae_m"])
                    rmse = _get(split_payload, [task, split_set, "rmse_m"])
                    n = _get(split_payload, [task, split_set, "n"], 0)
                    print(
                        f"{p.name},{split_name},{task},{split_set},{mae:.2f},{rmse:.2f},{n}"
                    )


if __name__ == "__main__":
    main()

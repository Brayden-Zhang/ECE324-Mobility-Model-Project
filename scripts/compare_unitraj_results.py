import argparse
import json
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Compare HMT UniTraj-style evals vs UniTraj external eval")
    parser.add_argument("--hmt", nargs="+", required=True, help="HMT eval JSON files")
    parser.add_argument("--unitraj", required=True, help="UniTraj external eval JSON file")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--mode", default="all", help="split mode key in HMT eval (e.g., all/random/temporal)")
    return parser.parse_args()


def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def main():
    args = parse_args()
    unitraj = load_json(args.unitraj)
    unitraj_metrics = unitraj.get("metrics", {})

    print("model,task,mae_m,rmse_m,n")
    for hmt_path in args.hmt:
        payload = load_json(hmt_path)
        name = Path(hmt_path).stem
        split_payload = payload.get("splits", {}).get(args.mode, {})
        for task, ext_metrics in unitraj_metrics.items():
            hmt_task = split_payload.get(task, {})
            hmt_metrics = hmt_task.get(args.split, {})
            if hmt_metrics:
                print(
                    f"{name},{task},{hmt_metrics.get('mae_m', 0.0):.2f},{hmt_metrics.get('rmse_m', 0.0):.2f},{hmt_metrics.get('n', 0)}"
                )
        # add UniTraj row once per file for easier CSV diff
        for task, m in unitraj_metrics.items():
            print(f"unitraj,{task},{m.get('mae_m', 0.0):.2f},{m.get('rmse_m', 0.0):.2f},{m.get('n', 0)}")


if __name__ == "__main__":
    main()

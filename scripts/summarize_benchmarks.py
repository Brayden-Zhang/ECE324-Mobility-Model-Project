import argparse
import json
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Summarize benchmark JSON outputs")
    parser.add_argument("files", nargs="+", help="benchmark JSON files")
    return parser.parse_args()


def get_metric(split_payload, path, default=0.0):
    cur = split_payload
    for key in path:
        if key not in cur:
            return default
        cur = cur[key]
    return cur


def main():
    args = parse_args()
    print(
        "file,split,next_top1,dest_top1,recon_l0,recon_l1,recon_l2,"
        "next_mae_m,next_rmse_m,dest_mae_m,dest_rmse_m"
    )
    for f in args.files:
        p = Path(f)
        with open(p, "r") as fp:
            payload = json.load(fp)
        for split_name, split_payload in payload.get("splits", {}).items():
            next_top1 = get_metric(split_payload, ["next_location_probe", "test", "top1"])
            dest_top1 = get_metric(split_payload, ["destination_probe", "test", "top1"])
            recon_l0 = get_metric(split_payload, ["reconstruction", "recon_acc_l0"])
            recon_l1 = get_metric(split_payload, ["reconstruction", "recon_acc_l1"])
            recon_l2 = get_metric(split_payload, ["reconstruction", "recon_acc_l2"])
            next_mae = get_metric(split_payload, ["next_location_regression_probe", "test", "mae_m"])
            next_rmse = get_metric(split_payload, ["next_location_regression_probe", "test", "rmse_m"])
            dest_mae = get_metric(split_payload, ["destination_regression_probe", "test", "mae_m"])
            dest_rmse = get_metric(split_payload, ["destination_regression_probe", "test", "rmse_m"])
            print(
                f"{p.name},{split_name},{next_top1:.4f},{dest_top1:.4f},{recon_l0:.4f},{recon_l1:.4f},{recon_l2:.4f},"
                f"{next_mae:.2f},{next_rmse:.2f},{dest_mae:.2f},{dest_rmse:.2f}"
            )


if __name__ == "__main__":
    main()

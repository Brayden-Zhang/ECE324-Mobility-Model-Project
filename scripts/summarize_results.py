import argparse
import json
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Summarize TrajectoryFM experiment JSON files")
    parser.add_argument("results", nargs="+", help="JSON paths (e.g., cache/results_*.json)")
    return parser.parse_args()


def fmt(x):
    if x is None:
        return "n/a"
    return f"{x:.4f}"


def main():
    args = parse_args()
    rows = []
    for p in args.results:
        path = Path(p)
        with open(path, "r") as f:
            payload = json.load(f)
        test = payload.get("final_test", {})
        rows.append(
            {
                "run": path.name,
                "dest_top1": test.get("dest_top1"),
                "dest_top5": test.get("dest_top5"),
                "token_acc_l0": test.get("token_acc_l0"),
                "token_acc_l2": test.get("token_acc_l2"),
                "flow_loss": test.get("flow_loss"),
            }
        )

    rows.sort(
        key=lambda r: (
            r["dest_top1"] if r["dest_top1"] is not None else -1.0,
            r["dest_top5"] if r["dest_top5"] is not None else -1.0,
            r["token_acc_l0"] if r["token_acc_l0"] is not None else -1.0,
        ),
        reverse=True,
    )

    print("run,dest_top1,dest_top5,token_acc_l0,token_acc_l2,flow_loss")
    for r in rows:
        print(
            f"{r['run']},{fmt(r['dest_top1'])},{fmt(r['dest_top5'])},"
            f"{fmt(r['token_acc_l0'])},{fmt(r['token_acc_l2'])},{fmt(r['flow_loss'])}"
        )


if __name__ == "__main__":
    main()

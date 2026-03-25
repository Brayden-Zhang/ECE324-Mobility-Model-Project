import json

def summarize(file_path):
    with open(file_path, "r") as f:
        data = json.load(f)

    print("\n LENGTH SENSITIVITY SUMMARY")

    for split, split_data in data.get("splits", {}).items():
        print(f"\nSplit: {split}")

        # handle nested structure safely
        metrics = split_data.get("metrics", split_data)

        for bucket in ["short", "medium", "long"]:
            if bucket in metrics:
                bucket_data = metrics[bucket]

                # try common metric names
                if "dest_top1" in bucket_data:
                    val = bucket_data["dest_top1"]
                elif isinstance(bucket_data, dict) and "mean" in bucket_data:
                    val = bucket_data["mean"]
                else:
                    continue

                print(f"{bucket}: {val:.3f}")

if __name__ == "__main__":
    summarize("cache/length_sensitivity_latest.json")
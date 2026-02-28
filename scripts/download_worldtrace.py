import argparse
import os

from datasets import load_dataset


def parse_args():
    parser = argparse.ArgumentParser(description="Download WorldTrace dataset from HF")
    parser.add_argument("--name", default="OpenTrace/WorldTrace")
    parser.add_argument("--split", default="train")
    parser.add_argument("--output", default="data/worldtrace")
    parser.add_argument("--num_proc", type=int, default=4)
    parser.add_argument("--max_samples", type=int, default=0)
    return parser.parse_args()


def main():
    args = parse_args()
    ds = load_dataset(args.name, split=args.split)
    if args.max_samples:
        ds = ds.select(range(args.max_samples))
    os.makedirs(args.output, exist_ok=True)
    path = os.path.join(args.output, f"{args.split}.parquet")
    ds.to_parquet(path)
    print(f"saved to {path}")


if __name__ == "__main__":
    main()

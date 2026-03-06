import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from route_rangers.cli import run_benchmarks as rb


def parse_args():
    parser = argparse.ArgumentParser(description="Commuting zone destination probe")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--local_data", type=str, required=True)
    parser.add_argument("--cz_csv", type=str, required=True)
    parser.add_argument("--max_len", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_samples", type=int, default=50000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", type=str, default="")
    return parser.parse_args()


def masked_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    m = mask.unsqueeze(-1).float()
    return (x * m).sum(dim=1) / m.sum(dim=1).clamp(min=1.0)


def load_commuting_zones(path: str):
    import pandas as pd
    from shapely import wkt
    from shapely.strtree import STRtree

    df = pd.read_csv(path)
    if "geography" not in df.columns or "fbcz_id" not in df.columns:
        raise ValueError("commuting zone CSV must have geography and fbcz_id columns")
    geoms = [wkt.loads(g) for g in df["geography"].tolist()]
    ids = df["fbcz_id"].astype(str).tolist()
    tree = STRtree(geoms)
    geom_index = {id(g): i for i, g in enumerate(geoms)}
    return tree, geom_index, ids


def assign_zones(tree, geom_index, ids, points):
    from shapely.geometry import Point

    labels = []
    for lat, lon in points:
        pt = Point(float(lon), float(lat))
        cand = tree.query(pt)
        label = None
        for geom in cand:
            if geom.contains(pt):
                idx = geom_index[id(geom)]
                label = ids[idx]
                break
        labels.append(label)
    return labels


def main():
    args = parse_args()
    rb.set_seed(args.seed)

    pack = rb.load_backbone(args.checkpoint, device=args.device, override_max_len=args.max_len, disable_graph=False)
    raw_records = rb.load_local_data(args.local_data)
    dataset = rb.FixedTrajectoryDataset(raw_records, max_len=args.max_len, sample_limit=args.max_samples)

    tree, geom_index, ids = load_commuting_zones(args.cz_csv)

    dest_points = []
    for sample in dataset.samples:
        coords = sample["coords"].numpy()
        attn = sample["attention_mask"].numpy()
        vlen = int(attn.sum())
        if vlen <= 0:
            dest_points.append((np.nan, np.nan))
            continue
        dest = coords[vlen - 1]
        dest_points.append((dest[0], dest[1]))

    labels = assign_zones(tree, geom_index, ids, dest_points)
    label_set = sorted({l for l in labels if l is not None})
    label_map = {l: i for i, l in enumerate(label_set)}

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=rb.collate_fixed,
    )

    feats = []
    y = []
    idx_base = 0
    for batch in loader:
        coords = batch["coords"].to(args.device)
        timestamps = batch["timestamps"].to(args.device)
        attention = batch["attention_mask"].to(args.device)

        outputs, _, _, _, _ = rb.forward_backbone(batch, pack, device=args.device, max_len=args.max_len, mask=None)
        pooled = masked_mean(outputs["step_hidden"], attention).detach().cpu().numpy()

        bsz = pooled.shape[0]
        for i in range(bsz):
            label = labels[idx_base + i]
            if label is None:
                continue
            feats.append(pooled[i])
            y.append(label_map[label])
        idx_base += bsz

    if len(y) < 10:
        raise RuntimeError("not enough labeled samples for commuting zone probe")

    X = np.stack(feats, axis=0)
    y = np.asarray(y, dtype=np.int64)

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, random_state=args.seed, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=args.seed, stratify=y_temp
    )

    clf = LogisticRegression(max_iter=200, n_jobs=4, multi_class="multinomial")
    clf.fit(X_train, y_train)

    val_acc = accuracy_score(y_val, clf.predict(X_val))
    test_acc = accuracy_score(y_test, clf.predict(X_test))

    metrics = {
        "cz_val_acc": float(val_acc),
        "cz_test_acc": float(test_acc),
        "num_classes": int(len(label_set)),
        "n": int(len(y)),
    }
    print(metrics)

    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w") as f:
            json.dump(
                {
                    "checkpoint": args.checkpoint,
                    "data": args.local_data,
                    "cz_csv": args.cz_csv,
                    "metrics": metrics,
                },
                f,
                indent=2,
            )
        print(f"saved {out}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Run MoveGPT-style next-POI and user-identification probes."""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from route_rangers.cli import run_benchmarks as rb


POI_KEYS = (
    "poi_sequence",
    "poi_ids",
    "poi_id_seq",
    "pois",
    "trajectory_poi_ids",
)
CITY_KEYS = ("city", "city_name", "region", "dataset_city")
USER_KEYS = ("user_id", "uid", "user", "userId")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate next-POI and user-identification probes"
    )
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--local_data", type=str, required=True)
    parser.add_argument("--max_len", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sample_limit", type=int, default=0)
    parser.add_argument("--probe_epochs", type=int, default=8)
    parser.add_argument("--probe_lr", type=float, default=2e-3)
    parser.add_argument("--probe_weight_decay", type=float, default=1e-4)
    parser.add_argument("--probe_batch_size", type=int, default=2048)
    parser.add_argument("--max_points", type=int, default=300000)
    parser.add_argument("--split_mode", choices=["random", "temporal"], default="random")
    parser.add_argument("--train_cities", type=str, default="")
    parser.add_argument("--test_cities", type=str, default="")
    parser.add_argument("--skip_user_identification", action="store_true")
    parser.add_argument("--output", type=str, default="")
    return parser.parse_args()


def _pick_first(record: dict, keys: Sequence[str]):
    for k in keys:
        if k in record and record[k] is not None:
            return record[k]
    return None


def extract_poi_seq(record: dict) -> Optional[List[str]]:
    raw = _pick_first(record, POI_KEYS)
    if raw is None:
        return None
    if isinstance(raw, np.ndarray):
        raw = raw.tolist()
    if not isinstance(raw, (list, tuple)):
        return None
    out = []
    for x in raw:
        if x is None:
            continue
        s = str(x).strip()
        if not s:
            continue
        out.append(s)
    return out if len(out) >= 2 else None


def extract_city(record: dict) -> str:
    city = _pick_first(record, CITY_KEYS)
    return str(city).strip() if city is not None else "unknown"


def extract_user(record: dict) -> str:
    user = _pick_first(record, USER_KEYS)
    return str(user).strip() if user is not None else "unknown"


class POIDataset(Dataset):
    def __init__(self, records: List[dict], max_len: int, sample_limit: int = 0):
        samples = []
        for rec in records:
            poi_seq = extract_poi_seq(rec)
            if poi_seq is None:
                continue
            base = rb.preprocess_record(rec, max_len=max_len)
            if base is None:
                continue
            vlen = int(base["attention_mask"].sum().item())
            if vlen < 2:
                continue
            aligned = min(vlen, len(poi_seq))
            if aligned < 2:
                continue

            if aligned != vlen:
                base["coords"][aligned:] = 0
                base["timestamps"][aligned:] = 0
                base["attention_mask"][aligned:] = 0
                vlen = aligned

            samples.append(
                {
                    "coords": base["coords"],
                    "timestamps": base["timestamps"],
                    "attention_mask": base["attention_mask"],
                    "start_ts": base["start_ts"],
                    "poi_seq": poi_seq[:vlen],
                    "city": extract_city(rec),
                    "user": extract_user(rec),
                }
            )
            if sample_limit > 0 and len(samples) >= sample_limit:
                break
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        return self.samples[idx]


def collate_poi(batch: List[dict]) -> dict:
    return {
        "coords": torch.stack([b["coords"] for b in batch], dim=0),
        "timestamps": torch.stack([b["timestamps"] for b in batch], dim=0),
        "attention_mask": torch.stack([b["attention_mask"] for b in batch], dim=0),
        "start_ts": torch.tensor([b["start_ts"] for b in batch], dtype=torch.float32),
        "poi_seq": [b["poi_seq"] for b in batch],
        "city": [b["city"] for b in batch],
        "user": [b["user"] for b in batch],
    }


def parse_city_list(raw: str) -> List[str]:
    if not raw.strip():
        return []
    return [x.strip() for x in raw.split(",") if x.strip()]


def split_indices_by_city(
    dataset: POIDataset,
    seed: int,
    train_cities: Optional[List[str]] = None,
    test_cities: Optional[List[str]] = None,
) -> Tuple[List[int], List[int], List[int], Dict[str, int]]:
    train_cities = train_cities or []
    test_cities = test_cities or []
    if not train_cities and not test_cities:
        tr, va, te = rb.split_indices(dataset, mode="random", seed=seed)
        return tr, va, te, {}

    train_cities_set = set(train_cities)
    test_cities_set = set(test_cities)

    test_idx = []
    rest_idx = []
    for i, s in enumerate(dataset.samples):
        city = s["city"]
        if test_cities_set and city in test_cities_set:
            test_idx.append(i)
        elif train_cities_set and city in train_cities_set:
            rest_idx.append(i)
        elif not train_cities_set and not test_cities_set:
            rest_idx.append(i)

    if not rest_idx or not test_idx:
        return [], [], [], {}

    rng = np.random.default_rng(seed)
    rng.shuffle(rest_idx)
    n_train = max(1, int(len(rest_idx) * 0.85))
    train_idx = rest_idx[:n_train]
    val_idx = rest_idx[n_train:] or rest_idx[-1:]
    city_counts = {
        "train_records": len(train_idx),
        "val_records": len(val_idx),
        "test_records": len(test_idx),
    }
    return train_idx, val_idx, test_idx, city_counts


def build_vocab(values: List[str]) -> Dict[str, int]:
    uniq = sorted(set(values))
    return {v: i for i, v in enumerate(uniq)}


def _evaluate_ranking(logits: torch.Tensor, y: torch.Tensor) -> Dict[str, float]:
    if logits.shape[0] == 0:
        return {
            "top1": 0.0,
            "top5": 0.0,
            "top10": 0.0,
            "recall1": 0.0,
            "recall5": 0.0,
            "recall10": 0.0,
            "precision5": 0.0,
            "precision10": 0.0,
            "ndcg5": 0.0,
            "ndcg10": 0.0,
            "mrr": 0.0,
            "loss": 0.0,
        }

    c = logits.shape[1]
    ks = [1, min(5, c), min(10, c)]
    sorted_idx = torch.argsort(logits, dim=1, descending=True)
    y_col = y.unsqueeze(1)
    hit_positions = (sorted_idx == y_col).nonzero(as_tuple=False)

    ranks = torch.full((logits.shape[0],), fill_value=c + 1, dtype=torch.float32)
    ranks[hit_positions[:, 0]] = hit_positions[:, 1].float() + 1.0

    out = {
        "loss": torch.nn.functional.cross_entropy(logits, y).item(),
        "mrr": (1.0 / ranks).mean().item(),
    }

    for name, k in [("1", ks[0]), ("5", ks[1]), ("10", ks[2])]:
        hits = (ranks <= float(k)).float()
        out[f"top{name}"] = hits.mean().item()
        out[f"recall{name}"] = hits.mean().item()
        out[f"precision{name}"] = (hits / float(k)).mean().item()
        out[f"ndcg{name}"] = (
            (hits / torch.log2(ranks + 1.0)).mean().item()
            if k > 1
            else hits.mean().item()
        )
    return out


def train_linear_probe(
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    val_x: torch.Tensor,
    val_y: torch.Tensor,
    test_x: torch.Tensor,
    test_y: torch.Tensor,
    num_classes: int,
    epochs: int,
    lr: float,
    weight_decay: float,
    batch_size: int,
    device: str,
) -> Dict[str, Dict[str, float]]:
    head = torch.nn.Linear(train_x.shape[-1], num_classes).to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=weight_decay)

    train_x = train_x.to(device)
    train_y = train_y.to(device)
    val_x = val_x.to(device)
    val_y = val_y.to(device)
    test_x = test_x.to(device)
    test_y = test_y.to(device)

    best_state = None
    best_val = float("inf")
    for _ in range(max(1, epochs)):
        perm = torch.randperm(train_x.shape[0], device=device)
        x_ep = train_x[perm]
        y_ep = train_y[perm]
        for i in range(0, x_ep.shape[0], batch_size):
            xb = x_ep[i : i + batch_size]
            yb = y_ep[i : i + batch_size]
            logits = head(xb)
            loss = torch.nn.functional.cross_entropy(logits, yb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

        with torch.no_grad():
            vloss = torch.nn.functional.cross_entropy(head(val_x), val_y).item()
            if vloss < best_val:
                best_val = vloss
                best_state = {k: v.detach().cpu().clone() for k, v in head.state_dict().items()}

    if best_state is not None:
        head.load_state_dict(best_state)

    with torch.no_grad():
        train_m = _evaluate_ranking(head(train_x), train_y)
        val_m = _evaluate_ranking(head(val_x), val_y)
        test_m = _evaluate_ranking(head(test_x), test_y)
    return {"train": train_m, "val": val_m, "test": test_m}


def collect_next_poi_features(
    dataset: POIDataset,
    indices: List[int],
    poi_vocab: Dict[str, int],
    pack: rb.BackbonePack,
    device: str,
    max_len: int,
    batch_size: int,
    max_points: int,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, int]]:
    if not indices:
        return torch.zeros((0, pack.ckpt_args["embed_dim"])), torch.zeros((0,), dtype=torch.long), {
            "samples": 0,
            "unknown_poi": 0,
        }

    subset = [dataset.samples[i] for i in indices]
    loader = DataLoader(
        subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_poi,
    )

    xs, ys = [], []
    unknown_poi = 0
    total = 0

    for batch in loader:
        outputs, _, _, _, attention = rb.forward_backbone(
            batch, pack, device=device, max_len=max_len, mask=None
        )
        step_hidden = outputs["step_hidden"].detach().cpu()
        attn = attention.detach().cpu()
        for b, seq in enumerate(batch["poi_seq"]):
            vlen = int(attn[b].sum().item())
            if vlen <= 1:
                continue
            for t in range(1, vlen):
                total += 1
                nxt = seq[t]
                cls = poi_vocab.get(nxt)
                if cls is None:
                    unknown_poi += 1
                    continue
                xs.append(step_hidden[b, t - 1])
                ys.append(cls)
                if max_points > 0 and len(xs) >= max_points:
                    break
            if max_points > 0 and len(xs) >= max_points:
                break
        if max_points > 0 and len(xs) >= max_points:
            break

    if not xs:
        return torch.zeros((0, pack.ckpt_args["embed_dim"])), torch.zeros((0,), dtype=torch.long), {
            "samples": 0,
            "unknown_poi": unknown_poi,
        }
    x = torch.stack(xs, dim=0)
    y = torch.tensor(ys, dtype=torch.long)
    return x, y, {"samples": total, "unknown_poi": unknown_poi}


def collect_user_features(
    dataset: POIDataset,
    indices: List[int],
    user_vocab: Dict[str, int],
    pack: rb.BackbonePack,
    device: str,
    max_len: int,
    batch_size: int,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, int]]:
    if not indices:
        return torch.zeros((0, pack.ckpt_args["embed_dim"])), torch.zeros((0,), dtype=torch.long), {
            "records": 0,
            "unknown_user": 0,
        }

    subset = [dataset.samples[i] for i in indices]
    loader = DataLoader(
        subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_poi,
    )

    xs, ys = [], []
    unknown_user = 0
    total = 0

    for batch in loader:
        outputs, _, _, _, attention = rb.forward_backbone(
            batch, pack, device=device, max_len=max_len, mask=None
        )
        pooled = rb.masked_mean(outputs["step_hidden"], attention).detach().cpu()
        for i, user in enumerate(batch["user"]):
            total += 1
            cls = user_vocab.get(user)
            if cls is None:
                unknown_user += 1
                continue
            xs.append(pooled[i])
            ys.append(cls)

    if not xs:
        return torch.zeros((0, pack.ckpt_args["embed_dim"])), torch.zeros((0,), dtype=torch.long), {
            "records": total,
            "unknown_user": unknown_user,
        }
    x = torch.stack(xs, dim=0)
    y = torch.tensor(ys, dtype=torch.long)
    return x, y, {"records": total, "unknown_user": unknown_user}


def run_eval(
    dataset: POIDataset,
    pack: rb.BackbonePack,
    *,
    seed: int,
    max_len: int,
    batch_size: int,
    max_points: int,
    probe_epochs: int,
    probe_lr: float,
    probe_weight_decay: float,
    probe_batch_size: int,
    split_mode: str,
    train_cities: Optional[List[str]] = None,
    test_cities: Optional[List[str]] = None,
    skip_user_identification: bool = False,
    device: str,
) -> dict:
    if train_cities or test_cities:
        train_idx, val_idx, test_idx, split_stats = split_indices_by_city(
            dataset,
            seed=seed,
            train_cities=train_cities,
            test_cities=test_cities,
        )
    else:
        train_idx, val_idx, test_idx = rb.split_indices(dataset, mode=split_mode, seed=seed)
        split_stats = {
            "train_records": len(train_idx),
            "val_records": len(val_idx),
            "test_records": len(test_idx),
        }

    if not train_idx or not val_idx or not test_idx:
        return {"error": "insufficient split data", "split": split_stats}

    train_poi = []
    train_users = []
    for i in train_idx:
        seq = dataset.samples[i]["poi_seq"]
        if len(seq) > 1:
            train_poi.extend(seq[1:])
        train_users.append(dataset.samples[i]["user"])

    poi_vocab = build_vocab(train_poi)
    user_vocab = build_vocab(train_users)

    train_x, train_y, train_sup = collect_next_poi_features(
        dataset,
        train_idx,
        poi_vocab,
        pack,
        device,
        max_len,
        batch_size,
        max_points,
    )
    val_x, val_y, val_sup = collect_next_poi_features(
        dataset,
        val_idx,
        poi_vocab,
        pack,
        device,
        max_len,
        batch_size,
        max_points,
    )
    test_x, test_y, test_sup = collect_next_poi_features(
        dataset,
        test_idx,
        poi_vocab,
        pack,
        device,
        max_len,
        batch_size,
        max_points,
    )

    if min(train_x.shape[0], val_x.shape[0], test_x.shape[0]) == 0:
        return {
            "error": "insufficient next_poi samples after vocabulary alignment",
            "split": split_stats,
            "support": {
                "train": train_sup,
                "val": val_sup,
                "test": test_sup,
            },
        }

    next_poi = train_linear_probe(
        train_x,
        train_y,
        val_x,
        val_y,
        test_x,
        test_y,
        num_classes=len(poi_vocab),
        epochs=probe_epochs,
        lr=probe_lr,
        weight_decay=probe_weight_decay,
        batch_size=probe_batch_size,
        device=device,
    )

    out = {
        "split": split_stats,
        "next_poi": {
            "num_classes": len(poi_vocab),
            "support": {
                "train": train_sup,
                "val": val_sup,
                "test": test_sup,
            },
            **next_poi,
        },
    }

    if not skip_user_identification and len(user_vocab) >= 2:
        u_train_x, u_train_y, u_train_sup = collect_user_features(
            dataset,
            train_idx,
            user_vocab,
            pack,
            device,
            max_len,
            batch_size,
        )
        u_val_x, u_val_y, u_val_sup = collect_user_features(
            dataset,
            val_idx,
            user_vocab,
            pack,
            device,
            max_len,
            batch_size,
        )
        u_test_x, u_test_y, u_test_sup = collect_user_features(
            dataset,
            test_idx,
            user_vocab,
            pack,
            device,
            max_len,
            batch_size,
        )
        if min(u_train_x.shape[0], u_val_x.shape[0], u_test_x.shape[0]) > 0:
            uid = train_linear_probe(
                u_train_x,
                u_train_y,
                u_val_x,
                u_val_y,
                u_test_x,
                u_test_y,
                num_classes=len(user_vocab),
                epochs=probe_epochs,
                lr=probe_lr,
                weight_decay=probe_weight_decay,
                batch_size=probe_batch_size,
                device=device,
            )
            out["user_identification"] = {
                "num_classes": len(user_vocab),
                "support": {
                    "train": u_train_sup,
                    "val": u_val_sup,
                    "test": u_test_sup,
                },
                **uid,
            }

    return out


def main():
    args = parse_args()
    rb.set_seed(args.seed)

    records = rb.load_local_data(args.local_data)
    dataset = POIDataset(records, max_len=args.max_len, sample_limit=args.sample_limit)
    if len(dataset) < 20:
        raise RuntimeError(
            "Too few valid POI samples. Ensure records contain trajectory and POI sequences."
        )

    pack = rb.load_backbone(args.checkpoint, device=args.device, override_max_len=args.max_len)

    train_cities = parse_city_list(args.train_cities)
    test_cities = parse_city_list(args.test_cities)

    result = {
        "checkpoint": args.checkpoint,
        "dataset": args.local_data,
        "settings": {
            "max_len": args.max_len,
            "split_mode": args.split_mode,
            "probe_epochs": args.probe_epochs,
            "probe_lr": args.probe_lr,
            "max_points": args.max_points,
            "train_cities": train_cities,
            "test_cities": test_cities,
        },
        "dataset_stats": {
            "records": len(dataset),
            "cities": len(set(s["city"] for s in dataset.samples)),
        },
    }
    result["results"] = run_eval(
        dataset,
        pack,
        seed=args.seed,
        max_len=args.max_len,
        batch_size=args.batch_size,
        max_points=args.max_points,
        probe_epochs=args.probe_epochs,
        probe_lr=args.probe_lr,
        probe_weight_decay=args.probe_weight_decay,
        probe_batch_size=args.probe_batch_size,
        split_mode=args.split_mode,
        train_cities=train_cities,
        test_cities=test_cities,
        skip_user_identification=args.skip_user_identification,
        device=args.device,
    )

    print(json.dumps(result, indent=2))
    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, indent=2) + "\n")
        print(f"saved {out}")


if __name__ == "__main__":
    main()
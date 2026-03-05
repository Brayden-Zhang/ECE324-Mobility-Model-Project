from utils.hmt_model import TrajectoryFMHMT
from utils.hmt import HMTConfig, HMTTokenizer, TimeFeatures
from utils.context import OSMContextIndex, context_tensor_from_index
import argparse
import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run benchmark suite for TrajectoryFM checkpoints")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--local_data", type=str, default="")
    parser.add_argument("--max_len", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sample_limit", type=int, default=0)
    parser.add_argument("--recon_mask_ratio", type=float, default=0.3)
    parser.add_argument("--probe_epochs", type=int, default=6)
    parser.add_argument("--probe_lr", type=float, default=2e-3)
    parser.add_argument("--probe_weight_decay", type=float, default=1e-4)
    parser.add_argument("--probe_batch_size", type=int, default=2048)
    parser.add_argument("--max_probe_points", type=int, default=200000)
    parser.add_argument(
        "--split_mode",
        type=str,
        default="both",
        choices=[
            "both",
            "random",
            "temporal"])
    parser.add_argument("--disable_graph", action="store_true")
    parser.add_argument(
        "--output",
        type=str,
        default="cache/benchmark_results.json")
    return parser.parse_args()


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_ckpt_path(path: str) -> str:
    """Resolve checkpoint paths (possibly absolute) to this repo checkout."""
    if not path:
        return path
    p = Path(path)
    if p.exists():
        return str(p)

    # Relative path from repo root.
    if not p.is_absolute():
        candidate = ROOT / p
        if candidate.exists():
            return str(candidate)

    # Common case: checkpoint stored an absolute path from another machine.
    name = p.name
    for base in (ROOT, ROOT / "data"):
        candidate = base / name
        if candidate.exists():
            return str(candidate)
    return path


def load_local_data(path: str) -> List[dict]:
    import pandas as pd

    if path.endswith(".parquet"):
        df = pd.read_parquet(path)
    else:
        df = pd.read_pickle(path)
    return df.to_dict(orient="records")


def parse_times(times) -> np.ndarray:
    if times is None:
        return np.zeros((0,), dtype=np.float32)
    arr = np.asarray(times)
    if arr.size == 0:
        return np.zeros((0,), dtype=np.float32)
    if np.issubdtype(arr.dtype, np.number):
        return arr.astype(np.float32, copy=False)
    if np.issubdtype(arr.dtype, np.datetime64):
        ts = arr.astype("datetime64[ns]").astype(np.int64) / 1e9
        return ts.astype(np.float32)
    try:
        import pandas as pd

        dt = pd.to_datetime(arr, errors="coerce", utc=True, format="mixed")
        ts = dt.view("int64").to_numpy(dtype=np.float64) / 1e9
        invalid = ~np.isfinite(ts)
        if invalid.any():
            ts[invalid] = np.arange(ts.shape[0], dtype=np.float64)[invalid]
        return ts.astype(np.float32)
    except Exception:
        out = np.zeros((arr.shape[0],), dtype=np.float32)
        for i, _ in enumerate(arr):
            out[i] = float(i)
        return out


def _is_valid_latlon(lat: np.ndarray, lon: np.ndarray) -> bool:
    finite = np.isfinite(lat) & np.isfinite(lon)
    if finite.sum() == 0:
        return False
    lat_f = lat[finite]
    lon_f = lon[finite]
    return float(
        np.max(
            np.abs(lat_f))) <= 90.0 and float(
        np.max(
            np.abs(lon_f))) <= 180.0


def _haversine_np(lat1, lon1, lat2, lon2) -> np.ndarray:
    lat1_r = np.deg2rad(lat1)
    lon1_r = np.deg2rad(lon1)
    lat2_r = np.deg2rad(lat2)
    lon2_r = np.deg2rad(lon2)
    dlat = lat2_r - lat1_r
    dlon = lon2_r - lon1_r
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1_r) * \
        np.cos(lat2_r) * np.sin(dlon / 2.0) ** 2
    c = 2.0 * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))
    return 6371000.0 * c


def _orientation_score(lat: np.ndarray, lon: np.ndarray) -> float:
    finite = np.isfinite(lat) & np.isfinite(lon)
    if finite.sum() < 2:
        return float("inf")
    lat = lat[finite]
    lon = lon[finite]
    if lat.shape[0] > 256:
        idx = np.linspace(
            0,
            lat.shape[0] - 1,
            num=256,
            endpoint=True).round().astype(
            np.int64)
        lat = lat[idx]
        lon = lon[idx]
    step_d = _haversine_np(lat[:-1], lon[:-1], lat[1:], lon[1:])
    if step_d.size == 0:
        return float("inf")
    med = float(np.median(step_d))
    p90 = float(np.percentile(step_d, 90))
    mx = float(np.max(step_d))
    lat95 = float(np.percentile(np.abs(lat), 95))
    polar_penalty = max(0.0, lat95 - 75.0) * 5000.0
    return med + 0.25 * p90 + 0.02 * mx + polar_penalty


def normalize_lon_lat(points) -> Tuple[np.ndarray, np.ndarray]:
    arr = np.asarray(points, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[1] < 2:
        raise ValueError(
            "trajectory must be a list/array of [lat, lon] or [lon, lat]")

    lat_a, lon_a = arr[:, 0], arr[:, 1]
    lat_b, lon_b = arr[:, 1], arr[:, 0]

    valid_a = _is_valid_latlon(lat_a, lon_a)
    valid_b = _is_valid_latlon(lat_b, lon_b)
    if valid_a and not valid_b:
        return lat_a, lon_a
    if valid_b and not valid_a:
        return lat_b, lon_b
    if not valid_a and not valid_b:
        # Keep historical behavior on malformed records.
        return lat_a, lon_a

    # Ambiguous case: both orders are range-valid.
    score_a = _orientation_score(lat_a, lon_a)
    score_b = _orientation_score(lat_b, lon_b)
    # Only flip when swapped order is materially more plausible.
    if score_b + 1e-6 < score_a * 0.8:
        return lat_b, lon_b
    return lat_a, lon_a


def deterministic_downsample(length: int, max_len: int) -> np.ndarray:
    if length <= max_len:
        return np.arange(length, dtype=np.int64)
    # Uniformly sample indices along trajectory length.
    idx = np.linspace(0, length - 1, num=max_len, endpoint=True)
    idx = np.round(idx).astype(np.int64)
    idx = np.clip(idx, 0, length - 1)
    # Ensure non-decreasing and exactly max_len points.
    idx = np.maximum.accumulate(idx)
    return idx[:max_len]


def preprocess_record(record: dict, max_len: int) -> Optional[dict]:
    traj = record.get("trajectory")
    if traj is None:
        traj = record.get("traj")
    if traj is None:
        traj = record.get("points")
    if traj is None:
        return None
    times = record.get("time")
    if times is None:
        times = record.get("times")
    if times is None:
        times = record.get("timestamp")

    lat, lon = normalize_lon_lat(traj)
    valid = np.isfinite(lat) & np.isfinite(lon)
    if valid.sum() < 2:
        return None
    lat = lat[valid]
    lon = lon[valid]
    if times is None:
        ts = np.arange(lat.shape[0], dtype=np.float32)
    else:
        ts = parse_times(times)
        if ts.shape[0] != valid.shape[0]:
            n = min(ts.shape[0], lat.shape[0])
            lat = lat[:n]
            lon = lon[:n]
            ts = ts[:n]
        else:
            ts = ts[valid]
    if lat.shape[0] < 2:
        return None

    idx = deterministic_downsample(lat.shape[0], max_len)
    lat = lat[idx]
    lon = lon[idx]
    ts = ts[idx]

    coords = np.zeros((max_len, 2), dtype=np.float32)
    timestamps = np.zeros((max_len,), dtype=np.float32)
    attention = np.zeros((max_len,), dtype=np.float32)
    vlen = lat.shape[0]
    coords[:vlen, 0] = lat
    coords[:vlen, 1] = lon
    timestamps[:vlen] = ts
    attention[:vlen] = 1.0
    return {
        "coords": torch.from_numpy(coords),
        "timestamps": torch.from_numpy(timestamps),
        "attention_mask": torch.from_numpy(attention),
        "start_ts": float(ts[0]),
    }


class FixedTrajectoryDataset(Dataset):
    def __init__(
            self,
            records: List[dict],
            max_len: int,
            sample_limit: int = 0):
        processed = []
        for rec in records:
            p = preprocess_record(rec, max_len=max_len)
            if p is not None:
                processed.append(p)
                if sample_limit > 0 and len(processed) >= sample_limit:
                    break
        self.samples = processed

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        return self.samples[idx]


def collate_fixed(batch: List[dict]) -> dict:
    return {
        "coords": torch.stack([b["coords"] for b in batch], dim=0),
        "timestamps": torch.stack([b["timestamps"] for b in batch], dim=0),
        "attention_mask": torch.stack(
            [b["attention_mask"] for b in batch], dim=0
        ),
        "start_ts": torch.tensor(
            [b["start_ts"] for b in batch], dtype=torch.float32
        ),
    }


def split_indices(dataset: FixedTrajectoryDataset, mode: str,
                  seed: int) -> Tuple[List[int], List[int], List[int]]:
    n = len(dataset)
    idx = list(range(n))
    if n < 10:
        return idx, idx, idx
    if mode == "temporal":
        idx = sorted(idx, key=lambda i: dataset.samples[i]["start_ts"])
    else:
        rng = random.Random(seed)
        rng.shuffle(idx)
    n_train = max(1, int(n * 0.70))
    n_val = max(1, int(n * 0.15))
    n_test = n - n_train - n_val
    if n_test <= 0:
        n_test = 1
        if n_train > 1:
            n_train -= 1
        else:
            n_val = max(1, n_val - 1)
    train_idx = idx[:n_train]
    val_idx = idx[n_train: n_train + n_val]
    test_idx = idx[n_train + n_val:]
    if not val_idx:
        val_idx = train_idx[-1:]
    if not test_idx:
        test_idx = val_idx
    return train_idx, val_idx, test_idx


@dataclass
class BackbonePack:
    model: TrajectoryFMHMT
    tokenizer: HMTTokenizer
    time_encoder: TimeFeatures
    ckpt_args: dict
    context_index: Optional[OSMContextIndex]


def safe_load_state_dict(
        module: torch.nn.Module,
        incoming: dict,
        module_name: str):
    current = module.state_dict()
    loadable = {}
    skipped = 0
    for k, v in incoming.items():
        if k not in current:
            skipped += 1
            continue
        if current[k].shape != v.shape:
            skipped += 1
            continue
        loadable[k] = v
    missing, unexpected = module.load_state_dict(loadable, strict=False)
    if skipped:
        print(
            f"warning: {module_name} skipped {skipped} "
            "incompatible/missing checkpoint tensors"
        )
    if missing:
        print(
            f"warning: {module_name} missing {len(missing)} "
            "keys after partial checkpoint load"
        )
    if unexpected:
        print(
            f"warning: {module_name} has {len(unexpected)} "
            "unexpected keys after partial checkpoint load"
        )


def load_backbone(
        checkpoint_path: str,
        device: str,
        override_max_len: int,
        disable_graph: bool = False) -> BackbonePack:
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    ckpt_args = dict(ckpt["args"])

    # Make checkpoints portable across machines by resolving embedded file
    # paths.
    for key in ("h3_vocab", "osm_context"):
        before = ckpt_args.get(key, "")
        after = resolve_ckpt_path(before) if isinstance(
            before, str) else before
        if before and after != before:
            print(f"info: resolved {key}: {before} -> {after}")
            ckpt_args[key] = after

    tokenizer_cfg = HMTConfig(
        mode=ckpt_args["tokenizer"],
        res0=ckpt_args["res0"],
        res1=ckpt_args["res1"],
        res2=ckpt_args["res2"],
        vocab_l0=ckpt_args["vocab_l0"],
        vocab_l1=ckpt_args["vocab_l1"],
        vocab_l2=ckpt_args["vocab_l2"],
        hash_tokens=ckpt_args.get("hash_tokens", False),
        h3_vocab=ckpt_args.get("h3_vocab", ""),
    )
    base_feature_dim = 17
    context_dim = ckpt_args.get(
        "osm_context_dim",
        0) if ckpt_args.get("osm_context") else 0
    feature_dim = base_feature_dim + \
        context_dim if context_dim > 0 else base_feature_dim

    tokenizer = HMTTokenizer(
        tokenizer_cfg,
        feature_dim=feature_dim,
        embed_dim=ckpt_args["embed_dim"]).to(device)
    time_encoder = TimeFeatures(ckpt_args["embed_dim"]).to(device)
    model = TrajectoryFMHMT(
        vocab_l0=ckpt_args["vocab_l0"],
        vocab_l1=ckpt_args["vocab_l1"],
        vocab_l2=ckpt_args["vocab_l2"],
        embed_dim=ckpt_args["embed_dim"],
        depth=ckpt_args["depth"],
        heads=ckpt_args["heads"],
        dropout=ckpt_args["dropout"],
        context_dim=context_dim,
        trip_feat_dim=4 if ckpt_args.get("use_trip_features", True) else 0,
        max_seq_len=override_max_len * 3 + 16,
        use_graph=(
            False if disable_graph else ckpt_args.get("use_graph", False)
        ),
        graph_layers=ckpt_args.get("graph_layers", 0),
        graph_knn=ckpt_args.get("graph_knn", 8),
        graph_temporal_window=ckpt_args.get("graph_temporal_window", 2),
        graph_same_region=ckpt_args.get("graph_same_region", True),
        step_attention_window=ckpt_args.get("step_attention_window", 0),
        use_spacetime=ckpt_args.get("space_time_encoder", False),
        spacetime_freqs=ckpt_args.get("space_time_freqs", 6),
        macro_region_vocab=ckpt_args.get("macro_region_vocab", 0),
        macro_dist_dim=ckpt_args.get("macro_dist_dim", 0),
    ).to(device)

    safe_load_state_dict(model, ckpt["model"], module_name="model")
    safe_load_state_dict(
        tokenizer,
        ckpt.get(
            "tokenizer",
            {}),
        module_name="tokenizer")
    if "time_encoder" in ckpt:
        safe_load_state_dict(
            time_encoder,
            ckpt["time_encoder"],
            module_name="time_encoder")
    else:
        print(
            "warning: checkpoint missing time_encoder state; "
            "using fresh initialization"
        )

    for m in (model, tokenizer, time_encoder):
        m.eval()
        for p in m.parameters():
            p.requires_grad = False

    context_index = None
    if ckpt_args.get("osm_context"):
        context_path = ckpt_args["osm_context"]
        if context_path and Path(context_path).exists():
            context_index = OSMContextIndex(
                context_path, ckpt_args["osm_context_dim"])
        else:
            print(
                "warning: checkpoint requested OSM context but file missing: "
                f"{context_path}"
            )

    return BackbonePack(
        model=model,
        tokenizer=tokenizer,
        time_encoder=time_encoder,
        ckpt_args=ckpt_args,
        context_index=context_index)


def compute_trip_features(coords: torch.Tensor,
                          attention_mask: torch.Tensor) -> torch.Tensor:
    deltas = coords[:, 1:] - coords[:, :-1]
    seg_dist = torch.sqrt((deltas**2).sum(dim=-1) + 1e-12)
    step_dist = torch.zeros_like(attention_mask)
    step_dist[:, 1:] = seg_dist
    step_dist = step_dist * attention_mask

    cum_dist = torch.cumsum(step_dist, dim=1)
    total_dist = step_dist.sum(dim=1, keepdim=True).clamp(min=1e-6)
    trip_len = attention_mask.sum(dim=1, keepdim=True).clamp(min=1.0)
    denom_steps = (trip_len - 1.0).clamp(min=1.0)

    step_idx = torch.cumsum(attention_mask, dim=1) - 1.0
    progress_step = (step_idx / denom_steps) * attention_mask
    progress_dist = (cum_dist / total_dist) * attention_mask
    trip_len_log = torch.log1p(trip_len).expand_as(
        attention_mask) * attention_mask
    total_dist_log = torch.log1p(total_dist).expand_as(
        attention_mask) * attention_mask
    return torch.stack([progress_step, progress_dist,
                       trip_len_log, total_dist_log], dim=-1)


def sample_mask(attention_mask: torch.Tensor, ratio: float,
                generator: torch.Generator) -> torch.Tensor:
    bsz, seq_len = attention_mask.shape
    out = torch.zeros((bsz, seq_len), dtype=torch.bool,
                      device=attention_mask.device)
    for b in range(bsz):
        vlen = int(attention_mask[b].sum().item())
        if vlen <= 0:
            continue
        num = max(1, int(vlen * ratio))
        perm = torch.randperm(
            vlen,
            generator=generator,
            device=attention_mask.device)
        idx = perm[:num]
        out[b, idx] = True
    return out


def forward_backbone(batch: dict,
                     pack: BackbonePack,
                     device: str,
                     max_len: int,
                     mask: Optional[torch.Tensor] = None):
    coords = batch["coords"].to(device)
    timestamps = batch["timestamps"].to(device)
    attention = batch["attention_mask"].to(device)
    context = None
    if pack.context_index is not None:
        context = context_tensor_from_index(
            pack.context_index, coords, pack.ckpt_args["res1"])

    tokens_l0, tokens_l1, tokens_l2, _ = pack.tokenizer(
        coords,
        timestamps,
        context,
        attention_mask=attention,
    )
    tokens_l0 = tokens_l0.to(device)
    tokens_l1 = tokens_l1.to(device)
    tokens_l2 = tokens_l2.to(device)

    if mask is not None:
        masked_l0 = tokens_l0.clone()
        masked_l1 = tokens_l1.clone()
        masked_l2 = tokens_l2.clone()
        masked_l0[mask] = pack.ckpt_args["vocab_l0"]
        masked_l1[mask] = pack.ckpt_args["vocab_l1"]
        masked_l2[mask] = pack.ckpt_args["vocab_l2"]
    else:
        masked_l0, masked_l1, masked_l2 = tokens_l0, tokens_l1, tokens_l2

    trip_features = None
    if pack.ckpt_args.get("use_trip_features", True):
        trip_features = compute_trip_features(coords, attention)

    with torch.no_grad():
        try:
            time_embed = pack.time_encoder(timestamps, attention)
        except TypeError:
            time_embed = pack.time_encoder(timestamps)
        outputs = pack.model(
            masked_l0,
            masked_l1,
            masked_l2,
            time_embed,
            attention,
            timestamps=timestamps,
            context=context,
            trip_features=trip_features,
            coords=coords,
            region_mask_ratio=0.0,
            # Avoid leaking masked targets through region grouping during
            # reconstruction.
            region_source_l1=masked_l1 if mask is not None else tokens_l1,
            region_source_l2=masked_l2 if mask is not None else tokens_l2,
        )
    return outputs, tokens_l0, tokens_l1, tokens_l2, attention


def masked_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    m = mask.unsqueeze(-1).float()
    return (x * m).sum(dim=1) / m.sum(dim=1).clamp(min=1.0)


def run_reconstruction(loader: DataLoader,
                       pack: BackbonePack,
                       device: str,
                       max_len: int,
                       mask_ratio: float,
                       seed: int) -> Dict[str,
                                          float]:
    rng = torch.Generator(device=device)
    rng.manual_seed(seed)
    total = {"acc_l0": 0.0, "acc_l1": 0.0, "acc_l2": 0.0, "n": 0}

    for batch in loader:
        attention = batch["attention_mask"].to(device)
        mask = sample_mask(attention, mask_ratio, generator=rng)
        outputs, t0, t1, t2, _ = forward_backbone(
            batch, pack, device=device, max_len=max_len, mask=mask)
        n = int(mask.sum().item())
        if n == 0:
            continue
        p0 = outputs["step_logits"]["l0"].argmax(dim=-1)
        p1 = outputs["step_logits"]["l1"].argmax(dim=-1)
        p2 = outputs["step_logits"]["l2"].argmax(dim=-1)
        total["acc_l0"] += (p0[mask] == t0[mask]).float().sum().item()
        total["acc_l1"] += (p1[mask] == t1[mask]).float().sum().item()
        total["acc_l2"] += (p2[mask] == t2[mask]).float().sum().item()
        total["n"] += n

    if total["n"] == 0:
        return {"recon_acc_l0": 0.0, "recon_acc_l1": 0.0, "recon_acc_l2": 0.0}
    n = total["n"]
    return {
        "recon_acc_l0": total["acc_l0"] / n,
        "recon_acc_l1": total["acc_l1"] / n,
        "recon_acc_l2": total["acc_l2"] / n,
    }


def collect_next_location_features(
    loader: DataLoader,
    pack: BackbonePack,
    device: str,
    max_len: int,
    max_points: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    feats, labels = [], []
    seen = 0
    for batch in loader:
        outputs, _, t1, _, attention = forward_backbone(
            batch, pack, device=device, max_len=max_len, mask=None)
        step_hidden = outputs["step_hidden"].detach().cpu()
        t1_cpu = t1.detach().cpu()
        attn_cpu = attention.detach().cpu()
        bsz = step_hidden.shape[0]
        for b in range(bsz):
            vlen = int(attn_cpu[b].sum().item())
            if vlen <= 1:
                continue
            x = step_hidden[b, : vlen - 1]
            y = t1_cpu[b, 1:vlen]
            feats.append(x)
            labels.append(y)
            seen += x.shape[0]
            if max_points > 0 and seen >= max_points:
                break
        if max_points > 0 and seen >= max_points:
            break
    if not feats:
        return torch.zeros(
            (0, pack.ckpt_args["embed_dim"])), torch.zeros(
            (0,), dtype=torch.long)
    x = torch.cat(feats, dim=0)
    y = torch.cat(labels, dim=0).long()
    if max_points > 0 and x.shape[0] > max_points:
        x = x[:max_points]
        y = y[:max_points]
    return x, y


def collect_destination_features(
    loader: DataLoader,
    pack: BackbonePack,
    device: str,
    max_len: int,
    max_points: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    feats, labels = [], []
    seen = 0
    for batch in loader:
        outputs, _, t1, _, attention = forward_backbone(
            batch, pack, device=device, max_len=max_len, mask=None)
        pooled_step = masked_mean(outputs["step_hidden"], attention)
        pooled_mid = masked_mean(
            outputs["mid_hidden"],
            outputs["mid_mask"],
        ) if outputs["mid_hidden"].shape[1] > 0 else torch.zeros_like(
            pooled_step
        )
        x = torch.cat([pooled_step, pooled_mid], dim=-1).detach().cpu()
        last_idx = attention.sum(dim=1).long().clamp(min=1) - 1
        y = t1.gather(1, last_idx.unsqueeze(1)).squeeze(
            1).detach().cpu().long()
        feats.append(x)
        labels.append(y)
        seen += x.shape[0]
        if max_points > 0 and seen >= max_points:
            break
    if not feats:
        return torch.zeros(
            (0, pack.ckpt_args["embed_dim"] * 2)
        ), torch.zeros((0,), dtype=torch.long)
    x = torch.cat(feats, dim=0)
    y = torch.cat(labels, dim=0)
    if max_points > 0 and x.shape[0] > max_points:
        x = x[:max_points]
        y = y[:max_points]
    return x, y


def evaluate_logits(logits: torch.Tensor,
                    targets: torch.Tensor) -> Dict[str, float]:
    if logits.shape[0] == 0:
        return {"top1": 0.0, "top5": 0.0, "loss": 0.0}
    loss = torch.nn.functional.cross_entropy(logits, targets).item()
    pred = logits.argmax(dim=-1)
    top1 = (pred == targets).float().mean().item()
    k = min(5, logits.shape[-1])
    topk = logits.topk(k, dim=-1).indices
    top5 = (topk == targets.unsqueeze(-1)).any(dim=-1).float().mean().item()
    return {"top1": top1, "top5": top5, "loss": loss}


def haversine_m(pred_latlon: torch.Tensor,
                true_latlon: torch.Tensor) -> torch.Tensor:
    # pred/true: [N, 2] with [lat, lon] in degrees
    if pred_latlon.shape[0] == 0:
        return torch.zeros((0,), dtype=torch.float32,
                           device=pred_latlon.device)
    lat1 = torch.deg2rad(pred_latlon[:, 0])
    lon1 = torch.deg2rad(pred_latlon[:, 1])
    lat2 = torch.deg2rad(true_latlon[:, 0])
    lon2 = torch.deg2rad(true_latlon[:, 1])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = torch.sin(dlat / 2).pow(2) + torch.cos(lat1) * \
        torch.cos(lat2) * torch.sin(dlon / 2).pow(2)
    c = 2.0 * torch.atan2(torch.sqrt(a.clamp(min=0.0, max=1.0)),
                          torch.sqrt((1.0 - a).clamp(min=0.0)))
    return 6371000.0 * c


def evaluate_regression_meters(
    pred_latlon: torch.Tensor,
    true_latlon: torch.Tensor,
) -> Dict[str, float]:
    if pred_latlon.shape[0] == 0:
        return {"mae_m": 0.0, "rmse_m": 0.0, "mse_deg2": 0.0}
    d_m = haversine_m(pred_latlon, true_latlon)
    mae_m = d_m.mean().item()
    rmse_m = torch.sqrt((d_m.pow(2)).mean()).item()
    mse_deg2 = ((pred_latlon - true_latlon).pow(2).sum(dim=-1)).mean().item()
    return {"mae_m": mae_m, "rmse_m": rmse_m, "mse_deg2": mse_deg2}


def collect_next_location_regression(
    loader: DataLoader,
    pack: BackbonePack,
    device: str,
    max_len: int,
    max_points: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    feats, targets = [], []
    seen = 0
    for batch in loader:
        outputs, _, _, _, attention = forward_backbone(
            batch, pack, device=device, max_len=max_len, mask=None)
        step_hidden = outputs["step_hidden"].detach().cpu()
        coords = batch["coords"].detach().cpu()
        attn_cpu = attention.detach().cpu()
        bsz = step_hidden.shape[0]
        for b in range(bsz):
            vlen = int(attn_cpu[b].sum().item())
            if vlen <= 1:
                continue
            x = step_hidden[b, : vlen - 1]
            y = coords[b, 1:vlen, :2]
            feats.append(x)
            targets.append(y)
            seen += x.shape[0]
            if max_points > 0 and seen >= max_points:
                break
        if max_points > 0 and seen >= max_points:
            break
    if not feats:
        return torch.zeros(
            (0, pack.ckpt_args["embed_dim"])), torch.zeros(
            (0, 2))
    x = torch.cat(feats, dim=0)
    y = torch.cat(targets, dim=0).float()
    if max_points > 0 and x.shape[0] > max_points:
        x = x[:max_points]
        y = y[:max_points]
    return x, y


def collect_destination_regression(
    loader: DataLoader,
    pack: BackbonePack,
    device: str,
    max_len: int,
    max_points: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    feats, targets = [], []
    seen = 0
    for batch in loader:
        outputs, _, _, _, attention = forward_backbone(
            batch, pack, device=device, max_len=max_len, mask=None)
        pooled_step = masked_mean(outputs["step_hidden"], attention)
        pooled_mid = (
            masked_mean(outputs["mid_hidden"], outputs["mid_mask"])
            if outputs["mid_hidden"].shape[1] > 0
            else torch.zeros_like(pooled_step)
        )
        x = torch.cat([pooled_step, pooled_mid], dim=-1).detach().cpu()
        coords = batch["coords"].detach().cpu()
        attn_cpu = attention.detach().cpu()
        y_list = []
        for b in range(coords.shape[0]):
            last = int(attn_cpu[b].sum().item()) - 1
            last = max(last, 0)
            y_list.append(coords[b, last, :2].float())
        y = torch.stack(y_list, dim=0)
        feats.append(x)
        targets.append(y)
        seen += x.shape[0]
        if max_points > 0 and seen >= max_points:
            break
    if not feats:
        return torch.zeros(
            (0, pack.ckpt_args["embed_dim"] * 2)), torch.zeros((0, 2))
    x = torch.cat(feats, dim=0)
    y = torch.cat(targets, dim=0)
    if max_points > 0 and x.shape[0] > max_points:
        x = x[:max_points]
        y = y[:max_points]
    return x, y


def train_regression_probe(
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    val_x: torch.Tensor,
    val_y: torch.Tensor,
    test_x: torch.Tensor,
    test_y: torch.Tensor,
    epochs: int,
    lr: float,
    weight_decay: float,
    batch_size: int,
    device: str,
) -> Dict[str, Dict[str, float]]:
    in_dim = train_x.shape[-1]
    head = torch.nn.Linear(in_dim, 2).to(device)
    opt = torch.optim.AdamW(
        head.parameters(),
        lr=lr,
        weight_decay=weight_decay)

    train_x = train_x.to(device)
    val_x = val_x.to(device)
    test_x = test_x.to(device)
    train_y = train_y.to(device).float()
    val_y = val_y.to(device).float()
    test_y = test_y.to(device).float()

    # Normalize targets for stable optimization, then decode to lat/lon for
    # reporting.
    y_mean = train_y.mean(dim=0, keepdim=True)
    y_std = train_y.std(dim=0, keepdim=True).clamp(min=1e-6)

    train_y_n = (train_y - y_mean) / y_std
    val_y_n = (val_y - y_mean) / y_std

    best_state = None
    best_val = float("inf")
    for _ in range(max(1, epochs)):
        if train_x.shape[0] > 0:
            perm = torch.randperm(train_x.shape[0], device=device)
            train_x = train_x[perm]
            train_y_n = train_y_n[perm]
            for start in range(0, train_x.shape[0], batch_size):
                end = start + batch_size
                xb = train_x[start:end]
                yb = train_y_n[start:end]
                pred_n = head(xb)
                loss = torch.nn.functional.mse_loss(pred_n, yb)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()

        with torch.no_grad():
            val_pred_n = head(val_x)
            val_loss = torch.nn.functional.mse_loss(val_pred_n, val_y_n).item()
            if val_loss < best_val:
                best_val = val_loss
                best_state = {k: v.detach().cpu().clone()
                              for k, v in head.state_dict().items()}

    if best_state is not None:
        head.load_state_dict(best_state)

    with torch.no_grad():
        train_pred = head(train_x) * y_std + y_mean
        val_pred = head(val_x) * y_std + y_mean
        test_pred = head(test_x) * y_std + y_mean
        train_metrics = evaluate_regression_meters(train_pred, train_y)
        val_metrics = evaluate_regression_meters(val_pred, val_y)
        test_metrics = evaluate_regression_meters(test_pred, test_y)
    return {"train": train_metrics, "val": val_metrics, "test": test_metrics}


def train_probe(
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
    in_dim = train_x.shape[-1]
    head = torch.nn.Linear(in_dim, num_classes).to(device)
    opt = torch.optim.AdamW(
        head.parameters(),
        lr=lr,
        weight_decay=weight_decay)

    train_x = train_x.to(device)
    train_y = train_y.to(device)
    val_x = val_x.to(device)
    val_y = val_y.to(device)
    test_x = test_x.to(device)
    test_y = test_y.to(device)

    best_state = None
    best_val = -1.0
    for _ in range(max(1, epochs)):
        if train_x.shape[0] > 0:
            perm = torch.randperm(train_x.shape[0], device=device)
            train_x = train_x[perm]
            train_y = train_y[perm]
            for start in range(0, train_x.shape[0], batch_size):
                end = start + batch_size
                xb = train_x[start:end]
                yb = train_y[start:end]
                logits = head(xb)
                loss = torch.nn.functional.cross_entropy(logits, yb)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()

        with torch.no_grad():
            val_logits = head(val_x)
            val_metrics = evaluate_logits(val_logits, val_y)
            if val_metrics["top1"] > best_val:
                best_val = val_metrics["top1"]
                best_state = {k: v.detach().cpu().clone()
                              for k, v in head.state_dict().items()}

    if best_state is not None:
        head.load_state_dict(best_state)

    with torch.no_grad():
        train_metrics = evaluate_logits(head(train_x), train_y)
        val_metrics = evaluate_logits(head(val_x), val_y)
        test_metrics = evaluate_logits(head(test_x), test_y)
    return {"train": train_metrics, "val": val_metrics, "test": test_metrics}


def subset_loader(
        dataset: FixedTrajectoryDataset,
        indices: List[int],
        batch_size: int,
        num_workers: int) -> DataLoader:
    class _Subset(Dataset):
        def __init__(self, parent: FixedTrajectoryDataset, idxs: List[int]):
            self.parent = parent
            self.idxs = idxs

        def __len__(self):
            return len(self.idxs)

        def __getitem__(self, i):
            return self.parent[self.idxs[i]]

    ds = _Subset(dataset, indices)
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fixed)


def run_split_suite(
    name: str,
    dataset: FixedTrajectoryDataset,
    indices: Tuple[List[int], List[int], List[int]],
    pack: BackbonePack,
    args,
) -> Dict[str, dict]:
    train_idx, val_idx, test_idx = indices
    print(
        f"[{name}] sizes train={len(train_idx)} "
        f"val={len(val_idx)} test={len(test_idx)}"
    )
    train_loader = subset_loader(
        dataset,
        train_idx,
        args.batch_size,
        args.num_workers)
    val_loader = subset_loader(
        dataset,
        val_idx,
        args.batch_size,
        args.num_workers)
    test_loader = subset_loader(
        dataset,
        test_idx,
        args.batch_size,
        args.num_workers)

    recon = run_reconstruction(
        test_loader,
        pack,
        device=args.device,
        max_len=args.max_len,
        mask_ratio=args.recon_mask_ratio,
        seed=args.seed + 123,
    )

    nx_train_x, nx_train_y = collect_next_location_features(
        train_loader, pack, args.device, args.max_len, args.max_probe_points)
    nx_val_x, nx_val_y = collect_next_location_features(
        val_loader, pack, args.device, args.max_len, max(
            1, args.max_probe_points // 3))
    nx_test_x, nx_test_y = collect_next_location_features(
        test_loader, pack, args.device, args.max_len, max(
            1, args.max_probe_points // 3))
    next_loc = train_probe(
        nx_train_x,
        nx_train_y,
        nx_val_x,
        nx_val_y,
        nx_test_x,
        nx_test_y,
        num_classes=pack.ckpt_args["vocab_l1"],
        epochs=args.probe_epochs,
        lr=args.probe_lr,
        weight_decay=args.probe_weight_decay,
        batch_size=args.probe_batch_size,
        device=args.device,
    )

    d_train_x, d_train_y = collect_destination_features(
        train_loader, pack, args.device, args.max_len, args.max_probe_points)
    d_val_x, d_val_y = collect_destination_features(
        val_loader, pack, args.device, args.max_len, max(
            1, args.max_probe_points // 3))
    d_test_x, d_test_y = collect_destination_features(
        test_loader, pack, args.device, args.max_len, max(
            1, args.max_probe_points // 3))
    destination = train_probe(
        d_train_x,
        d_train_y,
        d_val_x,
        d_val_y,
        d_test_x,
        d_test_y,
        num_classes=pack.ckpt_args["vocab_l1"],
        epochs=args.probe_epochs,
        lr=args.probe_lr,
        weight_decay=args.probe_weight_decay,
        batch_size=args.probe_batch_size,
        device=args.device,
    )

    nxr_train_x, nxr_train_y = collect_next_location_regression(
        train_loader, pack, args.device, args.max_len, args.max_probe_points
    )
    nxr_val_x, nxr_val_y = collect_next_location_regression(
        val_loader, pack, args.device, args.max_len, max(
            1, args.max_probe_points // 3))
    nxr_test_x, nxr_test_y = collect_next_location_regression(
        test_loader, pack, args.device, args.max_len, max(
            1, args.max_probe_points // 3))
    next_loc_reg = train_regression_probe(
        nxr_train_x,
        nxr_train_y,
        nxr_val_x,
        nxr_val_y,
        nxr_test_x,
        nxr_test_y,
        epochs=args.probe_epochs,
        lr=args.probe_lr,
        weight_decay=args.probe_weight_decay,
        batch_size=args.probe_batch_size,
        device=args.device,
    )

    dr_train_x, dr_train_y = collect_destination_regression(
        train_loader, pack, args.device, args.max_len, args.max_probe_points
    )
    dr_val_x, dr_val_y = collect_destination_regression(
        val_loader, pack, args.device, args.max_len, max(
            1, args.max_probe_points // 3))
    dr_test_x, dr_test_y = collect_destination_regression(
        test_loader, pack, args.device, args.max_len, max(
            1, args.max_probe_points // 3))
    destination_reg = train_regression_probe(
        dr_train_x,
        dr_train_y,
        dr_val_x,
        dr_val_y,
        dr_test_x,
        dr_test_y,
        epochs=args.probe_epochs,
        lr=args.probe_lr,
        weight_decay=args.probe_weight_decay,
        batch_size=args.probe_batch_size,
        device=args.device,
    )

    return {
        "reconstruction": recon,
        "next_location_probe": next_loc,
        "destination_probe": destination,
        "next_location_regression_probe": next_loc_reg,
        "destination_regression_probe": destination_reg,
    }


def main():
    args = parse_args()
    set_seed(args.seed)

    pack = load_backbone(
        args.checkpoint,
        device=args.device,
        override_max_len=args.max_len,
        disable_graph=args.disable_graph,
    )
    ckpt_args = pack.ckpt_args
    local_data = args.local_data or ckpt_args.get("local_data", "")
    if not local_data:
        raise ValueError(
            "No local_data provided and checkpoint args do not include "
            "local_data"
        )
    if not Path(local_data).exists():
        raise FileNotFoundError(f"local_data not found: {local_data}")

    raw_records = load_local_data(local_data)
    dataset = FixedTrajectoryDataset(
        raw_records,
        max_len=args.max_len,
        sample_limit=args.sample_limit)
    if len(dataset) < 10:
        raise RuntimeError(f"not enough benchmark samples: {len(dataset)}")
    print(
        f"loaded benchmark dataset: {len(dataset)} samples from {local_data}")

    split_modes = [
        "random",
        "temporal"] if args.split_mode == "both" else [
        args.split_mode]
    results = {
        "checkpoint": args.checkpoint,
        "dataset": local_data,
        "samples": len(dataset),
        "settings": {
            "max_len": args.max_len,
            "recon_mask_ratio": args.recon_mask_ratio,
            "probe_epochs": args.probe_epochs,
            "probe_lr": args.probe_lr,
            "probe_weight_decay": args.probe_weight_decay,
            "max_probe_points": args.max_probe_points,
        },
        "splits": {},
    }
    for mode in split_modes:
        idx = split_indices(dataset, mode=mode, seed=args.seed)
        results["splits"][mode] = run_split_suite(
            mode, dataset, idx, pack, args)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"saved benchmark results: {out}")


if __name__ == "__main__":
    main()

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

try:
    import h3
except Exception:  # optional
    h3 = None

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import run_benchmarks as rb  # noqa: E402


@dataclass
class EvalResult:
    mae_m: float
    rmse_m: float
    n: int
    n_total: int = 0


def parse_args():
    parser = argparse.ArgumentParser(description="UniTraj-style recovery/prediction eval for TrajectoryFM-HMT")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--local_data", type=str, default="")
    parser.add_argument("--max_len", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sample_limit", type=int, default=0)
    parser.add_argument("--split_mode", type=str, default="both", choices=["both", "random", "temporal", "all"])
    parser.add_argument("--task", type=str, default="both", choices=["both", "recovery", "prediction"])
    parser.add_argument("--mask_ratio", type=float, default=0.5)
    parser.add_argument("--pred_steps", type=int, default=5)
    parser.add_argument("--centroid_level", type=str, default="l0", choices=["l0", "l1", "l2"])
    parser.add_argument("--centroid_samples", type=int, default=0)
    parser.add_argument("--centroid_fraction", type=float, default=0.0)
    parser.add_argument("--coord_noise_std_m", type=float, default=0.0)
    parser.add_argument("--input_drop_ratio", type=float, default=0.0)
    parser.add_argument(
        "--coord_mode",
        type=str,
        default="auto",
        choices=["auto", "degrees", "meters"],
        help="Distance mode for coords. Auto selects meters if |coord| > 360, else degrees.",
    )
    parser.add_argument("--disable_graph", action="store_true")
    parser.add_argument(
        "--exclude_unknown",
        action="store_true",
        default=None,
        help="Exclude points mapped to the unknown H3 id (vocab-1) when computing distance metrics. "
        "Defaults to True when an H3 vocab is configured (since the unknown id aggregates many cells).",
    )
    parser.add_argument(
        "--include_unknown",
        action="store_true",
        help="Include points mapped to the unknown H3 id in distance metrics (usually makes MAE/RMSE meaningless).",
    )
    parser.add_argument(
        "--use_regression",
        action="store_true",
        help="Use a trained regression head on step embeddings to predict coordinates instead of "
        "centroid lookup. This avoids the vocab-coverage problem and produces directly-comparable "
        "MAE/RMSE in meters against UniTraj.",
    )
    parser.add_argument("--regression_epochs", type=int, default=8)
    parser.add_argument("--regression_lr", type=float, default=2e-3)
    parser.add_argument("--regression_batch_size", type=int, default=2048)
    parser.add_argument("--output", type=str, default="cache/unitraj_eval.json")
    return parser.parse_args()


def _haversine_m(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    # pred/target: [N, 2] lat, lon in degrees
    if pred.numel() == 0:
        return pred.new_zeros((0,))
    deg2rad = math.pi / 180.0
    lat1 = pred[:, 0] * deg2rad
    lon1 = pred[:, 1] * deg2rad
    lat2 = target[:, 0] * deg2rad
    lon2 = target[:, 1] * deg2rad
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = torch.sin(dlat / 2) ** 2 + torch.cos(lat1) * torch.cos(lat2) * torch.sin(dlon / 2) ** 2
    c = 2 * torch.asin(torch.clamp(a.sqrt(), max=1.0))
    return 6371000.0 * c


def _euclidean_m(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    if pred.numel() == 0:
        return pred.new_zeros((0,))
    diff = pred - target
    return torch.sqrt((diff ** 2).sum(dim=-1))


def _distance_m(pred: torch.Tensor, target: torch.Tensor, coord_mode: str) -> torch.Tensor:
    if coord_mode == "meters":
        return _euclidean_m(pred, target)
    return _haversine_m(pred, target)


def _apply_coord_noise(coords: torch.Tensor, attention: torch.Tensor, std_m: float, generator: torch.Generator) -> torch.Tensor:
    if std_m <= 0:
        return coords
    lat = coords[..., 0]
    lon = coords[..., 1]
    lat_rad = lat * (math.pi / 180.0)
    m_per_deg_lat = 111320.0
    m_per_deg_lon = (111320.0 * torch.cos(lat_rad)).clamp(min=1e-3)
    try:
        noise = torch.randn_like(coords, generator=generator) * std_m
    except TypeError:
        noise = torch.randn(coords.shape, device=coords.device, dtype=coords.dtype) * std_m
    noise_lat = noise[..., 0] / m_per_deg_lat
    noise_lon = noise[..., 1] / m_per_deg_lon
    noisy = coords.clone()
    mask = attention.bool()
    noisy[..., 0] = lat + noise_lat * mask
    noisy[..., 1] = lon + noise_lon * mask
    return noisy


def _apply_input_drop(attention: torch.Tensor, drop_ratio: float, generator: torch.Generator) -> torch.Tensor:
    if drop_ratio <= 0:
        return attention
    drop_ratio = max(0.0, min(1.0, float(drop_ratio)))
    out = attention.clone()
    bsz = attention.shape[0]
    for b in range(bsz):
        vlen = int(attention[b].sum().item())
        if vlen <= 1:
            continue
        num = max(1, int(vlen * drop_ratio))
        perm = torch.randperm(vlen, generator=generator, device=attention.device)
        idx = perm[:num]
        out[b, idx] = 0.0
    return out


def _build_prediction_mask(attention: torch.Tensor, pred_steps: int) -> torch.Tensor:
    bsz, seq_len = attention.shape
    mask = torch.zeros((bsz, seq_len), dtype=torch.bool, device=attention.device)
    for b in range(bsz):
        vlen = int(attention[b].sum().item())
        if vlen <= pred_steps:
            continue
        start = vlen - pred_steps
        mask[b, start:vlen] = True
    return mask


def _make_loader(dataset, indices, batch_size, num_workers, shuffle=False):
    subset = Subset(dataset, indices)
    return DataLoader(
        subset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=rb.collate_fixed,
    )


def _pick_level(tokens_l0, tokens_l1, tokens_l2, level: str):
    if level == "l1":
        return tokens_l1
    if level == "l2":
        return tokens_l2
    return tokens_l0


def _level_vocab(ckpt_args: Dict, level: str) -> int:
    if level == "l1":
        return int(ckpt_args["vocab_l1"])
    if level == "l2":
        return int(ckpt_args["vocab_l2"])
    return int(ckpt_args["vocab_l0"])


def build_token_centroids(
    loader: DataLoader,
    pack: rb.BackbonePack,
    level: str,
    device: str,
    max_samples: int,
) -> Tuple[torch.Tensor, int]:
    vocab = _level_vocab(pack.ckpt_args, level)
    sum_lat = np.zeros((vocab,), dtype=np.float64)
    sum_lon = np.zeros((vocab,), dtype=np.float64)
    count = np.zeros((vocab,), dtype=np.int64)
    total_lat = 0.0
    total_lon = 0.0
    total_n = 0
    seen = 0

    for batch in loader:
        coords = batch["coords"].to(device)
        timestamps = batch["timestamps"].to(device)
        attention = batch["attention_mask"].to(device)
        context = None
        if pack.context_index is not None:
            context = rb.context_tensor_from_index(pack.context_index, coords, pack.ckpt_args["res1"])

        with torch.no_grad():
            tokens_l0, tokens_l1, tokens_l2, _ = pack.tokenizer(
                coords,
                timestamps,
                context,
                attention_mask=attention,
            )
        tokens = _pick_level(tokens_l0, tokens_l1, tokens_l2, level).cpu().numpy()
        coords_np = coords.cpu().numpy()
        attn_np = attention.cpu().numpy()

        for b in range(tokens.shape[0]):
            mask = attn_np[b] > 0.5
            if not mask.any():
                continue
            tok = tokens[b][mask]
            pts = coords_np[b][mask]
            valid = (tok >= 0) & (tok < vocab)
            if not valid.any():
                continue
            tok = tok[valid]
            pts = pts[valid]
            np.add.at(sum_lat, tok, pts[:, 0])
            np.add.at(sum_lon, tok, pts[:, 1])
            np.add.at(count, tok, 1)
            total_lat += float(pts[:, 0].sum())
            total_lon += float(pts[:, 1].sum())
            total_n += int(pts.shape[0])
        seen += tokens.shape[0]
        if max_samples > 0 and seen >= max_samples:
            break

    if total_n <= 0:
        global_mean = (0.0, 0.0)
    else:
        global_mean = (total_lat / total_n, total_lon / total_n)

    centroids = np.zeros((vocab, 2), dtype=np.float32)
    for i in range(vocab):
        if count[i] > 0:
            centroids[i, 0] = sum_lat[i] / count[i]
            centroids[i, 1] = sum_lon[i] / count[i]
        else:
            centroids[i, 0] = global_mean[0]
            centroids[i, 1] = global_mean[1]
    return torch.from_numpy(centroids), seen


def load_h3_centroids(h3_vocab_path: str, level: str) -> torch.Tensor:
    if not h3_vocab_path:
        raise ValueError("h3_vocab_path is required for H3 centroids")
    if h3 is None:
        raise RuntimeError("h3 is not installed; cannot compute H3 centroids")
    import json
    from pathlib import Path

    path = Path(h3_vocab_path)
    with open(path, "r") as f:
        payload = json.load(f)
    if level == "l1":
        cells = payload.get("cells_l1", [])
    elif level == "l2":
        cells = payload.get("cells_l2", [])
    else:
        cells = payload.get("cells_l0", [])
    centroids = np.zeros((len(cells), 2), dtype=np.float32)
    for i, cell in enumerate(cells):
        if hasattr(h3, "cell_to_latlng"):
            lat, lon = h3.cell_to_latlng(cell)
        else:
            lat, lon = h3.h3_to_geo(cell)
        centroids[i, 0] = float(lat)
        centroids[i, 1] = float(lon)
    return torch.from_numpy(centroids)


def infer_coord_mode(dataset, sample_limit: int = 256) -> Tuple[str, float]:
    max_abs = 0.0
    if len(dataset) == 0:
        return "degrees", max_abs
    limit = min(sample_limit, len(dataset))
    for i in range(limit):
        coords = dataset.samples[i]["coords"]
        if isinstance(coords, torch.Tensor):
            val = float(coords.abs().max().item())
        else:
            val = float(np.abs(coords).max())
        if val > max_abs:
            max_abs = val
    mode = "meters" if max_abs > 360.0 else "degrees"
    return mode, max_abs


def evaluate_task(
    loader: DataLoader,
    pack: rb.BackbonePack,
    centroids: torch.Tensor,
    level: str,
    task: str,
    args,
    generator: torch.Generator,
    coord_mode: str,
) -> EvalResult:
    total_abs = 0.0
    total_sq = 0.0
    total_n = 0
    total_mask = 0
    device = args.device
    vocab = _level_vocab(pack.ckpt_args, level)
    centroids = centroids.cpu()
    exclude_unknown = bool(getattr(args, "exclude_unknown", False))
    if getattr(args, "include_unknown", False):
        exclude_unknown = False
    unknown_id = None
    if exclude_unknown and pack.ckpt_args.get("tokenizer") == "h3" and pack.ckpt_args.get("h3_vocab"):
        unknown_id = vocab - 1

    for batch in loader:
        coords_true = batch["coords"].clone()
        attention_true = batch["attention_mask"].clone()

        coords = coords_true.to(device)
        attention = attention_true.to(device)
        if args.input_drop_ratio > 0:
            attention = _apply_input_drop(attention, args.input_drop_ratio, generator)
        if args.coord_noise_std_m > 0:
            coords = _apply_coord_noise(coords, attention, args.coord_noise_std_m, generator)

        batch_in = {
            "coords": coords,
            "timestamps": batch["timestamps"].to(device),
            "attention_mask": attention,
            "start_ts": batch["start_ts"],
        }

        if task == "recovery":
            mask = rb.sample_mask(attention_true.to(device), args.mask_ratio, generator=generator)
        else:
            mask = _build_prediction_mask(attention_true.to(device), args.pred_steps)

        outputs, t0, t1, t2, _ = rb.forward_backbone(
            batch_in,
            pack,
            device=device,
            max_len=args.max_len,
            mask=mask,
        )

        preds = _pick_level(
            outputs["step_logits"]["l0"].argmax(dim=-1),
            outputs["step_logits"]["l1"].argmax(dim=-1),
            outputs["step_logits"]["l2"].argmax(dim=-1),
            level,
        ).cpu()
        true_tokens = _pick_level(t0, t1, t2, level).cpu()

        mask_cpu = mask.cpu()
        total_mask += int(mask_cpu.sum().item())
        true_coords = coords_true.cpu()
        select = mask_cpu & (preds < vocab)
        if unknown_id is not None:
            select = select & (true_tokens != unknown_id)
        if select.sum().item() == 0:
            continue

        pred_tokens = preds[select].long()
        pred_coords = centroids[pred_tokens]
        true_coords = true_coords[select]
        d = _distance_m(pred_coords, true_coords, coord_mode)
        if d.numel() == 0:
            continue
        total_abs += float(d.sum().item())
        total_sq += float((d**2).sum().item())
        total_n += int(d.numel())

    if total_n == 0:
        return EvalResult(mae_m=0.0, rmse_m=0.0, n=0, n_total=total_mask)
    mae = total_abs / total_n
    rmse = math.sqrt(total_sq / total_n)
    return EvalResult(mae_m=mae, rmse_m=rmse, n=total_n, n_total=total_mask)


# --------------- Regression-based evaluation (avoids centroid/vocab issues) ---------------


def _collect_regression_features(
    loader: DataLoader,
    pack: rb.BackbonePack,
    device: str,
    max_len: int,
    task: str,
    mask_ratio: float,
    pred_steps: int,
    generator: torch.Generator,
    noise_std_m: float = 0.0,
    drop_ratio: float = 0.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Collect (embedding, true_coord) pairs for masked/predicted positions."""
    feats, targets = [], []
    for batch in loader:
        coords_true = batch["coords"].clone()
        attention_true = batch["attention_mask"].clone()
        coords = coords_true.to(device)
        attention = attention_true.to(device)
        if drop_ratio > 0:
            attention = _apply_input_drop(attention, drop_ratio, generator)
        if noise_std_m > 0:
            coords = _apply_coord_noise(coords, attention, noise_std_m, generator)

        batch_in = {
            "coords": coords,
            "timestamps": batch["timestamps"].to(device),
            "attention_mask": attention,
            "start_ts": batch["start_ts"],
        }
        if task == "recovery":
            mask = rb.sample_mask(attention_true.to(device), mask_ratio, generator=generator)
        else:
            mask = _build_prediction_mask(attention_true.to(device), pred_steps)

        outputs, _, _, _, _ = rb.forward_backbone(
            batch_in, pack, device=device, max_len=max_len, mask=mask,
        )
        step_hidden = outputs["step_hidden"].detach().cpu()
        mask_cpu = mask.cpu()
        coords_cpu = coords_true.cpu()

        for b in range(step_hidden.shape[0]):
            sel = mask_cpu[b].bool()
            if sel.sum() == 0:
                continue
            feats.append(step_hidden[b][sel])
            targets.append(coords_cpu[b][sel, :2])

    if not feats:
        return torch.zeros((0, pack.ckpt_args["embed_dim"])), torch.zeros((0, 2))
    return torch.cat(feats, dim=0), torch.cat(targets, dim=0)


def _train_regression_head(
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    val_x: torch.Tensor,
    val_y: torch.Tensor,
    epochs: int,
    lr: float,
    batch_size: int,
    device: str,
) -> torch.nn.Linear:
    """Train a small linear head to map embeddings -> (lat, lon)."""
    in_dim = train_x.shape[-1]
    head = torch.nn.Linear(in_dim, 2).to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=1e-4)

    train_x = train_x.to(device)
    train_y = train_y.to(device).float()
    val_x = val_x.to(device)
    val_y = val_y.to(device).float()

    y_mean = train_y.mean(dim=0, keepdim=True)
    y_std = train_y.std(dim=0, keepdim=True).clamp(min=1e-6)
    train_yn = (train_y - y_mean) / y_std
    val_yn = (val_y - y_mean) / y_std

    best_state = None
    best_val = float("inf")
    for _ in range(max(1, epochs)):
        perm = torch.randperm(train_x.shape[0], device=device)
        train_x = train_x[perm]
        train_yn = train_yn[perm]
        for start in range(0, train_x.shape[0], batch_size):
            xb = train_x[start : start + batch_size]
            yb = train_yn[start : start + batch_size]
            pred = head(xb)
            loss = torch.nn.functional.mse_loss(pred, yb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        with torch.no_grad():
            val_loss = torch.nn.functional.mse_loss(head(val_x), val_yn).item()
            if val_loss < best_val:
                best_val = val_loss
                best_state = {k: v.cpu().clone() for k, v in head.state_dict().items()}
    if best_state:
        head.load_state_dict(best_state)
    # Store normalization stats on the head for inference
    head._y_mean = y_mean.cpu()
    head._y_std = y_std.cpu()
    return head


def _predict_with_head(head: torch.nn.Linear, x: torch.Tensor, device: str) -> torch.Tensor:
    x = x.to(device)
    with torch.no_grad():
        pred_n = head(x)
    y_mean = head._y_mean.to(device)
    y_std = head._y_std.to(device)
    return (pred_n * y_std + y_mean).cpu()


def evaluate_task_regression(
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    pack: rb.BackbonePack,
    task: str,
    args,
    generator: torch.Generator,
    coord_mode: str,
) -> Dict[str, dict]:
    """Evaluate recovery/prediction using a regression head instead of token centroids."""
    print(f"  [regression] collecting {task} train features...")
    train_x, train_y = _collect_regression_features(
        train_loader, pack, args.device, args.max_len, task,
        args.mask_ratio, args.pred_steps, generator,
        args.coord_noise_std_m, args.input_drop_ratio,
    )
    print(f"  [regression] collecting {task} val features...")
    val_x, val_y = _collect_regression_features(
        val_loader, pack, args.device, args.max_len, task,
        args.mask_ratio, args.pred_steps, generator,
        args.coord_noise_std_m, args.input_drop_ratio,
    )
    print(f"  [regression] collecting {task} test features...")
    test_x, test_y = _collect_regression_features(
        test_loader, pack, args.device, args.max_len, task,
        args.mask_ratio, args.pred_steps, generator,
        args.coord_noise_std_m, args.input_drop_ratio,
    )
    print(f"  [regression] train={train_x.shape[0]} val={val_x.shape[0]} test={test_x.shape[0]}")

    if train_x.shape[0] == 0 or test_x.shape[0] == 0:
        empty = EvalResult(mae_m=0.0, rmse_m=0.0, n=0).__dict__
        return {"train": empty, "val": empty, "test": empty}

    head = _train_regression_head(
        train_x, train_y, val_x, val_y,
        epochs=args.regression_epochs,
        lr=args.regression_lr,
        batch_size=args.regression_batch_size,
        device=args.device,
    )

    results = {}
    for split_name, sx, sy in [("train", train_x, train_y), ("val", val_x, val_y), ("test", test_x, test_y)]:
        pred = _predict_with_head(head, sx, args.device)
        d = _distance_m(pred, sy, coord_mode)
        mae = float(d.mean().item()) if d.numel() > 0 else 0.0
        rmse = float(torch.sqrt((d ** 2).mean()).item()) if d.numel() > 0 else 0.0
        results[split_name] = EvalResult(mae_m=mae, rmse_m=rmse, n=int(d.numel())).__dict__

    return results


def run_eval(args) -> Dict:
    rb.set_seed(args.seed)
    pack = rb.load_backbone(
        args.checkpoint,
        device=args.device,
        override_max_len=args.max_len,
        disable_graph=args.disable_graph,
    )
    local_data = args.local_data or pack.ckpt_args.get("local_data", "")
    if not local_data:
        raise ValueError("No local_data provided and checkpoint args do not include local_data")
    if not Path(local_data).exists():
        raise FileNotFoundError(f"local_data not found: {local_data}")

    raw_records = rb.load_local_data(local_data)
    dataset = rb.FixedTrajectoryDataset(raw_records, max_len=args.max_len, sample_limit=args.sample_limit)
    if len(dataset) < 10:
        raise RuntimeError(f"not enough samples: {len(dataset)}")

    coord_mode = getattr(args, "coord_mode", "auto")
    coord_max_abs = None
    if coord_mode == "auto":
        coord_mode, coord_max_abs = infer_coord_mode(dataset)
        print(f"[unitraj_eval] coord_mode auto -> {coord_mode} (max_abs={coord_max_abs:.3f})")
    else:
        _, coord_max_abs = infer_coord_mode(dataset)

    split_modes = ["random", "temporal"] if args.split_mode == "both" else [args.split_mode]
    # Default exclude_unknown to True when an H3 vocab is present, unless explicitly overridden.
    if getattr(args, "exclude_unknown", None) is None:
        args.exclude_unknown = bool(pack.ckpt_args.get("tokenizer") == "h3" and pack.ckpt_args.get("h3_vocab"))
    if getattr(args, "include_unknown", False):
        args.exclude_unknown = False
    results = {
        "checkpoint": args.checkpoint,
        "dataset": local_data,
        "samples": len(dataset),
        "settings": {
            "max_len": args.max_len,
            "mask_ratio": args.mask_ratio,
            "pred_steps": args.pred_steps,
            "centroid_level": args.centroid_level,
            "centroid_samples": args.centroid_samples,
            "centroid_fraction": args.centroid_fraction,
            "coord_noise_std_m": args.coord_noise_std_m,
            "input_drop_ratio": args.input_drop_ratio,
            "coord_mode": coord_mode,
            "coord_max_abs": coord_max_abs,
            "exclude_unknown": bool(args.exclude_unknown),
        },
        "splits": {},
    }

    for mode in split_modes:
        if mode == "all":
            full_idx = list(range(len(dataset)))
            train_idx, val_idx, test_idx = full_idx, full_idx, full_idx
        else:
            train_idx, val_idx, test_idx = rb.split_indices(dataset, mode=mode, seed=args.seed)
        train_loader = _make_loader(
            dataset, train_idx, args.batch_size, args.num_workers, shuffle=False
        )
        val_loader = _make_loader(
            dataset, val_idx, args.batch_size, args.num_workers, shuffle=False
        )
        test_loader = _make_loader(
            dataset, test_idx, args.batch_size, args.num_workers, shuffle=False
        )

        centroid_limit = args.centroid_samples
        if centroid_limit <= 0 and args.centroid_fraction > 0:
            centroid_limit = max(1, int(len(train_idx) * args.centroid_fraction))

        centroids = None
        used = 0
        centroid_source = "data"
        if (
            centroid_limit <= 0
            and pack.ckpt_args.get("tokenizer") == "h3"
            and pack.ckpt_args.get("h3_vocab")
        ):
            try:
                centroids = load_h3_centroids(pack.ckpt_args.get("h3_vocab"), args.centroid_level)
                centroid_source = "h3"
            except Exception as exc:
                print(f"warning: failed to load H3 centroids ({exc}); falling back to data centroids")
                centroids = None

        if centroids is None:
            centroids, used = build_token_centroids(
                train_loader,
                pack,
                args.centroid_level,
                device=args.device,
                max_samples=centroid_limit,
            )
            centroid_source = "data"

        rng = torch.Generator(device=args.device)
        rng.manual_seed(args.seed + 13)

        split_results = {
            "centroid_level": args.centroid_level,
            "centroid_samples_used": used,
            "centroid_source": centroid_source,
        }
        tasks = ["recovery", "prediction"] if args.task == "both" else [args.task]

        use_regression = getattr(args, "use_regression", False)
        for task in tasks:
            if use_regression:
                split_results[task] = evaluate_task_regression(
                    train_loader, val_loader, test_loader,
                    pack, task, args, rng, coord_mode,
                )
                # Also compute centroid-based metrics for comparison
                split_results[task + "_centroid"] = {
                    "train": evaluate_task(
                        train_loader, pack, centroids, args.centroid_level, task, args, rng, coord_mode
                    ).__dict__,
                    "test": evaluate_task(
                        test_loader, pack, centroids, args.centroid_level, task, args, rng, coord_mode
                    ).__dict__,
                }
            else:
                split_results[task] = {
                    "train": evaluate_task(
                        train_loader, pack, centroids, args.centroid_level, task, args, rng, coord_mode
                    ).__dict__,
                    "val": evaluate_task(
                        val_loader, pack, centroids, args.centroid_level, task, args, rng, coord_mode
                    ).__dict__,
                    "test": evaluate_task(
                        test_loader, pack, centroids, args.centroid_level, task, args, rng, coord_mode
                    ).__dict__,
                }

        results["splits"][mode] = split_results

    return results


def main():
    args = parse_args()
    results = run_eval(args)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"saved UniTraj-style eval results: {out}")


if __name__ == "__main__":
    main()

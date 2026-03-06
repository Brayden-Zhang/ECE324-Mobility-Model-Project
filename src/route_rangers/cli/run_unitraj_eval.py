import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

try:
    import h3
except Exception:  # optional
    h3 = None

from route_rangers.cli import run_benchmarks as rb


@dataclass
class EvalResult:
    mae_m: float
    rmse_m: float
    n: int
    n_total: int = 0


def parse_args():
    parser = argparse.ArgumentParser(
        description="UniTraj-style recovery/prediction eval for TrajectoryFM-HMT"
    )
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--local_data", type=str, default="")
    parser.add_argument("--max_len", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sample_limit", type=int, default=0)
    parser.add_argument(
        "--split_mode",
        type=str,
        default="both",
        choices=["both", "random", "temporal", "all"],
    )
    parser.add_argument(
        "--task", type=str, default="both", choices=["both", "recovery", "prediction"]
    )
    parser.add_argument("--mask_ratio", type=float, default=0.5)
    parser.add_argument("--pred_steps", type=int, default=5)
    parser.add_argument(
        "--centroid_level", type=str, default="l0", choices=["l0", "l1", "l2"]
    )
    parser.add_argument("--centroid_samples", type=int, default=0)
    parser.add_argument("--centroid_fraction", type=float, default=0.0)
    parser.add_argument("--coord_noise_std_m", type=float, default=0.0)
    parser.add_argument("--input_drop_ratio", type=float, default=0.0)
    parser.add_argument(
        "--coord_mode",
        type=str,
        default="auto",
        choices=["auto", "degrees", "meters"],
        help=(
            "Distance mode for coords. Auto selects meters if |coord| > 360, "
            "else degrees."
        ),
    )
    parser.add_argument("--disable_graph", action="store_true")
    parser.add_argument(
        "--exclude_unknown",
        action="store_true",
        default=None,
        help=(
            "Exclude points mapped to unknown H3 id (vocab-1) in distance "
            "metrics. Defaults to True when an H3 vocab is configured."
        ),
    )
    parser.add_argument(
        "--include_unknown",
        action="store_true",
        help=(
            "Include unknown H3 id in distance metrics (usually makes "
            "MAE/RMSE less meaningful)."
        ),
    )
    parser.add_argument(
        "--use_regression",
        "--use-regression",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Use a trained regression head on step embeddings to predict "
            "coordinates instead of centroid lookup. Enabled by default for "
            "apples-to-apples MAE/RMSE comparison with UniTraj."
        ),
    )
    parser.add_argument("--regression_epochs", type=int, default=8)
    parser.add_argument("--regression_lr", type=float, default=2e-3)
    parser.add_argument("--regression_batch_size", type=int, default=2048)
    parser.add_argument(
        "--include_centroid_baseline",
        action="store_true",
        help=(
            "When using regression eval, also compute centroid-based fallback "
            "metrics (slower; diagnostics only)."
        ),
    )
    parser.add_argument("--output", type=str, default="cache/unitraj_eval.json")
    return parser.parse_args()


def _extract_trajectory(record: dict):
    traj = record.get("trajectory")
    if traj is None:
        traj = record.get("traj")
    if traj is None:
        traj = record.get("points")
    return traj


def infer_coord_order_stats(raw_records, sample_limit: int = 0) -> Dict[str, object]:
    """Audit inferred raw coordinate ordering before normalization."""
    considered = 0
    invalid = 0
    detected_lat_lon = 0
    detected_lon_lat = 0
    ambiguous_kept_lat_lon = 0
    ambiguous_swapped_to_lat_lon = 0

    for rec in raw_records:
        traj = _extract_trajectory(rec)
        if traj is None:
            invalid += 1
            continue

        arr = np.asarray(traj, dtype=np.float32)
        if arr.ndim != 2 or arr.shape[1] < 2:
            invalid += 1
            continue

        lat_a, lon_a = arr[:, 0], arr[:, 1]
        lat_b, lon_b = arr[:, 1], arr[:, 0]

        valid_a = rb._is_valid_latlon(lat_a, lon_a)
        valid_b = rb._is_valid_latlon(lat_b, lon_b)
        if valid_a and not valid_b:
            detected_lat_lon += 1
        elif valid_b and not valid_a:
            detected_lon_lat += 1
        elif not valid_a and not valid_b:
            invalid += 1
        else:
            score_a = rb._orientation_score(lat_a, lon_a)
            score_b = rb._orientation_score(lat_b, lon_b)
            if score_b + 1e-6 < score_a * 0.8:
                detected_lon_lat += 1
                ambiguous_swapped_to_lat_lon += 1
            else:
                detected_lat_lon += 1
                ambiguous_kept_lat_lon += 1

        considered += 1
        if sample_limit > 0 and considered >= sample_limit:
            break

    records_total = len(raw_records)
    inferred = "unknown"
    if detected_lat_lon > detected_lon_lat:
        inferred = "mostly_lat_lon"
    elif detected_lon_lat > detected_lat_lon:
        inferred = "mostly_lon_lat"
    elif detected_lat_lon == detected_lon_lat and detected_lat_lon > 0:
        inferred = "mixed_tie"

    return {
        "normalized_output_order": "lat_lon",
        "inference_policy": "auto_normalize_lon_lat_plausibility_v2",
        "records_total": records_total,
        "records_considered": considered,
        "detected_lat_lon": detected_lat_lon,
        "detected_lon_lat": detected_lon_lat,
        "ambiguous_kept_lat_lon": ambiguous_kept_lat_lon,
        "ambiguous_swapped_to_lat_lon": ambiguous_swapped_to_lat_lon,
        "invalid_or_unusable": invalid,
        "inferred_majority_raw_order": inferred,
    }


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
    a = (
        torch.sin(dlat / 2) ** 2
        + torch.cos(lat1) * torch.cos(lat2) * torch.sin(dlon / 2) ** 2
    )
    c = 2 * torch.asin(torch.clamp(a.sqrt(), max=1.0))
    return 6371000.0 * c


def _euclidean_m(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    if pred.numel() == 0:
        return pred.new_zeros((0,))
    diff = pred - target
    return torch.sqrt((diff**2).sum(dim=-1))


def _distance_m(
    pred: torch.Tensor, target: torch.Tensor, coord_mode: str
) -> torch.Tensor:
    if coord_mode == "meters":
        return _euclidean_m(pred, target)
    return _haversine_m(pred, target)


def _apply_coord_noise(
    coords: torch.Tensor,
    attention: torch.Tensor,
    std_m: float,
    generator: torch.Generator,
) -> torch.Tensor:
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
        noise = (
            torch.randn(coords.shape, device=coords.device, dtype=coords.dtype) * std_m
        )
    noise_lat = noise[..., 0] / m_per_deg_lat
    noise_lon = noise[..., 1] / m_per_deg_lon
    noisy = coords.clone()
    mask = attention.bool()
    noisy[..., 0] = lat + noise_lat * mask
    noisy[..., 1] = lon + noise_lon * mask
    return noisy


def _apply_input_drop(
    attention: torch.Tensor, drop_ratio: float, generator: torch.Generator
) -> torch.Tensor:
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
            context = rb.context_tensor_from_index(
                pack.context_index, coords, pack.ckpt_args["res1"]
            )

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
    if (
        exclude_unknown
        and pack.ckpt_args.get("tokenizer") == "h3"
        and pack.ckpt_args.get("h3_vocab")
    ):
        unknown_id = vocab - 1

    for batch in loader:
        coords_true = batch["coords"].clone()
        attention_true = batch["attention_mask"].clone()

        coords = coords_true.to(device)
        attention = attention_true.to(device)
        if args.input_drop_ratio > 0:
            attention = _apply_input_drop(attention, args.input_drop_ratio, generator)
        if args.coord_noise_std_m > 0:
            coords = _apply_coord_noise(
                coords, attention, args.coord_noise_std_m, generator
            )

        batch_in = {
            "coords": coords,
            "timestamps": batch["timestamps"].to(device),
            "attention_mask": attention,
            "start_ts": batch["start_ts"],
        }

        if task == "recovery":
            mask = rb.sample_mask(
                attention_true.to(device), args.mask_ratio, generator=generator
            )
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


# --------------- Regression-based evaluation (avoids centroid/vocab issue


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
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Collect (feature, anchor_coord, true_coord) on selected positions."""
    feats, anchors, targets = [], [], []
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
            mask = rb.sample_mask(
                attention_true.to(device), mask_ratio, generator=generator
            )
        else:
            mask = _build_prediction_mask(attention_true.to(device), pred_steps)

        outputs, _, _, _, _ = rb.forward_backbone(
            batch_in,
            pack,
            device=device,
            max_len=max_len,
            mask=mask,
        )
        step_hidden = outputs["step_hidden"].detach().cpu()
        mask_cpu = mask.cpu()
        coords_obs = coords.detach().cpu()
        attention_obs = attention.detach().cpu()
        coords_cpu = coords_true.cpu()

        for b in range(step_hidden.shape[0]):
            sel = mask_cpu[b].bool()
            if sel.sum() == 0:
                continue
            valid_obs = attention_obs[b].bool()
            visible = valid_obs & (~sel)
            if visible.sum() == 0:
                visible = valid_obs
            vis_idx = visible.nonzero(as_tuple=False).squeeze(-1)
            if vis_idx.numel() == 0:
                continue

            selected_idx = sel.nonzero(as_tuple=False).squeeze(-1)
            vlen = int(valid_obs.sum().item())
            anchor_list = []
            extra_list = []
            for idx_t in selected_idx.tolist():
                prev = vis_idx[vis_idx <= idx_t]
                nxt = vis_idx[vis_idx >= idx_t]
                if prev.numel() == 0 and nxt.numel() == 0:
                    continue
                p_idx = int(prev[-1].item()) if prev.numel() > 0 else int(nxt[0].item())
                n_idx = int(nxt[0].item()) if nxt.numel() > 0 else int(prev[-1].item())

                p_coord = coords_obs[b, p_idx, :2]
                n_coord = coords_obs[b, n_idx, :2]
                if n_idx != p_idx:
                    w = float(idx_t - p_idx) / float(n_idx - p_idx)
                    anchor = p_coord * (1.0 - w) + n_coord * w
                else:
                    # For tail prediction (no future visible point):
                    # first-order
                    # extrapolation
                    # from the last two observed points is a strong,
                    # leakage-free baseline.
                    anchor = p_coord
                    if prev.numel() >= 2:
                        p2_idx = int(prev[-2].item())
                        p2_coord = coords_obs[b, p2_idx, :2]
                        denom = float(max(1, p_idx - p2_idx))
                        velocity = (p_coord - p2_coord) / denom
                        anchor = p_coord + velocity * float(idx_t - p_idx)
                    elif nxt.numel() >= 2:
                        n2_idx = int(nxt[1].item())
                        n2_coord = coords_obs[b, n2_idx, :2]
                        denom = float(max(1, n2_idx - n_idx))
                        velocity = (n2_coord - n_coord) / denom
                        anchor = n_coord - velocity * float(n_idx - idx_t)

                rel_pos = float(idx_t) / float(max(1, vlen - 1))
                span_ratio = float(abs(n_idx - p_idx)) / float(max(1, vlen))
                extra = torch.tensor(
                    [anchor[0].item(), anchor[1].item(), rel_pos, span_ratio],
                    dtype=step_hidden.dtype,
                )
                anchor_list.append(anchor)
                extra_list.append(extra)

            if not anchor_list:
                continue

            anchor_t = torch.stack(anchor_list, dim=0)
            extra_t = torch.stack(extra_list, dim=0)
            hidden_t = step_hidden[b][selected_idx[: anchor_t.shape[0]]]
            feat_t = torch.cat([hidden_t, extra_t], dim=-1)

            feats.append(feat_t)
            anchors.append(anchor_t)
            targets.append(coords_cpu[b][selected_idx[: anchor_t.shape[0]], :2])

    if not feats:
        fdim = int(pack.ckpt_args["embed_dim"]) + 4
        return torch.zeros((0, fdim)), torch.zeros((0, 2)), torch.zeros((0, 2))
    return torch.cat(feats, dim=0), torch.cat(anchors, dim=0), torch.cat(targets, dim=0)


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


def _predict_with_head(
    head: torch.nn.Linear, x: torch.Tensor, device: str
) -> torch.Tensor:
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
    """Evaluate recovery/prediction with regression vs token centroids."""
    same_train_val = val_loader is train_loader
    same_train_test = test_loader is train_loader

    print(f"  [regression] collecting {task} train features...")
    train_x, train_anchor, train_y = _collect_regression_features(
        train_loader,
        pack,
        args.device,
        args.max_len,
        task,
        args.mask_ratio,
        args.pred_steps,
        generator,
        args.coord_noise_std_m,
        args.input_drop_ratio,
    )

    if same_train_val:
        val_x, val_anchor, val_y = train_x, train_anchor, train_y
    else:
        print(f"  [regression] collecting {task} val features...")
        val_x, val_anchor, val_y = _collect_regression_features(
            val_loader,
            pack,
            args.device,
            args.max_len,
            task,
            args.mask_ratio,
            args.pred_steps,
            generator,
            args.coord_noise_std_m,
            args.input_drop_ratio,
        )

    if same_train_test:
        test_x, test_anchor, test_y = train_x, train_anchor, train_y
    else:
        print(f"  [regression] collecting {task} test features...")
        test_x, test_anchor, test_y = _collect_regression_features(
            test_loader,
            pack,
            args.device,
            args.max_len,
            task,
            args.mask_ratio,
            args.pred_steps,
            generator,
            args.coord_noise_std_m,
            args.input_drop_ratio,
        )
    print(
        f"  [regression] train={train_x.shape[0]} "
        f"val={val_x.shape[0]} test={test_x.shape[0]}"
    )

    if train_x.shape[0] == 0 or test_x.shape[0] == 0:
        empty = EvalResult(mae_m=0.0, rmse_m=0.0, n=0).__dict__
        return {"train": empty, "val": empty, "test": empty}

    # Learn residuals over a trajectory-aware interpolation anchor.
    train_res = train_y - train_anchor
    val_res = val_y - val_anchor
    head = _train_regression_head(
        train_x,
        train_res,
        val_x,
        val_res,
        epochs=args.regression_epochs,
        lr=args.regression_lr,
        batch_size=args.regression_batch_size,
        device=args.device,
    )

    with torch.no_grad():
        val_pred_res = _predict_with_head(head, val_x, args.device)
    val_pred = val_anchor + val_pred_res
    d_val_base = _distance_m(val_anchor, val_y, coord_mode)
    d_val_res = _distance_m(val_pred, val_y, coord_mode)
    mae_val_base = float(d_val_base.mean().item()) if d_val_base.numel() > 0 else 0.0
    mae_val_res = float(d_val_res.mean().item()) if d_val_res.numel() > 0 else 0.0
    use_residual = mae_val_res <= mae_val_base * 1.02

    results = {}
    for split_name, sx, sb, sy in [
        ("train", train_x, train_anchor, train_y),
        ("val", val_x, val_anchor, val_y),
        ("test", test_x, test_anchor, test_y),
    ]:
        if use_residual:
            pred = sb + _predict_with_head(head, sx, args.device)
        else:
            pred = sb
        d = _distance_m(pred, sy, coord_mode)
        mae = float(d.mean().item()) if d.numel() > 0 else 0.0
        rmse = float(torch.sqrt((d**2).mean()).item()) if d.numel() > 0 else 0.0
        results[split_name] = EvalResult(
            mae_m=mae, rmse_m=rmse, n=int(d.numel())
        ).__dict__

    results["_meta"] = {
        "strategy": (
            "anchor_interpolation_plus_residual"
            if use_residual
            else "anchor_interpolation_only"
        ),
        "val_anchor_mae_m": mae_val_base,
        "val_residual_mae_m": mae_val_res,
    }
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
        raise ValueError(
            "No local_data provided and checkpoint args do not include local_data"
        )
    if not Path(local_data).exists():
        raise FileNotFoundError(f"local_data not found: {local_data}")

    raw_records = rb.load_local_data(local_data)
    coord_order = infer_coord_order_stats(raw_records, sample_limit=args.sample_limit)
    dataset = rb.FixedTrajectoryDataset(
        raw_records, max_len=args.max_len, sample_limit=args.sample_limit
    )
    if len(dataset) < 10:
        raise RuntimeError(f"not enough samples: {len(dataset)}")

    coord_mode = getattr(args, "coord_mode", "auto")
    coord_max_abs = None
    if coord_mode == "auto":
        coord_mode, coord_max_abs = infer_coord_mode(dataset)
        print(
            "[unitraj_eval] coord_mode auto -> "
            f"{coord_mode} (max_abs={coord_max_abs:.3f})"
        )
    else:
        _, coord_max_abs = infer_coord_mode(dataset)

    split_modes = (
        ["random", "temporal"] if args.split_mode == "both" else [args.split_mode]
    )
    # Default exclude_unknown to True when an H3 vocab is present, unless
    # explicitly overridden.
    if getattr(args, "exclude_unknown", None) is None:
        args.exclude_unknown = bool(
            pack.ckpt_args.get("tokenizer") == "h3" and pack.ckpt_args.get("h3_vocab")
        )
    if getattr(args, "include_unknown", False):
        args.exclude_unknown = False
    results = {
        "checkpoint": args.checkpoint,
        "dataset": local_data,
        "samples": len(dataset),
        "coord_order": coord_order,
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
            "use_regression": bool(args.use_regression),
            "include_centroid_baseline": bool(args.include_centroid_baseline),
        },
        "splits": {},
    }

    for mode in split_modes:
        if mode == "all":
            full_idx = list(range(len(dataset)))
            train_idx, val_idx, test_idx = full_idx, full_idx, full_idx
            train_loader = _make_loader(
                dataset, train_idx, args.batch_size, args.num_workers, shuffle=False
            )
            val_loader = train_loader
            test_loader = train_loader
        else:
            train_idx, val_idx, test_idx = rb.split_indices(
                dataset, mode=mode, seed=args.seed
            )
            train_loader = _make_loader(
                dataset, train_idx, args.batch_size, args.num_workers, shuffle=False
            )
            val_loader = _make_loader(
                dataset, val_idx, args.batch_size, args.num_workers, shuffle=False
            )
            test_loader = _make_loader(
                dataset, test_idx, args.batch_size, args.num_workers, shuffle=False
            )

        rng = torch.Generator(device=args.device)
        rng.manual_seed(args.seed + 13)

        need_centroids = (not bool(args.use_regression)) or bool(
            args.include_centroid_baseline
        )
        centroids = None
        used = 0
        centroid_source = "skipped"
        if need_centroids:
            centroid_limit = args.centroid_samples
            if centroid_limit <= 0 and args.centroid_fraction > 0:
                centroid_limit = max(1, int(len(train_idx) * args.centroid_fraction))

            if (
                centroid_limit <= 0
                and pack.ckpt_args.get("tokenizer") == "h3"
                and pack.ckpt_args.get("h3_vocab")
            ):
                try:
                    centroids = load_h3_centroids(
                        pack.ckpt_args.get("h3_vocab"), args.centroid_level
                    )
                    centroid_source = "h3"
                except Exception as exc:
                    print(
                        "warning: failed to load H3 centroids "
                        f"({exc}); falling back to data centroids"
                    )
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
                    train_loader,
                    val_loader,
                    test_loader,
                    pack,
                    task,
                    args,
                    rng,
                    coord_mode,
                )
                if bool(args.include_centroid_baseline):
                    split_results[task + "_centroid"] = {
                        "train": evaluate_task(
                            train_loader,
                            pack,
                            centroids,
                            args.centroid_level,
                            task,
                            args,
                            rng,
                            coord_mode,
                        ).__dict__,
                        "test": evaluate_task(
                            test_loader,
                            pack,
                            centroids,
                            args.centroid_level,
                            task,
                            args,
                            rng,
                            coord_mode,
                        ).__dict__,
                    }
            else:
                split_results[task] = {
                    "train": evaluate_task(
                        train_loader,
                        pack,
                        centroids,
                        args.centroid_level,
                        task,
                        args,
                        rng,
                        coord_mode,
                    ).__dict__,
                    "val": evaluate_task(
                        val_loader,
                        pack,
                        centroids,
                        args.centroid_level,
                        task,
                        args,
                        rng,
                        coord_mode,
                    ).__dict__,
                    "test": evaluate_task(
                        test_loader,
                        pack,
                        centroids,
                        args.centroid_level,
                        task,
                        args,
                        rng,
                        coord_mode,
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

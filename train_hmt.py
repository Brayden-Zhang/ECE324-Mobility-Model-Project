import argparse
import json
import math
import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data import random_split

from utils.hmt import HMTConfig, HMTTokenizer, TimeFeatures
from utils.hmt_dataset import (
    WorldTraceIterableDataset,
    WorldTraceMapDataset,
    WorldTraceZipIterableDataset,
    collate_batch,
)
from utils.hmt_model import TrajectoryFMHMT
from utils.flow import sample_rectified_flow_targets, flow_matching_loss
from utils.context import OSMContextIndex, context_tensor_from_index
from utils.macro_dataset import MacroDistributionDataset

LENGTH_SCALE_CENTER = 0.5
LENGTH_SCALE_FACTOR = 2.0
MIN_LOCAL_MASK_RATIO = 0.01
MAX_LOCAL_MASK_RATIO = 0.95


def parse_args():
    parser = argparse.ArgumentParser(description="Train TrajectoryFM with hierarchical mobility tokens")
    parser.add_argument("--hf_name", type=str, default="OpenTrace/WorldTrace")
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--streaming", action="store_true")
    parser.add_argument("--data_mode", type=str, default="hf_zip", choices=["hf_zip", "hf_stream", "local"])
    parser.add_argument("--worldtrace_file", type=str, default="Trajectory.zip")
    parser.add_argument("--worldtrace_local_path", type=str, default="")
    parser.add_argument("--shuffle_buffer", type=int, default=1000)
    parser.add_argument("--take", type=int, default=0, help="limit records for debug; <=0 means full split")
    parser.add_argument("--local_data", type=str, default="")
    parser.add_argument("--val_local_data", type=str, default="")
    parser.add_argument("--test_local_data", type=str, default="")
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--test_ratio", type=float, default=0.1)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--cpu_threads", type=int, default=0)
    parser.add_argument("--max_len", type=int, default=200)
    parser.add_argument("--mask_ratio", type=float, default=0.3)
    parser.add_argument("--mask_ratio_min", type=float, default=0.1)
    parser.add_argument("--mask_curriculum_steps", type=int, default=20000)
    parser.add_argument("--span_mask_prob", type=float, default=0.5)
    parser.add_argument("--span_lambda", type=float, default=3.0)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--max_steps", type=int, default=2000)
    parser.add_argument("--eval_interval", type=int, default=200)
    parser.add_argument("--save_interval", type=int, default=500)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=0.05)
    parser.add_argument("--warmup_steps", type=int, default=500)
    parser.add_argument("--min_lr_ratio", type=float, default=0.1)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--accum_steps", type=int, default=1)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--amp", dest="amp", action="store_true")
    parser.add_argument("--no_amp", dest="amp", action="store_false")
    parser.set_defaults(amp=True)

    parser.add_argument("--tokenizer", type=str, default="h3", choices=["h3", "vq"])
    parser.add_argument("--res0", type=int, default=9)
    parser.add_argument("--res1", type=int, default=7)
    parser.add_argument("--res2", type=int, default=5)
    parser.add_argument("--vocab_l0", type=int, default=16384)
    parser.add_argument("--vocab_l1", type=int, default=4096)
    parser.add_argument("--vocab_l2", type=int, default=1024)
    parser.add_argument("--hash_tokens", dest="hash_tokens", action="store_true")
    parser.add_argument("--no_hash_tokens", dest="hash_tokens", action="store_false")
    parser.set_defaults(hash_tokens=False)
    parser.add_argument("--h3_vocab", type=str, default="")

    parser.add_argument("--embed_dim", type=int, default=256)
    parser.add_argument("--depth", type=int, default=8)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--use_graph", action="store_true")
    parser.add_argument("--graph_layers", type=int, default=2)
    parser.add_argument("--graph_knn", type=int, default=8)
    parser.add_argument("--graph_temporal_window", type=int, default=2)
    parser.add_argument("--step_attention_window", type=int, default=0)
    parser.add_argument("--graph_same_region", dest="graph_same_region", action="store_true")
    parser.add_argument("--no_graph_same_region", dest="graph_same_region", action="store_false")
    parser.set_defaults(graph_same_region=True)

    parser.add_argument("--space_time_encoder", dest="space_time_encoder", action="store_true")
    parser.add_argument("--no_space_time_encoder", dest="space_time_encoder", action="store_false")
    parser.set_defaults(space_time_encoder=False)
    parser.add_argument("--space_time_freqs", type=int, default=6)

    parser.add_argument("--flow_weight", type=float, default=1.0)
    parser.add_argument("--token_weight", type=float, default=1.0)
    parser.add_argument("--dest_weight", type=float, default=0.3)
    parser.add_argument("--vq_weight", type=float, default=1.0)
    parser.add_argument("--region_weight", type=float, default=0.5)
    parser.add_argument("--consistency_weight", type=float, default=0.2)
    parser.add_argument("--region_mask_ratio", type=float, default=0.2)
    parser.add_argument("--region_mask_ratio_min", type=float, default=0.05)
    parser.add_argument("--region_mask_curriculum_steps", type=int, default=20000)
    parser.add_argument("--coord_noise_std", type=float, default=0.0)
    parser.add_argument("--use_trip_features", dest="use_trip_features", action="store_true")
    parser.add_argument("--no_trip_features", dest="use_trip_features", action="store_false")
    parser.set_defaults(use_trip_features=True)
    parser.add_argument("--length_weighted_loss", action="store_true", help="normalize token loss per-trajectory length")
    parser.add_argument(
        "--length_adaptive_masking",
        action="store_true",
        help="scale masking ratio by trajectory length to improve robustness across short/long trips",
    )
    parser.add_argument(
        "--length_mask_alpha",
        type=float,
        default=0.3,
        help="strength of length-adaptive masking (0 disables extra scaling)",
    )
    parser.add_argument(
        "--use_length_adapter",
        action="store_true",
        help="enable length-adaptive hidden gating before token heads",
    )

    parser.add_argument("--osm_context", type=str, default="")
    parser.add_argument("--osm_context_dim", type=int, default=16)
    parser.add_argument("--max_eval_batches", type=int, default=25)
    parser.add_argument("--results_path", type=str, default="")
    parser.add_argument("--ckpt_prefix", type=str, default="hmt")
    parser.add_argument("--resume", type=str, default="", help="resume from checkpoint (model/tokenizer/time)")
    parser.add_argument("--resume_optimizer", action="store_true", help="also resume optimizer/scheduler state")
    parser.add_argument("--macro_data", type=str, default="", help="macro distribution npz for multi-task training")
    parser.add_argument("--macro_batch_size", type=int, default=256)
    parser.add_argument("--macro_weight", type=float, default=1.0)
    parser.add_argument("--macro_mix_prob", type=float, default=0.5)
    parser.add_argument("--macro_eval_batches", type=int, default=25)
    return parser.parse_args()


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_macro_loaders(args):
    if not args.macro_data:
        args.macro_region_vocab = 0
        args.macro_dist_dim = 0
        return None, None, None
    dataset = MacroDistributionDataset(args.macro_data, normalize=True)
    n = len(dataset)
    if n <= 0:
        args.macro_region_vocab = 0
        args.macro_dist_dim = 0
        return None, None, None
    args.macro_region_vocab = dataset.region_vocab
    args.macro_dist_dim = dataset.dist_dim

    if n < 10:
        loader = DataLoader(
            dataset,
            batch_size=args.macro_batch_size,
            shuffle=True,
            num_workers=min(args.num_workers, 2),
        )
        return loader, None, None

    val_n = int(n * args.val_ratio) if args.val_ratio > 0 else 0
    test_n = int(n * args.test_ratio) if args.test_ratio > 0 else 0
    train_n = max(1, n - val_n - test_n)
    if train_n + val_n + test_n > n:
        train_n = n
        val_n = 0
        test_n = 0
    gen = torch.Generator().manual_seed(args.seed + 17)
    splits = [train_n, val_n, test_n]
    train_ds, val_ds, test_ds = random_split(dataset, splits, generator=gen)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.macro_batch_size,
        shuffle=True,
        num_workers=min(args.num_workers, 2),
    )
    val_loader = (
        DataLoader(val_ds, batch_size=args.macro_batch_size, shuffle=False, num_workers=0)
        if val_n > 0
        else None
    )
    test_loader = (
        DataLoader(test_ds, batch_size=args.macro_batch_size, shuffle=False, num_workers=0)
        if test_n > 0
        else None
    )
    return train_loader, val_loader, test_loader


def macro_batch_loss(model, time_encoder, batch, device: str) -> torch.Tensor:
    region_idx = batch["region_idx"].to(device)
    time_ts = batch["time_ts"].to(device)
    dist = batch["dist"].to(device)
    attn = torch.ones((region_idx.shape[0], 1), device=device)
    time_embed = time_encoder(time_ts.unsqueeze(1), attn)[:, 0]
    logits = model.macro_logits(region_idx, time_embed)
    return F.kl_div(F.log_softmax(logits.float(), dim=-1), dist, reduction="batchmean")


def evaluate_macro(model, time_encoder, loader, device: str, max_batches: int) -> float:
    if loader is None:
        return 0.0
    model.eval()
    total_loss = 0.0
    total_n = 0
    with torch.no_grad():
        for i, batch in enumerate(loader):
            if max_batches > 0 and i >= max_batches:
                break
            loss = macro_batch_loss(model, time_encoder, batch, device)
            bs = batch["dist"].shape[0]
            total_loss += float(loss.item()) * bs
            total_n += int(bs)
    if total_n == 0:
        return 0.0
    return total_loss / total_n


def load_h3_vocab_sizes(path: str):
    import json
    from pathlib import Path

    vocab_path = Path(path)
    with open(vocab_path, "r") as f:
        payload = json.load(f)
    sizes = (
        len(payload.get("cells_l0", [])),
        len(payload.get("cells_l1", [])),
        len(payload.get("cells_l2", [])),
    )
    return sizes


def load_local_data(path):
    import pandas as pd

    if path.endswith(".parquet"):
        df = pd.read_parquet(path)
    else:
        df = pd.read_pickle(path)
    samples = df.to_dict(orient="records")
    return samples


def make_dataloader(args, dataset):
    is_iterable = isinstance(dataset, torch.utils.data.IterableDataset)
    workers = 0 if is_iterable else args.num_workers
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=not is_iterable,
        num_workers=workers,
        pin_memory=args.device.startswith("cuda"),
        persistent_workers=workers > 0,
        collate_fn=collate_batch,
    )


def _local_splits(samples, seed: int, val_ratio: float, test_ratio: float):
    n = len(samples)
    if n < 3:
        return samples, samples, samples
    rng = random.Random(seed)
    idx = list(range(n))
    rng.shuffle(idx)
    n_val = int(n * max(0.0, min(0.49, val_ratio)))
    n_test = int(n * max(0.0, min(0.49, test_ratio)))
    if n_val + n_test >= n:
        n_val = max(1, n // 5)
        n_test = max(1, n // 5)
    n_train = max(1, n - n_val - n_test)
    train_idx = idx[:n_train]
    val_idx = idx[n_train : n_train + n_val]
    test_idx = idx[n_train + n_val :]
    if not val_idx:
        val_idx = train_idx[: min(len(train_idx), max(1, n // 10))]
    if not test_idx:
        test_idx = val_idx
    train = [samples[i] for i in train_idx]
    val = [samples[i] for i in val_idx]
    test = [samples[i] for i in test_idx]
    return train, val, test


def build_dataloaders(args, take_override=None):
    take = take_override
    if take is None:
        take = args.take if args.take > 0 else None

    data_mode = args.data_mode
    if args.streaming:
        data_mode = "hf_stream"
    if args.local_data:
        data_mode = "local"

    if data_mode == "local":
        train_samples = load_local_data(args.local_data)
        if take is not None:
            train_samples = train_samples[:take]

        if args.val_local_data:
            val_samples = load_local_data(args.val_local_data)
            if take is not None:
                val_samples = val_samples[:take]
            auto_test_samples = val_samples
        else:
            train_samples, val_samples, auto_test_samples = _local_splits(
                train_samples,
                seed=args.seed,
                val_ratio=args.val_ratio,
                test_ratio=args.test_ratio,
            )

        if args.test_local_data:
            test_samples = load_local_data(args.test_local_data)
            if take is not None:
                test_samples = test_samples[:take]
        else:
            test_samples = auto_test_samples

        train_ds = WorldTraceMapDataset(train_samples, max_len=args.max_len, mask_ratio=args.mask_ratio)
        val_ds = WorldTraceMapDataset(val_samples, max_len=args.max_len, mask_ratio=args.mask_ratio)
        test_ds = WorldTraceMapDataset(test_samples, max_len=args.max_len, mask_ratio=args.mask_ratio)
        return make_dataloader(args, train_ds), make_dataloader(args, val_ds), make_dataloader(args, test_ds)
    elif data_mode == "hf_stream":
        train_ds = WorldTraceIterableDataset(
            args.hf_name,
            split=args.split,
            max_len=args.max_len,
            mask_ratio=args.mask_ratio,
            shuffle_buffer=args.shuffle_buffer,
            take=take,
        )
        eval_take = take if take is not None else min(5000, args.batch_size * 200)
        val_ds = WorldTraceIterableDataset(
            args.hf_name,
            split=args.split,
            max_len=args.max_len,
            mask_ratio=args.mask_ratio,
            shuffle_buffer=args.shuffle_buffer,
            take=eval_take,
        )
        return make_dataloader(args, train_ds), make_dataloader(args, val_ds), make_dataloader(args, val_ds)
    else:
        train_ds = WorldTraceZipIterableDataset(
            args.hf_name,
            filename=args.worldtrace_file,
            max_len=args.max_len,
            mask_ratio=args.mask_ratio,
            shuffle_buffer=args.shuffle_buffer,
            take=take,
            local_path=args.worldtrace_local_path,
            seed=args.seed,
        )
        eval_take = take if take is not None else min(5000, args.batch_size * 200)
        val_ds = WorldTraceZipIterableDataset(
            args.hf_name,
            filename=args.worldtrace_file,
            max_len=args.max_len,
            mask_ratio=args.mask_ratio,
            shuffle_buffer=args.shuffle_buffer,
            take=eval_take,
            local_path=args.worldtrace_local_path,
            seed=args.seed + 17,
        )
        return make_dataloader(args, train_ds), make_dataloader(args, val_ds), make_dataloader(args, val_ds)


def mask_tokens(tokens: torch.Tensor, mask_indices: torch.Tensor, attention_mask: torch.Tensor, mask_id: int):
    masked = tokens.clone()
    mask = torch.zeros_like(tokens, dtype=torch.bool)
    for b in range(tokens.shape[0]):
        valid_len = int(attention_mask[b].sum().item())
        for idx in mask_indices[b]:
            if idx < 0:
                continue
            if idx < valid_len:
                mask[b, idx] = True
        masked[b, mask[b]] = mask_id
    return masked, mask


def scheduled_ratio(step: int, min_ratio: float, max_ratio: float, curriculum_steps: int) -> float:
    lo = min(min_ratio, max_ratio)
    hi = max(min_ratio, max_ratio)
    if curriculum_steps <= 0:
        return hi
    progress = min(1.0, float(step) / float(curriculum_steps))
    return lo + (hi - lo) * progress


def sample_mask_indices(
    attention_mask: torch.Tensor,
    mask_ratio: float,
    span_mask_prob: float,
    span_lambda: float,
    length_adaptive: bool = False,
    length_alpha: float = 0.3,
) -> torch.Tensor:
    mask_ratio = max(0.0, min(1.0, float(mask_ratio)))
    bsz, _ = attention_mask.shape
    indices = []
    max_m = 1
    for b in range(bsz):
        valid_len = int(attention_mask[b].sum().item())
        if valid_len <= 0:
            idx = torch.empty((0,), dtype=torch.long, device=attention_mask.device)
            indices.append(idx)
            continue
        local_ratio = mask_ratio
        if length_adaptive:
            length_frac = float(valid_len) / float(max(1, attention_mask.shape[1]))
            # Center at 0.5 and scale by 2.0 so length_scale spans [-1, 1] from shortest to longest trips.
            length_scale = (length_frac - LENGTH_SCALE_CENTER) * LENGTH_SCALE_FACTOR
            local_ratio = mask_ratio * (1.0 + float(length_alpha) * length_scale)
            local_ratio = max(MIN_LOCAL_MASK_RATIO, min(MAX_LOCAL_MASK_RATIO, local_ratio))
        target = max(1, int(valid_len * local_ratio))
        selected = torch.zeros((valid_len,), dtype=torch.bool, device=attention_mask.device)
        while int(selected.sum().item()) < target:
            remaining = target - int(selected.sum().item())
            if random.random() < span_mask_prob and valid_len > 2:
                start = random.randint(0, valid_len - 1)
                span = max(1, int(np.random.poisson(max(span_lambda, 1e-3))))
                span = min(span, remaining)
                end = min(valid_len, start + span)
                selected[start:end] = True
            else:
                start = random.randint(0, valid_len - 1)
                selected[start] = True
        idx = selected.nonzero(as_tuple=False).squeeze(-1)
        indices.append(idx)
        max_m = max(max_m, int(idx.shape[0]))

    mask_idx = torch.full((bsz, max_m), -1, dtype=torch.long, device=attention_mask.device)
    for b, idx in enumerate(indices):
        if idx.numel() > 0:
            mask_idx[b, : idx.numel()] = idx
    return mask_idx


def apply_coordinate_noise(coords: torch.Tensor, attention_mask: torch.Tensor, noise_std: float) -> torch.Tensor:
    if noise_std <= 0:
        return coords
    noise = torch.randn_like(coords) * noise_std
    noisy = coords + noise * attention_mask.unsqueeze(-1)
    noisy_lat = noisy[..., 0].clamp(min=-90.0, max=90.0)
    noisy_lon = noisy[..., 1].clamp(min=-180.0, max=180.0)
    return torch.stack([noisy_lat, noisy_lon], dim=-1)


def masked_cross_entropy(logits: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor):
    # logits: [B, L, V], targets: [B, L], mask: [B, L]
    if mask.sum() == 0:
        return torch.tensor(0.0, device=logits.device)
    valid = mask & (targets >= 0) & (targets < logits.shape[-1])
    if valid.sum() == 0:
        return torch.tensor(0.0, device=logits.device)
    logits = logits[valid]
    targets = targets[valid]
    return torch.nn.functional.cross_entropy(logits, targets)


def masked_cross_entropy_weighted(logits: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor):
    if mask.sum() == 0:
        return torch.tensor(0.0, device=logits.device)
    valid = mask & (targets >= 0) & (targets < logits.shape[-1])
    if valid.sum() == 0:
        return torch.tensor(0.0, device=logits.device)
    bsz, seq_len, vocab = logits.shape
    loss = torch.nn.functional.cross_entropy(
        logits.view(-1, vocab),
        targets.view(-1),
        reduction="none",
    ).view(bsz, seq_len)
    valid_f = valid.float()
    loss = loss * valid_f
    denom = valid_f.sum(dim=1).clamp(min=1.0)
    loss_per = loss.sum(dim=1) / denom
    return loss_per.mean()


def sanitize_targets(targets: torch.Tensor, vocab_size: int) -> torch.Tensor:
    if vocab_size <= 0:
        return targets
    return targets.clamp(min=0, max=vocab_size - 1)


def heteroscedastic_dest_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    log_var: torch.Tensor,
) -> torch.Tensor:
    ce = torch.nn.functional.cross_entropy(logits, targets, reduction="none")
    precision = torch.exp(-log_var)
    return (precision * ce + log_var).mean()


def build_micro_flows(step_to_region: torch.Tensor, attention_mask: torch.Tensor, region_mask: torch.Tensor):
    # step_to_region: [B, L], region_mask: [B, R]
    bsz, seq_len = step_to_region.shape
    region_count = region_mask.shape[1]
    flows = torch.zeros((bsz, region_count, region_count), device=step_to_region.device)
    for b in range(bsz):
        valid_len = int(attention_mask[b].sum().item())
        if valid_len <= 1 or region_count == 0:
            continue
        idx = step_to_region[b, :valid_len]
        for t in range(valid_len - 1):
            i = idx[t].item()
            j = idx[t + 1].item()
            if i >= 0 and j >= 0 and i < region_count and j < region_count:
                flows[b, i, j] += 1.0
    return flows


def flow_consistency_loss(meso_logits: torch.Tensor, micro_flows: torch.Tensor, region_mask: torch.Tensor):
    if micro_flows.numel() == 0:
        return torch.tensor(0.0, device=meso_logits.device)
    dest_mask = region_mask.unsqueeze(1)
    min_score = -65504.0 if meso_logits.dtype == torch.float16 else -1e9
    masked_logits = meso_logits.masked_fill(~dest_mask, min_score)
    meso = torch.softmax(masked_logits, dim=-1)
    micro = micro_flows / (micro_flows.sum(dim=-1, keepdim=True) + 1e-6)
    row_mask = (micro_flows.sum(dim=-1) > 0) & region_mask
    if row_mask.sum() == 0:
        return torch.tensor(0.0, device=meso_logits.device)
    diff = (meso - micro).pow(2) * dest_mask.float()
    diff = diff.sum(dim=-1) / dest_mask.sum(dim=-1).clamp(min=1.0)
    return (diff * row_mask.float()).sum() / (row_mask.sum() + 1e-6)


def compute_trip_features(coords: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    # coords: [B, L, 2], attention_mask: [B, L]
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
    trip_len_log = torch.log1p(trip_len).expand_as(attention_mask) * attention_mask
    total_dist_log = torch.log1p(total_dist).expand_as(attention_mask) * attention_mask

    feats = torch.stack([progress_step, progress_dist, trip_len_log, total_dist_log], dim=-1)
    return feats


def evaluate(model, tokenizer, time_encoder, dataloader, args, context_index=None):
    model.eval()
    total_token_loss = 0.0
    total_flow_loss = 0.0
    total_dest_loss = 0.0
    total_region_loss = 0.0
    total_consistency_loss = 0.0
    total_token_acc_l0 = 0.0
    total_token_acc_l1 = 0.0
    total_token_acc_l2 = 0.0
    total_dest_top1 = 0.0
    total_dest_top5 = 0.0
    total_batches = 0
    use_amp = args.amp and args.device.startswith("cuda")
    autocast_device = "cuda" if args.device.startswith("cuda") else "cpu"
    autocast_dtype = torch.float16 if autocast_device == "cuda" else torch.bfloat16
    with torch.no_grad():
        for batch in dataloader:
            coords = batch["coords"].to(args.device)
            timestamps = batch["timestamps"].to(args.device)
            attention = batch["attention_mask"].to(args.device)
            mask_indices = sample_mask_indices(
                attention,
                mask_ratio=args.mask_ratio,
                span_mask_prob=args.span_mask_prob,
                span_lambda=args.span_lambda,
                length_adaptive=args.length_adaptive_masking,
                length_alpha=args.length_mask_alpha,
            )

            context = None
            if context_index is not None:
                context = context_tensor_from_index(context_index, coords, args.res1)

            tokens_l0, tokens_l1, tokens_l2, vq_loss = tokenizer(
                coords,
                timestamps,
                context,
                attention_mask=attention,
            )
            tokens_l0 = tokens_l0.to(args.device)
            tokens_l1 = tokens_l1.to(args.device)
            tokens_l2 = tokens_l2.to(args.device)
            tokens_l0 = sanitize_targets(tokens_l0, args.vocab_l0)
            tokens_l1 = sanitize_targets(tokens_l1, args.vocab_l1)
            tokens_l2 = sanitize_targets(tokens_l2, args.vocab_l2)
            tokens_l0 = sanitize_targets(tokens_l0, args.vocab_l0)
            tokens_l1 = sanitize_targets(tokens_l1, args.vocab_l1)
            tokens_l2 = sanitize_targets(tokens_l2, args.vocab_l2)

            masked_l0, mask = mask_tokens(tokens_l0, mask_indices, attention, args.vocab_l0)
            masked_l1, _ = mask_tokens(tokens_l1, mask_indices, attention, args.vocab_l1)
            masked_l2, _ = mask_tokens(tokens_l2, mask_indices, attention, args.vocab_l2)
            trip_features = compute_trip_features(coords, attention) if args.use_trip_features else None

            with torch.autocast(device_type=autocast_device, dtype=autocast_dtype, enabled=use_amp):
                time_embed = time_encoder(timestamps, attention)
                outputs = model(
                    masked_l0,
                    masked_l1,
                    masked_l2,
                    time_embed,
                    attention,
                    timestamps=timestamps,
                    context=context,
                    trip_features=trip_features,
                    coords=coords,
                    region_mask_ratio=args.region_mask_ratio,
                    region_source_l1=tokens_l1,
                    region_source_l2=tokens_l2,
                )

                if args.length_weighted_loss:
                    token_loss = (
                        masked_cross_entropy_weighted(outputs["step_logits"]["l0"], tokens_l0, mask)
                        + masked_cross_entropy_weighted(outputs["step_logits"]["l1"], tokens_l1, mask)
                        + masked_cross_entropy_weighted(outputs["step_logits"]["l2"], tokens_l2, mask)
                    )
                else:
                    token_loss = (
                        masked_cross_entropy(outputs["step_logits"]["l0"], tokens_l0, mask)
                        + masked_cross_entropy(outputs["step_logits"]["l1"], tokens_l1, mask)
                        + masked_cross_entropy(outputs["step_logits"]["l2"], tokens_l2, mask)
                    )
                region_loss = (
                    masked_cross_entropy(
                        outputs["region_logits"]["l1"], outputs["mid_ids"], outputs["mid_mlm_mask"]
                    )
                    + masked_cross_entropy(
                        outputs["region_logits"]["l2"], outputs["coarse_ids"], outputs["coarse_mlm_mask"]
                    )
                )

                x_t, target_v, t = sample_rectified_flow_targets(coords)
                pred_v = model.flow_head(outputs["step_hidden"], x_t, t)
                flow_loss = flow_matching_loss(pred_v, target_v, attention)

                # destination loss (mid-level)
                last_idx = attention.sum(dim=1).long() - 1
                dest_targets = tokens_l1.gather(1, last_idx.unsqueeze(1)).squeeze(1)
                dest_loss = heteroscedastic_dest_loss(
                    outputs["dest_logits"],
                    dest_targets,
                    outputs["dest_log_var"],
                )

                micro_flows = build_micro_flows(outputs["step_to_mid"], attention, outputs["mid_mask"])
                meso_logits = model.meso_flow_logits(outputs["mid_hidden"], outputs["mid_mask"])
                consistency_loss = flow_consistency_loss(meso_logits, micro_flows, outputs["mid_mask"])

            # token accuracy (all levels)
            if mask.sum() > 0:
                preds_l0 = outputs["step_logits"]["l0"].argmax(dim=-1)
                preds_l1 = outputs["step_logits"]["l1"].argmax(dim=-1)
                preds_l2 = outputs["step_logits"]["l2"].argmax(dim=-1)
                acc_l0 = (preds_l0[mask] == tokens_l0[mask]).float().mean().item()
                acc_l1 = (preds_l1[mask] == tokens_l1[mask]).float().mean().item()
                acc_l2 = (preds_l2[mask] == tokens_l2[mask]).float().mean().item()
            else:
                acc_l0 = 0.0
                acc_l1 = 0.0
                acc_l2 = 0.0

            dest_pred = outputs["dest_logits"].argmax(dim=-1)
            dest_top1 = (dest_pred == dest_targets).float().mean().item()
            k = min(5, outputs["dest_logits"].shape[-1])
            dest_topk = outputs["dest_logits"].topk(k, dim=-1).indices
            dest_top5 = (
                (dest_topk == dest_targets.unsqueeze(-1)).any(dim=-1).float().mean().item()
            )

            total_token_loss += token_loss.item()
            total_flow_loss += flow_loss.item()
            total_dest_loss += dest_loss.item()
            total_region_loss += region_loss.item()
            total_consistency_loss += consistency_loss.item()
            total_token_acc_l0 += acc_l0
            total_token_acc_l1 += acc_l1
            total_token_acc_l2 += acc_l2
            total_dest_top1 += dest_top1
            total_dest_top5 += dest_top5
            total_batches += 1
            if args.max_eval_batches > 0 and total_batches >= args.max_eval_batches:
                break
    if total_batches == 0:
        return {}
    return {
        "token_loss": total_token_loss / total_batches,
        "flow_loss": total_flow_loss / total_batches,
        "dest_loss": total_dest_loss / total_batches,
        "region_loss": total_region_loss / total_batches,
        "consistency_loss": total_consistency_loss / total_batches,
        "token_acc_l0": total_token_acc_l0 / total_batches,
        "token_acc_l1": total_token_acc_l1 / total_batches,
        "token_acc_l2": total_token_acc_l2 / total_batches,
        "dest_top1": total_dest_top1 / total_batches,
        "dest_top5": total_dest_top5 / total_batches,
    }


def main():
    args = parse_args()
    os.makedirs("checkpoints", exist_ok=True)
    set_seed(args.seed)
    if args.cpu_threads and args.cpu_threads > 0:
        torch.set_num_threads(args.cpu_threads)
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision("high")

    if args.h3_vocab:
        v0, v1, v2 = load_h3_vocab_sizes(args.h3_vocab)
        args.vocab_l0 = v0
        args.vocab_l1 = v1
        args.vocab_l2 = v2

    tokenizer_cfg = HMTConfig(
        mode=args.tokenizer,
        res0=args.res0,
        res1=args.res1,
        res2=args.res2,
        vocab_l0=args.vocab_l0,
        vocab_l1=args.vocab_l1,
        vocab_l2=args.vocab_l2,
        hash_tokens=args.hash_tokens,
        h3_vocab=args.h3_vocab,
    )
    # feature dim:
    # lat/lon, relative lat/lon, deltas, speed, heading(2), log_dt, dt_norm,
    # speed_per_dt, accel, and 4 periodic time features = 17
    base_feature_dim = 17
    if args.osm_context_dim > 0 and args.osm_context:
        feature_dim = base_feature_dim + args.osm_context_dim
    else:
        feature_dim = base_feature_dim
    tokenizer = HMTTokenizer(tokenizer_cfg, feature_dim=feature_dim, embed_dim=args.embed_dim).to(args.device)
    time_encoder = TimeFeatures(args.embed_dim).to(args.device)

    macro_train_loader, macro_val_loader, macro_test_loader = build_macro_loaders(args)

    model = TrajectoryFMHMT(
        vocab_l0=args.vocab_l0,
        vocab_l1=args.vocab_l1,
        vocab_l2=args.vocab_l2,
        embed_dim=args.embed_dim,
        depth=args.depth,
        heads=args.heads,
        dropout=args.dropout,
        context_dim=args.osm_context_dim if args.osm_context else 0,
        trip_feat_dim=4 if args.use_trip_features else 0,
        max_seq_len=args.max_len * 3 + 16,
        use_graph=args.use_graph,
        graph_layers=args.graph_layers,
        graph_knn=args.graph_knn,
        graph_temporal_window=args.graph_temporal_window,
        graph_same_region=args.graph_same_region,
        step_attention_window=args.step_attention_window,
        use_spacetime=args.space_time_encoder,
        spacetime_freqs=args.space_time_freqs,
        macro_region_vocab=getattr(args, "macro_region_vocab", 0),
        macro_dist_dim=getattr(args, "macro_dist_dim", 0),
        use_length_adapter=args.use_length_adapter,
    ).to(args.device)

    optim_params = list(model.parameters())
    tok_params = [p for p in tokenizer.parameters() if p.requires_grad]
    time_params = [p for p in time_encoder.parameters() if p.requires_grad]
    if tok_params:
        optim_params.extend(tok_params)
    if time_params:
        optim_params.extend(time_params)
    optim = torch.optim.AdamW(optim_params, lr=args.lr, weight_decay=args.weight_decay)
    use_amp = args.amp and args.device.startswith("cuda")
    autocast_device = "cuda" if args.device.startswith("cuda") else "cpu"
    autocast_dtype = torch.float16 if autocast_device == "cuda" else torch.bfloat16
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    else:
        scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    def lr_lambda(current_step: int):
        warmup = max(1, args.warmup_steps)
        if current_step < warmup:
            return float(current_step + 1) / float(warmup)
        progress = (current_step - warmup) / float(max(1, args.max_steps - warmup))
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return args.min_lr_ratio + (1.0 - args.min_lr_ratio) * cosine

    scheduler = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda=lr_lambda)

    context_index = None
    if args.osm_context:
        context_index = OSMContextIndex(args.osm_context, args.osm_context_dim)

    start_step = 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location="cpu")
        model.load_state_dict(ckpt.get("model", {}), strict=False)
        tokenizer.load_state_dict(ckpt.get("tokenizer", {}), strict=False)
        if "time_encoder" in ckpt:
            time_encoder.load_state_dict(ckpt.get("time_encoder", {}), strict=False)
        start_step = int(ckpt.get("step", 0))
        if args.resume_optimizer:
            if "optimizer" in ckpt:
                optim.load_state_dict(ckpt["optimizer"])
            if "scheduler" in ckpt:
                scheduler.load_state_dict(ckpt["scheduler"])
        else:
            scheduler.last_epoch = start_step - 1

    train_loader, val_loader, test_loader = build_dataloaders(args)
    if hasattr(train_loader, "dataset") and hasattr(train_loader.dataset, "__len__"):
        try:
            print(
                f"data_sizes train={len(train_loader.dataset)} val={len(val_loader.dataset)} test={len(test_loader.dataset)}"
            )
        except Exception:
            pass

    step = start_step
    macro_iter = iter(macro_train_loader) if macro_train_loader is not None else None

    def next_macro_batch():
        nonlocal macro_iter
        if macro_train_loader is None:
            return None
        try:
            return next(macro_iter)
        except StopIteration:
            macro_iter = iter(macro_train_loader)
            return next(macro_iter)
    optim.zero_grad(set_to_none=True)
    for epoch in range(args.epochs):
        for micro_step, batch in enumerate(train_loader):
            model.train()
            coords = batch["coords"].to(args.device)
            timestamps = batch["timestamps"].to(args.device)
            attention = batch["attention_mask"].to(args.device)
            mask_ratio = scheduled_ratio(
                step=step,
                min_ratio=args.mask_ratio_min,
                max_ratio=args.mask_ratio,
                curriculum_steps=args.mask_curriculum_steps,
            )
            region_mask_ratio = scheduled_ratio(
                step=step,
                min_ratio=args.region_mask_ratio_min,
                max_ratio=args.region_mask_ratio,
                curriculum_steps=args.region_mask_curriculum_steps,
            )
            mask_indices = sample_mask_indices(
                attention,
                mask_ratio=mask_ratio,
                span_mask_prob=args.span_mask_prob,
                span_lambda=args.span_lambda,
                length_adaptive=args.length_adaptive_masking,
                length_alpha=args.length_mask_alpha,
            )

            coords_in = apply_coordinate_noise(coords, attention, args.coord_noise_std)

            context = None
            if context_index is not None:
                context = context_tensor_from_index(context_index, coords_in, args.res1)

            tokens_l0, tokens_l1, tokens_l2, vq_loss = tokenizer(
                coords_in,
                timestamps,
                context,
                attention_mask=attention,
            )
            tokens_l0 = tokens_l0.to(args.device)
            tokens_l1 = tokens_l1.to(args.device)
            tokens_l2 = tokens_l2.to(args.device)

            masked_l0, mask = mask_tokens(tokens_l0, mask_indices, attention, args.vocab_l0)
            masked_l1, _ = mask_tokens(tokens_l1, mask_indices, attention, args.vocab_l1)
            masked_l2, _ = mask_tokens(tokens_l2, mask_indices, attention, args.vocab_l2)
            trip_features = compute_trip_features(coords_in, attention) if args.use_trip_features else None

            with torch.autocast(device_type=autocast_device, dtype=autocast_dtype, enabled=use_amp):
                time_embed = time_encoder(timestamps, attention)
                outputs = model(
                    masked_l0,
                    masked_l1,
                    masked_l2,
                    time_embed,
                    attention,
                    timestamps=timestamps,
                    context=context,
                    trip_features=trip_features,
                    coords=coords_in,
                    region_mask_ratio=region_mask_ratio,
                    region_source_l1=tokens_l1,
                    region_source_l2=tokens_l2,
                )

                if args.length_weighted_loss:
                    token_loss = (
                        masked_cross_entropy_weighted(outputs["step_logits"]["l0"], tokens_l0, mask)
                        + masked_cross_entropy_weighted(outputs["step_logits"]["l1"], tokens_l1, mask)
                        + masked_cross_entropy_weighted(outputs["step_logits"]["l2"], tokens_l2, mask)
                    )
                else:
                    token_loss = (
                        masked_cross_entropy(outputs["step_logits"]["l0"], tokens_l0, mask)
                        + masked_cross_entropy(outputs["step_logits"]["l1"], tokens_l1, mask)
                        + masked_cross_entropy(outputs["step_logits"]["l2"], tokens_l2, mask)
                    )
                region_loss = (
                    masked_cross_entropy(
                        outputs["region_logits"]["l1"], outputs["mid_ids"], outputs["mid_mlm_mask"]
                    )
                    + masked_cross_entropy(
                        outputs["region_logits"]["l2"], outputs["coarse_ids"], outputs["coarse_mlm_mask"]
                    )
                )
                x_t, target_v, t = sample_rectified_flow_targets(coords)
                pred_v = model.flow_head(outputs["step_hidden"], x_t, t)
                flow_loss = flow_matching_loss(pred_v, target_v, attention)

                last_idx = attention.sum(dim=1).long() - 1
                dest_targets = tokens_l1.gather(1, last_idx.unsqueeze(1)).squeeze(1)
                dest_loss = heteroscedastic_dest_loss(
                    outputs["dest_logits"],
                    dest_targets,
                    outputs["dest_log_var"],
                )

                micro_flows = build_micro_flows(outputs["step_to_mid"], attention, outputs["mid_mask"])
                meso_logits = model.meso_flow_logits(outputs["mid_hidden"], outputs["mid_mask"])
                consistency_loss = flow_consistency_loss(meso_logits, micro_flows, outputs["mid_mask"])

                macro_loss = torch.tensor(0.0, device=args.device)
                if (
                    macro_train_loader is not None
                    and model.macro_head is not None
                    and random.random() < args.macro_mix_prob
                ):
                    macro_batch = next_macro_batch()
                    if macro_batch is not None:
                        macro_loss = macro_batch_loss(model, time_encoder, macro_batch, args.device)

                loss = (
                    args.token_weight * token_loss
                    + args.region_weight * region_loss
                    + args.flow_weight * flow_loss
                    + args.dest_weight * dest_loss
                    + args.consistency_weight * consistency_loss
                    + args.vq_weight * vq_loss
                    + args.macro_weight * macro_loss
                )

            loss_to_backprop = loss / max(1, args.accum_steps)
            if use_amp:
                scaler.scale(loss_to_backprop).backward()
            else:
                loss_to_backprop.backward()

            should_step = (micro_step + 1) % max(1, args.accum_steps) == 0
            if should_step:
                if args.grad_clip > 0:
                    if use_amp:
                        scaler.unscale_(optim)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                did_step = True
                if use_amp:
                    scale_before = scaler.get_scale()
                    scaler.step(optim)
                    scaler.update()
                    did_step = scaler.get_scale() >= scale_before
                else:
                    optim.step()
                optim.zero_grad(set_to_none=True)
                if did_step:
                    scheduler.step()

                if step % 20 == 0:
                    lr = optim.param_groups[0]["lr"]
                    print(
                        f"step={step} lr={lr:.2e} loss={loss.item():.4f} token={token_loss.item():.4f} "
                        f"region={region_loss.item():.4f} flow={flow_loss.item():.4f} "
                        f"cons={consistency_loss.item():.4f} dest={dest_loss.item():.4f} "
                        f"vq={vq_loss.detach().item():.4f} macro={macro_loss.item():.4f} "
                        f"mask_ratio={mask_ratio:.3f} "
                        f"region_mask_ratio={region_mask_ratio:.3f}"
                    )

                if step > 0 and step % args.eval_interval == 0:
                    metrics = evaluate(model, tokenizer, time_encoder, val_loader, args, context_index)
                    if macro_val_loader is not None and model.macro_head is not None:
                        metrics["macro_kl"] = evaluate_macro(
                            model,
                            time_encoder,
                            macro_val_loader,
                            args.device,
                            max_batches=args.macro_eval_batches,
                        )
                    print("eval", metrics)

                if step > 0 and step % args.save_interval == 0:
                    ckpt = {
                        "model": model.state_dict(),
                        "tokenizer": tokenizer.state_dict(),
                        "time_encoder": time_encoder.state_dict(),
                        "optimizer": optim.state_dict(),
                        "scheduler": scheduler.state_dict(),
                        "step": step,
                        "args": vars(args),
                    }
                    torch.save(ckpt, Path("checkpoints") / f"{args.ckpt_prefix}_step_{step}.pt")

                step += 1
                if step >= args.max_steps:
                    break
        if step >= args.max_steps:
            break

    final_ckpt = {
        "model": model.state_dict(),
        "tokenizer": tokenizer.state_dict(),
        "time_encoder": time_encoder.state_dict(),
        "optimizer": optim.state_dict(),
        "scheduler": scheduler.state_dict(),
        "step": step,
        "args": vars(args),
    }
    final_ckpt_path = Path("checkpoints") / f"{args.ckpt_prefix}_final_step_{step}.pt"
    torch.save(final_ckpt, final_ckpt_path)
    print(f"saved_final_checkpoint={final_ckpt_path}")

    val_metrics = evaluate(model, tokenizer, time_encoder, val_loader, args, context_index)
    if macro_val_loader is not None and model.macro_head is not None:
        val_metrics["macro_kl"] = evaluate_macro(
            model,
            time_encoder,
            macro_val_loader,
            args.device,
            max_batches=args.macro_eval_batches,
        )
    test_metrics = evaluate(model, tokenizer, time_encoder, test_loader, args, context_index)
    if macro_test_loader is not None and model.macro_head is not None:
        test_metrics["macro_kl"] = evaluate_macro(
            model,
            time_encoder,
            macro_test_loader,
            args.device,
            max_batches=args.macro_eval_batches,
        )
    print("final_val", val_metrics)
    print("final_test", test_metrics)

    if args.results_path:
        payload = {
            "args": vars(args),
            "final_val": val_metrics,
            "final_test": test_metrics,
        }
        out_path = Path(args.results_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"saved_results={out_path}")


if __name__ == "__main__":
    main()

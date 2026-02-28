import math
import os
import random
import zipfile
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, IterableDataset
from torch.utils.data import get_worker_info

try:
    from datasets import load_dataset
except Exception:  # optional
    load_dataset = None

try:
    from huggingface_hub import hf_hub_download
except Exception:  # optional
    hf_hub_download = None

MIN_POINTS = 36
MAX_POINTS = 600
MIN_SAMPLING_RATIO = 0.35


def logarithmic_sampling_ratio(length, min_points=MIN_POINTS, max_points=MAX_POINTS, min_ratio=MIN_SAMPLING_RATIO):
    if length <= min_points:
        return 1.0
    if length >= max_points:
        return min_ratio
    ratio = 1.0 - math.log(length - min_points + 1) / math.log(max_points - min_points + 1) * (1.0 - min_ratio)
    return max(ratio, min_ratio)


def safe_parse_times(times: List) -> np.ndarray:
    if len(times) == 0:
        return np.zeros((0,), dtype=np.float32)
    # Fast paths for already-numeric and datetime-like arrays/series.
    if isinstance(times, np.ndarray):
        arr = times
    else:
        arr = np.asarray(times)

    if np.issubdtype(arr.dtype, np.number):
        return arr.astype(np.float32, copy=False)

    if np.issubdtype(arr.dtype, np.datetime64):
        ts = arr.astype("datetime64[ns]").astype(np.int64) / 1e9
        return ts.astype(np.float32)

    # Vectorized datetime parsing for object/string arrays (much faster than per-row parsing).
    try:
        import pandas as pd

        dt = pd.to_datetime(arr, errors="coerce", utc=True, format="mixed")
        ts = dt.view("int64").to_numpy(dtype=np.float64) / 1e9
        invalid = ~np.isfinite(ts)
        if invalid.any():
            # Preserve sequence order for invalid entries.
            ts[invalid] = np.arange(ts.shape[0], dtype=np.float64)[invalid]
        return ts.astype(np.float32)
    except Exception:
        parsed = []
        for i, t in enumerate(arr):
            if isinstance(t, (int, float, np.integer, np.floating)):
                parsed.append(float(t))
                continue
            try:
                parsed.append(datetime.fromisoformat(str(t)).timestamp())
            except Exception:
                parsed.append(float(i))
        return np.asarray(parsed, dtype=np.float32)


def normalize_lon_lat(points: List) -> Tuple[np.ndarray, np.ndarray]:
    # points are [lat, lon] or [lon, lat]. Heuristic on range.
    arr = np.asarray(points, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[1] < 2:
        raise ValueError("trajectory must be a list of [lat, lon] pairs")
    lat = arr[:, 0]
    lon = arr[:, 1]
    if np.any(np.abs(lat) > 90) and np.any(np.abs(lon) <= 90):
        lat, lon = lon, lat
    return lat, lon


class TrajectoryProcessor:
    def __init__(self, max_len: int, mask_ratio: float):
        self.max_len = max_len
        self.mask_ratio = mask_ratio
        self.sampling_ratios = [logarithmic_sampling_ratio(length) for length in range(MIN_POINTS, MAX_POINTS + 1)]

    def resample(self, lat: np.ndarray, lon: np.ndarray, times: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        length = len(lat)
        if length == 0:
            return lat, lon, times

        if random.random() < 0.3 and length >= 360:
            if length > 540:
                interval = random.randint(8, 15)
            elif length > 360:
                interval = random.randint(6, 10)
            else:
                interval = random.randint(3, 6)
            # uniform stride sampling by time
            base = times[0]
            bins = ((times - base) // interval).astype(np.int64)
            _, idx = np.unique(bins, return_index=True)
            idx = np.sort(idx)
        else:
            if length <= MIN_POINTS:
                ratio = 1.0
            elif length >= MAX_POINTS:
                ratio = MIN_SAMPLING_RATIO
            else:
                ratio = self.sampling_ratios[length - MIN_POINTS]
            num = max(2, int(length * ratio))
            idx = np.random.choice(np.arange(length), size=num, replace=False)
            idx = np.sort(idx)
        lat = lat[idx]
        lon = lon[idx]
        times = times[idx]
        return lat, lon, times

    def pad_or_truncate(self, arr: np.ndarray, fill: float = 0.0) -> Tuple[np.ndarray, np.ndarray]:
        length = len(arr)
        if length >= self.max_len:
            return arr[: self.max_len], np.ones((self.max_len,), dtype=np.float32)
        padded = np.full((self.max_len,) + arr.shape[1:], fill, dtype=arr.dtype)
        padded[:length] = arr
        mask = np.zeros((self.max_len,), dtype=np.float32)
        mask[:length] = 1.0
        return padded, mask

    def make_mask_indices(self, length: int) -> np.ndarray:
        length = min(length, self.max_len)
        num = max(1, int(length * self.mask_ratio))
        return np.random.choice(np.arange(length), size=num, replace=False)


class WorldTraceMapDataset(Dataset):
    def __init__(self, samples: List[dict], max_len: int, mask_ratio: float):
        prepared = []
        for sample in samples:
            rec = prepare_record(sample)
            if rec is not None:
                prepared.append(rec)
        self.samples = prepared
        self.proc = TrajectoryProcessor(max_len, mask_ratio)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        rec = self.samples[idx]
        return materialize_sample(rec["lat"], rec["lon"], rec["times"], self.proc)


class WorldTraceIterableDataset(IterableDataset):
    def __init__(self, hf_name: str, split: str, max_len: int, mask_ratio: float, shuffle_buffer: int = 1000, take: Optional[int] = None):
        if load_dataset is None:
            raise RuntimeError(
                "HF streaming requires `datasets` + `pyarrow`. Install both or load system Arrow (e.g., `module load arrow/21.0.0`)."
            )
        self.dataset = load_dataset(hf_name, split=split, streaming=True)
        self.dataset = self.dataset.shuffle(buffer_size=shuffle_buffer, seed=42)
        self.take = take
        self.proc = TrajectoryProcessor(max_len, mask_ratio)

    def __iter__(self):
        count = 0
        for record in self.dataset:
            try:
                sample = process_record(record, self.proc)
            except Exception:
                continue
            if sample is None:
                continue
            yield sample
            count += 1
            if self.take and count >= self.take:
                break


def _pick_column(columns_lower: Dict[str, str], names: List[str]) -> Optional[str]:
    for name in names:
        if name in columns_lower:
            return columns_lower[name]
    return None


class WorldTraceZipIterableDataset(IterableDataset):
    def __init__(
        self,
        hf_name: str,
        filename: str,
        max_len: int,
        mask_ratio: float,
        shuffle_buffer: int = 1000,
        take: Optional[int] = None,
        local_path: str = "",
        seed: int = 42,
    ):
        self.hf_name = hf_name
        self.filename = filename
        self.take = take
        self.shuffle_buffer = max(1, int(shuffle_buffer))
        self.seed = seed
        self.proc = TrajectoryProcessor(max_len, mask_ratio)
        self.source_path = self._resolve_source(local_path)

    def _resolve_source(self, local_path: str) -> str:
        if local_path:
            if not os.path.exists(local_path):
                raise FileNotFoundError(f"WorldTrace source not found: {local_path}")
            if os.path.isdir(local_path):
                candidate = os.path.join(local_path, self.filename)
                if os.path.exists(candidate):
                    return candidate
                raise FileNotFoundError(
                    f"WorldTrace source directory does not contain {self.filename}: {local_path}"
                )
            return local_path
        if hf_hub_download is None:
            raise RuntimeError("huggingface_hub is required to download WorldTrace zip/csv files")
        cache_dir = os.environ.get("HUGGINGFACE_HUB_CACHE")
        if not cache_dir:
            hf_home = os.environ.get("HF_HOME")
            if hf_home:
                cache_dir = os.path.join(hf_home, "hub")
        return hf_hub_download(
            repo_id=self.hf_name,
            filename=self.filename,
            repo_type="dataset",
            cache_dir=cache_dir,
        )

    def _record_from_csv_handle(self, csv_handle) -> Optional[dict]:
        import pandas as pd

        keep_cols = {"time", "timestamp", "datetime", "latitude", "longitude", "lat", "lon"}
        df = pd.read_csv(csv_handle, usecols=lambda c: str(c).strip().lower() in keep_cols)
        if len(df) < 2:
            return None

        cols = {str(c).strip().lower(): c for c in df.columns}
        lat_col = _pick_column(cols, ["latitude", "lat"])
        lon_col = _pick_column(cols, ["longitude", "lon"])
        if lat_col is None or lon_col is None:
            return None
        time_col = _pick_column(cols, ["time", "timestamp", "datetime"])

        lat = df[lat_col].to_numpy(dtype=np.float32)
        lon = df[lon_col].to_numpy(dtype=np.float32)
        if time_col is None:
            times = np.arange(len(df), dtype=np.float32)
        else:
            times = df[time_col].tolist()
        return {"trajectory": np.stack([lat, lon], axis=-1).tolist(), "time": times}

    def _iter_raw_records(self):
        path = self.source_path
        if path.lower().endswith(".csv"):
            with open(path, "rb") as f:
                rec = self._record_from_csv_handle(f)
                if rec is not None:
                    yield rec
            return

        if not path.lower().endswith(".zip"):
            raise ValueError(f"Unsupported WorldTrace file: {path}. Expected .zip or .csv")

        with zipfile.ZipFile(path, "r") as zf:
            for info in zf.infolist():
                if info.is_dir():
                    continue
                if not info.filename.lower().endswith(".csv"):
                    continue
                try:
                    with zf.open(info, "r") as f:
                        rec = self._record_from_csv_handle(f)
                except Exception:
                    continue
                if rec is not None:
                    yield rec

    def __iter__(self):
        worker = get_worker_info()
        worker_id = worker.id if worker is not None else 0
        rng = random.Random(self.seed + worker_id)

        emitted = 0
        buffer = []

        for record in self._iter_raw_records():
            if self.take is not None and emitted >= self.take:
                break
            try:
                sample = process_record(record, self.proc)
            except Exception:
                continue
            if sample is None:
                continue
            if self.shuffle_buffer <= 1:
                yield sample
                emitted += 1
                continue

            if len(buffer) < self.shuffle_buffer:
                buffer.append(sample)
                continue

            idx = rng.randrange(len(buffer))
            out = buffer[idx]
            buffer[idx] = sample
            yield out
            emitted += 1

        while buffer and (self.take is None or emitted < self.take):
            idx = rng.randrange(len(buffer))
            out = buffer[idx]
            buffer[idx] = buffer[-1]
            buffer.pop()
            yield out
            emitted += 1


def process_record(record: dict, proc: TrajectoryProcessor):
    prepared = prepare_record(record)
    if prepared is None:
        return None
    return materialize_sample(prepared["lat"], prepared["lon"], prepared["times"], proc)


def prepare_record(record: dict):
    traj = record.get("trajectory")
    if traj is None:
        traj = record.get("traj")
    if traj is None:
        traj = record.get("points")
    times = record.get("time")
    if times is None:
        times = record.get("times")
    if times is None:
        times = record.get("timestamp")
    if traj is None:
        raise KeyError("record missing trajectory field")

    lat, lon = normalize_lon_lat(traj)
    valid = np.isfinite(lat) & np.isfinite(lon)
    if valid.sum() < 2:
        return None
    lat = lat[valid]
    lon = lon[valid]
    if times is None:
        times = list(range(len(lat)))
    times = safe_parse_times(times)
    if len(times) != len(valid):
        n = min(len(times), len(lat))
        lat = lat[:n]
        lon = lon[:n]
        times = times[:n]
    else:
        times = times[valid]
    if len(lat) < 2:
        return None
    return {
        "lat": lat.astype(np.float32, copy=False),
        "lon": lon.astype(np.float32, copy=False),
        "times": times.astype(np.float32, copy=False),
    }


def materialize_sample(lat: np.ndarray, lon: np.ndarray, times: np.ndarray, proc: TrajectoryProcessor):
    lat, lon, times = proc.resample(lat, lon, times)
    coords = np.stack([lat, lon], axis=-1).astype(np.float32)
    intervals = np.zeros_like(times, dtype=np.float32)
    if len(times) > 1:
        intervals[1:] = times[1:] - times[:-1]
    coords, attn_mask = proc.pad_or_truncate(coords)
    intervals, _ = proc.pad_or_truncate(intervals)
    times, _ = proc.pad_or_truncate(times)
    mask_indices = proc.make_mask_indices(int(attn_mask.sum()))
    sample = {
        "coords": torch.from_numpy(coords),
        "intervals": torch.from_numpy(intervals),
        "timestamps": torch.from_numpy(times),
        "attention_mask": torch.from_numpy(attn_mask),
        "mask_indices": torch.from_numpy(mask_indices).long(),
    }
    return sample


def collate_batch(batch: List[dict]) -> dict:
    coords = torch.stack([b["coords"] for b in batch], dim=0)
    intervals = torch.stack([b["intervals"] for b in batch], dim=0)
    timestamps = torch.stack([b["timestamps"] for b in batch], dim=0)
    attention = torch.stack([b["attention_mask"] for b in batch], dim=0)
    # pad mask indices to max length in batch
    max_m = max(b["mask_indices"].shape[0] for b in batch)
    mask_idx = torch.full((len(batch), max_m), -1, dtype=torch.long)
    for i, b in enumerate(batch):
        m = b["mask_indices"]
        mask_idx[i, : m.shape[0]] = m
    return {
        "coords": coords,
        "intervals": intervals,
        "timestamps": timestamps,
        "attention_mask": attention,
        "mask_indices": mask_idx,
    }

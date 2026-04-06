"""Dataset helpers for the HMT training entrypoint."""

from __future__ import annotations

import io
import zipfile
from pathlib import Path
from typing import BinaryIO, Iterable

import numpy as np
import torch
from torch.utils.data import Dataset, IterableDataset


def _load_records(path: str | Path | BinaryIO) -> list[dict]:
    import pandas as pd

    if isinstance(path, (str, Path)):
        data_path = Path(path)
        if data_path.suffix == ".parquet":
            frame = pd.read_parquet(data_path)
        else:
            frame = pd.read_pickle(data_path)
    else:
        frame = pd.read_pickle(path)
    return frame.to_dict(orient="records")


def _parse_times(times: object, length: int) -> np.ndarray:
    if times is None:
        return np.arange(length, dtype=np.float32)

    arr = np.asarray(times)
    if arr.size == 0:
        return np.arange(length, dtype=np.float32)
    if np.issubdtype(arr.dtype, np.number):
        return arr.astype(np.float32, copy=False)
    if np.issubdtype(arr.dtype, np.datetime64):
        return (arr.astype("datetime64[ns]").astype(np.int64) / 1e9).astype(np.float32)

    try:
        import pandas as pd

        parsed = pd.to_datetime(arr, errors="coerce", utc=True, format="mixed")
        ts = parsed.view("int64").to_numpy(dtype=np.float64) / 1e9
        invalid = ~np.isfinite(ts)
        if invalid.any():
            ts[invalid] = np.arange(ts.shape[0], dtype=np.float64)[invalid]
        return ts.astype(np.float32)
    except Exception:
        return np.arange(length, dtype=np.float32)


def _normalize_lon_lat(points: object) -> tuple[np.ndarray, np.ndarray]:
    arr = np.asarray(points, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[1] < 2:
        raise ValueError("trajectory must be a 2D array of coordinate pairs")

    lat_lon = arr[:, :2]
    lon_lat = arr[:, [1, 0]]

    def _valid(coords: np.ndarray) -> bool:
        lat = coords[:, 0]
        lon = coords[:, 1]
        finite = np.isfinite(lat) & np.isfinite(lon)
        if finite.sum() == 0:
            return False
        return (
            float(np.max(np.abs(lat[finite]))) <= 90.0
            and float(np.max(np.abs(lon[finite]))) <= 180.0
        )

    if _valid(lat_lon) and not _valid(lon_lat):
        return lat_lon[:, 0], lat_lon[:, 1]
    if _valid(lon_lat) and not _valid(lat_lon):
        return lon_lat[:, 0], lon_lat[:, 1]
    return lat_lon[:, 0], lat_lon[:, 1]


def _downsample_indices(length: int, max_len: int) -> np.ndarray:
    if length <= max_len:
        return np.arange(length, dtype=np.int64)
    idx = np.linspace(0, length - 1, num=max_len, endpoint=True)
    idx = np.round(idx).astype(np.int64)
    idx = np.clip(idx, 0, length - 1)
    return np.maximum.accumulate(idx)[:max_len]


def _record_to_sample(record: dict, max_len: int) -> dict | None:
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

    lat, lon = _normalize_lon_lat(traj)
    valid = np.isfinite(lat) & np.isfinite(lon)
    if valid.sum() < 2:
        return None

    lat = lat[valid]
    lon = lon[valid]
    ts = _parse_times(times, len(valid))
    if ts.shape[0] == valid.shape[0]:
        ts = ts[valid]
    else:
        n = min(ts.shape[0], lat.shape[0])
        lat = lat[:n]
        lon = lon[:n]
        ts = ts[:n]
    if lat.shape[0] < 2:
        return None

    idx = _downsample_indices(lat.shape[0], max_len)
    lat = lat[idx]
    lon = lon[idx]
    ts = ts[idx]

    coords = np.zeros((max_len, 2), dtype=np.float32)
    timestamps = np.zeros((max_len,), dtype=np.float32)
    attention = np.zeros((max_len,), dtype=np.float32)
    length = lat.shape[0]
    coords[:length, 0] = lat
    coords[:length, 1] = lon
    timestamps[:length] = ts
    attention[:length] = 1.0

    return {
        "coords": torch.from_numpy(coords),
        "timestamps": torch.from_numpy(timestamps),
        "attention_mask": torch.from_numpy(attention),
    }


class WorldTraceMapDataset(Dataset):
    """Map-style WorldTrace dataset built from in-memory records."""

    def __init__(
        self,
        records: Iterable[dict],
        max_len: int = 200,
        mask_ratio: float = 0.3,
    ) -> None:
        del mask_ratio
        self.samples = []
        for record in records:
            sample = _record_to_sample(record, max_len=max_len)
            if sample is not None:
                self.samples.append(sample)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict:
        return self.samples[index]


class WorldTraceIterableDataset(IterableDataset):
    """Streaming-compatible WorldTrace dataset.

    This compatibility implementation currently supports local files through the
    ``local_path`` argument. Remote Hugging Face streaming is intentionally not
    reimplemented here.
    """

    def __init__(
        self,
        hf_name: str,
        split: str,
        max_len: int = 200,
        mask_ratio: float = 0.3,
        shuffle_buffer: int = 1000,
        take: int | None = None,
        local_path: str = "",
        seed: int = 42,
    ) -> None:
        del hf_name, split, mask_ratio, shuffle_buffer, seed
        self.max_len = max_len
        self.take = take
        self.local_path = local_path

    def __iter__(self):
        if not self.local_path:
            raise RuntimeError(
                "WorldTraceIterableDataset requires local_path in this repository checkout."
            )
        count = 0
        for record in _load_records(self.local_path):
            sample = _record_to_sample(record, max_len=self.max_len)
            if sample is None:
                continue
            yield sample
            count += 1
            if self.take is not None and self.take > 0 and count >= self.take:
                break


class WorldTraceZipIterableDataset(IterableDataset):
    """Read WorldTrace records from a local zip archive when available."""

    def __init__(
        self,
        hf_name: str,
        filename: str,
        max_len: int = 200,
        mask_ratio: float = 0.3,
        shuffle_buffer: int = 1000,
        take: int | None = None,
        local_path: str = "",
        seed: int = 42,
    ) -> None:
        del hf_name, filename, mask_ratio, shuffle_buffer, seed
        self.max_len = max_len
        self.take = take
        self.local_path = local_path

    def __iter__(self):
        if not self.local_path:
            raise RuntimeError(
                "WorldTraceZipIterableDataset requires worldtrace_local_path in this checkout."
            )
        zip_path = Path(self.local_path)
        if not zip_path.exists():
            raise FileNotFoundError(f"WorldTrace zip not found: {zip_path}")

        count = 0
        with zipfile.ZipFile(zip_path) as archive:
            members = [
                name
                for name in archive.namelist()
                if name.endswith(".pkl") or name.endswith(".parquet")
            ]
            if not members:
                raise RuntimeError(f"No pickle/parquet files found in {zip_path}")
            member = members[0]
            with archive.open(member) as handle:
                payload = io.BytesIO(handle.read())
            for record in _load_records(payload):
                sample = _record_to_sample(record, max_len=self.max_len)
                if sample is None:
                    continue
                yield sample
                count += 1
                if self.take is not None and self.take > 0 and count >= self.take:
                    break


def collate_batch(batch: list[dict]) -> dict:
    """Collate WorldTrace samples for training and evaluation."""

    return {
        "coords": torch.stack([item["coords"] for item in batch], dim=0),
        "timestamps": torch.stack([item["timestamps"] for item in batch], dim=0),
        "attention_mask": torch.stack(
            [item["attention_mask"] for item in batch],
            dim=0,
        ),
    }

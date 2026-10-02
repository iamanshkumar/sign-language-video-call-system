"""PyTorch Dataset / DataLoaders over the sequence index (lazy, no MediaPipe)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from asl.landmarks.layout import assemble_features, feature_dim
from asl.sequences.resample import resample_interval

_AUX = ["flags", "pose_vis"]


def load_sequence(path: Path, groups: list[str], start: int, end: int, window: int) -> np.ndarray:
    """[window, F] float32 for one stored clip; reads only the needed groups."""
    with np.load(path) as z:
        arrays = {k: z[k] for k in groups + [k for k in _AUX if k in z.files]}
    seq = assemble_features(resample_interval(arrays, groups, start, end, window), groups)
    assert seq.shape == (window, feature_dim(groups)) and np.isfinite(seq).all()
    return seq


class SequenceDataset(Dataset):
    """Sequences of one split, one window size and one landmark configuration.

    __getitem__ -> (float32 tensor [window, F], int64 label). `metadata(i)` gives the
    traceable source of item i.
    """

    def __init__(self, index: pd.DataFrame, arrays_dir: Path, split: str, window: int, configuration: str,
                 configurations: dict[str, list[str]], label_mapping: dict, preload: bool = False) -> None:
        if split not in ("train", "val", "test"):
            raise ValueError(split)
        rows = index[index["split"] == split].sort_values("sample_id", ignore_index=True)
        mapped = rows["word"].map(label_mapping["word_to_index"])
        if mapped.isna().any() or not (mapped.astype("int64") == rows["label"]).all():
            raise ValueError("index labels do not match the label mapping")
        self.rows, self.arrays_dir = rows, Path(arrays_dir)
        self.split, self.window, self.configuration = split, window, configuration
        self.groups = configurations[configuration]
        self.feature_dim = feature_dim(self.groups)
        self.num_classes = label_mapping["num_classes"]
        self._cache = [self._load(i) for i in range(len(rows))] if preload else None

    def _load(self, i: int) -> np.ndarray:
        r = self.rows.iloc[i]
        return load_sequence(self.arrays_dir / r.array_file, self.groups, int(r.active_start_frame),
                             int(r.active_end_frame), self.window)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor]:
        seq = self._cache[i] if self._cache is not None else self._load(i)
        return torch.from_numpy(seq), torch.tensor(int(self.rows.iloc[i].label), dtype=torch.long)

    def metadata(self, i: int) -> dict:
        r = self.rows.iloc[i]
        return {"sample_id": r.sample_id, "repo_path": r.repo_path, "word": r.word, "label": int(r.label),
                "split": r.split, "window": self.window, "configuration": self.configuration,
                "array_file": r.array_file, "active_interval": (int(r.active_start_frame), int(r.active_end_frame)),
                "active_frames": int(r.active_frames), "stretch": self.window / int(r.active_frames)}


def _seed_worker(worker_id: int) -> None:
    seed = torch.initial_seed() % 2**32
    np.random.seed(seed)


def make_dataloaders(datasets: dict[str, SequenceDataset], batch_size: int, num_workers: int, seed: int,
                     drop_last_train: bool = False) -> dict[str, DataLoader]:
    """Train: shuffled with a seeded generator. Val/test: fixed order, never shuffled."""
    loaders = {}
    for split, ds in datasets.items():
        train = split == "train"
        g = torch.Generator()
        g.manual_seed(seed)
        loaders[split] = DataLoader(ds, batch_size=batch_size, shuffle=train, drop_last=train and drop_last_train,
                                    num_workers=num_workers, generator=g if train else None,
                                    worker_init_fn=_seed_worker if num_workers else None)
    return loaders

"""Small synthetic landmark stores for smoke tests (no dataset, no MediaPipe).

Each class has a distinct hand trajectory frequency plus noise, so a model can learn it.
Files are written through the real v2 `pack_arrays` / `save_arrays`, and the index through
the real `build_index`, so the whole Step 2 -> Step 3 path is exercised. Active intervals
vary from 8 to 80 frames, so short intervals (< 30) are included.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from asl.landmarks.layout import N_LANDMARKS, SLICES
from asl.landmarks.preprocess import ActiveInterval
from asl.landmarks.process import array_name, pack_arrays, save_arrays
from asl.sequences.index import build_index


def make_synthetic_store(root: Path, words: list[str], mapping: dict, out_cfg: dict, windows: list[int],
                         per_class: dict[str, int] | None = None, seed: int = 0) -> tuple[pd.DataFrame, Path]:
    per_class = per_class or {"train": 6, "val": 2, "test": 2}
    rng = np.random.default_rng(seed)
    arrays_dir = root / "arrays"
    arrays_dir.mkdir(parents=True, exist_ok=True)
    meta, frozen = [], []
    for c, word in enumerate(words):
        for split, n in per_class.items():
            for k in range(n):
                sid = f"synthetic/{word}_{split}_{k}.mp4"
                length = int(rng.integers(8, 81))
                rest_before, rest_after = int(rng.integers(0, 15)), int(rng.integers(0, 15))
                t = rest_before + length + rest_after
                start, end = rest_before, rest_before + length - 1
                tt = np.linspace(0, 1, length)
                norm = rng.normal(0, 0.02, (t, N_LANDMARKS, 3))
                for g, amp in (("left_hand", 1.0), ("right_hand", 0.6), ("pose", 0.2), ("face", 0.05)):
                    sig = amp * np.sin(2 * np.pi * (1 + c) * tt + 0.3 * c)
                    norm[start:end + 1, SLICES[g], 0] += sig[:, None]
                    norm[start:end + 1, SLICES[g], 1] += amp * c / len(words)
                present = np.ones((t, 4), bool)
                present[:start, 2:] = False  # hands at rest outside the interval
                present[end + 1:, 2:] = False
                norm[~present[:, 2], SLICES["left_hand"]] = 0
                norm[~present[:, 3], SLICES["right_hand"]] = 0
                arrays = pack_arrays(norm, present, present.copy(), np.zeros_like(present), np.ones((t, 33)),
                                     np.zeros((t, 3)), np.ones(t), np.ones(t, bool), ActiveInterval(start, end), out_cfg)
                name = array_name(sid)
                save_arrays(arrays, arrays_dir / name)
                meta.append({"sample_id": sid, "repo_path": sid, "word": word, "split": split, "rejected": False,
                             "rejection_reason": None, "array_file": name, "active_start_frame": start,
                             "active_end_frame": end, "std_frame_count": t})
                frozen.append({"sample_id": sid, "repo_path": sid, "word": word, "split": split, "sha256": sid,
                               "dup_group_size": "1", "split_group": sid, "aug_source": "", "source_proxy": "synthetic"})
    idx, _ = build_index(pd.DataFrame(meta), pd.DataFrame(frozen), mapping, arrays_dir, windows, 30)
    return idx, arrays_dir

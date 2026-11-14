"""Shared helpers for the E0 evaluation scripts (docs/evaluation_protocol.md). OFFLINE ONLY.

Read-only use of experiment runs, the dev100 store and saved webcam sessions; nothing here is used by
training or by the inference pipeline.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from asl.config import PROJECT_ROOT
from asl.landmarks.layout import GROUP_INDEX, OBSERVED, PRESENT, assemble_features
from asl.models.recurrent import build_model
from asl.sequences.resample import resample_interval

EXP = PROJECT_ROOT / "experiments" / "dev100_hp"
SEEDS = [42, 1, 2, 3, 4]
GROUPS = ["left_hand", "right_hand", "pose"]          # hands_pose
WINDOW = 30
N_BOOT, BOOT_SEED = 2000, 0


def run_dirs(model: str) -> dict[int, Path]:
    """The 5 seed runs of the reference comparison (seed 42 has no suffix)."""
    base = f"hands_pose_w30_{model}_e100"
    return {s: EXP / (base if s == 42 else f"{base}_s{s}") for s in SEEDS}


def load_run(run_dir: Path) -> tuple[torch.nn.Module, dict]:
    """Rebuild a finished run's selected checkpoint (best.pt) in eval mode."""
    meta = json.loads((run_dir / "metadata.json").read_text())
    if meta.get("checkpoint") != "best.pt" or meta["feature_config"] != "hands_pose" or meta["window_size"] != WINDOW:
        raise ValueError(f"{run_dir}: not a hands_pose w30 run selected on validation")
    ck = torch.load(run_dir / "best.pt", map_location="cpu", weights_only=True)
    model = build_model(meta["model"], meta["feature_dim"], meta["vocabulary_size"], meta["neural_config"])
    model.load_state_dict(ck["state_dict"])
    return model.eval(), meta


@torch.no_grad()
def logits(model: torch.nn.Module, x: np.ndarray, batch: int = 256) -> np.ndarray:
    out = [model(torch.from_numpy(np.ascontiguousarray(x[i:i + batch], np.float32))).numpy()
           for i in range(0, len(x), batch)]
    return np.concatenate(out) if out else np.zeros((0, 0), np.float32)


# ---------------------------------------------------------------- metrics and bootstrap
def macro_f1(y: np.ndarray, p: np.ndarray) -> float:
    """Macro-F1 over the union of true and predicted labels (same definition as
    asl.models.metrics.classification_metrics)."""
    labels = np.union1d(y, p)
    tp = np.array([np.sum((y == c) & (p == c)) for c in labels], float)
    fp = np.array([np.sum((y != c) & (p == c)) for c in labels], float)
    fn = np.array([np.sum((y == c) & (p != c)) for c in labels], float)
    den = 2 * tp + fp + fn
    return float(np.mean(np.where(den > 0, 2 * tp / np.maximum(den, 1), 0.0)))


def boot_indices(n: int, n_boot: int = N_BOOT, seed: int = BOOT_SEED, strata: np.ndarray | None = None) -> np.ndarray:
    """Bootstrap index sets [n_boot, n]. With `strata` (e.g. the true class of each clip) clips are resampled
    within each stratum, so per-class supports stay fixed (unstratified resampling biases macro-F1 downward
    when classes have ~7 clips)."""
    rng = np.random.default_rng(seed)
    if strata is None:
        return rng.integers(0, n, size=(n_boot, n))
    strata = np.asarray(strata)
    out = np.empty((n_boot, n), np.int64)
    pos = 0
    for c in np.unique(strata):
        members = np.where(strata == c)[0]
        out[:, pos:pos + len(members)] = members[rng.integers(0, len(members), size=(n_boot, len(members)))]
        pos += len(members)
    return out


def ci(values: np.ndarray) -> tuple[float, float]:
    lo, hi = np.percentile(values, [2.5, 97.5])
    return float(lo), float(hi)


# ---------------------------------------------------------------- predeclared perturbations (protocol §6)
POSE_PAIRS = [(1, 4), (2, 5), (3, 6), (7, 8), (9, 10), (11, 12), (13, 14), (15, 16), (17, 18), (19, 20),
              (21, 22), (23, 24), (25, 26), (27, 28), (29, 30), (31, 32)]
PERTURBATIONS = ["clean", "hips_hidden", "hips_visible", "hold_start", "hold_end", "repeat_x2",
                 "drop_minor_hand", "hand_dropout_20", "mirror", "rotate_10", "noise_002"]


def _interval(arrays: dict, s: int, e: int) -> dict:
    return {k: np.array(arrays[k][s:e + 1]) for k in GROUPS + ["flags", "pose_vis"]}


def _drop_hand(a: dict, g: str, frames: np.ndarray) -> None:
    a[g][frames] = 0.0
    a["flags"][frames, GROUP_INDEX[g]] = 0


def perturb(arrays, start: int, end: int, name: str, seed: int = 0) -> dict:
    """Stored arrays of one clip -> arrays of its active interval after one predeclared perturbation
    (all coordinates are shoulder-normalized, origin at the shoulder centre)."""
    a = _interval(arrays, start, end)
    n = len(a["flags"])
    rng = np.random.default_rng(seed)
    if name == "clean":
        pass
    elif name == "hips_hidden":
        a["pose_vis"][:, 23:25] = 0.0
    elif name == "hips_visible":
        a["pose_vis"][:, 23:25] = 0.95
    elif name in ("hold_start", "hold_end"):
        k = max(1, int(round(0.5 * n)))
        idx = np.r_[np.zeros(k, int), np.arange(n)] if name == "hold_start" else np.r_[np.arange(n), np.full(k, n - 1)]
        a = {key: v[idx] for key, v in a.items()}
    elif name == "repeat_x2":
        a = {key: np.concatenate([v, v]) for key, v in a.items()}
    elif name == "drop_minor_hand":
        obs = {g: ((a["flags"][:, GROUP_INDEX[g]] & OBSERVED) != 0).sum() for g in ("left_hand", "right_hand")}
        minor = "left_hand" if obs["left_hand"] < obs["right_hand"] else "right_hand"
        _drop_hand(a, minor, np.arange(n))
    elif name == "hand_dropout_20":
        for g in ("left_hand", "right_hand"):
            _drop_hand(a, g, np.where(rng.random(n) < 0.2)[0])
    elif name == "mirror":
        for g in GROUPS:
            a[g][..., 0] *= -1.0
        a["left_hand"], a["right_hand"] = a["right_hand"].copy(), a["left_hand"].copy()
        li, ri = GROUP_INDEX["left_hand"], GROUP_INDEX["right_hand"]
        a["flags"][:, [li, ri]] = a["flags"][:, [ri, li]]
        for i, j in POSE_PAIRS:
            a["pose"][:, [i, j]] = a["pose"][:, [j, i]]
            a["pose_vis"][:, [i, j]] = a["pose_vis"][:, [j, i]]
    elif name == "rotate_10":
        th = np.deg2rad(10.0)
        rot = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]], np.float32)
        for g in GROUPS:
            pres = (a["flags"][:, GROUP_INDEX[g]] & PRESENT) != 0
            a[g][pres, :, :2] = a[g][pres, :, :2] @ rot.T
    elif name == "noise_002":
        for g in GROUPS:
            pres = (a["flags"][:, GROUP_INDEX[g]] & PRESENT) != 0
            a[g][pres] += rng.normal(0.0, 0.02, a[g][pres].shape).astype(np.float32)
    else:
        raise ValueError(name)
    return a


def features(a: dict) -> np.ndarray:
    """Perturbed interval arrays -> [30, 261] with the unchanged training resampling and assembly."""
    seq = assemble_features(resample_interval(a, GROUPS, 0, len(a["flags"]) - 1, WINDOW), GROUPS)
    assert seq.shape == (WINDOW, 261) and np.isfinite(seq).all()
    return seq

"""Training-only temporal augmentation of a stored clip's active interval (E1).

Applied to the stored (shoulder-normalized) arrays of the active interval before the unchanged uniform
resampling and feature assembly, so the model input format is identical. Never applied to validation,
test or live inputs. Each operation keeps the sign's own frames in their original order and never
changes the label; it only changes how long parts of the sign last:

- hold_start / hold_end: the first / last frame is held for 10-50 % of the interval length (a pause
  before or after signing, as seen in webcam segments);
- tempo: a monotone piecewise-linear time warp with local speed factors in [2/3, 3/2]; landmarks are
  interpolated between neighbouring frames exactly like the training resampling (never reordered);
- repeat_span: one extra cycle of a sub-span whose end returns to where it started: the splice from the
  span's last frame back to its first frame is no larger than min(`loop_tol` shoulder widths, the
  largest frame-to-frame step already in the clip), with the same hands present on both sides, so the
  repetition contains no jump; if no such sub-span exists the clip is left unchanged (counted).

With probability `p` one operation (uniform) is applied; otherwise the clip is unchanged.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from asl.landmarks.layout import GROUP_INDEX, OBSERVED

OPS = ("hold_start", "hold_end", "tempo", "repeat_span")
HANDS = ("left_hand", "right_hand")


@dataclass(frozen=True)
class TemporalAugmentConfig:
    p: float = 0.5
    hold_min: float = 0.1
    hold_max: float = 0.5
    tempo_min: float = 2 / 3
    tempo_max: float = 1.5
    tempo_knots: int = 2
    loop_tol: float = 0.15          # shoulder widths
    loop_min_frac: float = 0.25     # sub-span at least this share of the interval ...
    loop_min_frames: int = 6        # ... and at least this many frames

    def to_dict(self) -> dict:
        return asdict(self)


def _take(a: dict, idx: np.ndarray) -> dict:
    return {k: v[idx] for k, v in a.items()}


def _interp(a: dict, pos: np.ndarray) -> dict:
    """Arrays at fractional frame positions, like the training resampling: a group is linearly
    interpolated where it is present in both neighbouring frames, otherwise the nearest frame is used;
    flags come from the nearest frame."""
    from asl.landmarks.layout import PRESENT
    i = np.floor(pos).astype(int)
    j = np.minimum(i + 1, len(a["flags"]) - 1)
    w = (pos - i).astype(np.float32)
    near = np.where(w >= 0.5, j, i)
    out = {"flags": a["flags"][near].copy()}
    for k, v in a.items():
        if k == "flags":
            continue
        v = np.asarray(v)
        lin = v[i] * (1 - w).reshape(-1, *[1] * (v.ndim - 1)) + v[j] * w.reshape(-1, *[1] * (v.ndim - 1))
        if k in GROUP_INDEX:
            pres = (a["flags"][:, GROUP_INDEX[k]] & PRESENT) != 0
            both = (pres[i] & pres[j]).reshape(-1, *[1] * (v.ndim - 1))
            out[k] = np.where(both, lin, v[near]).astype(v.dtype)
        else:
            out[k] = lin.astype(np.float64 if v.dtype.kind in "iu" else v.dtype)
    return out


def _hand_centroids(a: dict) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    out = {}
    for g in HANDS:
        if g in a:
            obs = (a["flags"][:, GROUP_INDEX[g]] & OBSERVED) != 0
            out[g] = (obs, np.asarray(a[g], np.float64)[:, :, :2].mean(axis=1))
    return out


def _natural_max_step(hc: dict) -> float:
    """Largest centroid step between consecutive observed frames of any hand (bridging detection gaps)."""
    best = 0.0
    for obs, cen in hc.values():
        k = np.where(obs)[0]
        if len(k) > 1:
            best = max(best, float(np.linalg.norm(np.diff(cen[k], axis=0), axis=1).max()))
    return best


def find_loop(a: dict, cfg: TemporalAugmentConfig, rng: np.random.Generator) -> tuple[int, int] | None:
    """A random sub-span [i, j] (i < j) that can be played once more without a jump: the splice goes
    from frame j back to frame i + 1, so the same hands must be present at j and i + 1, and for every
    hand its last observed position up to j and its first observed position from i + 1 may differ by at
    most min(loop_tol, the largest step already in the clip). Or None."""
    n = len(a["flags"])
    min_len = max(cfg.loop_min_frames, int(np.ceil(cfg.loop_min_frac * n)))
    if n < min_len + 2:
        return None
    hc = _hand_centroids(a)
    if not hc:
        return None
    tol = min(cfg.loop_tol, _natural_max_step(hc))
    ii, jj = np.triu_indices(n - 1, k=min_len)          # i in [0, n-2], j >= i + min_len, j <= n-2 ...
    nxt = ii + 1                                        # ... the splice lands on i + 1
    ok = np.ones(len(ii), bool)
    any_hand = np.zeros(len(ii), bool)
    idx = np.arange(n)
    for obs, cen in hc.values():
        if not obs.any():
            continue
        last_upto = np.maximum.accumulate(np.where(obs, idx, -1))              # last observed frame <= k
        first_from = np.minimum.accumulate(np.where(obs, idx, n)[::-1])[::-1]  # first observed frame >= k
        a_, b_ = last_upto[jj], first_from[nxt]
        valid = (a_ >= 0) & (b_ < n)
        # the hand as seen just before the splice and just after it must be (nearly) in the same place
        d = np.linalg.norm(cen[np.clip(a_, 0, n - 1)] - cen[np.clip(b_, 0, n - 1)], axis=1)
        ok &= (obs[nxt] == obs[jj]) & (~valid | (d <= tol))
        any_hand |= obs[jj]
    cand = np.where(ok & any_hand)[0]
    if len(cand) == 0:
        return None
    k = cand[int(rng.integers(len(cand)))]
    return int(ii[k]), int(jj[k])


def apply_op(a: dict, op: str, cfg: TemporalAugmentConfig, rng: np.random.Generator) -> tuple[dict, bool]:
    """One operation on interval arrays (time on axis 0). Returns (arrays, changed)."""
    n = len(a["flags"])
    if op in ("hold_start", "hold_end"):
        k = max(1, int(round(rng.uniform(cfg.hold_min, cfg.hold_max) * n)))
        idx = np.r_[np.zeros(k, int), np.arange(n)] if op == "hold_start" else np.r_[np.arange(n), np.full(k, n - 1)]
        return _take(a, idx), True
    if op == "tempo":
        if n < 3:
            return a, False
        knots = np.sort(rng.uniform(0, n - 1, cfg.tempo_knots))
        edges = np.r_[0.0, knots, n - 1.0]
        speeds = rng.uniform(cfg.tempo_min, cfg.tempo_max, len(edges) - 1)
        pos = []
        for (lo, hi), sp in zip(zip(edges[:-1], edges[1:]), speeds):
            steps = max(1, int(np.ceil((hi - lo) / sp)))   # ceil: local speed never above sp
            pos.append(np.linspace(lo, hi, steps, endpoint=False))
        pos = np.clip(np.r_[np.concatenate(pos), n - 1.0], 0, n - 1)
        assert (np.diff(pos) >= 0).all()
        return _interp(a, pos), True
    if op == "repeat_span":
        loop = find_loop(a, cfg, rng)
        if loop is None:
            return a, False
        i, j = loop
        idx = np.r_[np.arange(0, j + 1), np.arange(i + 1, j + 1), np.arange(j + 1, n)]
        return _take(a, idx), True
    raise ValueError(op)


class TemporalAugment:
    """Callable used by the training Dataset: (interval arrays) -> (augmented interval arrays)."""

    def __init__(self, cfg: TemporalAugmentConfig, seed: int):
        self.cfg, self.seed, self.rng = cfg, seed, np.random.default_rng(seed)
        self.counts = {op: 0 for op in OPS} | {"none": 0, "repeat_span_no_loop": 0}

    def describe(self) -> dict:
        return {"type": "temporal", "seed": self.seed, **self.cfg.to_dict()}

    def __call__(self, a: dict) -> dict:
        if self.rng.random() >= self.cfg.p:
            self.counts["none"] += 1
            return a
        op = OPS[int(self.rng.integers(len(OPS)))]
        out, changed = apply_op(a, op, self.cfg, self.rng)
        self.counts[op if changed else f"{op}_no_loop" if op == "repeat_span" else "none"] += 1
        return out

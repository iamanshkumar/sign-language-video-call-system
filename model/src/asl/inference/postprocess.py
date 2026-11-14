"""Optional, configurable post-processing of model predictions. Defaults are pass-through.

Neither filter changes the model input or the model; they only decide which predictions are
reported. Both are off unless configured (threshold 0.0, smoothing window 1).
"""
from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass


@dataclass(frozen=True)
class Prediction:
    status: str                      # "ok" | "low_confidence" | "rejected"
    label: str | None                # class name (None when rejected)
    index: int | None
    confidence: float | None         # softmax probability of the top class
    top_k: tuple[tuple[str, float], ...] = ()
    reason: str = ""                 # rejection / filtering reason


class ConfidenceFilter:
    """Marks predictions whose top probability is below `threshold` as low_confidence."""

    def __init__(self, threshold: float = 0.0) -> None:
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("threshold must be in [0, 1]")
        self.threshold = threshold

    def __call__(self, p: Prediction) -> Prediction:
        if p.status != "ok" or p.confidence is None or p.confidence >= self.threshold:
            return p
        return Prediction("low_confidence", p.label, p.index, p.confidence, p.top_k,
                          f"confidence {p.confidence:.3f} < threshold {self.threshold:.3f}")


class Smoother:
    """Majority vote over the last `window` accepted predictions.

    `update` returns the label to emit, or None. A label is emitted when it holds at least
    `min_votes` of the last `window` accepted ("ok") predictions and differs from the label
    emitted last (no repeated emission of the same sign). window=1, min_votes=1 is pass-through:
    every accepted prediction is emitted, including repeats.
    """

    def __init__(self, window: int = 1, min_votes: int = 1) -> None:
        if window < 1 or not 1 <= min_votes <= window:
            raise ValueError("need window >= 1 and 1 <= min_votes <= window")
        self.window, self.min_votes = window, min_votes
        self.history: deque[str] = deque(maxlen=window)
        self.last_emitted: str | None = None

    @property
    def passthrough(self) -> bool:
        return self.window == 1 and self.min_votes == 1

    def update(self, p: Prediction) -> str | None:
        if p.status != "ok" or p.label is None:
            return None
        if self.passthrough:
            self.last_emitted = p.label
            return p.label
        self.history.append(p.label)
        label, votes = Counter(self.history).most_common(1)[0]
        if votes >= self.min_votes and label != self.last_emitted:
            self.last_emitted = label
            return label
        return None

    def reset(self) -> None:
        self.history.clear()
        self.last_emitted = None

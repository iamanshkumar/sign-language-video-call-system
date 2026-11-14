"""Isolated-sign recognition for live or recorded landmark streams.

The model input is built with exactly the training code path: per-frame MediaPipe Holistic
landmarks -> `preprocess_raw` (30 FPS resampling, active signing interval, rejection rules, gap
handling, shoulder normalization, storage dtypes) -> `resample_interval` (active interval -> W
frames) -> `assemble_features` (feature order of the configuration). Nothing here re-implements
preprocessing. One prediction is made per detected signing segment (isolated words, not
sentences).
"""
from __future__ import annotations

import json
import math
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from asl.inference.postprocess import ConfidenceFilter, Prediction, Smoother
from asl.inference.spec import preprocessing_spec, spec_differences
from asl.landmarks.extract import RawLandmarks, holistic_result_to_frame, make_holistic
from asl.landmarks.layout import GROUP_INDEX, GROUPS, N_LANDMARKS, SLICES, assemble_features
from asl.landmarks.process import preprocess_raw
from asl.landmarks.storage import apply_profile
from asl.models.recurrent import build_model
from asl.sequences.labels import vocabulary_hash
from asl.sequences.resample import resample_interval

REQUIRED_KEYS = ("state_dict", "model", "input_dim", "num_classes", "hidden_size", "num_layers", "dropout",
                 "window", "feature_config", "groups", "label_mapping_sha256", "preprocessing")
HANDS = [GROUP_INDEX["left_hand"], GROUP_INDEX["right_hand"]]


class ModelLoadError(RuntimeError):
    """The checkpoint or class mapping is missing, corrupt, or does not match this code/config."""


class InputError(ValueError):
    """An input has the wrong shape or contains invalid values."""


class SignRecognizer:
    """A loaded, validated model plus the exact preprocessing it was trained with."""

    def __init__(self, model: torch.nn.Module, info: dict, classes: list[str], step2_cfg: dict, top_k: int = 5):
        self.model, self.info, self.classes = model.eval(), info, classes
        self.window, self.feature_dim = int(info["window"]), int(info["input_dim"])
        self.groups = list(info["groups"])
        self.cfg = apply_profile(step2_cfg, info["preprocessing"]["storage_profile"])
        self.top_k = max(1, min(top_k, len(classes)))

    # ------------------------------------------------------------------ loading
    @classmethod
    def load(cls, model_path: Path, class_mapping_path: Path, step2_cfg: dict, step3_cfg: dict,
             top_k: int = 5) -> "SignRecognizer":
        model_path, class_mapping_path = Path(model_path), Path(class_mapping_path)
        for p in (model_path, class_mapping_path):
            if not p.is_file():
                raise ModelLoadError(f"not found: {p}")
        try:
            ck = torch.load(model_path, map_location="cpu", weights_only=True)
        except Exception as e:  # corrupt / not a torch file / unsafe content
            raise ModelLoadError(f"cannot read checkpoint {model_path}: {type(e).__name__}: {e}") from e
        missing = [k for k in REQUIRED_KEYS if k not in ck]
        if missing:
            raise ModelLoadError(f"{model_path} is not an exported model (missing {missing}); "
                                 "export it with scripts/s4_04_export_model.py")
        try:
            cm = json.loads(class_mapping_path.read_text(encoding="utf-8"))
            classes = [cm["index_to_class"][str(i)] for i in range(int(cm["num_classes"]))]
        except Exception as e:
            raise ModelLoadError(f"invalid class mapping {class_mapping_path}: {type(e).__name__}: {e}") from e
        if len(classes) != ck["num_classes"]:
            raise ModelLoadError(f"class mapping has {len(classes)} classes, model has {ck['num_classes']}")
        if vocabulary_hash(classes) != ck["label_mapping_sha256"]:
            raise ModelLoadError("class mapping does not match the label mapping the model was trained with")
        current = preprocessing_spec(step2_cfg, step3_cfg, ck["preprocessing"].get("storage_profile"),
                                     ck["feature_config"], int(ck["window"]))
        diff = spec_differences(ck["preprocessing"], current)
        if diff:
            raise ModelLoadError(f"preprocessing differs from training in {diff}; refusing to load")
        if current["feature_dim"] != ck["input_dim"] or current["groups"] != list(ck["groups"]):
            raise ModelLoadError("feature dimension / groups of the checkpoint do not match its preprocessing")
        try:
            model = build_model(ck["model"], int(ck["input_dim"]), int(ck["num_classes"]),
                                {"hidden_size": ck["hidden_size"], "num_layers": ck["num_layers"],
                                 "dropout": ck["dropout"]})
            model.load_state_dict(ck["state_dict"], strict=True)
        except Exception as e:
            raise ModelLoadError(f"weights do not fit the recorded architecture: {type(e).__name__}: {e}") from e
        info = {k: v for k, v in ck.items() if k != "state_dict"}
        return cls(model, info, classes, step2_cfg, top_k)

    # ------------------------------------------------------------------ model input -> prediction
    @torch.no_grad()
    def predict_features(self, x) -> Prediction:
        """x: [window, feature_dim] (one sequence, already built by the training pipeline)."""
        a = np.asarray(x)
        if a.ndim != 2 or a.shape != (self.window, self.feature_dim):
            raise InputError(f"expected input of shape ({self.window}, {self.feature_dim}), got {a.shape}")
        if not np.issubdtype(a.dtype, np.number):
            raise InputError(f"expected a numeric array, got dtype {a.dtype}")
        a = a.astype(np.float32, copy=False)
        if not np.isfinite(a).all():
            raise InputError("input contains NaN or Inf")
        prob = torch.softmax(self.model(torch.from_numpy(np.ascontiguousarray(a))[None]), dim=1)[0]
        conf, idx = torch.topk(prob, self.top_k)
        top = tuple((self.classes[int(i)], float(c)) for c, i in zip(conf, idx))
        return Prediction("ok", top[0][0], int(idx[0]), top[0][1], top)

    def features_from_raw(self, raw: RawLandmarks) -> tuple[np.ndarray | None, dict]:
        """Training preprocessing on raw per-frame landmarks -> ([window, F] or None, stats)."""
        arrays, st = preprocess_raw(raw, self.cfg)
        if arrays is None or st.get("rejection_reason"):
            return None, st
        s, e = (int(v) for v in arrays["active_interval"])
        seq = assemble_features(resample_interval(arrays, self.groups, s, e, self.window), self.groups)
        return seq.astype(np.float32), st

    def predict_raw(self, raw: RawLandmarks) -> Prediction:
        """One clip / segment of raw landmarks -> prediction, or status 'rejected' with the training
        rejection reason (e.g. too_few_decoded_frames, no_active_signing_interval)."""
        seq, st = self.features_from_raw(raw)
        if seq is None:
            return Prediction("rejected", None, None, None, (), st.get("rejection_reason") or "unknown")
        return self.predict_features(seq)


# ---------------------------------------------------------------- streaming
@dataclass(frozen=True)
class SignEvent:
    prediction: Prediction            # model output after the confidence filter
    emitted: str | None               # label to show (after smoothing), or None
    start_s: float
    end_s: float
    frames: int
    forced: bool                      # segment cut at max_segment_s instead of a rest
    last_hand_s: float | None = None  # stream time of the segment's last frame with a hand (end of signing)
    first_hand_s: float | None = None  # first frame of the hand run that opened the segment (start of signing)
    suppressed: bool = False           # within cooldown_s of the previous segment: classified, not emitted
    run_start_s: float | None = None   # first hand frame of the uninterrupted hand run: earlier than first_hand_s
                                       # only for a segment continuing one cut at max_segment_s


class StreamingRecognizer:
    """Segments a live frame stream into signing segments and classifies each one.

    A segment starts when a hand is seen in `min_hand_run` consecutive frames (the training
    definition of a meaningful detection) and includes up to `pre_roll_s` of earlier frames; it
    ends after `end_rest_s` without hands, or is cut at `max_segment_s`. Each segment is passed
    through the training preprocessing as one clip. Frames must arrive with increasing timestamps.
    """

    def __init__(self, recognizer: SignRecognizer, width: int, height: int, *, pre_roll_s: float = 0.5,
                 end_rest_s: float = 0.4, max_segment_s: float = 4.0, min_hand_run: int | None = None,
                 confidence_threshold: float = 0.0, smoothing_window: int = 1, smoothing_min_votes: int = 1,
                 cooldown_s: float = 0.0):
        if width <= 0 or height <= 0:
            raise ValueError("frame width/height must be positive")
        if not (pre_roll_s >= 0 and end_rest_s > 0 and max_segment_s > end_rest_s):
            raise ValueError("need pre_roll_s >= 0, end_rest_s > 0 and max_segment_s > end_rest_s")
        if cooldown_s < 0:
            raise ValueError("cooldown_s must be >= 0")
        self.rec, self.width, self.height = recognizer, int(width), int(height)
        self.pre_roll_s, self.end_rest_s, self.max_segment_s = pre_roll_s, end_rest_s, max_segment_s
        self.min_hand_run = min_hand_run or recognizer.cfg["rejection"]["active_interval_min_hand_run"]
        self.filter = ConfidenceFilter(confidence_threshold)
        self.smoother = Smoother(smoothing_window, smoothing_min_votes)
        # cooldown_s > 0: a segment whose hand run starts less than cooldown_s after the previous
        # segment's last frame (e.g. the rest of a sign cut at max_segment_s, or hand movement right
        # after a sign) is still classified but marked suppressed and not emitted. 0 = off.
        self.cooldown_s = float(cooldown_s)
        self._prev_end_t: float | None = None
        self._frames: deque = deque()     # (t, coords, observed, vis, repaired)
        self._run = 0                      # consecutive frames with a hand
        self._active_start: float | None = None
        self._last_hand_t: float | None = None
        self._run_start: float | None = None   # run_start_s of the open segment
        self._carry: tuple[float, float] | None = None  # after a forced cut: (run start, last hand time)
        self._last_t = -math.inf
        self._holistic = None

    # -------------------------------------------------------------- input
    def push(self, t: float, coords, observed, pose_vis) -> SignEvent | None:
        """One frame: coords [543, 3] (MediaPipe image-normalized), observed [4] bool per group
        (pose, face, left_hand, right_hand), pose visibility [33]. Returns an event when a
        segment ends."""
        t = float(t)
        if not math.isfinite(t) or t <= self._last_t:
            raise InputError(f"timestamps must be finite and increasing (got {t} after {self._last_t})")
        c = np.asarray(coords, np.float32)
        o = np.asarray(observed, bool).copy()
        v = np.asarray(pose_vis, np.float32)
        if c.shape != (N_LANDMARKS, 3) or o.shape != (4,) or v.shape != (GROUPS["pose"],):
            raise InputError(f"frame shapes must be (543, 3), (4,), (33,); got {c.shape}, {o.shape}, {v.shape}")
        c, repaired = self._drop_invalid_groups(c, o)
        if not np.isfinite(v).all():
            v = np.where(np.isfinite(v), v, 0.0).astype(np.float32)
        self._last_t = t
        self._frames.append((t, c, o, v, repaired))
        has_hand = bool(o[HANDS].any())
        self._run = self._run + 1 if has_hand else 0
        if has_hand:
            self._last_hand_t = t
        if self._active_start is None:
            if self._run >= self.min_hand_run:
                first_hand_t = self._frames[-self.min_hand_run][0]
                self._active_start = first_hand_t
                # hands never rested since a forced cut: same signing run, not a new sign
                cont = self._carry is not None and first_hand_t - self._carry[1] < self.end_rest_s
                self._run_start = self._carry[0] if cont else first_hand_t
                self._carry = None
            else:
                self._trim(t - self.pre_roll_s)
                return None
        if self._last_hand_t is not None and t - self._last_hand_t >= self.end_rest_s:
            return self._finalize(forced=False)
        if t - self._active_start >= self.max_segment_s:
            return self._finalize(forced=True)
        return None

    def push_rgb(self, frame_rgb: np.ndarray, t: float) -> SignEvent | None:
        """Run MediaPipe Holistic (training settings) on one RGB frame, then `push`."""
        if self._holistic is None:
            self._holistic = make_holistic(self.rec.cfg["mediapipe"])
        res = self._holistic.process(frame_rgb)
        c, o, v, _ = holistic_result_to_frame(res)
        return self.push(t, c, o, v)

    def flush(self) -> SignEvent | None:
        """End of stream: classify an open segment, if any."""
        return self._finalize(forced=False) if self._active_start is not None else None

    def close(self) -> None:
        if self._holistic is not None:
            self._holistic.close()
            self._holistic = None

    # -------------------------------------------------------------- internals
    @staticmethod
    def _drop_invalid_groups(c: np.ndarray, o: np.ndarray) -> tuple[np.ndarray, int]:
        """A group with NaN/Inf is treated as not detected (same rule as offline extraction)."""
        bad = 0
        for g in GROUPS:
            gi, sl = GROUP_INDEX[g], SLICES[g]
            if o[gi] and not np.isfinite(c[sl]).all():
                o[gi] = False
                bad += 1
            if not o[gi]:
                c[sl] = 0.0
        return c, bad

    def _trim(self, keep_from_t: float) -> None:
        """Drop frames older than keep_from_t, but always keep the last `min_hand_run` frames
        (a starting hand run must stay in the buffer even with pre_roll_s = 0)."""
        while len(self._frames) > self.min_hand_run and self._frames[0][0] < keep_from_t:
            self._frames.popleft()

    def _finalize(self, forced: bool) -> SignEvent:
        start = self._active_start - self.pre_roll_s
        frames = [f for f in self._frames if f[0] >= start]
        raw = RawLandmarks(
            times=np.array([f[0] for f in frames], np.float64), coords=np.stack([f[1] for f in frames]),
            observed=np.stack([f[2] for f in frames]), pose_vis=np.stack([f[3] for f in frames]),
            repaired_nonfinite=int(sum(f[4] for f in frames)), width=self.width, height=self.height,
            codec="live", container_fps=None, decode_s=0.0, mediapipe_s=0.0)
        pred = self.filter(self.rec.predict_raw(raw))
        suppressed = (self.cooldown_s > 0 and self._prev_end_t is not None
                      and self._active_start - self._prev_end_t < self.cooldown_s)
        emitted = None if suppressed else self.smoother.update(pred)
        event = SignEvent(pred, emitted, frames[0][0], frames[-1][0], len(frames), forced,
                          self._last_hand_t, self._active_start, suppressed, self._run_start)
        self._carry = (self._run_start, self._last_hand_t) if forced else None
        self._prev_end_t = frames[-1][0]
        # keep the most recent frames as pre-roll for the next segment
        last_t = self._frames[-1][0]
        self._frames = deque(f for f in self._frames if f[0] >= last_t - self.pre_roll_s)
        self._active_start, self._last_hand_t, self._run = None, None, 0
        return event

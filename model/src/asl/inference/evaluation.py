"""Live-evaluation helpers: frame sources, attempt/event matching and scoring (no model code).

Used by scripts/s6_03_webcam_eval.py. Kept free of camera/GUI calls (except inside CameraSource)
so the scoring can be unit-tested.
"""
from __future__ import annotations

import time
from collections import Counter
from dataclasses import dataclass, field

import numpy as np


# ---------------------------------------------------------------- frame sources
class CameraSource:
    """OpenCV camera. read() -> (ok, rgb frame, t seconds since start on the monotonic clock).

    A width/height/fps can be requested (the camera may deliver something else; the delivered
    values are recorded). Unthreaded, read() blocks in cap.read(): if processing a frame takes
    longer than one camera period, the next read waits for the following frame and every second
    frame is skipped (measured in session 2: 43.5 ms loop vs 33.3 ms period -> 15 fps), and drops
    are not observable (`dropped` is None).

    threaded=True: a reader thread calls cap.read() continuously, timestamps each frame when it
    arrives and keeps only the newest; read() returns the newest frame not yet returned (waiting
    only if none is new). Frames replaced before being read are counted in `dropped`.
    """

    kind = "camera"

    def __init__(self, index: int, width: int | None = None, height: int | None = None, fps: float | None = None,
                 capture=None, threaded: bool = False):
        import cv2

        self.cv2 = cv2
        self.cap = capture if capture is not None else cv2.VideoCapture(index)
        self.requested = {"width": width, "height": height, "fps": fps}
        for prop, val in ((cv2.CAP_PROP_FRAME_WIDTH, width), (cv2.CAP_PROP_FRAME_HEIGHT, height),
                          (cv2.CAP_PROP_FPS, fps)):
            if val:
                self.cap.set(prop, val)
        self.t0 = time.monotonic()
        self.threaded = threaded
        self.dropped = 0 if threaded else None
        self.captured = 0
        ok, frame = self.cap.read()
        if not ok:
            self.cap.release()
            raise RuntimeError(f"cannot read from camera {index}: not authorized or no device "
                               "(macOS: System Settings > Privacy & Security > Camera, enable the app "
                               "running this command, then restart it)")
        self.height, self.width = frame.shape[:2]
        self.reported_fps = float(self.cap.get(cv2.CAP_PROP_FPS) or 0) or None
        self._first = (frame, time.monotonic() - self.t0)
        if threaded:
            import threading

            self._cond = threading.Condition()
            self._latest = (frame, self._first[1], 1)  # (frame, t, sequence number)
            self.captured, self._returned_seq, self._eof, self._stop = 1, 0, False, False
            self._thread = threading.Thread(target=self._reader, daemon=True)
            self._thread.start()

    def _reader(self) -> None:
        seq = 1
        while not self._stop:
            ok, frame = self.cap.read()
            t = time.monotonic() - self.t0
            with self._cond:
                if not ok:
                    self._eof = True
                    self._cond.notify_all()
                    return
                seq += 1
                self._latest, self.captured = (frame, t, seq), seq
                self._cond.notify_all()

    def read(self):
        if self.threaded:
            with self._cond:
                while self._latest[2] <= self._returned_seq and not self._eof:
                    if not self._cond.wait(timeout=2.0):
                        return False, None, None
                if self._latest[2] <= self._returned_seq:
                    return False, None, None
                frame, t, seq = self._latest
                self.dropped += seq - self._returned_seq - 1
                self._returned_seq = seq
            return True, self.cv2.cvtColor(frame, self.cv2.COLOR_BGR2RGB), t
        if self._first is not None:
            (frame, t), self._first = self._first, None
        else:
            ok, frame = self.cap.read()
            if not ok:
                return False, None, None
            t = time.monotonic() - self.t0
        return True, self.cv2.cvtColor(frame, self.cv2.COLOR_BGR2RGB), t

    def now(self) -> float:
        return time.monotonic() - self.t0

    def close(self):
        if self.threaded:
            self._stop = True
            self._thread.join(timeout=2.0)
        self.cap.release()


class VideoReplaySource:
    """A recorded clip replayed in real time, as a camera would deliver it (SIMULATION ONLY).

    read() waits until the next frame's timestamp on the wall clock; if processing fell behind, it
    returns the newest frame that is due and counts the skipped ones as dropped - like a live
    camera that always shows the current image. Timestamps are the clip's own frame times.
    """

    kind = "video_replay"

    def __init__(self, path: str, clock=time.monotonic, sleep=time.sleep):
        import av

        with av.open(path) as c:
            s = c.streams.video[0]
            fps = float(s.average_rate) if s.average_rate else 30.0
            self.frames = [(f.time if f.time is not None else i / fps, f.to_ndarray(format="rgb24"))
                           for i, f in enumerate(c.decode(s))]
        if not self.frames:
            raise RuntimeError(f"no decodable frames in {path}")
        t_first = self.frames[0][0]
        self.frames = [(t - t_first, img) for t, img in self.frames]
        self.height, self.width = self.frames[0][1].shape[:2]
        self.clock, self.sleep = clock, sleep
        self.t0 = clock()
        self.i = 0
        self.dropped = 0

    def now(self) -> float:
        return self.clock() - self.t0

    def read(self):
        if self.i >= len(self.frames):
            return False, None, None
        due = self.now()
        j = self.i
        while j + 1 < len(self.frames) and self.frames[j + 1][0] <= due:
            j += 1  # newer frames are already due: skip to the newest one
        self.dropped += j - self.i
        t, img = self.frames[j]
        if t > due:
            self.sleep(t - due)
        self.i = j + 1
        return True, img, t

    def close(self):
        pass


# ---------------------------------------------------------------- attempts and scoring
@dataclass
class Attempt:
    index: int
    expected: str
    start_s: float              # "GO / sign now" prompt (stream time)
    end_s: float                # end of the signing window
    ready_start_s: float | None = None  # "get ready" prompt (the sign's name is shown from here)
    go_gate_wait_s: float | None = None  # extra WAIT time spent waiting for the hands to go down
    hands_down_at_go: bool | None = None  # False: GO was given after max wait with hands still visible
    hand_frames: int = 0
    frames: int = 0
    events: list = field(default_factory=list)        # (event, emitted_at_s) answering this attempt
    early_events: list = field(default_factory=list)  # signed during this attempt's READY phase


def assign_events(attempts: list[Attempt], events: list[tuple], grace_s: float) -> list[tuple]:
    """Attach each (SignEvent, emitted_at_s) to an attempt.

    - signing started (recorded `run_start_s`/`first_hand_s`) inside an attempt's GO window [start, end) -> that
      attempt's answers, however late it ends (session 3: a sign forced-cut past the 5 s window, and
      the tail segment after the cut, were pushed into the next prompt's WAIT phase);
    - otherwise by end of signing (`last_hand_s`): [start, end + grace) of an attempt -> its answers
      (a segment started in WAIT and ended after GO stays here and is scored as pre_go); the grace
      never extends into the next attempt's READY phase;
    - [ready_start, start) of an attempt -> that attempt's `early_events` (signed before "GO";
      reported, never scored as the answer);
    - anything else -> returned as orphans.
    """
    order = sorted(attempts, key=lambda a: a.start_s)
    limits = {}
    for a, nxt in zip(order, order[1:] + [None]):
        stop = a.end_s + grace_s
        if nxt is not None:
            stop = min(stop, nxt.ready_start_s if nxt.ready_start_s is not None else nxt.start_s)
        limits[id(a)] = stop
    orphans = []
    for ev, at in events:
        # only a recorded first hand frame: start_s includes the pre-roll, so older logs keep the old rule
        first = _recorded_run_start(ev)
        a = None if first is None else next((a for a in order if a.start_s <= first < a.end_s), None)
        if a is not None:
            a.events.append((ev, at))
            continue
        key = ev.last_hand_s if ev.last_hand_s is not None else ev.end_s
        a = next((a for a in order if a.start_s <= key < limits[id(a)]), None)
        if a is not None:
            a.events.append((ev, at))
            continue
        a = next((a for a in order if a.ready_start_s is not None and a.ready_start_s <= key < a.start_s), None)
        (a.early_events.append((ev, at)) if a is not None else orphans.append((ev, at)))
    return orphans


def _recorded_run_start(ev) -> float | None:
    """Start of signing: first hand frame of the uninterrupted hand run (a segment continuing a forced
    cut starts where the cut segment's run started), or None if the event did not record it."""
    for k in ("run_start_s", "first_hand_s"):
        if getattr(ev, k, None) is not None:
            return getattr(ev, k)
    return None


def _signing_start(ev) -> float:
    """Start of signing (falls back to the segment start, incl. pre-roll, for old events)."""
    r = _recorded_run_start(ev)
    return r if r is not None else ev.start_s


def _outcome(p, expected: str) -> str:
    return "correct" if p.status == "ok" and p.label == expected else "wrong" if p.status == "ok" else p.status


def attempt_row(a: Attempt) -> dict:
    """One scored attempt (strict scoring).

    The answer is the first event of the attempt that is not suppressed (cooldown) and whose
    signing started at or after GO (first hand frame >= GO). Events whose hands were already up
    before GO mix rest and sign; they are reported as `pre_go_events`, not scored. If an attempt has
    only such events its outcome is "pre_go_only". `legacy_*` keeps the previous rule (first event
    of the attempt, whatever its start) so earlier sessions stay comparable.
    """
    live = [(e, at) for e, at in a.events if not getattr(e, "suppressed", False)]
    post = [(e, at) for e, at in live if _signing_start(e) >= a.start_s]
    pre = [(e, at) for e, at in live if _signing_start(e) < a.start_s]
    row = {"attempt": a.index, "expected": a.expected, "ready_start_s": a.ready_start_s, "start_s": a.start_s,
           "end_s": a.end_s, "events": len(a.events), "duplicates": max(0, len(post) - 1),
           "suppressed_events": len(a.events) - len(live), "pre_go_events": len(pre),
           "pre_go_labels": [e.prediction.label for e, _ in pre],
           "early_events": len(a.early_events), "go_gate_wait_s": a.go_gate_wait_s,
           "hands_down_at_go": a.hands_down_at_go,
           "early_labels": [e.prediction.label for e, _ in a.early_events],
           "hand_visible_fraction": a.hand_frames / a.frames if a.frames else 0.0}
    if a.events:
        e0 = a.events[0][0]
        row["legacy_outcome"] = _outcome(e0.prediction, a.expected)
        row["legacy_predicted"] = e0.prediction.label
    else:
        row["legacy_outcome"], row["legacy_predicted"] = "missed", None
    if not post:
        return {**row, "outcome": "pre_go_only" if pre else "missed", "status": None, "predicted": None,
                "confidence": None, "in_top5": False,
                "reason": "only segments that started before GO" if pre else "no segment detected",
                "delay_s": None, "segment_frames": None, "forced_cut": None, "segment_start_s": None,
                "first_hand_s": None, "last_hand_s": None, "started_before_go": bool(pre) or None}
    ev, at = post[0]
    p = ev.prediction
    return {**row, "outcome": _outcome(p, a.expected), "status": p.status, "predicted": p.label,
            "confidence": p.confidence, "in_top5": any(l == a.expected for l, _ in p.top_k), "reason": p.reason,
            "delay_s": None if ev.last_hand_s is None else at - ev.last_hand_s, "segment_frames": ev.frames,
            "forced_cut": ev.forced, "segment_start_s": ev.start_s, "first_hand_s": _signing_start(ev),
            "last_hand_s": ev.last_hand_s, "started_before_go": False,
            "top3": [(l, round(c, 3)) for l, c in p.top_k[:3]]}


def summarize(rows: list[dict]) -> dict:
    n = len(rows)
    out = Counter(r["outcome"] for r in rows)
    answered = [r for r in rows if r["status"] == "ok"]
    delays = [r["delay_s"] for r in rows if r["delay_s"] is not None]
    conf_ok = [r["confidence"] for r in rows if r["outcome"] == "correct"]
    conf_bad = [r["confidence"] for r in rows if r["outcome"] == "wrong"]
    per_sign: dict[str, list] = {}
    for r in rows:
        per_sign.setdefault(r["expected"], []).append(r["outcome"] == "correct")

    def q(xs, p):
        return float(np.percentile(xs, p)) if xs else None
    return {
        "attempts": n, "outcomes": dict(out),
        "accuracy_all_attempts": out["correct"] / n if n else None,
        "accuracy_when_answered": (out["correct"] / len(answered)) if answered else None,
        "top5_rate_when_answered": (sum(r["in_top5"] for r in answered) / len(answered)) if answered else None,
        "attempts_with_duplicates": sum(r["duplicates"] > 0 for r in rows),
        "early_events_total": sum(r.get("early_events", 0) for r in rows),
        "pre_go_events_total": sum(r.get("pre_go_events", 0) for r in rows),
        "suppressed_events_total": sum(r.get("suppressed_events", 0) for r in rows),
        "attempts_with_hands_up_before_go": sum(bool(r.get("started_before_go")) for r in rows),
        "legacy_accuracy_all_attempts": (sum(r.get("legacy_outcome") == "correct" for r in rows) / n) if n else None,
        "scoring": "strict: first non-suppressed segment whose first hand frame is at/after GO",
        "rejection_reasons": dict(Counter(r["reason"] for r in rows if r["status"] == "rejected")),
        "delay_s": {"n": len(delays), "median": q(delays, 50), "p95": q(delays, 95)},
        "confidence_correct_median": q(conf_ok, 50), "confidence_wrong_median": q(conf_bad, 50),
        "per_sign_accuracy": {k: sum(v) / len(v) for k, v in sorted(per_sign.items())},
        "mean_hand_visible_fraction": float(np.mean([r["hand_visible_fraction"] for r in rows])) if rows else None,
    }


# ---------------------------------------------------------------- input diagnostics (no images needed)
def frame_diagnostics(times, observed, shoulder_dx) -> dict:
    """Per-session input statistics from per-frame MediaPipe output.

    times [N] s; observed [N, 4] bool (pose, face, left_hand, right_hand); shoulder_dx [N] = |x(11) -
    x(12)| in MediaPipe image-normalized units (= shoulder width / frame width) or NaN without pose.
    Comparable to the training store: shoulder_width_over_frame_width (dev100 median 0.371) and the
    hand mix (signers in the dataset mostly sign with the right hand).
    """
    t = np.asarray(times, float)
    o = np.asarray(observed, bool).reshape(-1, 4)
    dx = np.asarray(shoulder_dx, float)
    gaps = np.diff(t) * 1000.0
    left, right = o[:, 2], o[:, 3]
    hands = left | right
    sw = dx[np.isfinite(dx) & o[:, 0]]

    def frac(m):
        return float(m.mean()) if len(m) else None
    return {
        "frames": int(len(t)),
        "frame_interval_ms": {"median": float(np.median(gaps)) if len(gaps) else None,
                              "p95": float(np.percentile(gaps, 95)) if len(gaps) else None,
                              "share_over_50ms": frac(gaps > 50.0)},
        "processed_fps": float(1000.0 / np.median(gaps)) if len(gaps) else None,
        "pose_present": frac(o[:, 0]),
        "any_hand_present": frac(hands),
        "both_hands": frac(left & right),
        "left_hand_only": frac(left & ~right),
        "right_hand_only": frac(right & ~left),
        "shoulder_width_over_frame_width_median": float(np.median(sw)) if len(sw) else None,
    }


def framing_advice(shoulder_ratio: float | None, band: tuple[float, float] = (0.257, 0.421)) -> str:
    """Guidance from shoulder width / frame width vs the training interquartile range (dev100:
    25 % 0.257, median 0.371, 75 % 0.421)."""
    if shoulder_ratio is None or not np.isfinite(shoulder_ratio):
        return "framing: shoulders not visible - show your upper body"
    lo, hi = band
    pct = f"shoulders {100 * shoulder_ratio:.0f}% of width (training {100 * lo:.0f}-{100 * hi:.0f}%)"
    if shoulder_ratio < lo:
        return f"framing: {pct} - move closer"
    if shoulder_ratio > hi:
        return f"framing: {pct} - move back"
    return f"framing: {pct} - OK"

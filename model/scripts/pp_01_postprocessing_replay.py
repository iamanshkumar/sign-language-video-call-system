"""Paper Objective 5 (OFFLINE): streaming post-processing replayed on saved webcam sessions.

  PYTHONPATH=src .venv/bin/python scripts/pp_01_postprocessing_replay.py --out reports/paper/postprocessing

Replays the saved landmarks of the core webcam development sessions (S3, S4, 5a, 5b, 5c-1, 5c-2; one signer,
>= 25 fps; reports/e0/webcam_replay_set) through StreamingRecognizer with the live model of each session
(the exported baseline BiLSTM candidate) and a predeclared grid of post-processing settings, changing one
setting at a time from the live defaults (cooldown 0, threshold 0, smoothing off):
- cooldown_s in {0.0, 0.5, 1.0}  (a segment starting within cooldown_s of the previous one is not emitted);
- confidence_threshold in {0.0, 0.4, 0.57, 0.7}  (0.57 = threshold chosen on validation for 80 % precision,
  reports/step6/calibration_hands_pose_w30_bilstm_e100_s3.json).
Scoring is what a user sees: per attempt, the first EMITTED label among the segments whose signing started
at/after GO (strict rule) is the answer; further emitted labels in the attempt are duplicates; emitted
labels outside every attempt are spurious. Attempts flagged in the replay-set manifest (signed during WAIT,
pre-GO only) are excluded from the core metrics. Development data (one signer, already inspected):
descriptive only. Settings (0, 0) must reproduce the manifest's replay answers. Session files are only
read; --out must be new or empty.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from asl.config import PROJECT_ROOT  # noqa: E402
from asl.inference.evaluation import Attempt, _signing_start, assign_events  # noqa: E402

REPLAY_SET = PROJECT_ROOT / "reports" / "e0" / "webcam_replay_set"
COOLDOWNS = [0.0, 0.5, 1.0]
THRESHOLDS = [0.0, 0.4, 0.57, 0.7]
GRID = [{"cooldown_s": c, "confidence_threshold": 0.0} for c in COOLDOWNS] + \
       [{"cooldown_s": 0.0, "confidence_threshold": t} for t in THRESHOLDS if t > 0]


def _module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def score_attempt(a: Attempt) -> dict:
    """User-facing outcome of one attempt from its assigned events (see module docstring)."""
    post = [(e, at) for e, at in a.events if _signing_start(e) >= a.start_s]
    emitted = [e for e, _ in post if e.emitted is not None]
    first = emitted[0] if emitted else None
    return {"segments_after_go": len(post), "emitted": len(emitted),
            "answer": first.emitted if first else None,
            "answer_confidence": first.prediction.confidence if first else None,
            "correct": bool(first is not None and first.emitted == a.expected),
            "answered": first is not None, "duplicate_emissions": max(0, len(emitted) - 1),
            "no_segment": len(post) == 0,
            "suppressed": sum(1 for e, _ in post if getattr(e, "suppressed", False)),
            "low_confidence": sum(1 for e, _ in post if e.prediction.status == "low_confidence")}


def summarize(rows: pd.DataFrame) -> dict:
    n = len(rows)
    ans = rows[rows.answered]
    return {"attempts": n, "correct": int(rows.correct.sum()), "top1": rows.correct.mean() if n else None,
            "answered": int(rows.answered.sum()), "coverage": rows.answered.mean() if n else None,
            "precision_of_answers": ans.correct.mean() if len(ans) else None,
            "attempts_with_duplicates": int((rows.duplicate_emissions > 0).sum()),
            "duplicate_emissions": int(rows.duplicate_emissions.sum()),
            "no_segment": int(rows.no_segment.sum()), "suppressed_segments": int(rows.suppressed.sum()),
            "low_confidence_segments": int(rows.low_confidence.sum())}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", required=True)
    out = PROJECT_ROOT / ap.parse_args().out
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"{out} exists and is not empty")
    rp = _module(PROJECT_ROOT / "scripts" / "s6_05_replay_session.py", "replay")
    e02 = _module(PROJECT_ROOT / "scripts" / "e0_02_webcam_replay_set.py", "e02")
    man = pd.read_csv(REPLAY_SET / "manifest.csv")
    sessions = {k: v for k, v in e02.SESSIONS.items() if v[2]}            # core sessions only
    rows, spurious = [], []
    for setting in GRID:
        for key, (folder, cond, _) in sessions.items():
            S = PROJECT_ROOT / e02.R / folder
            events, settings, attempts = rp.replay(S, setting)
            orphans = assign_events(attempts, events, 1.5)
            spurious.append({**setting, "session": key, "spurious_emissions": sum(1 for e, _ in orphans if e.emitted is not None)})
            for a in attempts:
                m = man[(man.session == key) & (man.attempt == a.index)].iloc[0]
                rows.append({**setting, "session": key, "attempt": a.index, "expected": a.expected,
                             "excluded": m.excluded if isinstance(m.excluded, str) else "",
                             "manifest_replay_answer": m.get("replay_predicted"), **score_attempt(a)})
    R = pd.DataFrame(rows)
    base = R[(R.cooldown_s == 0) & (R.confidence_threshold == 0)]
    seg = base.manifest_replay_answer.notna()
    parity = bool((base[seg].answer.fillna("") == base[seg].manifest_replay_answer).all())
    core = R[R.excluded == ""]
    table = []
    for (c, t), g in core.groupby(["cooldown_s", "confidence_threshold"], sort=False):
        sp = sum(x["spurious_emissions"] for x in spurious if x["cooldown_s"] == c and x["confidence_threshold"] == t)
        table.append({"cooldown_s": c, "confidence_threshold": t, **summarize(g), "spurious_emissions_outside_attempts": sp})
    T = pd.DataFrame(table)
    by_session = core.groupby(["cooldown_s", "confidence_threshold", "session"]).apply(
        lambda g: pd.Series(summarize(g))).reset_index()
    out.mkdir(parents=True, exist_ok=True)
    R.to_csv(out / "per_attempt.csv", index=False)
    T.to_csv(out / "summary_by_setting.csv", index=False)
    by_session.to_csv(out / "summary_by_setting_and_session.csv", index=False)
    summary = {"sessions": list(sessions), "grid": GRID, "default_reproduces_replay_set_answers": parity,
               "core_attempts": int(len(core) / len(GRID)),
               "note": "one signer, development data, baseline BiLSTM candidate (live model); descriptive only"}
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    pd.set_option("display.width", 220)
    print(json.dumps(summary, indent=1))
    print(T.round(3).to_string(index=False))
    print(f"saved to {out.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()

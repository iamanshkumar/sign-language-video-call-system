"""Step 6.5: replay a recorded webcam session's landmarks offline (no camera, no MediaPipe).

  PYTHONPATH=src .venv/bin/python scripts/s6_05_replay_session.py \
      --session reports/step6/webcam_eval/<session> [--cooldown-s 1.0] [--tag cooldown1]

Needs a session recorded with --save-landmarks (landmarks.npz) and prompt times (attempts.csv with
ready_start_s/start_s/end_s). The saved per-frame landmarks are pushed through StreamingRecognizer
with the session's settings (plus any overrides) and scored with the current rules. With no
overrides the replay must reproduce the recorded events exactly - this is checked and reported.
Output: <session>/replay_<tag>/ (attempts.csv, events.jsonl, summary.json). The session's own files
are never modified.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from asl.config import PROJECT_ROOT, STEP2_CONFIG, load_yaml
from asl.inference import SignRecognizer, StreamingRecognizer
from asl.inference.evaluation import Attempt, assign_events, attempt_row, summarize


def replay(session: Path, overrides: dict) -> tuple[list[tuple], dict, list[Attempt]]:
    summ = json.loads((session / "summary.json").read_text())
    z = np.load(session / "landmarks.npz")
    md = PROJECT_ROOT / summ["model_dir"]
    rec = SignRecognizer.load(md / "best_model.pt", md / "class_mapping.json", load_yaml(STEP2_CONFIG),
                              load_yaml(PROJECT_ROOT / "configs" / "step3_sequences.yaml"))
    settings = {**summ["settings"], **overrides}
    sr = StreamingRecognizer(rec, int(z["width"]), int(z["height"]), **settings)
    events = []
    for t, c, o, v in zip(z["t"], z["coords"], z["observed"], z["pose_vis"]):
        ev = sr.push(float(t), c, o, v)
        if ev:
            events.append((ev, float(t)))  # offline: emitted when the closing frame is processed
    ev = sr.flush()
    if ev:
        events.append((ev, float(z["t"][-1])))
    att = pd.read_csv(session / "attempts.csv")
    attempts = [Attempt(int(r.attempt), r.expected, float(r.start_s), float(r.end_s), ready_start_s=float(r.ready_start_s))
                for r in att.itertuples()]
    hands = z["observed"][:, 2] | z["observed"][:, 3]
    for a in attempts:  # same definition as the live tool: frames in the GO window with any hand
        m = (z["t"] >= a.start_s) & (z["t"] < a.end_s)
        a.frames, a.hand_frames = int(m.sum()), int(hands[m].sum())
    return events, settings, attempts


def matches_recorded(events: list[tuple], recorded: list[dict]) -> bool:
    if len(events) != len(recorded):
        return False
    for (e, _), r in zip(events, recorded):
        p = e.prediction
        same_conf = (p.confidence is None and r["confidence"] is None) or (
            p.confidence is not None and r["confidence"] is not None and abs(p.confidence - r["confidence"]) < 1e-5)
        if not (p.status == r["status"] and p.label == r["label"] and same_conf and e.frames == r["frames"]
                and abs(e.start_s - r["start_s"]) < 1e-6 and abs((e.last_hand_s or 0) - (r["last_hand_s"] or 0)) < 1e-6):
            return False
    return True


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--session", required=True)
    ap.add_argument("--cooldown-s", type=float, default=None)
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()
    session = PROJECT_ROOT / args.session
    for f in ("landmarks.npz", "attempts.csv", "summary.json"):
        if not (session / f).exists():
            raise SystemExit(f"{session / f} missing (record the session with --save-landmarks)")
    overrides = {} if args.cooldown_s is None else {"cooldown_s": args.cooldown_s}
    events, settings, attempts = replay(session, overrides)
    recorded = [json.loads(l) for l in (session / "events.jsonl").read_text().splitlines()]
    grace = min(float(np.median([a.start_s - a.ready_start_s for a in attempts])) / 2, 1.5)
    orphans = assign_events(attempts, events, grace)
    rows = [attempt_row(a) for a in attempts]
    summary = summarize(rows)
    summary.update({"source_session": args.session, "settings": settings, "overrides": overrides,
                    "replay_reproduces_recorded_events": matches_recorded(events, recorded) if not overrides else None,
                    "events": len(events), "events_outside_attempts": len(orphans),
                    "note": "offline replay of saved landmarks; delays exclude live processing time"})
    out = session / f"replay_{args.tag or ('cooldown' + str(args.cooldown_s) if overrides else 'default')}"
    out.mkdir(exist_ok=True)
    pd.DataFrame(rows).to_csv(out / "attempts.csv", index=False)
    with open(out / "events.jsonl", "w") as f:
        for e, at in events:
            p = e.prediction
            f.write(json.dumps({"emitted_at_s": at, "start_s": e.start_s, "first_hand_s": e.first_hand_s, "run_start_s": e.run_start_s,
                                "last_hand_s": e.last_hand_s, "end_s": e.end_s, "frames": e.frames, "forced": e.forced,
                                "suppressed": e.suppressed, "status": p.status, "label": p.label,
                                "confidence": p.confidence, "reason": p.reason, "emitted": e.emitted,
                                "top_k": p.top_k}) + "\n")
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str) + "\n")
    print(json.dumps({k: v for k, v in summary.items() if k != "per_sign_accuracy"}, indent=1, default=str))
    print(f"saved to {out.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()

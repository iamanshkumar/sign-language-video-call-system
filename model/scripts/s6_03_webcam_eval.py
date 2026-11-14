"""Step 6.3: guided live evaluation with a webcam and a human signer (or a simulated replay).

Webcam session (a person signs the prompted words; nothing but landmarks-derived results is saved):
  PYTHONPATH=src .venv/bin/python scripts/s6_03_webcam_eval.py --camera 0 \
      --model-dir models/candidates/hands_pose_w30_bilstm_e100_s3 \
      --signer-id S01 --lighting "daylight, front" --camera-position "laptop, arm's length" [--preview]

Simulation (pipeline check only, NOT a real-world result): replay a recorded clip in real time:
  PYTHONPATH=src .venv/bin/python scripts/s6_03_webcam_eval.py --video clip.mp4 --expected africa \
      --model-dir models/candidates/hands_pose_w30_bilstm_e100_s3

The webcam session prompts each sign `--attempts` times: a `--ready-s` pause ("get ready, hands
down"), then a `--attempt-s` window ("SIGN NOW", sign once, then lower the hands). Every frame goes
through MediaPipe Holistic (training settings) and the StreamingRecognizer with the settings in
configs/inference.yaml. The first event whose end of signing falls in an attempt's window (+ grace)
is that attempt's answer; further events are counted as duplicates. Delay = time the event was
emitted minus the last frame with a hand. Results: reports/step6/webcam_eval/<session>/
(attempts.csv, events.jsonl, summary.json, summary.md). No video or images are written.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from asl.config import PROJECT_ROOT, STEP2_CONFIG, load_yaml
from asl.inference import SignRecognizer, StreamingRecognizer
from asl.inference.evaluation import (Attempt, CameraSource, VideoReplaySource, assign_events, attempt_row,
                                      frame_diagnostics, framing_advice, summarize)
from asl.landmarks.extract import holistic_result_to_frame, make_holistic
from asl.landmarks.layout import GROUP_INDEX

HANDS = [GROUP_INDEX["left_hand"], GROUP_INDEX["right_hand"]]
POSE0 = 0  # pose block starts at landmark 0 of the 543 (layout.SLICES["pose"])
TRAIN_VERIFICATION = PROJECT_ROOT / "reports" / "step2" / "dev100_verification.json"


def default_signs(classes: list[str], step: int = 10) -> list[str]:
    """Every `step`-th class in vocabulary order: a fixed rule, not chosen from model results."""
    return classes[::step]


def main() -> None:
    ap = argparse.ArgumentParser()
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--camera", type=int)
    src.add_argument("--video", help="SIMULATION: replay a recorded clip in real time")
    ap.add_argument("--expected", help="--video only: the clip's true label")
    ap.add_argument("--model-dir", default=None)
    ap.add_argument("--signs", default=None, help="comma-separated labels (default: every 10th vocabulary class)")
    ap.add_argument("--attempts", type=int, default=2)
    ap.add_argument("--ready-s", type=float, default=3.0)
    ap.add_argument("--attempt-s", type=float, default=5.0)
    ap.add_argument("--signer-id", default="unspecified", help="pseudonym; the signer must not be in the dataset")
    ap.add_argument("--lighting", default="")
    ap.add_argument("--camera-position", default="")
    ap.add_argument("--notes", default="")
    ap.add_argument("--preview", action="store_true", help="show the camera with prompts (OpenCV window)")
    ap.add_argument("--width", type=int, default=640, help="camera only: requested width (dataset: 98%% 640x480)")
    ap.add_argument("--height", type=int, default=480, help="camera only: requested height")
    ap.add_argument("--fps", type=float, default=30.0, help="camera only: requested frame rate")
    ap.add_argument("--threaded-capture", action=argparse.BooleanOptionalAction, default=True,
                    help="camera only: read frames in a background thread and always process the newest "
                         "(avoids waiting for the next frame when a loop overruns one camera period)")
    ap.add_argument("--preview-every", type=int, default=3,
                    help="redraw the preview every N processed frames (imshow+waitKey cost ~13 ms per draw)")
    ap.add_argument("--hands-down-s", type=float, default=0.5,
                    help="GO is shown only after no hand has been visible for this long (0 = no gate)")
    ap.add_argument("--max-wait-s", type=float, default=6.0,
                    help="give GO anyway after this much extra waiting (recorded as hands_down_at_go=False)")
    ap.add_argument("--framing-check-s", type=float, default=5.0,
                    help="camera only: show framing guidance for this long before the first prompt")
    ap.add_argument("--save-landmarks", action="store_true",
                    help="also save per-frame MediaPipe landmarks (landmarks.npz; no images) for offline diagnosis")
    ap.add_argument("--out", default="reports/step6/webcam_eval")
    args = ap.parse_args()
    cfg = load_yaml(PROJECT_ROOT / "configs" / "inference.yaml")
    md = PROJECT_ROOT / (args.model_dir or cfg["model_dir"])
    rec = SignRecognizer.load(md / "best_model.pt", md / "class_mapping.json", load_yaml(STEP2_CONFIG),
                              load_yaml(PROJECT_ROOT / "configs" / "step3_sequences.yaml"), top_k=cfg["top_k"])
    if args.video and not args.expected:
        raise SystemExit("--video needs --expected <label>")
    signs = [args.expected] if args.video else (args.signs.split(",") if args.signs else default_signs(rec.classes))
    unknown = [s for s in signs if s not in rec.classes]
    if unknown:
        raise SystemExit(f"not in the model's vocabulary: {unknown}")

    source = VideoReplaySource(args.video) if args.video else CameraSource(args.camera, args.width, args.height, args.fps,
                                                                            threaded=args.threaded_capture)
    sr = StreamingRecognizer(rec, source.width, source.height, **cfg["segmentation"], **cfg["postprocess"])
    hol = make_holistic(rec.cfg["mediapipe"])
    cv2 = None
    if args.preview:
        import cv2
    plan = ([(0, args.expected, 0.0, 0.0)] if args.video else
            [(k, s, 0.0, 0.0) for k, s in enumerate(x for x in signs for _ in range(args.attempts))])
    attempts: list[Attempt] = []
    events: list[tuple] = []
    frame_times, mp_ms, observed = [], [], []
    frame_log, lm_t, lm_c, lm_o, lm_v = [], [], [], [], []

    state = {"n": 0, "last_hand_t": None}

    def recent_ratio(n: int = 15):
        vals = [r["shoulder_dx"] for r in frame_log[-n:] if r["pose"] and np.isfinite(r["shoulder_dx"])]
        return float(np.median(vals)) if vals else None

    def step(label: str) -> bool:
        """Process one frame; returns False at end of stream."""
        import time as _t
        t_read = _t.perf_counter()
        ok, rgb, t = source.read()
        if not ok:
            return False
        t0 = _t.perf_counter()
        c, o, v, _ = holistic_result_to_frame(hol.process(rgb))
        mp_ms.append(1000 * (_t.perf_counter() - t0))
        frame_times.append(t)
        observed.append((t, bool(o[HANDS].any())))
        dx = abs(float(c[POSE0 + 11, 0] - c[POSE0 + 12, 0])) if o[0] else float("nan")
        frame_log.append({"t": t, "read_wait_ms": 1000 * (t0 - t_read), "mediapipe_ms": mp_ms[-1],
                          "pose": bool(o[0]), "left_hand": bool(o[2]), "right_hand": bool(o[3]), "shoulder_dx": dx,
                          "phase": label.split(":")[0]})
        if o[HANDS].any():
            state["last_hand_t"] = t
        if args.save_landmarks:
            lm_t.append(t), lm_c.append(c.astype(np.float32)), lm_o.append(o.copy()), lm_v.append(v.astype(np.float32))
        ev = sr.push(t, c, o, v)
        if ev:
            events.append((ev, source.now()))
            p = ev.prediction
            print(f"  -> {p.status:14s} {str(p.label):14s} conf={p.confidence if p.confidence is None else round(p.confidence, 3)} "
                  f"{p.reason}")
        state["n"] += 1
        if cv2 is not None and state["n"] % max(1, args.preview_every) == 0:
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            hands_up = bool(o[HANDS].any())
            if label.startswith("GO"):
                colour, text = (0, 255, 0), label
            elif label.startswith("WAIT") and hands_up:
                colour, text = (0, 0, 255), label + "  - lower your hands out of view"
            else:
                colour, text = (0, 165, 255), label
            cv2.putText(bgr, text, (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, colour, 2)
            cv2.putText(bgr, framing_advice(recent_ratio()), (12, bgr.shape[0] - 14), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (255, 255, 255), 1)
            cv2.imshow("sign evaluation (q to quit)", bgr)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                return False
        return True

    alive = True
    if args.video:
        attempts.append(Attempt(0, args.expected, 0.0, float("inf"), ready_start_s=0.0))
        while alive:
            alive = step(f"replay: {args.expected}")
    else:
        if args.framing_check_s > 0:
            print("FRAMING CHECK: sit so your upper body is visible; rest your hands OUT of view (e.g. in your lap)")
            t_end = source.now() + args.framing_check_s
            while alive and source.now() < t_end:
                alive = step("CHECK: framing")
            print("    " + framing_advice(recent_ratio(60)))
        for k, sign, _, _ in plan:
            ready_start = source.now()
            print(f"[{k + 1}/{len(plan)}] WAIT - hands down and out of view - next sign: {sign.upper()} (do not sign yet)")
            t_end = ready_start + args.ready_s
            while alive and source.now() < t_end:
                alive = step(f"WAIT: next {sign}")
            # gated GO: only once no hand has been seen for hands_down_s (or after max_wait_s)
            gate_start, hands_down = source.now(), True
            while alive and args.hands_down_s > 0:
                lh = state["last_hand_t"]
                if lh is None or source.now() - lh >= args.hands_down_s:
                    break
                if source.now() - gate_start >= args.max_wait_s:
                    hands_down = False
                    print("    (hands still visible - giving GO anyway; this attempt is flagged)")
                    break
                alive = step(f"WAIT: next {sign}")
            go = source.now()
            print(f"    GO: sign {sign.upper()} once, then lower your hands out of view")
            a = Attempt(k, sign, go, go + args.attempt_s, ready_start_s=ready_start,
                        go_gate_wait_s=go - gate_start, hands_down_at_go=hands_down)
            while alive and source.now() < a.end_s:
                alive = step(f"GO: {sign}")
            attempts.append(a)
            if not alive:
                break
        t_end = source.now() + cfg["segmentation"]["end_rest_s"] + 0.5  # let the last segment close
        while alive and source.now() < t_end:
            alive = step("done")
    ev = sr.flush()
    if ev:
        events.append((ev, source.now()))
    hol.close()
    source.close()
    if cv2 is not None:
        cv2.destroyAllWindows()

    grace = min(args.ready_s / 2, 1.5)
    orphans = assign_events(attempts, events, grace)
    for a in attempts:
        win = [h for t, h in observed if a.start_s <= t < min(a.end_s, frame_times[-1] + 1)]
        a.frames, a.hand_frames = len(win), sum(win)
    rows = [attempt_row(a) for a in attempts]
    ft = np.diff(frame_times)
    fl = pd.DataFrame(frame_log)
    diag = frame_diagnostics(fl["t"], fl[["pose", "left_hand", "right_hand"]].assign(face=False)[
        ["pose", "face", "left_hand", "right_hand"]].to_numpy(), fl["shoulder_dx"]) if len(fl) else {}
    if len(fl):
        diag["read_wait_ms_median"] = float(fl["read_wait_ms"].median())
        diag["loop_ms_median"] = float(np.median(np.diff(fl["t"])) * 1000) if len(fl) > 1 else None
    ref = None
    if TRAIN_VERIFICATION.exists():
        v = json.loads(TRAIN_VERIFICATION.read_text())
        ref = {"shoulder_width_over_frame_width_median": v["normalization"]["shoulder_width_over_image_width"].get("50%"),
               "inside_active_interval": v["missing_landmarks"].get("frames_inside_active_interval")}
    summary = summarize(rows)
    summary.update({
        "mode": source.kind, "real_world": source.kind == "camera",
        "signer_id": args.signer_id, "lighting": args.lighting, "camera_position": args.camera_position,
        "notes": args.notes, "model_dir": str(md.relative_to(PROJECT_ROOT)), "model": rec.info.get("experiment_id"),
        "settings": {**cfg["segmentation"], **cfg["postprocess"]}, "frame_size": [source.width, source.height],
        "frames": len(frame_times),
        # camera frames never processed: counted with --threaded-capture, None (unobservable) without
        "dropped_frames": source.dropped,
        "camera_fps_measured": (getattr(source, "captured", 0) / frame_times[-1])
        if getattr(source, "threaded", False) and frame_times and frame_times[-1] > 0 else None,
        "camera_requested": getattr(source, "requested", None),
        "camera_reported_fps": getattr(source, "reported_fps", None),
        "camera_frames_captured": getattr(source, "captured", None),
        "capture_options": {"threaded": getattr(source, "threaded", None), "preview_every": args.preview_every,
                            "hands_down_s": args.hands_down_s, "max_wait_s": args.max_wait_s},
        "framing_at_end": framing_advice(diag.get("shoulder_width_over_frame_width_median") if diag else None),
        "input_diagnostics": diag, "training_reference": ref,
        "capture_fps_median": float(1 / np.median(ft)) if len(ft) else None,  # = processed fps (name kept)
        "processed_fps_median": float(1 / np.median(ft)) if len(ft) else None,
        "mediapipe_ms": {"median": float(np.median(mp_ms)), "p95": float(np.percentile(mp_ms, 95))} if mp_ms else None,
        "events_outside_attempts": len(orphans),
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds")})
    session = f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}_{source.kind}_{args.signer_id}"
    out = PROJECT_ROOT / args.out / session
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out / "attempts.csv", index=False)
    fl.to_csv(out / "frames.csv", index=False)  # per-frame timing/visibility only (no images, no landmarks)
    if args.save_landmarks and lm_t:
        np.savez_compressed(out / "landmarks.npz", t=np.array(lm_t), coords=np.stack(lm_c), observed=np.stack(lm_o),
                            pose_vis=np.stack(lm_v), width=source.width, height=source.height)
    with open(out / "events.jsonl", "w") as f:
        for ev, at in events:
            p = ev.prediction
            f.write(json.dumps({"emitted_at_s": at, "start_s": ev.start_s, "first_hand_s": ev.first_hand_s, "run_start_s": ev.run_start_s, "end_s": ev.end_s,
                                "last_hand_s": ev.last_hand_s, "frames": ev.frames, "forced": ev.forced,
                                "status": p.status, "label": p.label, "confidence": p.confidence,
                                "reason": p.reason, "emitted": ev.emitted, "top_k": p.top_k}) + "\n")
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str) + "\n")
    tag = "REAL-WORLD webcam session" if summary["real_world"] else "SIMULATION (recorded clip replay) - not a real-world result"
    lines = [f"# Live evaluation - {tag}\n", "| attempt | expected | WAIT at s | GO at s | outcome | predicted | confidence | top-5 hit | started before GO | early (WAIT) signs | delay s | dup | hand visible | reason |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['attempt']} | {r['expected']} | {r['ready_start_s']:.2f} | {r['start_s']:.2f} | {r['outcome']} | {r['predicted']} | "
                     f"{'' if r['confidence'] is None else round(r['confidence'], 3)} | {r['in_top5']} | "
                     f"{r['started_before_go']} | {r['early_labels']} | "
                     f"{'' if r['delay_s'] is None else round(r['delay_s'], 2)} | {r['duplicates']} | "
                     f"{r['hand_visible_fraction']:.2f} | {r['reason']} |")
    lines += ["", "```json", json.dumps({k: v for k, v in summary.items() if k != "per_sign_accuracy"}, indent=1, default=str), "```"]
    (out / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nsaved to {out.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()

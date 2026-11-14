"""Step 6.1: run the streaming sign recognizer on a video file or a webcam (prints one line per sign).

  PYTHONPATH=src .venv/bin/python scripts/s6_01_run_inference.py --video clip.mp4 [--model-dir models/candidates/<id>]
  PYTHONPATH=src .venv/bin/python scripts/s6_01_run_inference.py --camera 0 [--max-seconds 30]

Frames go through MediaPipe Holistic with the training settings, then the exact training
preprocessing (see asl.inference.recognizer). Isolated words only.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from asl.config import PROJECT_ROOT, STEP2_CONFIG, load_yaml
from asl.inference import SignRecognizer, StreamingRecognizer


def build(model_dir: Path, width: int, height: int, cfg: dict) -> StreamingRecognizer:
    rec = SignRecognizer.load(model_dir / "best_model.pt", model_dir / "class_mapping.json", load_yaml(STEP2_CONFIG),
                              load_yaml(PROJECT_ROOT / "configs" / "step3_sequences.yaml"), top_k=cfg["top_k"])
    return StreamingRecognizer(rec, width, height, **cfg["segmentation"], **cfg["postprocess"])


def report(ev) -> None:
    p = ev.prediction
    print(json.dumps({"start_s": round(ev.start_s, 2), "end_s": round(ev.end_s, 2), "frames": ev.frames,
                      "status": p.status, "label": p.label, "confidence": None if p.confidence is None else round(p.confidence, 3),
                      "emitted": ev.emitted, "reason": p.reason, "forced_cut": ev.forced,
                      "top3": [(l, round(c, 3)) for l, c in p.top_k[:3]]}))


def main() -> None:
    ap = argparse.ArgumentParser()
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--video")
    src.add_argument("--camera", type=int)
    ap.add_argument("--model-dir", default=None, help="default: model_dir from configs/inference.yaml")
    ap.add_argument("--max-seconds", type=float, default=60.0, help="camera only")
    args = ap.parse_args()
    cfg = load_yaml(PROJECT_ROOT / "configs" / "inference.yaml")
    model_dir = PROJECT_ROOT / (args.model_dir or cfg["model_dir"])
    if args.video:
        import av
        with av.open(args.video) as c:
            st = c.streams.video[0]
            sr = build(model_dir, st.codec_context.width, st.codec_context.height, cfg)
            fps = float(st.average_rate) if st.average_rate else 30.0
            for i, frame in enumerate(c.decode(st)):
                t = frame.time if frame.time is not None else i / fps
                ev = sr.push_rgb(frame.to_ndarray(format="rgb24"), t)
                if ev:
                    report(ev)
    else:
        import cv2
        cap = cv2.VideoCapture(args.camera)
        ok, frame = cap.read()
        if not ok:
            raise SystemExit(f"cannot read from camera {args.camera} (permission or device problem)")
        sr = build(model_dir, frame.shape[1], frame.shape[0], cfg)
        t0 = time.monotonic()
        while ok and time.monotonic() - t0 < args.max_seconds:
            ev = sr.push_rgb(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), time.monotonic() - t0)
            if ev:
                report(ev)
            ok, frame = cap.read()
        cap.release()
    ev = sr.flush()
    if ev:
        report(ev)
    sr.close()


if __name__ == "__main__":
    main()

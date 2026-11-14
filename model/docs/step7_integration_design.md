# Step 7 — Application integration design (proposal; not implemented)

Implementation is deferred until the webcam evaluation (docs/step6_webcam_evaluation.md) has been run
with an unseen signer. This document fixes the inference contract so the service and frontend can be
built and tested against it.

## What exists today (inspected)

- `src/components/Call.tsx`: one `RTCPeerConnection` per remote client (1:1), STUN plus optional
  TURN (short-lived coturn credentials from `/api/turn`, added after this document was written), local camera
  from `getUserMedia({audio: true, video: {ideal 1280×720, facingMode: "user"}})`, tracks added with
  `addTrack`; no `RTCDataChannel`. The local preview is a `<video>` element (not CSS-mirrored).
- Signaling: Next.js route `src/app/api/signaling/[roomId]` backed by MongoDB, polled every 1 s
  (offer/answer/ICE/leave).
- No FastAPI service exists in the repository yet; the recognizer is `asl.inference` (Python).

## Where inference runs and how data moves

Only the signer's own camera is needed, so recognition runs on the **signer's side**; the other peer
only needs the resulting words.

**Recommended (A): local inference service + existing peer connection for results.**

```
signer's browser                         signer's machine                    other peer
local MediaStream (already captured) ──frames (JPEG, ≤640×480, ≤30 fps)──▶ FastAPI /v1/sign-stream (WebSocket, localhost)
          ▲                                         │  MediaPipe + StreamingRecognizer (training preprocessing)
          └──────────── sign events (JSON) ◀────────┘
caption shown locally ──RTCDataChannel "captions" on the EXISTING RTCPeerConnection──▶ caption overlay
```

- No second video stream leaves the machine: frames are taken from the track the call already uses
  (draw to an `OffscreenCanvas` / `requestVideoFrameCallback`) and sent to localhost only; across the
  network only small caption messages travel, on a data channel added to the existing connection.
- Parity with training is guaranteed because the same Python MediaPipe + preprocessing is used
  (`holistic_result_to_frame` → `StreamingRecognizer`). Frames should be downscaled to ~640×480 (98 % of
  the dataset's EDA sample) — this needs a quick parity/accuracy check because JPEG compression and
  resolution differ from the dataset videos.
- Cost: a Python process per signing user; ~28.5 ms MediaPipe per frame on an M2 (≈35 fps max), shared
  with the browser's own video encoding.

Alternatives considered:
- (B) MediaPipe in the browser (JS) and landmarks to a hosted service: lighter, but a different MediaPipe
  build/model; landmark parity with the 0.10.21 Python Holistic is **not** guaranteed and would have to be
  measured on recorded clips (s6_02-style feature comparison) before use.
- (C) A server-side WebRTC peer (e.g. aiortc) receiving the video: sends a redundant video stream
  upstream and adds a media server; rejected for now.

## Inference contract — WebSocket `GET /v1/sign-stream`

Client → server
- Text `{"type": "start", "session_id": str, "frame_width": int, "frame_height": int,
  "settings": {"confidence_threshold"?: float, "smoothing_window"?: int, "smoothing_min_votes"?: int}}`
  — segmentation/post-processing overrides only; the model input contract is fixed by the checkpoint.
- Binary frame: 8-byte little-endian float64 capture timestamp in seconds (client monotonic clock,
  strictly increasing) followed by one JPEG image (RGB, the declared width/height).
- Text `{"type": "flush"}` (classify an open segment), `{"type": "stop"}`.

Server → client
- `{"type": "ready", "model": {"experiment_id", "num_classes", "window": 30, "feature_dim": 261,
  "vocabulary_sha256"}, "settings": {...}}`
- `{"type": "sign", "segment": {"start_s", "end_s", "last_hand_s", "frames", "forced_cut"},
  "status": "ok" | "low_confidence" | "rejected", "label": str | null, "confidence": float | null,
  "top_k": [[label, prob], ...], "emitted": str | null, "reason": str, "server_ms": float}` — one per
  detected segment; `emitted` is what a UI should display (after threshold and smoothing).
- `{"type": "stats", "fps_processed", "frames_dropped", "mediapipe_ms_median"}` every few seconds.
- `{"type": "error", "code": "bad_frame" | "bad_timestamp" | "bad_size" | "overloaded" | "model_unavailable",
  "message": str, "fatal": bool}` — bad frames are skipped (non-fatal); a model that fails to load makes
  `GET /v1/health` return 503 and the socket close with `model_unavailable`.

REST: `GET /v1/health` (model loaded, experiment id, versions), `GET /v1/labels` (index → word).

Back-pressure: the server keeps only the newest unprocessed frame (drops older ones and counts them),
like a live camera; it never queues unboundedly.

Peer caption message (data channel `captions`): `{"type": "caption", "word": str, "confidence": float,
"t": float}` for emitted words only.

Scope: isolated words, one per segment. No sentence building, grammar or translation; the UI should
label output as recognised signs with confidence, not as translated speech.

## Tests to write with the implementation

- Contract tests with FastAPI's TestClient: start/ready, a recorded clip sent as JPEG frames produces
  the same `sign` events as `s6_01 --video` on that clip (JPEG round-trip tolerance to be measured),
  bad timestamp / bad JPEG / wrong size → error messages, health 503 when the model cannot load,
  back-pressure drops counted.
- Frontend: data-channel caption round trip between two browser tabs.

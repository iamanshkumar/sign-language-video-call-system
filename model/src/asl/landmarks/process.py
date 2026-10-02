"""Per-video Step 2 pipeline (methodology v2): extract once -> 30 FPS -> full-clip metadata
-> active signing interval -> rejection rules -> gap handling -> shoulder normalization
-> compact storage. Returns one metadata record per video."""
from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np

from asl.landmarks import preprocess as pp
from asl.landmarks.extract import RawLandmarks, extract_landmarks
from asl.landmarks.layout import GROUP_INDEX, GROUP_ORDER, INTERPOLATED, OBSERVED, PRESENT, SLICES, feature_dim

ARRAY_VERSION = 2
HANDS = [GROUP_INDEX["left_hand"], GROUP_INDEX["right_hand"]]


def array_name(sample_id: str) -> str:
    """Filesystem-safe, stable file name (sample ids contain spaces, '•', '/')."""
    return hashlib.sha1(sample_id.encode("utf-8")).hexdigest()[:20] + ".npz"


def _per_landmark(mask_by_group: np.ndarray) -> np.ndarray:
    out = np.zeros((mask_by_group.shape[0], SLICES["right_hand"].stop), bool)
    for g in GROUP_ORDER:
        out[:, SLICES[g]] = mask_by_group[:, GROUP_INDEX[g]][:, None]
    return out


def coord_dtype(out_cfg: dict, group: str) -> np.dtype:
    """`output.coord_dtype` is one dtype for all groups or a {group: dtype} mapping."""
    d = out_cfg["coord_dtype"]
    return np.dtype(d[group] if isinstance(d, dict) else d)


def pack_arrays(norm: np.ndarray, present: np.ndarray, observed: np.ndarray, interpolated: np.ndarray,
                pose_vis: np.ndarray, center: np.ndarray, scale: np.ndarray, shoulder_measured: np.ndarray,
                interval: pp.ActiveInterval | None, out_cfg: dict) -> dict[str, np.ndarray]:
    """Build the storage-format (v2) arrays from float landmark data. norm: [T, 543, 3]."""
    flags = (observed.astype(np.uint8) * OBSERVED | present.astype(np.uint8) * PRESENT
             | interpolated.astype(np.uint8) * INTERPOLATED).astype(np.uint8)
    arrays: dict[str, np.ndarray] = {g: norm[:, SLICES[g]].astype(coord_dtype(out_cfg, g))
                                     for g in out_cfg["groups_to_store"]}
    arrays.update(
        pose_vis=np.clip(pose_vis, 0, 1).astype(coord_dtype(out_cfg, "pose")),
        flags=flags,
        center=center.astype(np.float32),
        scale=scale.astype(np.float32),
        shoulder_measured=shoulder_measured.astype(np.uint8),
        active_interval=np.array([interval.start, interval.end] if interval else [-1, -1], np.int32),
        version=np.array(ARRAY_VERSION, np.int32),
    )
    return arrays


def preprocess_raw(raw: RawLandmarks, cfg: dict) -> tuple[dict | None, dict]:
    """Apply Step 2 rules to extracted landmarks. Returns (arrays or None if not saved, stats)."""
    pre, rules, out_cfg = cfg["preprocessing"], cfg["rejection"], cfg["output"]
    order = np.argsort(raw.times, kind="stable")
    times = raw.times[order]
    n = len(times)
    span = float(times[-1] - times[0]) if n > 1 else 0.0
    fps = (n - 1) / span if span > 0 else (raw.container_fps or pre["target_fps"])
    # Full-video metadata (never altered by the active-interval logic).
    st: dict = {
        "original_fps": fps, "container_fps": raw.container_fps, "original_frame_count": n,
        "original_duration_s": n / fps, "width": raw.width, "height": raw.height, "codec": raw.codec,
        "nonfinite_detections_repaired": raw.repaired_nonfinite,
        "std_frame_count": pp.standardized_length(n, fps, pre["target_fps"]),
    }
    if n < rules["min_decoded_frames"]:
        return None, {**st, "rejection_reason": "too_few_decoded_frames"}

    coords = pp.to_pixel_space(raw.coords[order].astype(np.float64), raw.width, raw.height)
    _, c30, obs30, vis30 = pp.resample_uniform(times, coords, raw.observed[order],
                                               raw.pose_vis[order].astype(np.float64), fps, pre["target_fps"])
    t30 = len(c30)
    st["std_frame_count"] = t30
    for g in GROUP_ORDER:  # full clip
        miss = ~obs30[:, GROUP_INDEX[g]]
        st[f"{g}_missing_frames"] = int(miss.sum())
        st[f"{g}_missing_pct"] = 100.0 * float(miss.mean())
        st[f"{g}_longest_missing_run"] = pp.longest_run(miss)
    hands = obs30[:, HANDS]
    both = ~hands.any(axis=1)
    st["both_hands_missing_frames"] = int(both.sum())
    st["both_hands_missing_pct"] = 100.0 * float(both.mean())
    st["both_hands_missing_runs"] = json.dumps([[s0, ln] for s0, ln in pp.runs(both)])  # [start, length]
    st["longest_both_hands_missing_run"] = pp.longest_run(both)
    st["fullclip_rule_rejection"] = pp.fullclip_rejection_reason(n, hands, rules)  # v1, comparison only

    iv = pp.active_interval(hands, rules["active_interval_min_hand_run"])
    st.update(active_start_frame=iv.start if iv else None, active_end_frame=iv.end if iv else None,
              active_frames=iv.frames if iv else 0,
              active_start_s=iv.start / pre["target_fps"] if iv else None,
              active_end_s=(iv.end + 1) / pre["target_fps"] if iv else None,
              rest_frames_before=iv.start if iv else t30, rest_frames_after=t30 - 1 - iv.end if iv else t30)
    if iv:
        seg = obs30[iv.start : iv.end + 1]
        q = pp.hand_quality(seg[:, HANDS])
        st["active_both_hands_missing_pct"] = 100.0 * q["both_missing_frac"]
        st["active_longest_both_hands_missing_run"] = q["longest_both_missing_run"]
        st["active_left_hand_missing_pct"] = 100.0 * float((~seg[:, GROUP_INDEX["left_hand"]]).mean())
        st["active_right_hand_missing_pct"] = 100.0 * float((~seg[:, GROUP_INDEX["right_hand"]]).mean())
    st["short_active_interval"] = bool(iv is not None and iv.frames < rules["short_active_interval_frames"])

    reason = pp.rejection_reason(n, hands, rules)
    if reason and not (out_cfg.get("keep_arrays_for_hand_rejections", False) and iv is not None):
        return None, {**st, "rejection_reason": reason}

    present = np.zeros_like(obs30)
    interp = np.zeros_like(obs30)
    for g in GROUP_ORDER:
        gi, sl = GROUP_INDEX[g], SLICES[g]
        r = pp.fill_gaps(c30[:, sl], obs30[:, gi], pre["max_interp_gap"])
        c30[:, sl], present[:, gi], interp[:, gi] = r.coords, r.present, r.interpolated
        st[f"{g}_interpolated_frames"] = int(r.interpolated.sum())
        st[f"{g}_zero_filled_frames"] = int(r.zero_filled.sum())
    vis30 = pp.fill_gaps(vis30[:, :, None], obs30[:, GROUP_INDEX["pose"]], pre["max_interp_gap"]).coords[:, :, 0]

    bf = pp.body_frame(c30[:, SLICES["pose"]], present[:, GROUP_INDEX["pose"]], vis30, pre["shoulder_visibility_min"])
    st["shoulder_frames_measured"] = int(bf.measured.sum())
    st["shoulder_frames_carried"] = int((~bf.measured).sum())
    if not bf.ok:
        return None, {**st, "rejection_reason": reason or "normalization_no_valid_shoulders"}
    norm = pp.normalize(c30, _per_landmark(present), bf)
    if not (np.isfinite(norm).all() and np.isfinite(vis30).all()):
        return None, {**st, "rejection_reason": reason or "invalid_values_unrepairable"}
    arrays = pack_arrays(norm, present, obs30, interp, vis30, bf.center, bf.scale, bf.measured, iv, out_cfg)
    if not all(np.isfinite(arrays[g]).all() for g in out_cfg["groups_to_store"]):
        return None, {**st, "rejection_reason": reason or "invalid_values_unrepairable"}  # e.g. float16 overflow
    return arrays, {**st, "rejection_reason": reason}


def save_arrays(arrays: dict, path: Path, compress: bool = False) -> int:
    """Atomic write (tmp + rename) so an interrupted run never leaves a partial file."""
    tmp = path.with_suffix(".tmp.npz")
    (np.savez_compressed if compress else np.savez)(tmp, **arrays)
    os.replace(tmp, path)
    return path.stat().st_size


def process_video(video_path: str, info: dict, cfg: dict, arrays_dir: Path) -> dict:
    """info: sample_id, repo_path, word, split. Never raises; failures become rejections."""
    t0 = time.perf_counter()
    rec = {**info, "array_file": None, "bytes": 0, "rejected": True, "rejection_reason": None, "error": "",
           "format_version": ARRAY_VERSION, "coord_dtype": json.dumps(cfg["output"]["coord_dtype"]),
           "stored_groups": json.dumps(cfg["output"]["groups_to_store"]),
           "configurations": json.dumps({k: feature_dim(v) for k, v in cfg["configurations"].items()})}
    try:
        raw = extract_landmarks(video_path, cfg["mediapipe"])
    except Exception as e:
        rec.update(rejection_reason="decode_or_mediapipe_error", error=f"{type(e).__name__}: {e}"[:300],
                   total_s=time.perf_counter() - t0)
        return rec
    rec.update(decode_s=raw.decode_s, mediapipe_s=raw.mediapipe_s)
    t1 = time.perf_counter()
    try:
        arrays, st = preprocess_raw(raw, cfg)
    except Exception as e:
        rec.update(rejection_reason="preprocessing_error", error=f"{type(e).__name__}: {e}"[:300],
                   total_s=time.perf_counter() - t0)
        return rec
    rec.update(st)
    rec["rejected"] = st["rejection_reason"] is not None
    if arrays is not None:
        name = array_name(info["sample_id"])
        rec["bytes"] = save_arrays(arrays, arrays_dir / name, cfg["output"].get("compress", False))
        rec["array_file"] = name
    rec["array_kept_for_rejected"] = bool(rec["rejected"] and arrays is not None)
    rec["post_s"] = time.perf_counter() - t1
    rec["total_s"] = time.perf_counter() - t0
    return rec

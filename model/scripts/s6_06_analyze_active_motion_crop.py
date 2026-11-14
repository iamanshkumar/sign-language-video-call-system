"""Step 6.6 (OFFLINE EXPERIMENT ONLY): active-motion cropping of saved webcam segments.

Exploratory analysis, not part of the inference pipeline. Re-classifies each strictly scored answer
segment of recorded webcam sessions twice - unchanged (baseline) and with the hand-raising /
hand-lowering portions removed - using the unchanged checkpoint and training preprocessing. Method and
predeclared settings: reports/step6/experiments/active_motion_crop_20261020/METHOD.md; results of the
original run: RESULTS.md, per_attempt.csv, summary.json and crop_boundaries_<session>.png there.

  PYTHONPATH=src .venv/bin/python scripts/s6_06_analyze_active_motion_crop.py \
      --out reports/step6/experiments/active_motion_crop_20261020/verify_<tag>

Inputs (read only): the sessions' landmarks.npz, events.jsonl, attempts.csv and summary.json (default:
S3 = 20261019T084115Z_camera_S01, S4 = 20261020T045903Z_camera_S01; override with --session NAME=DIR),
and the checkpoint named in each session's summary.json (models/candidates/hands_pose_w30_bilstm_e100_s3).
Dependencies: the project's asl package, numpy, pandas, matplotlib, and scripts/s6_05_replay_session.py.
Output: --out (required; created; refused if it exists and is not empty, so earlier results are never
overwritten): per_attempt.csv, summary.json, crop_boundaries_<session>.png. Nothing else is written.

Checks reported in summary.json: the session replay reproduces every recorded event, each rebuilt
baseline segment reproduces its recorded label/confidence, and the no-crop control equals the baseline.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
from math import comb
from pathlib import Path

import numpy as np
import pandas as pd

from asl.config import PROJECT_ROOT, STEP2_CONFIG, load_yaml
from asl.inference import SignRecognizer, StreamingRecognizer
from asl.inference.evaluation import assign_events, attempt_row
from asl.landmarks.extract import RawLandmarks

DEFAULT_SESSIONS = {"S3": "reports/step6/webcam_eval/20261019T084115Z_camera_S01",
                    "S4": "reports/step6/webcam_eval/20261020T045903Z_camera_S01"}
THETAS = [0.5, 1.0, 2.0]          # vertical-speed thresholds, shoulder widths/s (predeclared, all reported)
LH, RH = slice(501, 522), slice(522, 543)
MIN_KEEP_S = 0.5                  # shorter crop -> crop_failed, baseline result kept
SMOOTH = 5                        # centered moving average, frames


def _replay_module():
    spec = importlib.util.spec_from_file_location("replay", PROJECT_ROOT / "scripts" / "s6_05_replay_session.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def segment_arrays(z, i0: int, i1: int):
    """Frames i0..i1 of a saved session after the recognizer's invalid-group repair."""
    c, o = [], []
    for i in range(i0, i1 + 1):
        oi = np.asarray(z["observed"][i], bool).copy()   # repaired in place
        ci, _ = StreamingRecognizer._drop_invalid_groups(np.asarray(z["coords"][i], np.float32).copy(), oi)
        c.append(ci)
        o.append(oi)
    v = np.asarray(z["pose_vis"][i0:i1 + 1], np.float32)
    return np.stack(c), np.stack(o), np.where(np.isfinite(v), v, 0.0).astype(np.float32)


def make_raw(z, t, c, o, v) -> RawLandmarks:
    return RawLandmarks(times=np.asarray(t, np.float64), coords=c, observed=o, pose_vis=v, repaired_nonfinite=0,
                        width=int(z["width"]), height=int(z["height"]), codec="live", container_fps=None,
                        decode_s=0.0, mediapipe_s=0.0)


def motion(width: int, height: int, t, c, o) -> dict:
    """Smoothed active-hand height (shoulder widths below the shoulder line) and its velocity over the
    segment's hand range [a, b]."""
    hands = o[:, 2] | o[:, 3]
    hi = np.where(hands)[0]
    a, b = hi[0], hi[-1]
    gi, sl = (3, RH) if o[a:b + 1, 3].sum() >= o[a:b + 1, 2].sum() else (2, LH)
    pose = o[:, 0]
    sw = np.median(np.abs(c[pose, 11, 0] - c[pose, 12, 0]))
    ysh = np.where(pose, (c[:, 11, 1] + c[:, 12, 1]) / 2, np.nan)
    ysh = pd.Series(ysh).interpolate(limit_direction="both").to_numpy()
    h = np.full(len(t), np.nan)
    m = o[:, gi]
    h[m] = (c[m, sl, 1].mean(axis=1) - ysh[m]) * height / (sw * width)
    hs = pd.Series(h, index=t).interpolate(method="index", limit_area="inside").to_numpy()
    seg = pd.Series(hs[a:b + 1].copy()).bfill().ffill().to_numpy()
    sm = np.convolve(np.pad(seg, SMOOTH // 2, mode="edge"), np.ones(SMOOTH) / SMOOTH, mode="valid")
    return dict(a=a, b=b, hand="right" if gi == 3 else "left", sm=sm, vel=np.gradient(sm, t[a:b + 1]), hands=hands)


def crop(mo: dict, theta: float) -> tuple[int, int]:
    """Frame indices [s, e] kept: initial rise (v < -theta) and final drop (v > +theta) removed."""
    a, b, vel, hands = mo["a"], mo["b"], mo["vel"], mo["hands"]
    n = b - a + 1
    s, e = 0, n - 1
    if math.isfinite(theta):
        if vel[0] < -theta:
            while s < n and vel[s] < -theta:
                s += 1
        if vel[-1] > theta:
            while e >= 0 and vel[e] > theta:
                e -= 1
    s, e = a + s, a + e
    while s <= e and not hands[s]:
        s += 1
    while e >= s and not hands[e]:
        e -= 1
    return s, e


def sign_test(k: int, n: int) -> float:
    if n == 0:
        return 1.0
    return min(1.0, sum(comb(n, i) for i in range(0, min(k, n - k) + 1)) / 2 ** n * 2)


def run(sessions: dict[str, Path], out: Path) -> dict:
    rp = _replay_module()
    rows, seg_store, parity = [], {}, {}
    rec = None
    for sname, S in sessions.items():
        events, _, attempts = rp.replay(S, {})
        recorded = [json.loads(l) for l in (S / "events.jsonl").read_text().splitlines()]
        parity[sname] = {"events": len(events), "replay_reproduces_recorded_events": rp.matches_recorded(events, recorded)}
        assign_events(attempts, events, 1.5)
        z = np.load(S / "landmarks.npz")
        T = z["t"]
        if rec is None:
            md = PROJECT_ROOT / json.loads((S / "summary.json").read_text())["model_dir"]
            rec = SignRecognizer.load(md / "best_model.pt", md / "class_mapping.json", load_yaml(STEP2_CONFIG),
                                      load_yaml(PROJECT_ROOT / "configs" / "step3_sequences.yaml"), top_k=5)
        for a in attempts:
            r = attempt_row(a)
            if r["status"] != "ok":
                rows.append(dict(session=sname, attempt=a.index, sign=a.expected, eligible=False, note=r["outcome"]))
                continue
            ev = next(e for e, _ in a.events if e.last_hand_s == r["last_hand_s"] and e.start_s == r["segment_start_s"])
            i0, i1 = int(np.searchsorted(T, ev.start_s)), int(np.searchsorted(T, ev.end_s, side="right")) - 1
            t = T[i0:i1 + 1]
            c, o, v = segment_arrays(z, i0, i1)
            base = rec.predict_raw(make_raw(z, t, c, o, v))
            same = base.label == ev.prediction.label and abs(base.confidence - ev.prediction.confidence) < 1e-5
            mo = motion(int(z["width"]), int(z["height"]), t, c, o)
            seg_store[(sname, a.index)] = dict(t=t, mo=mo, crops={})
            hand_dur = t[mo["b"]] - t[mo["a"]] + 1 / 30

            def pack(p, expected=a.expected):
                labs = [l for l, _ in p.top_k]
                return dict(label=p.label, conf=p.confidence, status=p.status, reason=p.reason,
                            correct=p.label == expected, top5=expected in labs,
                            rank=(labs.index(expected) + 1) if expected in labs else None)
            row = dict(session=sname, attempt=a.index, sign=a.expected, eligible=True, forced=ev.forced,
                       hand=mo["hand"], hand_range_s=round(hand_dur, 3), recorded_label=ev.prediction.label,
                       baseline_matches_recorded=bool(same), **{f"base_{k}": x for k, x in pack(base).items()})
            for th in [math.inf] + THETAS:
                s, e = crop(mo, th)
                keep = (t[e] - t[s] + 1 / 30) if e >= s else 0.0
                tag = "ctrl" if not math.isfinite(th) else f"th{th}"
                seg_store[(sname, a.index)]["crops"][tag] = (s, e)
                failed = keep < MIN_KEEP_S
                if failed:
                    p = base
                else:
                    o2, c2 = o.copy(), c.copy()
                    outside = np.ones(len(t), bool)
                    outside[s:e + 1] = False
                    o2[outside, 2] = o2[outside, 3] = False      # hands out of view outside the crop
                    c2[np.ix_(outside, np.r_[501:543])] = 0.0
                    p = rec.predict_raw(make_raw(z, t, c2, o2, v))
                sm, a0, b0 = mo["sm"], mo["a"], mo["b"]
                row.update({f"{tag}_cut_start_s": round(t[s] - t[a0], 3), f"{tag}_cut_end_s": round(t[b0] - t[e], 3),
                            f"{tag}_crop_s": round(keep, 3), f"{tag}_retained": round(keep / hand_dur, 3),
                            f"{tag}_crop_failed": failed,
                            f"{tag}_rise_ok": bool(sm[s - a0] <= sm[0] + 1e-9) if not failed else None,
                            f"{tag}_lower_ok": bool(sm[e - a0] <= sm[-1] + 1e-9) if not failed else None,
                            **{f"{tag}_{k}": x for k, x in pack(p).items()}})
            rows.append(row)

    df = pd.DataFrame(rows)
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "per_attempt.csv", index=False)
    el = df[df.eligible]
    summ = {"parity": parity, "eligible": int(len(el)),
            "ineligible": df[~df.eligible].reindex(columns=["session", "attempt", "note"]).to_dict("records"),
            "baseline_all_match_recorded": bool(el.baseline_matches_recorded.all()),
            "control_identical_to_baseline": bool((el.ctrl_label == el.base_label).all()
                                                  and np.allclose(el.ctrl_conf, el.base_conf, atol=1e-6)),
            "variants": {}}
    for tag in ["base", "ctrl"] + [f"th{t}" for t in THETAS]:
        d = {}
        for scope, g in [("all", el)] + [(s, g) for s, g in el.groupby("session")]:
            d[scope] = {"n": int(len(g)), "correct": int(g[f"{tag}_correct"].sum()), "top5": int(g[f"{tag}_top5"].sum()),
                        "rejected": int((g[f"{tag}_status"] != "ok").sum())}
            if tag.startswith("th"):
                w2c = int((~g.base_correct & g[f"{tag}_correct"]).sum())
                c2w = int((g.base_correct & ~g[f"{tag}_correct"]).sum())
                d[scope].update(wrong_to_correct=w2c, correct_to_wrong=c2w, sign_test_p=round(sign_test(w2c, w2c + c2w), 3),
                                label_changed=int((g[f"{tag}_label"] != g.base_label).sum()),
                                crop_failed=int(g[f"{tag}_crop_failed"].sum()),
                                uncropped=int(((g[f"{tag}_cut_start_s"] == 0) & (g[f"{tag}_cut_end_s"] == 0)).sum()),
                                crop_s_median=float(g[f"{tag}_crop_s"].median()),
                                retained_median=float(g[f"{tag}_retained"].median()),
                                boundary_checks_failed=int(((g[f"{tag}_rise_ok"] == False)  # noqa: E712
                                                            | (g[f"{tag}_lower_ok"] == False)).sum()))  # noqa: E712
        summ["variants"][tag] = d
    summ["correct_by_sign"] = {tag: el.groupby("sign")[f"{tag}_correct"].sum().astype(int).to_dict()
                               for tag in ["base"] + [f"th{t}" for t in THETAS]}
    (out / "summary.json").write_text(json.dumps(summ, indent=2, default=str) + "\n")
    _plots(seg_store, el, list(sessions), out)
    return summ


def _plots(seg_store: dict, el: pd.DataFrame, names: list[str], out: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    for sname in names:
        keys = sorted(k for k in seg_store if k[0] == sname)
        fig, axs = plt.subplots(5, 4, figsize=(16, 15))
        for ax, k in zip(axs.flat, keys):
            st = seg_store[k]
            t, mo = st["t"], st["mo"]
            ax.plot(t[mo["a"]:mo["b"] + 1] - t[mo["a"]], mo["sm"], color="k", lw=1.2)
            ax.invert_yaxis()
            for tag, col in [("th0.5", "tab:blue"), ("th1.0", "tab:orange"), ("th2.0", "tab:green")]:
                s, e = st["crops"][tag]
                ax.axvspan(t[s] - t[mo["a"]], t[e] - t[mo["a"]], color=col, alpha=0.12)
                ax.axvline(t[s] - t[mo["a"]], color=col, lw=0.8)
                ax.axvline(t[e] - t[mo["a"]], color=col, lw=0.8)
            r = el[(el.session == k[0]) & (el.attempt == k[1])].iloc[0]
            ax.set_title(f"{k[0]} #{k[1]} {r.sign} [{mo['hand'][0]}] base={r.base_label}", fontsize=8)
            ax.tick_params(labelsize=7)
        fig.suptitle(f"{sname}: active-hand height (shoulder widths below shoulder line; up = higher) vs time from "
                     "first hand frame; crop spans: blue th0.5, orange th1.0, green th2.0", fontsize=9)
        fig.tight_layout()
        fig.savefig(out / f"crop_boundaries_{sname}.png", dpi=90)
        plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", required=True, help="new or empty output folder (relative to the project root)")
    ap.add_argument("--session", action="append", default=None, metavar="NAME=DIR",
                    help="session to analyse (repeatable); default: S3 and S4")
    args = ap.parse_args()
    out = PROJECT_ROOT / args.out
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"{out} exists and is not empty - choose a new folder (results are never overwritten)")
    pairs = dict(s.split("=", 1) for s in args.session) if args.session else DEFAULT_SESSIONS
    sessions = {k: PROJECT_ROOT / v for k, v in pairs.items()}
    for S in sessions.values():
        for f in ("landmarks.npz", "events.jsonl", "attempts.csv", "summary.json"):
            if not (S / f).exists():
                raise SystemExit(f"{S / f} missing")
    summ = run(sessions, out)
    print(json.dumps({k: summ[k] for k in ("parity", "eligible", "baseline_all_match_recorded",
                                           "control_identical_to_baseline")}, indent=1))
    for tag, d in summ["variants"].items():
        print(tag, json.dumps(d["all"]))
    print(f"saved to {out.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()

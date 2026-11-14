"""E0.2 (OFFLINE): frozen webcam development replay set + paired evaluation of models on it.

  # build once (refuses to overwrite) and evaluate the 10 reference runs and the exported candidate
  PYTHONPATH=src .venv/bin/python scripts/e0_02_webcam_replay_set.py --out reports/e0/webcam_replay_set

One segment per attempt of the saved webcam sessions (the answer segment under the current strict
scoring; the segmenter does not depend on the classifier), frozen as frame ranges into each session's
landmarks.npz plus the [30, 261] features from the unchanged training preprocessing. Every model is then
evaluated on identical inputs, so comparisons are paired (exact McNemar on attempts). Attempts without a
segment count as wrong for every model. Development data only (one signer, already inspected):
docs/evaluation_protocol.md §4. Validity checks: the exported candidate reproduces the strict replay answer of every segment and
the live answers of sessions scored live with the strict rules; its source run reproduces the candidate. Inputs are read only; --out must be new or empty.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from math import comb
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from e0_common import SEEDS, load_run, logits, run_dirs  # noqa: E402

from asl.config import PROJECT_ROOT, STEP2_CONFIG, load_yaml  # noqa: E402
from asl.inference import SignRecognizer  # noqa: E402
from asl.inference.evaluation import assign_events, attempt_row  # noqa: E402

R = "reports/step6/webcam_eval/"
# session key -> (folder, condition label, core subset member)
SESSIONS = {
    "S2": ("20261019T075328Z_camera_S01", "natural (15 fps capture)", False),
    "S3": ("20261019T084115Z_camera_S01", "natural", True),
    "S4": ("20261020T045903Z_camera_S01", "controlled, sign once", True),
    "5a": ("20261020T063535Z_camera_S01", "protocol v1 SINGLE", True),
    "5b": ("20261020T063812Z_camera_S01", "protocol v1 REPEATED", True),
    "5c-1": ("20261020T064102Z_camera_S01", "protocol v1 REPEATED", True),
    "5c-2": ("20261020T064353Z_camera_S01", "protocol v1 REPEATED", True),
    "5d": ("20261020T064645Z_camera_S01", "protocol v1 SINGLE (24 fps, invalid block)", False),
}
CANDIDATE = "models/candidates/hands_pose_w30_bilstm_e100_s3"
# sessions whose live attempts.csv was scored with the current strict rules (S2, S3 were scored live with
# earlier rules; their current answers are checked against the replay instead)
STRICT_LIVE = ["S4", "5a", "5b", "5c-1", "5c-2", "5d"]


def _module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def mcnemar_exact(b: int, c: int) -> float:
    n = b + c
    return 1.0 if n == 0 else min(1.0, 2 * sum(comb(n, i) for i in range(0, min(b, c) + 1)) / 2 ** n)


def build(rec: SignRecognizer) -> tuple[pd.DataFrame, np.ndarray, dict]:
    s606 = _module(PROJECT_ROOT / "scripts" / "s6_06_analyze_active_motion_crop.py", "s606")
    rp = s606._replay_module()
    rows, feats, shas = [], [], {}
    for key, (folder, cond, core) in SESSIONS.items():
        S = PROJECT_ROOT / R / folder
        shas[key] = hashlib.sha256((S / "landmarks.npz").read_bytes()).hexdigest()
        events, _, attempts = rp.replay(S, {})
        recorded = [json.loads(l) for l in (S / "events.jsonl").read_text().splitlines()]
        if not rp.matches_recorded(events, recorded):
            raise RuntimeError(f"{key}: replay does not reproduce the recorded events")
        assign_events(attempts, events, 1.5)
        live = pd.read_csv(S / "attempts.csv")
        z = np.load(S / "landmarks.npz")
        T = z["t"]
        for a in attempts:
            r = attempt_row(a)
            lv = live[live.attempt == a.index].iloc[0]
            flags = []
            if r["early_events"] > 0:
                flags.append("signed_during_wait")
            if r["outcome"] == "pre_go_only":
                flags.append("pre_go")
            if "hands_down_at_go" in lv and str(lv.hands_down_at_go) == "False":
                flags.append("hands_not_down_at_go")
            row = dict(session=key, folder=folder, condition=cond, core=core, attempt=a.index, expected=a.expected,
                       excluded=";".join(flags), live_outcome=lv.outcome, live_predicted=lv.predicted,
                       has_segment=r["status"] == "ok", i0=-1, i1=-1, forced=r["forced_cut"], feature_row=-1)
            if r["status"] == "ok":
                ev = next(e for e, _ in a.events if e.last_hand_s == r["last_hand_s"] and e.start_s == r["segment_start_s"])
                i0, i1 = int(np.searchsorted(T, ev.start_s)), int(np.searchsorted(T, ev.end_s, side="right")) - 1
                c, o, v = s606.segment_arrays(z, i0, i1)
                seq, st = rec.features_from_raw(s606.make_raw(z, T[i0:i1 + 1], c, o, v))
                row.update(i0=i0, i1=i1, feature_row=len(feats), replay_predicted=r["predicted"])
                feats.append(seq)
            rows.append(row)
    return pd.DataFrame(rows), np.stack(feats).astype(np.float32), shas


def evaluate(man: pd.DataFrame, X: np.ndarray, model: torch.nn.Module, classes: list[str]) -> pd.DataFrame:
    lg = logits(model, X)
    top = np.argsort(-lg, axis=1)[:, :5]
    out = man[["session", "attempt", "expected", "core", "excluded", "has_segment"]].copy()
    pred, top5 = [], []
    for r in man.itertuples():
        if r.feature_row < 0:
            pred.append(None)
            top5.append(False)
        else:
            labs = [classes[k] for k in top[r.feature_row]]
            pred.append(labs[0])
            top5.append(r.expected in labs)
    out["predicted"], out["top5"] = pred, top5
    out["correct"] = out.predicted == out.expected
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", required=True)
    out = PROJECT_ROOT / ap.parse_args().out
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"{out} exists and is not empty (the replay set is frozen; use a new folder)")
    md = PROJECT_ROOT / CANDIDATE
    rec = SignRecognizer.load(md / "best_model.pt", md / "class_mapping.json", load_yaml(STEP2_CONFIG),
                              load_yaml(PROJECT_ROOT / "configs" / "step3_sequences.yaml"), top_k=5)
    man, X, shas = build(rec)
    classes = rec.classes
    # validity: the candidate reproduces the strict answer of every replayed segment, and the recorded live
    # answer of every session that was scored live with the current (strict) rules
    cand = evaluate(man, X, rec.model.eval(), classes)
    seg = man.has_segment.to_numpy()
    reproduces_replay = bool((cand.predicted[seg].to_numpy() == man.replay_predicted[seg].to_numpy()).all())
    live_ans = man.live_outcome.isin(["correct", "wrong"]) & man.session.isin(STRICT_LIVE)
    reproduces = bool((cand.predicted[live_ans].to_numpy() == man.live_predicted[live_ans].to_numpy()).all())
    out.mkdir(parents=True, exist_ok=True)
    man.to_csv(out / "manifest.csv", index=False)
    np.savez_compressed(out / "features.npz", X=X)
    results, preds = [], {"candidate_bilstm_s3": cand}
    for m in ("lstm", "bilstm"):
        for s, d in run_dirs(m).items():
            model, meta = load_run(d)
            preds[f"{m}_s{s}"] = evaluate(man, X, model, classes)
    same_as_candidate = bool((preds["bilstm_s3"].predicted.fillna("") == cand.predicted.fillna("")).all())
    for name, p in preds.items():
        for scope, mask in (("core_valid", p.core & (p.excluded.fillna("") == "")), ("all", np.ones(len(p), bool))):
            q = p[mask]
            results.append(dict(model=name, scope=scope, attempts=len(q), correct=int(q.correct.sum()),
                                top1=q.correct.mean(), top5=q.top5.mean()))
        p.to_csv(out / f"predictions_{name}.csv", index=False)
    res = pd.DataFrame(results)
    res.to_csv(out / "summary_by_model.csv", index=False)
    core = preds["candidate_bilstm_s3"].core & (preds["candidate_bilstm_s3"].excluded.fillna("") == "")
    paired = []
    for s in SEEDS:
        b, l = preds[f"bilstm_s{s}"][core].correct.to_numpy(), preds[f"lstm_s{s}"][core].correct.to_numpy()
        paired.append(dict(seed=s, bilstm_only=int((b & ~l).sum()), lstm_only=int((l & ~b).sum()),
                           exact_mcnemar_p=mcnemar_exact(int((b & ~l).sum()), int((l & ~b).sum()))))
    pd.DataFrame(paired).to_csv(out / "bilstm_vs_lstm_paired.csv", index=False)
    by_session = pd.concat([p.assign(model=n) for n, p in preds.items()]).groupby(["model", "session"]).correct.mean().unstack()
    by_session.to_csv(out / "top1_by_session.csv")
    summary = {"sessions": {k: {"folder": v[0], "condition": v[1], "core": v[2], "landmarks_sha256": shas[k]} for k, v in SESSIONS.items()},
               "attempts": len(man), "attempts_with_segment": int(man.has_segment.sum()),
               "core_valid_attempts": int(core.sum()),
               "candidate_reproduces_strict_replay_answers": reproduces_replay,
               "candidate_reproduces_live_answers_of_strict_sessions": reproduces, "strict_live_sessions": STRICT_LIVE, "bilstm_s3_run_equals_candidate": same_as_candidate,
               "status": "development data: one signer (S01), already inspected; secondary evidence only"}
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=1))
    print(res.to_string(index=False))
    print(f"saved to {out.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()

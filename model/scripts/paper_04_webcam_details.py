"""Paper details from saved single-signer webcam artifacts (no model training, no test data).

  PYTHONPATH=src .venv/bin/python scripts/paper_04_webcam_details.py --out reports/paper/draft/webcam_details

1. Cause of each duplicate emission in the default post-processing replay (reports/paper/postprocessing):
   the sessions are replayed with the live model and default settings; a duplicate is a continuation when the
   answer segment was cut at the segment cap and the duplicate's hand run started inside that segment.
2. Per-sign correct counts on the frozen webcam replay set for all 20 models (LSTM, BiLSTM baseline, E1, E2;
   5 seeds each), from the saved prediction files.
--out must be new or empty.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import pandas as pd

from asl.config import PROJECT_ROOT
from asl.inference.evaluation import _signing_start, assign_events

R = PROJECT_ROOT / "reports"
SEEDS = [42, 1, 2, 3, 4]


def _module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def duplicate_causes() -> pd.DataFrame:
    rp = _module(PROJECT_ROOT / "scripts/s6_05_replay_session.py", "rp")
    e02 = _module(PROJECT_ROOT / "scripts/e0_02_webcam_replay_set.py", "e02")
    pp = pd.read_csv(R / "paper/postprocessing/per_attempt.csv")
    d = pp[(pp.cooldown_s == 0) & (pp.confidence_threshold == 0) & pp.excluded.isna() & (pp.duplicate_emissions > 0)]
    rows = []
    for key, g in d.groupby("session"):
        events, _, attempts = rp.replay(PROJECT_ROOT / e02.R / e02.SESSIONS[key][0], {})
        assign_events(attempts, events, 1.5)
        for a in attempts:
            if a.index not in set(g.attempt):
                continue
            em = [e for e, _ in a.events if _signing_start(e) >= a.start_s and e.emitted is not None]
            first, dup = em[0], em[1]
            cont = bool(first.forced and dup.run_start_s is not None and dup.run_start_s < dup.first_hand_s - 1e-9)
            rows.append(dict(session=key, attempt=a.index, answer_forced_cut=first.forced, continuation_of_cut_sign=cont,
                             gap_after_answer_last_hand_s=round(dup.first_hand_s - first.last_hand_s, 2),
                             answer=first.emitted, duplicate=dup.emitted))
    return pd.DataFrame(rows)


def per_sign_counts() -> pd.DataFrame:
    man = pd.read_csv(R / "e0/webcam_replay_set/manifest.csv")
    core = (man.core & man.excluded.isna()).to_numpy()
    files = {"LSTM": "e0/webcam_replay_set/predictions_lstm_s{}.csv", "BiLSTM": "e0/webcam_replay_set/predictions_bilstm_s{}.csv",
             "E1": "e1/comparison/webcam_predictions_e1_s{}.csv", "E2": "e2/comparison/webcam_predictions_e2_s{}.csv"}
    rows = []
    for m, f in files.items():
        for s in SEEDS:
            p = pd.read_csv(R / f.format(s))[core]
            for sign, g in p.groupby("expected"):
                rows.append(dict(model=m, seed=s, sign=sign, attempts=len(g), correct=int(g.correct.astype(bool).sum())))
    t = pd.DataFrame(rows)
    return t.groupby(["sign", "model"]).agg(model_attempts=("attempts", "sum"), correct=("correct", "sum")).reset_index()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", required=True)
    out = PROJECT_ROOT / ap.parse_args().out
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"{out} exists and is not empty")
    out.mkdir(parents=True, exist_ok=True)
    dc = duplicate_causes()
    dc.to_csv(out / "duplicate_causes.csv", index=False)
    ps = per_sign_counts()
    ps.to_csv(out / "per_sign_correct_20_models.csv", index=False)
    summary = {"duplicate_emissions": len(dc), "continuations_of_cut_sign": int(dc.continuation_of_cut_sign.sum()),
               "other": dc[~dc.continuation_of_cut_sign][["session", "attempt", "gap_after_answer_last_hand_s"]].to_dict("records"),
               "never_or_rarely_recognised": ps[ps.sign.isin(["africa", "computer", "forbid"])].groupby("sign")
               .agg(model_attempts=("model_attempts", "sum"), correct=("correct", "sum")).to_dict("index")}
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=int) + "\n")
    print(json.dumps(summary, indent=1, default=int))


if __name__ == "__main__":
    main()

"""Step 6.4: confidence calibration of an exported model on the stored dev100 sequences.

  PYTHONPATH=src .venv/bin/python scripts/s6_04_confidence_calibration.py --model-dir models/candidates/<id>

Reports, for validation and test separately: expected calibration error (15 equal-width bins),
mean confidence of correct vs wrong predictions, and accuracy / coverage when predictions below a
confidence threshold are withheld. A suggested threshold is derived from VALIDATION ONLY (lowest
threshold whose accepted validation predictions reach --target-precision); test numbers at that
threshold are reported as a check, never used to choose it. Nothing in configs/ is changed.
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd
import torch

from asl.config import PROJECT_ROOT, STEP2_CONFIG, load_yaml
from asl.inference import SignRecognizer
from asl.landmarks.storage import available_configurations
from asl.sequences.dataset import SequenceDataset
from asl.sequences.labels import load_label_mapping

THRESHOLDS = [0.0, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


def ece(conf: np.ndarray, correct: np.ndarray, bins: int = 15) -> float:
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(conf, edges[1:-1]), 0, bins - 1)
    return float(sum(abs(correct[idx == b].mean() - conf[idx == b].mean()) * (idx == b).mean()
                     for b in range(bins) if (idx == b).any()))


def coverage_table(conf, correct, thresholds=THRESHOLDS) -> list[dict]:
    rows = []
    for t in thresholds:
        keep = conf >= t
        rows.append({"threshold": t, "coverage": float(keep.mean()),
                     "accuracy_of_accepted": float(correct[keep].mean()) if keep.any() else None,
                     "wrong_accepted_per_100": float(100 * (keep & ~correct).mean())})
    return rows


def suggest_threshold(conf, correct, target: float) -> float | None:
    for t in np.round(np.arange(0.0, 1.0, 0.01), 2):
        keep = conf >= t
        if keep.any() and correct[keep].mean() >= target:
            return float(t)
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--target-precision", type=float, default=0.80)
    args = ap.parse_args()
    c2, c3 = load_yaml(STEP2_CONFIG), load_yaml(PROJECT_ROOT / "configs" / "step3_sequences.yaml")
    md = PROJECT_ROOT / args.model_dir
    rec = SignRecognizer.load(md / "best_model.pt", md / "class_mapping.json", c2, c3)
    st = c3["stores"][rec.info["store"]]
    configs = available_configurations(c2["configurations"], rec.info["preprocessing"]["stored_groups"])
    index = pd.read_parquet(PROJECT_ROOT / st["index_dir"] / "index.parquet")
    mapping = load_label_mapping(PROJECT_ROOT / st["label_mapping"])
    res = {}
    for split in ("val", "test"):
        ds = SequenceDataset(index, PROJECT_ROOT / st["landmarks_dir"] / "arrays", split, rec.window,
                             rec.info["feature_config"], configs, mapping)
        x = torch.stack([ds[i][0] for i in range(len(ds))])
        y = ds.rows["label"].to_numpy()
        with torch.no_grad():
            prob = torch.softmax(rec.model(x), 1).numpy()
        conf, pred = prob.max(1), prob.argmax(1)
        correct = pred == y
        res[split] = {"n": len(y), "accuracy": float(correct.mean()), "ece_15_bins": ece(conf, correct),
                      "mean_confidence": float(conf.mean()),
                      "median_conf_correct": float(np.median(conf[correct])),
                      "median_conf_wrong": float(np.median(conf[~correct])),
                      "wrong_with_conf_ge_0.8": int((~correct & (conf >= 0.8)).sum()),
                      "coverage": coverage_table(conf, correct), "_conf": conf, "_correct": correct}
    t = suggest_threshold(res["val"]["_conf"], res["val"]["_correct"], args.target_precision)
    check = None
    if t is not None:
        keep = res["test"]["_conf"] >= t
        check = {"threshold": t, "test_coverage": float(keep.mean()),
                 "test_accuracy_of_accepted": float(res["test"]["_correct"][keep].mean()) if keep.any() else None}
    for s in res.values():
        s.pop("_conf"), s.pop("_correct")
    out = {"model": rec.info.get("experiment_id"), "target_precision": args.target_precision,
           "suggested_threshold_from_validation": t, "test_check_at_that_threshold": check,
           "splits": res, "note": "threshold chosen on validation only; configs/inference.yaml unchanged"}
    rep = PROJECT_ROOT / "reports" / "step6"
    rep.mkdir(parents=True, exist_ok=True)
    (rep / f"calibration_{out['model']}.json").write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()

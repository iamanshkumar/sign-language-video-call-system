"""Step 4.1: smoke test of the training/evaluation pipeline on a SYNTHETIC store.

Uses the real 2,207-class label mapping (models have 2,207 outputs) with 10 vocabulary
words as synthetic classes that have train/val/test samples. Checks, for all 3
configurations x 3 windows, Dataset -> DataLoader -> forward -> loss -> backward; then runs
full LSTM, BiLSTM and RF experiments (training, validation selection, checkpoint reload,
test evaluation, latency), repeats one experiment to check reproducibility, and writes
reports/step4/smoke/smoke_report.json. Nothing here is a real result.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone

import pandas as pd
import torch

from asl.config import PROJECT_ROOT, STEP2_CONFIG, load_yaml
from asl.landmarks.layout import feature_dim
from asl.models.experiment import ExperimentSpec, run_experiment
from asl.models.recurrent import build_model
from asl.models.train import load_checkpoint, set_seed
from asl.sequences.dataset import SequenceDataset, make_dataloaders
from asl.sequences.labels import load_label_mapping
from asl.sequences.synthetic import make_synthetic_store

STEP3, STEP4 = PROJECT_ROOT / "configs" / "step3_sequences.yaml", PROJECT_ROOT / "configs" / "step4_training.yaml"


def main() -> None:
    c2, c3, c4 = load_yaml(STEP2_CONFIG), load_yaml(STEP3), load_yaml(STEP4)
    mapping = load_label_mapping(PROJECT_ROOT / c3["labels"]["mapping_path"])
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    root = PROJECT_ROOT / c4["output"]["experiments_dir"]
    store = f"smoke_synthetic_{stamp}"
    store_dir = root / "_stores" / store
    words = sorted(mapping["word_to_index"])[:10]
    index, arrays_dir = make_synthetic_store(store_dir, words, mapping, c2["output"], c3["sequences"]["windows"], seed=0)
    configs = c2["configurations"]
    report: dict = {"store": store, "words": words, "vocabulary_size": mapping["num_classes"],
                    "samples": index["split"].value_counts().to_dict(),
                    "short_active": int(index["short_active_interval"].sum()), "forward_backward": {}}

    # 1. Dataset -> DataLoader -> forward -> loss -> backward for every config x window x model
    set_seed(c4["seed"])
    for fc in configs:
        for w in c3["sequences"]["windows"]:
            ds = SequenceDataset(index, arrays_dir, "train", w, fc, configs, mapping)
            x, y = next(iter(make_dataloaders({"train": ds}, 8, 0, c4["seed"])["train"]))
            for name in ("lstm", "bilstm"):
                m = build_model(name, feature_dim(configs[fc]), mapping["num_classes"], c4["neural"])
                logits = m(x)
                loss = torch.nn.functional.cross_entropy(logits, y)
                loss.backward()
                grads = all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters())
                report["forward_backward"][f"{fc}_w{w}_{name}"] = {
                    "x": list(x.shape), "logits": list(logits.shape), "loss": float(loss), "finite_gradients": bool(grads)}

    # 2. full experiments (reduced epochs: smoke only)
    results_csv = PROJECT_ROOT / c4["output"]["reports_dir"] / "smoke" / "smoke_results.csv"
    runs = {}
    for spec in (ExperimentSpec("lstm", "hands", 30), ExperimentSpec("bilstm", "hands_pose", 45),
                 ExperimentSpec("rf", "hands_pose_face", 60)):
        r = run_experiment(spec, store=store, index=index, arrays_dir=arrays_dir, mapping=mapping,
                           configurations=configs, cfg=c4, experiments_root=root, results_csv=results_csv,
                           max_epochs=12, notes="SMOKE TEST on synthetic data (max_epochs=12)")
        runs[spec.experiment_id] = {"exp_dir": str(r["exp_dir"].relative_to(PROJECT_ROOT)), **r["row"],
                                    "evaluations": r["meta"]["evaluations"],
                                    "checkpoint_reload_reproduces_val_f1": r["meta"].get("checkpoint_reload_reproduces_val_f1"),
                                    "latency": r["meta"]["latency"], "train_seconds": r["meta"]["train_seconds"]}
    report["experiments"] = runs

    # 3. reproducibility: same experiment again in a second store name -> identical history and metrics
    again = run_experiment(ExperimentSpec("lstm", "hands", 30), store=store + "_repeat", index=index,
                           arrays_dir=arrays_dir, mapping=mapping, configurations=configs, cfg=c4,
                           experiments_root=root, results_csv=results_csv, max_epochs=12,
                           notes="SMOKE TEST reproducibility repeat")
    h1 = pd.read_csv(root / store / "hands_w30_lstm" / "history.csv").drop(columns="seconds")
    h2 = pd.read_csv(again["exp_dir"] / "history.csv").drop(columns="seconds")
    report["reproducible_history"] = bool(h1.equals(h2))
    report["reproducible_test_metrics"] = (json.loads((root / store / "hands_w30_lstm" / "metrics_test.json").read_text())
                                           == json.loads((again["exp_dir"] / "metrics_test.json").read_text()))
    # 4. checkpoint is reloadable into a fresh model
    m = build_model("lstm", feature_dim(configs["hands"]), mapping["num_classes"], c4["neural"])
    ck = load_checkpoint(root / store / "hands_w30_lstm" / "best.pt", m)
    report["checkpoint_reload"] = {"epoch": ck["epoch"], "val_macro_f1": ck["val_macro_f1"],
                                   "out_features": m.fc.out_features}
    out = PROJECT_ROOT / c4["output"]["reports_dir"] / "smoke"
    out.mkdir(parents=True, exist_ok=True)
    (out / "smoke_report.json").write_text(json.dumps(report, indent=2, default=str) + "\n")
    print(json.dumps({k: report[k] for k in ("store", "samples", "short_active", "reproducible_history",
                                             "reproducible_test_metrics", "checkpoint_reload")}, indent=2, default=str))
    fb = report["forward_backward"]
    print(f"forward/backward: {sum(v['finite_gradients'] for v in fb.values())}/{len(fb)} ok; "
          f"logits {fb['hands_pose_face_w60_bilstm']['logits']}")
    for k, v in runs.items():
        print(k, {kk: v[kk] for kk in ("epochs", "best_epoch", "best_val_macro_f1", "test_accuracy", "test_macro_f1",
                                       "mean_latency_ms", "fps", "checkpoint_reload_reproduces_val_f1")},
              f"train {v['train_seconds']:.1f}s")


if __name__ == "__main__":
    main()

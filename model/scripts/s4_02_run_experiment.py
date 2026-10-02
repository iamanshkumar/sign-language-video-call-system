"""Step 4.2: run ONE experiment (there is deliberately no "run the whole matrix" mode).

Example: --store benchmark_v2_mixed --model lstm --feature-config hands --window 30
A store without validation data is refused unless --allow-no-validation is given; such a
run has no model selection and reports train-fit metrics only (never generalization).
"""
from __future__ import annotations

import argparse
import json

import pandas as pd

from asl.config import PROJECT_ROOT, STEP2_CONFIG, load_yaml
from asl.landmarks.pipeline import ResourceMonitor
from asl.models.experiment import ExperimentSpec, run_experiment
from asl.sequences.labels import load_label_mapping

STEP3, STEP4 = PROJECT_ROOT / "configs" / "step3_sequences.yaml", PROJECT_ROOT / "configs" / "step4_training.yaml"


def main() -> None:
    c2, c3, c4 = load_yaml(STEP2_CONFIG), load_yaml(STEP3), load_yaml(STEP4)
    m = c4["experiment_matrix"]
    ap = argparse.ArgumentParser()
    ap.add_argument("--store", required=True, choices=sorted(c3["stores"]))
    ap.add_argument("--model", required=True, choices=m["neural_models"] + (["rf"] if m["rf_baseline"] else []))
    ap.add_argument("--feature-config", required=True, choices=m["feature_configs"])
    ap.add_argument("--window", required=True, type=int, choices=m["windows"])
    ap.add_argument("--allow-no-validation", action="store_true")
    ap.add_argument("--notes", default="")
    args = ap.parse_args()

    st = c3["stores"][args.store]
    index = pd.read_parquet(PROJECT_ROOT / st["index_dir"] / "index.parquet")
    mapping = load_label_mapping(PROJECT_ROOT / c3["labels"]["mapping_path"])
    spec = ExperimentSpec(args.model, args.feature_config, args.window)
    with ResourceMonitor() as mon:
        r = run_experiment(spec, store=args.store, index=index, arrays_dir=PROJECT_ROOT / st["landmarks_dir"] / "arrays",
                           mapping=mapping, configurations=c2["configurations"], cfg=c4,
                           experiments_root=PROJECT_ROOT / c4["output"]["experiments_dir"],
                           results_csv=PROJECT_ROOT / c4["output"]["reports_dir"] / "results" / "experiment_results.csv",
                           allow_no_validation=args.allow_no_validation, notes=args.notes)
    meta = json.loads((r["exp_dir"] / "metadata.json").read_text())
    meta["resources"] = mon.summary()
    (r["exp_dir"] / "metadata.json").write_text(json.dumps(meta, indent=2, default=str) + "\n")
    print(json.dumps({"exp_dir": str(r["exp_dir"].relative_to(PROJECT_ROOT)), "row": r["row"],
                      "evaluations": meta["evaluations"], "train_seconds": meta["train_seconds"],
                      "resources": meta["resources"], "latency": meta["latency"]}, indent=2, default=str))


if __name__ == "__main__":
    main()

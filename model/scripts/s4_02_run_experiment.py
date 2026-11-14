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
from asl.landmarks.pipeline import load_metadata
from asl.landmarks.storage import available_configurations, store_groups
from asl.models.experiment import ExperimentSpec, export_model, preflight, run_experiment
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
    ap.add_argument("--max-epochs", type=int, default=None, help="override neural.max_epochs (default: config)")
    ap.add_argument("--tag", default="", help="variant name appended to the experiment id (own directory)")
    ap.add_argument("--dropout", type=float, default=None, help="override neural.dropout (default: config)")
    ap.add_argument("--seed", type=int, default=None,
                    help="override the seed (weights init, dropout, training shuffle order); default: config")
    ap.add_argument("--augment", choices=["none", "temporal"], default="none",
                    help="training-only augmentation (asl.sequences.augment); recorded in config.json/metadata")
    ap.add_argument("--no-test-eval", action="store_true",
                    help="build no test Dataset and run no test inference (development runs; docs/evaluation_protocol.md)")
    ap.add_argument("--drop-pose-landmarks", default=None, metavar="A-B",
                    help="feature ablation: remove the x, y, z and visibility columns of pose landmarks A..B "
                         "(e.g. 23-32 = hips, knees, ankles, feet) for every split; recorded in config.json")
    ap.add_argument("--export-dir", default=None,
                    help="after the run, write <dir>/best_model.pt and <dir>/class_mapping.json (neural models)")
    args = ap.parse_args()

    if args.dropout is not None:  # the override is recorded in the experiment's config.json / metadata
        c4 = {**c4, "neural": {**c4["neural"], "dropout": args.dropout}}
    if args.seed is not None:  # recorded in config.json / metadata / results row like the default seed
        c4 = {**c4, "seed": args.seed}
    augment = None
    if args.augment == "temporal":
        from asl.sequences.augment import TemporalAugment, TemporalAugmentConfig
        augment = TemporalAugment(TemporalAugmentConfig(), seed=c4["seed"])
        c4 = {**c4, "train_augmentation": augment.describe()}
    if args.no_test_eval:
        c4 = {**c4, "evaluate_test": False}
    drop_columns: list[int] = []
    if args.drop_pose_landmarks:
        from asl.landmarks.layout import pose_landmark_columns
        a, b = (int(x) for x in args.drop_pose_landmarks.split("-"))
        drop_columns = pose_landmark_columns(c2["configurations"][args.feature_config], range(a, b + 1))
        c4 = {**c4, "feature_ablation": {"drop_pose_landmarks": [a, b], "dropped_columns": drop_columns}}
    st = c3["stores"][args.store]
    index = pd.read_parquet(PROJECT_ROOT / st["index_dir"] / "index.parquet")
    # a store may carry its own (subset) label mapping, e.g. the 100-class dev vocabulary
    mapping = load_label_mapping(PROJECT_ROOT / st.get("label_mapping", c3["labels"]["mapping_path"]))
    arrays_dir = PROJECT_ROOT / st["landmarks_dir"] / "arrays"
    configs = available_configurations(c2["configurations"],
                                       store_groups(load_metadata(PROJECT_ROOT / st["landmarks_dir"])))
    spec = ExperimentSpec(args.model, args.feature_config, args.window)
    if spec.feature_config not in configs:
        raise SystemExit(f"store {args.store} does not hold the groups of {spec.feature_config}; available: {list(configs)}")
    if index["split"].eq("val").any():  # train-only pipeline checks have nothing to preflight against
        pre = preflight(index, arrays_dir, spec, configs, mapping, drop_columns=drop_columns)
        print("preflight:", json.dumps(pre))
    with ResourceMonitor() as mon:
        r = run_experiment(spec, store=args.store, index=index, arrays_dir=arrays_dir,
                           mapping=mapping, configurations=configs, cfg=c4,
                           experiments_root=PROJECT_ROOT / c4["output"]["experiments_dir"],
                           results_csv=PROJECT_ROOT / c4["output"]["reports_dir"] / "results" / "experiment_results.csv",
                           allow_no_validation=args.allow_no_validation, notes=args.notes,
                           max_epochs=args.max_epochs, tag=args.tag, train_augment=augment,
                           evaluate_test=not args.no_test_eval, drop_columns=drop_columns)
    meta = json.loads((r["exp_dir"] / "metadata.json").read_text())
    meta["resources"] = mon.summary()
    (r["exp_dir"] / "metadata.json").write_text(json.dumps(meta, indent=2, default=str) + "\n")
    print(json.dumps({"exp_dir": str(r["exp_dir"].relative_to(PROJECT_ROOT)), "row": r["row"],
                      "evaluations": meta["evaluations"], "train_seconds": meta["train_seconds"],
                      "resources": meta["resources"], "latency": meta["latency"]}, indent=2, default=str))
    if args.export_dir:
        info = export_model(r["exp_dir"], PROJECT_ROOT / args.export_dir, mapping, configs)
        print("exported:", json.dumps(info, default=str))


if __name__ == "__main__":
    main()

"""Step 4.4: export an already-trained experiment as a validated inference CANDIDATE (no retraining).

  PYTHONPATH=src .venv/bin/python scripts/s4_04_export_model.py \
      --experiment experiments/dev100_hp/hands_pose_w30_bilstm_e100

Writes models/candidates/<experiment_id>/{best_model.pt, class_mapping.json, export_report.json}.
Never touches models/best_model.pt (promotion is scripts/s4_05_promote_model.py). Before export it
checks the architecture, feature count, class count, label mapping and preprocessing contract;
after export it reloads the candidate through the inference loader and requires its predictions
to equal the original checkpoint's. The candidate directory appears only if every check passes.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from asl.config import PROJECT_ROOT, STEP2_CONFIG, load_yaml
from asl.inference.recognizer import SignRecognizer
from asl.inference.spec import preprocessing_spec, spec_digest
from asl.landmarks.layout import feature_dim
from asl.landmarks.pipeline import load_metadata
from asl.landmarks.storage import available_configurations, store_groups
from asl.models.experiment import export_model
from asl.models.recurrent import build_model
from asl.models.train import load_checkpoint
from asl.sequences.dataset import SequenceDataset
from asl.sequences.labels import load_label_mapping

STEP3, STEP4 = PROJECT_ROOT / "configs" / "step3_sequences.yaml", PROJECT_ROOT / "configs" / "step4_training.yaml"


def profile_for_store(meta: pd.DataFrame, step2: dict) -> str | None:
    """Name of the storage profile the store was written with (None = default output config)."""
    acc = meta[meta["array_file"].map(lambda v: isinstance(v, str) and v != "")]
    groups = store_groups(meta)
    dtypes = {json.dumps(json.loads(d), sort_keys=True) for d in acc["coord_dtype"]}
    if len(dtypes) != 1:
        raise ValueError(f"store mixes coordinate dtypes: {dtypes}")
    dtype = json.loads(dtypes.pop())
    for name, prof in step2.get("storage_profiles", {}).items():
        if list(prof["groups_to_store"]) == groups and prof["coord_dtype"] == dtype:
            return name
    if list(step2["output"]["groups_to_store"]) == groups and step2["output"]["coord_dtype"] == dtype:
        return None
    raise ValueError(f"store layout {groups} / {dtype} matches no configured storage profile")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiment", required=True, help="experiment directory, e.g. experiments/dev100_hp/<id>")
    ap.add_argument("--out-dir", default=None, help="default: models/candidates/<experiment_id>")
    ap.add_argument("--check-samples", type=int, default=64, help="real test sequences for the logits check")
    args = ap.parse_args()
    c2, c3 = load_yaml(STEP2_CONFIG), load_yaml(STEP3)
    exp = (PROJECT_ROOT / args.experiment).resolve()
    meta = json.loads((exp / "metadata.json").read_text())
    out = PROJECT_ROOT / (args.out_dir or f"models/candidates/{meta['experiment_id']}")
    if out.exists():
        raise SystemExit(f"{out} exists; candidates are never overwritten")
    st = c3["stores"][meta["store"]]
    mapping = load_label_mapping(PROJECT_ROOT / st.get("label_mapping", c3["labels"]["mapping_path"]))
    lmeta = load_metadata(PROJECT_ROOT / st["landmarks_dir"])
    configs = available_configurations(c2["configurations"], store_groups(lmeta))
    profile = profile_for_store(lmeta, c2)
    fc, window = meta["feature_config"], int(meta["window_size"])

    # ---------------------------------------------------------------- checks before export
    pre = {
        "neural_model": meta["model"] in ("lstm", "bilstm"),
        "selected_on_validation": meta.get("checkpoint") == "best.pt" and meta.get("selection") == "best validation macro-F1",
        "checkpoint_reload_reproduced_val_f1": bool(meta.get("checkpoint_reload_reproduces_val_f1")),
        "feature_config_available_in_store": fc in configs,
        "groups_match": fc in configs and configs[fc] == meta["groups"],
        "feature_dim_261": fc in configs and feature_dim(configs[fc]) == meta["feature_dim"] == 261,
        "class_count_matches_mapping": meta["vocabulary_size"] == mapping["num_classes"],
        "label_mapping_hash_matches": meta["label_mapping_sha256"] == mapping["vocabulary_sha256"],
        "window_is_a_configured_window": window in c3["sequences"]["windows"],
    }
    ref = build_model(meta["model"], meta["feature_dim"], meta["vocabulary_size"], meta["neural_config"])
    try:
        ck = load_checkpoint(exp / "best.pt", ref)  # strict load: architecture must fit the weights
        pre["weights_fit_architecture"] = True
    except Exception as e:  # noqa: BLE001 - report, do not export
        pre["weights_fit_architecture"] = False
        ck = {"error": f"{type(e).__name__}: {e}"}
    pre["checkpoint_epoch_is_best_epoch"] = ck.get("epoch") == meta.get("best_epoch")
    if not all(pre.values()):
        print(json.dumps(pre, indent=1))
        raise SystemExit("pre-export checks failed; nothing exported")
    spec = preprocessing_spec(c2, c3, profile, fc, window)

    # ---------------------------------------------------------------- export into a temp dir, then verify
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix=".export_", dir=out.parent))
    try:
        info = export_model(exp, tmp, mapping, configs, preprocessing=spec)
        rec = SignRecognizer.load(tmp / "best_model.pt", tmp / "class_mapping.json", c2, c3)  # full load-time checks
        ref.eval()
        index = pd.read_parquet(PROJECT_ROOT / st["index_dir"] / "index.parquet")
        ds = SequenceDataset(index, PROJECT_ROOT / st["landmarks_dir"] / "arrays", "test", window, fc, configs, mapping)
        n = min(args.check_samples, len(ds))
        x = torch.stack([ds[i][0] for i in range(n)])
        with torch.no_grad():
            la, lb = ref(x), rec.model(x)
            rng = torch.Generator().manual_seed(0)
            xr = torch.randn(8, window, meta["feature_dim"], generator=rng)
            ra, rb = ref(xr), rec.model(xr)
            all_test = torch.stack([ds[i][0] for i in range(len(ds))])
            pred_all = rec.model(all_test).argmax(1).numpy()
        stored = pd.read_csv(exp / "predictions_test.csv")
        same_order = stored["sample_id"].tolist() == ds.rows["sample_id"].tolist()
        via_api = [rec.predict_features(ds[i][0].numpy()).index for i in range(n)]
        post = {
            "logits_identical_on_real_test_sequences": bool(torch.equal(la, lb)),
            "logits_identical_on_fixed_random_inputs": bool(torch.equal(ra, rb)),
            "test_predictions_equal_experiment_predictions": bool(same_order and np.array_equal(pred_all, stored["y_pred"].to_numpy())),
            "inference_api_predictions_equal": via_api == la.argmax(1).tolist(),
            "loaded_through_inference_loader": True,
        }
    except Exception:
        shutil.rmtree(tmp, ignore_errors=True)
        raise
    report = {"created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
              "source_experiment": str(exp.relative_to(PROJECT_ROOT)), "experiment_id": meta["experiment_id"],
              "model": meta["model"], "seed": meta["seed"], "best_epoch": meta["best_epoch"],
              "metrics": meta["evaluations"], "model_info": {k: v for k, v in info.items() if k != "preprocessing"},
              "storage_profile": profile, "preprocessing_spec_sha256": spec_digest(spec),
              "checks_before_export": pre, "checks_after_export": post,
              "check_samples": n, "test_sequences_compared": len(ds),
              "all_checks_pass": all(pre.values()) and all(post.values())}
    (tmp / "export_report.json").write_text(json.dumps(report, indent=2, default=str) + "\n")
    if not report["all_checks_pass"]:
        failed = tmp.with_name(f"{out.name}.FAILED")
        tmp.rename(failed)
        print(json.dumps(report, indent=1, default=str))
        sys.exit(f"post-export checks failed; inspect {failed}")
    tmp.chmod(0o755)  # mkdtemp creates 0700
    tmp.rename(out)
    print(json.dumps({"candidate": str(out.relative_to(PROJECT_ROOT)), "checks_before_export": pre,
                      "checks_after_export": post, "metrics": meta["evaluations"]}, indent=1, default=str))


if __name__ == "__main__":
    main()

"""Runs exactly ONE experiment (model x feature configuration x window) end to end.

Train on TRAIN, select on VALIDATION (neural), reload the selected checkpoint from disk,
check it reproduces the selection value, then evaluate TRAIN (reported separately, never
as performance), VALIDATION and finally TEST once. Experiment directories are never
overwritten; the results table is append-only.
"""
from __future__ import annotations

import json
import platform
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy import sparse
from torch.utils.data import DataLoader

from asl.config import PROJECT_ROOT
from asl.landmarks.layout import feature_dim
from asl.models import latency as lat
from asl.models.metrics import classification_metrics, confusion, per_class_table, summary, top_confusions
from asl.models.recurrent import build_model, count_parameters
from asl.models.rf import build_rf, temporal_statistics
from asl.models.train import fit, load_checkpoint, predict, resolve_device, set_seed
from asl.sequences.dataset import SequenceDataset, load_sequence, make_dataloaders

SPLITS = ("train", "val", "test")
RESULT_COLUMNS = [
    "run_utc", "store", "experiment_id", "model", "feature_config", "window_size", "seed", "train_samples",
    "validation_samples", "test_samples", "train_classes", "vocabulary_size", "epochs", "best_epoch",
    "best_val_macro_f1", "test_accuracy", "test_macro_precision", "test_macro_recall", "test_macro_f1",
    "test_weighted_f1", "mean_latency_ms", "median_latency_ms", "p95_latency_ms", "fps", "latency_device",
    "checkpoint_path", "selection", "notes",
    # added after Step 4 v1: train / val reported next to test so a generalization gap is visible
    "train_accuracy", "train_macro_f1", "val_accuracy", "val_macro_f1", "train_test_accuracy_gap",
]


@dataclass(frozen=True)
class ExperimentSpec:
    model: str            # lstm | bilstm | rf
    feature_config: str   # hands | hands_pose | hands_pose_face
    window: int           # 30 | 45 | 60

    @property
    def experiment_id(self) -> str:
        return f"{self.feature_config}_w{self.window}_{self.model}"


def experiment_matrix(cfg: dict) -> list[ExperimentSpec]:
    m = cfg["experiment_matrix"]
    specs = [ExperimentSpec(model, fc, w) for fc in m["feature_configs"] for w in m["windows"]
             for model in m["neural_models"]]
    if m["rf_baseline"]:
        specs += [ExperimentSpec("rf", fc, w) for fc in m["feature_configs"] for w in m["windows"]]
    return specs


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _rel(p: Path) -> str:
    try:
        return str(Path(p).resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(p)


def _environment(device: torch.device, threads: int) -> dict:
    return {"python": platform.python_version(), "torch": torch.__version__, "numpy": np.__version__,
            "platform": platform.platform(), "machine": platform.machine(), "device": device.type,
            **({"gpu": torch.cuda.get_device_name(device)} if device.type == "cuda" else {}), "cpu_threads": threads}


def _write_eval(d: Path, name: str, y: np.ndarray, p: np.ndarray, ds: SequenceDataset, words: list[str],
                vocab: int, label: str) -> dict:
    """metrics_<name>.json, confusion_<name>.npz, per_class_<name>.csv, top_confusions_<name>.csv,
    predictions_<name>.csv. Returns the full metrics dict."""
    m = {**classification_metrics(y, p, vocab), "evaluated_split": label}
    (d / f"metrics_{name}.json").write_text(json.dumps(m, indent=2) + "\n")
    cm = confusion(y, p, vocab)
    sparse.save_npz(d / f"confusion_{name}.npz", cm)
    per_class_table(y, p, words).to_csv(d / f"per_class_{name}.csv", index=False)
    top_confusions(cm, words).to_csv(d / f"top_confusions_{name}.csv", index=False)
    rows = ds.rows
    pd.DataFrame({"sample_id": rows["sample_id"], "word": rows["word"], "y_true": y, "y_pred": p,
                  "pred_word": [words[i] for i in p], "correct": y == p, "active_frames": rows["active_frames"],
                  "short_active_interval": rows["short_active_interval"],
                  "stretch": ds.window / rows["active_frames"]}).to_csv(d / f"predictions_{name}.csv", index=False)
    return m


def _curves(history: pd.DataFrame, out: Path, title: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), dpi=110)
    axes[0].plot(history["epoch"], history["train_loss"], label="train")
    if history["val_loss"].notna().any():
        axes[0].plot(history["epoch"], history["val_loss"], label="validation")
    axes[0].set(xlabel="epoch", ylabel="cross-entropy loss", title="loss")
    axes[1].plot(history["epoch"], history["train_macro_f1"], label="train (training mode)")
    if history["val_macro_f1"].notna().any():
        axes[1].plot(history["epoch"], history["val_macro_f1"], label="validation")
    axes[1].set(xlabel="epoch", ylabel="macro-F1", title="macro-F1")
    for ax in axes:
        ax.legend()
        ax.grid(alpha=0.3)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def _append_result(path: Path, row: dict) -> None:
    """Append-only; an older table with fewer columns is extended (old rows kept unchanged)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    new = pd.DataFrame([{c: row.get(c) for c in RESULT_COLUMNS}])
    if path.exists():
        old = pd.read_csv(path)
        if list(old.columns) != RESULT_COLUMNS:
            pd.concat([old.reindex(columns=RESULT_COLUMNS), new], ignore_index=True).to_csv(path, index=False)
            return
        new.to_csv(path, mode="a", header=False, index=False)
    else:
        new.to_csv(path, index=False)


def _e2e_fn(ds: SequenceDataset, n: int, predict_one):
    rows = ds.rows.head(n)
    items = [(ds.arrays_dir / r.array_file, int(r.active_start_frame), int(r.active_end_frame))
             for r in rows.itertuples(index=False)]
    state = {"i": 0}

    def fn():
        path, s, e = items[state["i"] % len(items)]
        state["i"] += 1
        return predict_one(load_sequence(path, ds.groups, s, e, ds.window))
    return fn, len(items)


def run_experiment(spec: ExperimentSpec, *, store: str, index: pd.DataFrame, arrays_dir: Path, mapping: dict,
                   configurations: dict[str, list[str]], cfg: dict, experiments_root: Path, results_csv: Path,
                   max_epochs: int | None = None, allow_no_validation: bool = False, notes: str = "",
                   tag: str = "") -> dict:
    """`tag` names a variant of the same spec (e.g. "e100" for a longer budget): it is appended to
    the experiment id, so the run gets its own directory and results row."""
    run_id = f"{spec.experiment_id}_{tag}" if tag else spec.experiment_id
    counts = {s: int((index["split"] == s).sum()) for s in SPLITS}
    if counts["train"] == 0:
        raise ValueError("no training samples in the index")
    has_val = counts["val"] > 0
    if not has_val and not allow_no_validation:
        raise ValueError("no validation samples: refusing to run without model selection "
                         "(pass allow_no_validation=True explicitly; metrics are then train-fit only)")
    exp_dir = Path(experiments_root) / store / run_id
    if exp_dir.exists():
        raise FileExistsError(f"{exp_dir} exists; experiments are never overwritten")

    rt, seed, vocab = cfg["runtime"], cfg["seed"], mapping["num_classes"]
    device = resolve_device(rt["device"])
    torch.set_num_threads(rt["cpu_threads"])
    if rt.get("deterministic", True):
        torch.use_deterministic_algorithms(True, warn_only=True)
    set_seed(seed)
    words = sorted(mapping["word_to_index"], key=mapping["word_to_index"].get)
    groups = configurations[spec.feature_config]
    fdim = feature_dim(groups)
    dsets = {s: SequenceDataset(index, arrays_dir, s, spec.window, spec.feature_config, configurations, mapping)
             for s in SPLITS if counts[s] > 0}
    exp_dir.mkdir(parents=True)
    (exp_dir / "config.json").write_text(json.dumps({"spec": asdict(spec), "step4": cfg}, indent=2) + "\n")
    meta: dict = {"experiment_id": run_id, "tag": tag, "store": store, "model": spec.model,
                  "feature_config": spec.feature_config, "groups": groups, "window_size": spec.window,
                  "feature_dim": fdim, "seed": seed, "samples": counts,
                  "train_classes": int(index.loc[index["split"] == "train", "word"].nunique()),
                  "vocabulary_size": vocab, "label_mapping_sha256": mapping["vocabulary_sha256"],
                  "environment": _environment(device, rt["cpu_threads"]), "started_utc": _now(), "notes": notes}
    lc = cfg["latency"]
    eval_splits = ["train", "val", "test"] if has_val else []
    evals: dict[str, dict] = {}

    if spec.model == "rf":
        stats = {s: (np.stack([temporal_statistics(ds[i][0].numpy()) for i in range(len(ds))]),
                     ds.rows["label"].to_numpy()) for s, ds in dsets.items()}
        t0 = time.perf_counter()
        rf = build_rf(cfg["random_forest"]).fit(*stats["train"])
        meta["train_seconds"] = time.perf_counter() - t0
        joblib.dump(rf, exp_dir / "model.joblib")
        rf = joblib.load(exp_dir / "model.joblib")  # evaluate what was saved
        preds = {s: rf.predict(stats[s][0]) for s in stats}
        meta.update(rf_config=cfg["random_forest"], rf_feature_dim=4 * fdim, checkpoint="model.joblib",
                    selection="none (no hyperparameter search; fixed configuration)", epochs=None, best_epoch=None,
                    params=None)
        for s in (eval_splits or ["train"]):
            name = s if has_val else "train_fit"
            label = {"train": "train (fit; not generalization)", "val": "val", "test": "test"}[s]
            evals[name] = _write_eval(exp_dir, name, stats[s][1], preds[s], dsets[s], words, vocab, label)
        meta["best_val_macro_f1"] = evals["val"]["macro_f1"] if has_val else None
        ref_split = "test" if "test" in stats else "train"
        X = stats[ref_split][0]
        meta["latency"] = []
        for b in lc["batch_sizes"]:
            xb = np.resize(X, (b, X.shape[1]))
            r = lat.callable_latency(lambda: rf.predict(xb), b, lc["timed_runs"], lc["warmup_runs"],
                                     "RF predict on precomputed statistics")
            meta["latency"].append({**r, "device": "cpu"})
        fn, n = _e2e_fn(dsets[ref_split], lc["end_to_end_samples"],
                        lambda seq: rf.predict(temporal_statistics(seq)[None]))
        meta["latency"].append(lat.callable_latency(fn, 1, max(n, lc["end_to_end_samples"]), min(5, n),
                                                    "end-to-end per sample: load .npz + resample + assemble + "
                                                    "statistics + model (no MediaPipe)"))
        checkpoint = exp_dir / "model.joblib"
    else:
        nc = cfg["neural"]
        batch = nc["batch_size"]
        t0 = time.perf_counter()
        while True:
            set_seed(seed)
            model = build_model(spec.model, fdim, vocab, nc)
            loaders = make_dataloaders({s: dsets[s] for s in ("train", "val") if s in dsets}, batch, 0, seed)
            try:
                out = fit(model, loaders["train"], loaders.get("val"), nc, exp_dir, device, max_epochs=max_epochs,
                          allow_no_validation=allow_no_validation)
                break
            except torch.OutOfMemoryError:
                if batch == nc["fallback_batch_size"]:
                    raise
                batch = nc["fallback_batch_size"]
        meta["train_seconds"] = time.perf_counter() - t0
        history = pd.DataFrame(out["history"])
        history.to_csv(exp_dir / "history.csv", index=False)
        _curves(history, exp_dir / "training_curves.png", f"{store} / {run_id}")
        meta.update(neural_config=nc, batch_size_used=batch, params=count_parameters(model),
                    epochs=out["epochs_run"], best_epoch=out["best_epoch"],
                    best_val_macro_f1=out["best_val_macro_f1"], stopped_early=out["stopped_early"],
                    checkpoint=out["checkpoint"], selection=out["selection"],
                    checkpoint_epoch=out["checkpoint_epoch"], max_epochs_used=out["max_epochs_used"])
        # reload the selected checkpoint from disk; every evaluation uses the reloaded model
        checkpoint = exp_dir / out["checkpoint"]
        model = build_model(spec.model, fdim, vocab, nc)
        load_checkpoint(checkpoint, model)
        model.to(device)
        eval_loaders = make_dataloaders({f"{s}_eval": dsets[s] for s in dsets}, batch, 0, seed)
        if has_val:
            y, p, _ = predict(model, eval_loaders["val_eval"], device)
            reloaded = classification_metrics(y, p, vocab)["macro_f1"]
            meta["reloaded_val_macro_f1"] = reloaded
            meta["checkpoint_reload_reproduces_val_f1"] = bool(abs(reloaded - out["best_val_macro_f1"]) < 1e-12)
        for s in (eval_splits or ["train"]):  # test last, once
            if s not in dsets:
                continue
            y, p, _ = predict(model, eval_loaders[f"{s}_eval"], device)
            name = s if has_val else "train_fit"
            label = {"train": "train (selected checkpoint, eval mode; not generalization)" if has_val
                     else "train (fit; not generalization)", "val": "val", "test": "test"}[s]
            evals[name] = _write_eval(exp_dir, name, y, p, dsets[s], words, vocab, label)
        meta["latency"] = [lat.model_latency(model, spec.window, fdim, b, device, lc["warmup_runs"], lc["timed_runs"])
                           for b in lc["batch_sizes"]]
        ref = dsets["test"] if "test" in dsets else dsets["train"]

        @torch.no_grad()
        def one(seq):
            return model(torch.from_numpy(seq)[None].to(device)).argmax(1).cpu()
        model.eval()
        fn, n = _e2e_fn(ref, lc["end_to_end_samples"], one)
        meta["latency"].append(lat.callable_latency(fn, 1, max(n, lc["end_to_end_samples"]), min(5, n),
                                                    "end-to-end per sample: load .npz + resample + assemble + "
                                                    "model (no MediaPipe)"))

    meta["evaluations"] = {k: summary(v) for k, v in evals.items()}
    tr, te, va = evals.get("train"), evals.get("test"), evals.get("val")
    if tr and te:
        meta["generalization_gap"] = {"train_minus_test_accuracy": tr["accuracy"] - te["accuracy"],
                                      "train_minus_test_macro_f1": tr["macro_f1"] - te["macro_f1"]}
    meta["finished_utc"] = _now()
    (exp_dir / "metadata.json").write_text(json.dumps(meta, indent=2, default=str) + "\n")

    l1 = meta["latency"][0]
    row = {"run_utc": meta["finished_utc"], "store": store, "experiment_id": run_id, "model": spec.model,
           "feature_config": spec.feature_config, "window_size": spec.window, "seed": seed,
           "train_samples": counts["train"], "validation_samples": counts["val"], "test_samples": counts["test"],
           "train_classes": meta["train_classes"], "vocabulary_size": vocab, "epochs": meta["epochs"],
           "best_epoch": meta["best_epoch"], "best_val_macro_f1": meta["best_val_macro_f1"],
           "test_accuracy": te and te["accuracy"], "test_macro_precision": te and te["macro_precision"],
           "test_macro_recall": te and te["macro_recall"], "test_macro_f1": te and te["macro_f1"],
           "test_weighted_f1": te and te["weighted_f1"], "mean_latency_ms": l1["mean_ms"],
           "median_latency_ms": l1["median_ms"], "p95_latency_ms": l1["p95_ms"],
           "fps": l1["throughput_samples_per_s"], "latency_device": l1.get("device", "cpu"),
           "checkpoint_path": _rel(checkpoint), "selection": meta["selection"], "notes": notes,
           "train_accuracy": tr and tr["accuracy"], "train_macro_f1": tr and tr["macro_f1"],
           "val_accuracy": va and va["accuracy"], "val_macro_f1": va and va["macro_f1"],
           "train_test_accuracy_gap": meta.get("generalization_gap", {}).get("train_minus_test_accuracy")}
    _append_result(Path(results_csv), row)
    return {"exp_dir": exp_dir, "row": row, "meta": meta, "evals": evals}


def preflight(index: pd.DataFrame, arrays_dir: Path, spec: ExperimentSpec, configurations: dict[str, list[str]],
              mapping: dict, full: bool = True) -> dict:
    """Checks before training: every label is a valid index of `mapping`, every split has samples
    and every sequence is float32 [window, F] and finite. Raises ValueError on any problem."""
    fdim = feature_dim(configurations[spec.feature_config])
    nc, w2i = mapping["num_classes"], mapping["word_to_index"]
    problems, out = [], {"window": spec.window, "feature_dim": fdim, "num_classes": nc, "splits": {}}
    if not index["label"].between(0, nc - 1).all():
        problems.append("label outside [0, num_classes)")
    if not (index["word"].map(w2i) == index["label"]).all():
        problems.append("index label does not match the mapping")
    for s in SPLITS:
        ds = SequenceDataset(index, arrays_dir, s, spec.window, spec.feature_config, configurations, mapping)
        if not len(ds):
            problems.append(f"no {s} samples")
            continue
        shapes, finite = set(), True
        for i in range(len(ds) if full else min(len(ds), 64)):
            x, y = ds[i]
            shapes.add((tuple(x.shape), str(x.dtype)))
            finite &= bool(torch.isfinite(x).all())
        if shapes != {((spec.window, fdim), "torch.float32")} or not finite:
            problems.append(f"{s}: shapes {shapes}, finite {finite}")
        out["splits"][s] = {"samples": len(ds), "classes": int(ds.rows["label"].nunique()),
                            "item_shapes": sorted(str(x) for x in shapes), "all_finite": finite}
    if problems:
        raise ValueError(f"preflight failed: {problems}")
    out["labels_valid"] = True
    return out


def export_model(exp_dir: Path, out_dir: Path, mapping: dict, configurations: dict[str, list[str]]) -> dict:
    """Copy the selected checkpoint of a finished neural experiment to <out_dir>/best_model.pt and
    write <out_dir>/class_mapping.json. The .pt holds the weights plus everything needed to rebuild
    the model and its input (architecture, window, feature groups, label mapping hash)."""
    meta = json.loads((exp_dir / "metadata.json").read_text())
    if meta["model"] == "rf" or meta.get("checkpoint") != "best.pt":
        raise ValueError("only a neural experiment selected on validation (best.pt) can be exported")
    ck = torch.load(exp_dir / "best.pt", map_location="cpu", weights_only=True)
    words = sorted(mapping["word_to_index"], key=mapping["word_to_index"].get)
    if meta["vocabulary_size"] != len(words) or meta["label_mapping_sha256"] != mapping["vocabulary_sha256"]:
        raise ValueError("experiment was trained with a different label mapping")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    nc = meta["neural_config"]
    model_info = {"model": meta["model"], "input_dim": meta["feature_dim"], "num_classes": len(words),
                  "hidden_size": nc["hidden_size"], "num_layers": nc["num_layers"], "dropout": nc["dropout"],
                  "window": meta["window_size"], "feature_config": meta["feature_config"],
                  "groups": configurations[meta["feature_config"]], "epoch": ck["epoch"],
                  "val_macro_f1": ck["val_macro_f1"], "label_mapping_sha256": mapping["vocabulary_sha256"],
                  "source_experiment": _rel(exp_dir), "store": meta["store"]}
    torch.save({"state_dict": ck["state_dict"], **model_info}, out_dir / "best_model.pt")
    (out_dir / "class_mapping.json").write_text(json.dumps(
        {"num_classes": len(words), "vocabulary_sha256": mapping["vocabulary_sha256"],
         "index_to_class": {str(i): w for i, w in enumerate(words)}, "class_to_index": mapping["word_to_index"]},
        indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    return model_info


def load_exported_model(path: Path) -> tuple[torch.nn.Module, dict]:
    """Rebuild an exported model (eval mode) and return it with its metadata."""
    ck = torch.load(path, map_location="cpu", weights_only=True)
    info = {k: v for k, v in ck.items() if k != "state_dict"}
    model = build_model(info["model"], info["input_dim"], info["num_classes"],
                        {"hidden_size": info["hidden_size"], "num_layers": info["num_layers"], "dropout": info["dropout"]})
    model.load_state_dict(ck["state_dict"])
    return model.eval(), info

"""Step 2.10: integrity verification of a vocabulary landmark store (+ its sequence index).

Checks every accepted array, not a sample of them:
  1 integrity (keys, dtypes, finiteness, frame counts, metadata/label)   2 feature layout (F, order)
  3 missing-landmark representation    4 shoulder normalization    5 SequenceDataset windows
  6 split separation    7 measured storage    8 (optional) landmark overlay on raw frames
Writes reports/step2/<tag>_verification.{md,json} (+ figure). Exit 1 if any check fails.
This verifies the video -> landmarks -> .npz -> Dataset pipeline only, not model quality.
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from asl.config import PROJECT_ROOT, STEP2_CONFIG, load_config, load_yaml
from asl.data.split import load_frozen_split
from asl.data.vocabulary import load_vocabulary
from asl.landmarks.layout import (GROUP_INDEX, GROUPS, INTERPOLATED, LEFT_SHOULDER, OBSERVED, PRESENT, RIGHT_SHOULDER,
                                  assemble_features, feature_dim, feature_names, group_feature_dim)
from asl.landmarks.outputs import check_outputs, check_split_preserved, load_arrays, unreferenced_arrays
from asl.landmarks.pipeline import DOWNLOAD_FAILED, hf_fetcher, load_metadata
from asl.landmarks.storage import available_configurations, store_groups
from asl.sequences.dataset import SequenceDataset, make_dataloaders
from asl.sequences.index import SPLITS, check_index_leakage
from asl.sequences.labels import load_label_mapping

STEP3 = PROJECT_ROOT / "configs" / "step3_sequences.yaml"
EXPECTED_KEYS = {"pose", "left_hand", "right_hand", "pose_vis", "flags", "center", "scale", "shoulder_measured",
                 "active_interval", "version"}
FLOAT_KEYS = ["pose", "left_hand", "right_hand", "pose_vis", "center", "scale"]


def layout_table(groups: list[str]) -> list[dict]:
    rows, off = [], 0
    for g in groups:
        n = GROUPS[g]
        blocks = [(f"{g} x,y,z", n * 3, f"{n} landmarks x 3, interleaved x0,y0,z0,x1,..."), (f"{g} present", 1, "0/1")]
        if g == "pose":
            blocks.append(("pose visibility", n, "MediaPipe visibility x present"))
        for name, width, what in blocks:
            rows.append({"block": name, "columns": f"{off}-{off + width - 1}", "width": width, "contents": what})
            off += width
    return rows


def visualize(samples: pd.DataFrame, arrays_dir: Path, out_png: Path, c1) -> list[dict]:
    """Raw frames with de-normalized landmarks (norm * scale + center = pixel coordinates)."""
    import av
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fetch = hf_fetcher(c1["dataset"]["repo_id"], c1["dataset"]["revision"], c1["probe"]["download_retries"])
    fig, axes = plt.subplots(len(samples), 3, figsize=(12, 3.4 * len(samples)), dpi=100, squeeze=False)
    info = []
    with tempfile.TemporaryDirectory(dir=c1.path("raw_cache")) as tmp:
        for row_i, r in enumerate(samples.itertuples(index=False)):
            dest = Path(tmp) / "v.mp4"
            got = fetch(r.repo_path, r.sha256, dest)
            if not got["download_ok"]:
                info.append({"sample_id": r.sample_id, "error": got.get("download_error")})
                continue
            a = load_arrays(arrays_dir / r.array_file)
            with av.open(str(dest)) as c:
                frames = [(f.time, f.to_ndarray(format="rgb24")) for f in c.decode(c.streams.video[0])]
            t0 = frames[0][0] or 0.0
            times = np.array([t if t is not None else i / 30 for i, (t, _) in enumerate(frames)])
            s, e = (int(x) for x in a["active_interval"])
            picks = [s, (s + e) // 2, e]
            errs = []
            for col, k in enumerate(picks):
                fi = int(np.argmin(np.abs(times - (t0 + k / 30.0))))
                img = frames[fi][1]
                ax = axes[row_i, col]
                ax.imshow(img)
                for g, colr in (("pose", "lime"), ("left_hand", "red"), ("right_hand", "deepskyblue")):
                    if not (a["flags"][k, GROUP_INDEX[g]] & PRESENT):
                        continue
                    px = a[g][k].astype(np.float64) * a["scale"][k] + a["center"][k]
                    pts = px[:25] if g == "pose" else px  # upper body only for pose
                    ax.scatter(pts[:, 0], pts[:, 1], s=6 if g != "pose" else 10, c=colr)
                    errs.append(float(np.mean((px[:, 0] < -0.1 * img.shape[1]) | (px[:, 0] > 1.1 * img.shape[1]))))
                obs = {g: bool(a["flags"][k, GROUP_INDEX[g]] & OBSERVED) for g in ("left_hand", "right_hand")}
                ax.set_title(f"{r.word} [{r.split}] frame {k} (src {fi})\nL={'obs' if obs['left_hand'] else '-'} "
                             f"R={'obs' if obs['right_hand'] else '-'}", fontsize=8)
                ax.axis("off")
            info.append({"sample_id": r.sample_id, "word": r.word, "split": r.split, "frames_drawn": picks,
                         "fraction_points_outside_image": float(np.mean(errs)) if errs else None})
    fig.suptitle("De-normalized stored landmarks on raw frames (green pose, red = subject's left hand, "
                 "blue = subject's right hand)", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png)
    plt.close(fig)
    return info


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--landmarks-dir", required=True)
    ap.add_argument("--index-dir", required=True)
    ap.add_argument("--vocabulary", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--visualize", type=int, default=0, help="number of samples to overlay on raw frames")
    args = ap.parse_args()
    c1, c2, c3 = load_config(), load_yaml(STEP2_CONFIG), load_yaml(STEP3)
    lm, idx_dir = PROJECT_ROOT / args.landmarks_dir, PROJECT_ROOT / args.index_dir
    arrays_dir = lm / "arrays"
    lock = json.loads((c1.path("splits") / f"split_{c1['split']['version']}.lock.json").read_text())
    voc = load_vocabulary(PROJECT_ROOT / args.vocabulary, lock["sha256"])
    mapping = load_label_mapping(PROJECT_ROOT / "data" / "sequences" / f"label_mapping_{voc['name']}.json")
    frozen = load_frozen_split(c1.path("splits"), c1["split"]["version"])
    meta_all = load_metadata(lm)
    meta = meta_all[meta_all["rejection_reason"].ne(DOWNLOAD_FAILED)]
    idx = pd.read_parquet(idx_dir / "index.parquet")
    groups = store_groups(meta)
    configs = available_configurations(c2["configurations"], groups)
    checks: dict[str, bool] = {}
    R: dict = {"landmarks_dir": args.landmarks_dir, "index_dir": args.index_dir, "vocabulary": voc["name"],
               "stored_groups": groups, "configurations": {k: feature_dim(v) for k, v in configs.items()}}

    # ---------------------------------------------------------------- 1 integrity
    probs = check_outputs(meta, arrays_dir, configs)
    probs += check_split_preserved(meta, frozen)
    unref = unreferenced_arrays(meta, arrays_dir)
    acc = meta[~meta["rejected"].astype(bool)]
    per_file, bad = [], []
    for r in acc.itertuples(index=False):
        with np.load(arrays_dir / r.array_file) as z:
            keys = set(z.files)
            a = {k: z[k] for k in z.files}
        t = int(r.std_frame_count)
        expect_t = int(round(r.original_frame_count / r.original_fps * 30))
        row = {"sample_id": r.sample_id, "word": r.word, "split": r.split, "T": t,
               "decoded_frames": int(r.original_frame_count), "fps": float(r.original_fps),
               "keys_ok": keys == EXPECTED_KEYS, "no_face": "face" not in keys,
               "float32": all(a[k].dtype == np.float32 for k in FLOAT_KEYS),
               "flags_uint8": a["flags"].dtype == np.uint8,
               "finite": all(np.isfinite(a[k]).all() for k in FLOAT_KEYS),
               "frames_match_duration": abs(t - expect_t) <= 1 and t == a["flags"].shape[0],
               "label_in_vocabulary": r.word in mapping["word_to_index"]}
        row["ok"] = all(v for k, v in row.items() if isinstance(v, (bool, np.bool_)))
        per_file.append(row)
        if not row["ok"]:
            bad.append(row)
    pf = pd.DataFrame(per_file)
    checks["1_integrity"] = not probs and not unref and not bad
    R["integrity"] = {"metadata_records": len(meta), "accepted_arrays": len(acc),
                      "rejected": meta.loc[meta["rejected"].astype(bool), "rejection_reason"].value_counts().to_dict(),
                      "check_outputs_problems": probs[:20], "unreferenced_arrays": unref[:20],
                      "files_failing": bad[:20], "keys": sorted(EXPECTED_KEYS),
                      "T_frames": pf["T"].describe().round(1).to_dict(),
                      "decoded_fps": pf["fps"].describe().round(2).to_dict()}

    # ---------------------------------------------------------------- 2 feature layout
    hp = configs["hands_pose"]
    names = feature_names(hp)
    lay_ok = feature_dim(hp) == 261 and len(names) == 261
    r0 = acc.iloc[0]
    a0 = load_arrays(arrays_dir / r0.array_file)
    f0 = assemble_features(a0, hp)
    off, pieces = 0, []
    for g in hp:  # recompute each block from the raw arrays and compare with the assembled matrix
        n = GROUPS[g]
        pres = ((a0["flags"][:, GROUP_INDEX[g]] & PRESENT) != 0).astype(np.float32)
        pieces.append(np.array_equal(f0[:, off:off + 3 * n], a0[g].reshape(len(pres), -1)))
        pieces.append(np.array_equal(f0[:, off + 3 * n], pres))
        if g == "pose":
            pieces.append(np.array_equal(f0[:, off + 3 * n + 1:off + 3 * n + 1 + n], a0["pose_vis"] * pres[:, None]))
        off += group_feature_dim(g)
    checks["2_feature_layout"] = lay_ok and f0.shape[1] == 261 and f0.dtype == np.float32 and all(pieces)
    R["feature_layout"] = {"F_hands_pose": int(f0.shape[1]), "F_hands": feature_dim(configs["hands"]),
                           "blocks": layout_table(hp), "first_names": names[:4], "presence_columns":
                           [i for i, n in enumerate(names) if n.endswith("_present")],
                           "assembled_equals_raw_blocks": all(pieces)}

    # ---------------------------------------------------------------- 3 missing landmarks
    cases = {"left_hand_missing_only": 0, "right_hand_missing_only": 0, "both_hands_missing": 0,
             "pose_missing": 0, "interpolated_hand_frames": 0}
    inside = {"frames_inside_active_interval": 0, "left_hand_missing_only": 0, "right_hand_missing_only": 0,
              "both_hands_missing": 0, "pose_missing": 0}
    examples: dict[str, str] = {}
    rep_ok = True
    for r in acc.itertuples(index=False):
        a = load_arrays(arrays_dir / r.array_file)
        f = assemble_features(a, hp)
        pres = {g: (a["flags"][:, GROUP_INDEX[g]] & PRESENT) != 0 for g in hp}
        L, Rh, P = pres["left_hand"], pres["right_hand"], pres["pose"]
        masks = {"left_hand_missing_only": ~L & Rh, "right_hand_missing_only": L & ~Rh, "both_hands_missing": ~L & ~Rh,
                 "pose_missing": ~P}
        s_, e_ = (int(x) for x in a["active_interval"])
        act = np.zeros(len(L), bool)
        act[s_:e_ + 1] = True
        inside["frames_inside_active_interval"] += int(act.sum())
        for k, m in masks.items():
            cases[k] += int(m.sum())
            inside[k] += int((m & act).sum())
            if m.any() and k not in examples:
                examples[k] = f"{r.sample_id} frame {int(np.flatnonzero(m)[0])}"
        for g in ("left_hand", "right_hand"):
            interp = (a["flags"][:, GROUP_INDEX[g]] & INTERPOLATED) != 0
            cases["interpolated_hand_frames"] += int(interp.sum())
            rep_ok &= bool(pres[g][interp].all())  # interpolated => present
        off = 0
        for g in hp:  # absent => all coordinates 0 and presence feature 0; present => presence 1
            n = GROUPS[g]
            block, pcol = f[:, off:off + 3 * n], f[:, off + 3 * n]
            rep_ok &= bool((block[~pres[g]] == 0).all() and (pcol[~pres[g]] == 0).all() and (pcol[pres[g]] == 1).all())
            if g == "pose":
                rep_ok &= bool((f[~pres[g], off + 3 * n + 1:off + 3 * n + 1 + n] == 0).all())
            off += group_feature_dim(g)
    checks["3_missing_landmarks"] = rep_ok
    R["missing_landmarks"] = {"frames": cases, "frames_inside_active_interval": inside, "first_example": examples, "representation_consistent": rep_ok,
                              "rule": "absent group -> its x,y,z (and pose visibility) are 0.0 and its presence "
                                      "column is 0; gaps <= 5 frames are interpolated and marked present"}

    # ---------------------------------------------------------------- 4 normalization
    mid_err, dist_err, carried_ok, scale_rel = [], [], True, []
    n_meas = n_carried = 0
    for r in acc.itertuples(index=False):
        a = load_arrays(arrays_dir / r.array_file)
        sm = a["shoulder_measured"].astype(bool)
        P = (a["flags"][:, GROUP_INDEX["pose"]] & PRESENT) != 0
        ls, rs = a["pose"][:, LEFT_SHOULDER].astype(np.float64), a["pose"][:, RIGHT_SHOULDER].astype(np.float64)
        m = sm & P
        mid_err.append(np.abs((ls[m] + rs[m]) / 2).max() if m.any() else 0.0)
        dist_err.append(np.abs(np.linalg.norm((ls[m] - rs[m])[:, :2], axis=1) - 1).max() if m.any() else 0.0)
        n_meas += int(sm.sum())
        n_carried += int((~sm).sum())
        last = None
        for t in range(len(sm)):  # carried frames reuse the last measured frame (or the first one if leading)
            if sm[t]:
                last = t
            else:
                ref = last if last is not None else int(np.flatnonzero(sm)[0])
                carried_ok &= bool(np.array_equal(a["center"][t], a["center"][ref]) and a["scale"][t] == a["scale"][ref])
        scale_rel.append(float(np.median(a["scale"])) / float(r.width))
    checks["4_normalization"] = max(mid_err) < 1e-4 and max(dist_err) < 1e-4 and carried_ok and n_meas > 0
    R["normalization"] = {"rule": "pixel coords (x*W, y*H, z*W); origin = midpoint of pose 11/12 (shoulders); "
                                  "scale = 2-D shoulder distance; frames with shoulder visibility < 0.5 reuse the "
                                  "last measured frame (leading frames the first)",
                          "max_abs_normalized_shoulder_midpoint": float(max(mid_err)),
                          "max_abs_normalized_shoulder_distance_minus_1": float(max(dist_err)),
                          "frames_measured": n_meas, "frames_carried": n_carried,
                          "carried_frames_reuse_reference": carried_ok,
                          "shoulder_width_over_image_width": pd.Series(scale_rel).describe().round(3).to_dict()}

    # ---------------------------------------------------------------- 5 SequenceDataset
    ds_res, ds_ok = {}, True
    voc_index = {c["class"]: c["index"] for c in voc["classes"]}
    ds_ok &= mapping["num_classes"] == voc["num_classes"] == 100 and all(
        mapping["word_to_index"][w] == i for w, i in voc_index.items())
    for w in c3["sequences"]["windows"]:
        for split in SPLITS:
            ds = SequenceDataset(idx, arrays_dir, split, w, "hands_pose", configs, mapping)
            if not len(ds):
                ds_ok = False
                continue
            xb, yb = next(iter(make_dataloaders({"b": ds}, 8, 0, 0)["b"]))
            shapes, labels_ok = set(), True
            for i in range(len(ds)):
                x, y = ds[i]
                shapes.add(tuple(x.shape))
                labels_ok &= int(y) == voc_index[ds.metadata(i)["word"]]
            ok = shapes == {(w, 261)} and labels_ok and xb.dtype.is_floating_point and str(xb.dtype) == "torch.float32"
            ds_ok &= ok
            ds_res[f"w{w}/{split}"] = {"items": len(ds), "item_shape": sorted(shapes), "batch": list(xb.shape),
                                       "dtype": str(xb.dtype), "labels_match_vocabulary": labels_ok}
    checks["5_sequence_dataset"] = ds_ok
    R["sequence_dataset"] = ds_res

    # ---------------------------------------------------------------- 6 splits
    leaks = check_index_leakage(idx)
    m = idx.merge(frozen[["sample_id", "split"]], on="sample_id", suffixes=("", "_frozen"))
    ids = {s: set(idx.loc[idx.split == s, "sample_id"]) for s in SPLITS}
    disjoint = all(not (ids[a] & ids[b]) for a in SPLITS for b in SPLITS if a < b)
    ds_split_ok = True
    for split in SPLITS:
        ds = SequenceDataset(idx, arrays_dir, split, 30, "hands_pose", configs, mapping)
        ds_split_ok &= set(ds.rows["split"]) <= {split} and set(ds.rows["sample_id"]) == ids[split]
    checks["6_split_separation"] = not leaks and bool((m["split"] == m["split_frozen"]).all()) and disjoint and ds_split_ok
    R["splits"] = {"per_split": {s: len(ids[s]) for s in SPLITS}, "leakage_problems": leaks,
                   "index_split_equals_frozen": bool((m["split"] == m["split_frozen"]).all()),
                   "sample_ids_disjoint": disjoint, "datasets_contain_only_their_split": ds_split_ok}

    # ---------------------------------------------------------------- 7 storage
    man = pd.read_parquet(c1.path("metadata") / "manifest.parquet")
    sizes = man[man["eligible_pre_probe"]].set_index("sample_id")["size_bytes"]
    npz = acc["bytes"].astype(float)
    full = load_frozen_split(c1.path("splits"), c1["split"]["version"])
    full = full[full["word"].isin(voc_index)]
    acc_rate = len(acc) / len(meta)
    R["storage"] = {"videos_processed": len(meta), "accepted": len(acc), "acceptance_rate": acc_rate,
                    "raw_download_mb": float(sizes.reindex(meta["sample_id"]).sum() / 1e6),
                    "npz_total_mb": float(npz.sum() / 1e6), "npz_mean_kb": float(npz.mean() / 1e3),
                    "npz_median_kb": float(npz.median() / 1e3), "bytes_per_frame": float(npz.sum() / acc["std_frame_count"].sum()),
                    "vocabulary_videos": len(full), "vocabulary_raw_download_gb": float(sizes.reindex(full["sample_id"]).sum() / 1e9),
                    "estimated_vocabulary_npz_gb": float(npz.mean() * len(full) * acc_rate / 1e9),
                    "mean_seconds_per_video_per_worker": float(meta["total_s"].mean())}
    checks["7_storage_measured"] = True

    # ---------------------------------------------------------------- 8 visualization
    if args.visualize:
        pick = (idx.merge(acc[["sample_id"]], on="sample_id").merge(frozen[["sample_id", "sha256"]], on="sample_id",
                                                                     suffixes=("", "_f")))
        pick = pd.concat([pick[pick.split == s].head(1) for s in SPLITS] +
                         [pick[pick.split == "train"].iloc[1:args.visualize - 2]]).head(args.visualize)
        png = PROJECT_ROOT / "reports" / "step2" / "figures" / f"{args.tag}_landmarks.png"
        R["visualization"] = {"figure": str(png.relative_to(PROJECT_ROOT)),
                              "samples": visualize(pick, arrays_dir, png, c1)}

    R["checks"] = checks
    R["all_checks_pass"] = all(checks.values())
    out = PROJECT_ROOT / "reports" / "step2"
    (out / f"{args.tag}_verification.json").write_text(json.dumps(R, indent=2, default=str) + "\n")
    lines = [f"# Landmark store verification - `{args.tag}`\n",
             f"Store `{args.landmarks_dir}`, index `{args.index_dir}`, vocabulary `{voc['name']}`. Generated by "
             "`scripts/s2_10_verify_store.py`. This verifies the video -> landmarks -> .npz -> Dataset pipeline; it "
             "says nothing about recognition accuracy.\n",
             "| check | result |\n|---|---|"] + [f"| {k} | {'PASS' if v else 'FAIL'} |" for k, v in checks.items()]
    lines += ["", "## Feature vector (hands_pose, F = 261)\n", "| block | columns | width | contents |", "|---|---|---|---|"]
    lines += [f"| {b['block']} | {b['columns']} | {b['width']} | {b['contents']} |" for b in R["feature_layout"]["blocks"]]
    for sec in ("integrity", "missing_landmarks", "normalization", "splits", "storage", "visualization"):
        if sec in R:
            lines += ["", f"## {sec}\n", "```json", json.dumps(R[sec], indent=1, default=str), "```"]
    lines += ["", "## sequence_dataset\n", "| window/split | items | item shape | batch | labels match |", "|---|---|---|---|---|"]
    lines += [f"| {k} | {v['items']} | {v['item_shape']} | {v['batch']} | {v['labels_match_vocabulary']} |"
              for k, v in ds_res.items()]
    (out / f"{args.tag}_verification.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({"checks": checks, "storage": R["storage"], "missing": cases,
                      "normalization": {k: R["normalization"][k] for k in ("max_abs_normalized_shoulder_midpoint",
                                        "max_abs_normalized_shoulder_distance_minus_1", "frames_carried")}},
                     indent=1, default=str))
    if not R["all_checks_pass"]:
        sys.exit(1)


if __name__ == "__main__":
    main()

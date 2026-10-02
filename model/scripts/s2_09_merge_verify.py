"""Step 2.9: merge the shards of a run into one landmark store and verify its integrity.

Fails (exit 1) on: a planned sample without a final record, pending download failures,
a sample in more than one shard, array/metadata inconsistencies, disagreement with the
frozen split, or a shard run with a different code fingerprint. Also reports per-split
statistics and the cross-platform parity comparison (if the parity list was extracted).
Writes data/landmarks/<run>_store/ (never overwritten) and reports/step2/<run>_extraction_report.md.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys

import numpy as np
import pandas as pd

from asl.config import PROJECT_ROOT, STEP2_CONFIG, load_config, load_yaml
from asl.data.split import load_frozen_split
from asl.landmarks.layout import GROUP_INDEX, GROUP_ORDER, OBSERVED
from asl.landmarks.outputs import check_outputs, check_split_preserved, load_arrays, unreferenced_arrays
from asl.landmarks.pipeline import DOWNLOAD_FAILED, load_metadata
from asl.sequences.index import interval_bin


def link_or_copy(src, dst) -> None:
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def split_stats(meta: pd.DataFrame) -> dict:
    out = {}
    for split, g in meta.groupby("split"):
        acc = g[~g["rejected"].astype(bool)]
        out[split] = {"videos": int(len(g)), "accepted": int(len(acc)), "accepted_pct": 100 * len(acc) / len(g),
                      "rejections": g.loc[g["rejected"].astype(bool), "rejection_reason"].value_counts().to_dict(),
                      "short_active_interval": int(acc["short_active_interval"].astype(bool).sum()),
                      "active_bins": acc["active_frames"].map(interval_bin).value_counts().to_dict()
                      if "active_frames" in acc else {},
                      "classes_accepted": int(acc["word"].nunique()), "bytes": int(acc["bytes"].sum())}
    return out


def _s(v) -> str:
    return v if isinstance(v, str) else ""


def _i(v):
    return None if v is None or pd.isna(v) else int(v)


def parity(run, bench_dir) -> dict | None:
    pdir = run / "parity"
    if not pdir.exists():
        return None
    remote = load_metadata(pdir).set_index("sample_id")
    local = pd.read_parquet(bench_dir / "meta_parts" / "part_00000.parquet").set_index("sample_id")
    rows = []
    for sid, r in remote.iterrows():
        l = local.loc[sid]
        rec = {"sample_id": sid, "same_decision": _s(r["rejection_reason"]) == _s(l["rejection_reason"]),
               "same_interval": (_i(r.get("active_start_frame")), _i(r.get("active_end_frame")))
               == (_i(l.get("active_start_frame")), _i(l.get("active_end_frame")))}
        if isinstance(r["array_file"], str) and isinstance(l["array_file"], str):
            a, b = load_arrays(pdir / "arrays" / r["array_file"]), load_arrays(bench_dir / "arrays" / l["array_file"])
            if a["flags"].shape == b["flags"].shape:
                both = ((a["flags"] & 2) != 0) & ((b["flags"] & 2) != 0)
                for g in GROUP_ORDER:
                    m = both[:, GROUP_INDEX[g]]
                    d = np.abs(a[g].astype(np.float32) - b[g].astype(np.float32))[m]
                    rec[f"{g}_max_abs_diff"] = float(d.max()) if d.size else 0.0
                    rec[f"{g}_mean_abs_diff"] = float(d.mean()) if d.size else 0.0
                rec["observed_flag_agreement"] = float(((a["flags"] & OBSERVED) == (b["flags"] & OBSERVED)).mean())
            else:
                rec["length_mismatch"] = True
        rows.append(rec)
    df = pd.DataFrame(rows)
    return {"videos": int(len(df)), "same_decision": int(df["same_decision"].sum()),
            "same_interval": int(df["same_interval"].sum()),
            "summary": df.drop(columns=["sample_id"]).select_dtypes("number").describe().round(5).to_dict()}


def main() -> None:
    c1, c2 = load_config(), load_yaml(STEP2_CONFIG)
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-name", required=True)
    args = ap.parse_args()
    run = PROJECT_ROOT / "data" / "landmarks" / "runs" / args.run_name
    plan = json.loads((run / "plan.json").read_text())
    store = PROJECT_ROOT / "data" / "landmarks" / f"{args.run_name}_store"
    if store.exists():
        raise SystemExit(f"{store} exists; merged stores are never overwritten")

    problems: list[str] = []
    expected, metas, manifests, unref = [], [], [], 0
    for i in range(plan["shards"]):
        expected.append(pd.read_csv(run / "shards" / f"shard_{i:02d}_of_{plan['shards']:02d}.csv", dtype=str,
                                    keep_default_na=False))
        sdir = run / "shards" / f"shard_{i:02d}"
        if not sdir.exists():
            problems.append(f"shard {i}: not run")
            continue
        m = load_metadata(sdir)
        m["shard_dir"] = str(sdir)
        metas.append(m)
        unref += len(unreferenced_arrays(m, sdir / "arrays"))
        for f in sorted(sdir.glob("run_manifest_*.json")):
            manifests.append(json.loads(f.read_text()))
    exp = pd.concat(expected, ignore_index=True)
    meta = pd.concat(metas, ignore_index=True) if metas else pd.DataFrame(columns=["sample_id"])
    if meta["sample_id"].duplicated().any():
        problems.append("a sample has records in more than one shard")
    missing = sorted(set(exp["sample_id"]) - set(meta["sample_id"]))
    pending = meta.loc[meta["rejection_reason"].eq(DOWNLOAD_FAILED), "sample_id"].tolist() if len(meta) else []
    extra = sorted(set(meta["sample_id"]) - set(exp["sample_id"]))
    if missing:
        problems.append(f"{len(missing)} planned samples have no record (e.g. {missing[:3]})")
    if pending:
        problems.append(f"{len(pending)} download failures still pending: re-run the shard(s)")
    if extra:
        problems.append(f"{len(extra)} records not in the plan")
    if not manifests or not all(m.get("fingerprint_matches_plan") for m in manifests):
        problems.append("missing or mismatching shard run manifests")

    frozen = load_frozen_split(c1.path("splits"), c1["split"]["version"])
    final = meta[meta["rejection_reason"].ne(DOWNLOAD_FAILED)].copy() if len(meta) else meta
    tmp = store.with_name(store.name + ".tmp")
    shutil.rmtree(tmp, ignore_errors=True)
    (tmp / "arrays").mkdir(parents=True)
    (tmp / "meta_parts").mkdir()
    for r in final.itertuples():
        if isinstance(r.array_file, str) and r.array_file:
            link_or_copy(os.path.join(r.shard_dir, "arrays", r.array_file), tmp / "arrays" / r.array_file)
    final = final.drop(columns="shard_dir")
    final.to_parquet(tmp / "meta_parts" / "part_00000.parquet", index=False)
    problems += [f"consistency: {p}" for p in check_outputs(final, tmp / "arrays", c2["configurations"])[:20]]
    problems += [f"split: {p}" for p in check_split_preserved(final, frozen)[:20]]

    report = {"run_name": args.run_name, "planned": int(len(exp)), "records": int(len(final)),
              "missing": len(missing), "pending_download_failures": len(pending),
              "unreferenced_arrays_ignored": unref, "hosts": sorted({json.dumps(m["host"], sort_keys=True) for m in manifests}),
              "by_split": split_stats(final) if len(final) else {},
              "decode_errors": final.loc[final["error"].fillna("") != "", ["sample_id", "error"]].to_dict("records")
              if len(final) else [],
              "storage_bytes": int(final["bytes"].sum()) if len(final) else 0,
              "parity": parity(run, PROJECT_ROOT / "data" / "landmarks" / "benchmark_v2_mixed"),
              "problems": problems, "integrity_ok": not problems}
    (tmp / "verification.json").write_text(json.dumps(report, indent=2, default=str) + "\n")
    if problems:
        print(json.dumps(report, indent=2, default=str))
        print(f"INTEGRITY FAILED; partial merge left in {tmp}", file=sys.stderr)
        sys.exit(1)
    tmp.rename(store)
    rep = PROJECT_ROOT / "reports" / "step2" / f"{args.run_name}_extraction_report.md"
    L = [f"# Step 2 extraction run `{args.run_name}`\n", f"Planned {report['planned']} videos; records {report['records']}; "
         f"integrity OK. Store: `data/landmarks/{args.run_name}_store/`. Hosts: {report['hosts']}\n",
         "| split | videos | accepted | accepted % | short active | classes | MB | rejections |", "|---|---|---|---|---|---|---|---|"]
    for s, v in report["by_split"].items():
        L.append(f"| {s} | {v['videos']} | {v['accepted']} | {v['accepted_pct']:.1f} | {v['short_active_interval']} | "
                 f"{v['classes_accepted']} | {v['bytes'] / 1e6:.1f} | {v['rejections']} |")
    L += [f"\nDecode errors: {len(report['decode_errors'])}; unreferenced arrays from interrupted batches (ignored): {unref}.",
          f"\nParity vs M2 benchmark extraction: {json.dumps(report['parity'], default=str)}\n"]
    rep.write_text("\n".join(L) + "\n")
    print(json.dumps({k: report[k] for k in ("planned", "records", "missing", "by_split", "storage_bytes", "integrity_ok")},
                     indent=2, default=str))
    print(f"parity: {json.dumps(report['parity'], default=str)[:600]}")


if __name__ == "__main__":
    main()

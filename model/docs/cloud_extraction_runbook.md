# Cloud runbook — Step 2 extraction + Step 3 sequences for the frozen VAL and TEST splits

Runs the **unchanged** Step 2 pipeline (methodology v2, mixed precision, active-interval
rules) and the unchanged Step 3 sequence preparation on a remote machine. Validated end to
end on the M2 with a 12-video dry run (`dryrun_valtest_m2`, 6 val + 6 test + 3 parity).

## Workload (frozen split, measured inputs; time is an extrapolation)

| | val + test | (train, not approved) |
|---|---|---|
| videos | 32,623 (16,272 val + 16,351 test), 2,207 classes | 75,993 |
| video download (streamed, deleted per batch) | 16.3 GB | 38.0 GB |
| landmark output (≈ 93.5 % accepted × 0.316 MB) | ≈ 9.6 GB | ≈ 22.5 GB |
| CPU work at the M2's measured 15.5 frames/s per worker | ≈ 49 worker-hours | ≈ 115 worker-hours |

Wall time ≈ worker-hours / workers, **if** a cloud vCPU is as fast as an M2 core (not
measured — the first shard's log reports the real rate). One worker ≈ 0.6 GB RAM.

## Machine requirements

- **Linux x86_64**, glibc ≥ 2.28 (e.g. Ubuntu 20.04/22.04/24.04). MediaPipe 0.10.21 has
  **no Linux ARM wheel** — ARM instances (AWS Graviton, Ampere) will not work.
- Python 3.12; CPU-only is fine (the pinned MediaPipe Holistic runs on CPU).
- Disk ≥ 40 GB free (bundle 0.12 GB, outputs ≈ 10 GB, per-batch video cache, merge copies).
- RAM ≈ 0.6 GB × workers + 2 GB.
- Optional but recommended: a Hugging Face token (`HF_TOKEN`, read-only) to avoid HTTP 429
  rate limits; it is sent only to `huggingface.co`.

## Steps

On the M2 (plan + bundle; N = number of machines, one shard each):

```bash
PYTHONPATH=src .venv/bin/python scripts/s2_07_prepare_extraction.py --run-name valtest_v2 --splits val test --shards N --parity 20
scripts/cloud/make_bundle.sh valtest_v2        # dist/valtest_v2_bundle.tar.gz + .sha256
```

On each cloud machine:

```bash
shasum -a 256 -c valtest_v2_bundle.tar.gz.sha256 && mkdir asl && tar -xzf valtest_v2_bundle.tar.gz -C asl && cd asl
scripts/cloud/setup.sh                         # venv, pinned deps, self-tests
export HF_TOKEN=...                            # optional
scripts/cloud/run_shard.sh valtest_v2 parity 2 # 20 benchmark videos: cross-platform parity (machine 0 only)
scripts/cloud/run_shard.sh valtest_v2 <i>      # this machine's shard; workers = nproc - 1
tail -f data/landmarks/runs/valtest_v2/logs/<i>.log
```

`s2_08` refuses to start if the code/config fingerprint differs from the plan. Every
command is resumable: re-run it after any interruption; finished videos are skipped and
HTTP 429 download failures are retried.

After all shards finish (on one machine holding every shard directory, or on the M2 after
copying `data/landmarks/runs/valtest_v2/` back):

```bash
PYTHONPATH=src .venv/bin/python scripts/s2_09_merge_verify.py --run-name valtest_v2      # exit 1 on any integrity problem
PYTHONPATH=src .venv/bin/python scripts/s3_04_prepare_store.py \
    --landmarks-dir data/landmarks/valtest_v2_store --index-dir data/sequences/valtest_v2
```

Copy back to the M2: `data/landmarks/valtest_v2_store/`, `data/sequences/valtest_v2/`,
`data/landmarks/runs/valtest_v2/` (logs, manifests) and
`reports/step2/valtest_v2_extraction_report.md`, e.g. `rsync -av --progress host:asl/data/... data/...`.

## What is verified

- `s2_09`: every planned sample has exactly one final record; no pending download
  failures; no sample in two shards; every shard ran with the planned fingerprint;
  array/metadata consistency (shapes, mixed-precision dtypes, zero-fill, flags, active
  interval); split, word and path identical to the frozen split; per-split acceptance,
  rejection reasons, short intervals, active-interval bins, storage; parity vs the M2.
- `s3_04`: the unchanged Step 3 `build_index` (frozen split, shared 2,207-class mapping,
  leakage checks) and generation of every sequence for 3 configurations × 3 windows ×
  splits, checked for shape, finiteness, presence flags and bit-identical regeneration.

## Dry-run result on the M2 (12 videos)

Integrity OK; 12/12 accepted (val 6, test 6; 6 short intervals kept); index built, all
sequence checks passed; parity (same machine): identical decisions and active intervals,
pose identical, face/hands ≤ 1 float16 ulp (0.00049) because the benchmark store was
converted float32 → float16 while fresh extraction rounds float64 → float16 directly.

# Research Paper Reference Guide

**Manuscript audited:** `~/Downloads/Real_Time_ASL_Recognition_Research_Paper.pdf` ("Real-Time American Sign Language
Recognition and Video Calling System", 9 pages, text extracted with macOS PDFKit). Not modified.
**Evidence base:** project repository (`model/` and the Next.js app), saved reports, run metadata and predictions; the 12
supplied PDFs in `~/Documents/research_papers/`; limited external metadata checks (Crossref, arXiv, publisher pages,
Hugging Face) carried out earlier in this project and marked *external* below.
**No training, inference, experiment or test-set evaluation was run to produce this guide.**
Paths are relative to `model/` unless they start with `src/`, `package.json` or `README.md` (repository root) or `~`.

Labels used: **VERIFIED** (seen in a repository file or supplied PDF), **UNVERIFIED** (not confirmable from available
evidence), **AUTHOR INPUT REQUIRED** (only you can supply it), *external* (confirmed outside the supplied PDFs).

> **Note on a second draft.** `reports/paper/draft/manuscript.md` (and `.docx`) is a separate, evidence-checked
> draft produced earlier in this project. It is *not* your manuscript; it is referred to below only as a source of
> already-verified wording and tables. Its own known corrections are listed in §D and §H.

---

## A. Executive summary

**Overall quality.** The manuscript is well organised and clearly written, and its background material (Introduction,
Related Work, LSTM/BiLSTM description, the limitation on signer independence) is largely sound. However, it is written
as a **design proposal for a system that was not built as described**. The Results section is empty
("[To be written after the experiments are completed.]"), Section X (Conclusion) is referenced but missing, Fig. 1 is
referenced but absent, and several methodological sections describe components that do not exist in the code.

**Most important corrections (each verified against the repository):**

1. **Integration claims are false.** The paper states that the recognizer runs inside the WebRTC call with browser-side
   MediaPipe, a FastAPI service, a WebSocket signalling server and captions over a WebRTC data channel (Abstract;
   III.C item 7; V.A; V.G; VIII). None of this exists: the app has no recognition, captions, data channel, FastAPI or
   WebSocket code (`src/components/Call.tsx`; `package.json`; `README.md`: signalling is HTTP + MongoDB); FastAPI
   appears in the repository only as a "NOT built" label in the pipeline figure script.
2. **The experiment grid was not run.** The 18-configuration grid (3 landmark sets × 2 models × 3 windows), the
   hands + pose + face configuration, 60-frame windows and the Random Forest baseline were **not** run
   (`experiments/dev100_hp/` contains no face, 60-frame or RF run). What was run: LSTM vs BiLSTM (5 seeds),
   E1 temporal augmentation, E2 upper-body pose ablation, a hands-only comparison (validation only), one single-seed
   45-frame LSTM, one dropout-0.5 run.
3. **Vocabulary size is wrong throughout.** Models were trained and evaluated on **100 classes**, not 2,208
   (`data/sequences/dev100/index_summary.json`). Statements built on 2,208 classes (III.C item 1, VI.B, VI.C) must change.
4. **Several method details contradict the implementation:** feature dimensions (HP is 261, not 228; HPF is 1,666,
   not 1,633), resampling (linear interpolation over the *active signing interval*, not nearest-frame over the whole
   clip), split order (frozen split *before* rejection, at group level), scheduler monitor (validation macro-F1, not
   loss), hardware (CPU, not MPS), streaming design (hand-activity segmentation, not a stride-5 sliding window),
   post-processing (cooldown and confidence threshold; not the 0.75/consensus/duplicate rule).
5. **Streaming evaluation differs.** No synthetic continuous streams, DPR or MSR were computed. Streaming was evaluated
   with **one signer's** webcam sessions and replays (development data).
6. **Citation problems:** wrong reference numbers in the Introduction (e.g. [12] cited for WebRTC integration is a
   Transformer paper), most Related Work statements carry no citation number, several descriptions misstate the
   sources (SLRNet, GCAR, PBRC, Sign-Meet), one source has an **IEEE Expression of Concern** ([7]), two sources are
   low-reliability ([2], [4]), and several references have wrong or missing metadata ([10], [11], [12]); essential
   references (LSTM, BiLSTM, AdamW, MediaPipe primary papers) are missing.
7. **Test-set use must be disclosed** (13 development runs + auxiliary checks + export equality/logits checks; §D).

**Sections that can largely be retained:** Index Terms, I (with citation fixes and scope edits), II.B–II.D (with
corrections), III.A first paragraph, III.B (scope; edit the "sliding window/captions" sentences), IV.A (update counts),
IV.C interpolation paragraph, IV.D normalization equation, V.B, V.D, V.E, VI.C metric definitions (Eq. 8), IX items 1–4.

**Main unresolved issues:** Results must be written from saved evidence (§C); test-set disclosure (§D); ethics and
signer consent; repository/code availability; dataset provenance; venue requirements (§G).

---

## B. Section-by-section writing guide

Status codes: **KEEP**, **UPDATE**, **INCOMPLETE**, **RESTART**. 35 items audited.

### B.1 Title — UPDATE
- **Purpose:** state the contribution precisely.
- **Problem:** "Real-Time American Sign Language Recognition and Video Calling System" implies an integrated system
  with real-time recognition in calls; the recognizer is not integrated (Correction A.1).
- **Cover:** isolated-word recognition, 100 classes, landmarks + recurrent networks; optionally "streaming prototype" and
  "video-calling application (separate)". Avoid "translation".
- **Evidence:** `src/components/Call.tsx`; `docs/step7_integration_design.md` (proposal only).

### B.2 Author block — INCOMPLETE
- Five authors, GL Bajaj Institute of Technology and Management, are printed. Four e-mails for five authors: author 2's
  footnote marker and e-mail ordering do not match (1, 2, 3, 5 listed; "Vatsal Mittal" (3) has no e-mail and
  "srivastavasakshi126" is labelled 3 though it appears to belong to author 4). **AUTHOR INPUT REQUIRED:** check
  the e-mail–author mapping and the corresponding-author marker.

### B.3 Abstract — RESTART
- **Purpose:** summarise what was done and found.
- **Accurate and keep:** motivation sentence; MediaPipe landmarks + PyTorch LSTM/BiLSTM; Hugging Face ASL dataset; isolated
  words; metrics include accuracy, precision, recall, F1, latency, FPS, duplicates, missed signs.
- **Inaccurate:** "detects … words … and provides text captions from the video communication"; "a sliding-window
  approach was investigated"; "three landmark configurations including … hands with pose and face"; "temporal window
  sizes of 30, 45, and 60 frames … was studied"; "precision, recall and F1 … for individual words" (per-class
  precision/recall are not reported; per-class F1 exists for baseline/E1/E2); "end-to-end pipeline with … Next.js
  and FastAPI was implemented"; "aim to find … best configuration" (no grid was run). No results are given.
- **Cover, in order:** context → task (100 isolated words; not continuous/translation) → data (4,838 accepted of
  5,131 candidates) → models (LSTM vs BiLSTM, 5 seeds; E1; E2) → key validation and one-time test results → latency →
  one-signer streaming result (descriptive) → limits (not signer-independent; one signer; not integrated).
- **Numbers and sources:** §C tables C.1, C.4, C.5, C.6, C.8, C.9.
- **Avoid:** "real-time captions in calls", "continuous", "signer-independent", "significant" (5 seeds: minimum
  two-sided sign-test p = 0.0625), calling E2 an "improvement", "rules fixed in advance" for all decisions (only E1/E2).

### B.4 Index Terms — KEEP
- "landmark-based recognition", "real-time inference", "WebRTC" are acceptable if the body states that WebRTC is a
  separate app. Consider adding "isolated sign recognition", "data augmentation".

### B.5 I. Introduction — UPDATE
- **Purpose:** problem, motivation, gap, aims.
- **Keep:** paragraphs 1–2 (communication barrier; landmark efficiency; temporal nature), paragraph on real-time
  constraints, the isolated-vs-continuous discussion.
- **Corrections:**
  1. Paragraph 1 (barriers for deaf/hard-of-hearing users) has **no citation** → citation gap (none of the supplied
     PDFs establishes prevalence; Bragg et al. 2019 (*external*, doi:10.1145/3308561.3353774) discusses the field and
     Deaf-community perspective).
  2. "MediaPipe + temporal networks [1], [4], [5]": [1] and [5] support it (MediaPipe Holistic + LSTM). **[4] does not**:
     Sign-Meet's method section uses a 17-keypoint pose estimator feeding a 7-layer MLP (its abstract says TCN; the
     paper is internally inconsistent).
  3. "LSTMs and Bidirectional LSTMs … [3], [4], [8], [10]": [3] uses a single-layer LSTM (detection, not recognition);
     [4] MLP/TCN; [8] graph convolutional network (GCAR); [10] bidirectional echo-state reservoirs, not LSTM. Use
     [1], [5], [6], [9] (MediaPipe/CNN + LSTM) and [2] (BiLSTM, low reliability) instead; add Hochreiter &
     Schmidhuber 1997 and Schuster & Paliwal 1997 / Graves & Schmidhuber 2005 for the models themselves.
  4. "TCNs, graph networks, Transformers … [3], [4], [8], [10]": cite [8] (graph) and [12] (Transformer); [4] only as a
     TCN *claim* in its abstract.
  5. "pose estimation and temporal networks in real-time video-conferencing [7]": [7] is a CSLR review with an
     **Expression of Concern**; the correct source for video-conferencing is [3] (Google detection blog).
  6. "integrated sign recognition into WebRTC … [12]": wrong — [12] is the Stack Transformer paper. The supplied source
     is [2] (low reliability, see E).
  7. "longer temporal windows increase recognition accuracy": **contradicted by your own data** (45 frames:
     val macro-F1 0.518 vs 0.600 at 30 frames, single seed; `reports/step4/results/experiment_results.csv`). State it
     as a hypothesis or remove.
  8. Final paragraph (planned H/HP/HPF × 30/45/60 grid, sliding window, Next.js + FastAPI integration) → rewrite to
     what was done: LSTM vs BiLSTM, E1, E2, hands-only (validation), exploratory 30 vs 45, one-signer streaming
     prototype; the call app is separate.
- **Author decision:** the 30-vs-45 comparison belongs in Results only (your decision), not as an objective.

### B.6 II. Related Work (lead paragraph) — KEEP
- General; add a citation for the survey statement (Rastgoo et al. 2021 or Koller 2020, both *external*-verified).

### B.7 II.A Landmark-Based SLR — UPDATE
- **No reference numbers** are attached to any study named here; add them.
- **Factual corrections against supplied PDFs:**
  - SLRNet [6]: the PDF describes MediaPipe Holistic (543 landmarks), 30-frame sequences, a CNN spatial block and a
    **stacked LSTM**; "temporal convolutional layers with a region-based segmentation" is **not supported** (no such
    wording found). It recognises letters (and, per the abstract, 10 words; the dataset section describes only
    letters — internal inconsistency).
  - Srivastava et al. [5]: MediaPipe Holistic, **spatial** normalisation of landmarks ("standard scale and position"),
    30 frames per video, LSTM; "temporal normalization procedure" is **not supported**. Task is effectively isolated
    recognition of 45 ISL signs despite the "continuous" title.
  - PBRC [10]: input is **MediaPipe hand-joint coordinates only**; "hand and body pose landmarks" is wrong (and
    contradicts your own Section III.A, which correctly says hand joints).
  - "Generalized Convolutional Associative Memories": wrong expansion. [8] defines **GCAR** as a "two-stream multistage
    graph convolution with attention and residual connection".
  - Stacked Spatio-temporal Attention [12]: skeleton keypoints, word-level and fingerspelling recognition — acceptable.
- **Normalisation paragraph:** acceptable; cite [5] for normalisation to scale/position; your own method uses
  shoulder-centred, shoulder-width-scaled coordinates (§B.20).
- **Keypoint-error sentence:** supported in spirit by [11] (SignRep argues landmark-only models suffer from missing and
  misdetected keypoints).

### B.8 II.B Temporal Deep Learning — UPDATE
- Add citation numbers. SLRNet [6] stacked LSTM over 30 frames — OK. "continuous ISL recognition systems use multiple
  LSTM layers" → [5] (and [1]). MobileNetV2-LSTM with attention → [9] (RGB, WLASL100, 5-fold CV 84.65 %) — OK.
- Sign-Meet [4] "employs a TCN": only its abstract says so; its method is an MLP — state the inconsistency or drop.
- GCAR "separable temporal convolutional layers" — UNVERIFIED wording (the PDF confirms graph convolution with
  attention and residual connections); check the PDF's method section before keeping "separable".
- PBRC: ESN reservoirs, readout trained by ridge regression, faster training (18.67 s vs 55+ min for a Bi-GRU) — OK.
- Closing sentence "different studies use predetermined sequence lengths" — fair.

### B.9 II.C Advanced Spatial-Temporal Models — UPDATE
- GCAR [8] two streams (joint and motion) with attention — OK; add [8].
- SSTAN [12]: name used in your paper; the PDF title is "Stack Transformer Based Spatial-Temporal Attention Model…".
  Check the acronym against the PDF before using "SSTAN".
- SignRep [11]: Hiera ViT, masked autoencoding, sign priors, RGB at inference — OK. "performance can decrease as the
  vocabulary becomes larger" — general; not tied to a specific finding → soften or cite a specific result.

### B.10 II.D Real-Time Recognition and Video Communication — UPDATE
- Google [3]: pose estimation + optical flow + single-layer LSTM for **signing detection**, active-speaker trigger;
  91.5 % accuracy at 3.5 ms per frame (blog figures) — OK. Note it is a blog post; the underlying SLRTP 2020 paper is
  not supplied (UNVERIFIED metadata).
- "SIGN-MEET … built upon the same ideas": UNVERIFIED (the PDF does not cite or build on [3]); state only what it
  describes, and note it has no dataset or evaluation protocol.
- "Multiple works have demonstrated MediaPipe + BiLSTM + WebRTC": only [2] in your set, and it has no stated dataset,
  split or protocol → "one system has been described [2]".

### B.11 II.E Synthesis — UPDATE
- Keep the isolated-vs-continuous argument.
- Final paragraph says the study "isolates the effect of different MediaPipe representations … window sizes … 30, 45,
  60" — not what was done. Rewrite to your actual comparisons (§C).
- "The papers reviewed lack detailed discussion of streaming effects" — fair for the reviewed set; phrase as "among the
  works reviewed".

### B.12 III.A Research Gap — UPDATE
- Paragraph 1: "[3], [4]" for video-conferencing — OK ([3] detection, [4] concept). "[2] tested on a live video call" —
  OK but flag reliability. "[1], [5], [6]" MediaPipe + LSTM — OK.
- Paragraph 2: "PBRC [10] processes only hand-joint positions" — correct (contradicts II.A; fix II.A).
- "we did not identify an evaluation of whether pose and face landmarks justify their cost" — your study evaluates only
  hands-only vs hands + full pose and upper-body vs full pose (validation); **face not evaluated**. Adjust the gap so
  that it is one you actually address.
- "[7]" for "real-time processing is an open challenge" — the statement does appear in [7], but [7] carries an
  **Expression of Concern** for uncredited copying of Alyami, Luqman & Hammoudeh, "Reviewing 25 years of continuous
  sign language recognition research…", Information Processing & Management, 2024 (named in the notice;
  bibliographic details UNVERIFIED). Replace [7] with that review after verifying it, or with Koller 2020.
- Research questions 1–2: RQ1 mentions sliding-window sizes and "adapted to continuous online recognition" → restate
  around architectures, augmentation, landmark subsets, and latency; RQ2 (post-processing) is answerable from the
  one-signer replay (§C.10).

### B.13 III.B Scope — UPDATE
- Keep: isolated words only; no sentence translation; fixed vocabulary.
- Remove: "During a video call, the model is applied … through a sliding window, and the recognized words are displayed
  as text captions" (not implemented). Replace with: the recognizer runs as a separate Python streaming prototype that
  segments signs by hand activity; the call app does not use it.

### B.14 III.C Contributions — RESTART
- Items 1, 2, 4, 6, 7 are not supported as written (2,208-word training; face configuration; 60-frame window; 18
  configurations; confusion matrices across all configurations; integration with captions). Item 3 (LSTM vs BiLSTM
  under identical conditions) is supported (`docs/step5_vocabulary.md`, seed study). Item 5 is partially supported
  (cooldown/threshold by replay; one signer).
- **Cover instead (supported):** documented landmark pipeline and 100-class subset with rejection rules; five-seed
  LSTM vs BiLSTM with bootstrap CIs; pre-registered E1 and E2 with validation decision rules and one-time final test;
  precision/recall from saved predictions; latency measurements and one-signer streaming evaluation with post-processing;
  separate WebRTC calling app (not integrated).
- Remove the section roadmap's reference to "Section VIII" describing integration, and "Section X" unless you add it.

### B.15 IV.A Dataset Description — UPDATE
- **Verified:** Hugging Face `akasheroor/American-Sign-Language-Dataset`, pinned revision
  `e7979505c0dff7072ef36d45b3cddfffb50ba871` (`configs/step1_data.yaml`). Dataset card: 108,618 videos, 2,208 words, ≥ 30 per
  word, `dataset.csv` with `word`, `video_path`; MIT licence; credit to Zahid Yasin Mittha (2025); "scraped, collected
  from multiple sources" with sources not named (*external*, dataset card).
- **Your own reconciliation differs:** 108,616 files and **2,207** classes after excluding one listed-but-missing file
  and one duplicate CSV row; the CSV columns are `word`, `videos` (bare filenames), not `video_path`
  (`reports/step1/eda_report.md`; `reports/step1/stats.json`). Report both: card figures and reconciled figures.
- "duplicate and corrupted videos had been removed beforehand" — the card's claim; your audit found byte-identical
  clips under different labels and augmented copies (`docs/step5_vocabulary.md`: 968 classes share byte-identical clips
  with another label). Do not repeat the card's claim without that caveat.
- **Missing:** the 100-class subset and how it was selected (metadata rules; no model results used;
  `docs/step5_vocabulary.md`, `configs/vocabulary_100.json`).

### B.16 IV.B Video Preprocessing — UPDATE
- **Keep:** decode; resample to 30 FPS; ≥ 30 decoded frames; interpolation-based gap repair; logged rejections.
- **Corrections:**
  - Resampling to 30 FPS is applied to **landmark sequences** after MediaPipe (`docs/step2_methodology.md` step 3), not to
    the video before extraction.
  - Rules 3–4 apply **inside the active signing interval**, not to all frames; rest frames are not counted
    (`docs/step2_methodology.md`).
  - Missing rules: **no active signing interval** (no run of ≥ 3 consecutive hand frames) and **shoulders never valid**.
  - "windows of 30, 45 and 60 frames correspond to about 1, 1.5 and 2 seconds" — **not true for the implemented
    method**: the active interval is resampled to W frames, so W is a temporal resolution, not a duration
    (`docs/step3_methodology.md` §8).
  - Placeholders "[N]/[M]/[a]–[d]": fill with the 100-class figures: **5,131 candidates; 4,838 accepted; 293 rejected
    = 150 both hands missing > 30 % of the active interval, 101 < 30 decoded frames, 36 both hands missing > 15
    consecutive frames, 6 no active signing interval; 0 decode errors** (`data/sequences/dev100/index_summary.json`;
    `docs/step5_vocabulary.md` Stage 2A). The full 108,618 collection was **not** processed.
  - "Frames are not resized … normalized coordinates … resolution does not affect scale": landmarks are converted to
    **pixel units** (x·W, y·H, z·W) before shoulder normalisation (`docs/step2_methodology.md` step 8). Adjust.

### B.17 IV.C Landmark Extraction — UPDATE
- **Verified:** MediaPipe Holistic 0.10.21, legacy API `mp.solutions.holistic` (`src/asl/landmarks/extract.py:38`), CPU,
  `model_complexity=1`, fresh instance per clip; 33 pose (+ visibility), 21 per hand, 468 face-mesh landmarks
  (`docs/step2_methodology.md`).
- **Corrections:**
  - Face landmarks were **not stored** for the 100-class store (`docs/step5_vocabulary.md` "Storage profile
    hands_pose_f32"); HPF was never trained.
  - Feature dimensions: **H = 128** (correct); **HP = 261**, not 228 — 128 hand + 99 pose coordinates + 1 pose presence
    + **33 pose visibility** values; **HPF = 1,666**, not 1,633 (`docs/step2_methodology.md` "Configurations";
    `docs/step5_vocabulary.md` feature table). Presence flags are **one per landmark group**, not per landmark
    component.
  - "x and y normalized to the image size" → then converted to pixels (see IV.B).
- **Keep:** interpolation of gaps ≤ 5 frames; zero-fill + presence 0 for longer and edge gaps; the rationale paragraph.
  "five-frame threshold fixed before any experiments" — plausible but UNVERIFIED (documented as part of the approved
  v2 methodology; no timestamped pre-registration).

### B.18 IV.D Normalization and Sequence Construction — UPDATE
- **Keep:** Eq. (1) with c = shoulder midpoint (pose 11/12) and s = shoulder distance. Specify **2-D** distance.
- **Corrections:**
  - Shoulders invalid (visibility < 0.5) → reuse last valid frame; leading frames use first valid frame; **never valid →
    clip rejected** (not "handled by the missing-landmark policy") (`docs/step2_methodology.md` step 8).
  - Sequence construction: the **active signing interval** (first to last run of ≥ 3 consecutive hand frames) is
    resampled to W frames with p_k = start + k(L−1)/(W−1) and **linear interpolation** where the group is present
    in both neighbours (nearest frame otherwise); no frame duplication, no padding (`docs/step3_methodology.md` §3).
    "whole clip … nearest available frame … selects some frames more than once" is wrong on all three counts.
  - Labels: 0…99 in Unicode order of the 100-class vocabulary (`data/sequences/label_mapping_dev_100.json`).

### B.19 IV.E Dataset Split — UPDATE
- **Corrections:** the 70/15/15 split (seed 42) was made **once on the full reconciled collection before any
  extraction**, grouping byte-identical files (SHA-256) with their augmentation families so they cannot cross splits,
  stratified by class, then frozen with a checksum; rejected clips keep their assignment (`README.md` "Rules applied in
  Step 1"; `configs/step1_data.yaml`; `scripts/s1_04_make_split.py`). It was **not** made "on the processed clips".
- Accepted counts: **3,407 / 725 / 706**; every class in every split; minimum 26 / 4 / 4 per class
  (`data/sequences/dev100/index.parquet`). Median 33 training clips per class (26–45) (`reports/paper/tables_figures_v2/T1.csv`).
- **Keep:** not signer-independent (no signer IDs). Add: the only grouping is 4 filename collections (2,917 / 1,146 /
  708 / 67), all present in every split (`docs/evaluation_protocol.md` §1).

### B.20 V.A Overall Architecture — RESTART
- Describes browser MediaPipe, rolling buffer, FastAPI, captions — not implemented.
- **Cover instead:** offline pipeline (Fig. 1 panel A of `reports/paper/draft/F1_pipeline.png`) and the online Python
  prototype: OpenCV capture (threaded), MediaPipe Holistic per frame, hand-activity segmentation, same preprocessing as
  training, BiLSTM, optional post-processing, word label (`docs/step6_inference.md` §3; `configs/inference.yaml`).
  Parity verified: features bit-identical to training on 20 dataset clips; replay reproduced recorded events.
- The call app is separate (B.32).

### B.21 V.B Landmark Configurations — UPDATE
- Rationale paragraph is good (keep). Change to the configurations actually compared: hands-only (128), hands + full
  pose (261), hands + upper-body pose (221, pose landmarks 23–32 removed). Face: motivation can stay as background, but
  state it was not evaluated.

### B.22 V.C Temporal Windowing — RESTART
- Stride-5 sliding window, 1/1.5/2 s windows and "queried six times per second" are not implemented.
- **Cover instead:** resampling to W = 30 (temporal resolution); exploratory 45-frame single-seed LSTM; streaming uses
  segmentation: start after 3 consecutive hand frames with 0.5 s pre-roll, end after 0.4 s without hands, cut at 4 s
  (`configs/inference.yaml`).

### B.23 V.D LSTM — KEEP (minor update)
- Eqs. (2)–(7) are the standard LSTM; cite Hochreiter & Schmidhuber 1997 (*external*). The PDF extraction shows
  Eqs. (2)–(5) run together on one line — check layout.
- Verified: 2 layers, 128 hidden, dropout 0.30 between layers **and** before the classifier; classifier on final
  hidden state; 100 outputs; 345,188 parameters at 261 features (`docs/step4_methodology.md` §4; run `metadata.json`).

### B.24 V.E BiLSTM — KEEP (minor update)
- Verified: forward state after last step ‖ backward state after first step = 256 → linear; 821,348 parameters
  (261 features); 780,388 (E2, 221); 685,156 (hands-only, 128). "Roughly twice the recurrent work" — consistent with
  measured 1.72 vs 0.89 ms median per sign in training runs (`reports/step4/results/experiment_results.csv`).
- Cite Schuster & Paliwal 1997 / Graves & Schmidhuber 2005.

### B.25 V.F Post-processing — RESTART
- The 0.75 threshold, 5-output consensus with ≥ 3 votes and "suppress until a different word" rule are **not
  implemented**. Implemented (off by default; `configs/inference.yaml`, `docs/step6_inference.md`): confidence
  threshold (evaluated at 0.4 / 0.57 / 0.7), **cooldown** (segment starting within 0.5 or 1.0 s of the previous one is
  classified but not emitted), majority-vote smoothing (**implemented, not evaluated**). The 0.57 threshold was chosen
  on validation for 80 % precision (`reports/step6/calibration_hands_pose_w30_bilstm_e100_s3.json`).

### B.26 V.G Real-Time Inference — RESTART
- Browser extraction, FastAPI, per-session buffers, forwarding to remote participant — not implemented.
- **Cover:** the Python `StreamingRecognizer` (B.20/B.22); the live pipeline accepts only 261-feature models, so E1/E2
  were evaluated by offline replay; live sessions used the baseline BiLSTM seed 3.

### B.27 VI.A Experimental Configurations — RESTART
- 18 configurations, RF baseline, synthetic streams — none run. **Cover the actual experiments** (§C.3).

### B.28 VI.B Training Configuration — UPDATE
- **Keep:** PyTorch, cross-entropy, AdamW lr 1e-3, wd 1e-4, ReduceLROnPlateau factor 0.5 patience 3, early stopping
  patience 7 on validation macro-F1, best-epoch weights, batch 64.
- **Corrections:** scheduler monitors **validation macro-F1**, not loss; max **100** epochs (one early run 50);
  batch size used = **64** (fill placeholder); **5 seeds (42, 1, 2, 3, 4)**; deterministic algorithms; **CPU, 4
  threads** — not MPS — on an arm64 laptop with 8 logical cores and 8 GB RAM, macOS 27.0.1, Python 3.12.6,
  PyTorch 2.14.1, NumPy 1.26.4 (run `metadata.json` "environment"; `reports/step6/latency_*.json`). "Apple M2" appears
  only in documentation → UNVERIFIED (AUTHOR INPUT REQUIRED to state it). Remove the RF settings placeholder (RF not
  run). Remove "accuracy would be misleading with 2,208 classes … most frequent word" (100 roughly balanced classes;
  macro-F1 is the primary metric by protocol, `docs/evaluation_protocol.md` §2).
- **Add:** E1 augmentation parameters (p = 0.5; hold 10–50 %; tempo 2/3–3/2; loop tolerance 0.15 shoulder widths;
  training only) (`src/asl/sequences/augment.py`; E1 `metadata.json`); E2 removes 40 columns (197–226, 251–260).
- **Add:** single-seed development variations (50 vs 100 epochs; dropout 0.5 → val macro-F1 0.599 vs 0.600; 45
  frames) — so "no hyperparameter search" must not be claimed.

### B.29 VI.C Evaluation Metrics — UPDATE
- **Keep:** accuracy; per-class P/R; macro averaging; Eq. (8) (it equals the implemented macro-F1: mean of per-class F1
  over the 100 classes, zero_division = 0) (`src/asl/models/metrics.py`).
- **Corrections:** "2,208 classes, confusion matrix impractical" → 100 classes; "inference latency … process a single
  frame" → measured per **sign** (one 30-frame window, batch 1); end-to-end latency via network/FastAPI not measured;
  "[mean and 95th percentile] over [number]" → **median and p95**: model forward 500 timed runs after 20 warm-ups;
  MediaPipe on 1,595 frames of 20 test clips; segment close on 20 segments; live label delay from session logs
  (`docs/step6_inference.md` §4).
- **Streaming metrics:** DPR/MSR on synthetic streams were **not computed**. What exists: attempts correct, duplicates
  (attempts with duplicate emissions; duplicate emissions), missed (no segment), coverage and precision of answers
  under thresholds, on 92 core webcam attempts of one signer (`reports/paper/postprocessing/summary_by_setting.csv`).
  Eqs. (9)–(10) must be removed or redefined to match these.
- **Add:** uncertainty — 5 seeds mean ± SD; class-stratified bootstrap 95 % CIs, 2,000 resamples, seed 0; paired
  bootstrap; exact McNemar on webcam attempts (`docs/evaluation_protocol.md` §2). Add decision rules (§3 improvement;
  §3a non-inferiority).
- **Add:** precision/recall averaging: macro (unweighted over 100 classes) and weighted; micro = accuracy.

### B.30 VI (missing subsection) Evaluation protocol and test-set use — INCOMPLETE
- Must be added; see §D for full inventory and required disclosure.

### B.31 VII. Results and Discussion — INCOMPLETE (empty)
- Write from §C only. Suggested order: data → LSTM vs BiLSTM → representation (validation) → sequence length
  (exploratory) → E1/E2 (validation, rules) → one-time final test (accuracy, macro-F1, macro-P, macro-R) → overfitting →
  robustness diagnostics → latency → one-signer streaming → post-processing. Separate validation, final test,
  exploratory and webcam results explicitly.
- **Interpretation guidance:** BiLSTM > LSTM in all seeds (not "significant" at seed level); E1 met the improvement rule;
  E2 met only non-inferiority on validation although its test scores were higher — do not re-label E2 as an improvement;
  hands-only beat hands + full pose on validation (no augmentation, not tested on test); MediaPipe dominates latency;
  webcam accuracy 21–24 points below test accuracy but on 8–10 signs and development data — not like-for-like.
- **Discussion points available:** precision > recall by 0.022–0.028 (errors spread across predicted classes);
  overfitting (train ≈ 100 %, gap 28.6–32.5 points); calibration over-confident on webcam.

### B.32 VIII. Real-Time Video Communication System — RESTART
- Every functional claim except "Next.js front end" and "WebRTC peer-to-peer" is unsupported: no MediaPipe in the
  browser, no FastAPI, no WebSocket, no data channel, no captions, no recognition server.
- **Verified facts:** Next.js 16.2.10, React 19.2.4 (`package.json`); one-to-one WebRTC; signalling via
  `GET/POST/DELETE /api/signaling/[roomId]` with MongoDB-stored presence and signalling documents (TTL); STUN servers
  `stun.l.google.com:19302`, `stun1.l.google.com:19302` (`src/components/Call.tsx:117–120`); accounts with bcrypt
  (`package.json`, `src/lib/auth.ts`); media peer-to-peer (`README.md`). Recognizer not integrated
  (`docs/step7_integration_design.md` is a proposal).
- Present this section as the communication platform built, with integration as future work. Screenshots/deployment:
  AUTHOR INPUT REQUIRED.

### B.33 IX. Limitations — UPDATE
- **Keep:** 1 (not signer-independent), 2 (no grammar/translation), 3 (fixed vocabulary), 4 (landmark quality).
- **Correct:** item 4's "rejection statistics … reflect the frequency of this issue in the test set" → rejections were
  across all splits (test: 61 of 767 candidates rejected = 36 + 7 + 18; `index_summary.json`). Items 5–6 (synthetic
  streams; duplicate-suppression rule) describe things not implemented → remove.
- **Add:** test-set reuse (§D); one signer/camera/room; development webcam data; one hardware platform; not integrated;
  live pipeline 261-only (E1/E2 offline); face/60-frame/RF not done; hands-only not tested with augmentation; overfitting;
  five seeds only (no seed-level significance); per-class P/R rest on ~7 test clips.

### B.34 X. Conclusion (missing) — INCOMPLETE
- Referenced in the roadmap but absent. Cover: what was built and found (§C), what it does not establish, future work
  (signer-independent multi-signer evaluation with consent; face and longer-window studies; hands-only + augmentation;
  E2 in live pipeline; WebRTC integration and in-call latency; target-domain calibration; continuous recognition).

### B.35 Fig. 1 — INCOMPLETE
- "Fig. [1]" is a placeholder; no figure is included. A verified pipeline figure exists:
  `reports/paper/draft/F1_pipeline.{png,svg,pdf}` (`scripts/paper_02_pipeline_figure.py`). Its Evaluation box says
  "earlier test uses disclosed" — accurate only after §D is disclosed. It labels FastAPI as "NOT built" (correct).

### B.36 Equations — UPDATE
- Eq. (1) keep (add 2-D distance). Eqs. (2)–(7) keep (layout). Eq. (8) keep. Eqs. (9)–(10) remove or redefine (no
  synthetic streams). Optional: add the resampling formula p_k = start + k(L−1)/(W−1).

### B.37 References — UPDATE (see §E for each entry)

### B.38 Ethics / consent, data & code availability, acknowledgements — INCOMPLETE (absent)
- AUTHOR INPUT REQUIRED (§G).

---

## C. Consolidated results and methodology (verified)

### C.1 Data
| item | value | source |
|---|---|---|
| Dataset | `akasheroor/American-Sign-Language-Dataset` @ `e7979505c0dff7072ef36d45b3cddfffb50ba871`, MIT licence | `configs/step1_data.yaml`; HF card (*external*) |
| Card figures | 108,618 videos, 2,208 words | HF card (*external*); `reports/step1/eda_report.md` |
| Reconciled | 108,616 files, 2,207 classes | `reports/step1/stats.json` |
| Frozen split | 70/15/15, seed 42, group-aware (SHA-256 + augmentation families), stratified | `configs/step1_data.yaml`; `README.md` |
| 100-class candidates | 5,131 | `data/sequences/dev100/index_summary.json` |
| Accepted / rejected | 4,838 / 293 | same |
| Rejections | 150 hands-missing fraction; 101 < 30 frames; 36 hands-missing run; 6 no active interval | same (`excluded_by_split_reason`, summed) |
| Accepted split | 3,407 / 725 / 706; min per class 26 / 4 / 4; median train 33 (26–45) | `index.parquet`; `tables_figures_v2/T1.csv` |
| Signer IDs | none; 4 filename collections in every split | `docs/evaluation_protocol.md` §1 |

### C.2 Features and models
| item | value | source |
|---|---|---|
| MediaPipe | Holistic 0.10.21, legacy `mp.solutions.holistic`, CPU, model_complexity 1 | `src/asl/landmarks/extract.py:38`; `docs/step2_methodology.md` |
| Features | hands 128; hands + pose 261; E2 221 (pose 23–32 removed: cols 197–226, 251–260) | `docs/step5_vocabulary.md`; E2 `metadata.json` |
| Sequence | active interval resampled to 30 frames (one run 45) | `docs/step3_methodology.md` |
| LSTM / BiLSTM | 2 × 128, dropout 0.3; 345,188 / 821,348 params (261); E2 780,388; hands-only 685,156 | run `metadata.json` |
| Training | AdamW 1e-3 / 1e-4, CE, batch 64, ≤ 100 epochs, LR ×0.5 patience 3 and early stop 7 on val macro-F1, best checkpoint, seeds 42/1/2/3/4, CPU 4 threads | `docs/step5_vocabulary.md` Stage 2B; `metadata.json` |
| E1 | p 0.5: hold start/end 10–50 %, tempo 2/3–3/2, loop repeat (tol 0.15); training only | `src/asl/sequences/augment.py`; `reports/e1/README.md` |

### C.3 Experiments actually run
| experiment | runs | test evaluated |
|---|---|---|
| hands_pose_w30_lstm (50 ep) | 1 | yes (development) |
| hands_pose_w30_lstm_e100 + s1–s4 | 5 | yes (development) |
| hands_pose_w30_lstm_e100_do05 (dropout 0.5) | 1 | yes (development) |
| hands_pose_w45_lstm_e100 | 1 | yes (development) |
| hands_pose_w30_bilstm_e100 + s1–s4 (baseline) | 5 | yes (development); predictions reused in final |
| E1 …_augtemp + s1–s4 | 5 | once, final |
| E2 …_augtemp_upperpose + s1–s4 | 5 | once, final |
| hands_w30_bilstm_e100 + s1–s4 (hands-only) | 5 | **no** |
Not run: hands + pose + face, 60 frames, Random Forest, 18-configuration grid, synthetic streams. Source: `experiments/dev100_hp/`.

### C.4 LSTM vs BiLSTM (hands + pose, 30 frames, 5 seeds) — test computed during development
| split | model | accuracy % | macro-F1 |
|---|---|---|---|
| val | LSTM | 60.0 ± 3.2 | 0.594 ± 0.033 |
| val | BiLSTM | 66.6 ± 0.9 | 0.662 ± 0.010 |
| test | LSTM | 57.5 ± 3.8 (CI 54.8–60.3) | 0.570 ± 0.043 |
| test | BiLSTM | 63.9 ± 1.0 (CI 61.1–66.6) | 0.634 ± 0.010 |
| test paired | BiLSTM − LSTM | +6.3 pts (+4.9 to +7.7) | +0.064 (+0.049 to +0.078) |
Source: `reports/e0/bootstrap/seed_mean.csv`, `bilstm_minus_lstm.csv`. BiLSTM better in all 5 seeds; sign test p = 0.0625 (minimum possible) (`docs/step5_vocabulary.md`).

### C.5 Validation-only comparisons
| comparison | mean val macro-F1 | paired change (95 % CI) | seeds | source |
|---|---|---|---|---|
| hands-only (128) vs hands + full pose (261), BiLSTM, no aug. | 0.685 vs 0.662 | +0.023 (+0.007 to +0.040) | 5/5 higher; 59 classes up, 41 down | `reports/paper/hands_only/comparison/summary.json` |
| E1 vs baseline | 0.691 vs 0.662 | +0.028 (+0.017 to +0.041); unrounded 0.02848 | 5/5; improvement rule met | `reports/e1/comparison/summary.json` |
| E2 vs E1 | 0.701 vs 0.691 | +0.010 (−0.002 to +0.023); unrounded ≈ 0.0102 | 4/5; non-inferiority met; improvement not met | `reports/e2/comparison/summary.json` |
| Train − val accuracy gap | baseline 32.5, E1 30.1, E2 28.6 pts | — | — | E1/E2 `per_seed.csv` |
Display convention: macro-F1 to 3 decimals; the E1 gain 0.02848 displays as +0.028 (earlier summaries' +0.029 came from rounding twice).

### C.6 Exploratory (single seed, LSTM)
| setting | val macro-F1 | test macro-F1 (development) | source |
|---|---|---|---|
| 30 frames | 0.600 | 0.612 | `reports/step4/results/experiment_results.csv` |
| 45 frames | 0.518 | 0.500 | same |
| dropout 0.5 (30 frames) | 0.599 | 0.586 | same |

### C.7 One-time final test (706 clips, 5 seeds; baseline predictions reused)
| model | accuracy % | macro-F1 | source |
|---|---|---|---|
| Baseline BiLSTM | 63.9 ± 1.0 (61.1–66.6) | 0.634 ± 0.010 (0.600–0.656) | `reports/paper/final_test/seed_mean.csv` |
| E1 | 66.1 ± 1.8 (63.5–69.0) | 0.658 ± 0.019 (0.625–0.681) | same |
| E2 | 67.8 ± 1.6 (65.3–70.6) | 0.675 ± 0.017 (0.643–0.698) | same |
| E1 − baseline | +2.3 pts (+1.1 to +3.6) | +0.024 (+0.012 to +0.038) | `final_test/paired.csv` |
| E2 − E1 | +1.6 pts (+0.4 to +2.8) | +0.016 (+0.004 to +0.030) | same |
| E2 − baseline | +3.9 pts (+2.5 to +5.3) | +0.041 (+0.026 to +0.057) | same |
All steps better in 5/5 seeds; per class E2 > baseline in 66 classes, < in 34 (`final_test/per_class.csv`).

### C.8 Precision / recall (from saved predictions; no new inference)
| model | macro-P | macro-R | macro-F1 | weighted-F1 |
|---|---|---|---|---|
| LSTM (development predictions) | 0.596 ± 0.051 | 0.574 ± 0.040 | 0.570 ± 0.043 | 0.572 ± 0.042 |
| Baseline | 0.665 ± 0.012 | 0.637 ± 0.011 | 0.634 ± 0.010 | 0.636 ± 0.010 |
| E1 | 0.689 ± 0.021 | 0.661 ± 0.019 | 0.658 ± 0.019 | 0.659 ± 0.019 |
| E2 | 0.703 ± 0.018 | 0.678 ± 0.017 | 0.675 ± 0.017 | 0.675 ± 0.016 |
Source: `reports/paper/final_test_prf/seed_mean.csv` (`scripts/paper_05_precision_recall.py`); 100 reconciliation checks, max |diff| 0 (`reconciliation.csv`).

### C.9 Latency (one arm64 laptop, CPU, 4 threads)
| component | median | p95 | source |
|---|---|---|---|
| MediaPipe Holistic per frame | 28.5 ms | 29.7 ms | `reports/step6/latency_hands_pose_w30_bilstm_e100_s3.json` |
| BiLSTM forward, batch 1 (per sign) | 1.68 ms | 1.80 ms | same |
| Training-run medians LSTM / BiLSTM | 0.89 / 1.72 ms | — | `experiment_results.csv` |
| Segment close (preprocess + model) | 4.1 ms | 4.9 ms | latency json |
| Streaming push per frame | 0.007 ms | 0.009 ms | latency json |
| Live processed FPS | 15.0 (unthreaded) → 29.8–29.9 (threaded) | — | session `summary.json` |
| Label delay after hands lowered | 0.44–0.46 s | — | session `attempts.csv` |
Not measured: in-call latency, browser, network, other hardware.

### C.10 One-signer webcam (DEVELOPMENT data; not generalization)
| item | value | source |
|---|---|---|
| Core replay set | 92 attempts = 104 core − 12 "signed during WAIT"; sessions S3, S4 (10 signs), 5a, 5b, 5c-1, 5c-2 (8 signs; BLIND, DIRECTOR dropped) | `reports/e0/webcam_replay_set/manifest.csv` |
| Metric | strict top-1: first emitted segment whose hand run starts in the GO window; mean over 5 seeds | `docs/step6_webcam_evaluation.md`; `docs/evaluation_protocol.md` §4 |
| Top-1 (top-5) | LSTM 29.8 (57.2); BiLSTM 42.4 (63.0); E1 43.9 (66.5); E2 43.7 (69.6) | `reports/e0/webcam_replay_set/summary_by_model.csv`; `reports/e1/comparison/webcam_replay_paired.csv`; `reports/e2/comparison/webcam_replay_paired.csv` |
| E1/E2 vs baseline | no seed significant (exact McNemar) | same |
| E1/E2 status | offline replay only (live pipeline accepts 261 features) | `docs/step6_inference.md` |
| Live sessions (baseline seed 3) | S3 13/20, S4 9/20 correct; 0 missed; 9 and 2 attempts with duplicates | session summaries |
| Post-processing replay (baseline) | none: 37/92 correct, 14 duplicates; cooldown 0.5/1.0 s: 0 duplicates, answers unchanged; threshold 0.4: 76.1 % coverage / 50.0 % precision; 0.57: 54.3 / 60.0; 0.7: 43.5 / 70.0 | `reports/paper/postprocessing/summary_by_setting.csv` |
| Calibration (seed 3) | ECE val 0.093, test 0.119; threshold 0.57 → test 77.7 % precision at 74.9 % coverage | `reports/step6/calibration_hands_pose_w30_bilstm_e100_s3.json` |
| Never recognised | AFRICA, COMPUTER (all 20 models); FORBID once in 240 model-attempts | `reports/paper/draft/webcam_details/summary.json` |

### C.11 Robustness (validation, accuracy change vs clean, pts)
hold_start −17.1 → −6.8 (E1); repeat_x2 −10.9 → −8.0; drop_minor_hand −29.2 (not reduced); mirror −17.4; hips_visible
−3.0 → 0.0 (E2). Source: `reports/paper/tables_figures_v2/T11.csv`.

---

## D. Test-set-use audit

| # | use | runs / items | type | source |
|---|---|---|---|---|
| 1 | Test metrics computed at the end of development training runs | 13 runs (C.3, "yes (development)") | **performance**, reported | `reports/step4/results/experiment_results.csv` (rows with test values) |
| 2 | Bootstrap CIs on the saved test predictions of those runs | analysis, no inference | performance analysis | `reports/e0/bootstrap/` |
| 3 | Confidence calibration and threshold check on test (candidate seed 3) | 1 | **performance** (ECE, precision/coverage at 0.57) | `reports/step6/calibration_*.json` |
| 4 | Latency/parity: features and predictions on 20 test clips compared with stored values | 1 | equality/parity only | `scripts/s6_02_measure_latency.py`; `reports/step6/latency_*.json` |
| 5 | Simulated live replay of 3 test clips (bottle, fruit, calm) | 3 | pipeline check; correctness was observed (3/3) | `reports/step6/webcam_eval/*_video_replay_dataset-test-clip`; `docs/step6_webcam_evaluation.md` |
| 6 | Export: logits on 64 real test sequences compared with source checkpoint | 1 | numerical equality only | `scripts/s4_04_export_model.py:61,109–125`; `models/candidates/hands_pose_w30_bilstm_e100_s3/export_report.json` |
| 7 | Export: argmax on all 706 test sequences compared with stored predictions | 1 | equality only (`test_sequences_compared: 706`) | same |
| 8 | One-time final evaluation of E1 and E2 (10 runs, inference once) after development closed; baseline predictions reused | 1 event | **final performance** | `reports/paper/final_test/summary.json`; `scripts/e3_01_final_test.py` |
| 9 | Precision/recall from saved predictions | 0 inference | analysis | `reports/paper/final_test_prf/` |
| — | Hands-only runs | 0 | — | `metadata.json` `test_evaluated: false` |

**Disclosure required:** (a) the baseline's and LSTM's test results were known during development (items 1, 3), but no
model choice used them (`docs/evaluation_protocol.md` §1); (b) items 4–7 exercised the test split for correctness only;
(c) the evaluation protocol with E1/E2 decision rules is dated 2026-10-23, the same day the first E1 run started
(12:13 UTC, E1 `metadata.json`) — the order within that day is UNVERIFIED; (d) E1/E2 were evaluated once.
**Correction affecting other files:** `reports/paper/tables_figures_v2/T12.csv`/`TABLES.md` and the earlier draft list
"13 runs and three auxiliary checks" and omit items 6–7; `scripts/paper_03_verify_claims.py` hard-codes "three auxiliary
checks"; Fig. 1's Evaluation box says "earlier test uses disclosed". Update all consistently if you reuse them.

---

## E. Literature and citation matrix

### E.1 Supplied references as numbered in your paper
| ref | supplied PDF | what it establishes | metadata issues | reliability / guidance |
|---|---|---|---|---|
| [1] Anithadevi et al. | MediaPipe-LSTM-Enhanced… (OCR only) | MediaPipe Holistic + stacked LSTM on ~2,000 self-collected ISL videos; reports 97 % accuracy | *Engineering Reports* 7, e70142, 2025, doi:10.1002/eng2.70142 printed; "no. 7" UNVERIFIED | Mainstream publisher; internal inconsistencies (30 vs 26 classes; 80:20 vs 92:8 split; 97 % accuracy vs F1 94.88 %); not signer-independent. Cite as feasibility, not as a comparable result. Numbers read by OCR — spot-check. |
| [2] Saravana Perumal et al. | sign_language_detection_in_video_and_real_time_video_calls.pdf | MediaPipe + CNN-BiLSTM-attention with WebRTC; no dataset/protocol | Your year "2026" — PDF: Nov–Dec 2025, Vol 5(06), pp. 195–199, ISSN 2583-0368, doi:10.59256/ijrtmr.20250506025 | **Low reliability**: no dataset, split or protocol; its own reference list misattributes the 1997 LSTM paper. Cite only as "a system has been described", or omit. |
| [3] Moryossef (blog) | Developing Real-Time… (web printout) | Signing **detection** (not recognition) for video calls; pose + optical flow; single-layer LSTM 91.5 % at 3.5 ms | Blog post by Amit Moryossef, Google Research, 1 Oct 2020 — single author ("et al." wrong); URL research.google/blog/developing-real-time-automatic-sign-language-detection-for-video-conferencing/ | Reliable source for the demo; underlying SLRTP 2020 paper not supplied (UNVERIFIED metadata). |
| [4] Chandrasekaran et al. | K_S_Chandrasekaran_et_al_V7I5003.pdf | Concept of a meeting platform with ISL recognition, speech-to-text, avatar | Journal printed as "International Journal of Multidisciplinary Research Transactions (IJMRT)" (your "Int. J. Multidiscip. Res. Technol." is wrong); Vol 7 Issue 5, 2025, pp. 27–35; doi:10.5281/zenodo.15348353 (Zenodo) | **Low reliability**: abstract (TCN) contradicts method (MLP); 98 % from one binary confusion matrix without data origin. Do not cite as evidence of TCN or of accuracy. |
| [5] Srivastava et al. | 2411.04517v1.pdf | MediaPipe Holistic, normalisation, 30-frame sequences, LSTM; 45 ISL signs; 88.23 % on 95:5 random split | PDF prints doi:10.1007/s11277-024-11356-0 but **no year/volume/pages**; your "vol. 137, pp. 1455–1468, 2024" is UNVERIFIED — check the Springer page | Reliable venue; isolated in practice despite "continuous" title. |
| [6] Kamble (SLRNet) | 2506.11154v1.pdf | MediaPipe Holistic 543 landmarks, 30 frames, CNN + stacked LSTM; 86.7 % val accuracy | arXiv:2506.11154, 2025 — OK | Preprint; inconsistencies (26 letters vs "+10 words"; 78 ms vs < 70 ms latency). Your II.A description ("temporal convolutional layers with region-based segmentation") is **not supported**. |
| [7] Khan et al. | Deep_Learning_Approaches… | CSLR survey; lists real-time processing and signer variability as open | IEEE Access 13, 55524–55544, 2025, doi:10.1109/ACCESS.2025.3554046 (your entry lacks venue) | **IEEE Expression of Concern** (uncredited copying of Alyami, Luqman & Hammoudeh, IP&M 2024). **Do not cite**; replace with the original review (verify its metadata) or Koller 2020. |
| [8] Miah et al. | Sign_Language_Recognition_Using_Graph… | GCAR: two-stream multistage graph convolution with attention and residual connection; WLASL100 top-1 63.25 % (abstract's 90.31 % is top-k) | Add IEEE Access 12, 34553–34569, 2024, doi:10.1109/ACCESS.2024.3372425 | Reliable. Fix "Generalized Convolutional Associative Memories". |
| [9] Kumari & Anand | electronics-13-01229-v2.pdf | RGB MobileNetV2 + attention LSTM; WLASL100; 84.65 % mean 5-fold CV | Add doi:10.3390/electronics13071229 | Reliable; pooled CV, not the official split. |
| [10] Singh et al. (PBRC) | 2512.19451v1.pdf | Hand-joint MediaPipe input; bidirectional ESN reservoirs; WLASL100 60.85 % ± 1.38; training 18.67 s | Author order in PDF: Singh, Syulistyo, **Tamukoh, Tanaka** (yours swaps the last two); add arXiv:2512.19451; accepted to NOLTA, IEICE (pages not printed) | Reliable preprint. |
| [11] Wong et al. (SignRep) | 2503.08529v1.pdf | Self-supervised RGB (Hiera MAE); WLASL2000 61.05 % | PDF is arXiv:2503.08529v1, no venue printed; your "ICCV 2025" is UNVERIFIED — check | Reliable; contrast for RGB SOTA. |
| [12] Hirooka et al. | 2503.16855v3.pdf | Stacked spatial-temporal Transformer on skeletons; WLASL100 mean 81.59 % ± 1.19 (best 82.95 %) | **Title wrong** (PDF: "…for Dynamic Sign Language and Fingerspelling Recognition"); **authors wrong** — PDF lists Hirooka, Murakami, Miah, Hasan, Hwang, Shin (no "Y. Akiba"; Hasan missing in yours; order differs between PDF metadata and page 1 — verify on arXiv) | Reliable preprint. |
| [13] Mittha dataset | — | Dataset card | Add revision hash, MIT licence, access date; note copy at `akasheroor/…` duplicated from `ZahidYasinMittha/…` (*external*) | Provenance caveat (unnamed scraped sources). |
| [14] Google MediaPipe | — | — | Generic link; cite documentation for the **legacy Holistic solution** used (UNVERIFIED URL) plus primary papers: Lugaresi et al. arXiv:1906.08172; Bazarevsky et al. arXiv:2006.10204; Zhang et al. arXiv:2006.10214 (*external*) | Current Holistic Landmarker docs list 478 face points (not 468). |
| [15] PyTorch site | — | — | Prefer Paszke et al., NeurIPS 2019, arXiv:1912.01703 (*external*) | — |
| [16] Next.js docs | — | Framework used (16.2.10) | Add access date | OK |
| [17] FastAPI docs | — | **Not used in the project** | Remove | — |
| [18] webrtc.org | — | WebRTC | Prefer W3C Recommendation "WebRTC 1.0: Real-Time Communication Between Browsers", 26 Jan 2021 (*external*) | — |

### E.2 Claim → citation checks (main problems)
| manuscript claim | cited | adequate? | fix |
|---|---|---|---|
| Barriers for deaf/HoH users (I ¶1) | none | gap | find a verified source (AUTHOR INPUT REQUIRED) |
| MediaPipe + temporal networks effective (I ¶2) | [1],[4],[5] | partly | drop [4]; add [6] |
| LSTM/BiLSTM explored (I ¶3) | [3],[4],[8],[10] | **no** | use [1],[5],[6],[9],[2]; add Hochreiter 1997, Schuster 1997 |
| Real-time video-conferencing (I ¶4) | [7] | **no** (EoC; wrong topic) | [3] |
| WebRTC integration (I ¶4) | [12] | **wrong ref** | [2] (low reliability) |
| SLRNet uses TCN + region segmentation (II.A) | none | **misstated** | CNN + stacked LSTM [6] |
| ISL temporal normalisation (II.A) | none | misstated | spatial normalisation [5] |
| PBRC uses hand + body pose (II.A) | none | **wrong** | hand joints [10] |
| GCAR = "Generalized Convolutional Associative Memories" (II.A) | none | **wrong** | graph convolution with attention and residual connection [8] |
| Sign-Meet uses TCN (II.B, II.D) | none | inconsistent source | state inconsistency or omit [4] |
| Sign-Meet built on Google ideas (II.D) | none | unsupported | remove |
| Real-time processing open challenge (III.A) | [7] | **EoC** | replace |
| 30-frame windows common (V.C) | [5],[6] | yes | — |
| Dataset of 2,208 words (III.C) | [13] | card figure; you trained on 100 | state both |

### E.3 Missing essential references (verified *external*, earlier in this project)
Hochreiter & Schmidhuber, Neural Computation 9(8):1735–1780, 1997, doi:10.1162/neco.1997.9.8.1735 · Schuster & Paliwal,
IEEE TSP 45(11):2673–2681, 1997, doi:10.1109/78.650093 · Graves & Schmidhuber, Neural Networks 18(5–6):602–610, 2005,
doi:10.1016/j.neunet.2005.06.042 · Loshchilov & Hutter, ICLR 2019, arXiv:1711.05101 (AdamW) · Lugaresi et al.
arXiv:1906.08172 (MediaPipe) · Efron, Ann. Stat. 7(1):1–26, 1979 (bootstrap) · McNemar, Psychometrika 12(2):153–157,
1947 · Pedregosa et al., JMLR 12:2825–2830, 2011 (scikit-learn metrics) · Li et al. WLASL, WACV 2020, arXiv:1910.11006 ·
Vaezi Joze & Koller MS-ASL, BMVC 2019, arXiv:1812.01053 · Bragg et al., ASSETS 2019, pp. 16–31 · Koller, arXiv:2008.09918.
Not verified (do not use without checking): SPOTER (WACV-W 2022); Camgoz et al. CVPR 2018; ASL Citizen's NeurIPS venue
(arXiv:2304.05934 verified).

---

## F. Figures and tables audit
| item | issue | action |
|---|---|---|
| Fig. [1] | placeholder; no figure | use `reports/paper/draft/F1_pipeline.png` (verified pipeline) or draw your own matching the implementation |
| Tables | none in the manuscript | Results need tables: dataset (C.1), experiments (C.3), LSTM vs BiLSTM (C.4), validation ladder (C.5), final test + P/R (C.7–C.8), latency (C.9), webcam/post-processing (C.10), test use (§D). Generated sources: `reports/paper/tables_figures_v2/T1–T12.csv`, `TABLES.md` |
| Existing generated figures | F2 validation per seed (validation only); F3 latency; F4 threshold coverage/precision; F5 coverage vs accuracy (dataset vs webcam) | `reports/paper/tables_figures_v2/*.png`; label "validation" / "one signer" in captions |
| Eqs. (2)–(5) | run together in the PDF layout | check typesetting |
| Eqs. (9)–(10) | metrics never computed | remove/redefine |
| Placeholders | [N], [M], [a]–[d], [64/32], [number of trees], [hardware], [mean and 95th percentile], [number], [source of neutral segments], [with some tolerance δ], [all 18 configurations / …], [and window lengths], [Screenshots…] | fill from §C or delete with the unimplemented content |
| Units/terms | "snippets" vs "clips"; "Hands with pose" vs "hands + pose"; "SSTAN" acronym not in source | standardise |

---

## G. Author-input checklist (AUTHOR INPUT REQUIRED)
1. **Author e-mail mapping** — confirm which e-mail belongs to which author and the corresponding author (B.2).
2. **Ethics** — institutional approval, exemption, or a statement that none was required. No record exists in the repository.
3. **Signer S01** — identity relation to the project (e.g. author), informed consent for recording and publication, and
   what was stored (repository shows landmarks, predictions, timings; no video/images — `docs/step6_webcam_evaluation.md`).
4. **Code/data availability** — repository URL or statement; whether landmark arrays/predictions will be shared.
5. **Hardware** — confirm whether "Apple M2" may be stated (only arm64, 8 cores, 8 GB are logged).
6. **Camera** — built-in vs iPhone Continuity camera for the webcam sessions (logs record only "mode: camera").
7. **Target venue** — page/word limits, reference style, figure/table numbering.
8. **Dataset provenance** — supervisor decision on using a dataset whose card says videos were scraped from unnamed sources.
9. **Reference metadata to confirm** — Srivastava [5] volume/pages; SignRep [11] venue; Hirooka [12] author order;
   Anithadevi [1] issue; replacement for [7].
10. **Whether to retain low-reliability sources** [2] and [4].
11. **App screenshots / deployment details** for Section VIII.
12. **Scope wording** — confirm 30-vs-45 appears only in Results (your decision) and the title wording.

---

## H. Prioritized completion plan

**Mandatory (affects multiple sections)**
1. Remove/restate all integration claims (Abstract, III.B, III.C-7, V.A, V.G, VIII, Title) — B.1, B.3, B.13, B.14, B.20, B.26, B.32.
2. Replace the planned 18-configuration design with the experiments actually run (Abstract, I, II.E, III.A, III.C, VI.A, Eqs. 9–10) — C.3.
3. Correct vocabulary size to 100 classes everywhere (III.C, IV, VI.B, VI.C).
4. Correct method details: features 261/221/128; active-interval linear resampling; split order; scheduler monitor; CPU hardware; segmentation; implemented post-processing (IV.B–E, V.C, V.F, VI.B).
5. Write Results (VII) from §C; add test-set disclosure (§D) and protocol (decision rules, bootstrap).
6. Fix citations: wrong numbers in I; add numbers to II; correct SLRNet/PBRC/GCAR/Sign-Meet descriptions; drop [7]; fix [10]/[12] metadata; add LSTM/BiLSTM/AdamW/MediaPipe primary refs; remove [17].
7. Add Conclusion (X), Ethics, Data/Code availability (§G).
8. Fill or delete every placeholder (§F).
9. Rewrite Limitations items 4–6 and add missing limitations (B.33).
10. If you reuse generated tables/figures: update T12, `TABLES.md`, Fig. 1 box and `paper_03_verify_claims.py` for the export checks (§D) — cross-file correction.

**Optional improvements**
- Per-class analysis (largest gains/losses) from `reports/paper/final_test/per_class.csv`.
- Robustness table (C.11) in Results.
- Discussion of calibration (C.10) and overfitting.
- Mention Google SLRTP 2020 paper after verifying it.

**Future work (do not attempt now)**
Signer-independent/multi-signer evaluation with consent; hands + pose + face; multi-seed 45/60 frames; hands-only with
augmentation; RF baseline; E2 in live pipeline; WebRTC integration with captions and in-call latency; target-domain
threshold calibration; evaluating smoothing; continuous recognition.

---

## I. Final consistency checklist
- [ ] **Numbers:** every value matches §C and its source path; 3-decimal macro-F1; 1-decimal percentages; E1 gain +0.028; E2 +0.010.
- [ ] **Counts:** 100 classes; 5,131 / 4,838 / 293; 3,407 / 725 / 706; 261 / 221 / 128 features; 92 webcam attempts.
- [ ] **Terminology:** "isolated word", "temporal sequence length" (not sliding window), "one signer", "validation" vs "test", "one-time final test", "non-inferiority" vs "improvement", "E1", "E2", "baseline".
- [ ] **Scope:** no claims of integration, captions, continuous recognition, translation, signer independence, real-world accuracy, face/60-frame results, RF results.
- [ ] **Statistics:** no seed-level significance; CIs described as clip-level bootstrap; McNemar only for webcam attempts.
- [ ] **Test disclosure:** all uses in §D stated; E1/E2 once; baseline predictions reused.
- [ ] **Methods:** active-interval resampling; split before rejection; scheduler on val macro-F1; CPU; post-processing as implemented.
- [ ] **Citations:** every literature claim numbered; descriptions match the PDFs; [7] removed; metadata fixed; no reference to FastAPI.
- [ ] **Ethics/consent, availability, authors:** filled by you (§G).
- [ ] **Reproducibility:** commands and paths (`reports/paper/draft/manuscript.md` Appendix A lists them).
- [ ] **Formatting:** no remaining bracket placeholders; equations typeset; figures referenced in order; captions state split and signer scope.
- [ ] **Claims check:** after writing, compare each number against this guide; optionally adapt `scripts/paper_03_verify_claims.py` to your text.

---
*Limitations of this guide:* the Anithadevi et al. PDF has no text layer and was read by OCR earlier; Hugging Face,
Crossref and arXiv checks were made earlier in this project and not repeated here; the calling app was inspected as
code, not run.

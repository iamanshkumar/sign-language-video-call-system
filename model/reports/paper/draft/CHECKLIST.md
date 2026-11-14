# Paper checklist (2026-10-27)

Draft: `sections.md` (Results, Discussion, Limitations, Conclusion). Figure F1: `F1_pipeline.{pdf,svg,png}`
(`scripts/paper_02_pipeline_figure.py`). Tables T1–T12 and figures F2–F5:
`reports/paper/tables_figures_v2/` (`scripts/paper_01_tables.py`). Number check: `claims_verification.md`
(`scripts/paper_03_verify_claims.py`, 61 claims, 0 failed). Webcam details: `webcam_details/`
(`scripts/paper_04_webcam_details.py`).

## Research objectives
| # | objective as stated | status | where in the draft | required wording |
|---|---|---|---|---|
| 1 | landmark representations: hands, hands + pose, hands + pose + face | **partially supported** (validation only): hands vs hands + full pose; upper-body vs full pose. Face **not evaluated** (not extracted). | 5.3, T3, Discussion | narrow to "hands-only, hands + full pose and hands + upper-body pose"; state face as not evaluated |
| 2 | temporal sliding-window sizes | **not tested** as sliding windows; only a single-seed LSTM comparison of 30 vs 45 resampled frames | 5.4, T4, Limitation 6 | rename to "temporal sequence length (exploratory)" or remove from the objectives |
| 3 | accuracy vs real-time latency | **supported** on one laptop CPU | 5.2, 5.6, T2, T6, F3 | "on one laptop CPU"; no claim for the video call |
| 4 | isolated-word recognition adapted to continuous online inference | **supported as a single-signer demonstration** of streaming recognition of isolated words separated by pauses | 5.7, T7, T8 | not "continuous sign language"; one signer; not representative accuracy |
| 5 | post-processing for duplicate / unreliable predictions | **partially supported** (one signer): cooldown and confidence threshold evaluated by replay; smoothing not evaluated | 5.8, T9, T10, F4, F5 | one signer; smoothing listed as not evaluated |
| metrics | accuracy, precision, recall, macro-F1, FPS, latency, duplicates, missed signs | accuracy, macro-F1, FPS, latency, duplicates, missed signs reported. **Macro precision / recall** exist per development run (`metrics_*.json`) but are **not in the tables** and were not computed for E1 / E2 test | — | decide (see item 4 below) |

## Unresolved items (authors)
1. **References.** No paper draft or reference list exists in the repository. Citation placeholders in the
   draft must be filled from sources the authors have read: [REF: dataset (Hugging Face
   `akasheroor/American-Sign-Language-Dataset`) and its licence], [REF: MediaPipe Holistic], [REF: LSTM],
   [REF: bidirectional RNN/LSTM], [REF: AdamW], [REF: bootstrap], plus related work. The 12 PDFs in
   `~/Documents/research_papers` could not be read here (no PDF text tool); none was cited.
2. **Sections not drafted:** Abstract, Introduction, Related Work, Methods. Methods must describe the
   preprocessing, the strict scoring rules of the webcam protocol, and the evaluation protocol
   (`docs/evaluation_protocol.md`, including the §3a non-inferiority rule and its change log).
3. **Objective wording:** narrow objectives 1, 2 and 5 as in the table above, consistently in the
   Abstract, Introduction and Conclusion.
4. **Precision / recall:** either drop them from the stated metrics, or compute macro precision and recall
   for the final models from the already saved test predictions
   (`reports/paper/final_test/predictions_test_all.csv`; no new inference). Needs your decision.
5. **E1 gain rounding:** the E1 report and earlier summaries say +0.029; the unrounded value is 0.0285
   (draft: +0.028). Use +0.028 throughout or report 0.0285.
6. **Webcam signer:** confirm that signer S01 is not in the dataset (assumed, not verifiable without signer
   IDs) and add an ethics / consent statement for the recordings.
7. **Table and figure numbering:** map T1–T12 / F1–F5 to the journal's numbering; T2 and T12 should stay
   adjacent to the disclosure text. Figure F2 shows validation only (as labelled).
8. **Dataset licence and preprocessing counts** (videos rejected per rule) belong in Methods
   (`data/sequences/dev100/index_summary.json`).
9. **Optional, not required for a defensible paper:** hands-only with augmentation; a signer-independent or
   multi-signer evaluation; integration of E2 into the live pipeline. Listed as future work only.

## Must not be claimed
- signer independence, or generalization to unseen signers;
- real-world accuracy from the single-signer webcam results;
- face-landmark effects; any sliding-window result;
- video-call integration or its latency;
- E2 as an "improvement" under the predefined validation rule (it met non-inferiority only);
- seed-level statistical significance (minimum two-sided sign-test p = 0.0625 with five seeds);
- that cooldown or calibrated thresholds generalize beyond this signer.

## Terminology (used consistently in the draft)
"temporal sequence length" (not sliding window); "one signer" / "single-signer"; "validation" vs "test";
"one-time final test evaluation"; "non-inferiority rule" / "improvement rule"; "E1" = temporal augmentation;
"E2" = E1 + upper-body pose (221 features); "baseline" = BiLSTM, hands + full pose, 261 features.

## Update 2026-10-27 — full manuscript
- Full manuscript: `manuscript.md` (all sections, 14 tables, 5 figures, 30 references). `sections.md` is kept
  unchanged as the earlier results-only draft.
- Number check: `manuscript_claims_verification.md` — 90 checks (76 numeric/text claims + 14 tables identical
  to `tables_figures_v2/TABLES.md`), 0 failed:
  `PYTHONPATH=src .venv/bin/python scripts/paper_03_verify_claims.py --draft reports/paper/draft/manuscript.md --out reports/paper/draft/manuscript_claims_verification.md --tables reports/paper/tables_figures_v2/TABLES.md`
- Item 4 resolved: macro precision / recall computed from saved predictions (no inference):
  `PYTHONPATH=src .venv/bin/python scripts/paper_05_precision_recall.py` → `reports/paper/final_test_prf/`;
  100 reconciliation checks against `final_test/per_run.csv` and the runs' `metrics_test.json`, max |diff| 0.
- Item 5 resolved: three-decimal convention; E1 gain shown as +0.028 with the unrounded 0.02848 (0.0285) stated once.
- Item 1 resolved except as listed in the manuscript's open items: references verified (Crossref / arXiv / publisher
  pages, or as printed in the local PDFs). Not cited: the IEEE Access CSLR review (IEEE Expression of Concern),
  the IJMRT "Sign-Meet" and IJRTMR video-call papers (venue/evaluation not verifiable), SLRNet (internal
  inconsistencies), SPOTER and Camgoz et al. 2018 (no primary page could be opened).
- Still open (authors): author list; ethics/consent statement for S01; repository URL; dataset provenance
  (card says videos were scraped from unnamed sources, MIT licence); journal table numbering; PDF build
  (no LaTeX/pandoc toolchain in the repository).
- Word version: `manuscript.docx` (A4, Times New Roman 11 pt, table captions above tables, figures embedded,
  page numbers). Generated from `manuscript.md` by `reports/paper/build_tools/md2docx.js`
  (`npm install docx@9.5.1` in any scratch folder, then `node md2docx.js reports/paper/draft/manuscript.md reports/paper/draft/manuscript.docx`).
  Edit `manuscript.md` and regenerate, or edit the .docx directly (then the .md is no longer the source).

# Draft: Results, Discussion, Limitations, Conclusion

All numbers are taken from `reports/paper/tables_figures_v2/TABLES.md` (tables T1–T12, figures F2–F5) and
`reports/paper/final_test/`; figure F1 is `reports/paper/draft/F1_pipeline.{pdf,svg,png}`. Every number in
this draft is checked by `scripts/paper_03_verify_claims.py` (output: `claims_verification.md`).
Validation results, test results and single-signer webcam results are reported separately.
Bracketed items such as [REF: MediaPipe] are citation placeholders to be filled by the authors.

---

## 5 Results

### 5.1 Data and evaluation protocol
The dataset contains 100 ASL word classes and 4,838 accepted video clips, split into 3,407 training,
725 validation and 706 test clips, with a median of 33 training clips per class (range 26–45) (Table T1).
The dataset provides no signer identities, so the split is **not signer-independent**; test results
therefore estimate performance on recordings similar to those in the dataset, not on unseen signers.
All model and design decisions were made on the validation split, with checkpoints selected on validation
macro-F1. Results are reported as the mean ± standard deviation over five training seeds (42, 1, 2, 3, 4),
with 95 % confidence intervals from a class-stratified bootstrap over clips (2,000 resamples). Paired
comparisons use the same seeds and the same bootstrap resamples. The test split was used during
development for 13 training runs and three auxiliary checks (Table T12); none of these results was used
to choose a model. The two final models (E1, E2) were evaluated on test once, after development was
closed.

### 5.2 Recurrent architecture
With hands and full-body pose landmarks (261 features per frame) and 30-frame sequences, the
bidirectional LSTM outperformed the unidirectional LSTM (Table T2). Mean test accuracy was 63.9 ± 1.0 %
(95 % CI 61.1–66.6) for the BiLSTM and 57.5 ± 3.8 % for the LSTM; mean test macro-F1 was 0.634 ± 0.010 and
0.570 ± 0.043. The paired test difference was +6.3 percentage points in accuracy (95 % CI +4.9 to +7.7) and
+0.064 in macro-F1 (+0.049 to +0.078). The BiLSTM was also less sensitive to the training seed
(standard deviation of test accuracy 1.0 versus 3.8 points). These test results were computed during
development and are reported as such.

### 5.3 Landmark representation
Two representation comparisons were run on the validation split, each with five paired seeds and each
within a fixed training setting (Table T3).

*Hands only versus hands and full-body pose (BiLSTM, no augmentation).* Removing all pose landmarks
(128 instead of 261 features) increased mean validation macro-F1 from 0.662 to 0.685, a paired change of
+0.023 (95 % CI +0.007 to +0.040), with all five seeds higher. Per class, 59 classes improved and 41
declined; the largest declines were for signs articulated at a body or face location, such as VOTE,
REGISTER and PLEASE. The hand coordinates remain expressed relative to the shoulders, so the hands-only
features still carry the body reference frame.

*Upper-body versus full-body pose (BiLSTM with temporal augmentation).* Removing the lower-body pose
landmarks 23–32 (hips, knees, ankles and feet; 221 instead of 261 features) changed mean validation
macro-F1 from 0.691 to 0.701, a paired change of +0.010 (95 % CI −0.002 to +0.023). This comparison was
judged by a non-inferiority rule fixed before the experiment (Section 5.5).

A representation that adds face landmarks was not evaluated: face landmarks were not extracted for this
dataset. Hands-only features were not combined with temporal augmentation and were not evaluated on test.

### 5.4 Temporal sequence length
Each sign's active interval is resampled to a fixed number of frames. Only one comparison of this length
exists: a single-seed LSTM run with 45 frames reached a validation macro-F1 of 0.518 against 0.600 with
30 frames (Table T4). Because it rests on a single seed and a single architecture, we report it as
exploratory. The streaming system segments signs by hand activity rather than with a sliding window, so
sliding-window sizes were not studied.

### 5.5 Training improvements: temporal augmentation (E1) and upper-body representation (E2)
Starting from the BiLSTM baseline, two changes were evaluated in sequence, each against a paired
reference and a decision rule fixed in advance (Table T5, Figure F2).

*E1, temporal augmentation.* During training only, each clip was, with probability 0.5, given a hold at
its start or end, a mild tempo change, or one extra cycle of a sub-span whose end returns to its start.
Mean validation macro-F1 rose from 0.662 to 0.691, a paired gain of +0.028 (95 % CI +0.017 to +0.041), with
all five seeds improved. This met the predefined improvement rule (gain ≥ 0.02, at least four of five
seeds better, no class with F1 = 0).

*E2, upper-body pose on top of E1.* As reported in Section 5.3, the paired validation change was +0.010
(95 % CI −0.002 to +0.023); per seed it ranged from −0.018 to +0.027, with four of five seeds higher. E2
met the predefined non-inferiority rule (mean change ≥ −0.01 and lower confidence bound ≥ −0.02) but not
the improvement rule (gain ≥ 0.02).

*Final test evaluation.* After development was closed, E1 and E2 were evaluated once on the test split;
the baseline's saved test predictions were reused without re-running it (Table T5b). Mean test accuracy was
66.1 ± 1.8 % for E1 (95 % CI 63.5–69.0) and 67.8 ± 1.6 % for E2 (65.3–70.6), against 63.9 ± 1.0 % for the
baseline. Mean test macro-F1 was 0.658 ± 0.019 for E1 and 0.675 ± 0.017 for E2, against 0.634 ± 0.010.
Paired by seed, E2 exceeded the baseline by +3.9 percentage points in accuracy (95 % CI +2.5 to +5.3) and
+0.041 in macro-F1 (+0.026 to +0.057); E1 exceeded the baseline by +2.3 points (+1.1 to +3.6), and E2
exceeded E1 by +1.6 points (+0.4 to +2.8). Each step was better in all five seeds. Per class, E2 improved
test F1 over the baseline for 66 classes and reduced it for 34.

*Overfitting.* Training accuracy remained close to 100 % for all three models. The gap between training
and validation accuracy narrowed from 32.5 points (baseline) to 30.1 (E1) and 28.6 (E2), mainly because
validation accuracy increased.

*Robustness diagnostics.* On validation clips with synthetic perturbations (Table T11), temporal
augmentation reduced the accuracy loss caused by a static hold at the start of a sign from −17.1 to −6.8
points and by repeating the sign from −10.9 to −8.0 points. These perturbations resemble the augmentation
operations, so this reduction is partly expected by construction. Sensitivity to losing one hand
(−29.2 points for the baseline) and to mirroring (−17.4) was not reduced. E2 removed the sensitivity to
hip-landmark visibility (−3.0 to 0.0 points) because those inputs are no longer present.

### 5.6 Latency and throughput
All timings were measured on one Apple-silicon laptop CPU (8 GB RAM, four PyTorch threads) (Table T6,
Figure F3). MediaPipe Holistic landmark extraction took a median of 28.5 ms per frame (p95 29.7 ms),
which limits a single process to about 35 frames per second. The BiLSTM classifier took a median of
1.68 ms per sign (batch size 1); in the training runs the LSTM took 0.89 ms and the BiLSTM 1.72 ms.
Closing a segment, including the training preprocessing and the classifier, took a median of 4.1 ms, and
per-frame streaming bookkeeping 0.007 ms. In live capture, reading frames in the main loop limited
processing to 15.0 frames per second; reading them in a background thread raised this to 29.8–29.9
frames per second with the camera's 30 frames per second. A recognised word was emitted a median of
0.44–0.46 s after the signer lowered their hands, dominated by the 0.4 s rest needed to close a segment.

### 5.7 Online recognition (single signer)
The streaming recognizer was evaluated with one signer who is not in the dataset. These results describe
this signer and recording setup only and are not an estimate of performance on unseen signers.
Offline and live recognition were consistent: on 20 dataset clips, streaming features and predictions
matched the offline pipeline exactly, and replaying each recorded session's saved landmarks reproduced
every recorded event. On a frozen set of 92 webcam attempts (Table T7), mean top-1 accuracy over five
seeds was 29.8 % for the LSTM, 42.4 % for the BiLSTM baseline, 43.9 % for E1 and 43.7 % for E2; mean top-5
accuracy was 57.2 %, 63.0 %, 66.5 % and 69.6 %. Paired comparisons on these attempts showed no
statistically significant difference between E1 or E2 and the baseline in any seed. For this signer, AFRICA and
COMPUTER were never recognised by any of the 20 models (four model types, five seeds each), and FORBID
was recognised once (by one E2 seed) in 240 model-attempts. In two live sessions with the
baseline model (Table T8), 13 and 9 of 20 attempts were correct, no sign was missed, and 9 and 2 attempts
produced a duplicate prediction.

### 5.8 Post-processing
Duplicate predictions arose mainly when a sign lasting longer than the 4 s segment cap was split and
its continuation classified again: of the 14 duplicate emissions in the replay below, 13 were such
continuations and one was a second burst of movement starting 0.57 s after the hands were last seen in the first segment. Post-processing settings were evaluated by replaying the 92 single-signer
webcam attempts with the baseline model (Table T9, Figure F4). Without post-processing, 37 attempts were
correct and 14 attempts produced one duplicate emission each. A cooldown of 0.5 s or 1.0 s, which
suppresses a segment starting shortly after the previous one, removed all 14 duplicate emissions while
leaving every attempt's answer unchanged. A confidence threshold traded coverage for precision: at 0.4,
76.1 % of attempts were answered with 50.0 % precision; at 0.57, 54.3 % with 60.0 %; at 0.7, 43.5 % with
70.0 %. The threshold of 0.57 had been chosen on validation data for 80 % precision; on the dataset's
test split it gave 77.7 % precision at 74.9 % coverage, whereas on the webcam attempts it gave 60.0 %
precision (Table T10, Figure F5). No sign was missed under any setting. Temporal smoothing of
predictions was implemented but not evaluated.

---

## 6 Discussion

**Validation and test findings for E2.** E2 was accepted under a non-inferiority rule that was written
down before it was trained: on validation its mean gain over E1 was +0.010, below the 0.02 required for an
improvement, and only its non-inferiority was established. On the one-time test evaluation E2 exceeded E1
by +1.6 points of accuracy and +0.016 macro-F1 in all five seeds. We report both and do not reclassify E2
as an improvement on the basis of the test result: the selection rule was fixed before the result was
known, the test split is not signer-independent, and the validation and test estimates are compatible with
a small effect of uncertain size. The appropriate reading is that removing the lower-body landmarks did not
harm accuracy and may help slightly, while removing inputs that carry no linguistic information for signs
produced above the waist.

**Representation.** On validation, hands-only features outperformed hands with full-body pose, and the
upper-body pose subset was not worse than full-body pose. Together these suggest that, in this dataset,
full-body pose adds more noise than useful information to a model with about 33 training clips per
class. Signs articulated at a body or face location lost accuracy without pose, which indicates that some
body-relative information is useful; whether hands-only or upper-body pose is preferable with temporal
augmentation was not tested.

**Temporal augmentation.** E1 gave the largest validated gain and reduced sensitivity to holds and
repetitions, which occur in live signing but are rare in the dataset clips. Its effect on the
single-signer webcam set was small and not significant, so the offline gain cannot be assumed to transfer
to live use.

**Gap between dataset and live performance.** Accuracy on the single-signer webcam attempts (42.4–43.9 % mean
top-1) was 21–24 points below the corresponding test accuracy on the dataset (63.9–67.8 %). Signs that were never recognised for this signer
point to differences between how this signer and the dataset's signers produce those signs. One signer
cannot show how large this gap is in general.

**Real-time feasibility.** Landmark extraction dominated the per-frame cost (28.5 ms), while the classifier
added under 2 ms per sign, so the BiLSTM's accuracy gain over the LSTM costs less than 1 ms per sign. With
threaded capture the pipeline kept up with a 30 frames-per-second camera on a laptop CPU. The main
delay perceived by a user is the 0.4 s rest used to detect the end of a sign.

**Post-processing.** For this signer a short cooldown removed duplicate predictions without changing any
answer, because the duplicates were almost all continuations of the same sign. Confidence thresholds calibrated on
dataset clips were over-optimistic on webcam input, so thresholds need calibration on data from the target
setting.

---

## 7 Limitations

1. **Not signer-independent.** The dataset has no signer identities; the split cannot exclude the same
   signers or recording sessions from training and test. Test accuracy does not measure generalization to
   unseen signers.
2. **Test-set reuse.** The test split was evaluated for 13 development runs and three auxiliary checks
   before the final evaluation (Table T12). No model choice used test results, but the baseline's test
   figures were known during development.
3. **One webcam signer.** All live and replay results come from one signer, one camera and one room; the
   webcam set was also inspected during development. These results are descriptive and not representative
   of real-world accuracy.
4. **Seed-level statistics.** With five seeds, a two-sided sign test cannot fall below p = 0.0625; the
   confidence intervals reflect variation across clips, not across independent datasets or signers.
5. **Representation study is incomplete.** Face landmarks were not extracted; hands-only features were
   evaluated only on validation and without augmentation; no representation was tested on unseen signers.
6. **No sliding-window study.** Only one single-seed comparison of sequence length exists.
7. **Isolated words only.** The online system recognises single words separated by pauses; it does not
   recognise continuous signing or sentences.
8. **Not integrated.** The FastAPI inference service and delivery of recognised words into the video
   call were not built. The live pipeline accepts only 261-feature models, so E1 and E2 were evaluated
   offline; the live sessions used the baseline model.
9. **One hardware platform.** Latency was measured on one laptop CPU; other devices and browsers were
   not measured.
10. **Overfitting remains.** Training accuracy is close to 100 % and the training–validation gap is about
    29 points for the final model.

---

## 8 Conclusion

We built and evaluated a landmark-based recognizer for 100 isolated ASL words and a streaming pipeline
that runs in real time on a laptop CPU. A bidirectional LSTM outperformed a unidirectional LSTM by 6.3
points of test accuracy. Temporal augmentation improved validation macro-F1 by 0.028, and removing
lower-body pose landmarks was non-inferior on validation; on a single final test evaluation, the resulting
model reached 67.8 % accuracy and 0.675 macro-F1, 3.9 points above the baseline. Hands-only features
outperformed hands with full-body pose on validation. Landmark extraction, not classification, dominates
the processing cost, and a short cooldown removed duplicate predictions in single-signer replays. Because
the dataset split is not signer-independent and live evaluation involved one signer, these results do not
establish performance for unseen signers. Future work should evaluate signer-independent splits and
multiple new signers, test hands-only and upper-body representations with augmentation, integrate the
recognizer into the video call, and calibrate post-processing on target-domain data.

---

## Figure and table captions

- **Figure F1.** System pipeline. A: offline data preparation, landmark extraction, preprocessing,
  training and evaluation. B: online streaming inference evaluated with one signer. Solid boxes are
  implemented and used for the reported results; dashed boxes are implemented but evaluated offline only
  or disabled by default; dotted grey boxes are proposed or not integrated.
- **Figure F2.** Validation macro-F1 per seed for the baseline, E1 and E2 (grey lines connect the same
  seed; black: mean of five seeds). Validation split only.
- **Figure F3.** Median processing time per component on one laptop CPU.
- **Figure F4.** Coverage and precision of answers for confidence thresholds 0, 0.4, 0.57 and 0.7 on 92
  webcam attempts from one signer (baseline model).
- **Figure F5.** Accuracy of accepted predictions against coverage for the validation and test splits
  (baseline candidate, seed 3) and for the single-signer webcam attempts.
- **Table T1.** Dataset and split. **T2.** LSTM versus BiLSTM (validation; test computed during
  development). **T3.** Landmark representation (validation). **T4.** Sequence length (exploratory,
  single seed). **T5.** Development ladder (validation, with the one-time test result). **T5b.** One-time
  final test evaluation. **T6.** Latency and throughput. **T7.** Webcam replay benchmark (one signer).
  **T8.** Live sessions (one signer). **T9.** Post-processing replay (one signer). **T10.** Confidence
  calibration. **T11.** Robustness diagnostics (validation). **T12.** Test-set use disclosure.

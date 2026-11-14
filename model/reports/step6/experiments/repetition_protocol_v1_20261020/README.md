# Repetition protocol v1 (blocks 5a–5d) — validity, manipulation check, descriptive results (2026-10-20)

Analysis of the five recordings of the approved single-vs-repeated protocol. Read-only on the recordings;
analysis script scratch-only (not in the repository). Files: `blocks.csv` (one row per recording),
`per_attempt.csv` (80 attempts: outcome, exclusion, movement measures). The movement measures use the
hold/repetition method and its training-derived thresholds unchanged
(`../hold_repetition_20261020/summary.json`, p25 thresholds).

## Recordings (all follow the predeclared sign order; replay reproduces every live event)
| block | condition | folder | written (UTC) | frame size |
|---|---|---|---|---|
| 5a | SINGLE | `20261020T063535Z_camera_S01` | 06:35:35 | 640×360 |
| 5b | REPEATED | `20261020T063812Z_camera_S01` | 06:38:12 | 640×360 |
| 5c-1 | REPEATED | `20261020T064102Z_camera_S01` | 06:41:02 | 640×360 |
| 5c-2 | REPEATED | `20261020T064353Z_camera_S01` | 06:43:53 | 640×480 |
| 5d | SINGLE | `20261020T064645Z_camera_S01` | 06:46:45 | 640×480 |

The folder timestamp is written when a block ends; blocks ran back to back (~2.5–2.8 min apart). 640×480
was requested every time; the camera delivered 640×360 for the first three recordings and 640×480 after.

## Validity (predeclared: ≥ 12 valid attempts, ≥ 25 fps, ≤ 2 % dropped frames)
| block | camera fps | processed fps | dropped | valid attempts | result |
|---|---|---|---|---|---|
| 5a | 29.7 | 30.0 | 48/4038 (1.19 %) | 13/16 | pass |
| 5b | 30.0 | 30.0 | 36/4051 (0.89 %) | 16/16 | pass |
| 5c-1 | 29.9 | 30.0 | 28/4054 (0.69 %) | 15/16 | pass |
| 5c-2 | 30.0 | 29.8 | 19/4073 (0.47 %) | 15/16 | pass |
| 5d | **24.0** | **23.9** | 17/3245 (0.52 %) | 16/16 | **fail (fps)** |

Exclusions applied (objective, prediction-independent): signing during WAIT (an early event) — 5a #2, #9,
#11; 5c-1 #11; 5c-2 #3. No attempt had GO given with hands up; no pre-GO answers. The paper mistake log
was not available, so self-reported mistakes could not be applied. 5c-2 has 2 events outside attempts,
both during the framing check before the first prompt (not scored).

## Manipulation check (predeclared: REPEATED longer movement time than SINGLE in ≥ 6 of 8 signs)
| comparison | signs with R longer | overall moving-time ratio R/S | result |
|---|---|---|---|
| 5a vs 5b + 5c-2 (valid blocks, provisional primary) | 5/8 | 1.10 | **fail** |
| 5a vs 5b + 5c-1 | 4/8 | 1.00 | fail |
| 5a + 5d vs 5b + 5c-2 (includes invalid 5d; not admissible) | 7/8 | 1.17 | — |

Full hand-run durations: median 2.85 s (5a) vs 3.13 s (5b), 3.07 s (5c-1), 3.21 s (5c-2), 2.99 s (5d).
Three cycles versus one should roughly double the movement time; a 0–17 % difference does not show that
the conditions were performed differently. Per-attempt cycle counts cannot be established: the burst count
misses slow cyclic movement and no video exists. Conclusion: **manipulation not verified**; by the
predeclared rule the SINGLE vs REPEATED comparison is inconclusive.

## Descriptive results (not interpretable as a repetition effect; see above)
| block | correct (all) | correct (valid) | top-5 (valid) | duplicates | forced cuts | median delay |
|---|---|---|---|---|---|---|
| 5a SINGLE | 8/16 | 7/13 | 8/13 | 1 | 2 | 0.41 s |
| 5b REPEATED | 4/16 | 4/16 | 9/16 | 2 | 3 | 0.40 s |
| 5c-1 REPEATED | 5/16 | 5/15 | 9/15 | 3 | 5 | 0.40 s |
| 5c-2 REPEATED | 4/16 | 4/15 | 7/15 | 2 | 4 | 0.40 s |
| 5d SINGLE (invalid) | 5/16 | 5/16 | 9/16 | 1 | 2 | 0.42 s |

No missed, rejected or pre-GO answers in any block. MORE and PLEASE were correct in nearly every block;
AFRICA, COMPUTER and FORBID were wrong in every block. The SINGLE–REPEATED difference in the valid blocks
comes from SHOWER (5a 2/2, all other blocks 0/2) and TUESDAY (5a 1/1, 5c-1 1/2, others 0). The two
SINGLE blocks differ as much from each other on these signs (5a vs 5d: SHOWER 2/2 vs 0/2) as SINGLE
differs from REPEATED, and the two 5c recordings (same condition and order) differ on 3 signs.
The predeclared within-sign permutation statistic is computed in the analysis (p ≈ 0.03 for the valid
blocks) but is not interpretable: the manipulation check failed, SINGLE rests on one block (2 attempts
per sign or fewer), and block order, camera format and framing all changed between blocks.

## Framing (shoulder width / frame width, median; training median 0.371, interquartile 0.257–0.421)
5a 0.210, 5b 0.195, 5c-1 0.198, 5c-2 0.175, 5d 0.204 — all below the training range and below sessions
3 (0.255) and 4 (0.277); every block ended with "move closer".

## 5c recordings
Both are complete and pass validity. No transcript or log in the repository explains the repeat, so the
later recording (5c-2) is provisionally primary as instructed; 5c-1 is reported alongside in every table.
The repeat was not part of the protocol (which allowed re-recording only an invalid block).

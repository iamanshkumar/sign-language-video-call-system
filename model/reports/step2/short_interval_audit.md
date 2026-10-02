# Step 2 - class-level short-active-interval audit (benchmark)

Short = active signing interval < 30 frames (flag `short_active_interval`, not a rejection). Population: the 374 benchmark clips accepted under methodology v2 (re-evaluated from stored Phase A detections; no MediaPipe re-run).

> **Limitation:** the benchmark has ~1 accepted clip per class (see below) and is not proportional to the dataset, so per-class rates are indicative only; category-level rates are more reliable. A dataset-wide class audit needs the full extraction.

## Overall

- Short clips: **111 / 374 (29.7%)**; active frames among short clips: {'count': 111.0, 'mean': 21.5, 'std': 6.41, 'min': 3.0, '25%': 18.0, '50%': 23.0, '75%': 26.5, 'max': 29.0} (median 0.77 s); 7 have < 10 active frames.
- Clip length vs active frames: Spearman ρ = 0.72 (p = 4.9e-61).
- Active share of clip (%): short {'count': 111.0, 'mean': 38.7, 'std': 12.7, 'min': 4.5, '25%': 30.7, '50%': 38.7, '75%': 47.7, 'max': 67.5}; not short {'count': 263.0, 'mean': 52.6, 'std': 16.4, 'min': 20.7, '25%': 42.4, '50%': 49.2, '75%': 57.5, 'max': 100.0}.

## Classes

- Classes with accepted clips: 357; with ≥1 short clip: 110; **all accepted clips short: 107** (these classes would have no benchmark clip left if short intervals were rejected).
- Classes with ≥2 accepted clips: 15; of these, with any short: 4, all short: 1.

Classes with ≥2 accepted clips and at least one short clip:

| word | accepted | short | min_active_frames | median_active_frames | short_pct |
|---|---|---|---|---|---|
| mirror | 2 | 2 | 22.0 | 24.0 | 100.0 |
| airplane | 2 | 1 | 26.0 | 31.5 | 50.0 |
| enemy | 2 | 1 | 24.0 | 29.0 | 50.0 |
| pour | 2 | 1 | 18.0 | 32.0 | 50.0 |

Classes whose every accepted benchmark clip is short (107): `mirror`, `absolutelynothing`, `allway`, `basic`, `bat`, `bitch`, `blowmind`, `blue`, `board`, `boss`, `boyfriend`, `busted`, `c`, `chin`, `chopsticks`, `comeout`, `committee`, `consume`, `cost`, `death`, `dilemma`, `disney`, `dribble`, `dry`, `easytodo`, `flipflops`, `flipswitch`, `floor`, `freedom`, `gallaudet`, `head`, `headphones`, `hen`, `hundred`, `ignore`, `lie`, `lightning`, `m`, `microphone`, `mock`, `more`, `mosquito`, `mouse`, `near`, `never`, `none`, `nonsense`, `normal`, `nosebleed`, `now`, `offthepoint`, `owe`, `paragraph`, `phone`, `pile`, `pirate`, `popcorn`, `precious`, `prefer`, `q`, `razor`, `reduce`, `replace`, `restaurant`, `roof`, `room`, `secretary`, `select`, `set`, `shortperson`, `signature`, `slingshot`, `small`, `sneeze`, `snob`, `soap`, `something`, `spate`, `spitout`, `spoon`, `staff`, `sting`, `stir`, `story`, `strawberry`, `strong`, `sushi`, `suspend`, `talk`, `taste`, `three`, `toothbrush`, `topic`, `truck`, `two`, `twomore`, `verb`, `vest`, `videocamera`, `wash`, `wheelchair`, `who`, `will`, `window`, `wolf`, `workshop`, `zipper`

Full per-class table: `reports/step2/tables/short_interval_by_class.csv`.

## Clip categories

### label_type

| label_type | accepted | short | short_pct |
|---|---|---|---|
| single_word | 353 | 108 | 30.6 |
| fingerspelled_letter | 14 | 3 | 21.4 |
| multi_word | 7 | 0 | 0.0 |

χ² test (short vs not short): p = 0.169 approximate: expected counts < 5

### source_proxy

| source_proxy | accepted | short | short_pct |
|---|---|---|---|
| numericid-WORD | 265 | 89 | 33.6 |
| word_video_n | 47 | 4 | 8.5 |
| word_timestamp_n | 34 | 11 | 32.4 |
| word_n | 26 | 7 | 26.9 |
| word_only | 2 | 0 | 0.0 |

χ² test (short vs not short): p = 0.0109 approximate: expected counts < 5

### clip_frames_30fps

| clip_frames_30fps | accepted | short | short_pct |
|---|---|---|---|
| 60-89 | 169 | 40 | 23.7 |
| >=90 | 117 | 2 | 1.7 |
| 45-59 | 59 | 45 | 76.3 |
| <45 | 29 | 24 | 82.8 |

χ² test (short vs not short): p = 1.01e-31 

### fps_band

| fps_band | accepted | short | short_pct |
|---|---|---|---|
| ~30 | 322 | 96 | 29.8 |
| <29.5 | 32 | 8 | 25.0 |
| >30.5 | 20 | 7 | 35.0 |

χ² test (short vs not short): p = 0.737 

### augmented

| augmented | accepted | short | short_pct |
|---|---|---|---|
| False | 362 | 108 | 29.8 |
| True | 12 | 3 | 25.0 |
### in_duplicate_group

| in_duplicate_group | accepted | short | short_pct |
|---|---|---|---|
| False | 342 | 107 | 31.3 |
| True | 32 | 4 | 12.5 |
## Shortest active intervals (accepted)

| word | sample_id | active_frames | std_frame_count | rest_frames_before | rest_frames_after | active_both_hands_missing_pct |
|---|---|---|---|---|---|---|
| wheelchair | part_4/6471960992477459-WHEELCHAIR.mp4 | 3.0 | 67 | 14.0 | 50.0 | 0.0 |
| chin | part_2/1921109044689493-CHIN 2.mp4 | 4.0 | 67 | 17.0 | 46.0 | 0.0 |
| flipswitch | part_5/8545363405007698-FLIP SWITCH.mp4 | 5.0 | 49 | 15.0 | 29.0 | 0.0 |
| never | part_1/17168186292515686-NEVER.mp4 | 6.0 | 52 | 9.0 | 37.0 | 0.0 |
| head | part_4/708769017680714-HEAD.mp4 | 7.0 | 47 | 9.0 | 31.0 | 0.0 |
| precious | part_4/559768801316793-PRECIOUS.mp4 | 8.0 | 33 | 4.0 | 21.0 | 0.0 |
| death | part_3/4951174780421559-DEATH.mp4 | 9.0 | 79 | 26.0 | 44.0 | 0.0 |
| taste | part_2/3583049668962337-TASTE.mp4 | 10.0 | 38 | 13.0 | 15.0 | 0.0 |
| two | part_5/7391350877622198-TWO.mp4 | 10.0 | 38 | 9.0 | 19.0 | 0.0 |
| dry | part_6/9862882272897495-DRY.mp4 | 11.0 | 45 | 7.0 | 27.0 | 0.0 |
| toothbrush | part_1/08384216500026387-TOOTHBRUSH.mp4 | 12.0 | 40 | 13.0 | 15.0 | 0.0 |
| popcorn | part_5/888614784762559-POPCORN.mp4 | 13.0 | 47 | 14.0 | 20.0 | 0.0 |
| wolf | part_1/08673843529675529-WOLF.mp4 | 14.0 | 45 | 8.0 | 23.0 | 0.0 |
| signature | part_1/09602770669006189-SIGNATURE.mp4 | 14.0 | 33 | 8.0 | 11.0 | 0.0 |
| consume | part_3/5377518041774723-CONSUME.mp4 | 14.0 | 77 | 14.0 | 49.0 | 0.0 |

Very short intervals are worth a visual spot-check: they may be quick signs, or clips where the hands are mostly outside the frame/undetected even during the sign.


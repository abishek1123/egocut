# Research scripts (not needed to run the system)

These are the scripts behind the numbers in the main README. They were run from our development tree,
so paths inside may need adjusting.

| Script | What it did |
|---|---|
| `make_dev_split.py` | Held out EPIC-KITCHENS-100 training participants P04, P22, P30 as a dev set (detector heads retrained without them) |
| `run_af_val.py` | Ran an ActionFormer verb/noun head over a feature set and saved its detections |
| `tune_dev.py`, `tune_rules.py` | Tuned verb-noun composition and the coverage selector on the dev split, under the kit rules (>= 4 s segments, 2000-action cap) |
| `eval_joint_dev.py` | Dev-split ablation of joint per-anchor composition and recognition re-scoring |
| `compose_eval.py` | Our implementations of action mAP (tIoU 0.1-0.5) and action coverage (>= 50 % of the action inside the highlights) |
| `pick_bgr.py`, `compare_feats.py` | Checked our MP4 feature extraction against the official EPIC features (found the BGR channel-order fix; cosine 0.74 -> 0.967) |
| `imp_search.py`, `cv.py` | Chose the interest-model weights on SumMe, with 5-fold cross-validation |
| `pod_final.sh` | Release check on a fresh GPU machine: install, fetch weights, run `run.sh` with the network off, compare with `outputs/`, pin versions |
| `pod_test_pkg.sh` | End-to-end package test: runs `run.sh` exactly as the organisers do and checks validity and timing |

All tuning and every number we report come from the dev split or SumMe. EPIC-KITCHENS-100 validation and FPVSum
annotations (the test set) were never used for training, tuning or model selection.

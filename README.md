# EgoCut — Action-Aware Egocentric Video Summarization (Track 4)

EgoCut turns a first-person video into (1) a list of detected **verb–noun actions** and (2) a **~15 % highlight** chosen to cover as many *distinct* actions as possible, plus a video-level **category label** — in the exact 6-field JSON the track requires.

```
bash run.sh <video.mp4> <out.json>
```

## 0. Submission materials

| Item | Where |
|---|---|
| Presentation | [`docs/VIT_DAX.pptx`](docs/VIT_DAX.pptx) |
| Demo video (under 3 min) | [`docs/EgoCut_DAX_demo.mp4`](docs/EgoCut_DAX_demo.mp4) |
| Example outputs | [`outputs/`](outputs) (187 JSON files, one per manifest video) |
| Highlight reels and answer clips | [`docs/demo/`](docs/demo) |
| Model weights | GitHub release [`weights-v1`](https://github.com/abishek1123/egocut/releases/tag/weights-v1) |

## 1. Setup

**Option A — Docker (recommended, GPU)**
```bash
docker build -t egocut .            # downloads ~3.5 GB of weights during the build
docker run --gpus all -v /videos:/data -v /preds:/out egocut /data/P01_11.MP4 /out/P01_11.json
```

**Option B — local Python (tested: 3.12, CUDA 12.9)**
```bash
sudo apt-get install -y ffmpeg
pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu129
pip install -r requirements.txt
bash weights/download_weights.sh    # one-time, ~3.5 GB from our GitHub release, checksum-verified
bash run.sh /videos/P01_11.MP4 outputs/P01_11.json
```
Tested on NVIDIA RTX A6000, RTX 6000 Ada and RTX 4090; on the 4090 a short clip takes about 30 s end to end, most of it loading the models (peak RAM about 3.8 GB). It also runs on CPU, but the two ViT-L backbones make that very slow.

**Offline / isolated runs.** After the one-time weight download, `run.sh` needs no network: every model loads from `weights/` and Hugging Face is put in offline mode. We tested this on a fresh GPU machine with the network switched off: the outputs matched our submitted files (top-200 actions identical on all three test videos).

**Option C — Windows:** same Python setup, then `run.bat C:\videos\P01_11.MP4 outputs\P01_11.json`.

**Validate:** `python code/validate_local.py --dir outputs/` (our rule-by-rule implementation of the kit's checks; use the official `validate.py` if you have it).

## 2. How it works

```
MP4 ──ffmpeg (stream, 30 fps)──┬─► SigLIP frame embeddings (1 fps) ──► label (zero-shot over 15 class prompts)
                               │
                               │   label == cooking:
                               ├─► VideoMAE-L, EPIC-finetuned VERB backbone ─┐  16-frame snippets, stride 8
                               ├─► VideoMAE-L, EPIC-finetuned NOUN backbone ─┤
                               │        ActionFormer VERB + NOUN heads (shared anchor grid)
                               │        ─► joint per-anchor verb×noun composition (top-10×10, class-wise soft-NMS)
                               │        ─► recognition re-scoring with the backbones' own classifiers
                               │        ─► actions (≤ 2000)  ─► EgoCut coverage selector ─► highlights
                               │
                               │   otherwise:
                               └─► SlowFast-R50 motion + SigLIP novelty + centre prior ─► interest model ─► highlights

highlights: sorted, non-overlapping, every piece ≥ 4.1 s, total ≤ 14.9 % of the video
```

* **Strong features.** VideoMAE-L backbones fine-tuned on EPIC-KITCHENS-100 (verb and noun, OpenTAD release), run directly on the MP4.
* **Joint verb–noun detection.** Most systems detect verbs and nouns separately and pair them afterwards. EgoCut composes them **per anchor**: for every temporal position it takes the 10 best verbs × 10 best nouns, scores each pair `sqrt(p_verb · p_noun)`, fuses boundaries weighted by confidence, and suppresses duplicates per verb–noun class. This alone lifts held-out action mAP from 24.1 to 25.1.
* **Recognition re-scoring.** Each detected segment is re-scored with the backbones' own EPIC action classifiers averaged over the segment: `score × (p_verb · p_noun)^0.6`. Held-out mAP goes from 25.1 to 29.0.
* **Coverage-optimal highlight selection (our main contribution).** Action coverage is the largest metric (40 %) and counts *distinct* verb–noun pairs. Instead of ranking frames by "importance", the EgoCut selector greedily adds the window with the highest **expected number of new distinct actions per second of budget**, then fills any leftover budget. Every piece is ≥ 4.1 s, so no time is wasted by the ≥ 4 s rule.
* **Interest model for non-kitchen video.** For sports/FPV footage, highlights come from an unsupervised score: centre prior + SigLIP visual novelty + SlowFast motion magnitude/change. Its weights were chosen on **SumMe**, an unrelated public summarization dataset, never on the test videos.
* **Fallbacks.** If the VideoMAE weights are missing, kitchen video goes through a SlowFast-R50 + ActionFormer branch. If anything fails, `code/fallback.py` writes a valid minimal file.

## 3. Results (held-out development data)

The test videos' annotations are treated as hidden test labels, so every number here comes from data we held out ourselves:
**EPIC dev** = all videos of EPIC-KITCHENS-100 *training* participants P04, P22 and P30, removed from detector training; **SumMe** = a public summarization dataset unrelated to the test set.

| Metric (weight) | Data | EgoCut |
|---|---|---|
| Action coverage (40 %): ≥ 50 % of the action inside the highlights | EPIC dev | **45.6 %** (54.0 % with any overlap) |
| Action mAP (25 %), tIoU 0.1–0.5 | EPIC dev | **29.0** |
| Highlight F1 (25 %), top-15 % consensus frames | SumMe (the set its weights were chosen on) | **31.0** |
| Label accuracy (10 %) | — | zero-shot SigLIP, no training or tuning |

**Ablation on EPIC dev**

| Step | Action mAP | Coverage any / ≥ 50 % |
|---|---|---|
| VideoMAE-L + separate verb/noun pairing | 24.1 | 50.8 / 43.1 |
| + joint per-anchor composition | 25.1 | — |
| + recognition re-scoring | 29.0 | 53.2 / 45.0 |
| + budget 14.5 % → 14.9 % | 29.0 | 54.0 / 45.6 |

**No test labels were used for training, tuning or model selection.** Detector heads were trained on EPIC-KITCHENS training videos without P04/P22/P30, every hyper-parameter was chosen on that dev split, and the interest model was chosen on SumMe. Class labels are never read from file names.

**Known limits.** Two FPVSum manifest videos (Scuba_KD74Ywuy9ng, Scuba_Um4kqB8bQBg) are no longer downloadable; the organisers excluded them from scoring. Their files in `outputs/` were written by the generic `code/fallback.py` path (label `other`, no actions, one middle block) so the folder still holds one valid file per manifest row. Three kitchen videos are shorter than 27.5 s, where a single ≥ 4 s highlight is already more than 15 % of the video; those files go slightly over budget by necessity. Non-kitchen video gets no action list because the action vocabulary is kitchen-only.

## 4. Data and models used

| Component | Source | Licence |
|---|---|---|
| VideoMAE-L EPIC-KITCHENS-100 verb/noun backbones | sming256/OpenTAD (VideoMAE, MCG-NJU) | Apache-2.0 / CC BY-NC 4.0 |
| ActionFormer heads on VideoMAE-L features | trained by us (ActionFormer code, happyharrycn/actionformer_release) | MIT |
| SlowFast-R50, EPIC-KITCHENS-100 | epic-kitchens/C1-Action-Recognition (SlowFast.pyth) | Apache-2.0 |
| SigLIP base patch16-224 | google/siglip-base-patch16-224 (downloaded from Hugging Face on first run) | Apache-2.0 |
| EPIC-KITCHENS-100 train annotations | epic-kitchens/epic-kitchens-100-annotations | CC BY-NC 4.0 |
| SumMe (dev for the interest model) | gyglim / ETH | research use |

All our trained weights are in the GitHub release `weights-v1`, with SHA-256 checksums (`weights/download_weights.sh`).

## 5. Robustness

`run.sh` never leaves a video without a valid file: if the main pipeline fails for any reason (corrupt video, out of memory, missing library), `code/fallback.py` (standard library only) writes a minimal valid prediction. The system `ffmpeg` is preferred over the bundled one because very new static builds can hit thread limits inside containers.

## 6. Brownie features (beyond the track output)

These run on the JSON files alone: Python + NumPy, no GPU, no model. They never touch `run.sh` or `outputs/`.

| Feature | Command | What you get |
|---|---|---|
| Highlight reel | `python code/make_reel.py video.mp4 outputs/<id>.json reel.mp4` | The highlights cut from the original footage, each captioned with the actions detected inside it |
| Timeline viewer | `python code/make_viewer.py video.mp4 outputs/<id>.json viewer.html` | A self-contained page with the video, detected actions and highlights on one timeline |
| Ask your memory | `python code/brownie/ask.py outputs/<id>.json "when did I wash the plate?" --video video.mp4 --clips clips/` | Plain-English questions matched to the verb and noun classes (EPIC-KITCHENS synonym lists), answered with time stamps and clips. Also `--at 2:35` and `--between 1:00 2:00` |
| Any-length summary | `python code/brownie/summarize.py outputs/<id>.json --seconds 30 --video video.mp4 --reel recap.mp4` | The coverage selector re-plans the highlight for any budget. On P01_11 (9 min): 30 s keeps 25 distinct detected actions, 60 s keeps 37, the submitted 14.9 % keeps 46 |

## 7. Layout

```
run.sh / run.bat        entry point (contract: <video.mp4> <out.json>)
code/egocut_infer.py    full pipeline (also batch mode: --list videos.txt --outdir outputs/)
code/vmae.py            VideoMAE-L backbone + snippet feature extraction
code/joint.py           joint per-anchor verb-noun composition + recognition re-scoring
code/compose.py         verb-noun composition (SlowFast fallback branch)
code/selector_fast.py   EgoCut coverage selector
code/write_official.py  official schema writer + highlight fixing (sorted, non-overlapping, ≥ 4.1 s)
code/fallback.py        crash-safe minimal output
code/brownie/           ask your memory + any-length summary (extras, not used by run.sh)
code/validate_local.py  schema/rule checker
code/third_party/       vendored ActionFormer (MIT) and EPIC SlowFast (Apache-2.0), NumPy NMS
weights/                download_weights.sh, egocut_config.json (all tuned hyper-parameters)
outputs/                our 187 predictions (one per manifest row)
```

#!/usr/bin/env bash
# EgoCut - Track 4 run contract:   bash run.sh <video.mp4> <out.json>
# Writes exactly one JSON in the official 6-field schema. If anything in the main pipeline fails,
# a standard-library fallback still writes a valid file, so a single bad video never breaks a batch.
set -u
if [ $# -ne 2 ]; then echo "usage: bash run.sh <video.mp4> <out.json>" >&2; exit 2; fi
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VIDEO="$1"; OUT="$2"; PY="${PYTHON:-python3}"
mkdir -p "$(dirname "$OUT")"
if [ ! -f "$HERE/weights/SlowFast.pyth" ] || [ ! -f "$HERE/weights/vmae_backbone_verb.pth" ] || [ ! -f "$HERE/weights/actionformer_vmae_verb.pth.tar" ] || [ ! -s "$HERE/weights/siglip/model.safetensors" ]; then
  echo "[run.sh] weights missing - fetching them once (see README)" >&2
  bash "$HERE/weights/download_weights.sh" || echo "[run.sh] weight download failed" >&2
fi
# everything is local once the weights are in place: never touch the network at inference time (isolated re-runs)
if [ -s "$HERE/weights/siglip/model.safetensors" ]; then export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1; fi
FF_ARGS=()
if command -v ffmpeg >/dev/null 2>&1; then FF_ARGS=(--ffmpeg "$(command -v ffmpeg)"); fi
"$PY" "$HERE/code/egocut_infer.py" "$VIDEO" "$OUT" "${FF_ARGS[@]}" || echo "[run.sh] main pipeline exited with an error" >&2
"$PY" "$HERE/code/fallback.py" "$VIDEO" "$OUT"

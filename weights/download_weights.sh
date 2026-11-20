#!/usr/bin/env bash
# Fetch all model weights (about 3.5 GB). Every file is public; a team mirror is tried first because
# Google Drive often rate-limits automated downloads of the original ActionFormer checkpoints.
set -uo pipefail
W="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; cd "$W"
PY="${PYTHON:-python3}"
MIRROR="${EGOCUT_MIRROR:-https://github.com/abishek1123/egocut/releases/download/weights-v1}"
get() {  # get <file> <url...>: first URL that works wins
  local f="$1"; shift
  [ -s "$f" ] && { echo "have $f"; return 0; }
  for u in "$@"; do
    echo "fetching $f <- $u"
    if curl -fL --retry 3 -o "$f.part" "$u"; then mv "$f.part" "$f"; return 0; fi
  done
  rm -f "$f.part"; return 1
}
# 1) SlowFast-R50 trained on EPIC-KITCHENS-100 (official EPIC-KITCHENS release)
get SlowFast.pyth "$MIRROR/SlowFast.pyth" "https://www.dropbox.com/s/uxb6i2xkn91xqzi/SlowFast.pyth?dl=1"
# 2) ActionFormer verb / noun detectors (official ActionFormer EPIC-KITCHENS-100 checkpoints, epoch 20)
for spec in verb:1Ta4ggKSj2YcszSrDbePlHe1ECF1CFKK4 noun:1OTlxeiWj8JE9n1-LsRYogHmqgUdsE5PR; do
  h="${spec%%:*}"; id="${spec#*:}"; f="actionformer_$h.pth.tar"
  if ! get "$f" "$MIRROR/$f"; then
    "$PY" -m pip install -q gdown >/dev/null 2>&1 || true
    "$PY" -m gdown "$id" -O "/tmp/af_$h.zip"
    "$PY" - "$h" <<'PYEOF'
import zipfile, sys, glob, shutil
h = sys.argv[1]; d = f'/tmp/af_{h}'; zipfile.ZipFile(f'/tmp/af_{h}.zip').extractall(d)
shutil.copy(sorted(glob.glob(f'{d}/**/*.pth.tar', recursive=True))[-1], f'actionformer_{h}.pth.tar')
PYEOF
  fi
done
# 3) VideoMAE-L branch (main action detector): EPIC-finetuned VideoMAE-L backbones (OpenTAD release) + our ActionFormer heads
get vmae_backbone_verb.pth "$MIRROR/vmae_backbone_verb.pth" || { "$PY" -m gdown 1h7oLiNN5LTXau4HWmmzS_ekvuNdZkp-b -O vmae_backbone_verb.pth || true; }
get vmae_backbone_noun.pth "$MIRROR/vmae_backbone_noun.pth" || { "$PY" -m gdown 1nRuzJI4ej90vFsKCBSugRVOmxrR8urwW -O vmae_backbone_noun.pth || true; }
get actionformer_vmae_verb.pth.tar "$MIRROR/actionformer_vmae_verb.pth.tar" || echo "WARNING: VideoMAE verb head unavailable - run.sh falls back to the SlowFast branch"
get actionformer_vmae_noun.pth.tar "$MIRROR/actionformer_vmae_noun.pth.tar" || echo "WARNING: VideoMAE noun head unavailable - run.sh falls back to the SlowFast branch"
# 4) SigLIP image-text model (Hugging Face google/siglip-base-patch16-224), saved locally for offline runs
if [ ! -s siglip/model.safetensors ] && [ ! -s siglip/pytorch_model.bin ]; then
  get siglip.tgz "$MIRROR/siglip.tgz" && tar xzf siglip.tgz && rm -f siglip.tgz
fi
if [ ! -s siglip/model.safetensors ] && [ ! -s siglip/pytorch_model.bin ]; then
  "$PY" -c "from transformers import AutoModel, AutoProcessor as P; m='google/siglip-base-patch16-224'; AutoModel.from_pretrained(m).save_pretrained('siglip'); P.from_pretrained(m).save_pretrained('siglip')"
fi
[ -f SHA256SUMS ] || get SHA256SUMS "$MIRROR/SHA256SUMS" || true
[ -f SHA256SUMS ] && (sha256sum -c --ignore-missing SHA256SUMS || echo "WARNING: checksum mismatch - delete the file above and re-run")
echo "weights ready:"; ls -la "$W"

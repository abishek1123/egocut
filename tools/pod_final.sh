#!/usr/bin/env bash
# Final release check on a FRESH GPU pod, exactly as an organiser would do it:
#   1) install from requirements.txt, 2) fetch weights from the GitHub release, 3) run run.sh with the network OFF,
#   4) compare with our submitted outputs/, 5) pin exact versions, 6) publish SigLIP to the release and push the code.
# Usage (on the pod):  cd /workspace && tar xzf DAX_v5.tgz && bash DAX/tools/pod_final.sh
set -u
D=/workspace/DAX; V=/workspace/videos; O=/workspace/test_out; cd $D
step() { echo; echo "========== $* =========="; }

step "1. system packages + python requirements"
apt-get update -qq && apt-get install -y -qq ffmpeg git time curl >/dev/null && echo "apt ok"
pip install -q -r requirements.txt 2>&1 | grep -v -i "already\|warn\|notice" | tail -3

step "2. weights (GitHub release + SigLIP)"
bash weights/download_weights.sh 2>&1 | grep -i "fetching\|have\|warn\|OK\|FAILED\|mismatch" | head -20
ls -la weights weights/siglip | head -30

step "3. test videos"
mkdir -p $V $O
if [ ! -s $V/P03_26.MP4 ]; then
  [ -d /workspace/epic-dl ] || git clone -q https://github.com/epic-kitchens/epic-kitchens-download-scripts.git /workspace/epic-dl
  (cd /workspace/epic-dl && python3 epic_downloader.py --videos --specific-videos P03_26,P26_30,P02_13 --output-path /workspace/EPIC >/dev/null 2>&1)
  find /workspace/EPIC -name "P03_26.MP4" -o -name "P26_30.MP4" -o -name "P02_13.MP4" | xargs -I{} cp {} $V/
fi
ls -la $V

step "4. run.sh with the network switched off"
run_offline() {  # unshare -n = no network at all; if the container forbids it, point every proxy at a dead port
  if unshare -n true 2>/dev/null; then unshare -n "$@"
  else env HTTP_PROXY=http://127.0.0.1:9 HTTPS_PROXY=http://127.0.0.1:9 http_proxy=http://127.0.0.1:9 https_proxy=http://127.0.0.1:9 NO_PROXY= "$@"; fi
}
for f in $V/*.MP4 $V/*.mp4; do
  [ -e "$f" ] || continue; b=$(basename "${f%.*}")
  run_offline /usr/bin/time -f "$b: %e s wall, %M KB peak RAM" bash run.sh "$f" $O/$b.json 2>&1 | grep -i "s wall\|fallback\|error\|missing" | tail -3
done
python3 code/validate_local.py --dir $O -v 2>&1 | grep -v MISSING | tail -5
grep -l '"name": "other", "score": 0.0' $O/*.json 2>/dev/null && echo "!!! a video fell back to the minimal output - check the log above" || echo "no fallbacks: the real pipeline ran offline"

step "5. agreement with the submitted outputs/"
python3 - <<'PY'
import json, glob, os
for f in sorted(glob.glob('/workspace/test_out/*.json')):
    v = os.path.basename(f)[:-5]; a = json.load(open(f)); s = '/workspace/DAX/outputs/' + v + '.json'
    if not os.path.exists(s): print(v, 'not in outputs/'); continue
    b = json.load(open(s)); key = lambda x: (x['verb'], x['noun'], round(x['segment'][0], 1))
    ka = {key(x) for x in a['actions'][:200]}; kb = {key(x) for x in b['actions'][:200]}
    print(f"{v}: label {a['label']['name']} vs {b['label']['name']} | top-200 actions shared {len(ka & kb)}/{max(len(kb), 1)} | highlights identical: {a['highlights'] == b['highlights']}")
PY

step "6. pin exact versions into requirements.txt"
python3 - <<'PY'
import importlib.metadata as md, re
src = open('/workspace/DAX/requirements.txt').read().splitlines(); out = []
for line in src:
    m = re.match(r'^([A-Za-z0-9_.\-]+)', line)
    if not m or line.startswith('#'): out.append(line); continue
    try: out.append(f'{m.group(1)}=={md.version(m.group(1))}')
    except md.PackageNotFoundError: out.append(line)
open('/workspace/DAX/requirements.txt', 'w').write('\n'.join(out) + '\n'); print('\n'.join(out))
PY
python3 -c "import torch, sys; print('python', sys.version.split()[0], '| torch', torch.__version__, '| cuda', torch.version.cuda, '|', torch.cuda.get_device_name(0))"

step "7. publish SigLIP to the release + push the code"
echo "Paste the GitHub token (input hidden), then Enter:"; read -rs GH_TOKEN; export GH_TOKEN
R=abishek1123/egocut; API=https://api.github.com/repos/$R; AUTH="Authorization: Bearer $GH_TOKEN"
(cd weights && tar czf /workspace/siglip.tgz siglip)
REL=$(curl -s -H "$AUTH" $API/releases/tags/weights-v1 | python3 -c "import sys, json; print(json.load(sys.stdin)['id'])")
(cd weights && sha256sum SlowFast.pyth actionformer_vmae_*.pth.tar vmae_backbone_*.pth > /workspace/SHA256SUMS.new)
(cd /workspace && sha256sum siglip.tgz >> SHA256SUMS.new)
for a in siglip.tgz SHA256SUMS; do  # replace an existing asset of the same name
  id=$(curl -s -H "$AUTH" $API/releases/$REL/assets | python3 -c "import sys, json; print(next((x['id'] for x in json.load(sys.stdin) if x['name'] == '$a'), ''))")
  [ -n "$id" ] && curl -s -X DELETE -H "$AUTH" $API/releases/assets/$id
done
cp /workspace/SHA256SUMS.new /workspace/SHA256SUMS
for a in siglip.tgz SHA256SUMS; do
  curl -s -H "$AUTH" -H "Content-Type: application/octet-stream" -T /workspace/$a "https://uploads.github.com/repos/$R/releases/$REL/assets?name=$a" | python3 -c "import sys, json; d = json.load(sys.stdin); print(d.get('name'), d.get('size'), d.get('state'))"
done
cat > .gitignore <<'EOF'
weights/*
!weights/download_weights.sh
!weights/egocut_config.json
__pycache__/
*.pyc
EOF
git init -q 2>/dev/null; git checkout -q -B main
git -c user.name=abishek1123 -c user.email=abishek1123@users.noreply.github.com add -A
git -c user.name=abishek1123 -c user.email=abishek1123@users.noreply.github.com commit -q -m "EgoCut (team DAX): Samsung PRISM XR Hackathon 2026, Track 4 submission

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01KU3hdwATy3e7KxRj17HoSP" && echo committed
git push -q -f "https://x-access-token:$GH_TOKEN@github.com/$R.git" main && echo "pushed to https://github.com/$R"
unset GH_TOKEN
for a in siglip.tgz SHA256SUMS; do curl -s -o /dev/null -w "$a %{http_code}\n" -L https://github.com/$R/releases/download/weights-v1/$a; done
echo; echo "ALL DONE - paste everything from step 2 onward back to Claude."

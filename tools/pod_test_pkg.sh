#!/usr/bin/env bash
# Final package test on a GPU pod: real weights, run.sh exactly as the organizers call it, timing, validity,
# and agreement with the batch detections (preds/af_*_vmae_val_ours.pkl).
set -u; P=/workspace/EgoCut; W=/workspace/pod2; cd $P
pip install -q -r requirements.txt 2>&1 | grep -v -i "already\|warn" | tail -2
mkdir -p weights && cp -n $W/dl/vmae_backbone_verb.pth $W/dl/vmae_backbone_noun.pth weights/
for h in verb noun; do [ -s weights/actionformer_vmae_$h.pth.tar ] || python3 -c "
import torch, glob; ck = sorted(glob.glob('$W/af/ckpt/vmae_full_${h}_run/*.pth.tar'))[-1]
torch.save({'state_dict_ema': torch.load(ck, map_location='cpu', weights_only=False)['state_dict_ema']}, 'weights/actionformer_vmae_$h.pth.tar'); print('head', '$h', ck)"; done
bash weights/download_weights.sh 2>&1 | grep -i "fetching\|have\|warn\|ready" | head -12
mkdir -p /workspace/test_out
for v in P03_26 P26_30 P02_13; do
  /usr/bin/time -f "$v: %e s wall, %M KB peak RAM" bash run.sh $W/videos/$v.MP4 /workspace/test_out/$v.json 2>&1 | grep -v -i "warn" | tail -2
done
python3 code/validate_local.py --dir /workspace/test_out -v 2>&1 | tail -4
python3 - <<'PY'
import json, pickle, sys
sys.path.insert(0, '/workspace/EgoCut/code'); from compose import compose
cfg = json.load(open('/workspace/EgoCut/weights/egocut_config.json'))
P = {h: pickle.load(open(f'/workspace/pod2/preds/af_{h}_vmae_val_ours.pkl', 'rb')) for h in ('verb', 'noun')}
def dets(h, v):
    p = P[h]; return [(float(s), float(e), int(l), float(c)) for vid, s, e, l, c in zip(p['video-id'], p['t-start'], p['t-end'], p['label'], p['score']) if vid == v]
for v in ('P03_26', 'P26_30', 'P02_13'):
    a = json.load(open(f'/workspace/test_out/{v}.json'))['actions'][:200]
    b = compose(dets('verb', v), dets('noun', v), **cfg['compose'])[:200]
    voc = json.load(open('/workspace/EgoCut/code/data/supported_classes.json'))
    key = lambda x, names: (x['verb'] if names else voc['verbs'][x['verb']], x['noun'] if names else voc['nouns'][x['noun']], round(x['segment'][0] if names else x['start'], 1))
    ka = {key(x, True) for x in a}; kb = {key(x, False) for x in b}
    print(f'{v}: run.sh vs batch top-200 actions identical: {len(ka & kb)}/{len(ka)}')
PY

"""Tune composition + selector on the DEV split (held-out training participants). No validation labels used."""
import sys, os, json, pickle, itertools, collections, time
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(__file__))
from compose import compose
from compose_eval import fast_map, coverage, by_video, ts
from selector import select
ROOT = os.path.join(os.path.dirname(__file__), '..')
tag = sys.argv[1] if len(sys.argv) > 1 else 'dev'
V = by_video(pickle.load(open(f'{ROOT}/outputs_dev/af_verb_{tag}.pkl', 'rb')))
N = by_video(pickle.load(open(f'{ROOT}/outputs_dev/af_noun_{tag}.pkl', 'rb')))
csv = 'EPIC_100_train.csv'  # dev split = held-out TRAINING participants only
ann = pd.read_csv(f'{ROOT}/data/ek_annotations/{csv}')
info = pd.read_csv(f'{ROOT}/data/ek_annotations/EPIC_100_video_info.csv').set_index('video_id')
vids = sorted(v for v in V if v in set(ann.video_id))
gt = {v: [dict(verb=r.verb_class, noun=r.noun_class, start=ts(r.start_timestamp), end=ts(r.stop_timestamp)) for r in g.itertuples()]
      for v, g in ann[ann.video_id.isin(vids)].groupby('video_id')}
print(tag, 'videos:', len(vids), flush=True)
# 1) composition grid -> action mAP
grid = dict(top_v=[300, 1000], top_n=[300, 1000], k=[1, 3, 5], min_iou=[0.1, 0.3, 0.5], a=[0.5, 0.65], bound=['verb', 'avg'], sym=[False, True])
keys = list(grid); best = (-1, None); rows = []
t0 = time.time()
for vals in itertools.product(*grid.values()):
    p = dict(zip(keys, vals))
    pred = {v: compose(V[v], N.get(v, []), **p) for v in vids}
    m = fast_map(gt, pred); rows.append({**p, 'mAP': m})
    if m > best[0]:
        best = (m, p); print(f'new best mAP {100*m:.2f}  {p}  ({time.time()-t0:.0f}s)', flush=True)
pd.DataFrame(rows).sort_values('mAP', ascending=False).to_csv(f'{ROOT}/outputs_dev/compose_grid_{tag}.csv', index=False)
bp = best[1]; json.dump(bp, open(f'{ROOT}/outputs_dev/best_compose.json', 'w'))
print('BEST composition:', bp, 'mAP', round(100 * best[0], 2), flush=True)
# 2) selector grid -> coverage (both rules)
pred = {v: compose(V[v], N.get(v, []), **bp) for v in vids}
sg = dict(top=[200, 400, 1000], margin=[0.0, 0.3, 0.6], min_len=[1.0, 2.0, 3.0], budget=[0.145])
srows = []
for top, mg, ml, bu in itertools.product(*sg.values()):
    r = dict(top=top, margin=mg, min_len=ml, budget=bu)
    for m in ('any', 'half'):
        cs, fr = [], []
        for v in vids:
            dur = info.loc[v, 'duration']; hl = select(pred[v][:top], None, dur, budget=bu, min_len=ml, margin=mg, mode=m)
            cs.append(coverage(gt[v], hl, m)); fr.append(sum(h['end'] - h['start'] for h in hl) / dur)
        r[f'cov_{m}'] = np.mean(cs); r[f'frac_{m}'] = np.mean(fr)
    srows.append(r); print({k: (round(x, 3) if isinstance(x, float) else x) for k, x in r.items()}, flush=True)
sdf = pd.DataFrame(srows); sdf.to_csv(f'{ROOT}/outputs_dev/selector_grid_{tag}.csv', index=False)
for m in ('any', 'half'):
    b = sdf.sort_values(f'cov_{m}', ascending=False).iloc[0]
    print(f'BEST selector ({m}):', b.to_dict())

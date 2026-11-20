"""Re-tune under the official kit rules: highlight segments < 4.0 s are ignored in scoring; actions capped at 2000."""
import sys, os, json, pickle, time, itertools
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(__file__))
from compose import compose
from compose_eval import coverage, fast_map, by_video, ts
from selector_fast import select
ROOT = os.path.join(os.path.dirname(__file__), '..'); tag = sys.argv[1]
V = by_video(pickle.load(open(f'{ROOT}/outputs_dev/af_verb_{tag}.pkl', 'rb'))); N = by_video(pickle.load(open(f'{ROOT}/outputs_dev/af_noun_{tag}.pkl', 'rb')))
ann = pd.read_csv(f'{ROOT}/data/ek_annotations/EPIC_100_train.csv')  # dev split = held-out TRAINING participants only
info = pd.read_csv(f'{ROOT}/data/ek_annotations/EPIC_100_video_info.csv').set_index('video_id')
vids = sorted(v for v in V if v in set(ann.video_id))
gt = {v: [dict(verb=r.verb_class, noun=r.noun_class, start=ts(r.start_timestamp), end=ts(r.stop_timestamp)) for r in g.itertuples()] for v, g in ann[ann.video_id.isin(vids)].groupby('video_id')}
bp = dict(top_v=2000, top_n=2000, k=5, min_iou=0.1, a=0.5, bound='verb', sym=True, keep=2000)
pred = {v: compose(V[v], N.get(v, []), **bp) for v in vids}
print(f'{tag} action mAP with 2000-action cap: {100*fast_map(gt, pred):.2f}', flush=True)
def scored(hl): return [h for h in hl if h['end'] - h['start'] >= 4.0 - 1e-6]
def run(cfg, m):
    cs, fr = [], []
    for v in vids:
        dur = info.loc[v, 'duration']; hl = select(pred[v], dur, mode=m, **cfg)
        cs.append(coverage(gt[v], scored(hl), m)); fr.append(sum(h['end'] - h['start'] for h in hl) / dur)
    return round(100 * np.mean(cs), 2), round(100 * np.mean(fr), 2)
if tag == 'dev':
    rows = []
    for ml, mg, top in itertools.product([4.0, 4.5, 5.0, 6.0], [0.0, 0.5], [1000, 2000]):
        cfg = dict(top=top, margin=mg, min_len=ml, fill=True, budget=0.145); r = dict(cfg)
        for m in ('any', 'half'): r[f'cov_{m}'], r[f'frac_{m}'] = run(cfg, m)
        rows.append(r); print(r, flush=True)
    df = pd.DataFrame(rows); df.to_csv(f'{ROOT}/outputs_dev/selector_grid_rules_dev.csv', index=False)
    best = {}
    for m in ('any', 'half'):
        b = df.sort_values(f'cov_{m}', ascending=False).iloc[0]
        best[m] = dict(top=int(b.top), margin=float(b.margin), min_len=float(b.min_len), fill=True, budget=0.145)
    json.dump(best, open(f'{ROOT}/outputs_dev/best_selector.json', 'w')); print('BEST', best)
    # uniform 4 s clips baseline for reference
else:
    best = json.load(open(f'{ROOT}/outputs_dev/best_selector.json'))
    for m in ('any', 'half'):
        print(f'VAL coverage ({m}) {best[m]}: {run(best[m], m)}', flush=True)
        cs = []
        for v in vids:
            dur = info.loc[v, 'duration']; n = max(1, int(0.145 * dur / 4.0)); uni = [dict(start=s, end=s + 4.0) for s in np.linspace(0, max(dur - 4.0, 0), n)]
            cs.append(coverage(gt[v], scored(uni), m))
        print(f'VAL uniform 4 s clips ({m}): {100*np.mean(cs):.2f}', flush=True)

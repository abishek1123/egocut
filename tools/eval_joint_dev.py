import sys, time, json, numpy as np, pandas as pd
sys.path.insert(0, '/home/claude/joint'); sys.path.insert(1, '/home/claude/pkg/EgoCut/code'); sys.path.insert(2, '/home/claude/ego2/code')
from joint import joint, load_raw
from selector_fast import select
from compose_eval import coverage, fast_map, ts
E = '/home/claude/ego2'
ann = pd.read_csv(f'{E}/data/ek_annotations/EPIC_100_train.csv')
RAW = load_raw('dev'); vids = sorted(RAW)
gt = {v: [dict(verb=r.verb_class, noun=r.noun_class, start=ts(r.start_timestamp), end=ts(r.stop_timestamp)) for r in g.itertuples()] for v, g in ann[ann.video_id.isin(vids)].groupby('video_id')}
SEL = dict(top=2000, margin=0.25, min_len=4.1, fill=True, budget=0.145)
def run(name, cov=False, gamma=2.0, **kw):
    t = time.time(); pred = {v: joint(RAW[v], **kw) for v in vids}; m = 100 * fast_map(gt, pred); s = f'{name:40s} mAP {m:5.2f}'
    if cov:
        c = {}
        for mode in ('any', 'half'):
            c[mode] = 100 * np.mean([coverage(gt[v], select([dict(d, score=d['score'] ** gamma) for d in pred[v]], float(RAW[v]['duration']), mode='half', **SEL), mode) for v in vids])
        s += f"  cov any {c['any']:.1f} half {c['half']:.1f}"
    print(s + f'  ({time.time()-t:.0f}s)', flush=True); return m
if __name__ == '__main__':
    print('dev videos', len(vids), '| reference: current pairing mAP 24.13, coverage (half-select) any 50.8 half 43.1')
    run('joint default (EgoAction-like)', cov=True)
    for b in ('verb', 'mean', 'noun'): run(f'bound={b}', bound=b)
    for sg in (0.3, 0.6): run(f'sigma={sg}', sigma=sg)
    for A in (1500, 6000): run(f'A={A}', A=A)

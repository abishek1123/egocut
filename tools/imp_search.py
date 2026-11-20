"""Choose an unsupervised interest strategy on SumMe (dev). Metric mirrors Track 4: F1 vs top-15% consensus frames,
highlights = non-overlapping segments >= 4.1 s, total <= 14.5% of the video."""
import os, sys, glob, itertools, json
import numpy as np, scipy.io as sio

def z(x):
    x = np.asarray(x, float); s = x.std(); return (x - x.mean()) / s if s > 1e-8 else x * 0

def signals(sg, sf, T):
    t = np.arange(T); sg = sg[:T]
    sgn = sg / (np.linalg.norm(sg, axis=1, keepdims=True) + 1e-8)
    mean = sgn.mean(0); mean /= np.linalg.norm(mean) + 1e-8
    nov = 1 - sgn @ mean
    chg = np.r_[0, np.linalg.norm(np.diff(sgn, axis=0), axis=1)]
    # SlowFast: one feature per 16/30 s -> per-second by nearest index
    idx = np.clip((t * 30 / 16).astype(int), 0, len(sf) - 1); f = sf[idx]
    sfn = np.linalg.norm(f, axis=1)
    fn = f / (np.linalg.norm(f, axis=1, keepdims=True) + 1e-8); sfc = np.r_[0, np.linalg.norm(np.diff(fn, axis=0), axis=1)]
    fm = fn.mean(0); fm /= np.linalg.norm(fm) + 1e-8; sfnov = 1 - fn @ fm
    cen = -((t / max(T - 1, 1) - 0.5) ** 2)
    return dict(cen=z(cen), nov=z(nov), chg=z(chg), sfn=z(sfn), sfc=z(sfc), sfnov=z(sfnov))

def select(score, dur, L=6, smooth=5, budget=0.145):
    T = len(score)
    s = np.convolve(score, np.ones(smooth) / smooth, mode='same') if smooth > 1 else score
    win = np.convolve(s, np.ones(L), mode='valid')  # sum over each L-second window starting at i
    taken = np.zeros(T, bool); out = []; used = 0
    for i in np.argsort(-win):
        if taken[i:i + L].any() or used + L > budget * dur: continue
        taken[i:i + L] = True; out.append((float(i), float(min(i + L, dur)))); used += L
    return sorted(out)

def f1(segs, gtb, dur):
    n = len(gtb); fps = n / dur; p = np.zeros(n, bool)
    for a, b in segs: p[int(a * fps):int(np.ceil(b * fps))] = True
    tp = (p & gtb).sum(); return 0.0 if tp == 0 else 2 * tp / (p.sum() + gtb.sum())

def top15(score):
    k = int(round(0.15 * len(score))); b = np.zeros(len(score), bool); b[np.argsort(-score, kind='stable')[:k]] = True; return b

def load_summe(cache='cache', gtdir='summe_gt'):
    data = []
    for m in sorted(glob.glob(os.path.join(gtdir, '*.mat'))):
        name = os.path.basename(m)[:-4]; g = sio.loadmat(m, squeeze_me=True)
        c = np.load(os.path.join(cache, name + '.npz')); dur = float(g['video_duration'])
        T = min(len(c['siglip']), int(np.floor(dur)))
        data.append(dict(name=name, dur=dur, gt=top15(np.asarray(g['gt_score'], float)), sig=signals(c['siglip'], c['slowfast'], T)))
    return data

KEYS = ['cen', 'nov', 'chg', 'sfn', 'sfc', 'sfnov']
def evaluate(data, w, L, sm):
    return np.mean([f1(select(sum(w[k] * d['sig'][k] for k in KEYS), d['dur'], L, sm), d['gt'], d['dur']) for d in data])

if __name__ == '__main__':
    data = load_summe(); print('SumMe videos', len(data))
    base = {}
    for name, w in [('uniform-ish (no signal)', {k: 0 for k in KEYS})] + [(k + ' only', {**{x: 0 for x in KEYS}, k: 1}) for k in KEYS]:
        base[name] = evaluate(data, w, 6, 5)
    # uniform 4 s clips + middle block references
    uni = np.mean([f1([(s, s + 4) for s in np.linspace(0, max(d['dur'] - 4, 0), max(1, int(0.145 * d['dur'] / 4)))], d['gt'], d['dur']) for d in data])
    mid = np.mean([f1([(d['dur'] * 0.4275, d['dur'] * 0.5725)], d['gt'], d['dur']) for d in data])
    print(f'reference: uniform 4 s clips {100*uni:.1f} | middle block {100*mid:.1f}')
    for k, v in base.items(): print(f'  {k:28s} {100*v:.1f}')
    best = []
    grid = [-1, 0, 1, 2]
    for ws in itertools.product(grid, repeat=len(KEYS)):
        if not any(ws): continue
        w = dict(zip(KEYS, ws))
        for L in (4.1, 6, 8, 12):
            Li = int(np.ceil(L))
            for sm in (1, 5, 9):
                best.append((evaluate(data, w, Li, sm), w, Li, sm))
    best.sort(key=lambda x: -x[0])
    for b in best[:8]: print(f'{100*b[0]:.1f}  {b[1]}  L={b[2]} smooth={b[3]}')
    json.dump(dict(f1=best[0][0], w=best[0][1], L=best[0][2], smooth=best[0][3]), open('best_importance.json', 'w'))

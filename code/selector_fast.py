"""Fast vectorised EgoCut selector: greedy expected distinct-pair coverage per second, under a strict budget.
fill=True spends any leftover budget on the highest-confidence remaining candidates."""
import numpy as np

def _merge(ws):
    out = []
    for a, b in sorted(ws):
        if out and a <= out[-1][1]: out[-1][1] = max(out[-1][1], b)
        else: out.append([a, b])
    return out

def select(dets, duration, budget=0.145, min_len=2.0, margin=0.3, mode='half', top=1000, fill=True, importance=None, alpha=1.0):
    dets = sorted(dets, key=lambda d: -d['score'])[:top]
    cap = budget * duration
    if not dets:
        return []
    ds = np.array([d['start'] for d in dets]); de = np.array([d['end'] for d in dets]); sc = np.array([d['score'] for d in dets])
    pk = [(d['verb'], d['noun']) for d in dets]; uniq = {p: i for i, p in enumerate(dict.fromkeys(pk))}; pid = np.array([uniq[p] for p in pk])
    # candidate windows = padded detections (min length enforced)
    s = ds - margin; e = de + margin; short = (e - s) < min_len; c = (s + e) / 2
    s = np.where(short, c - min_len / 2, s); e = np.where(short, c + min_len / 2, e)
    s = np.clip(s, 0, duration); e = np.clip(e, 0, duration)
    ov = np.clip(np.minimum(e[:, None], de[None, :]) - np.maximum(s[:, None], ds[None, :]), 0, None)  # C x D
    hit = ov > 0 if mode == 'any' else ov >= 0.5 * np.maximum(de - ds, 1e-3)[None, :]
    G = np.zeros((len(s), len(uniq)))
    np.maximum.at(G, (np.nonzero(hit)[0], pid[np.nonzero(hit)[1]]), sc[np.nonzero(hit)[1]])
    have = np.zeros(len(uniq)); chosen = []; used = 0.0; alive = np.ones(len(s), bool)
    def new_len():
        L = e - s
        for a, b in _merge(chosen):
            L = L - np.clip(np.minimum(e, b) - np.maximum(s, a), 0, None)
        return L
    while True:
        L = new_len(); ok = alive & (L > 1e-6) & (used + L <= cap)
        if not ok.any(): break
        gain = np.clip(G - have[None, :], 0, None).sum(1)
        val = np.where(ok, gain / np.maximum(L, 1e-6), -1)
        i = int(np.argmax(val))
        if val[i] <= 1e-9:
            if not fill: break
            i = int(np.argmax(np.where(ok, sc, -1)))  # leftover budget: most confident remaining window
        chosen.append((s[i], e[i])); used += L[i]; have = np.maximum(have, G[i]); alive[i] = False
    out = []
    for a, b in sorted(chosen):
        if out and a <= out[-1][1]: out[-1][1] = max(out[-1][1], b)
        else: out.append([a, b])
    return [{'start': round(float(a), 2), 'end': round(float(b), 2), 'importance': 0.0} for a, b in out]

"""Combine verb + noun detections into verb-noun actions, run selectors, score everything.
Metric implementations only (action mAP at tIoU 0.1-0.5, action coverage)."""
import sys, os, pickle, collections, numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(__file__))
from selector import select
ROOT = os.path.join(os.path.dirname(__file__), '..')

def ts(s):
    h, m, x = s.split(':'); return int(h) * 3600 + int(m) * 60 + float(x)

def by_video(p):
    d = collections.defaultdict(list)
    for v, s, e, l, c in zip(p['video-id'], p['t-start'], p['t-end'], p['label'], p['score']):
        d[v].append((float(s), float(e), int(l), float(c)))
    return d

def compose(verbs, nouns, top_v=300, top_n=300, k_noun=3, min_iou=0.3, keep=1000):
    verbs = sorted(verbs, key=lambda x: -x[3])[:top_v]; nouns = sorted(nouns, key=lambda x: -x[3])[:top_n]
    if not verbs or not nouns:
        return []
    N = np.array(nouns)
    acts = []
    for s, e, vl, vc in verbs:
        inter = np.clip(np.minimum(e, N[:, 1]) - np.maximum(s, N[:, 0]), 0, None)
        iou = inter / (np.maximum(e, N[:, 1]) - np.minimum(s, N[:, 0]) + 1e-6)
        cand = np.where(iou >= min_iou)[0]
        cand = cand[np.argsort(-N[cand, 3])][:k_noun]
        for j in cand:
            acts.append(dict(verb=int(vl), noun=int(N[j, 2]), start=s, end=e, score=float(np.sqrt(vc * N[j, 3]))))
    acts.sort(key=lambda a: -a['score'])
    return acts[:keep]

def fast_map(gt, pred, ths=(0.1, 0.2, 0.3, 0.4, 0.5)):
    """EPIC-style action mAP: mean over action classes present in GT, averaged over tIoU thresholds."""
    G = collections.defaultdict(lambda: collections.defaultdict(list))
    for v, gl in gt.items():
        for g in gl:
            G[(g['verb'], g['noun'])][v].append((g['start'], g['end']))
    P = collections.defaultdict(list)
    for v, pl in pred.items():
        for p in pl:
            P[(p['verb'], p['noun'])].append((p['score'], v, p['start'], p['end']))
    out = []
    for t in ths:
        aps = []
        for c, gv in G.items():
            npos = sum(len(x) for x in gv.values())
            pl = sorted(P.get(c, []), key=lambda x: -x[0])
            used = {v: np.zeros(len(x), bool) for v, x in gv.items()}
            tp = np.zeros(len(pl))
            for i, (_, v, s, e) in enumerate(pl):
                g = gv.get(v)
                if not g:
                    continue
                g = np.array(g); inter = np.clip(np.minimum(e, g[:, 1]) - np.maximum(s, g[:, 0]), 0, None)
                iou = inter / (np.maximum(e, g[:, 1]) - np.minimum(s, g[:, 0]))
                iou[used[v]] = -1; j = int(np.argmax(iou))
                if iou[j] >= t:
                    tp[i] = 1; used[v][j] = True
            if len(pl) == 0:
                aps.append(0.0); continue
            ctp = np.cumsum(tp); rec = ctp / npos; prec = ctp / np.arange(1, len(pl) + 1)
            mrec = np.r_[0, rec, 1]; mpre = np.r_[0, prec, 0]
            mpre = np.maximum.accumulate(mpre[::-1])[::-1]
            idx = np.where(mrec[1:] != mrec[:-1])[0]
            aps.append(float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1])))
        out.append(np.mean(aps))
    return float(np.mean(out))

def coverage(gt, hl, mode):
    pairs = {(g['verb'], g['noun']) for g in gt}; got = set()
    for g in gt:
        L = max(g['end'] - g['start'], 1e-3)
        ov = sum(max(0.0, min(h['end'], g['end']) - max(h['start'], g['start'])) for h in hl)
        if (mode == 'any' and ov > 0) or (mode == 'half' and ov >= 0.5 * L):
            got.add((g['verb'], g['noun']))
    return len(got) / max(1, len(pairs))

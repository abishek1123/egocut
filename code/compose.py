"""Verb + noun -> verb-noun action composition, with tunable options.
Repeated action instances are kept (dedup belongs in the selector, not here)."""
import numpy as np

def _iou(s, e, A):
    inter = np.clip(np.minimum(e, A[:, 1]) - np.maximum(s, A[:, 0]), 0, None)
    return inter / (np.maximum(e, A[:, 1]) - np.minimum(s, A[:, 0]) + 1e-6)

def compose(verbs, nouns, top_v=500, top_n=500, k=3, min_iou=0.3, a=0.5, bound='verb', sym=False, keep=2000, nms_iou=0.9):
    V = np.array(sorted(verbs, key=lambda x: -x[3])[:top_v]); N = np.array(sorted(nouns, key=lambda x: -x[3])[:top_n])
    if len(V) == 0 or len(N) == 0:
        return []
    acts = []
    def pair(A, B, a_is_verb):
        for s, e, l, c in A:
            iou = _iou(s, e, B); idx = np.where(iou >= min_iou)[0]
            idx = idx[np.argsort(-B[idx, 3])][:k]
            for j in idx:
                vs, vl, vc = (c, l, (s, e)) if a_is_verb else (B[j, 3], B[j, 2], (B[j, 0], B[j, 1]))
                ns, nl, nc = (B[j, 3], B[j, 2], (B[j, 0], B[j, 1])) if a_is_verb else (c, l, (s, e))
                if bound == 'verb':   st, en = vc
                elif bound == 'noun': st, en = nc
                elif bound == 'avg':  st, en = (vc[0] + nc[0]) / 2, (vc[1] + nc[1]) / 2
                else:                 st, en = max(vc[0], nc[0]), min(vc[1], nc[1])
                if en <= st: st, en = vc
                acts.append((float(st), float(en), int(vl), int(nl), float(vs ** a * ns ** (1 - a) * (0.5 + 0.5 * iou[j]))))
    pair(V, N, True)
    if sym:
        pair(N, V, False)
    acts.sort(key=lambda x: -x[4])
    # remove near-identical duplicates of the same verb-noun pair (keeps genuine repeats)
    out, seen = [], {}
    for st, en, vl, nl, sc in acts:
        lst = seen.setdefault((vl, nl), [])
        if any(min(en, e2) - max(st, s2) > nms_iou * (max(en, e2) - min(st, s2)) for s2, e2 in lst):
            continue
        lst.append((st, en)); out.append(dict(verb=vl, noun=nl, start=st, end=en, score=sc))
        if len(out) >= keep:
            break
    return out

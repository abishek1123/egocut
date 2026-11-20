"""EgoCut action composition (EgoAction-style, CVPR 2026 EPIC challenge) + recognition re-scoring.
Verb and noun ActionFormer heads share one anchor grid, so each anchor carries a top-10 verb and a top-10 noun
distribution: pairs are scored sqrt(p_verb * p_noun), boundaries are the confidence-weighted mix of the two heads,
then class-wise soft-NMS. Each action is finally re-scored with the EPIC-finetuned VideoMAE-L recognition heads
averaged over its span: score * (p_cls(verb) * p_cls(noun)) ** beta. All settings were tuned on held-out
EPIC training participants P04/P22/P30."""
import numpy as np, torch

def joint(R, kv=10, kn=10, A=12000, pre=30000, sigma=0.6, min_score=0.001, maxn=2000, power=0.5, vote=0.75, stride=8, nf=16, fps=30.0):
    from libs.utils.nms import batched_nms
    pv = R['verb_p'].astype(np.float32)[:, :kv]; iv = R['verb_i'][:, :kv].astype(np.int64)
    pn = R['noun_p'].astype(np.float32)[:, :kn]; inn = R['noun_i'][:, :kn].astype(np.int64)
    Cv, Cn = pv[:, 0], pn[:, 0]; a = np.argsort(-(Cv * Cn))[:A]
    s = (pv[a, :, None] * pn[a, None, :]) ** power
    w = (Cn[a] / (Cn[a] + Cv[a] + 1e-6))[:, None]; seg = w * R['noun_seg'][a] + (1 - w) * R['verb_seg'][a]
    lab = iv[a][:, :, None] * 300 + inn[a][:, None, :]
    seg = np.repeat(seg[:, None, None, :], kv, 1).repeat(kn, 2).reshape(-1, 2); s = s.reshape(-1); lab = lab.reshape(-1)
    o = np.argsort(-s)[:pre]; seg, s, lab = seg[o], s[o], lab[o]
    segs, sc, lb = batched_nms(torch.from_numpy(seg).float(), torch.from_numpy(s).float(), torch.from_numpy(lab), 0.1, min_score, maxn,
                               use_soft_nms=True, multiclass=True, sigma=sigma, voting_thresh=vote)
    segs = ((segs * stride + 0.5 * nf) / fps).clamp(0, float(R['duration'])).numpy(); sc = sc.numpy(); lb = lb.numpy()
    return [dict(verb=int(l // 300), noun=int(l % 300), start=float(x[0]), end=float(x[1]), score=float(c)) for x, c, l in zip(segs, sc, lb)]

def softmax_probs(f, W, b):
    x = f.astype(np.float32) @ W.T + b; x = np.exp(x - x.max(1, keepdims=True)); return x / x.sum(1, keepdims=True)

def rescore(dets, pv, pn, beta=0.6, stride=8, nf=16, fps=30.0):
    T = len(pv); cs = np.vstack([np.zeros((1, pv.shape[1])), np.cumsum(pv, 0)]); cn = np.vstack([np.zeros((1, pn.shape[1])), np.cumsum(pn, 0)]); out = []
    off = 0.5 * nf
    for d in dets:
        a = int(np.clip(np.floor((d['start'] * fps - off) / stride) + 1, 0, T - 1)); b = int(np.clip(np.ceil((d['end'] * fps - off) / stride) + 1, a + 1, T))
        r = ((cs[b, d['verb']] - cs[a, d['verb']]) / (b - a)) * ((cn[b, d['noun']] - cn[a, d['noun']]) / (b - a))
        out.append(dict(d, score=d['score'] * (r + 1e-6) ** beta))
    return sorted(out, key=lambda x: -x['score'])

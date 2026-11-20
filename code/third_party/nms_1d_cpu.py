"""Pure NumPy drop-in for ActionFormer's compiled nms_1d_cpu extension (no C++ compiler needed on Windows).
Mirrors csrc/nms_cpu.cpp: nms(segs, scores, iou_threshold) and softnms(segs, scores, dets, ...)."""
import numpy as np
import torch

def _iou(x1, x2, area, i, rest):
    inter = np.clip(np.minimum(x2[i], x2[rest]) - np.maximum(x1[i], x1[rest]), 0, None)
    return inter / (area[i] + area[rest] - inter)

def nms(segs, scores, iou_threshold):
    s = segs.numpy().astype(np.float32); sc = scores.numpy().astype(np.float32)
    if len(s) == 0:
        return torch.empty(0, dtype=torch.long)
    x1, x2 = s[:, 0], s[:, 1]; area = x2 - x1 + 1e-6
    order = np.argsort(-sc, kind="stable"); keep = []
    while order.size:
        i = order[0]; keep.append(i)
        rest = order[1:]
        order = rest[_iou(x1, x2, area, i, rest) < iou_threshold]
    return torch.as_tensor(np.array(keep, dtype=np.int64))

def softnms(segs, scores, dets, iou_threshold, sigma, min_score, method):
    s = segs.numpy().astype(np.float32); sc = scores.numpy().astype(np.float32).copy()
    n = len(s)
    if n == 0:
        return torch.empty(0, dtype=torch.long)
    x1, x2 = s[:, 0], s[:, 1]; area = x2 - x1 + 1e-6
    alive = np.arange(n); out_idx = []; out_det = []
    while alive.size:
        j = int(np.argmax(sc[alive])); i = alive[j]
        out_idx.append(i); out_det.append((x1[i], x2[i], sc[i]))
        rest = np.delete(alive, j)
        if rest.size:
            ovr = _iou(x1, x2, area, i, rest)
            if method == 0:
                w = np.where(ovr >= iou_threshold, 0.0, 1.0)
            elif method == 1:
                w = np.where(ovr >= iou_threshold, 1 - ovr, 1.0)
            else:
                w = np.exp(-(ovr * ovr) / sigma)
            sc[rest] *= w
            rest = rest[sc[rest] >= min_score]
        alive = rest
    k = len(out_idx)
    dets[:k] = torch.as_tensor(np.array(out_det, dtype=np.float32))
    return torch.as_tensor(np.array(out_idx, dtype=np.int64))

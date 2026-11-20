"""Check our SlowFast features against the official ActionFormer EPIC features (cosine similarity per time step)."""
import sys, os, glob, numpy as np
ours_dir, official_dir = sys.argv[1], sys.argv[2]
for f in sorted(glob.glob(os.path.join(ours_dir, '*.npz'))):
    vid, tag = os.path.basename(f)[:-4].split('__')
    a = np.load(f)['feats']; o = np.load(os.path.join(official_dir, vid + '.npz')); b = o[o.files[0]]
    if b.shape[0] == 2304 and b.shape[1] != 2304: b = b.T
    best = None
    for off in range(-3, 4):
        x, y = (a[off:], b) if off >= 0 else (a, b[-off:])
        n = min(len(x), len(y)); x, y = x[:n], y[:n]
        cos = (x * y).sum(1) / (np.linalg.norm(x, axis=1) * np.linalg.norm(y, axis=1) + 1e-9)
        if best is None or cos.mean() > best[0]: best = (cos.mean(), off, n)
    print(f'{vid:8s} {tag:14s} ours {a.shape} official {b.shape}  best mean cosine {best[0]:.3f} at offset {best[1]} over {best[2]} steps')

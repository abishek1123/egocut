"""EPIC-KITCHENS SlowFast-R50 (8x8) feature extractor, matching ActionFormer's EPIC features:
32-frame clips at 30 fps, stride 16 frames, 2304-d = avg-pooled slow (2048) + fast (256) pathways, before the classifier."""
import os, sys, argparse, subprocess, time
import numpy as np, torch
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, 'third_party', 'slowfast_epic'))
from slowfast.config.defaults import get_cfg
from slowfast.models.video_model_builder import SlowFast

def build(weights, device):
    cfg = get_cfg(); cfg.merge_from_file(os.path.join(HERE, 'third_party', 'slowfast_epic', 'configs', 'EPIC-KITCHENS', 'SLOWFAST_8x8_R50.yaml'))
    cfg.NUM_GPUS = 1 if device == 'cuda' else 0
    model = SlowFast(cfg)
    if weights:
        ck = torch.load(weights, map_location='cpu')
        sd = ck.get('model_state', ck)
        missing, unexpected = model.load_state_dict(sd, strict=False)
        print('weights loaded; missing', len(missing), 'unexpected', len(unexpected), flush=True)
    head = model.head
    def feat_forward(inputs):  # pooled features before dropout / projection
        pool = [getattr(head, f'pathway{p}_avgpool')(inputs[p]) for p in range(head.num_pathways)]
        x = torch.cat(pool, 1).permute((0, 2, 3, 4, 1))
        return x.mean([1, 2, 3])
    head.forward = feat_forward
    return model.eval().to(device)

def decode(path, fps=30, short=256, ffmpeg='ffmpeg'):
    probe = subprocess.run([ffmpeg, '-i', path], capture_output=True, text=True).stderr
    import re
    m = re.search(r', (\d+)x(\d+)[, ]', probe); w, h = int(m.group(1)), int(m.group(2))
    if w >= h: nh, nw = short, int(round(w * short / h / 2)) * 2
    else: nw, nh = short, int(round(h * short / w / 2)) * 2
    cmd = [ffmpeg, '-loglevel', 'error', '-i', path, '-vf', f'fps={fps},scale={nw}:{nh}', '-pix_fmt', 'rgb24', '-f', 'rawvideo', '-']
    raw = subprocess.run(cmd, capture_output=True).stdout
    return np.frombuffer(raw, np.uint8).reshape(-1, nh, nw, 3)

def extract(model, frames, device, sr=2, stride=16, n_frames=32, crop='center', batch=8, alpha=4, bgr=False):
    T = len(frames); span = n_frames * sr
    starts = list(range(0, max(T - span, 0) + 1, stride)) if T >= span else [0]
    if T >= span and (T - span) % stride:  # cover the tail
        pass
    mean = torch.tensor([0.45, 0.45, 0.45], device=device).view(1, 3, 1, 1, 1); std = torch.tensor([0.225] * 3, device=device).view(1, 3, 1, 1, 1)
    H, W = frames.shape[1:3]
    if crop == 'center':
        c = min(H, W); y0, x0 = (H - c) // 2, (W - c) // 2
    feats = []
    with torch.no_grad(), torch.autocast(device_type='cuda', dtype=torch.float16, enabled=(device == 'cuda')):
        for b in range(0, len(starts), batch):
            clips = []
            for s in starts[b:b + batch]:
                idx = np.clip(np.arange(s, s + span, sr), 0, T - 1)
                cl = frames[idx]
                if crop == 'center': cl = cl[:, y0:y0 + c, x0:x0 + c]
                clips.append(torch.from_numpy(np.ascontiguousarray(cl)))
            x = torch.stack(clips).to(device).permute(0, 4, 1, 2, 3).float() / 255.0  # B C T H W
            if bgr: x = x[:, [2, 1, 0]]
            x = (x - mean) / std
            slow = torch.index_select(x, 2, torch.linspace(0, x.shape[2] - 1, x.shape[2] // alpha, device=device).long())
            feats.append(model([slow, x]).float().cpu())
    return torch.cat(feats).numpy()

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('video'); ap.add_argument('out')
    ap.add_argument('--weights', default=os.path.join(HERE, '..', 'weights', 'SlowFast.pyth'))
    ap.add_argument('--sr', type=int, default=2); ap.add_argument('--bgr', action='store_true'); ap.add_argument('--crop', default='center'); ap.add_argument('--ffmpeg', default='ffmpeg')
    a = ap.parse_args()
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    t = time.time(); fr = decode(a.video, ffmpeg=a.ffmpeg); print('decoded', fr.shape, f'{time.time()-t:.0f}s', flush=True)
    m = build(a.weights if os.path.exists(a.weights) else None, dev)
    t = time.time(); f = extract(m, fr, dev, sr=a.sr, crop=a.crop, bgr=a.bgr); print('features', f.shape, f'{time.time()-t:.0f}s', flush=True)
    np.savez(a.out, feats=f)

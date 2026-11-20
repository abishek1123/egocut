"""VideoMAE-L (EPIC-finetuned, OpenTAD release) feature extractor, plain PyTorch.
Architecture = mmaction VisionTransformer: ViT-L/16, 16 frames, tubelet 2, img 224, sinusoidal pos-embed,
qkv with q/v bias, mean pooling + fc_norm.  One 1024-d feature per 16-frame snippet, stride 8 frames at 30 fps.
Usage:
  python vmae.py keys  <ckpt>                         # print checkpoint keys (sanity)
  python vmae.py match <ckpt> <released_dir> <video...> # try preprocessing variants vs released features
  python vmae.py extract <ckpt> <out_dir> --variant V --list videos.txt
"""
import os, sys, math, json, subprocess, argparse, time
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F

def sinusoid(n, d):
    vec = torch.arange(d, dtype=torch.float64); vec = (vec - vec % 2) / d; vec = torch.pow(10000, -vec).view(1, -1)
    t = torch.arange(n).view(-1, 1) * vec; t[:, 0::2].sin_(); t[:, 1::2].cos_()
    return t.to(torch.float32).unsqueeze(0)

class Attn(nn.Module):
    def __init__(s, d, h):
        super().__init__(); s.h = h; s.qkv = nn.Linear(d, 3 * d, bias=False); s.q_bias = nn.Parameter(torch.zeros(d)); s.v_bias = nn.Parameter(torch.zeros(d)); s.proj = nn.Linear(d, d)
    def forward(s, x):
        B, N, C = x.shape
        b = torch.cat([s.q_bias, torch.zeros_like(s.v_bias), s.v_bias])
        q, k, v = F.linear(x, s.qkv.weight, b).reshape(B, N, 3, s.h, C // s.h).permute(2, 0, 3, 1, 4)
        return s.proj(F.scaled_dot_product_attention(q, k, v).transpose(1, 2).reshape(B, N, C))

class Block(nn.Module):
    def __init__(s, d, h):
        super().__init__(); s.norm1 = nn.LayerNorm(d, eps=1e-6); s.attn = Attn(d, h); s.norm2 = nn.LayerNorm(d, eps=1e-6)
        s.fc1 = nn.Linear(d, 4 * d); s.fc2 = nn.Linear(4 * d, d)
    def forward(s, x):
        x = x + s.attn(s.norm1(x)); return x + s.fc2(F.gelu(s.fc1(s.norm2(x))))

class ViT(nn.Module):
    def __init__(s, d=1024, depth=24, h=16, frames=16, img=224, p=16, tub=2):
        super().__init__(); s.proj = nn.Conv3d(3, d, (tub, p, p), (tub, p, p))
        s.n = (frames // tub) * (img // p) ** 2; s.register_buffer('pos', sinusoid(s.n, d), persistent=False)
        s.blocks = nn.ModuleList([Block(d, h) for _ in range(depth)]); s.fc_norm = nn.LayerNorm(d, eps=1e-6)
    def forward(s, x):  # x: B,3,T,H,W normalised
        x = s.proj(x).flatten(2).transpose(1, 2)
        x = x + (s.pos if x.shape[1] == s.n else sinusoid(x.shape[1], x.shape[2]).to(x))
        for b in s.blocks: x = b(x)
        m = x.mean(1)
        return s.fc_norm(m), m   # (with fc_norm, raw mean)

def load(ckpt, dev):
    sd = torch.load(ckpt, map_location='cpu', weights_only=False)
    sd = sd.get('state_dict', sd.get('model', sd.get('module', sd)))
    out = {}
    for k, v in sd.items():
        k = k.replace('module.', '').replace('backbone.', '')
        if k.startswith(('cls_head', 'head.', 'fc_cls')): continue
        k = (k.replace('patch_embed.projection.', 'proj.').replace('patch_embed.proj.', 'proj.')
              .replace('mlp.layers.0.0.', 'fc1.').replace('mlp.layers.1.', 'fc2.').replace('mlp.fc1.', 'fc1.').replace('mlp.fc2.', 'fc2.')
              .replace('ffn.layers.0.0.', 'fc1.').replace('ffn.layers.1.', 'fc2.'))
        out[k] = v
    m = ViT(); miss, unexp = m.load_state_dict(out, strict=False)
    miss = [k for k in miss if k != 'pos']; unexp = [k for k in unexp if k not in ('pos_embed',)]
    print(f'[vmae] loaded {ckpt}: missing {len(miss)} {miss[:6]} | unexpected {len(unexp)} {unexp[:6]}', flush=True)
    if 'pos_embed' in out and out['pos_embed'].shape[1] == m.n: m.pos = out['pos_embed'].float()
    return m.eval().to(dev).half()

MEAN = torch.tensor([123.675, 116.28, 103.53]).view(1, 3, 1, 1, 1); STD = torch.tensor([58.395, 57.12, 57.375]).view(1, 3, 1, 1, 1)

def frames(path, w=512, h=288, ff='ffmpeg', max_frames=0):
    """frames at 30 fps, resized to w x h (OpenTAD extracted from 30 fps 512x288 videos), uint8 T,H,W,3"""
    lim = ['-frames:v', str(max_frames)] if max_frames else []
    p = subprocess.run([ff, '-loglevel', 'error', '-threads', '8', '-i', path, '-vf', f'fps=30,scale={w}:{h}', *lim, '-pix_fmt', 'rgb24', '-f', 'rawvideo', '-'], stdout=subprocess.PIPE, check=True)
    return np.frombuffer(p.stdout, np.uint8).reshape(-1, h, w, 3)

def frame_chunks(path, w=512, h=288, ff='ffmpeg', chunk=1024):
    p = subprocess.Popen([ff, '-loglevel', 'error', '-threads', '8', '-i', path, '-vf', f'fps=30,scale={w}:{h}', '-pix_fmt', 'rgb24', '-f', 'rawvideo', '-'], stdout=subprocess.PIPE)
    fb = w * h * 3
    while True:
        buf = p.stdout.read(fb * chunk)
        if not buf: break
        n = len(buf) // fb
        yield np.frombuffer(buf[:n * fb], np.uint8).reshape(n, h, w, 3)
    p.wait()

@torch.no_grad()
def extract_stream(model, path, variant, dev, start=0, stride=8, length=16, bs=16, raw=False, ff='ffmpeg'):
    """same snippets as extract(), but decodes the video in chunks (constant memory)"""
    models = model if isinstance(model, (list, tuple)) else [model]
    buf = np.zeros((0, 288, 512, 3), np.uint8); base = 0; i = 0; feats = [[] for _ in models]; pending = []
    def run(clips):
        x = prep(torch.from_numpy(np.stack(clips)), variant, dev)
        for j, m in enumerate(models):
            a, b = m(x); feats[j].append((b if raw else a).float().cpu())
    pad = max(0, -start); start = max(0, start)
    for ch in frame_chunks(path, ff=ff):
        if pad: ch = np.concatenate([np.repeat(ch[:1], pad, 0), ch]); pad = 0
        buf = np.concatenate([buf, ch])
        while start + i * stride + length <= base + len(buf):
            s0 = start + i * stride - base; pending.append(buf[s0:s0 + length]); i += 1
            if len(pending) == bs: run(pending); pending = []
        keep = start + i * stride - base
        if keep > 0: buf = buf[keep:]; base += keep
    T = base + len(buf)
    while T > 0 and start + i * stride < T:  # tail snippets: clamp past-the-end frames to the last frame
        idx = np.clip(np.arange(start + i * stride, start + i * stride + length), 0, T - 1) - base
        pending.append(buf[np.clip(idx, 0, len(buf) - 1)]); i += 1
        if len(pending) == bs: run(pending); pending = []
    if pending: run(pending)
    res = [torch.cat(f).numpy() if f else np.zeros((0, 1024), np.float32) for f in feats]
    return res if isinstance(model, (list, tuple)) else res[0]

def prep(clip, variant, dev):
    """clip: uint8 tensor N,16,H,W,3 -> N,3,16,224,224 normalised"""
    x = clip.to(dev).permute(0, 4, 1, 2, 3).float()  # N,3,T,H,W
    N, C, T, H, W = x.shape
    x = x.transpose(1, 2).reshape(N * T, C, H, W)
    if variant in ('short224_center', 'short224_center_raw'):
        nh, nw = 224, int(round(W * 224 / H)); x = F.interpolate(x, (nh, nw), mode='bilinear', align_corners=False)
        o = (nw - 224) // 2; x = x[:, :, :, o:o + 224]
    elif variant == 'squash224':
        x = F.interpolate(x, (224, 224), mode='bilinear', align_corners=False)
    elif variant == 'short256_center':
        nh, nw = 256, int(round(W * 256 / H)); x = F.interpolate(x, (nh, nw), mode='bilinear', align_corners=False)
        oy, ox = (nh - 224) // 2, (nw - 224) // 2; x = x[:, :, oy:oy + 224, ox:ox + 224]
    x = x.reshape(N, T, C, 224, 224).transpose(1, 2)
    return ((x - MEAN.to(dev)) / STD.to(dev)).half()

@torch.no_grad()
def extract(model, fr, variant, dev, start=0, stride=8, length=16, bs=16, raw=False):
    """features for snippets starting at frame start + i*stride (frames beyond the end are clamped)"""
    T = len(fr); n = max(1, math.ceil((T - start) / stride)) if T > 0 else 0
    frt = torch.from_numpy(fr); feats = []
    for i0 in range(0, n, bs):
        idx = torch.stack([torch.clamp(torch.arange(start + i * stride, start + i * stride + length), 0, T - 1) for i in range(i0, min(n, i0 + bs))])
        a, b = model(prep(frt[idx], variant, dev))
        feats.append((b if raw else a).float().cpu())
    return torch.cat(feats).numpy()

def cos(a, b):
    a = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-8); b = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-8); return float((a * b).sum(1).mean())

if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('cmd'); ap.add_argument('ckpt'); ap.add_argument('rest', nargs='*')
    ap.add_argument('--variant', default='short224_center'); ap.add_argument('--start', type=int, default=0); ap.add_argument('--list', default='')
    a = ap.parse_args(); dev = 'cuda'
    if a.cmd == 'keys':
        sd = torch.load(a.ckpt, map_location='cpu', weights_only=False); sd = sd.get('state_dict', sd.get('model', sd))
        ks = list(sd.keys()); print(len(ks), 'keys'); [print(k, tuple(sd[k].shape)) for k in ks[:12] + ks[-8:]]; sys.exit()
    m = load(a.ckpt, dev)
    if a.cmd == 'match':
        ref_dir, vids = a.rest[0], a.rest[1:]; best = {}
        for v in vids:
            vid = os.path.splitext(os.path.basename(v))[0]; ref = np.load(os.path.join(ref_dir, vid + '.npy'))
            fr = frames(v, max_frames=30 * 120); print(f'{vid}: first {len(fr)} frames vs released {ref.shape}', flush=True)
            for var in ('short224_center', 'squash224', 'short256_center'):
                for raw in (False, True):
                    for st in (0, 4):
                        f = extract(m, fr, var, dev, start=st, raw=raw)
                        r = ref[:len(f)]; shifts = {}
                        for s in (-2, -1, 0, 1, 2):
                            aa, bb = (f[max(0, -s):], r[max(0, s):]); L = min(len(aa), len(bb)); shifts[s] = cos(aa[:L], bb[:L])
                        s_best = max(shifts, key=shifts.get); key = (var, 'raw' if raw else 'fcnorm', st)
                        best.setdefault(key, []).append(shifts[s_best])
                        print(f'   {var:16s} {"raw" if raw else "fcnorm":6s} start{st:+d}: best shift {s_best:+d} cos {shifts[s_best]:.3f}', flush=True)
        print('== mean cosine per variant ==')
        for k, v in sorted(best.items(), key=lambda kv: -np.mean(kv[1])): print(f'{np.mean(v):.4f}', k)
        json.dump({' '.join(map(str, k)): float(np.mean(v)) for k, v in best.items()}, open('match_result.json', 'w'), indent=1)
    if a.cmd == 'extract':
        out = a.rest[0]; os.makedirs(out, exist_ok=True); raw = a.variant.endswith('_raw')
        vids = [l.strip() for l in open(a.list) if l.strip()]
        for i, v in enumerate(vids):
            vid = os.path.splitext(os.path.basename(v))[0]; o = os.path.join(out, vid + '.npy')
            if os.path.exists(o): continue
            t = time.time(); f = extract_stream(m, v, a.variant.replace('_raw', ''), dev, start=a.start, raw=raw)
            np.save(o, f.astype(np.float16)); print(f'[{i+1}/{len(vids)}] {vid} {f.shape} {time.time()-t:.0f}s', flush=True)

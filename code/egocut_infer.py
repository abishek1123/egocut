"""EgoCut single-pass inference: MP4 -> official Track 4 JSON.
decode (stream) -> SlowFast-EPIC features -> ActionFormer verb + noun -> verb-noun composition
-> SigLIP frame embeddings (label + importance) -> EgoCut budgeted selector -> JSON.
Batch mode (--list) loads models once and caches features per video."""
import os, sys, json, argparse, subprocess, re, time
import numpy as np, torch
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.join(HERE, '..')
sys.path.insert(0, HERE)
from extract_slowfast import build as build_slowfast
from compose import compose
from selector_fast import select
from write_official import write, LABELS
from joint import joint, rescore, softmax_probs
CFG = json.load(open(os.path.join(ROOT, 'weights', 'egocut_config.json')))
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'

# ---------------- video ----------------
def probe(path, ff):
    err = subprocess.run([ff, '-i', path], capture_output=True, text=True).stderr
    m = re.search(r'Duration: (\d+):(\d+):([\d.]+)', err); dur = int(m.group(1)) * 3600 + int(m.group(2)) * 60 + float(m.group(3))
    m = re.search(r', (\d{2,5})x(\d{2,5})[, ]', err); return dur, int(m.group(1)), int(m.group(2))

def stream_frames(path, ff, short=256, fps=30, chunk=256):
    dur, w, h = probe(path, ff)
    if w >= h: nh, nw = short, int(round(w * short / h / 2)) * 2
    else: nw, nh = short, int(round(h * short / w / 2)) * 2
    p = subprocess.Popen([ff, '-loglevel', 'error', '-threads', '8', '-i', path, '-threads', '8', '-filter_threads', '4', '-vf', f'fps={fps},scale={nw}:{nh}', '-pix_fmt', 'rgb24', '-f', 'rawvideo', '-'], stdout=subprocess.PIPE)
    fb = nh * nw * 3
    while True:
        buf = p.stdout.read(fb * chunk)
        if not buf: break
        n = len(buf) // fb
        yield np.frombuffer(buf[:n * fb], np.uint8).reshape(n, nh, nw, 3)
    p.wait()

# ---------------- features ----------------
class Featurizer:
    def __init__(self, ff):
        self.ff = ff; self.sf_cfg = CFG['slowfast']
        self.sf = build_slowfast(os.path.join(ROOT, 'weights', 'SlowFast.pyth'), DEV)
        from transformers import AutoModel, AutoProcessor
        name = os.path.join(ROOT, 'weights', 'siglip') if os.path.isdir(os.path.join(ROOT, 'weights', 'siglip')) else CFG['siglip']
        self.sg = AutoModel.from_pretrained(name, torch_dtype=torch.float16 if DEV == 'cuda' else torch.float32).eval().to(DEV)
        self.proc = AutoProcessor.from_pretrained(name)
        self.mean = torch.tensor([0.45] * 3, device=DEV).view(1, 3, 1, 1, 1); self.std = torch.tensor([0.225] * 3, device=DEV).view(1, 3, 1, 1, 1)

    @torch.no_grad()
    def _sf_batch(self, clips):
        x = torch.from_numpy(np.stack(clips)).to(DEV).permute(0, 4, 1, 2, 3).float() / 255.0
        if self.sf_cfg.get('bgr'): x = x[:, [2, 1, 0]]  # EPIC SlowFast was trained on cv2 (BGR) frames
        x = (x - self.mean) / self.std
        slow = torch.index_select(x, 2, torch.linspace(0, x.shape[2] - 1, x.shape[2] // 4, device=DEV).long())
        with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=(DEV == 'cuda')):
            return self.sf([slow, x]).float().cpu().numpy()

    @torch.no_grad()
    def _sg_batch(self, imgs):
        inp = self.proc(images=list(imgs), return_tensors='pt').to(DEV)
        e = self.sg.get_image_features(pixel_values=inp['pixel_values'].to(self.sg.dtype))
        return torch.nn.functional.normalize(getattr(e, 'pooler_output', e).float(), dim=-1).cpu().numpy()

    def run(self, path):
        sr, crop = self.sf_cfg['sr'], self.sf_cfg['crop']; span = 32 * sr; stride = 16
        buf = None; base = 0; next_start = 0; clips = []; feats = []; sg_imgs = []; sg_feats = []; t = 0
        for ch in stream_frames(path, self.ff):
            buf = ch if buf is None else np.concatenate([buf, ch])
            for f in range(t, t + len(ch)):
                if f % 30 == 0: sg_imgs.append(ch[f - t])
            t += len(ch)
            if len(sg_imgs) >= 32: sg_feats.append(self._sg_batch(sg_imgs)); sg_imgs = []
            while next_start + span <= base + len(buf):
                c = buf[next_start - base: next_start - base + span: sr]
                if crop == 'center':
                    H, W = c.shape[1:3]; m = min(H, W); y0, x0 = (H - m) // 2, (W - m) // 2; c = c[:, y0:y0 + m, x0:x0 + m]
                clips.append(np.ascontiguousarray(c)); next_start += stride
                if len(clips) == 8: feats.append(self._sf_batch(clips)); clips = []
            drop = max(0, next_start - base)  # keep only frames still needed
            if drop: buf = buf[drop:]; base += drop
        if buf is not None and (next_start == 0 or len(feats) == 0 and not clips):  # very short video: one padded clip
            idx = np.clip(np.arange(0, span, sr), 0, len(buf) - 1); c = buf[idx]
            if crop == 'center':
                H, W = c.shape[1:3]; m = min(H, W); y0, x0 = (H - m) // 2, (W - m) // 2; c = c[:, y0:y0 + m, x0:x0 + m]
            clips.append(np.ascontiguousarray(c))
        if clips: feats.append(self._sf_batch(clips))
        if sg_imgs: sg_feats.append(self._sg_batch(sg_imgs))
        return np.concatenate(feats), (np.concatenate(sg_feats) if sg_feats else np.zeros((0, 1)))

    @torch.no_grad()
    def text(self, prompts):
        inp = self.proc(text=prompts, padding='max_length', return_tensors='pt').to(DEV)
        e = self.sg.get_text_features(**inp)
        return torch.nn.functional.normalize(getattr(e, 'pooler_output', e).float(), dim=-1).cpu().numpy()

# ---------------- detection ----------------
class Detector:
    """ActionFormer verb + noun heads. kind='slowfast' (SlowFast-R50 features, stride 16) or 'vmae' (VideoMAE-L, stride 8)."""
    SPEC = {'slowfast': ('epic_slowfast_{}.yaml', 'actionformer_{}.pth.tar', 16, 32), 'vmae': ('vmae_full_{}.yaml', 'actionformer_vmae_{}.pth.tar', 8, 16)}
    def __init__(self, kind='slowfast'):
        AF = os.path.join(HERE, 'third_party', 'actionformer'); sys.path.insert(0, AF); sys.path.insert(0, os.path.join(HERE, 'third_party'))
        from libs.core import load_config
        from libs.modeling import make_meta_arch
        cfg_name, w_name, self.stride, self.nf = self.SPEC[kind]; self.models = {}
        for h in ('verb', 'noun'):
            cfg = load_config(os.path.join(AF, 'configs', cfg_name.format(h)))
            m = make_meta_arch(cfg['model_name'], **cfg['model'])
            ck = os.path.join(ROOT, 'weights', w_name.format(h))
            if os.path.exists(ck):
                sd = torch.load(ck, map_location='cpu', weights_only=False)['state_dict_ema']; m.load_state_dict({k.replace('module.', '', 1): v for k, v in sd.items()})
            self.models[h] = m.eval().to(DEV)
    @torch.no_grad()
    def raw(self, fv, fn, duration):
        """per-anchor top-10 verb/noun probabilities and both boundary sets (verb and noun heads share the anchor grid)"""
        R = {'duration': float(duration)}
        for h, f in (('verb', fv), ('noun', fn)):
            m = self.models[h]
            item = {'video_id': 'x', 'feats': torch.from_numpy(np.ascontiguousarray(f.T.astype(np.float32))), 'segments': None, 'labels': None,
                    'fps': 30.0, 'duration': float(duration), 'feat_stride': self.stride, 'feat_num_frames': self.nf}
            bi, bm = m.preprocessing([item]); x, mk = m.backbone(bi, bm); ff, fm = m.neck(x, mk)
            pts = m.point_generator(ff); cl = m.cls_head(ff, fm); of = m.reg_head(ff, fm)
            mask = torch.cat([z.squeeze(1)[0] for z in fm]); P = torch.cat([z[0].permute(1, 0) for z in cl]).sigmoid() * mask[:, None]
            O = torch.cat([z[0].permute(1, 0) for z in of]); pts = torch.cat(pts)
            seg = torch.stack([pts[:, 0] - O[:, 0] * pts[:, 3], pts[:, 0] + O[:, 1] * pts[:, 3]], 1); keep = mask > 0
            p, idx = P[keep].topk(10, 1)
            R[f'{h}_p'] = p.float().cpu().numpy(); R[f'{h}_i'] = idx.cpu().numpy(); R[f'{h}_seg'] = seg[keep].float().cpu().numpy()
        assert len(R['verb_p']) == len(R['noun_p'])
        return R

    @torch.no_grad()
    def run(self, feats, duration, vid='x'):
        item = {'video_id': vid, 'feats': torch.from_numpy(np.ascontiguousarray(feats.T.astype(np.float32))), 'segments': None, 'labels': None,
                'fps': 30.0, 'duration': float(duration), 'feat_stride': self.stride, 'feat_num_frames': self.nf}
        out = {}
        for h, m in self.models.items():
            try: o = m([item])[0]
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache(); m.to('cpu'); o = m([item])[0]; m.to(DEV)
            out[h] = [(float(s), float(e), int(l), float(c)) for (s, e), l, c in zip(o['segments'].tolist(), o['labels'].tolist(), o['scores'].tolist())]
        return out['verb'], out['noun']

class VmaeFeaturizer:
    """EPIC-finetuned VideoMAE-L verb and noun backbones (OpenTAD release); one shared decode pass."""
    def __init__(self, ff):
        import vmae; self.vm = vmae; self.ff = ff; self.cfg = CFG.get('vmae', {})
        self.models = [vmae.load(os.path.join(ROOT, 'weights', f'vmae_backbone_{h}.pth'), DEV) for h in ('verb', 'noun')]
        self.heads = {}
        for h in ('verb', 'noun'):  # EPIC recognition heads stored in the same checkpoints (used for re-scoring)
            sd = torch.load(os.path.join(ROOT, 'weights', f'vmae_backbone_{h}.pth'), map_location='cpu', weights_only=False); sd = sd.get('state_dict', sd)
            self.heads[h] = (sd['cls_head.fc_cls.weight'].float().numpy(), sd['cls_head.fc_cls.bias'].float().numpy())
    def run(self, path):
        v = self.cfg.get('variant', 'short224_center'); raw = v.endswith('_raw')
        return self.vm.extract_stream(self.models, path, v.replace('_raw', ''), DEV, start=self.cfg.get('start', 0), raw=raw, ff=self.ff)

def fuse(a, b, wa, wb):
    """union of two detectors' (start, end, label, score) lists, scores weighted"""
    return [(s, e, l, c * wa) for s, e, l, c in a] + [(s, e, l, c * wb) for s, e, l, c in b]

# ---------------- label + importance ----------------
LABEL_PROMPTS = {'cooking': 'a first-person video of someone cooking or washing up in a kitchen', 'bike': 'a first-person video of riding a bicycle',
    'bikepolo': 'a first-person video of playing bike polo', 'boxing': 'a first-person video of boxing', 'horse': 'a first-person video of riding a horse',
    'jumping': 'a first-person video of skydiving or bungee jumping', 'longboarding': 'a first-person video of longboarding down a road',
    'motor': 'a first-person video of riding a motorcycle', 'parkour': 'a first-person video of parkour', 'plane': 'a first-person video from an airplane or glider cockpit',
    'rockclimbing': 'a first-person video of rock climbing', 'scuba': 'a first-person video of scuba diving underwater', 'skate': 'a first-person video of skateboarding',
    'ski': 'a first-person video of skiing on snow', 'surfing': 'a first-person video of surfing a wave'}

def label_and_importance(F, sg):
    names = list(LABEL_PROMPTS); T = F.text([LABEL_PROMPTS[n] for n in names])
    if len(sg) == 0: return 'other', 0.0, np.zeros(1)
    logits = sg @ T.T  # seconds x labels
    vid = np.exp(30 * logits.mean(0)); p = vid / vid.sum(); i = int(p.argmax())
    lab = names[i]
    pos = F.text([CFG['importance']['pos'].format(c=LABEL_PROMPTS[lab])])[0]; neg = F.text([CFG['importance']['neg']])[0]
    imp = sg @ pos - sg @ neg
    return lab, float(p[i]), imp

def importance_highlights(imp, dur, budget, seg_len=4.0, smooth=5):
    if len(imp) == 0: return []
    k = np.ones(smooth) / smooth; s = np.convolve(imp, k, mode='same')
    order = np.argsort(-s); taken = np.zeros(len(s), bool); out = []; used = 0.0; L = int(np.ceil(seg_len))
    for c in order:
        a = max(0, min(int(c) - L // 2, len(s) - L)); b = min(len(s), a + L)
        if taken[a:b].any() or used + (b - a) > budget * dur: continue
        taken[a:b] = True; out.append({'start': float(a), 'end': float(min(b, dur))}); used += b - a
    return sorted(out, key=lambda h: h['start'])

def _z(x):
    x = np.asarray(x, float); sd = x.std(); return (x - x.mean()) / sd if sd > 1e-8 else x * 0

def interest_highlights(sg, sf, dur, budget, w=None, L=5, smooth=5):
    """Unsupervised interest model, weights chosen on SumMe (dev): centre prior + SigLIP novelty + SlowFast motion signals."""
    w = w or CFG.get('interest', {}).get('w', {'cen': 2, 'nov': 2, 'chg': 0, 'sfn': 1, 'sfc': 1, 'sfnov': -1})
    T = min(len(sg), int(np.floor(dur)))
    if T < L or len(sf) == 0: return [{'start': 0.0, 'end': float(min(dur, max(L, 4.1)))}]
    t = np.arange(T); g = sg[:T] / (np.linalg.norm(sg[:T], axis=1, keepdims=True) + 1e-8)
    m = g.mean(0); m /= np.linalg.norm(m) + 1e-8
    f = sf[np.clip((t * 30 / 16).astype(int), 0, len(sf) - 1)]; fn = f / (np.linalg.norm(f, axis=1, keepdims=True) + 1e-8)
    fm = fn.mean(0); fm /= np.linalg.norm(fm) + 1e-8
    S = dict(cen=-((t / max(T - 1, 1) - 0.5) ** 2), nov=1 - g @ m, chg=np.r_[0, np.linalg.norm(np.diff(g, axis=0), axis=1)],
             sfn=np.linalg.norm(f, axis=1), sfc=np.r_[0, np.linalg.norm(np.diff(fn, axis=0), axis=1)], sfnov=1 - fn @ fm)
    score = sum(w[k] * _z(S[k]) for k in w)
    s = np.convolve(score, np.ones(smooth) / smooth, mode='same') if smooth > 1 else score
    win = np.convolve(s, np.ones(L), mode='valid'); taken = np.zeros(T, bool); out = []; used = 0
    for i in np.argsort(-win):
        if taken[i:i + L].any() or used + L > budget * dur: continue
        taken[i:i + L] = True; out.append({'start': float(i), 'end': float(min(i + L, dur))}); used += L
    return sorted(out, key=lambda h: h['start'])

# ---------------- main ----------------
VM = DV = None  # VideoMAE branch, set up in __main__ when enabled

def infer_one(path, out_json, F, D, team, cache=None):
    vid = os.path.splitext(os.path.basename(path))[0]
    dur = probe(path, F.ff)[0]
    cf = os.path.join(cache, vid + '.npz') if cache else None
    if cf and os.path.exists(cf):
        z = np.load(cf); feats, sg = z['slowfast'], z['siglip']
    else:
        feats, sg = F.run(path)
        if cf: np.savez(cf, slowfast=feats, siglip=sg)
    lab, lscore, imp = label_and_importance(F, sg)
    acts = []
    if lab == 'cooking':  # kitchen action vocabulary only makes sense for kitchen video
        if VM is not None:  # VideoMAE-L branch (default): EPIC-finetuned verb/noun backbones + ActionFormer heads
            vc = os.path.join(cache, vid + '.vmae.npz') if cache else None
            if vc and os.path.exists(vc): z = np.load(vc); fv, fn = z['verb'].astype(np.float32), z['noun'].astype(np.float32)
            else:
                fv, fn = VM.run(path)
                if vc: np.savez(vc, verb=fv.astype(np.float16), noun=fn.astype(np.float16))
            jc = CFG.get('joint', {})
            acts = joint(DV.raw(fv, fn, dur), **{k: v for k, v in jc.items() if k != 'rescore_beta'})
            acts = rescore(acts, softmax_probs(fv, *VM.heads['verb']), softmax_probs(fn, *VM.heads['noun']), beta=jc.get('rescore_beta', 0.6))
            g = CFG.get('sel_gamma', 1.0)
            hl = select([dict(d, score=d['score'] ** g) for d in acts], dur, mode=CFG.get('sel_mode', 'half'), **CFG['selector'])
        else:  # SlowFast-R50 branch (fallback when VideoMAE weights are absent)
            verbs, nouns = D.run(feats, dur, vid)
            acts = compose(verbs, nouns, **CFG['compose'])
            cand = [dict(d, score=d['score'] ** 2.0) for d in compose(verbs, nouns, **CFG.get('compose_sel', CFG['compose']))]
            hl = select(cand, dur, mode=CFG.get('sel_mode', 'half'), **CFG['selector'])
    else:
        ic = CFG.get('interest', {})
        hl = interest_highlights(sg, feats, dur, CFG['selector']['budget'], ic.get('w'), ic.get('L', 5), ic.get('smooth', 5))
    return write(out_json, team, vid, dur, lab, lscore, acts, hl)

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('video', nargs='?'); ap.add_argument('out', nargs='?')
    ap.add_argument('--list', help='text file of video paths (batch mode)'); ap.add_argument('--outdir', default=os.path.join(ROOT, 'outputs'))
    ap.add_argument('--cache', default=None); ap.add_argument('--team', default=CFG.get('team_id', 'EgoCut')); ap.add_argument('--ffmpeg', default=None)
    a = ap.parse_args()
    ff = a.ffmpeg
    if not ff:  # prefer the system ffmpeg (very new static builds can hit thread limits inside containers)
        import shutil; ff = shutil.which('ffmpeg')
        if not ff:
            try:
                import imageio_ffmpeg; ff = imageio_ffmpeg.get_ffmpeg_exe()
            except Exception: ff = 'ffmpeg'
    F = Featurizer(ff); D = None
    vw = [os.path.join(ROOT, 'weights', f) for f in ('vmae_backbone_verb.pth', 'vmae_backbone_noun.pth', 'actionformer_vmae_verb.pth.tar', 'actionformer_vmae_noun.pth.tar')]
    if CFG.get('vmae', {}).get('enabled') and all(os.path.exists(f) for f in vw): VM = VmaeFeaturizer(ff); DV = Detector('vmae')
    else: D = Detector('slowfast'); print('[egocut] VideoMAE weights missing - using the SlowFast detector branch', flush=True)
    if a.cache: os.makedirs(a.cache, exist_ok=True)
    jobs = [(a.video, a.out)] if a.video else [(p.strip(), os.path.join(a.outdir, os.path.splitext(os.path.basename(p.strip()))[0] + '.json')) for p in open(a.list) if p.strip()]
    os.makedirs(a.outdir, exist_ok=True)
    for i, (v, o) in enumerate(jobs):
        t = time.time()
        try:
            r = infer_one(v, o, F, D, a.team, a.cache)
            print(f'[{i+1}/{len(jobs)}] {os.path.basename(v)} -> {r["label"]["name"]} {len(r["actions"])} actions {len(r["highlights"])} highlights ({time.time()-t:.0f}s)', flush=True)
        except Exception as e:
            import traceback; print(f'[{i+1}/{len(jobs)}] {v} FAILED: {e}', flush=True); traceback.print_exc()

"""Run a pretrained ActionFormer head (verb or noun) on the official EPIC features.
Saves all detections to a pickle AND prints the official mAP in one pass.
Falls back to CPU for any video that does not fit in GPU memory (6 GB laptop GPU)."""
import argparse, os, sys, glob, pickle, time
import numpy as np, torch, torch.nn as nn
AF = os.path.join(os.path.dirname(__file__), '..', 'third_party', 'actionformer')
sys.path.insert(0, os.path.abspath(AF))
from libs.core import load_config
from libs.datasets import make_dataset
from libs.datasets.data_utils import trivial_batch_collator
from libs.modeling import make_meta_arch
from libs.utils import ANETdetection, fix_random_seed

ap = argparse.ArgumentParser()
ap.add_argument('head', choices=['verb', 'noun'])
ap.add_argument('--limit', type=int, default=0, help='only first N videos (smoke test)')
ap.add_argument('--config', default='')
ap.add_argument('--ckpt-dir', default='')
ap.add_argument('--tag', default='val')
a = ap.parse_args()
os.chdir(os.path.abspath(AF))
cfg = load_config(a.config or f'configs/epic_slowfast_{a.head}.yaml')
cfg['loader']['num_workers'] = 0
ckpt = sorted(glob.glob(os.path.join(a.ckpt_dir or f'pretrained/epic_slowfast_{a.head}_reproduce', '*.pth.tar')))[-1]
fix_random_seed(0, include_cuda=True)
ds = make_dataset(cfg['dataset_name'], False, cfg['val_split'], **cfg['dataset'])
if a.limit:
    ds.data_list = ds.data_list[:a.limit]
dl = torch.utils.data.DataLoader(ds, batch_size=1, num_workers=0, collate_fn=trivial_batch_collator, shuffle=False)
dev = 'cuda' if torch.cuda.is_available() else 'cpu'
model = make_meta_arch(cfg['model_name'], **cfg['model'])
sd = torch.load(ckpt, map_location='cpu')['state_dict_ema']
model.load_state_dict({k.replace('module.', '', 1): v for k, v in sd.items()})
model.eval().to(dev)
print('checkpoint', ckpt, '| device', dev, '| videos', len(ds))
res = {k: [] for k in ['video-id', 't-start', 't-end', 'label', 'score']}
t0 = time.time(); n_cpu = 0
for i, vl in enumerate(dl):
    with torch.no_grad():
        try:
            out = model(vl)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache(); n_cpu += 1
            model.to('cpu'); out = model(vl); model.to(dev)
    for o in out:
        n = o['segments'].shape[0]
        if n:
            res['video-id'] += [o['video_id']] * n
            res['t-start'].append(o['segments'][:, 0].cpu()); res['t-end'].append(o['segments'][:, 1].cpu())
            res['label'].append(o['labels'].cpu()); res['score'].append(o['scores'].cpu())
    if i % 10 == 0:
        print(f'{i+1}/{len(ds)} videos  {time.time()-t0:.0f}s  peak GPU {torch.cuda.max_memory_allocated()/2**30:.2f} GB' if dev == 'cuda' else f'{i+1}/{len(ds)}')
for k in ['t-start', 't-end', 'label', 'score']:
    res[k] = torch.cat(res[k]).numpy()
os.makedirs('../../outputs_dev', exist_ok=True)
pkl = os.path.abspath(f'../../outputs_dev/af_{a.head}_{a.tag}.pkl')
pickle.dump(res, open(pkl, 'wb'))
print('saved', pkl, len(res['score']), 'detections | videos run on CPU:', n_cpu)
if not a.limit:
    ev = ANETdetection(ds.json_file, ds.split[0], tiou_thresholds=ds.get_attributes()['tiou_thresholds'])
    ev.evaluate(res, verbose=True)
print('total time', round(time.time() - t0), 's')

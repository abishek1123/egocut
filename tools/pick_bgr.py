"""After the BGR re-check: keep bgr=True only if it beats the RGB result (0.738)."""
import re, json, os, sys
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
txt = open(sys.argv[1], encoding='utf-8', errors='ignore').read()
sc = {}
for tag, c in re.findall(r'(sr1_\w+?_bgr)\s+ours .*?best mean cosine ([\d.]+)', txt): sc.setdefault(tag, []).append(float(c))
avg = {k: sum(v) / len(v) for k, v in sc.items()}; print('BGR check:', {k: round(v, 3) for k, v in avg.items()})
cfgp = os.path.join(ROOT, 'weights', 'egocut_config.json'); cfg = json.load(open(cfgp))
if avg and max(avg.values()) > 0.738:
    best = max(avg, key=avg.get); cfg['slowfast'] = {'sr': 1, 'crop': best.split('_')[1], 'bgr': True}; cfg['extractor_cosine'] = round(avg[best], 3)
else:
    cfg['slowfast'] = {'sr': 1, 'crop': 'full', 'bgr': False}; cfg['extractor_cosine'] = 0.738
json.dump(cfg, open(cfgp, 'w'), indent=1); print('using', cfg['slowfast'], 'cosine', cfg['extractor_cosine'])

"""EgoCut brownie 2 - a summary of any length from the same detections.

  python code/brownie/summarize.py outputs/P01_11.json --seconds 30
  python code/brownie/summarize.py outputs/P01_11.json --fraction 0.05 --video P01_11.mp4 --reel reel_5pct.mp4

An XR device's token budget changes with the task: a 30 s recap, a 60 s story, the full 15 %. EgoCut's coverage
selector re-plans the highlight for any budget from the actions already in the JSON (no model, no GPU, under a
second), always maximising distinct verb-noun actions covered per second. Kitchen videos only: other videos have no
action list to plan from."""
import argparse, json, os, subprocess, sys, tempfile

HERE = os.path.dirname(os.path.abspath(__file__)); CODE = os.path.dirname(HERE)
sys.path.insert(0, CODE)
from selector_fast import select

def mmss(t): return f'{int(t // 60)}:{int(t % 60):02d}'

def plan(d, seconds):
    cfg = json.load(open(os.path.join(CODE, '..', 'weights', 'egocut_config.json')))['selector']
    dur = d['duration_sec']; budget = min(1.0, seconds / dur)
    dets = [dict(start=a['segment'][0], end=a['segment'][1], verb=a['verb'], noun=a['noun'], score=a['score']) for a in d['actions']]
    w = select(dets, dur, budget=budget, min_len=cfg.get('min_len', 4.1), margin=cfg.get('margin', 0.25), mode='half', top=cfg.get('top', 2000), fill=True)
    return [{'segment': [round(h['start'], 2), round(h['end'], 2)]} for h in w]

def covered(d, hl, top=2000):
    """distinct verb-noun pairs among the detections that have >= 50 % of their time inside the highlights"""
    pairs = set()
    for a in sorted(d['actions'], key=lambda a: -a['score'])[:top]:
        if a['score'] < 0.1: continue
        s, e = a['segment']; inside = sum(max(0, min(e, h['segment'][1]) - max(s, h['segment'][0])) for h in hl)
        if inside >= 0.5 * (e - s): pairs.add((a['verb'], a['noun']))
    return pairs

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('pred'); g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument('--seconds', type=float); g.add_argument('--fraction', type=float)
    ap.add_argument('--out'); ap.add_argument('--video'); ap.add_argument('--reel')
    a = ap.parse_args()
    d = json.load(open(a.pred)); dur = d['duration_sec']
    if not d['actions']: sys.exit(f"{d['video_id']}: no kitchen actions detected (label {d['label']['name']}); length control needs an action list.")
    target = a.seconds if a.seconds else a.fraction * dur
    hl = plan(d, target); total = sum(h['segment'][1] - h['segment'][0] for h in hl)
    print(f"{d['video_id']} ({mmss(dur)}): {len(hl)} pieces, {total:.1f} s = {100 * total / dur:.1f} % of the video, "
          f"covering {len(covered(d, hl))} distinct confident actions")
    for h in hl: print(f"  {mmss(h['segment'][0])} - {mmss(h['segment'][1])}")
    out = dict(d, highlights=hl)
    if a.out: json.dump(out, open(a.out, 'w')); print('json:', a.out)
    if a.video and a.reel:
        with tempfile.NamedTemporaryFile('w', suffix='.json', delete=False) as f: json.dump(out, f)
        subprocess.run([sys.executable, os.path.join(CODE, 'make_reel.py'), a.video, f.name, a.reel], check=True)
        os.unlink(f.name)

if __name__ == '__main__':
    main()

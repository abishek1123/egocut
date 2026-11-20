"""Our validation script (the organisers asked teams to write their own). Implements every check in TEAM_README.md
section 5 (updated kit), strictest reading where the wording allows two: highlights must lie inside [0, duration_sec]."""
import json, os, sys, csv, math, argparse
HERE = os.path.dirname(os.path.abspath(__file__))
KIT = os.path.join(HERE, 'data')
man = {r['video_id']: float(r['duration_sec']) for r in csv.DictReader(open(os.path.join(KIT, 'manifest.csv')))}
voc = json.load(open(os.path.join(KIT, 'supported_classes.json')))
def num(x): return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)
def check(path, team=None):
    E, W = [], []
    try: d = json.load(open(path))
    except Exception as e: return [f'unreadable JSON: {e}'], W, None
    for k in ('team_id', 'video_id', 'duration_sec', 'label', 'actions', 'highlights'):
        if k not in d: E.append(f'missing field {k}')
    if E: return E, W, None
    if not isinstance(d['team_id'], str) or not d['team_id']: E.append('team_id must be a non-empty string')
    if team and d['team_id'] != team: E.append('team_id differs from other files')
    vid = d['video_id']
    if os.path.basename(path) != f'{vid}.json': E.append('video_id does not match file name')
    if vid not in man: E.append('video_id not in manifest')
    dur = d['duration_sec']
    if not num(dur) or dur <= 0: E.append('duration_sec must be > 0'); dur = 0
    elif vid in man and abs(dur - man[vid]) > 0.5: E.append(f'duration_sec {dur} not within 0.5 s of manifest {man[vid]}')
    lb = d['label']
    if not isinstance(lb, dict) or not isinstance(lb.get('name'), str) or not num(lb.get('score')) or not 0 <= lb['score'] <= 1: E.append('label must be {name: str, score in [0,1]}')
    elif lb['name'] not in voc['labels']: E.append(f'label {lb["name"]} not in supported_classes.json labels')
    acts = d['actions']
    if not isinstance(acts, list) or len(acts) > 2000: E.append('actions must be a list of at most 2000')
    else:
        for i, a in enumerate(acts):
            sg = a.get('segment') if isinstance(a, dict) else None
            if not (isinstance(sg, list) and len(sg) == 2 and all(num(x) for x in sg) and 0 <= sg[0] < sg[1] <= dur + 0.5): E.append(f'action {i}: bad segment {sg}'); break
            if not isinstance(a.get('verb'), str) or not isinstance(a.get('noun'), str): E.append(f'action {i}: verb/noun must be strings'); break
            if a['verb'] not in voc['verbs'] or a['noun'] not in voc['nouns']: W.append(f'action {i}: verb/noun outside vocabulary'); break
            if not num(a.get('score')) or not 0 <= a['score'] <= 1: E.append(f'action {i}: score must be in [0,1]'); break
    hl = d['highlights']
    if not isinstance(hl, list) or not hl: E.append('highlights must be a non-empty list')
    else:
        segs = [h.get('segment') if isinstance(h, dict) else None for h in hl]
        if not all(isinstance(s, list) and len(s) == 2 and all(num(x) for x in s) and 0 <= s[0] < s[1] <= dur for s in segs): E.append('highlights: each segment needs 0 <= start < end <= duration_sec')
        else:
            if any(segs[i][1] > segs[i + 1][0] for i in range(len(segs) - 1)) or any(segs[i][0] > segs[i + 1][0] for i in range(len(segs) - 1)): E.append('highlights must be sorted and non-overlapping')
            if not any(s[1] - s[0] >= 4.0 for s in segs): E.append('need at least one highlight segment >= 4.0 s')
            if any(s[1] - s[0] < 4.0 for s in segs): W.append('some highlight segments < 4.0 s (ignored in scoring)')
            tot = sum(s[1] - s[0] for s in segs)
            if dur and tot > 0.15 * dur: W.append(f'highlights total {100*tot/dur:.1f}% > 15% budget')
    stamps = [x for a in (acts if isinstance(acts, list) else []) if isinstance(a, dict) for x in a.get('segment', [])] + [x for h in (hl if isinstance(hl, list) else []) if isinstance(h, dict) for x in h.get('segment', [])]
    if any(num(x) and abs(round(x, 2) - x) > 1e-9 for x in stamps): W.append('timestamps with more than 2 decimal places')
    return E, W, d['team_id']
if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--dir'); ap.add_argument('--file'); ap.add_argument('-v', action='store_true'); a = ap.parse_args()
    files = [a.file] if a.file else sorted(os.path.join(a.dir, f) for f in os.listdir(a.dir) if f.endswith('.json'))
    team = None; bad = 0; warn = 0
    for f in files:
        E, W, t = check(f, team); team = team or t
        if E: bad += 1; print('FAIL', os.path.basename(f), '|', '; '.join(E))
        if W: warn += 1
        if W and a.v: print('warn', os.path.basename(f), '|', '; '.join(W))
    if a.dir:
        missing = sorted(set(man) - {os.path.basename(f)[:-5] for f in files})
        if missing: print(f'MISSING {len(missing)} manifest videos, e.g. {missing[:5]}')
        print(f'{len(files)} of {len(man)} manifest files present' + (' (complete)' if not missing else ''))
    print(f'{len(files)} files checked: {len(files) - bad} pass, {bad} fail, {warn} with warnings')
    sys.exit(1 if bad else 0)

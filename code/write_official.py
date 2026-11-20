"""Write one prediction file in the official Track 4 schema (6 top-level fields)."""
import json, os
HERE = os.path.dirname(os.path.abspath(__file__))
VOCAB = json.load(open(os.path.join(HERE, 'data', 'supported_classes.json')))
VERBS, NOUNS, LABELS = VOCAB['verbs'], VOCAB['nouns'], set(VOCAB['labels'])
MIN_SEG, BUDGET = 4.1, 0.15  # 4.1 s: a safety margin so float rounding never drops a segment below the 4.0 s rule

def fix_highlights(hl, dur, anchor=None):
    """Sorted, strictly non-overlapping, every segment >= 4 s, at least one segment (required)."""
    segs = sorted([max(0.0, h['start']), min(dur, h['end'])] for h in hl if h['end'] > h['start'])
    merged = []
    for s, e in segs:
        if merged and s <= merged[-1][1] + 0.01: merged[-1][1] = max(merged[-1][1], e)
        else: merged.append([s, e])
    out = []
    for s, e in merged:
        if e - s < MIN_SEG:  # grow short pieces to 4 s around their centre (ignored by scoring otherwise)
            c = (s + e) / 2; s, e = max(0.0, c - MIN_SEG / 2), min(dur, c + MIN_SEG / 2)
            if e - s < MIN_SEG: s, e = max(0.0, e - MIN_SEG), min(dur, s + MIN_SEG)
        if out and s <= out[-1][1] + 0.01: out[-1][1] = max(out[-1][1], e)
        else: out.append([s, e])
    if not out:  # very short video: one 4 s segment (validity beats the budget)
        c = anchor if anchor is not None else dur / 2
        s = min(max(0.0, c - MIN_SEG / 2), max(0.0, dur - MIN_SEG)); out = [[s, min(dur, s + MIN_SEG)]]
    cap = 0.149 * dur  # budget guard: growing pieces to 4.1 s and outward rounding can overshoot 15 % slightly
    excess = sum(e - s for s, e in out) - cap
    while excess > 1e-6:
        cand = [i for i, (s, e) in enumerate(out) if e - s > MIN_SEG + 1e-6]
        if not cand: break  # very short video: one >= 4 s segment necessarily exceeds 15 %
        i = max(cand, key=lambda j: out[j][1] - out[j][0]); cut = min(excess, out[i][1] - out[i][0] - MIN_SEG)
        out[i][0] += cut / 2; out[i][1] -= cut / 2; excess -= cut
    import math
    return [{'segment': [math.ceil(s * 100) / 100, math.floor(e * 100) / 100] if e - s > MIN_SEG + 0.02 else [math.floor(s * 100) / 100, math.ceil(e * 100) / 100]} for s, e in out]

def write(path, team_id, video_id, duration, label, label_score, actions, highlights):
    assert label in LABELS, label
    acts = sorted(actions, key=lambda a: -a['score'])[:2000]
    acts = [{'segment': [round(max(0.0, a['start']), 2), round(min(duration + 0.5, a['end']), 2)],
             'verb': VERBS[a['verb']] if isinstance(a['verb'], int) else a['verb'],
             'noun': NOUNS[a['noun']] if isinstance(a['noun'], int) else a['noun'],
             'score': round(min(1.0, max(0.0, float(a['score']))), 4)} for a in acts]
    acts = [a for a in acts if a['segment'][1] > a['segment'][0]]
    anchor = (actions[0]['start'] + actions[0]['end']) / 2 if actions else None
    out = {'team_id': team_id, 'video_id': video_id, 'duration_sec': round(float(duration), 2),
           'label': {'name': label, 'score': round(min(1.0, max(0.0, float(label_score))), 4)},
           'actions': acts, 'highlights': fix_highlights(highlights, duration, anchor)}
    with open(path, 'w') as f: json.dump(out, f)
    return out

"""EgoCut brownie 1 - "Ask your memory": question a video's EgoCut output in plain English.

  python code/brownie/ask.py outputs/P01_11.json "when did I wash the plate?"
  python code/brownie/ask.py outputs/P01_11.json --at 2:35            # what was I doing at 2:35?
  python code/brownie/ask.py outputs/P01_11.json --between 1:00 2:00  # what did I do in that minute?
  add  --video P01_11.mp4 --clips out_dir/  to cut each answer from the original footage

Works on any EgoCut JSON, standard library + NumPy only (no GPU, no model). Words in the question are matched to the
98 verb and 301 noun classes through the EPIC-KITCHENS class synonym lists (e.g. dish -> plate, drizzle -> pour),
then the time-stamped actions EgoCut detected are searched, merged and ranked by confidence."""
import argparse, json, os, re, shutil, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__))
SYN = json.load(open(os.path.join(HERE, 'synonyms.json')))
STOP = set('a an the i me my we our you did do does when where what was were is are have had to of in on at for from with and or did it this that how many times'.split())

def mmss(t): return f'{int(t // 60)}:{int(t % 60):02d}'

def secs(s):
    p = [float(x) for x in s.split(':')]
    return p[0] * 60 + p[1] if len(p) == 2 else p[0]

def stems(w):
    """crude English stemming: washed->wash, cutting->cut, placed->place, dishes->dish"""
    out = {w}
    for suf in ('ing', 'ed', 'es', 's'):
        if w.endswith(suf) and len(w) > len(suf) + 2:
            b = w[:-len(suf)]; out |= {b, b + 'e'}
            if len(b) > 2 and b[-1] == b[-2]: out.add(b[:-1])
    if w.endswith('ies'): out.add(w[:-3] + 'y')
    return out

def parse(q):
    """map question words to verb and noun classes; exact class names beat synonyms, longer phrases beat shorter ones"""
    words = [w for w in re.findall(r"[a-z]+", q.lower()) if w not in STOP]
    grams = set()
    for i, w in enumerate(words):
        grams |= stems(w)
        if i + 1 < len(words): grams |= {w + ' ' + s for s in stems(words[i + 1])}
    def match(kind, tiers):
        best = {}
        for key, t in SYN[kind].items():
            for tier, name in tiers:
                m = [g for g in t[name] if g in grams]
                if m: best[key] = max(best.get(key, (0, 0)), (tier, max(len(g) for g in m)))
        return best
    nouns = match('nouns', ((3, 'key'), (2, 'full')))
    noun_words = {g for k in nouns for g in SYN['nouns'][k]['key'] + SYN['nouns'][k]['full'] if g in grams}
    allv = match('verbs', ((3, 'key'), (2, 'full'), (1, 'head')))
    verbs = {k: v for k, v in allv.items() if not ({g for n in ('key', 'full', 'head') for g in SYN['verbs'][k][n] if g in grams} <= noun_words)} or allv
    pick = lambda d: {k for k, v in d.items() if v[0] == max(x[0] for x in d.values())} if d else set()
    return pick(verbs), pick(nouns)

def merge(acts, gap=1.0):
    """join detections of the same verb-noun pair that touch or overlap"""
    out = []
    for a in sorted(acts, key=lambda a: (a['verb'], a['noun'], a['segment'][0])):
        if out and (a['verb'], a['noun']) == (out[-1]['verb'], out[-1]['noun']) and a['segment'][0] <= out[-1]['segment'][1] + gap:
            o = out[-1]; o['segment'][1] = max(o['segment'][1], a['segment'][1]); o['score'] = max(o['score'], a['score'])
        else:
            out.append({'verb': a['verb'], 'noun': a['noun'], 'segment': list(a['segment']), 'score': a['score']})
    return out

def cut(video, s, e, path, text):
    ff = shutil.which('ffmpeg')
    if not ff:
        try:
            import imageio_ffmpeg; ff = imageio_ffmpeg.get_ffmpeg_exe()
        except Exception: sys.exit('ffmpeg not found')
    font = next((f for f in ('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf', 'C:/Windows/Fonts/arialbd.ttf') if os.path.exists(f)), None)
    vf = 'scale=-2:720'
    if font:
        safe = text.replace(':', r'\:').replace("'", '')
        vf += f",drawtext=fontfile='{font.replace(':', chr(92) + ':')}':text='{safe}':x=24:y=24:fontsize=34:fontcolor=white:box=1:boxcolor=black@0.55:boxborderw=10"
    subprocess.run([ff, '-v', 'error', '-y', '-ss', f'{s:.2f}', '-to', f'{e:.2f}', '-i', video, '-vf', vf, '-c:v', 'libx264', '-preset', 'veryfast', '-crf', '23', '-an', path], check=True)

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('pred'); ap.add_argument('question', nargs='?')
    ap.add_argument('--at'); ap.add_argument('--between', nargs=2); ap.add_argument('--top', type=int, default=5)
    ap.add_argument('--min-score', type=float, default=0.05); ap.add_argument('--video'); ap.add_argument('--clips')
    a = ap.parse_args()
    d = json.load(open(a.pred)); acts = [x for x in d['actions'] if x['score'] >= a.min_score and 'other' not in (x['verb'], x['noun'])]
    print(f"{d['video_id']}  ({mmss(d['duration_sec'])}, label: {d['label']['name']}, {len(d['actions'])} actions detected)")
    if not acts:
        print('No kitchen actions were detected in this video, so there is nothing to search.'); return
    if a.at or a.between:
        lo, hi = (secs(a.at) - 1, secs(a.at) + 1) if a.at else (secs(a.between[0]), secs(a.between[1]))
        found = [x for x in merge(acts) if x['segment'][0] < hi and x['segment'][1] > lo]
        found = sorted(found, key=lambda x: -x['score'])[:a.top]
        title = f'At {a.at}' if a.at else f'Between {a.between[0]} and {a.between[1]}'
        print(f'{title}, you were most likely:')
        found.sort(key=lambda x: x['segment'][0])
    else:
        if not a.question: ap.error('give a question, --at or --between')
        vs, ns = parse(a.question)
        if not vs and not ns:
            print('I could not match any action or object in that question. Try words like wash, cut, open, plate, knife, fridge.'); return
        print('Looking for  verb: ' + (', '.join(sorted(vs)) or 'any') + '   object: ' + (', '.join(sorted(ns)) or 'any'))
        found = [x for x in merge(acts) if (not vs or x['verb'] in vs) and (not ns or x['noun'] in ns)]
        found = sorted(found, key=lambda x: -x['score'])[:a.top]
        if not found:
            print('No matching moment found in this video.'); return
        print(f'{len(found)} best moment(s), most confident first:')
    for i, x in enumerate(found, 1):
        s, e = x['segment']
        print(f"  {i}. {mmss(s)} - {mmss(e)}   {x['verb']} {x['noun']}   (confidence {x['score']:.2f})")
        if a.video and a.clips:
            os.makedirs(a.clips, exist_ok=True)
            p = os.path.join(a.clips, f"{d['video_id']}_{i}_{x['verb']}_{x['noun']}.mp4".replace(':', '-'))
            cut(a.video, max(0, s - 1), min(d['duration_sec'], e + 1), p, f"{mmss(s)}  {x['verb']} {x['noun']}")
            print(f'     clip: {p}')

if __name__ == '__main__':
    main()

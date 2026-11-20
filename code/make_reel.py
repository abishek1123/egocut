"""Compile the highlight reel: the selected highlight segments cut from the ORIGINAL footage, in order,
each captioned with its time range and the actions EgoCut detected inside it.
  python code/make_reel.py <video.mp4> <prediction.json> <reel.mp4> [--height 720]"""
import argparse, json, os, shutil, subprocess, tempfile, glob

FONTS = ['/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf', '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',
         'C:/Windows/Fonts/arialbd.ttf', 'C:/Windows/Fonts/arial.ttf', '/System/Library/Fonts/Supplemental/Arial Bold.ttf']

def ffmpeg_exe():
    ff = shutil.which('ffmpeg')
    if ff: return ff
    import imageio_ffmpeg; return imageio_ffmpeg.get_ffmpeg_exe()

def mmss(t): return f'{int(t // 60):02d}:{int(t % 60):02d}'

def caption(pred, s, e, k=3):
    """top-k distinct verb-noun pairs whose segment overlaps [s, e], by score"""
    seen, out = set(), []
    for a in sorted(pred.get('actions', []), key=lambda a: -a.get('score', 0)):
        a0, a1 = a['segment']
        if min(a1, e) - max(a0, s) <= 0.5 * min(a1 - a0, e - s): continue
        pair = f"{a['verb']} {a['noun']}"
        if pair in seen or 'other' in pair: continue
        seen.add(pair); out.append(pair)
        if len(out) == k: break
    return out

def main():
    ap = argparse.ArgumentParser(); ap.add_argument('video'); ap.add_argument('pred'); ap.add_argument('out'); ap.add_argument('--height', type=int, default=720)
    a = ap.parse_args(); pred = json.load(open(a.pred)); ff = ffmpeg_exe()
    font = next((f for f in FONTS if os.path.exists(f)), None)
    tmp = tempfile.mkdtemp(); parts = []
    hl = [h['segment'] for h in pred['highlights']]; total = sum(e - s for s, e in hl)
    head = f"{pred['label']['name'].upper()}  |  {len(hl)} highlights, {total:.0f}s of {pred['duration_sec']:.0f}s ({100 * total / pred['duration_sec']:.0f}%)"
    for i, (s, e) in enumerate(hl):
        acts = caption(pred, s, e)
        lines = [head, f"[{i + 1}/{len(hl)}]  {mmss(s)} - {mmss(e)}"] + ([' · '.join(acts)] if acts else [])
        vf = [f'scale=-2:{a.height}', 'fps=30']
        if font:
            for j, line in enumerate(lines):
                tf = os.path.join(tmp, f'c{i}_{j}.txt'); open(tf, 'w', encoding='utf-8').write(line)
                y = 24 + 44 * j
                vf.append(f"drawtext=fontfile='{font}':textfile='{tf}':expansion=none:x=24:y={y}:fontsize={30 if j < 2 else 34}:fontcolor=white:box=1:boxcolor=black@0.55:boxborderw=8")
        p = os.path.join(tmp, f'p{i:03d}.mp4'); parts.append(p)
        subprocess.run([ff, '-loglevel', 'error', '-y', '-ss', f'{s:.2f}', '-t', f'{e - s:.2f}', '-i', a.video, '-vf', ','.join(vf),
                        '-c:v', 'libx264', '-preset', 'veryfast', '-crf', '23', '-pix_fmt', 'yuv420p',
                        '-c:a', 'aac', '-ar', '48000', '-ac', '2', '-b:a', '128k', '-af', 'aresample=async=1', p], check=True)
    lst = os.path.join(tmp, 'list.txt'); open(lst, 'w').write(''.join(f"file '{p}'\n" for p in parts))
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    subprocess.run([ff, '-loglevel', 'error', '-y', '-f', 'concat', '-safe', '0', '-i', lst, '-c', 'copy', '-movflags', '+faststart', a.out], check=True)
    shutil.rmtree(tmp, ignore_errors=True); print(f'reel: {a.out}  ({len(hl)} segments, {total:.1f}s)')

if __name__ == '__main__':
    main()

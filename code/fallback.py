"""Safety net for run.sh: if the main pipeline failed or wrote an invalid file, write a minimal VALID prediction
(label 'other', no actions, one centred highlight block of ~14.5% of the video, every piece >= 4.1 s).
Standard library only, so it works even if torch/ffmpeg imports are broken."""
import json, os, re, subprocess, sys, shutil

def duration(video):
    for exe in (shutil.which('ffprobe'),):
        if exe:
            try:
                out = subprocess.run([exe, '-v', 'error', '-show_entries', 'format=duration', '-of', 'csv=p=0', video], capture_output=True, text=True, timeout=120).stdout.strip()
                return float(out)
            except Exception: pass
    ff = shutil.which('ffmpeg')
    if not ff:
        try:
            import imageio_ffmpeg; ff = imageio_ffmpeg.get_ffmpeg_exe()
        except Exception: ff = None
    if ff:
        err = subprocess.run([ff, '-i', video], capture_output=True, text=True).stderr
        m = re.search(r'Duration: (\d+):(\d+):([\d.]+)', err)
        if m: return int(m.group(1)) * 3600 + int(m.group(2)) * 60 + float(m.group(3))
    return None

def looks_valid(path, vid):
    try:
        d = json.load(open(path))
        return (d.get('video_id') == vid and d.get('duration_sec', 0) > 0 and isinstance(d.get('actions'), list)
                and isinstance(d.get('label'), dict) and any(h['segment'][1] - h['segment'][0] >= 4.0 for h in d.get('highlights', [])))
    except Exception:
        return False

def main(video, out, team='EgoCut'):
    vid = os.path.splitext(os.path.basename(video))[0]
    if os.path.exists(out) and looks_valid(out, vid): return 0
    here = os.path.dirname(os.path.abspath(__file__))
    try:  # manifest duration is the reference when the video is in it
        import csv
        man = {r['video_id']: float(r['duration_sec']) for r in csv.DictReader(open(os.path.join(here, 'data', 'manifest.csv')))}
    except Exception: man = {}
    try: team = json.load(open(os.path.join(here, '..', 'weights', 'egocut_config.json'))).get('team_id', team)
    except Exception: pass
    dur = man.get(vid) or duration(video) or 60.0
    L = max(4.1, 0.145 * dur); L = min(L, dur)
    s = max(0.0, dur / 2 - L / 2)
    d = {'team_id': team, 'video_id': vid, 'duration_sec': round(dur, 2), 'label': {'name': 'other', 'score': 0.0},
         'actions': [], 'highlights': [{'segment': [round(s, 2), round(min(dur, s + L), 2)]}]}
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True); json.dump(d, open(out, 'w'))
    print(f'[fallback] wrote minimal valid prediction for {vid}', file=sys.stderr); return 0

if __name__ == '__main__':
    sys.exit(main(sys.argv[1], sys.argv[2]))

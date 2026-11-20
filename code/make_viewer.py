"""Interactive timeline viewer for one prediction: video player + highlights + detected actions.
  python code/make_viewer.py <video.mp4> <prediction.json> <viewer.html>
The page references the video by relative path, so keep it next to (or under) the HTML file's folder."""
import json, os, sys, html

PAGE = r"""<!DOCTYPE html><html lang="en"><head><meta charset="utf-8"><title>EgoCut · __VID__</title>
<style>
:root{--bg:#0f1115;--panel:#171a21;--ink:#e8eaf0;--mute:#8b91a1;--hl:#f2b544;--line:#262a33;--acc:#5aa9ff}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:14px/1.4 system-ui,-apple-system,Segoe UI,Roboto,sans-serif}
header{display:flex;gap:18px;align-items:baseline;padding:14px 20px;border-bottom:1px solid var(--line)}
header h1{font-size:18px;margin:0}header .tag{background:var(--panel);border:1px solid var(--line);border-radius:999px;padding:2px 10px;color:var(--mute)}
main{display:grid;grid-template-columns:minmax(0,1fr) 320px;gap:16px;padding:16px 20px}
video{width:100%;background:#000;border-radius:8px;max-height:62vh}
aside{background:var(--panel);border:1px solid var(--line);border-radius:8px;padding:12px;overflow:auto;max-height:62vh}
aside h2{font-size:13px;color:var(--mute);margin:4px 0 8px;font-weight:600;text-transform:uppercase;letter-spacing:.04em}
.act{display:flex;justify-content:space-between;gap:8px;padding:5px 8px;border-radius:6px;margin-bottom:4px;background:#1e222b;cursor:pointer}
.act b{font-weight:600}.act span{color:var(--mute);font-variant-numeric:tabular-nums}
#tl{padding:0 20px 8px}canvas{width:100%;height:150px;display:block;cursor:pointer;border-radius:8px;background:var(--panel);border:1px solid var(--line)}
.legend{display:flex;gap:16px;color:var(--mute);font-size:12px;padding:6px 20px 16px}.sw{display:inline-block;width:10px;height:10px;border-radius:2px;margin-right:6px;vertical-align:-1px}
.btns{display:flex;gap:8px;margin:10px 0 0}button{background:#232834;color:var(--ink);border:1px solid var(--line);border-radius:6px;padding:6px 10px;cursor:pointer}button.on{border-color:var(--hl);color:var(--hl)}
</style></head><body>
<header><h1>EgoCut</h1><span class="tag" id="vid"></span><span class="tag" id="lab"></span><span class="tag" id="stat"></span></header>
<main><div><video id="v" controls preload="metadata" src="__SRC__"></video>
<div class="btns"><button id="play">▶ Play highlights only</button><button id="all">Play full video</button></div></div>
<aside><h2>Actions now</h2><div id="now"></div><h2 style="margin-top:14px">Highlights</h2><div id="hls"></div></aside></main>
<div id="tl"><canvas id="c"></canvas></div>
<div class="legend"><span><i class="sw" style="background:var(--hl)"></i>highlight (selected ≤15%)</span><span><i class="sw" style="background:var(--acc)"></i>detected action (opacity = confidence)</span><span>click the timeline to seek</span></div>
<script>
const P=__PRED__;const v=document.getElementById('v'),c=document.getElementById('c'),x=c.getContext('2d');
const D=P.duration_sec,H=P.highlights.map(h=>h.segment);const A=[...P.actions].sort((a,b)=>b.score-a.score).slice(0,400);
const tot=H.reduce((s,h)=>s+h[1]-h[0],0);const mm=t=>`${String(Math.floor(t/60)).padStart(2,'0')}:${String(Math.floor(t%60)).padStart(2,'0')}`;
document.getElementById('vid').textContent=P.video_id;document.getElementById('lab').textContent=`${P.label.name} (${(100*P.label.score).toFixed(0)}%)`;
document.getElementById('stat').textContent=`${H.length} highlights · ${tot.toFixed(0)}s of ${D.toFixed(0)}s (${(100*tot/D).toFixed(1)}%) · ${P.actions.length} actions`;
const lanes=[];A.forEach(a=>{let l=lanes.findIndex(e=>e<=a.segment[0]);if(l<0){l=lanes.length;lanes.push(0)}lanes[l]=a.segment[1];a.lane=l});
const hls=document.getElementById('hls');H.forEach((h,i)=>{const d=document.createElement('div');d.className='act';
 const acts=[...new Set(A.filter(a=>Math.min(a.segment[1],h[1])-Math.max(a.segment[0],h[0])>0).slice(0,3).map(a=>a.verb+' '+a.noun))];
 d.innerHTML=`<b>${i+1}. ${acts.join(' · ')||'highlight'}</b><span>${mm(h[0])}</span>`;d.onclick=()=>{v.currentTime=h[0];v.play()};hls.appendChild(d)});
function draw(){const W=c.width=c.clientWidth*devicePixelRatio,Hh=c.height=c.clientHeight*devicePixelRatio,s=W/D,top=28*devicePixelRatio;
 x.clearRect(0,0,W,Hh);x.fillStyle='#f2b544';H.forEach(h=>x.fillRect(h[0]*s,4*devicePixelRatio,Math.max(2,(h[1]-h[0])*s),18*devicePixelRatio));
 const nl=Math.max(1,lanes.length),lh=(Hh-top-6)/nl;A.forEach(a=>{x.globalAlpha=.25+.75*a.score;x.fillStyle='#5aa9ff';x.fillRect(a.segment[0]*s,top+a.lane*lh,Math.max(1,(a.segment[1]-a.segment[0])*s),Math.max(1,lh-1))});
 x.globalAlpha=1;x.fillStyle='#fff';x.fillRect(v.currentTime*s-1,0,2*devicePixelRatio,Hh)}
c.onclick=e=>{const r=c.getBoundingClientRect();v.currentTime=(e.clientX-r.left)/r.width*D};
let hlOnly=false;document.getElementById('play').onclick=e=>{hlOnly=!hlOnly;e.target.classList.toggle('on',hlOnly);if(hlOnly){v.currentTime=H[0][0];v.play()}};
document.getElementById('all').onclick=()=>{hlOnly=false;document.getElementById('play').classList.remove('on');v.currentTime=0;v.play()};
v.ontimeupdate=()=>{const t=v.currentTime;if(hlOnly){const cur=H.find(h=>t>=h[0]-.05&&t<h[1]);if(!cur){const nx=H.find(h=>h[0]>t);if(nx)v.currentTime=nx[0];else{v.pause();hlOnly=false}}}
 const now=A.filter(a=>a.segment[0]<=t&&t<=a.segment[1]).slice(0,8);document.getElementById('now').innerHTML=now.map(a=>`<div class="act"><b>${a.verb} ${a.noun}</b><span>${(100*a.score).toFixed(0)}%</span></div>`).join('')||'<div style="color:var(--mute)">—</div>';draw()};
addEventListener('resize',draw);draw();
</script></body></html>"""

def main(video, pred_path, out):
    pred = json.load(open(pred_path)); out_dir = os.path.dirname(os.path.abspath(out))
    src = os.path.relpath(os.path.abspath(video), out_dir).replace(os.sep, '/')
    page = (PAGE.replace('__PRED__', json.dumps(pred).replace('</', '<\\/')).replace('__SRC__', html.escape(src, quote=True))
            .replace('__VID__', html.escape(pred['video_id'])))
    os.makedirs(out_dir, exist_ok=True); open(out, 'w', encoding='utf-8').write(page); print('viewer:', out)

if __name__ == '__main__':
    main(*sys.argv[1:4])

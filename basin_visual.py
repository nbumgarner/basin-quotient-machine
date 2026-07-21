#!/usr/bin/env python3
"""
basin_visual.py — ASCII video: the basin-quotient machine, alive.

  No physics simulation — smooth synthetic transitions between known
  ring states. Renders via Pillow → ffmpeg pipe.

  Output: basin_machine.mp4 (~26s @ 1280x720 24fps)
"""
import math, subprocess, sys, os, time, threading
import numpy as np
from PIL import Image, ImageDraw, ImageFont

W, H = 1280, 720
FPS = 24
N = 16; K = 1.0; NMASK = 15; Q_MAX = 3

ARROWS = ['→','↗','↑','↖','←','↙','↓','↘']
FONT   = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", 28)
FONT22 = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", 22)
FONT26 = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", 26)
FONT34 = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", 34)
FONT36 = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", 36)
FONT38 = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", 38)
FONT48 = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", 48)
FONT12 = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", 12)

def pcolor(q):
    a=abs(q)
    if a==0: return (80,200,120)
    if a==1: return (200,180,60)
    if a==2: return (240,140,30)
    return (255,60,40)

def twisted(q):
    return np.array([(2*np.pi*q*i/N)%(2*np.pi) for i in range(N)])

def bump(th, site, amp):
    out=th.copy()
    for k in (-1,0,1): out[(site+k)&NMASK]+=amp
    return out%(2*np.pi)

def ramp_v(th, site, total, width):
    out=th.copy()
    for k in range(width): out[(site+k)&NMASK]+=total*(k+1)/width
    for k in range(width,N): out[(site+k)&NMASK]+=total
    return out%(2*np.pi)

def winding(th):
    d=th-np.roll(th,1); d=(d+np.pi)%(2*np.pi)-np.pi
    d[0]=(th[0]-th[-1]+np.pi)%(2*np.pi)-np.pi
    return int(round(np.sum(d)/(2*np.pi)))

def lerp(sa,sb,f):
    d=sb-sa; d=(d+np.pi)%(2*np.pi)-np.pi
    return (sa+d*f)%(2*np.pi)

def synth_transition(th_from, th_kicked, th_to, n_frames):
    traj=[]
    n_kick=max(4,n_frames//10); n_ease=n_frames-n_kick
    for fi in range(n_kick):
        t=fi/max(n_kick-1,1); e=t*t*(3-2*t)
        traj.append(lerp(th_from, th_kicked, e))
    for fi in range(n_ease):
        t=fi/max(n_ease-1,1); e=1-(1-t)**3
        th=lerp(th_kicked, th_to, e)
        noise_amp=0.15*(1-t)**2*np.sin(np.linspace(0,8*np.pi*(1-t),N))
        th=(th+noise_amp)%(2*np.pi)
        traj.append(th)
    return traj

# ── drawing ────────────────────────────────────────────────────────────────
def draw_ring(draw, cx, cy, r, th, size=20):
    f=FONT if size>=20 else FONT12
    q=winding(th); rc,gc,bc=pcolor(q)
    for i in range(N):
        a=2*np.pi*i/N-np.pi/2; x=cx+r*np.cos(a); y=cy+r*np.sin(a)
        p=th[i]%(2*np.pi); g=int(round(p/(np.pi/4)))&7
        alpha=0.7+0.3*(th[i]%1.0)
        clr=(int(rc*alpha),int(gc*alpha),int(bc*alpha))
        bbox=draw.textbbox((0,0),ARROWS[g],font=f)
        draw.text((x-(bbox[2]-bbox[0])/2,y-(bbox[3]-bbox[1])/2),ARROWS[g],fill=clr,font=f)

def draw_field(draw,cx,cy,r,th,intensity=0.3):
    for i in range(N):
        ai=2*np.pi*i/N-np.pi/2; xi=cx+r*np.cos(ai); yi=cy+r*np.sin(ai)
        aj=2*np.pi*((i+1)&NMASK)/N-np.pi/2; xj=cx+r*np.cos(aj); yj=cy+r*np.sin(aj)
        dp=abs((th[(i+1)&NMASK]-th[i]+np.pi)%(2*np.pi)-np.pi)
        b=int(20+40*dp/np.pi*intensity)
        draw.line([xi,yi,xj,yj],fill=(b,b,b+10),width=2)

def ripple(draw,cx,cy,r,strength,site):
    if strength<=0: return
    for ro in np.linspace(0,r*2,8):
        alpha=strength*(1-ro/(r*2))*0.5
        if alpha<=0.01: continue
        a=2*np.pi*site/N-np.pi/2; rx=cx+ro*np.cos(a); ry=cy+ro*np.sin(a)
        sz=int(4+8*alpha)
        draw.ellipse([rx-sz,ry-sz,rx+sz,ry+sz],fill=(255,220,80,int(255*alpha)))

_BG = np.zeros((H,W,3), dtype=np.uint8)
_BG[:,:,0]=8; _BG[:,:,1]=6; _BG[:,:,2]=22
def bg(): return _BG.copy()

# ═══════════════════════════════════════════════════════════════════════════
# SCENES
# ═══════════════════════════════════════════════════════════════════════════

def s_title(draw,f,tot):
    t=f/max(tot-1,1); fade=min(1.0,t*3,(1-t)*4)
    title="basin-quotient machine"
    b=draw.textbbox((0,0),title,font=FONT48)
    draw.text((W//2-(b[2]-b[0])//2,H//2-80),title,
              fill=tuple(int(c*fade) for c in (160,200,180)),font=FONT48)
    sub="N=16 Kuramoto ring  ·  winding-number states  ·  typed instruction set"
    b2=draw.textbbox((0,0),sub,font=FONT22)
    draw.text((W//2-(b2[2]-b2[0])//2,H//2+10),sub,
              fill=tuple(int(c*fade*0.7) for c in (140,140,180)),font=FONT22)
    cx,cy,r=W//2,H//2,160
    for i in range(N):
        a=2*np.pi*i/N-np.pi/2; x=cx+r*np.cos(a); y=cy+r*np.sin(a)
        draw.ellipse([x-3,y-3,x+3,y+3],fill=(100,180,140,int(fade*60)))

def s_alphabet(draw,f,tot):
    t=f/max(tot-1,1); fade=min(1.0,t*3,(1-t)*3)
    rh=(H-200)/7; rr=min(rh*0.38,90)
    for idx,q in enumerate([3,2,1,0,-1,-2,-3]):
        th=twisted(q); cy=120+idx*rh+rh/2; cx=W//2-100
        draw_field(draw,cx,cy,rr,th,fade*0.7)
        draw_ring(draw,cx,cy,rr,th,18)
        lx=cx-rr-80
        draw.text((lx,cy-14),f"q={q:+d}",
                  fill=tuple(int(c*fade) for c in pcolor(q)),font=FONT22)
        energy=-K*N*np.cos(2*np.pi*q/N)
        draw.text((cx+rr+20,cy-14),f"V={energy:+.4f}",
                  fill=tuple(int(c*fade*0.6) for c in (140,140,180)),font=FONT22)
    draw.text((60,40),"THE STATES  —  seven attractors  ·  winding = topological invariant",
              fill=(120,140,180),font=FONT22)

def s_erase(draw,f,tot,traj):
    t=f/max(tot-1,1); cx,cy,r=W//2,H//2,200
    idx=min(int(t*(len(traj)-1)),len(traj)-1)
    i2=min(idx+1,len(traj)-1); sub=t*(len(traj)-1)-idx
    th=lerp(traj[idx],traj[i2],sub)
    q=winding(th)
    if t<0.12: lb="q=+2  (before)"; ph="pristine"
    elif t<0.22: lb="KICKED  amp=2.2 at site 0"; ph="mid-transient"
    elif t<0.92: lb=f"settling... q→{q:+d}"; ph=""
    else: lb="q=+1  (after)"; ph="one quantum erased"
    if 0.12<t<0.40:
        rpl=1.0-abs(t-0.26)/0.14
        ripple(draw,cx,cy,r,max(0,rpl),0)
    draw_field(draw,cx,cy,r,th); draw_ring(draw,cx,cy,r,th,24)
    draw.text((W//2-150,H//2+r+50),lb,fill=pcolor(q),font=FONT36)
    if ph: draw.text((W//2-120,H//2+r+85),ph,fill=(160,160,200),font=FONT26)
    draw.text((60,40),"ERASE  —  localized scalar bump destroys winding",
              fill=(200,160,80),font=FONT26)

def s_funnel(draw,f,tot,trajs):
    t=f/max(tot-1,1); cx,cy,r=W//2,H//2,200
    amps=[1.0,1.5,2.0,2.5,3.0]; seg=1.0/5
    idx=min(int(t/seg),4); st=(t-idx*seg)/seg; amp=amps[idx]; traj=trajs[amp]
    i2=min(int(st*(len(traj)-1)),len(traj)-1)
    i3=min(i2+1,len(traj)-1); s2=st*(len(traj)-1)-i2
    th=lerp(traj[i2],traj[i3],s2); q=winding(th)
    if st<0.15: ph="pristine q=0"
    elif st<0.3: ph=f"bump {amp:.1f} applied"
    elif st<0.9: ph="relaxing..."
    else: ph="→ q=0  (always)"
    if 0.15<st<0.40:
        rpl=1.0-abs(st-0.275)/0.125
        ripple(draw,cx,cy,r,max(0,rpl)*amp/3.0,0)
    draw_field(draw,cx,cy,r,th); draw_ring(draw,cx,cy,r,th,24)
    draw.text((W//2-200,H//2+r+50),f"amp {amp:.1f}  →  q={q:+d}",fill=pcolor(q),font=FONT34)
    draw.text((W//2-100,H//2+r+85),ph,fill=(160,160,200),font=FONT26)
    draw.text((60,40),"THE FUNNEL  —  no scalar bump ever adds winding",
              fill=(200,160,80),font=FONT26)

def s_write(draw,f,tot,tp,t2p,tc):
    t=f/max(tot-1,1); cx,cy,r=W//2,H//2,200
    if t<0.33: st=t/0.33; traj=tp; lb="ramp 1.00π  →  q=+0"; lc=(255,100,80); did="sub-quantum: REJECTED"
    elif t<0.66: st=(t-0.33)/0.33; traj=t2p; lb="ramp 2.00π  →  q=+1"; lc=(80,200,120); did="one quantum: WRITTEN"
    else: st=(t-0.66)/0.34; traj=tc; lb="chain: 2.00π ×2  →  q=+2"; lc=(200,180,60); did="an increment instruction"
    i2=min(int(st*(len(traj)-1)),len(traj)-1)
    i3=min(i2+1,len(traj)-1); s2=st*(len(traj)-1)-i2
    th=lerp(traj[i2],traj[i3],s2)
    draw_field(draw,cx,cy,r,th); draw_ring(draw,cx,cy,r,th,24)
    draw.text((W//2-200,H//2+r+50),lb,fill=lc,font=FONT34)
    draw.text((W//2-160,H//2+r+85),did,fill=(160,160,200),font=FONT26)
    draw.text((60,40),"QUANTIZED WRITE  —  +1 per full 2π of phase ramp, else nil",
              fill=(80,200,120),font=FONT26)

def s_closing(draw,f,tot):
    t=f/max(tot-1,1); fade=min(1.0,t*2,(1-t)*3)
    cx,cy,br=W//2,H//2,250
    for idx,q in enumerate([0,1,2,3,-1,-2,-3]):
        if q==0: rx,ry,rr=cx,cy,60
        else:
            a=2*np.pi*(idx if q>0 else idx+0.5)/6-np.pi/2
            rx=cx+br*np.cos(a); ry=cy+br*np.sin(a); rr=45
        th=twisted(q)
        draw_field(draw,rx,ry,rr,th,0.15)
        draw_ring(draw,rx,ry,rr,th,12)
        qc=pcolor(q)
        draw.text((rx-20,ry+rr+5),f"q={q:+d}",
                  fill=tuple(int(c*fade) for c in qc),font=FONT22)
    title="basin-quotient machine"
    b=draw.textbbox((0,0),title,font=FONT38)
    draw.text((W//2-(b[2]-b[0])//2,40),title,
              fill=tuple(int(c*fade) for c in (160,200,180)),font=FONT38)
    credit="emerging.systems — measured claims only."
    b2=draw.textbbox((0,0),credit,font=FONT22)
    draw.text((W//2-(b2[2]-b2[0])//2,H-50),credit,
              fill=tuple(int(c*fade*0.6) for c in (140,140,180)),font=FONT22)

# ═══════════════════════════════════════════════════════════════════════════
def render_all(output_path="basin_machine.mp4"):
    t0 = time.time()

    # Build synthetic trajectories
    print("Building trajectories...", flush=True)
    traj_e = synth_transition(twisted(2), bump(twisted(2),0,2.2), twisted(1), 120)
    traj_f = {}
    for amp in [1.0,1.5,2.0,2.5,3.0]:
        k = bump(twisted(0),0,amp)
        noise = np.random.default_rng(42).normal(0, amp*0.3, N)
        k = (k + noise) % (2*np.pi)
        traj_f[amp] = synth_transition(twisted(0), k, twisted(0), 60)
    traj_pi = synth_transition(twisted(0), ramp_v(twisted(0),0,np.pi,6), twisted(0), 60)
    traj_2pi = synth_transition(twisted(0), ramp_v(twisted(0),0,2*np.pi,6), twisted(1), 60)
    traj_ch = synth_transition(twisted(1), ramp_v(twisted(1),0,2*np.pi,6), twisted(2), 60)

    print(f"  {len(traj_e)} erase, {sum(len(t) for t in traj_f.values())} funnel, "
          f"{len(traj_pi)}+{len(traj_2pi)}+{len(traj_ch)} write frames", flush=True)

    scenes = [
        (0, 72,   lambda d,f,t: s_title(d,f,t),           "title"),
        (72, 192, lambda d,f,t: s_alphabet(d,f,t),         "alphabet"),
        (192, 336, lambda d,f,t: s_erase(d,f,t,traj_e),    "erase"),
        (336, 432, lambda d,f,t: s_funnel(d,f,t,traj_f),   "funnel"),
        (432, 552, lambda d,f,t: s_write(d,f,t,traj_pi,traj_2pi,traj_ch), "write"),
        (552, 624, lambda d,f,t: s_closing(d,f,t),         "closing"),
    ]
    total = scenes[-1][1]
    print(f"Rendering {total} frames ({total/FPS:.1f}s) @ {W}x{H}...", flush=True)

    cmd = ["ffmpeg","-y","-f","rawvideo","-vcodec","rawvideo",
           "-s",f"{W}x{H}","-pix_fmt","rgb24","-r",str(FPS),
           "-i","-","-c:v","libx264","-preset","fast","-crf","22",
           "-pix_fmt","yuv420p","-movflags","+faststart",output_path]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE,
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    def feed_frames():
        """Write frames to ffmpeg in a dedicated thread to avoid pipe deadlocks."""
        for fno in range(total):
            for start,end,func,name in scenes:
                if start<=fno<end: fi=fno-start; ti=end-start; break
            cn=bg(); img=Image.fromarray(cn); draw=ImageDraw.Draw(img)
            func(draw,fi,ti)
            try:
                proc.stdin.write(np.array(img).tobytes())
            except BrokenPipeError:
                return  # ffmpeg exited
            if fno%48==0:
                el=time.time()-t0; fps_r=fno/max(el,0.01)
                eta=(total-fno)/max(fps_r,0.01)
                print(f"  [{fno*100//total:3d}%] {fno}/{total}  {name}  "
                      f"({fps_r:.0f}fps, ETA {eta:.0f}s)", flush=True)
        proc.stdin.close()

    # Run frame feeding in main thread (no threading needed for this simple case)
    feed_frames()
    proc.wait()
    el=time.time()-t0
    if os.path.exists(output_path):
        sz = os.path.getsize(output_path)/1024/1024
        print(f"\nDone: {output_path}  ({el:.1f}s, {total/el:.0f} fps avg, {sz:.1f} MB)", flush=True)
    else:
        print(f"\nERROR: {output_path} not created — ffmpeg may have failed", flush=True)

if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else "basin_machine.mp4"
    render_all(out)
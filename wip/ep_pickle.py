"""토스 편 — 빠른 컷(자막 조각마다 새 비트), 펀치줌·슬램 숫자·흔들림, 강조 자막, 효과음·BGM."""
import sys, os, json, re, subprocess
import numpy as np, cv2, soundfile as sf
from PIL import Image, ImageDraw
sys.path.insert(0, "/home/claude/tts"); sys.path.insert(0, "/home/claude/pb/video")
from lib import *
from pickle_script import LINES, CARDS, SOURCE
for _L in LINES:
    _L["img"] = [(_L["img"], "w")]
import sfx

SRC = "/home/claude/pb/video/flow_pickle"
VO = "/home/claude/tts/out_pickle/fast"
OUT = "/home/claude/pb/video/out/pickle"
os.makedirs(OUT, exist_ok=True)
LEAD, GAP, TAIL = 0.05, 0.16, 0.9
# 문장 뒤 쉼: 전환점·결말 앞은 길게, 실패 몽타주는 짧게
GAPS = {1: 0.22, 2: 0.4, 6: 0.3, 9: 0.4, 12: 0.4, 14: 0.3, 15: 0.48, 16: 0.4}

def norm(s): return re.sub(r"[^가-힣0-9a-zA-Z]", "", s)
def weight(s): return sum(1.6 if c.isdigit() else 1.0 for c in norm(s.replace("[", "").replace("]", "")))

_wm = None
def words(path):
    global _wm
    from faster_whisper import WhisperModel
    if _wm is None: _wm = WhisperModel("/home/claude/tts/models/faster-whisper-small", device="cpu", compute_type="int8")
    a = subprocess.run(["ffmpeg", "-loglevel", "error", "-i", path, "-ac", "1", "-ar", "16000", "-f", "f32le", "-"], capture_output=True).stdout
    segs, _ = _wm.transcribe(np.frombuffer(a, np.float32), language="ko", word_timestamps=True)
    return [(w.word, w.start) for s in segs for w in s.words]

def chunk_times(path, caps):
    if len(caps) <= 1: return [0.0] * len(caps)
    ws = words(path)
    if not ws: return [0.0] * len(caps)
    wl = [max(1, len(norm(w))) for w, _ in ws]; cum = np.cumsum([0] + wl)[:-1] / sum(wl)
    cw = [weight(c) for c in caps]; cf = np.cumsum([0] + cw)[:-1] / sum(cw)
    return [0.0 if f == 0 else max(0.0, ws[int(np.argmin(np.abs(cum - f)))][1] - 0.04) for f in cf]

def build():
    segs, caps, slams, ev, t = [], [], [], [], LEAD
    lines = []
    for L in LINES:
        p = f"{VO}/line{L['id']:02d}.wav"; d = sf.info(p).duration
        ct = chunk_times(p, L["caps"]) if L["caps"] else [0.0]
        lines.append((L, t, d, ct, p)); t += d + GAPS.get(L['id'], GAP)
    total = t - GAPS.get(LINES[-1]['id'], GAP) + TAIL
    for k, (L, st, d, ct, p) in enumerate(lines):
        nxt = lines[k + 1][1] if k + 1 < len(lines) else total
        for j, img in enumerate(L["img"]):
            s0 = (0.0 if k == 0 and j == 0 else st + (ct[j] if j < len(ct) else 0))
            s1 = st + ct[j + 1] if j + 1 < len(ct) and j + 1 < len(L["img"]) else nxt
            segs.append((img[0], img[1], s0, s1))
        for j, c in enumerate(L["caps"]):
            c0 = 0.0 if (k == 0 and j == 0) else st + ct[j]
            c1 = st + ct[j + 1] if j + 1 < len(ct) else st + d + 0.1
            caps.append((c0, c1, c))
            if L.get("slam", {}).get(j): slams.append((st + ct[j], L["slam"][j]))
        pass  # riser 제거 (긴장감 X)
        if "ding" in L: ev.append(("ding", st + ct[L["ding"]]))
    merged = []
    for sgm in segs:  # 같은 장면 연속은 합치고, 2.2초 미만 짧은 컷은 앞 장면에 흡수
        if merged and sgm[:2] == merged[-1][:2]:
            merged[-1] = (merged[-1][0], merged[-1][1], merged[-1][2], sgm[3])
        else:
            merged.append(sgm)
    segs = merged
    for i, (n, f, s0, s1) in enumerate(segs):
        if i == 0: continue
        prev = segs[i - 1][0]
        ev.append(("whoosh" if n != prev else "tick", s0 - (0.12 if n != prev else 0)))
    for s, _ in slams: ev.append(("impact", s))
    return dict(lines=[(L["id"], st, d, p) for L, st, d, ct, p in lines], segs=segs, caps=caps, slams=slams, ev=ev, total=total)

# ---------- 화면 ----------
SC = 1.16
_cache = {}
def assets(name):
    if name not in _cache:
        im = Image.open(f"{SRC}/{name}.jpg").convert("RGB").resize((int(W * SC), int(H * SC)), Image.LANCZOS)
        dep = np.asarray(Image.open(f"{SRC}/{name}_depth.png").resize(im.size, Image.BILINEAR), np.float32) / 255.0
        d = cv2.GaussianBlur(dep, (0, 0), 9)
        h, w = d.shape
        gx, gy = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
        m = d > np.quantile(d, 0.85)  # 가까운 피사체 중심 = 타이트 컷 초점
        fy, fx = np.mean(np.argwhere(m), axis=0)
        fx = np.clip(fx / w, 0.3, 0.7); fy = np.clip(fy / h, 0.32, 0.62)
        _cache[name] = (np.asarray(im), d, gx, gy, fx, fy)
    return _cache[name]

AMP = 30.0
def view(name, framing, x, dirn, punch, shake):
    img, d, gx, gy, fx, fy = assets(name)
    u = np.float32(ease_in_out(x) * 2 - 1)
    ox, oy = np.float32(dirn[0] * AMP * u), np.float32(dirn[1] * AMP * u)
    warped = cv2.remap(img, gx - d * ox, gy - d * oy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    h, w = d.shape
    if framing == "t": z, cx, cy = lerp(1.38, 1.46, x), fx, fy
    else:              z, cx, cy = lerp(1.0, 1.07, x), 0.5, 0.5
    z *= punch
    cw, ch = w / z, h / z
    l = np.clip(cx * w - cw / 2 + shake[0], 0, w - cw); t = np.clip(cy * h - ch / 2 + shake[1], 0, h - ch)
    return cv2.resize(warped[int(t):int(t + ch), int(l):int(l + cw)], (W, H), interpolation=cv2.INTER_LINEAR)

RNG = np.random.default_rng(7)
DUST = [(RNG.uniform(0, W), RNG.uniform(0, H), RNG.uniform(1.5, 4.0), RNG.uniform(-8, 8), RNG.uniform(-22, -6)) for _ in range(36)]
def dust(fr, t):
    ov = fr.copy()
    for x, y, r, vx, vy in DUST:
        cv2.circle(ov, (int((x + vx * t) % W), int((y + vy * t) % H)), int(r), (255, 244, 220), -1, cv2.LINE_AA)
    return cv2.addWeighted(ov, 0.3, fr, 0.7, 0)
ys = np.linspace(0, 1, H)[:, None]
SHADE = (1 - np.clip((ys - 0.5) / 0.5, 0, 1) ** 1.6 * 0.42)[..., None].astype(np.float32)

def cap_layer(text, p):
    parts = parse_marked(text); plain = "".join(s for s, _ in parts)
    size = 80; f = font(size, "black")
    while text_w(plain, f) > W - 110 and size > 50: size -= 4; f = font(size, "black")
    f2 = font(int(size * lerp(0.82, 1.0, ease_back(min(1, p)))), "black")
    layer = Image.new("RGBA", (W, H), (0, 0, 0, 0)); d = ImageDraw.Draw(layer)
    x = W / 2 - text_w(plain, f2) / 2
    for s, hl in parts:
        d.text((x, 1190), s, font=f2, fill=(BATON if hl else WHITE) + (255,), stroke_width=11, stroke_fill=(0, 0, 0, 255))
        x += text_w(s, f2)
    if p < 1: layer.putalpha(layer.split()[3].point(lambda v: int(v * min(1, p * 1.8))))
    return layer

def slam_layer(text, t):
    p = prog(t, 0, 0.22); out = prog(t, 1.0, 0.25)
    if p <= 0 or out >= 1: return None
    sc = lerp(1.9, 1.0, ease_out(p)); f = font(int(230 * sc), "black")
    layer = Image.new("RGBA", (W, H), (0, 0, 0, 0)); d = ImageDraw.Draw(layer)
    d.text((W / 2 - text_w(text, f) / 2, 560 - 115 * sc), text, font=f, fill=BATON + (255,), stroke_width=16, stroke_fill=(0, 0, 0, 255))
    a = min(1, p * 2) * (1 - out)
    layer.putalpha(layer.split()[3].point(lambda v: int(v * a)))
    return layer

LOGO = Image.open("/home/claude/pb/out/B_baton.png").convert("RGBA").resize((240, 240))
_m = Image.new("L", (240, 240), 0); ImageDraw.Draw(_m).ellipse([0, 0, 240, 240], fill=255); LOGO.putalpha(_m)
def end_card(fr, t):
    bg = Image.fromarray(fr).convert("RGBA")
    bg.alpha_composite(Image.new("RGBA", (W, H), (15, 13, 12, int(200 * ease_out(prog(t, 0, 0.35))))))
    p = ease_back(prog(t, 0.1, 0.3))
    if p > 0:
        s = max(1, int(240 * p)); bg.alpha_composite(LOGO.resize((s, s)), (int(W / 2 - s / 2), int(640 - s / 2)))
    for txt, y, sz, d0, col in [("직접 만든 앱을", 860, 76, 0.25, WHITE), ("[소개하고 싶다면]", 960, 76, 0.55, WHITE), ("DM 주세요", 1110, 56, 0.9, WHITE), ("@productbaton", 1200, 40, 1.1, (210, 205, 198))]:
        a = prog(t, d0, 0.2)
        if a > 0:
            parts = parse_marked(txt); plain = "".join(s for s, _ in parts); f = font(sz, "black")
            layer = Image.new("RGBA", bg.size, (0, 0, 0, 0)); dd = ImageDraw.Draw(layer); x = W / 2 - text_w(plain, f) / 2
            for s, hl in parts:
                dd.text((x, y), s, font=f, fill=(BATON if hl else col) + (255,), stroke_width=8, stroke_fill=(0, 0, 0, 255)); x += text_w(s, f)
            layer.putalpha(layer.split()[3].point(lambda v: int(v * a))); bg.alpha_composite(layer)
    return bg

_vc = {}
def vid_frames(name):
    path = f"{SRC}/v{name[1:]}.mp4"
    if not os.path.exists(path): return None
    if name not in _vc:
        cap = cv2.VideoCapture(path); fs = []
        while True:
            ok, f = cap.read()
            if not ok: break
            fs.append(cv2.resize(cv2.cvtColor(f, cv2.COLOR_BGR2RGB), (int(W * 1.08), int(H * 1.08)), interpolation=cv2.INTER_CUBIC))
        _vc[name] = fs
    return _vc[name]

def vview(name, t, dur, punch, shake):
    fs = vid_frames(name); n = len(fs)
    speed = max(0.6, min(1.0, (n / 24.0) / max(dur, 0.1)))
    fr = fs[min(n - 1, int(t * speed * 24))]
    h, w = fr.shape[:2]; z = lerp(1.0, 1.05, ease_in_out(t / dur)) * punch
    cw, ch = W * 1.08 / z / 1.08 * 1.0, H / z
    cw, ch = w / (z * 1.0), h / (z * 1.0)
    cw, ch = min(w, cw), min(h, ch)
    l = np.clip((w - cw) / 2 + shake[0], 0, w - cw); tt = np.clip((h - ch) / 2 + shake[1], 0, h - ch)
    return cv2.resize(fr[int(tt):int(tt + ch), int(l):int(l + cw)], (W, H), interpolation=cv2.INTER_LINEAR)

DIRS = [(1, 0), (0, -1), (-1, 0), (0, 1), (1, -1), (-1, -1)]
def make_seg_fn(i, P):
    name, framing, s0, s1 = P["segs"][i]; dur = s1 - s0
    src = P["segs"][i - 1][0] if name == "END" else name
    dirn = DIRS[i % len(DIRS)]
    def fn(t):
        gt = s0 + t
        punch = 1 + 0.07 * (1 - ease_out(prog(t, 0, 0.2))) if s0 > 0 else 1.0
        shake = (0.0, 0.0)
        for st, _ in P["slams"]:
            if 0 <= gt - st < 0.3:
                k = (1 - (gt - st) / 0.3) * 14; shake = (np.sin(gt * 90) * k, np.cos(gt * 77) * k)
        if name != "END" and vid_frames(name) is not None: fr = vview(name, t, dur, punch, shake)
        else: fr = view(src, framing if name != "END" else "w", t / dur if name != "END" else 1.0, dirn, punch, shake)
        fr = dust(fr, gt)
        fr = (fr.astype(np.float32) * SHADE).astype(np.uint8)
        for st, _ in P["slams"]:  # 슬램 순간 짧은 플래시
            if 0 <= gt - st < 0.08: fr = cv2.addWeighted(fr, 0.6, np.full_like(fr, 255), 0.4, 0)
        if name == "END": return end_card(fr, t)
        img = Image.fromarray(fr).convert("RGBA")
        for st, txt in P["slams"]:
            l = slam_layer(txt, gt - st)
            if l: img.alpha_composite(l)
        cur = [c for c in P["caps"] if c[0] <= gt < c[1]]
        if cur: img.alpha_composite(cap_layer(cur[-1][2], prog(gt, cur[-1][0], 0.15)))
        return img
    return fn

LIB = "/home/claude/pb/video/sfx_lib"
_lc = {}
def lib(name, vol=0.5, start=0.0, dur=None, fade=0.15):
    key = (name, start, dur)
    if key not in _lc:
        cmd = ["ffmpeg", "-loglevel", "error", "-ss", str(start), "-i", f"{LIB}/{name}.mp3"]
        if dur: cmd += ["-t", str(dur)]
        cmd += ["-ac", "1", "-ar", str(sfx.SR), "-f", "f32le", "-"]
        a = np.frombuffer(subprocess.run(cmd, capture_output=True).stdout, np.float32).astype(np.float64)
        a = a / (np.abs(a).max() + 1e-9)
        f = int(fade * sfx.SR)
        if dur and len(a) > f: a[-f:] *= np.linspace(1, 0, f)
        _lc[key] = a
    return _lc[key] * vol

def cap_at(P, text):
    for c0, c1, c in P["caps"]:
        if text in c: return c0
    return None

def mix(P):
    SR = sfx.SR; total = P["total"]; N = int(total * SR)
    voice = np.zeros(N + SR); fxb = np.zeros(N + 3 * SR)
    for _, st, d, p in P["lines"]:
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", p, "-ar", str(SR), "-ac", "1", "/tmp/l.wav"])
        a, _ = sf.read("/tmp/l.wav"); i = int(st * SR); voice[i:i + len(a)] += a
    gen = {"whoosh": lambda: lib("whoosh_1", 0.28), "tick": lambda: lib("click_0", 0.18), "pop": lambda: lib("softpop_1", 0.16),
           "impact": lambda: lib("softpop_0", 0.42), "riser": lambda: sfx.riser(1.25, 0.28), "ding": lambda: lib("notif_0", 0.32)}
    # 장면 맞춤 효과음 (Pixabay 무료 라이선스)
    extra = [("success_0", 0.20, 0.0, 1.6, cap_at(P, "한국 AI 비서")),
             ("warning_0", 0.15, 0.0, 1.0, cap_at(P, "카카오톡 기능을")),
             ("coin_1", 0.22, 0.0, None, cap_at(P, "월 매출 8천만")),
             ("notif_1", 0.24, 0.0, None, cap_at(P, "스레드에서 초대권")),
             ("typing_0", 0.14, 0.5, 2.0, cap_at(P, "KTX 표 예매")),
             ("error_0", 0.18, 0.0, None, cap_at(P, "허락한 적 없는"))]
    for name, vol, stt, dd, at in extra:
        if at is None: continue
        a = lib(name, vol, stt, dd); i = max(0, int(at * SR)); fxb[i:i + len(a)] += a[: len(fxb) - i]
    for k, at in P["ev"]:
        s = gen[k](); i = max(0, int(at * SR)); fxb[i:i + len(s)] += s[: len(fxb) - i]
    m = sfx.lofi(total)
    fade = int(1.2 * SR); m[-fade:] *= np.linspace(1, 0, fade)
    sf.write(f"{OUT}/voice.wav", voice[:N], SR); sf.write(f"{OUT}/fx.wav", fxb[:N], SR); sf.write(f"{OUT}/music.wav", m[:N], SR)
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", f"{OUT}/voice.wav", "-i", f"{OUT}/music.wav", "-i", f"{OUT}/fx.wav", "-filter_complex",
        "[0:a]highpass=f=70,acompressor=threshold=0.1:ratio=3:attack=5:release=80,loudnorm=I=-15:TP=-1.5:LRA=6,aformat=channel_layouts=stereo,asplit=2[v][sc];"
        "[1:a]volume=1.5,aformat=channel_layouts=stereo[m];[m][sc]sidechaincompress=threshold=0.1:ratio=1.3:attack=40:release=400[md];"
        "[2:a]volume=0.9,aformat=channel_layouts=stereo[f];"
        "[v][md][f]amix=inputs=3:duration=first:normalize=0,alimiter=limit=0.95[o]", "-map", "[o]", "-ar", str(SR), f"{OUT}/mix.wav"], check=True)

# ================= 티이유식 레이아웃 =================
GREY = (128, 128, 132); DIM = (70, 70, 74); BG = (14, 14, 16)
MEDIA_H = 1080
_fade = np.clip((np.arange(MEDIA_H) - 640) / 440, 0, 1)[:, None, None].astype(np.float32)
LINE_BY_ID = {L["id"]: L for L in LINES}

def media_frame(fr):
    top = fr[200:200 + MEDIA_H].astype(np.float32)
    top = top * (1 - _fade) + np.array(BG, np.float32) * _fade
    out = np.empty((H, W, 3), np.uint8); out[:] = BG; out[:MEDIA_H] = top.astype(np.uint8)
    return out

def text_parts(d, x, y, txt, f, col, hl=BATON, stroke=0):
    for s, h in parse_marked(txt):
        d.text((x, y), s, font=f, fill=(hl if h else col) + (255,), stroke_width=stroke, stroke_fill=(0, 0, 0, 255)); x += text_w(s, f)

def head_layer(img, head, label, t_in):
    a = ease_out(min(1, t_in / 0.35)); dy = int(40 * (1 - a))
    lay = Image.new("RGBA", (W, H), (0, 0, 0, 0)); d = ImageDraw.Draw(lay)
    if label: d.text((84, 1060 + dy), " ".join(label.upper()), font=font(26, "bold"), fill=GREY + (255,))
    f = font(104, "black"); y = 1108 + dy
    for k, line in enumerate(head):
        size = 104
        plain = line.replace("[", "").replace("]", "")
        while text_w(plain, font(size, "black")) > W - 280: size -= 6
        text_parts(d, 84, y, line, font(size, "black"), WHITE); y += size + 22
    d.line([(84, y + 44), (84 + 120, y + 44)], fill=BATON + (255,), width=6)
    lay.putalpha(lay.split()[3].point(lambda v: int(v * a))); img.alpha_composite(lay)

def cards_layer(img, n, t_in):
    lay = Image.new("RGBA", (W, H), (0, 0, 0, 0)); d = ImageDraw.Draw(lay)
    y0 = 1070
    for k, (yr, title, tag) in enumerate(CARDS):
        y = y0 + k * 118
        d.line([(84, y), (W - 200, y)], fill=(40, 40, 44, 255), width=2)
        if k >= n: 
            d.text((84, y + 24), f"0{k+1}", font=font(30, "bold"), fill=DIM + (255,)); d.text((160, y + 22), "· · ·", font=font(56, "bold"), fill=DIM + (255,)); continue
        a = 1.0 if k < n - 1 else ease_out(min(1, t_in / 0.3))
        cur = (k == n - 1)
        col = WHITE if cur else (170, 170, 174)
        d.text((84, y + 24), f"0{k+1}", font=font(30, "bold"), fill=GREY + (int(255 * a),))
        d.text((160, y + 10), yr, font=font(34, "bold"), fill=(BATON if cur else GREY) + (int(255 * a),))
        d.text((160, y + 50), title, font=font(54, "black"), fill=col + (int(255 * a),))
        tw = text_w(tag, font(34, "bold")); d.text((W - 200 - tw, y + 64), tag, font=font(34, "bold"), fill=(BATON if cur else GREY) + (int(255 * a),))
    d.line([(84, y0 + 3 * 118), (W - 200, y0 + 3 * 118)], fill=(40, 40, 44, 255), width=2)
    img.alpha_composite(lay)

def karaoke(img, text, p):
    plain = text.replace("[", "").replace("]", "")
    size = 60; f = font(size, "black")
    while text_w(plain, f) > W - 260 and size > 40: size -= 4; f = font(size, "black")
    lay = Image.new("RGBA", (W, H), (0, 0, 0, 0)); d = ImageDraw.Draw(lay)
    n = len(plain); k = int(round(n * min(1, p * 1.12)))
    x = W / 2 - text_w(plain, f) / 2; y = 1470
    d.text((x, y), plain, font=f, fill=(95, 95, 100, 255))
    d.text((x, y), plain[:k], font=f, fill=WHITE + (255,))
    img.alpha_composite(lay)

def chrome(img):
    d = ImageDraw.Draw(img)
    d.text((W - 84 - text_w(SOURCE, font(22, "bold")), 1040), SOURCE, font=font(22, "bold"), fill=(105, 105, 110, 255))

def make_seg_fn(i, P):
    name, framing, s0, s1 = P["segs"][i]; dur = s1 - s0
    src = P["segs"][i - 1][0] if name == "END" else name
    dirn = DIRS[i % len(DIRS)]
    lines = P["lines"]
    def fn(t):
        gt = s0 + t
        punch = 1 + 0.05 * (1 - ease_out(prog(t, 0, 0.25))) if s0 > 0 else 1.0
        if name != "END" and vid_frames(name) is not None: fr = vview(name, t, dur, punch, (0, 0))
        else: fr = view(src, "w", t / dur if name != "END" else 1.0, dirn, punch, (0, 0))
        fr = dust(fr, gt)
        if name == "END": return end_card(fr, t)
        img = Image.fromarray(media_frame(fr)).convert("RGBA")
        # 현재 라인
        cur = None
        for k, (lid, st, d, p) in enumerate(lines):
            nxt = lines[k + 1][1] if k + 1 < len(lines) else P["total"]
            if st - 0.05 <= gt < nxt: cur = (LINE_BY_ID[lid], st)
        if cur:
            L, st = cur
            # 헤드라인: 이 라인에 head가 없고 카드도 없으면 직전 head 유지
            if L.get("cards"): cards_layer(img, L["cards"], gt - st)
            else:
                hl = None
                for lid, st2, d2, p2 in lines:
                    LL = LINE_BY_ID[lid]
                    if st2 - 0.05 <= gt and LL.get("head"): hl = (LL, st2)
                    if st2 - 0.05 <= gt and LL.get("cards"): hl = None
                if hl: head_layer(img, hl[0]["head"], hl[0].get("label"), gt - hl[1])
        capc = [c for c in P["caps"] if c[0] <= gt < c[1]]
        if capc: c0, c1, txt = capc[-1]; karaoke(img, txt, (gt - c0) / max(0.3, c1 - c0))
        chrome(img)
        return img
    return fn

if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "plan":
        P = build(); json.dump(P, open(f"{OUT}/plan.json", "w"), ensure_ascii=False, indent=1)
        print("TOTAL", round(P["total"], 2), "segs", len(P["segs"]))
        for s in P["segs"]: print(s[0], s[1], round(s[2], 2), round(s[3], 2))
    else:
        P = json.load(open(f"{OUT}/plan.json"))
        if mode == "stills":
            for k, i in enumerate(map(int, sys.argv[2:])):
                n, f, s0, s1 = P["segs"][i]
                make_seg_fn(i, P)((s1 - s0) * 0.5).convert("RGB").save(f"{OUT}/still_{k}.jpg", quality=85)
        elif mode == "render":
            a, b = int(sys.argv[2]), int(sys.argv[3])
            for i in range(a, min(b, len(P["segs"]))):
                n, f, s0, s1 = P["segs"][i]; render(f"{OUT}/seg{i:02d}.mp4", s1 - s0, make_seg_fn(i, P))
            print("rendered", a, b)
        elif mode == "mix":
            mix(P); print("mixed")

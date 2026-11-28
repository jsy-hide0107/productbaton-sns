"""직접 합성한 BGM(업비트) + 효과음 — 저작권 걱정 없음."""
import numpy as np
SR = 44100
rng = np.random.default_rng(3)

def t_(d): return np.arange(int(d * SR)) / SR
def env(d, a=0.005, r=None, k=None):
    t = t_(d); e = np.minimum(1, t / a)
    if k: e *= np.exp(-t * k)
    return e
def lp(x, a):  # 1-pole lowpass
    y = np.empty_like(x); s = 0.0
    for i, v in enumerate(x): s += a * (v - s); y[i] = s
    return y
def lpf(x, a): # vectorized-ish lowpass via cumulative filter
    from scipy.signal import lfilter
    return lfilter([a], [1, a - 1], x)

def whoosh(d=0.35, up=True):
    n = rng.standard_normal(int(d * SR))
    from scipy.signal import butter, sosfilt
    out = np.zeros_like(n); seg = 8; L = len(n) // seg
    for i in range(seg):
        f = 600 + 3500 * (i / seg if up else 1 - i / seg)
        sos = butter(2, [f * 0.6, f * 1.4], btype="band", fs=SR, output="sos")
        out[i * L:(i + 1) * L] = sosfilt(sos, n[i * L:(i + 1) * L])
    e = np.sin(np.linspace(0, np.pi, len(out))) ** 1.5
    return out * e * 0.9
def pop(v=0.5):
    t = t_(0.09); f = 500 + 1100 * np.exp(-t * 45)
    return np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t * 40) * v
def tick(v=0.35):
    t = t_(0.03); return np.sin(2 * np.pi * 2600 * t) * np.exp(-t * 180) * v
def impact(v=1.0):
    t = t_(0.9); f = 70 * np.exp(-t * 3) + 38
    boom = np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t * 4.5)
    n = rng.standard_normal(len(t)) * np.exp(-t * 18) * 0.35
    return (boom + n) * v
def riser(d=1.3, v=0.6):
    n = rng.standard_normal(int(d * SR))
    from scipy.signal import butter, sosfilt
    out = np.zeros_like(n); seg = 16; L = len(n) // seg
    for i in range(seg):
        f = 300 + 5000 * (i / seg) ** 2
        sos = butter(2, [f * 0.7, f * 1.3], btype="band", fs=SR, output="sos")
        out[i * L:(i + 1) * L] = sosfilt(sos, n[i * L:(i + 1) * L])
    e = np.linspace(0, 1, len(out)) ** 2
    tone = np.sin(2 * np.pi * np.cumsum(np.linspace(200, 900, len(out))) / SR) * 0.25
    return (out + tone) * e * v
def ding(v=0.5):
    t = t_(1.2)
    return sum(np.sin(2 * np.pi * f * t) * a for f, a in [(1318.5, 1), (2637, .4), (3951, .15)]) * np.exp(-t * 4) * v * 0.6

def note(n): return 440 * 2 ** ((n - 69) / 12)
def music(total, bpm=102):
    from scipy.signal import butter, sosfilt
    buf = np.zeros(int((total + 2) * SR)); beat = 60 / bpm; bar = 4 * beat
    def add(sig, at):
        i = int(at * SR); j = min(len(buf), i + len(sig))
        if i < len(buf): buf[i:j] += sig[:j - i]
    kick = lambda: (np.sin(2 * np.pi * np.cumsum(55 + 90 * np.exp(-t_(0.3) * 30)) / SR) * np.exp(-t_(0.3) * 9))
    clap = lambda: sosfilt(butter(2, [900, 3000], btype="band", fs=SR, output="sos"), rng.standard_normal(int(0.18 * SR))) * np.exp(-t_(0.18) * 25) * 1.2
    hat = lambda v: sosfilt(butter(2, 7000, btype="high", fs=SR, output="sos"), rng.standard_normal(int(0.05 * SR))) * np.exp(-t_(0.05) * 80) * v
    def pluck(f, d, v):
        t = t_(d); s = 2 * ((f * t) % 1) - 1
        s = sosfilt(butter(2, 2800, fs=SR, output="sos"), s)
        return s * np.exp(-t * 6) * np.minimum(1, t / 0.004) * v
    def bass(f, d, v):
        t = t_(d); return (np.sin(2 * np.pi * f * t) + 0.3 * np.sin(4 * np.pi * f * t)) * np.minimum(1, t / 0.01) * np.exp(-t * 2.5) * v
    prog = [[57, 60, 64], [53, 57, 60], [48, 52, 55], [55, 59, 62]]  # Am F C G
    roots = [45, 41, 36, 43]
    t, i = 0.0, 0
    while t < total + 1:
        ch = prog[i % 4]
        for b in range(4):
            add(kick() * 0.9, t + b * beat)
            if b in (1, 3): add(clap() * 0.5, t + b * beat)
            for s in range(4): add(hat(0.12 if s % 2 else 0.06), t + b * beat + s * beat / 4)
            add(bass(note(roots[i % 4]), beat * 0.9, 0.4), t + b * beat)
        for k in range(8):  # 8분음 아르페지오
            add(pluck(note(ch[k % 3] + 12), beat * 0.6, 0.45), t + k * beat / 2)
        for n_ in ch:  # 부드러운 패드(폰 스피커에서도 들리는 중음역)
            tt = t_(bar); pad = (2 * ((note(n_) * tt) % 1) - 1)
            pad = sosfilt(butter(2, [400, 2200], btype="band", fs=SR, output="sos"), pad) * np.minimum(1, tt / 0.3) * np.minimum(1, (bar - tt) / 0.3) * 0.09
            add(pad, t)
        t += bar; i += 1
    m = buf[: int(total * SR)]
    return m / (np.max(np.abs(m)) + 1e-9)

def calm(total, bpm=72, seed=3):
    """긴장감 없는 lo-fi 패드: Fmaj7-Em7-Dm7-Cmaj7, 소프트 EP + 서브 + 약한 셰이커 + 바이닐 노이즈."""
    import numpy as np
    from scipy.signal import butter, lfilter
    rng = np.random.default_rng(seed)
    N = int(total * SR); out = np.zeros(N)
    beat = 60 / bpm; bar = beat * 4
    chords = [[53, 57, 60, 64], [52, 55, 59, 62], [50, 53, 57, 60], [48, 52, 55, 59]]
    hz = lambda m: 440 * 2 ** ((m - 69) / 12)
    t_bar = np.arange(int(bar * SR)) / SR
    env = (1 - np.exp(-t_bar / 0.08)) * np.exp(-t_bar / 2.8)
    k = 0; t0 = 0.0
    while t0 < total:
        ch = chords[k % 4]; seg = np.zeros(len(t_bar))
        for m in ch:
            f = hz(m + 12)
            seg += (np.sin(2 * np.pi * f * t_bar) + 0.25 * np.sin(4 * np.pi * f * t_bar) * np.exp(-t_bar / 0.6)) * env * 0.11
        seg += np.sin(2 * np.pi * hz(ch[0] - 12) * t_bar) * (1 - np.exp(-t_bar / 0.05)) * np.exp(-t_bar / 3.5) * 0.16
        i = int(t0 * SR); n = min(len(seg), N - i); out[i:i + n] += seg[:n]
        # 셰이커: 8분음표, 아주 작게
        for j in range(8):
            s = int((t0 + j * beat / 2) * SR)
            if s < N:
                L = int(0.06 * SR); nz = rng.standard_normal(L) * np.exp(-np.arange(L) / (0.015 * SR)) * (0.018 if j % 2 else 0.01)
                out[s:s + L] += nz[: max(0, N - s)]
        k += 1; t0 += bar
    b, a = butter(2, 3200 / (SR / 2), "low"); out = lfilter(b, a, out)
    vinyl = rng.standard_normal(N) * 0.004; b2, a2 = butter(2, [1500 / (SR / 2), 6000 / (SR / 2)], "band"); out += lfilter(b2, a2, vinyl)
    fade = int(1.0 * SR); out[:fade] *= np.linspace(0, 1, fade)
    return out / (np.abs(out).max() + 1e-9) * 0.6

def lofi(total, bpm=82, seed=11):
    """나레이션용 로파이: 82BPM 스윙, 부드러운 킥·림·셰이커, 9th 코드 EP, 베이스, 드문 멜로디, 바이닐.
    신나지 않게: 킥은 1·3박만, 스네어 대신 작은 림샷, 고음 컷."""
    import numpy as np
    from scipy.signal import butter, lfilter
    rng = np.random.default_rng(seed)
    N = int(total * SR); out = np.zeros(N)
    beat = 60 / bpm; bar = 4 * beat; sw = beat / 2 * 0.58  # 스윙 8분
    hz = lambda m: 440 * 2 ** ((m - 69) / 12)
    # Dm9 - G13 - Cmaj9 - Am9 (ii-V-I-vi), 따뜻하고 결론 없는 순환
    prog = [[50, 53, 57, 60, 64], [43, 53, 57, 59, 64], [48, 52, 55, 59, 62], [45, 52, 55, 60, 64]]
    bass = [38, 43, 36, 33]
    def add(sig, t):
        i = int(t * SR)
        if i >= N: return
        n = min(len(sig), N - i); out[i:i + n] += sig[:n]
    def ep(m, dur, vel):
        t = np.arange(int(dur * SR)) / SR; f = hz(m)
        e = (1 - np.exp(-t / 0.012)) * np.exp(-t / 1.6)
        trem = 1 + 0.06 * np.sin(2 * np.pi * 4.5 * t)
        return (np.sin(2 * np.pi * f * t) + 0.18 * np.sin(4 * np.pi * f * t) * np.exp(-t / 0.3) + 0.06 * np.sin(6 * np.pi * f * t) * np.exp(-t / 0.15)) * e * trem * vel
    def kick(v):
        t = np.arange(int(0.35 * SR)) / SR; f = 50 + 70 * np.exp(-t / 0.03)
        return np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t / 0.12) * v
    def rim(v):
        L = int(0.08 * SR); t = np.arange(L) / SR
        return (np.sin(2 * np.pi * 1700 * t) * 0.5 + rng.standard_normal(L) * 0.5) * np.exp(-t / 0.012) * v
    def hat(v):
        L = int(0.05 * SR); return rng.standard_normal(L) * np.exp(-np.arange(L) / (0.008 * SR)) * v
    mel = [None, (74, 0.5), None, None, (72, 1.0), None, None, None, (69, 0.5), None, (71, 0.5), None, None, None, None, None]
    k = 0; t0 = 0.0
    while t0 < total:
        ch = prog[k % 4]
        for j, m in enumerate(ch):  # 살짝 흩뿌린 코드
            add(ep(m + 12, bar * 0.98, 0.075 - j * 0.006), t0 + j * 0.012 + 0.02 * rng.random())
        tb = np.arange(int(bar * SR)) / SR
        add(np.sin(2 * np.pi * hz(bass[k % 4]) * tb) * (1 - np.exp(-tb / 0.02)) * np.exp(-tb / 2.2) * 0.22, t0)
        add(np.sin(2 * np.pi * hz(bass[k % 4] + 7) * tb[: int(beat * SR)]) * np.exp(-tb[: int(beat * SR)] / 0.4) * 0.12, t0 + 2.5 * beat)
        for b in range(4):
            tb0 = t0 + b * beat
            if b in (0, 2): add(kick(0.30 if b == 0 else 0.22), tb0)
            if b in (1, 3): add(rim(0.07), tb0)
            add(hat(0.035), tb0); add(hat(0.022), tb0 + sw)
        if k % 2 == 1 and k > 1:  # 2마디마다 아주 드문 멜로디
            for s, nv in enumerate(mel[(k // 2 % 2) * 8:(k // 2 % 2) * 8 + 8]):
                if nv: add(ep(nv[0] + 12, nv[1] * beat * 2, 0.045), t0 + s * beat / 2)
        k += 1; t0 += bar
    b, a = butter(2, 4200 / (SR / 2), "low"); out = lfilter(b, a, out)
    hp_b, hp_a = butter(1, 40 / (SR / 2), "high"); out = lfilter(hp_b, hp_a, out)
    crack = np.zeros(N); idx = rng.integers(0, N, int(total * 6)); crack[idx] = rng.standard_normal(len(idx)) * 0.05
    b2, a2 = butter(2, [1200 / (SR / 2), 7000 / (SR / 2)], "band")
    out += lfilter(b2, a2, crack + rng.standard_normal(N) * 0.003)
    f = int(1.2 * SR); out[:f] *= np.linspace(0, 1, f)
    return out / (np.abs(out).max() + 1e-9) * 0.6


def lofi_drive(total, bpm=92, seed=23):
    """텐션 있는 로파이: 92BPM, 마이너 진행(Am9-Fmaj9-Dm9-E7sus), 킥 싱코페이션, 2·4 스네어,
    16분 하이햇, 8분 베이스, 펌핑감. 시끄럽지 않게 고음은 6.5kHz에서 컷."""
    import numpy as np
    from scipy.signal import butter, lfilter
    rng = np.random.default_rng(seed)
    N = int(total * SR); out = np.zeros(N); drums = np.zeros(N)
    beat = 60 / bpm; bar = 4 * beat; s16 = beat / 4; sw = 0.54
    hz = lambda m: 440 * 2 ** ((m - 69) / 12)
    prog = [[57, 60, 64, 67, 71], [53, 57, 60, 64, 67], [50, 53, 57, 60, 64], [52, 57, 59, 62, 64]]
    bass = [45, 41, 38, 40]
    def add(buf, sig, t):
        i = int(t * SR)
        if i >= N or i < 0: return
        n = min(len(sig), N - i); buf[i:i + n] += sig[:n]
    def ep(m, dur, vel):
        t = np.arange(int(dur * SR)) / SR; f = hz(m)
        e = (1 - np.exp(-t / 0.01)) * np.exp(-t / 1.2)
        return (np.sin(2 * np.pi * f * t) + 0.22 * np.sin(4 * np.pi * f * t) * np.exp(-t / 0.25)) * e * vel
    def kick(v):
        t = np.arange(int(0.3 * SR)) / SR; f = 48 + 90 * np.exp(-t / 0.025)
        return np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t / 0.11) * v
    def snare(v):
        L = int(0.18 * SR); t = np.arange(L) / SR
        body = np.sin(2 * np.pi * 190 * t) * np.exp(-t / 0.04)
        nz = rng.standard_normal(L) * np.exp(-t / 0.06)
        b, a = butter(2, [1500 / (SR / 2), 6000 / (SR / 2)], "band")
        return (0.5 * body + lfilter(b, a, nz) * 1.4) * v
    def hat(v, open_=False):
        L = int((0.12 if open_ else 0.04) * SR)
        b, a = butter(2, 7000 / (SR / 2), "high")
        return lfilter(b, a, rng.standard_normal(L)) * np.exp(-np.arange(L) / ((0.04 if open_ else 0.007) * SR)) * v
    def bassn(m, dur, v):
        t = np.arange(int(dur * SR)) / SR; f = hz(m)
        return (np.sin(2 * np.pi * f * t) + 0.3 * np.sin(4 * np.pi * f * t)) * (1 - np.exp(-t / 0.006)) * np.exp(-t / 0.35) * v
    mel = [(76, 1), None, (74, 1), (72, 2), None, None, (71, 1), (72, 1)]
    k = 0; t0 = 0.0; pump = np.ones(N)
    while t0 < total:
        ch = prog[k % 4]
        for j, m in enumerate(ch):
            add(out, ep(m + 12, bar * 0.95, 0.07 - j * 0.005), t0 + j * 0.01)
        root = bass[k % 4]
        for e8, off in enumerate([0, 0, 12, 0, 7, 0, 12, 10]):
            tt = t0 + e8 * beat / 2 + (beat / 2 * (sw - 0.5) * 2 if e8 % 2 else 0)
            add(out, bassn(root + off - 12 if off < 12 else root, beat / 2 * 0.9, 0.2 if e8 % 2 == 0 else 0.13), tt)
        intro = k < 2
        for b in range(4):
            tb0 = t0 + b * beat
            if not intro:
                if b in (0, 2): add(drums, kick(0.42), tb0)
                if b == 1: add(drums, kick(0.26), tb0 + beat * 0.5 + beat / 2 * (sw - 0.5) * 2)
                if b in (1, 3): add(drums, snare(0.16), tb0)
            for s in range(4):
                ts = tb0 + s * s16 + (s16 * (sw - 0.5) * 2 if s % 2 else 0)
                acc = 0.05 if s == 0 else (0.035 if s == 2 else 0.022)
                add(drums, hat(acc, open_=(b == 3 and s == 2)), ts)
            if not intro:  # 킥에 맞춘 사이드체인 펌핑
                i0 = int(tb0 * SR); L = min(int(beat * SR), N - i0)
                if L > 0: pump[i0:i0 + L] = np.minimum(pump[i0:i0 + L], 0.72 + 0.28 * (1 - np.exp(-np.arange(L) / (0.09 * SR))))
        if k >= 4 and k % 4 in (1, 3):
            for s, nv in enumerate(mel):
                if nv: add(out, ep(nv[0] + 12, nv[1] * beat / 2 * 1.6, 0.05), t0 + s * beat / 2)
        k += 1; t0 += bar
    out *= pump
    b, a = butter(2, 6500 / (SR / 2), "low"); out = lfilter(b, a, out + drums)
    hp_b, hp_a = butter(1, 35 / (SR / 2), "high"); out = lfilter(hp_b, hp_a, out)
    crack = np.zeros(N); idx = rng.integers(0, N, int(total * 5)); crack[idx] = rng.standard_normal(len(idx)) * 0.04
    b2, a2 = butter(2, [1200 / (SR / 2), 7000 / (SR / 2)], "band")
    out += lfilter(b2, a2, crack + rng.standard_normal(N) * 0.0025)
    f = int(0.8 * SR); out[:f] *= np.linspace(0, 1, f)
    return out / (np.abs(out).max() + 1e-9) * 0.6

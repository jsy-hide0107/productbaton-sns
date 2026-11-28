"""문장 안 가변 속도: 기본은 빠르게, 핵심어·발음 뭉개지는 단어는 느리게.
원본 TTS → tidy(끝 여운 유지) → whisper 단어 타임스탬프로 핵심어 구간 찾기 → 구간별 atempo → 짧은 크로스페이드로 이어붙임."""
import re, subprocess, numpy as np, soundfile as sf
from faster_whisper import WhisperModel
_src = open("/home/claude/tts/speedmap_toss.py").read()
exec("def env_db" + _src.split("def env_db")[1].split("for i, sp in SPEED.items()")[0])
import pickle_script as S
SRC = "/home/claude/tts/out_pickle"; DST = f"{SRC}/fast"
BASE = {1:1.48, 2:1.50, 3:1.46, 4:1.48, 5:1.55, 6:1.48, 7:1.55, 8:1.52, 9:1.50, 10:1.45, 11:1.48, 12:1.50,
        13:1.38, 14:1.48, 15:1.40, 16:1.38, 17:1.42}
SLOW = 1.18
KEYS = {1:["챗지피티", "피클이라는"], 2:["하루 만에"], 3:["의대를", "박채근"], 4:["열 시간씩", "줌 수업"],
        5:["화면엔", "에이아이 아바타"], 6:["팔 개월 만에 월 매출 팔천만 원", "와이콤비네이터"], 7:["초대권으로"], 8:["등본을", "전화까지"],
        9:["생산성 일 위에"], 10:["대신 로그인해서"], 11:["사전 협의", "법적 문제"], 12:["유나이티드항공은"],
        13:["누가 열어줄"], 14:["공식 제휴를 하나씩 늘리는", "카카오는 자체 에이아이 비서"], 15:["저는 그 문이 열렸으면"], 16:["의대생이", "인공지능 비서"],
        17:["디엠 주세요"]}
m = WhisperModel("/home/claude/tts/models/faster-whisper-small", device="cpu", compute_type="int8")
norm = lambda s: re.sub(r"[^가-힣a-zA-Z0-9]", "", s)
def spans(text, keys):
    t = norm(text); out = []
    for k in keys:
        i = t.find(norm(k))
        if i >= 0: out.append((i / len(t), (i + len(norm(k))) / len(t)))
    return out
for L in S.LINES:
    i = L["id"]; a, sr = sf.read(f"{SRC}/line{i:02d}.wav"); a = a if a.ndim == 1 else a.mean(1)
    a = tidy(a, sr); dur = len(a) / sr
    a16 = subprocess.run(["ffmpeg", "-loglevel", "error", "-f", "f32le", "-ar", str(sr), "-ac", "1", "-i", "-", "-ar", "16000", "-f", "f32le", "-"],
                         input=a.astype(np.float32).tobytes(), capture_output=True).stdout
    segs, _ = m.transcribe(np.frombuffer(a16, np.float32), language="ko", word_timestamps=True)
    words = [(w.start, w.end, norm(w.word)) for s in segs for w in s.words if norm(w.word)]
    tot = sum(len(w[2]) for w in words) or 1
    cum = []; c = 0
    for w in words: cum.append((c / tot, (c + len(w[2])) / tot, w[0], w[1])); c += len(w[2])
    def t_at(frac, side):  # 글자 비율 → 시간
        for f0, f1, s0, s1 in cum:
            if f0 <= frac <= f1 + 1e-9:
                r = (frac - f0) / max(f1 - f0, 1e-9); return s0 + r * (s1 - s0)
        return dur if side else 0.0
    sl = []
    for f0, f1 in spans(L["tts"], KEYS.get(i, [])):
        s0 = max(0, t_at(f0, 0) - 0.06); s1 = min(dur, t_at(f1, 1) + 0.08); sl.append((s0, s1))
    sl.sort(); cuts = []; last = 0.0
    for s0, s1 in sl:
        if s0 > last: cuts.append((last, s0, BASE[i]))
        cuts.append((max(s0, last), s1, SLOW)); last = s1
    if last < dur: cuts.append((last, dur, BASE[i]))
    parts = []
    for s0, s1, sp in cuts:
        seg = a[int(s0 * sr):int(s1 * sr)]
        if len(seg) < sr * 0.03: continue
        sf.write("/tmp/seg.wav", seg, sr)
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", "/tmp/seg.wav", "-af", f"atempo={sp}", "/tmp/seg2.wav"], check=True)
        parts.append(sf.read("/tmp/seg2.wav")[0])
    xf = int(0.012 * sr); y = parts[0]
    for p in parts[1:]:
        if len(y) > xf and len(p) > xf:
            r = np.linspace(0, 1, xf); y = np.concatenate([y[:-xf], y[-xf:] * (1 - r) + p[:xf] * r, p[xf:]])
        else: y = np.concatenate([y, p])
    y = np.concatenate([y, np.zeros(int(0.04 * sr))])
    sf.write(f"{DST}/line{i:02d}.wav", y, sr)
    print(i, round(dur, 2), "->", round(len(y) / sr, 2), [(round(a, 2), round(b, 2)) for a, b in sl], flush=True)

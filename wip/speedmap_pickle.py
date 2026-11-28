"""문장별 가변 속도 (원본 TTS 대비 배율). 긴장 구간은 빠르게, 전환·결말은 여유 있게."""
import sys, subprocess, numpy as np, soundfile as sf
SPEED = {1:1.40, 2:1.42, 3:1.40, 4:1.42, 5:1.50, 6:1.42, 7:1.50, 8:1.42, 9:1.45, 10:1.36, 11:1.40, 12:1.45, 13:1.22, 14:1.40, 15:1.28, 16:1.25, 17:1.38}
SRC = "/home/claude/tts/out_pickle"; DST = f"{SRC}/fast"
def env_db(a, sr, win=0.02):
    n = int(win * sr); k = len(a) // n
    return 20 * np.log10(np.sqrt((a[:k*n].reshape(k, n) ** 2).mean(1)) + 1e-9), n
def tidy(a, sr):
    db, n = env_db(a, sr); peak = db.max()
    on = np.where(db > peak - 35)[0]
    s = max(0, (on[0] - 2) * n)                      # 앞 무음 제거 (40ms 여유)
    tail = np.where(db > peak - 50)[0][-1]           # 끝 여운은 -50dB까지 살림
    e = min(len(a), (tail + 5) * n)                  # + 100ms
    a = a[s:e].copy()
    # 문장 안 긴 쉼(>0.32s)만 0.22s로 줄임
    db, n = env_db(a, sr); quiet = db < peak - 38; out = []; i = 0; k = len(quiet)
    while i < k:
        j = i
        while j < k and quiet[j] == quiet[i]: j += 1
        seg = a[i*n:j*n]
        if quiet[i] and (j - i) * n / sr > 0.32 and i > 0 and j < k:
            keep = int(0.11 * sr); seg = np.concatenate([seg[:keep], seg[-keep:]])
        out.append(seg); i = j
    out.append(a[k*n:]); a = np.concatenate(out)
    f = int(0.06 * sr); a[-f:] *= np.linspace(1, 0, f) ** 2   # 자연스러운 페이드아웃
    return a
for i, sp in SPEED.items():
    a, sr = sf.read(f"{SRC}/line{i:02d}.wav"); a = a if a.ndim == 1 else a.mean(1)
    sf.write("/tmp/t.wav", tidy(a, sr), sr)
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", "/tmp/t.wav", "-af",
        f"atempo={sp},apad=pad_dur=0.04",
        f"{DST}/line{i:02d}.wav"], check=True)
    print(i, sp, round(len(a)/sr, 2), "->", round(sf.info(f"{DST}/line{i:02d}.wav").duration, 2))

"""Audio I/O and processing helpers (ffmpeg for decoding/encoding, numpy for the rest)."""
from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path

import librosa
import numpy as np
import pyloudnorm as pyln
import soundfile as sf


def require_ffmpeg() -> None:
    if not shutil.which("ffmpeg"):
        raise SystemExit("ffmpeg not found. Install it with:  brew install ffmpeg")


def decode(path: Path, sr: int, highpass_hz: int | None = None, filters: str | None = None) -> np.ndarray:
    """Decode any audio/video file ffmpeg understands to mono float32 at `sr`, optionally
    through a high-pass and/or an ffmpeg filter chain (e.g. "afftdn=nr=12")."""
    require_ffmpeg()
    cmd = ["ffmpeg", "-v", "error", "-nostdin", "-i", str(path), "-vn", "-ac", "1", "-ar", str(sr)]
    chain = [f for f in (f"highpass=f={highpass_hz}" if highpass_hz else None, filters) if f]
    if chain:
        cmd += ["-af", ",".join(chain)]
    cmd += ["-f", "f32le", "-"]
    proc = subprocess.run(cmd, capture_output=True)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg could not read {path}:\n{proc.stderr.decode(errors='replace')[-800:]}")
    return np.frombuffer(proc.stdout, dtype=np.float32).copy()


def analyze(wav: np.ndarray, sr: int) -> dict:
    """Rough quality stats for a voice recording."""
    if len(wav) == 0:
        return {"seconds": 0.0, "peak_dbfs": -120.0, "clipped_pct": 0.0, "snr_db": 0.0}
    rms = librosa.feature.rms(y=wav, frame_length=int(0.03 * sr), hop_length=int(0.015 * sr))[0]
    db = 20 * np.log10(rms + 1e-9)
    peak = float(np.max(np.abs(wav)))
    return {
        "seconds": round(len(wav) / sr, 2),
        "peak_dbfs": round(20 * np.log10(peak + 1e-9), 1),
        "clipped_pct": round(100 * float(np.mean(np.abs(wav) >= 0.999)), 3),
        # loud speech frames vs. the quietest frames (room noise between words)
        "snr_db": round(float(np.percentile(db, 95) - np.percentile(db, 10)), 1),
    }


def condense_pauses(wav: np.ndarray, sr: int, top_db: float = 40, max_gap: float = 0.35,
                    edge: float = 0.1, return_map: bool = False):
    """Trim leading/trailing silence and shorten long pauses, so the model's limited
    conditioning window (first ~10-15 s) is filled with actual speech.
    With `return_map`, also return [(start in the output, start in `wav`, length)] in samples."""
    intervals = librosa.effects.split(wav, top_db=top_db, frame_length=1024, hop_length=256)
    if len(intervals) == 0:
        return (wav[:0], []) if return_map else wav[:0]
    half_gap, edge_n = int(max_gap * sr) // 2, int(edge * sr)
    spans, prev_end = [], None  # (start, end) in wav, in output order
    for start, end in intervals:
        if prev_end is None:
            spans.append((max(0, start - edge_n), start))
        elif start - prev_end > 2 * half_gap:
            spans += [(prev_end, prev_end + half_gap), (start - half_gap, start)]
        else:
            spans.append((prev_end, start))
        spans.append((start, end))
        prev_end = end
    spans.append((prev_end, min(len(wav), prev_end + edge_n)))
    out = np.concatenate([wav[a:b] for a, b in spans])
    if not return_map:
        return out
    seg_map, pos = [], 0
    for a, b in spans:
        seg_map.append((pos, a, b - a))
        pos += b - a
    return out, seg_map


def to_source_time(t: float, seg_map: list, sr: int) -> float:
    """Map a time in condensed audio back to the original recording (seconds)."""
    n = int(t * sr)
    for out_start, src_start, length in seg_map:
        if out_start <= n < out_start + length:
            return (src_start + n - out_start) / sr
    return (seg_map[-1][1] + seg_map[-1][2]) / sr if seg_map else t


# --------------------------------------------------------------------------- reference windows

REF_WINDOW = 15.0  # Turbo's speech prompt uses the first 15 s of the reference (the decoder, 10 s)


def window_features(speech: np.ndarray, sr: int, noise_floor_db: float, window: float = REF_WINDOW,
                    hop: float = 2.5) -> list[dict]:
    """Stats for each `window`-second stretch of condensed speech: start (s), snr_db (loud frames
    over the recording's noise floor), rate (onsets per second, a speech-rate proxy) and
    pitch_var (spread of the pitch in semitones, from librosa's pyin)."""
    y = librosa.resample(speech.astype(np.float32), orig_sr=sr, target_sr=16000)
    hop_len, fps = 512, 16000 / 512
    f0, voiced, _ = librosa.pyin(y, fmin=65, fmax=400, sr=16000, frame_length=2048, hop_length=hop_len)
    db = 20 * np.log10(librosa.feature.rms(y=y, frame_length=1024, hop_length=hop_len)[0] + 1e-9)
    onsets = librosa.onset.onset_detect(y=y, sr=16000, hop_length=hop_len, units="time")
    feats, t, total = [], 0.0, len(y) / 16000
    while t + window <= total + 1e-6:
        a, b = int(t * fps), int((t + window) * fps)
        f = f0[a:b][voiced[a:b]]
        semis = 12 * np.log2(f / np.median(f)) if len(f) > 20 else np.zeros(1)
        feats.append({"start": round(t, 2),
                      "snr_db": round(float(np.percentile(db[a:b], 90) - noise_floor_db), 1),
                      "rate": round(float(np.sum((onsets >= t) & (onsets < t + window)) / window), 2),
                      "pitch_var": round(float(np.std(semis)), 2)})
        t += hop
    return feats


def _z(values) -> np.ndarray:
    v = np.asarray(values, dtype=float)
    return (v - v.mean()) / (v.std() + 1e-9)


def rank_windows(feats: list[dict], similarity: list[float]) -> list[float]:
    """Score each window (higher is better): clean (SNR), a typical pace for this speaker rather
    than the slow, careful start most recordings have, lively pitch, and sounding like the rest
    of the recording (speaker similarity). Each term is a z-score, capped so no single one wins."""
    snr = np.clip(_z([f["snr_db"] for f in feats]), -3, 1.5)
    pace = -np.abs(_z([f["rate"] for f in feats]))
    pitch = np.clip(_z([f["pitch_var"] for f in feats]), -3, 1.5)
    sim = np.clip(_z(similarity), -3, 1.5)
    return [round(float(x), 2) for x in snr + pace + pitch + sim]


def best_window_first(speech: np.ndarray, sr: int, start: float, window: float, total: float) -> np.ndarray:
    """The chosen window, then the speech that follows it, then what came before, up to `total`
    seconds, joined with 20 ms crossfades."""
    a, b = int(start * sr), int((start + window) * sr)
    pieces = [p for p in (speech[a:b], speech[b:], speech[:a]) if len(p)]
    xf = int(0.02 * sr)
    out = pieces[0]
    for p in pieces[1:]:
        if len(out) >= total * sr:
            break
        if len(p) > 2 * xf and len(out) > 2 * xf:
            ramp = np.linspace(0, 1, xf, dtype=np.float32)
            out = np.concatenate([out[:-xf], out[-xf:] * ramp[::-1] + p[:xf] * ramp, p[xf:]])
        else:
            out = np.concatenate([out, p])
    return out[: int(total * sr)]


def gain_to_lufs(wav: np.ndarray, sr: int, target: float) -> np.ndarray:
    try:
        loudness = pyln.Meter(sr).integrated_loudness(wav)
    except ValueError:  # shorter than one 400 ms block
        return wav
    if not np.isfinite(loudness):
        return wav
    return (wav * 10 ** ((target - loudness) / 20)).astype(np.float32)


def trim_silence(wav: np.ndarray, sr: int, top_db: float = 45,
                 pad_start: float = 0.03, pad_end: float = 0.08) -> np.ndarray:
    if len(wav) == 0:
        return wav
    _, (start, end) = librosa.effects.trim(wav, top_db=top_db, frame_length=512, hop_length=128)
    return wav[max(0, start - int(pad_start * sr)):min(len(wav), end + int(pad_end * sr))]


def fade(wav: np.ndarray, sr: int, ms: float = 8) -> np.ndarray:
    n = min(int(sr * ms / 1000), len(wav) // 2)
    if n <= 0:
        return wav
    wav = wav.copy()
    ramp = np.linspace(0.0, 1.0, n, dtype=np.float32)
    wav[:n] *= ramp
    wav[-n:] *= ramp[::-1]
    return wav


def silence(seconds: float, sr: int) -> np.ndarray:
    return np.zeros(int(round(seconds * sr)), dtype=np.float32)


def _ffmetadata(chapters: list[dict], duration: float) -> str:
    def esc(v: str) -> str:
        return "".join("\\" + ch if ch in "=;#\\\n" else ch for ch in v)
    lines = [";FFMETADATA1"]
    for i, ch in enumerate(chapters):
        end = chapters[i + 1]["start"] if i + 1 < len(chapters) else duration
        lines += ["", "[CHAPTER]", "TIMEBASE=1/1000", f"START={int(ch['start'] * 1000)}",
                  f"END={int(end * 1000)}", f"title={esc(ch['title'])}"]
    return "\n".join(lines) + "\n"


def chapter_count(path: Path) -> int | None:
    """Chapters ffprobe finds in a file (None when ffprobe isn't installed)."""
    if not shutil.which("ffprobe"):
        return None
    proc = subprocess.run(["ffprobe", "-v", "error", "-show_chapters", "-of", "csv=p=0", str(path)],
                          capture_output=True, text=True)
    return len([ln for ln in proc.stdout.splitlines() if ln.strip()]) if proc.returncode == 0 else None


def export(wav: np.ndarray, sr: int, out: Path, *, speed: float = 1.0, metadata: dict | None = None,
           bitrate: str = "128k", chapters: list[dict] | None = None) -> None:
    """Write .mp3 / .m4a / .wav with a peak limiter (-1 dBFS) and optional tempo change.
    `chapters` ([{start, title}], in output seconds) go in as ID3 CHAP frames / MP4 chapters."""
    require_ffmpeg()
    out.parent.mkdir(parents=True, exist_ok=True)
    ext = out.suffix.lower()
    codec = {".mp3": ["-c:a", "libmp3lame", "-b:a", bitrate, "-id3v2_version", "3"],
             ".m4a": ["-c:a", "aac", "-b:a", bitrate],
             ".wav": ["-c:a", "pcm_s16le"]}.get(ext)
    if codec is None:
        raise ValueError(f"Unsupported output format '{ext}' (use .mp3, .m4a or .wav)")
    filters = []
    if abs(speed - 1.0) > 1e-3:
        filters.append(f"atempo={speed:.3f}")
    filters.append("alimiter=limit=0.89:level=0")
    meta_args = []
    for key, value in (metadata or {}).items():
        if value:
            meta_args += ["-metadata", f"{key}={value}"]
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td) / "mix.wav"
        sf.write(tmp, wav.astype(np.float32), sr, subtype="FLOAT")  # float keeps overs for the limiter
        inputs = ["-i", str(tmp)]
        if chapters and ext in (".mp3", ".m4a"):
            meta = Path(td) / "chapters.txt"
            meta.write_text(_ffmetadata(chapters, len(wav) / sr / speed), encoding="utf-8")
            inputs += ["-i", str(meta), "-map", "0:a", "-map_chapters", "1"]
        cmd = ["ffmpeg", "-v", "error", "-nostdin", "-y", *inputs, "-af", ",".join(filters),
               "-ac", "1", "-ar", "44100", *codec, *meta_args, str(out)]
        proc = subprocess.run(cmd, capture_output=True)
        if proc.returncode != 0:
            raise RuntimeError(f"ffmpeg failed writing {out}:\n{proc.stderr.decode(errors='replace')[-800:]}")


# --------------------------------------------------------------------------- mastering a recording

def speech_segments(wav: np.ndarray, sr: int) -> list[tuple[float, float]]:
    """Where someone is speaking, as (start, end) seconds, from silero-vad (MIT; the model ships
    inside the package, so nothing is downloaded). Runs on the CPU."""
    import torch
    from silero_vad import get_speech_timestamps, load_silero_vad

    y = librosa.resample(wav.astype(np.float32), orig_sr=sr, target_sr=16000) if sr != 16000 else wav
    with torch.inference_mode():
        stamps = get_speech_timestamps(torch.from_numpy(np.ascontiguousarray(y)), load_silero_vad(),
                                       sampling_rate=16000, min_silence_duration_ms=150,
                                       speech_pad_ms=80, return_seconds=True)
    return [(float(t["start"]), float(t["end"])) for t in stamps]


def shorten_pauses(wav: np.ndarray, sr: int, segments: list[tuple[float, float]], max_pause: float = 0.8,
                   keep: float = 0.6, edge: float = 0.3) -> tuple[np.ndarray, dict]:
    """Cut every silence between speech segments that is longer than `max_pause` down to `keep`
    seconds (half kept after the speech before it, half before the speech after it), and the
    silence at the very start and end down to `edge`. Cuts get 10 ms fades.
    Returns (audio, stats) with stats: shortened (count), removed_sec, kept spans [(start, end)]."""
    if not segments:
        return wav, {"shortened": 0, "removed_sec": 0.0, "spans": [(0.0, len(wav) / sr)]}
    total = len(wav) / sr
    spans = [(max(0.0, segments[0][0] - edge), None)]
    shortened = 0
    for (_, end), (start, _) in zip(segments, segments[1:]):
        if start - end > max_pause:
            spans[-1] = (spans[-1][0], end + keep / 2)
            spans.append((start - keep / 2, None))
            shortened += 1
    spans[-1] = (spans[-1][0], min(total, segments[-1][1] + edge))
    fade = int(0.01 * sr)
    ramp = np.linspace(0.0, 1.0, fade, dtype=np.float32)
    pieces = []
    for a, b in spans:
        piece = wav[int(a * sr):int(b * sr)].copy()
        if len(piece) > 2 * fade:
            piece[:fade] *= ramp
            piece[-fade:] *= ramp[::-1]
        pieces.append(piece)
    out = np.concatenate(pieces)
    return out, {"shortened": shortened, "removed_sec": round(total - len(out) / sr, 1), "spans": spans}


def denoise_deepfilter(wav: np.ndarray, sr: int) -> np.ndarray:
    """DeepFilterNet3 noise suppression (MIT/Apache-2.0). Optional: install it yourself with
    `uv pip install deepfilternet` (it pins old versions of packaging and numpy, so it isn't a
    project dependency). Its small model downloads on first use."""
    try:
        import torch
        from df.enhance import enhance, init_df
    except ImportError as e:
        raise RuntimeError("--denoise deepfilter needs DeepFilterNet: uv pip install deepfilternet") from e
    import contextlib
    import io
    import warnings

    with warnings.catch_warnings(), contextlib.redirect_stderr(io.StringIO()):
        warnings.simplefilter("ignore")
        model, state, _ = init_df(log_level="ERROR")
        y = librosa.resample(wav, orig_sr=sr, target_sr=state.sr()) if sr != state.sr() else wav
        out = enhance(model, state, torch.from_numpy(np.ascontiguousarray(y, dtype=np.float32)).unsqueeze(0))
    out = out.squeeze(0).numpy().astype(np.float32)
    return librosa.resample(out, orig_sr=state.sr(), target_sr=sr) if sr != state.sr() else out

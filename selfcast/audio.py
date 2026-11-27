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


def decode(path: Path, sr: int, highpass_hz: int | None = None) -> np.ndarray:
    """Decode any audio/video file ffmpeg understands to mono float32 at `sr`."""
    require_ffmpeg()
    cmd = ["ffmpeg", "-v", "error", "-nostdin", "-i", str(path), "-vn", "-ac", "1", "-ar", str(sr)]
    if highpass_hz:
        cmd += ["-af", f"highpass=f={highpass_hz}"]
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
                    edge: float = 0.1) -> np.ndarray:
    """Trim leading/trailing silence and shorten long pauses, so the model's limited
    conditioning window (first ~6-15 s) is filled with actual speech."""
    intervals = librosa.effects.split(wav, top_db=top_db, frame_length=1024, hop_length=256)
    if len(intervals) == 0:
        return wav[:0]
    half_gap, edge_n = int(max_gap * sr) // 2, int(edge * sr)
    pieces, prev_end = [], None
    for start, end in intervals:
        if prev_end is None:
            pieces.append(wav[max(0, start - edge_n):start])
        elif start - prev_end > 2 * half_gap:
            pieces += [wav[prev_end:prev_end + half_gap], wav[start - half_gap:start]]
        else:
            pieces.append(wav[prev_end:start])
        pieces.append(wav[start:end])
        prev_end = end
    pieces.append(wav[prev_end:prev_end + edge_n])
    return np.concatenate(pieces)


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


def export(wav: np.ndarray, sr: int, out: Path, *, speed: float = 1.0, metadata: dict | None = None,
           bitrate: str = "128k") -> None:
    """Write .mp3 / .m4a / .wav with a peak limiter (-1 dBFS) and optional tempo change."""
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
        cmd = ["ffmpeg", "-v", "error", "-nostdin", "-y", "-i", str(tmp), "-af", ",".join(filters),
               "-ac", "1", "-ar", "44100", *codec, *meta_args, str(out)]
        proc = subprocess.run(cmd, capture_output=True)
        if proc.returncode != 0:
            raise RuntimeError(f"ffmpeg failed writing {out}:\n{proc.stderr.decode(errors='replace')[-800:]}")

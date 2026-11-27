"""Command-line entry point:  uv run selfcast <clone|podcast|voices> ...

Every command ends with one machine-readable line:  RESULT {...json...}
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import random
import shutil
import sys
import time
from pathlib import Path

from . import config as C
from .text import CHUNK_PAUSE, Pause, Speech, first_heading, parse_script, slugify, speech_bounds, word_count


def log(msg: str = "") -> None:
    print(msg, flush=True)


def die(msg: str) -> None:
    print(f"ERROR: {msg}", file=sys.stderr, flush=True)
    sys.exit(2)


def result(payload: dict) -> None:
    print("RESULT " + json.dumps(payload, ensure_ascii=False), flush=True)


def clock(sec: float) -> str:
    m, s = divmod(max(sec, 0.0), 60)
    return f"{int(m):02d}:{s:04.1f}"


def human(sec: float) -> str:
    m, s = divmod(int(round(sec)), 60)
    return f"{m}m {s:02d}s" if m else f"{s}s"


def heavy_engine_warning(engine: str) -> None:
    if engine in C.HEAVY_ENGINES:
        log(f"WARNING: the '{engine}' engine peaked at ~18 GB of memory in testing (turbo needs ~4-5 GB) "
            "and will be slow. Close other apps first.")


# --------------------------------------------------------------------------- voices

def cmd_voices(args: argparse.Namespace) -> None:
    from .voices import default_voice_name, list_voices

    voices, default = list_voices(), default_voice_name()
    if not voices:
        log("No voice profiles yet. Create one with:  uv run selfcast clone --recording <file> --name me")
    for v in voices:
        m = v.meta()
        log(f"{'*' if v.name == default else ' '} {v.name:<16} {m.get('reference_seconds', 0):5.1f}s of speech  "
            f"engines ready: {', '.join(v.prepared_engines()) or '-'}  created {m.get('created', '?')[:10]}")
    result({"voices": [{"name": v.name, "default": v.name == default,
                        "engines": v.prepared_engines(), **v.meta()} for v in voices],
            "default": default})


# --------------------------------------------------------------------------- clone

def cmd_clone(args: argparse.Namespace) -> None:
    import soundfile as sf

    from .audio import analyze, condense_pauses, decode, export, gain_to_lufs, trim_silence
    from .voices import NAME_RE, Voice, default_voice_name, set_default

    src = Path(args.recording).expanduser().resolve()
    if not src.is_file():
        die(f"recording not found: {src}")
    if not NAME_RE.match(args.name):
        die("voice name must be lowercase letters, digits, '-' or '_' (e.g. 'me', 'alex-mic2')")
    engines = [e.strip() for e in args.engines.split(",") if e.strip()]
    if bad := [e for e in engines if e not in C.ENGINES]:
        die(f"unknown engine(s): {', '.join(bad)} (choose from {', '.join(C.ENGINES)})")
    voice = Voice(args.name)
    if voice.dir.exists() and not args.force:
        die(f"voice '{args.name}' already exists. Re-run with --force to replace it.")

    # 1) Clean the recording into a reference clip
    log(f"Reading {src.name} ...")
    raw = decode(src, C.REF_SR, highpass_hz=70)
    raw_stats = analyze(raw, C.REF_SR)
    ref = condense_pauses(raw, C.REF_SR)[: int(C.MAX_REF_SECONDS * C.REF_SR)]
    ref = gain_to_lufs(ref, C.REF_SR, C.REF_LUFS)
    if (peak := float(abs(ref).max(initial=0.0))) > 0.95:
        ref = ref * (0.95 / peak)
    ref_seconds = len(ref) / C.REF_SR
    if ref_seconds < C.MIN_REF_SECONDS:
        die(f"only {ref_seconds:.1f}s of speech found in the recording; need at least "
            f"{C.MIN_REF_SECONDS:.0f}s (20-60s recommended).")

    warnings = []
    if ref_seconds < C.GOOD_REF_SECONDS:
        warnings.append(f"Only {ref_seconds:.0f}s of speech. It works, but 20-60s usually gives a closer likeness.")
    if raw_stats["snr_db"] < 25:
        warnings.append(f"Background noise is noticeable (speech-to-noise ~{raw_stats['snr_db']:.0f} dB). "
                        "The clone copies the room too; a quieter recording will sound better.")
    if raw_stats["clipped_pct"] > 0.1:
        warnings.append("The recording clips (too loud / too close to the mic) - distortion will carry over.")
    if raw_stats["seconds"] > 5 * 60:
        warnings.append(f"Long recording; only the first {C.MAX_REF_SECONDS:.0f}s of speech are used.")

    if voice.dir.exists():  # --force: replace the old profile entirely
        shutil.rmtree(voice.dir)
    voice.dir.mkdir(parents=True)
    shutil.copy2(src, voice.dir / f"source{src.suffix.lower()}")
    sf.write(voice.reference, ref, C.REF_SR, subtype="PCM_16")
    log(f"Reference clip: {ref_seconds:.1f}s of speech (from {raw_stats['seconds']:.1f}s recording)")

    # 2) Compute and cache the voice conditioning ("the clone") for each requested engine
    from .engine import Engine

    preview_engine, sims = None, {}
    t0 = time.time()
    for name in engines:
        heavy_engine_warning(name)
        log(f"Cloning voice for the {name} engine ...")
        eng = Engine(name, device=args.device, power=args.power, log=log)
        eng.use_voice(voice)
        if preview_engine is None:
            preview_engine = eng
    eng = preview_engine

    # 3) Preview + a rough similarity score
    log("Rendering a short preview ...")
    wav = trim_silence(eng.synthesize(args.preview_text, seed=42, language="en"), eng.sr)
    export(gain_to_lufs(wav, eng.sr, C.PODCAST_LUFS), eng.sr, voice.preview,
           metadata={"title": "Voice preview", "artist": args.display_name or args.name})
    sims[eng.name] = round(float(eng.speaker_embedding(ref, C.REF_SR) @ eng.speaker_embedding(wav, eng.sr)), 3)

    meta = {
        "name": args.name,
        "display_name": args.display_name or args.name,
        "created": dt.datetime.now().isoformat(timespec="seconds"),
        "source_file": str(src),
        "reference_seconds": round(ref_seconds, 2),
        "recording": raw_stats,
        "warnings": warnings,
        "speaker_similarity": sims,
    }
    voice.write_meta(meta)
    if args.default or default_voice_name() in (None, args.name):
        set_default(args.name)

    log(f"\nVoice '{args.name}' is ready ({human(time.time() - t0)}). Preview: {voice.preview}")
    log(f"Speaker similarity, reference vs. preview: {sims[eng.name]:.2f} (0-1, higher = closer)"
        f" | peak GPU memory {eng.peak_gpu_gb:.1f} GB")
    for w in warnings:
        log(f"  ! {w}")
    result({"voice": args.name, "dir": str(voice.dir), "preview": str(voice.preview),
            "reference_seconds": round(ref_seconds, 2), "speaker_similarity": sims[eng.name],
            "warnings": warnings, "default": default_voice_name() == args.name})


# --------------------------------------------------------------------------- podcast

def _episode_path(title: str) -> Path:
    base = f"{dt.date.today().isoformat()}-{slugify(title)}"
    folder, n = C.PODCASTS_DIR / base, 2
    while folder.exists():
        folder, n = C.PODCASTS_DIR / f"{base}-{n}", n + 1
    return folder / f"{folder.name}.mp3"


def _render_chunk(eng, text: str, seed: int, args: argparse.Namespace):
    """Synthesize one chunk; retry when the duration says the take is glitched."""
    from .audio import trim_silence

    lo, hi = speech_bounds(text)
    best, best_err = None, float("inf")
    for attempt in range(args.max_retries + 1):
        wav = eng.synthesize(text, seed + attempt * 7919, language=args.language,
                             exaggeration=args.exaggeration, cfg_weight=args.cfg_weight,
                             temperature=args.temperature)
        wav = trim_silence(wav, eng.sr)
        dur = len(wav) / eng.sr
        if lo <= dur <= hi:
            return wav, True, attempt + 1
        err = (lo - dur) / max(lo, 1e-3) if dur < lo else (dur - hi) / hi
        if err < best_err:
            best, best_err = wav, err
    return best, False, args.max_retries + 1


def cmd_podcast(args: argparse.Namespace) -> None:
    import numpy as np
    import soundfile as sf

    from .audio import export, fade, gain_to_lufs, silence
    from .voices import resolve

    # Validate everything cheap before loading the model
    if args.engine != "multilingual" and args.language != "en":
        die(f"the {args.engine} engine is English-only; use --engine multilingual --language {args.language}")
    if args.engine == "multilingual" and args.language not in C.MULTILINGUAL_LANGUAGES:
        die(f"unsupported language '{args.language}'. Supported: {', '.join(C.MULTILINGUAL_LANGUAGES)}")
    if not 0.7 <= args.speed <= 1.3:
        die("--speed must be between 0.7 and 1.3")

    if args.text is not None:
        raw = args.text
    elif args.file == "-":
        raw = sys.stdin.read()
    else:
        path = Path(args.file).expanduser()
        if not path.is_file():
            die(f"script not found: {path}")
        raw = path.read_text(encoding="utf-8")

    items, notes = parse_script(raw, engine=args.engine, max_chars=args.max_chars)
    chunks = [it for it in items if isinstance(it, Speech)]
    if not chunks:
        die("no speakable text found in the script.")
    voice = resolve(args.voice)
    meta = voice.meta()
    title = args.title or first_heading(raw) or f"Podcast {dt.date.today().isoformat()}"
    out = Path(args.out).expanduser().resolve() if args.out else _episode_path(title)
    redo = {int(x) for x in args.redo.split(",") if x.strip()} if args.redo else set()
    if bad := [n for n in redo if not 1 <= n <= len(chunks)]:
        die(f"--redo chunk numbers out of range (1-{len(chunks)}): {bad}")

    words = sum(word_count(c.text) for c in chunks)
    log(f"Voice '{voice.name}' | engine {args.engine} | power {args.power} | {len(chunks)} chunks, "
        f"~{int(words)} words (~{words / C.WORDS_PER_MINUTE:.1f} min of audio)")
    for n in notes:
        log(f"  note: {n}")
    heavy_engine_warning(args.engine)

    from .engine import Engine

    t_start = time.time()
    eng = None

    def load_engine():
        e = Engine(args.engine, device=args.device, power=args.power, log=log)
        e.use_voice(voice)  # first use of an engine for this voice computes + saves its clone
        return e

    if not voice.conds(args.engine).exists():
        eng = load_engine()
    conds_id = hashlib.sha1(voice.conds(args.engine).read_bytes()).hexdigest()[:12]
    cache = C.CACHE_DIR / voice.name
    cache.mkdir(parents=True, exist_ok=True)

    rendered: dict[int, np.ndarray] = {}
    flagged, cached, retries = [], 0, 0
    gen_chars, gen_secs = 0, 0.0
    paths = {c.index: cache / (hashlib.sha1(json.dumps(
        [args.engine, conds_id, args.language, args.exaggeration, args.cfg_weight, args.temperature,
         args.seed, c.text]).encode()).hexdigest() + ".wav") for c in chunks}
    fresh = {c.index for c in chunks if c.index in redo or not paths[c.index].exists()}
    todo_chars = sum(len(c.text) for c in chunks if c.index in fresh)
    if len(fresh) < len(chunks):
        log(f"{len(chunks) - len(fresh)} chunks unchanged since last time (from cache), {len(fresh)} to render")
    if fresh and eng is None:
        eng = load_engine()  # only load the ~4 GB model when something actually needs generating
    for c in chunks:
        path = paths[c.index]
        if c.index not in fresh:
            rendered[c.index], _ = sf.read(path, dtype="float32")
            cached += 1
            continue
        seed = random.randrange(1, 2**31) if c.index in redo else args.seed
        todo_chars -= len(c.text)
        t0 = time.time()
        wav, ok, tries = _render_chunk(eng, c.text, seed, args)
        took = time.time() - t0
        sf.write(path, wav, eng.sr)
        rendered[c.index] = wav
        retries += tries - 1
        if not ok:
            flagged.append(c.index)
        gen_chars, gen_secs = gen_chars + len(c.text), gen_secs + took
        rest = took * C.ECO_REST_RATIO if args.power == "eco" else 0.0
        eta = todo_chars * gen_secs / gen_chars * (1 + (C.ECO_REST_RATIO if args.power == "eco" else 0))
        log(f"[{c.index:>3}/{len(chunks)}] {len(wav) / eng.sr:5.1f}s audio in {took:5.1f}s"
            f"{' (retried)' if tries > 1 else ''}{' !check' if not ok else ''}"
            f" | ETA {human(eta)} | {c.text[:60]}{'...' if len(c.text) > 60 else ''}")
        if rest and todo_chars:
            time.sleep(rest)

    # Assemble: paragraph/heading pauses from the script, short gaps between chunks
    sr = eng.sr if eng else sf.info(str(paths[chunks[0].index])).samplerate
    parts, starts, t, prev_speech = [silence(0.4, sr)], {}, 0.4, False
    for it in items:
        if isinstance(it, Pause):
            parts.append(silence(it.seconds, sr))
            t += it.seconds
            prev_speech = False
            continue
        if prev_speech:
            parts.append(silence(CHUNK_PAUSE, sr))
            t += CHUNK_PAUSE
        starts[it.index] = t
        wav = fade(rendered[it.index], sr)
        parts.append(wav)
        t += len(wav) / sr
        prev_speech = True
    parts.append(silence(1.0, sr))
    mix = gain_to_lufs(np.concatenate(parts), sr, C.PODCAST_LUFS)

    artist = meta.get("display_name", voice.name)
    export(mix, sr, out, speed=args.speed, metadata={
        "title": title, "artist": artist, "album_artist": artist, "album": args.show or "Podcast",
        "date": str(dt.date.today().year), "genre": "Podcast",
        "comment": "Synthetic voice (cloned with consent; Chatterbox TTS, watermarked)"})
    duration = len(mix) / sr / args.speed

    # Sidecars: the exact script, and a chunk map for fixing individual sentences
    script_path, chunk_path = out.with_suffix(".script.md"), out.with_suffix(".chunks.txt")
    script_path.write_text(raw if raw.endswith("\n") else raw + "\n", encoding="utf-8")
    lines = ["# chunk  start     text   ('!' = duration looked off; listen there first)"]
    for c in chunks:
        mark = "!" if c.index in flagged else " "
        lines.append(f"{mark}{c.index:03d}    {clock(starts[c.index] / args.speed)}   {c.text}")
    chunk_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    elapsed = time.time() - t_start
    log(f"\nDone in {human(elapsed)}: {out}  ({human(duration)} long, peak GPU memory {eng.peak_gpu_gb if eng else 0:.1f} GB)")
    if flagged:
        log(f"  ! chunks {', '.join(map(str, flagged))} may be glitched - listen around those times "
            f"(see {chunk_path.name}) and re-render with --redo {','.join(map(str, flagged))}")
    result({"output": str(out), "duration_sec": round(duration, 1), "chunks": len(chunks),
            "from_cache": cached, "retries": retries, "flagged_chunks": flagged,
            "script": str(script_path), "chunk_map": str(chunk_path), "voice": voice.name,
            "engine": args.engine, "power": args.power, "elapsed_sec": round(elapsed, 1),
            "notes": notes})


# --------------------------------------------------------------------------- argparse

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="selfcast", description="Turn text into a podcast in your own cloned voice, locally.")
    sub = p.add_subparsers(dest="cmd", required=True)

    def power_args(sp: argparse.ArgumentParser) -> None:
        sp.add_argument("--power", choices=C.POWER_MODES, default=C.DEFAULT_POWER,
                        help="eco (default): slower, cooler, low memory. fast: full speed.")
        sp.add_argument("--device", choices=["cuda", "mps", "cpu"],
                        help="default: cuda if available, else mps (Apple GPU), else cpu")

    c = sub.add_parser("clone", help="Create a voice profile from a recording (one-time).")
    c.add_argument("--recording", "-r", required=True, help="any audio/video file (m4a, mp3, wav, mov...)")
    c.add_argument("--name", "-n", default="me", help="profile name (default: me)")
    c.add_argument("--display-name", help="name written as the podcast's artist tag")
    c.add_argument("--engines", default=C.DEFAULT_ENGINE,
                   help="engines to prepare now, comma-separated (others are prepared on first use)")
    c.add_argument("--force", action="store_true", help="replace an existing profile with this name")
    c.add_argument("--default", action="store_true", help="make this the default voice")
    c.add_argument("--preview-text", default=C.PREVIEW_TEXT)
    power_args(c)
    c.set_defaults(func=cmd_clone)

    q = sub.add_parser("podcast", help="Render a script into a podcast episode.")
    src = q.add_mutually_exclusive_group(required=True)
    src.add_argument("--file", "-f", help="script file (.md / .txt), or '-' for stdin")
    src.add_argument("--text", "-t", help="inline text")
    q.add_argument("--voice", "-v", help="voice profile (default: the default voice)")
    q.add_argument("--title", help="episode title (default: first '# heading', else the date)")
    q.add_argument("--show", help="show name, written as the album tag")
    q.add_argument("--out", "-o", help="output .mp3/.m4a/.wav (default: podcasts/<date>-<title>/...)")
    q.add_argument("--engine", "-e", choices=C.ENGINES, default=C.DEFAULT_ENGINE)
    q.add_argument("--language", "-l", default="en", help="multilingual engine only")
    q.add_argument("--speed", type=float, default=1.0, help="tempo 0.7-1.3 (pitch preserved)")
    q.add_argument("--seed", type=int, default=42)
    q.add_argument("--temperature", type=float, default=0.8)
    q.add_argument("--exaggeration", type=float, default=0.5, help="standard/multilingual only")
    q.add_argument("--cfg-weight", type=float, default=0.5, help="standard/multilingual only")
    q.add_argument("--max-chars", type=int, default=280, help="max characters per generated chunk")
    q.add_argument("--max-retries", type=int, default=2)
    q.add_argument("--redo", help="chunk numbers to re-take, e.g. 4,12 (see the .chunks.txt file)")
    power_args(q)
    q.set_defaults(func=cmd_podcast)

    v = sub.add_parser("voices", help="List voice profiles.")
    v.set_defaults(func=cmd_voices)
    return p


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)

"""Command-line entry point:  uv run selfcast <clone|podcast|master|voices> ...

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
from .text import (CHUNK_PAUSE, Pause, Speech, chapters, first_heading, parse_script, slugify, speech_bounds,
                   word_count)


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
        log(f"WARNING: the '{engine}' engine peaked at ~18 GB of memory in testing (turbo needs ~7 GB) "
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
    auto = str(args.ref_start).lower() == "auto"
    try:
        ref_start = 0.0 if auto else float(args.ref_start)
    except ValueError:
        die("--ref-start must be a number of seconds or 'auto'")
    if ref_start < 0 or not C.MIN_REF_SECONDS <= args.ref_seconds <= 300:
        die(f"--ref-start must be >= 0 and --ref-seconds between {C.MIN_REF_SECONDS:.0f} and 300")

    # 1) Clean the recording into a reference clip
    log(f"Reading {src.name} ...")
    raw = decode(src, C.REF_SR, highpass_hz=70)
    raw_stats = analyze(raw, C.REF_SR)
    if ref_start >= raw_stats["seconds"]:
        die(f"--ref-start {ref_start:g}s is past the end of the recording ({raw_stats['seconds']:.1f}s)")
    offset = int(ref_start * C.REF_SR)
    speech, seg_map = condense_pauses(raw[offset:], C.REF_SR, return_map=True)
    window_info, eng = None, None
    if auto:
        eng, window_info, speech = _choose_reference_window(raw, speech, seg_map, args)
    ref = speech[: int(args.ref_seconds * C.REF_SR)]
    if window_info is None:
        window_info = {"mode": "start", "start_sec": round(ref_start, 2)}
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
    if raw_stats["seconds"] > 5 * 60 and not auto:
        warnings.append(f"Long recording; only {args.ref_seconds:.0f}s of speech are used. Try --ref-start auto "
                        "to pick the best stretch.")

    if voice.dir.exists():  # --force: replace the old profile entirely
        shutil.rmtree(voice.dir)
    voice.dir.mkdir(parents=True)
    shutil.copy2(src, voice.dir / f"source{src.suffix.lower()}")
    sf.write(voice.reference, ref, C.REF_SR, subtype="PCM_16")
    log(f"Reference clip: {ref_seconds:.1f}s of speech (from {raw_stats['seconds']:.1f}s recording)")

    # 2) Compute and cache the voice conditioning ("the clone") for each requested engine
    from .engine import Engine, peak_memory_gb

    preview_engine, sims = None, {}
    t0 = time.time()
    first = eng  # already loaded to score windows in auto mode
    for name in engines:
        heavy_engine_warning(name)
        log(f"Cloning voice for the {name} engine ...")
        if first is None or first.name != name:
            eng = Engine(name, device=args.device, power=args.power, log=log)
        else:
            eng, first = first, None
        eng.use_voice(voice)
        if preview_engine is None:
            preview_engine = eng
    eng = preview_engine

    # 3) Preview + a rough similarity score
    log("Rendering a short preview ...")
    wav = trim_silence(eng.synthesize(args.preview_text, seed=42, language="en"), eng.sr)
    export(gain_to_lufs(wav, eng.sr, C.PODCAST_LUFS), eng.sr, voice.preview, loudness=C.PODCAST_LUFS,
           metadata={"title": "Voice preview", "artist": args.display_name or args.name})
    sims[eng.name] = round(float(eng.speaker_embedding(ref, C.REF_SR) @ eng.speaker_embedding(wav, eng.sr)), 3)

    meta = {
        "name": args.name,
        "display_name": args.display_name or args.name,
        "created": dt.datetime.now().isoformat(timespec="seconds"),
        "source_file": str(src),
        "reference_seconds": round(ref_seconds, 2),
        "reference_window": window_info,
        "recording": raw_stats,
        "warnings": warnings,
        "speaker_similarity": sims,
    }
    voice.write_meta(meta)
    if args.default or default_voice_name() in (None, args.name):
        set_default(args.name)

    log(f"\nVoice '{args.name}' is ready ({human(time.time() - t0)}). Preview: {voice.preview}")
    log(f"Speaker similarity, reference vs. preview: {sims[eng.name]:.2f} (0-1, higher = closer)"
        f" | peak memory {peak_memory_gb():.1f} GB (GPU {eng.peak_gpu_gb:.1f} GB)")
    for w in warnings:
        log(f"  ! {w}")
    result({"voice": args.name, "dir": str(voice.dir), "preview": str(voice.preview),
            "reference_seconds": round(ref_seconds, 2), "reference_window": window_info,
            "speaker_similarity": sims[eng.name],
            "warnings": warnings, "default": default_voice_name() == args.name})


def _choose_reference_window(raw, speech, seg_map, args):
    """--ref-start auto: score every 15 s stretch of speech and put the best one first, since
    Turbo builds its speech prompt from the first 15 s of the reference (the decoder from the
    first 10 s); only the speaker embedding hears the rest. Returns (engine, info, speech)."""
    import librosa
    import numpy as np

    from .audio import REF_WINDOW, best_window_first, rank_windows, to_source_time, window_features
    from .engine import Engine

    scan = speech[: int(C.REF_SCAN_SECONDS * C.REF_SR)]
    if len(scan) < (REF_WINDOW + 2.5) * C.REF_SR:
        log("Recording too short to choose a window; using it from the start.")
        return None, None, speech
    log(f"Looking for the best {REF_WINDOW:.0f}s of speech in {len(scan) / C.REF_SR:.0f}s ...")
    rms = librosa.feature.rms(y=raw, frame_length=int(0.03 * C.REF_SR), hop_length=int(0.015 * C.REF_SR))[0]
    noise_floor = float(np.percentile(20 * np.log10(rms + 1e-9), 10))
    feats = window_features(scan, C.REF_SR, noise_floor)
    heavy_engine_warning(args.engines.split(",")[0].strip())
    eng = Engine(args.engines.split(",")[0].strip(), device=args.device, power=args.power, log=log)
    whole = eng.speaker_embedding(scan, C.REF_SR)
    n = int(REF_WINDOW * C.REF_SR)
    sims = [float(eng.speaker_embedding(scan[int(f["start"] * C.REF_SR):][:n], C.REF_SR) @ whole) for f in feats]
    scores = rank_windows(feats, sims)
    for f, sim, sc in zip(feats, sims, scores):
        f.update(similarity=round(sim, 3), score=sc, source_start=round(to_source_time(f["start"], seg_map, C.REF_SR), 1))
    ranked = sorted(feats, key=lambda f: -f["score"])
    first = feats[0]
    log("  start   score   SNR   pace   pitch  similarity")
    for f in ranked[:3] + ([first] if first not in ranked[:3] else []):
        tag = "  <- chosen" if f is ranked[0] else ("  (the very start)" if f is first else "")
        log(f"  {clock(f['source_start'])} {f['score']:6.2f} {f['snr_db']:5.0f}dB {f['rate']:4.1f}/s "
            f"{f['pitch_var']:5.2f}st  {f['similarity']:.3f}{tag}")
    best = ranked[0]
    info = {"mode": "auto", "start_sec": best["source_start"], "seconds": REF_WINDOW,
            "score": best["score"], "first_window_score": first["score"],
            "features": {k: best[k] for k in ("snr_db", "rate", "pitch_var", "similarity")}}
    return eng, info, best_window_first(speech, C.REF_SR, best["start"], REF_WINDOW, args.ref_seconds)


# --------------------------------------------------------------------------- podcast

def _episode_path(title: str) -> Path:
    base = f"{dt.date.today().isoformat()}-{slugify(title)}"
    folder, n = C.PODCASTS_DIR / base, 2
    while folder.exists():
        folder, n = C.PODCASTS_DIR / f"{base}-{n}", n + 1
    return folder / f"{folder.name}.mp3"


def _duration_error(text: str, seconds: float) -> float:
    """0 when the take's length is plausible for its text, else how far outside (relative)."""
    lo, hi = speech_bounds(text)
    if lo <= seconds <= hi:
        return 0.0
    return (lo - seconds) / max(lo, 1e-3) if seconds < lo else (seconds - hi) / hi


def _take_info(text: str, wav, sr: int, check) -> dict:
    """What we know about a take; saved next to the cached audio so flags survive re-runs."""
    seconds = len(wav) / sr
    return {"duration": round(seconds, 2), "duration_ok": _duration_error(text, seconds) == 0.0,
            "qa": check.as_dict() if check is not None else None}


def _problems(text: str, info: dict, judge) -> list[str]:
    """Why a take still needs a listen ([] when it looks fine)."""
    from .qa import describe, is_acceptable

    out = []
    if not info.get("duration_ok", True):
        lo, hi = speech_bounds(text)
        out.append(f"{info['duration']:.1f}s long, expected {lo:.1f}-{hi:.1f}s")
    qa = info.get("qa")
    if judge is not None and qa and not is_acceptable(qa, judge.threshold):
        out.append(describe(qa))
    return out


def _script_with_reading(raw: str, items: list, normalized: bool) -> str:
    """The script as given, plus (inside an HTML comment, so re-rendering from this file reads
    exactly the same) the text the voice model was actually given, chunk by chunk."""
    out = raw if raw.endswith("\n") else raw + "\n"
    if not normalized:
        return out
    lines = ["", "<!-- selfcast: the text as it was read (after pronunciation and number normalization).",
             "     [n] = chunk number in .chunks.txt, (pause) = silence inserted between chunks.", ""]
    for it in items:
        if isinstance(it, Pause):
            lines.append(f"(pause {it.seconds:.2f}s)")
            continue
        lines.append(f"[{it.index}] {it.text.replace('--', '- -')}")
    return out + "\n".join(lines) + "\n-->\n"


def _render_chunk(eng, text: str, seed: int, args: argparse.Namespace, judge):
    """Synthesize one chunk. A take with an implausible length (truncated or rambling), or one
    Whisper hears as different words, is re-taken with a new seed, up to --max-retries times.
    Returns (audio, info, seconds spent synthesizing); the best take wins."""
    from .audio import trim_silence

    best, best_key, synth_secs, qa_rejects = None, None, 0.0, 0
    for attempt in range(args.max_retries + 1):
        t0 = time.time()
        wav = eng.synthesize(text, seed + attempt * 7919, language=args.language,
                             exaggeration=args.exaggeration, cfg_weight=args.cfg_weight,
                             temperature=args.temperature)
        wav = trim_silence(wav, eng.sr)
        synth_secs += time.time() - t0
        check = judge.check(text, wav, eng.sr) if judge is not None else None
        info = _take_info(text, wav, eng.sr, check)
        dur_err = _duration_error(text, info["duration"])
        qa_bad = check is not None and not check.ok
        key = (dur_err > 0, qa_bad, check.wer if check else 0.0, dur_err)
        if best_key is None or key < best_key:
            best, best_key = (wav, info), key
        if not key[0] and not key[1]:
            break
        qa_rejects += int(qa_bad and not key[0])
    wav, info = best
    info["tries"], info["qa_retakes"] = attempt + 1, qa_rejects
    return wav, info, synth_secs


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

    from .normalize import load_lexicon

    lex_file = Path(args.pronunciations).expanduser() if args.pronunciations else C.USER_LEXICON
    if args.pronunciations and not lex_file.is_file():
        die(f"pronunciation file not found: {lex_file}")
    try:
        lexicon = load_lexicon(lex_file)
    except ValueError as e:  # includes tomllib.TOMLDecodeError
        die(f"could not read the pronunciation file: {e}")
    normalize_text = args.normalize and args.engine != "multilingual"  # the rules are English
    items, notes = parse_script(raw, engine=args.engine, max_chars=args.max_chars,
                                lexicon=lexicon if normalize_text else None,
                                pacing_seed=args.seed if args.pacing == "natural" else None)
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

    judge = None
    if args.qa:
        from .qa import Judge

        judge = Judge(args.qa_threshold, variants=lexicon.accepted_variants(), log=log)

    rendered: dict[int, np.ndarray] = {}
    infos: dict[int, dict] = {}
    cached, retries, qa_retakes = 0, 0, 0
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
        path, info_path = paths[c.index], paths[c.index].with_suffix(".json")
        if c.index not in fresh:
            wav, wav_sr = sf.read(path, dtype="float32")
            rendered[c.index] = wav
            try:
                info = json.loads(info_path.read_text())
            except (FileNotFoundError, json.JSONDecodeError):  # cached before verdicts were saved
                info = _take_info(c.text, wav, wav_sr, None)
            if judge is not None:
                if info.get("qa") is None:  # cached before QA, or rendered with --no-qa
                    info["qa"] = judge.check(c.text, wav, wav_sr).as_dict()
                else:  # re-score the saved transcript, so rule or word-list changes apply
                    info["qa"] = judge.score(c.text, info["qa"]["heard"]).as_dict()
                info_path.write_text(json.dumps(info, ensure_ascii=False))
            infos[c.index] = info
            cached += 1
            continue
        seed = random.randrange(1, 2**31) if c.index in redo else args.seed
        todo_chars -= len(c.text)
        t0 = time.time()
        wav, info, synth_secs = _render_chunk(eng, c.text, seed, args, judge)
        took = time.time() - t0
        sf.write(path, wav, eng.sr)
        info_path.write_text(json.dumps(info, ensure_ascii=False))
        rendered[c.index], infos[c.index] = wav, info
        retries += info["tries"] - 1
        qa_retakes += info["qa_retakes"]
        problems = _problems(c.text, info, judge)
        # Eco mode rests the GPU for half the time it worked. Whisper runs on the CPU, so the time
        # spent transcribing counts towards that rest instead of adding to it.
        rest = max(0.0, synth_secs * C.ECO_REST_RATIO - (took - synth_secs)) if args.power == "eco" else 0.0
        gen_chars, gen_secs = gen_chars + len(c.text), gen_secs + took + rest
        eta = todo_chars * gen_secs / gen_chars
        qa_note = f" WER {info['qa']['wer']:.0%}" if info.get("qa") else ""
        log(f"[{c.index:>3}/{len(chunks)}] {len(wav) / eng.sr:5.1f}s audio in {took:5.1f}s{qa_note}"
            f"{' (retried)' if info['tries'] > 1 else ''}{' !check' if problems else ''}"
            f" | ETA {human(eta)} | {c.text[:60]}{'...' if len(c.text) > 60 else ''}")
        if rest and todo_chars:
            time.sleep(rest)
    reasons = {c.index: "; ".join(p) for c in chunks if (p := _problems(c.text, infos[c.index], judge))}
    flagged = sorted(reasons)

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

    episode_chapters = chapters(items, {i: t / args.speed for i, t in starts.items()}, title)
    artist = meta.get("display_name", voice.name)
    export(mix, sr, out, speed=args.speed, chapters=episode_chapters, loudness=C.PODCAST_LUFS, metadata={
        "title": title, "artist": artist, "album_artist": artist, "album": args.show or "Podcast",
        "date": str(dt.date.today().year), "genre": "Podcast",
        "comment": "Synthetic voice (cloned with consent; Chatterbox TTS, watermarked)"})
    duration = len(mix) / sr / args.speed

    # Sidecars: the exact script, a chunk map for fixing individual sentences, and the chapters
    script_path, chunk_path = out.with_suffix(".script.md"), out.with_suffix(".chunks.txt")
    chapters_path = out.with_suffix(".chapters.json")
    embedded = None
    if episode_chapters:
        chapters_path.write_text(json.dumps(episode_chapters, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        from .audio import chapter_count

        embedded = chapter_count(out) if out.suffix.lower() in (".mp3", ".m4a") else 0
        if embedded == 0 and out.suffix.lower() in (".mp3", ".m4a"):
            notes.append("This ffmpeg didn't write chapters into the audio file; they're in .chapters.json only.")
    elif chapters_path.exists():
        chapters_path.unlink()  # re-render of a script that no longer has ## headings
    script_path.write_text(_script_with_reading(raw, items, normalize_text), encoding="utf-8")
    lines = ["# chunk  start     text   ('!' = still sounded off after retries; listen there first)"]
    for c in chunks:
        mark = "!" if c.index in flagged else " "
        lines.append(f"{mark}{c.index:03d}    {clock(starts[c.index] / args.speed)}   {c.text}")
    if reasons:
        lines += ["", "# Why the '!' chunks were flagged:"]
        lines += [f"#  {i:03d}  {reasons[i]}" for i in flagged]
    chunk_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    from .engine import peak_memory_gb

    elapsed, peak_gpu, peak_mem = time.time() - t_start, (eng.peak_gpu_gb if eng else 0.0), peak_memory_gb()
    qa_stats = None
    if judge is not None:
        wers = [infos[c.index]["qa"]["wer"] for c in chunks if infos[c.index].get("qa")]
        qa_stats = {"model": judge.model_name, "threshold": judge.threshold, "checked": len(wers),
                    "mean_wer": round(sum(wers) / max(len(wers), 1), 3), "max_wer": max(wers, default=0.0),
                    "retakes": qa_retakes, "seconds": round(judge.seconds, 1)}
    log(f"\nDone in {human(elapsed)}: {out}  ({human(duration)} long, peak GPU memory {peak_gpu:.1f} GB, "
        f"process peak {peak_mem:.1f} GB)")
    if episode_chapters:
        log(f"  {len(episode_chapters)} chapters: {chapters_path.name}"
            + (", also embedded in the audio file" if embedded else ""))
    if qa_stats:
        log(f"  QA: mean WER {qa_stats['mean_wer']:.1%}, {qa_retakes} re-takes for misread words, "
            f"{qa_stats['seconds']:.0f}s transcribing")
    if flagged:
        log(f"  ! chunks {', '.join(map(str, flagged))} may be glitched - listen around those times "
            f"(see {chunk_path.name}) and re-render with --redo {','.join(map(str, flagged))}")
        for i in flagged:
            log(f"    {i:03d}: {reasons[i]}")
    result({"output": str(out), "duration_sec": round(duration, 1), "chunks": len(chunks),
            "from_cache": cached, "retries": retries, "flagged_chunks": flagged,
            "script": str(script_path), "chunk_map": str(chunk_path), "voice": voice.name,
            "engine": args.engine, "power": args.power, "elapsed_sec": round(elapsed, 1),
            "notes": notes, "flag_reasons": {str(i): reasons[i] for i in flagged}, "qa": qa_stats,
            "peak_gpu_gb": round(peak_gpu, 2), "peak_memory_gb": round(peak_mem, 2),
            "chapters": str(chapters_path) if episode_chapters else None,
            "chapter_count": len(episode_chapters), "chapters_embedded": bool(embedded)})


# --------------------------------------------------------------------------- master

MASTER_SR = 48000


def _write_transcript(words: list, path: Path) -> None:
    """One line per sentence, with its start time."""
    lines, sentence, start = [], [], None
    for word, t0, _ in words:
        start = t0 if start is None else start
        sentence.append(word)
        if word.endswith((".", "?", "!")):
            lines.append(f"[{clock(start)}] {' '.join(sentence)}")
            sentence, start = [], None
    if sentence:
        lines.append(f"[{clock(start)}] {' '.join(sentence)}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def cmd_master(args: argparse.Namespace) -> None:
    """Master your own recording of a script: denoise, shorten long pauses, normalize like a
    rendered episode, and find where each ## section starts to write chapters."""
    from .audio import (analyze, chapter_count, decode, denoise_deepfilter, export, gain_to_lufs,
                        shorten_pauses, speech_segments)
    from .engine import apply_power_mode, peak_memory_gb
    from .text import script_sections

    src = Path(args.recording).expanduser().resolve()
    if not src.is_file():
        die(f"recording not found: {src}")
    raw_script = None
    if args.script:
        script = Path(args.script).expanduser()
        if not script.is_file():
            die(f"script not found: {script}")
        raw_script = script.read_text(encoding="utf-8")
    if not 0.2 <= args.keep_pause <= args.max_pause:
        die("--keep-pause must be at least 0.2 s and no longer than --max-pause")
    title = args.title or (first_heading(raw_script) if raw_script else None) or src.stem
    out = Path(args.out).expanduser().resolve() if args.out else _episode_path(title)
    apply_power_mode(args.power, "cpu")  # everything here runs on the CPU
    t_start, notes = time.time(), []

    log(f"Reading {src.name} ...")
    afftdn = "afftdn=nr=12:nf=-50:tn=1" if args.denoise == "afftdn" else None
    wav = decode(src, MASTER_SR, highpass_hz=70, filters=afftdn)
    before = analyze(wav, MASTER_SR)
    if args.denoise == "deepfilter":
        log("Removing noise with DeepFilterNet ...")
        try:
            wav = denoise_deepfilter(wav, MASTER_SR)
        except RuntimeError as e:
            die(str(e))
    log("Finding speech and shortening long pauses ...")
    segments = speech_segments(wav, MASTER_SR)
    if not segments:
        die("no speech found in the recording.")
    wav, cut = shorten_pauses(wav, MASTER_SR, segments, max_pause=args.max_pause, keep=args.keep_pause)
    mix = gain_to_lufs(wav, MASTER_SR, C.PODCAST_LUFS)
    duration = len(mix) / MASTER_SR

    chapters, words, missing = [], [], []
    if raw_script is not None or args.transcript:
        from .normalize import load_lexicon
        from .qa import Judge, align_sections

        log("Transcribing with Whisper (CPU) ...")
        judge = Judge(C.QA_THRESHOLD, variants=load_lexicon(C.USER_LEXICON).accepted_variants(), log=log)
        words = judge.words(mix, MASTER_SR)
        sections = script_sections(raw_script) if raw_script else []
        if sections:
            found = align_sections(judge, sections, words)
            missing = [f["title"] for f in found if f["start"] is None]
            last = -1
            for f in found:
                if f["start"] is not None and int(f["start"]) > last:
                    chapters.append({"start": int(f["start"]), "title": f["title"]})
                    last = int(f["start"])
            if chapters and chapters[0]["start"] > 0:
                chapters.insert(0, {"start": 0, "title": title})
            if missing:
                notes.append("Couldn't find these sections in the recording, so they have no chapter: "
                             + ", ".join(f'"{m}"' for m in missing))
        elif raw_script is not None:
            notes.append("The script has no ## headings, so there are no chapters.")

    export(mix, MASTER_SR, out, chapters=chapters, loudness=C.PODCAST_LUFS, metadata={
        "title": title, "artist": args.artist, "album_artist": args.artist, "album": args.show or "Podcast",
        "date": str(dt.date.today().year), "genre": "Podcast"})
    chapters_path, transcript_path = out.with_suffix(".chapters.json"), out.with_suffix(".transcript.txt")
    embedded = None
    if chapters:
        chapters_path.write_text(json.dumps(chapters, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        embedded = chapter_count(out) if out.suffix.lower() in (".mp3", ".m4a") else 0
    if words:
        _write_transcript(words, transcript_path)
    if raw_script is not None:
        out.with_suffix(".script.md").write_text(raw_script if raw_script.endswith("\n") else raw_script + "\n",
                                                encoding="utf-8")

    elapsed, peak = time.time() - t_start, peak_memory_gb()
    log(f"\nDone in {human(elapsed)}: {out}  ({human(duration)} long, was {human(before['seconds'])}; "
        f"process peak {peak:.1f} GB)")
    log(f"  {cut['shortened']} long pauses shortened ({cut['removed_sec']:.0f}s removed), "
        f"-16 LUFS, peaks at -1 dBFS, denoise: {args.denoise}")
    if chapters:
        log(f"  {len(chapters)} chapters: {chapters_path.name}" + (", also embedded in the audio file" if embedded else ""))
        for ch in chapters:
            log(f"    {clock(ch['start'])}  {ch['title']}")
    for n in notes:
        log(f"  note: {n}")
    result({"output": str(out), "duration_sec": round(duration, 1), "original_sec": before["seconds"],
            "pauses_shortened": cut["shortened"], "removed_sec": cut["removed_sec"], "denoise": args.denoise,
            "chapters": str(chapters_path) if chapters else None, "chapter_count": len(chapters),
            "chapters_embedded": bool(embedded), "missing_sections": missing,
            "transcript": str(transcript_path) if words else None, "elapsed_sec": round(elapsed, 1),
            "peak_memory_gb": round(peak, 2), "notes": notes})


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
    c.add_argument("--ref-start", default="0",
                   help="where the reference starts, in seconds into the recording, or 'auto' to pick the best "
                        "15 s of speech and put it first (the model's style comes from the first 10-15 s)")
    c.add_argument("--ref-seconds", type=float, default=C.MAX_REF_SECONDS,
                   help=f"seconds of speech to keep (default {C.MAX_REF_SECONDS:.0f})")
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
    q.add_argument("--normalize", action=argparse.BooleanOptionalAction, default=True,
                   help="write out numbers, apply pronunciations.toml, turn colons into sentence breaks "
                        "(default: on; English engines only)")
    q.add_argument("--pronunciations", help="pronunciation file (default: pronunciations.toml in the project root, "
                                           "merged over the built-in one)")
    q.add_argument("--pacing", choices=("natural", "fixed"), default="natural",
                   help="natural (default): pauses vary with punctuation, sentence and paragraph length. "
                        "fixed: the same pause at every join")
    q.add_argument("--max-retries", type=int, default=2)
    q.add_argument("--qa", action=argparse.BooleanOptionalAction, default=C.QA_DEFAULT,
                   help="check every chunk with Whisper (CPU) and re-take misread ones (default: on)")
    q.add_argument("--qa-threshold", type=float, default=C.QA_THRESHOLD,
                   help=f"word error rate above which a take is re-done (default {C.QA_THRESHOLD})")
    q.add_argument("--redo", help="chunk numbers to re-take, e.g. 4,12 (see the .chunks.txt file)")
    power_args(q)
    q.set_defaults(func=cmd_podcast)

    m = sub.add_parser("master", help="Master your own recording of a script (denoise, pauses, loudness, chapters).")
    m.add_argument("--recording", "-r", required=True, help="your recording (any audio/video file)")
    m.add_argument("--script", "-s", help="the script you read (Markdown); its ## headings become chapters")
    m.add_argument("--out", "-o", help="output .mp3/.m4a/.wav (default: podcasts/<date>-<title>/...)")
    m.add_argument("--title", help="episode title (default: the script's first heading, else the file name)")
    m.add_argument("--show", help="show name, written as the album tag")
    m.add_argument("--artist", help="your name, written as the artist tag")
    m.add_argument("--denoise", choices=("afftdn", "deepfilter", "none"), default="afftdn",
                   help="afftdn (default): ffmpeg's FFT denoiser. deepfilter: DeepFilterNet3, stronger "
                        "(install it first: uv pip install deepfilternet)")
    m.add_argument("--max-pause", type=float, default=0.8,
                   help="silences between speech longer than this (s) are shortened (default 0.8, i.e. ~1 s pauses)")
    m.add_argument("--keep-pause", type=float, default=0.6, help="what they're shortened to (default 0.6)")
    m.add_argument("--transcript", action="store_true", help="write a timestamped transcript even without a script")
    m.add_argument("--power", choices=C.POWER_MODES, default=C.DEFAULT_POWER,
                   help="eco (default): low priority, 2 threads. fast: full speed.")
    m.set_defaults(func=cmd_master)

    v = sub.add_parser("voices", help="List voice profiles.")
    v.set_defaults(func=cmd_voices)
    return p


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)

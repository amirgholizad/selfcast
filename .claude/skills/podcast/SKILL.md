---
name: podcast
description: Turn text into a podcast episode spoken in the user's cloned voice, generated fully locally on this machine (Chatterbox TTS). Use when the user hands over text, a script or a file and wants it as a podcast, audio, narration or voice-over in their voice, asks to fix or re-take part of an episode, or types /podcast. Needs a voice profile made once with /clone-voice.
argument-hint: "[script file | inline text] [--title ...]"
---

# Text → podcast in the user's cloned voice

Run every command from the project root (the folder containing `pyproject.toml`).

## Steps

1. **Get the text.**
   - A file path → use that file as-is.
   - Inline text → save it **verbatim** with the Write tool to `scripts/YYYY-MM-DD-<short-slug>.md`,
     so there's a record and later fixes (`--redo`) can point at the same file.
   - Nothing given → ask for the text.
   - Read the text as written. Don't rewrite, summarize, add an intro or outro, or insert tags
     like `[laugh]` unless the user asks. If they want it "podcast-style", draft the rewrite,
     show it to them, save it as the script file, and only then render.

2. **Check for a voice:** `uv run selfcast voices`. If there's none, the user needs
   `/clone-voice` with a recording first.

3. **Estimate, then render.** Audio minutes ≈ words / 150. In the default eco mode on an Apple
   Silicon Mac, rendering takes ~1.5 min per audio minute, plus ~20 s to load the model. Tell
   the user the estimate when the episode is longer than ~2 minutes.

   ```bash
   uv run selfcast podcast --file scripts/<file>.md [--title "Episode title"] [--show "Show name"]
   ```

   - For anything over ~1 minute of audio, run it with Bash `run_in_background: true`. You're
     notified when it exits, so don't poll it.
   - Keep `--power eco` (the default). It takes ~7 GB of memory and leaves the machine
     responsive. Use `--power fast` only if the user asks for speed.
   - Never run two renders at once.

4. **Report** from the final `RESULT {...}` line: the output path as a link, the duration, and
   any `flagged_chunks` (with the reason from `flag_reasons`, e.g. what Whisper heard) or `notes`,
   and the `chapter_count` when there are chapters. Offer to play it: `open "<mp3>"` (default player) or `afplay "<mp3>"`.

Output goes to `podcasts/<date>-<title>/`: the `.mp3` (44.1 kHz mono, -16 LUFS, tagged, with
chapters from the `##` headings), the `.script.md` (the script, plus the text as it was read), a
`.chapters.json` (`[{start, title}]`, when there are `##` headings), and a `.chunks.txt` map giving
the start time of every chunk.

## Fixing part of an episode

- **"Something at 3:41 sounds off"**: read the episode's `.chunks.txt`, pick the chunk whose start
  time is ≤ 3:41 and the closest, then re-take only that chunk:
  `uv run selfcast podcast --file <same script> --out <same .mp3> --redo 23`
  (several: `--redo 4,23`). Every other chunk comes from the cache, so it takes seconds.
- **Wording change**: edit the script file and re-run with `--out <same .mp3>`. Only changed chunks are re-generated.
- **Mispronounced word**: add it to `pronunciations.toml` in the project root (git-ignored; create
  it if needed), e.g. `Nguyen = "Win"`, then re-run with the same `--out`. Only the chunks containing
  it are re-generated. Numbers, spelled-out letters ("C S S" → CSS) and colons are already handled
  automatically: don't rewrite them in the script. The episode's `.script.md` ends with the text as
  it was actually read.
- Chunks marked `!` in `.chunks.txt` still sounded off after retries: Whisper heard different words, or
  the take was an unusual length. The reason is listed at the bottom of `.chunks.txt`. Listen there first,
  and `--redo` them if they really are wrong. Whisper itself sometimes mishears names or new words: if a
  flag is a false alarm, add the word's spelling as heard to `pronunciations.toml` (see the README).

## The user recorded it themselves

If the user hands over **their own recording** of a script (not text to synthesize), master it
instead of rendering:

```bash
uv run selfcast master --recording "<file>" --script scripts/<file>.md
```

It denoises, shortens long pauses, normalizes to -16 LUFS / -1 dBFS and writes chapters from the
script's `##` headings (CPU only, ~1 minute per 5 minutes of audio). Report from `RESULT`: output,
`duration_sec` vs `original_sec`, `pauses_shortened`, `chapter_count`, and any `missing_sections`
(headings it couldn't find in the recording). Recordings are private: never commit or share them.

## Options (podcast)

| Flag | Default | Notes |
|---|---|---|
| `--voice NAME` | default voice | see `selfcast voices` |
| `--title`, `--show` | first `# heading`, "Podcast" | ID3 title / album tags; title also names the folder |
| `--speed 0.7-1.3` | 1.0 | tempo, pitch preserved |
| `--out PATH` | new folder in `podcasts/` | `.mp3`, `.m4a` or `.wav`; overwrites |
| `--power eco/fast` | eco | eco: low priority, 2 threads, GPU memory cap, rests between chunks |
| `--engine turbo/standard/multilingual` | turbo | **standard/multilingual peaked at ~18 GB on a 16 GB Mac**: only on explicit request, warn first |
| `--language xx` | en | multilingual only: ar da de el en es fi fr he hi it ja ko ms nl no pl pt ru sv sw tr zh |
| `--seed N` | 42 | a different take of the whole episode (and slightly different pauses) |
| `--pacing natural/fixed` | natural | natural pauses vary with punctuation and sentence length; fixed = old fixed gaps |
| `--qa` / `--no-qa` | on | transcribe every chunk with Whisper (CPU) and re-take misread ones |

## Script syntax (all optional)

- Blank line = paragraph pause · `# Heading` = read as its own phrase, with pauses around it · `## Heading` also starts a chapter · `- item` = one phrase each
- `[pause]` (1 s) or `[pause 2.5s]` · `---` = 1.5 s scene break · `<!-- notes -->` are never read
- Turbo expressive tags: `[laugh] [chuckle] [sigh] [gasp] [clear throat] [whispering] [dramatic] [happy] [sarcastic] [surprised]`
- Links keep their text and drop the URL. Bare URLs and code blocks are skipped.

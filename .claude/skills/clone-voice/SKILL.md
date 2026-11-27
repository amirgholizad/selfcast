---
name: clone-voice
description: One-time setup that clones the user's voice from a recording for this project's local podcast pipeline. Use when the user provides a voice recording or audio file to clone, asks to clone, re-clone, replace or add a voice profile, or types /clone-voice. After this, episodes are made with /podcast and no re-cloning is needed.
argument-hint: "[path to recording] [profile name]"
---

# Clone a voice (one-time setup)

Run every command from the project root (the folder containing `pyproject.toml`).
Cloning extracts the voice once and saves it to `voices/<name>/`; every later podcast reuses it.

## Steps

1. **Get the recording.** Take the path from the arguments or the message. Any format ffmpeg
   reads works: `.m4a` (Voice Memos), `.mp3`, `.wav`, `.aiff`, `.mov`/`.mp4` (audio is taken
   from video). Suggest keeping it in `recordings/`. If the user has no recording yet, give
   them the recording tips below and the reading passage in `recordings/README.md`, then stop.

2. **Consent.** This pipeline is for cloning the user's own voice. If the recording is someone
   else's, continue only if the user says that person agreed to be cloned. Otherwise, decline.

3. **Pick a name.** Default `me`. Run `uv run selfcast voices`. If the name exists,
   ask before replacing it: `--force` deletes the old profile.

4. **Clone** (~30-60 s; the very first run on a new machine also downloads ~3 GB of model weights):

   ```bash
   uv run selfcast clone --recording "<path>" --name me
   ```

   Optional: `--display-name "Their Name"` (written as the artist tag on episodes),
   `--default` (make it the default voice when several exist). Keep the default eco power mode.
   For recordings longer than ~30 s, add `--ref-start auto`: Turbo takes its delivery from the first
   10-15 s of the reference, and auto mode puts the best 15 s first instead of a stiff intro (the
   choice is printed; report the chosen start time). `--ref-start SECONDS` picks a start by hand.

5. **Report** from the final `RESULT {...}` line:
   - seconds of usable speech, and any `warnings` (noise, clipping, too short);
   - `speaker_similarity`: **≥ 0.90** strong match · **0.80-0.90** acceptable, judge by ear ·
     **< 0.80** likely off, re-record. (Measured during development: same-voice clones 0.92-0.95, other voices 0.58-0.84.)
   - the preview path. Offer to play it: `afplay voices/<name>/preview.mp3`.

6. **Ask the user to listen.** If it doesn't sound like them, go through the tips below and
   re-run with a better recording plus `--force`.

## Recording tips

- 30-60 s of continuous speech, one speaker, no music or TV in the background.
- A quiet room with soft furnishings (less echo). A phone or laptop mic is fine at ~20 cm, at a steady distance.
- Speak the way you want the podcast to sound. The clone copies energy and pace as well as timbre.
- Start talking right away, with no long pauses. Don't whisper or shout, and keep it from clipping.

## What gets stored (biometric data: keep private, never commit or upload)

`voices/<name>/`: `source.<ext>` (copy of the original), `reference.wav` (cleaned clip),
`conds-turbo.pt` (the clone itself), `preview.mp3`, `profile.json`. Clones for other engines
(`conds-standard.pt`, `conds-multilingual.pt`) are created automatically the first time they're used.

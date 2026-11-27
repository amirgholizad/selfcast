# Selfcast

Local voice cloning and text-to-podcast, built on Chatterbox TTS (`chatterbox-tts`, Resemble AI,
MIT). Developed and tested on a MacBook Air M4 / 16 GB. Everything runs offline once the model
weights are cached in `~/.cache/huggingface`.

- Skills: `/clone-voice` (one-time: recording → `voices/<name>/`) and `/podcast` (script → `podcasts/`).
- CLI: `uv run selfcast {clone,podcast,voices}`. Every command ends with a `RESULT {json}` line,
  which the skills parse, so keep it backwards compatible.
- Code: `selfcast/text.py` (script parsing and chunking), `selfcast/engine.py` (model loading,
  cached voice conditioning, synthesis, power modes), `selfcast/audio.py` (ffmpeg I/O, loudness),
  `selfcast/cli.py`, `selfcast/voices.py`, `selfcast/config.py`.

## Constraints

- **Memory and power:** the default is the `turbo` engine in `--power eco` (~7 GB process peak,
  flat over an episode; the end-of-render line prints it). Turbo's decoding is patched in
  `engine.py` to keep tensor shapes fixed: PyTorch on Apple GPUs caches a graph per shape and
  never frees it, which used to add ~5 GB over an episode. Keep that if you touch synthesis.
  `standard` and `multilingual` peaked at ~18 GB on a 16 GB Mac and pushed it into heavy swap:
  use them only on explicit request. Never run two renders at once.
- Python is pinned to 3.11 (chatterbox pins torch 2.6). `setuptools<81` is required because the
  watermarker (`resemble-perth`) imports `pkg_resources`.
- Output carries Resemble's inaudible Perth watermark. Keep it.
- `voices/`, `recordings/`, `podcasts/` and personal scripts hold biometric or private data.
  They are git-ignored: never commit, upload or share them.

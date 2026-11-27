# Contributing

Thanks for helping improve Selfcast. Bug fixes, platform reports, documentation and small,
well-scoped features are all welcome.

## Ground rules

- **Local only.** Selfcast must keep working offline once the model weights are downloaded.
  Please don't add cloud APIs, accounts, telemetry or anything that uploads audio or text.
- **Keep it light.** The default path (turbo engine, eco mode) should stay usable on a 16 GB
  laptop. Avoid heavy new dependencies, and mention memory use in your PR if you change
  anything in the synthesis path.
- **Never commit voice data.** No recordings, voice profiles (`voices/`), rendered episodes
  (`podcasts/`) or personal scripts. These are git-ignored; keep it that way. Use
  `scripts/example.md` or placeholder text in examples and bug reports.
- **Keep the watermark and the consent guidance.** Don't remove the Perth watermark, or the
  consent steps in the `/clone-voice` skill and the README.
- **One change per pull request.** Small, focused PRs are much easier to review and merge.
- **Match the existing style.** Python 3.11, 4-space indentation (see `.editorconfig`), type
  hints, and small functions. Script parsing goes in `selfcast/text.py`, audio processing in
  `selfcast/audio.py`, and model code in `selfcast/engine.py`.

## Before you open a pull request

```sh
uv sync
uv run pytest
uv run selfcast clone --recording <a recording of yourself> --name test --force
uv run selfcast podcast --file scripts/example.md --voice test
```

Then check:

- [ ] The example episode renders, and you listened to it end to end
- [ ] `.chunks.txt` timestamps line up with what you hear
- [ ] Re-running the same command renders nothing new (everything comes from the cache)
- [ ] `--redo` re-takes only the chunks you named
- [ ] Peak GPU memory (printed at the end) didn't grow noticeably in eco mode
- [ ] No errors or new warnings in the output

If you changed `selfcast/text.py` or `selfcast/normalize.py`, try a script with headings, lists,
`[pause 2s]`, author notes, expressive tags, numbers and acronyms, and check how it was split in
`.chunks.txt` and what was read at the end of `.script.md`. Add a test in `tests/` for new rules.

## Changing commands or the script format

If you add a CLI flag or script syntax, document it in the README tables **and** in
`.claude/skills/podcast/SKILL.md` (or `clone-voice/SKILL.md`), so the Claude Code skills know
about it too. Keep the `RESULT {...}` JSON line backwards compatible: the skills parse it.

## Reporting bugs and ideas

Use the [issue templates](https://github.com/amirgholizad/selfcast/issues/new/choose). For
bugs, include your OS, chip and RAM, the command you ran and the log output. Please **don't**
attach voice recordings or profiles.

## License

By contributing, you agree that your contributions are licensed under the project's [MIT License](LICENSE).

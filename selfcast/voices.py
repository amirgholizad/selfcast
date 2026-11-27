"""Voice profiles: voices/<name>/{reference.wav, profile.json, conds-<engine>.pt, preview.mp3}.

A profile is created once from a recording. The per-engine conditioning file (the "clone")
is computed from reference.wav the first time an engine needs it, then reused forever.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

from .config import VOICES_DIR

NAME_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,39}$")
DEFAULT_FILE = VOICES_DIR / "DEFAULT"


@dataclass(frozen=True)
class Voice:
    name: str

    @property
    def dir(self) -> Path:
        return VOICES_DIR / self.name

    @property
    def reference(self) -> Path:
        return self.dir / "reference.wav"

    @property
    def preview(self) -> Path:
        return self.dir / "preview.mp3"

    @property
    def meta_path(self) -> Path:
        return self.dir / "profile.json"

    def conds(self, engine: str) -> Path:
        return self.dir / f"conds-{engine}.pt"

    def exists(self) -> bool:
        return self.reference.exists()

    def meta(self) -> dict:
        try:
            return json.loads(self.meta_path.read_text())
        except (FileNotFoundError, json.JSONDecodeError):
            return {}

    def write_meta(self, meta: dict) -> None:
        self.meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n")

    def prepared_engines(self) -> list[str]:
        return sorted(p.stem.removeprefix("conds-") for p in self.dir.glob("conds-*.pt"))


def list_voices() -> list[Voice]:
    if not VOICES_DIR.exists():
        return []
    return sorted((Voice(p.name) for p in VOICES_DIR.iterdir() if (p / "reference.wav").exists()),
                  key=lambda v: v.name)


def default_voice_name() -> str | None:
    if DEFAULT_FILE.exists():
        name = DEFAULT_FILE.read_text().strip()
        if Voice(name).exists():
            return name
    voices = list_voices()
    return voices[0].name if len(voices) == 1 else None


def set_default(name: str) -> None:
    VOICES_DIR.mkdir(parents=True, exist_ok=True)
    DEFAULT_FILE.write_text(name + "\n")


def resolve(name: str | None) -> Voice:
    if name:
        voice = Voice(name)
        if not voice.exists():
            known = ", ".join(v.name for v in list_voices()) or "none yet"
            raise SystemExit(f"ERROR: voice '{name}' not found (available: {known}).")
        return voice
    default = default_voice_name()
    if default:
        return Voice(default)
    voices = list_voices()
    if not voices:
        raise SystemExit("ERROR: no voice profile yet. Clone one first:  uv run selfcast clone --recording <file> --name me")
    raise SystemExit("ERROR: several voices exist and none is the default; pass --voice "
                     f"({', '.join(v.name for v in voices)}).")

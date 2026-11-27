"""Turn a plain-text or Markdown script into speakable chunks and pauses.

Script conventions (all optional):
  - Blank line           -> paragraph break (longer pause)
  - "# Heading"          -> read as its own phrase, with a pause before and after
  - "- item" / "1. item" -> each item read as its own phrase
  - "---"                -> scene break (1.5 s pause)
  - [pause] / [pause 2s] -> explicit pause (default 1 s, max 10 s)
  - <!-- note -->        -> author notes, never read
  - [laugh] [chuckle] [sigh] ... -> expressive tags (turbo engine only; removed otherwise)
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

# Expressive tags the Chatterbox Turbo tokenizer understands natively.
TURBO_TAGS = frozenset({
    "laugh", "chuckle", "sigh", "gasp", "cough", "clear throat", "sniff", "groan", "shush",
    "angry", "fear", "surprised", "whispering", "advertisement", "dramatic", "narration",
    "crying", "happy", "sarcastic",
})

CHUNK_PAUSE = 0.3        # between chunks inside a paragraph (~ a sentence break)
PARAGRAPH_PAUSE = 0.75
HEADING_PAUSE = 1.0
SCENE_BREAK_PAUSE = 1.5
MAX_PAUSE = 10.0


@dataclass
class Speech:
    text: str
    index: int = 0  # 1-based chunk number, assigned after parsing


@dataclass
class Pause:
    seconds: float


_PAUSE_RE = re.compile(r"\[\s*pause(?:[\s:=]+(\d+(?:\.\d+)?)\s*(?:s|secs?|seconds?)?)?\s*\]", re.I)
_TAG_RE = re.compile(r"\[([^\[\]\n]{1,40})\]")
_SENT_END_RE = re.compile(r"[.!?…]+[\"'”’)\]]*(?=\s|$)")
_ABBREV_RE = re.compile(
    r"(?:\b(?:Mr|Mrs|Ms|Dr|Prof|Sr|Jr|St|Mt|vs|etc|approx|Inc|Ltd|Co|No|Vol|Fig|"
    r"Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)|\b[A-Z]|\be\.g|\bi\.e|\bU\.S|\bU\.K)\.$"
)
_HEADING_RE = re.compile(r"^#{1,6}\s+(.*?)\s*#*\s*$")
_LIST_RE = re.compile(r"^(?:[-*+•]|\d{1,3}[.)])\s+(.*)$")
_HR_RE = re.compile(r"^\s*(?:-{3,}|\*{3,}|_{3,})\s*$")
_WORD_RE = re.compile(r"[^\W_]+(?:['’][^\W_]+)*")
_CJK_RE = re.compile(r"[぀-ヿ㐀-䶿一-鿿가-힯]")


# --------------------------------------------------------------------------- cleaning

def _clean_inline(s: str) -> str:
    s = re.sub(r"!\[[^\]]*\]\([^)]*\)", "", s)                 # images
    s = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", s)              # [text](url) -> text
    s = re.sub(r"<https?://[^>]+>", "", s)
    s = re.sub(r"https?://\S+", "", s)                          # bare URLs
    s = re.sub(r"`([^`]*)`", r"\1", s)                          # inline code
    s = re.sub(r"(\*\*|__)(.+?)\1", r"\2", s)                   # bold
    s = re.sub(r"(?<![\w*])\*(?!\s)(.+?)(?<!\s)\*(?![\w*])", r"\1", s)  # *italic*
    s = re.sub(r"(?<!\w)_(?!\s)(.+?)(?<!\s)_(?!\w)", r"\1", s)  # _italic_
    s = re.sub(r"~~(.+?)~~", r"\1", s)
    s = re.sub(r"</?[a-zA-Z][^>]*>", "", s)                     # html tags
    s = s.replace(" ", " ")
    s = re.sub(r"\s+[—–-]{1,2}\s+|—|--", ", ", s)               # dashes -> spoken pause
    return s


def _tidy(s: str) -> str:
    s = re.sub(r"\s+", " ", s).strip()
    s = re.sub(r"\s+([,.;:!?])", r"\1", s)
    s = re.sub(r"([,;:])(?:\s*[,;:])+", r"\1", s)
    s = re.sub(r"([.!?])\s*,", r"\1", s)
    s = re.sub(r"^[,;:.\s]+", "", s)
    return s


def _finish(item: str) -> str:
    """End a heading / list item with punctuation so it is read as its own phrase."""
    item = item.rstrip()
    core = re.sub(r"[*_`~\]\)]+$", "", item)
    if core and core[-1] not in ".!?…,;:":
        item += "."
    return item


def _apply_tags(s: str, engine: str, unknown: set[str]) -> str:
    def repl(m: re.Match) -> str:
        tag = " ".join(m.group(1).lower().split())
        if tag in TURBO_TAGS:
            return f" [{tag}] " if engine == "turbo" else " "
        unknown.add(m.group(0))
        return " "
    return _TAG_RE.sub(repl, s)


# --------------------------------------------------------------------------- chunking

def _split_sentences(s: str) -> list[str]:
    out, start = [], 0
    for m in _SENT_END_RE.finditer(s):
        seg = s[start:m.end()]
        if m.end() < len(s) and _ABBREV_RE.search(seg):
            continue  # "Dr." / "e.g." / initials: not a sentence end
        out.append(seg.strip())
        start = m.end()
    if s[start:].strip():
        out.append(s[start:].strip())
    return [x for x in out if x]


def _greedy(pieces: list[str], max_chars: int) -> list[str]:
    out, cur = [], ""
    for p in pieces:
        if cur and len(cur) + 1 + len(p) > max_chars:
            out.append(cur)
            cur = p
        else:
            cur = f"{cur} {p}" if cur else p
    if cur:
        out.append(cur)
    return out


def _split_long(sentence: str, max_chars: int) -> list[str]:
    if len(sentence) <= max_chars:
        return [sentence]
    pieces = []
    for clause in re.split(r"(?<=[,;:])\s+", sentence):
        pieces.extend([clause] if len(clause) <= max_chars else _greedy(clause.split(), max_chars))
    return _greedy(pieces, max_chars)


def _emit_speech(part: str, items: list, engine: str, max_chars: int) -> None:
    if not _WORD_RE.search(_TAG_RE.sub(" ", part)) and not (engine == "turbo" and _TAG_RE.search(part)):
        return
    pieces = [p for s in _split_sentences(part) for p in _split_long(s, max_chars)]
    items.extend(Speech(chunk) for chunk in _greedy(pieces, max_chars))


def _emit(text: str, items: list, engine: str, max_chars: int, unknown: set[str], lexicon=None) -> None:
    parts = _PAUSE_RE.split(_clean_inline(text))
    # re.split with one group -> [text, seconds|None, text, seconds|None, ...]
    for i, part in enumerate(parts):
        if i % 2 == 1:
            items.append(Pause(min(float(part) if part else 1.0, MAX_PAUSE)))
            continue
        part = _tidy(_apply_tags(part or "", engine, unknown))
        if lexicon is None:
            _emit_speech(part, items, engine, max_chars)
            continue
        from .normalize import normalize

        # normalization may add pauses of its own (after a colon); split on them too
        for k, piece in enumerate(_PAUSE_RE.split(normalize(part, lexicon))):
            if k % 2 == 1:
                items.append(Pause(float(piece)))
            else:
                _emit_speech(_tidy(piece or ""), items, engine, max_chars)


def parse_script(raw: str, engine: str = "turbo", max_chars: int = 280,
                 lexicon=None) -> tuple[list, list[str]]:
    """Return (items, notes): items is an ordered list of Speech / Pause. With a `lexicon`
    (see normalize.py), the text is normalized to exactly what the voice model should read."""
    notes: list[str] = []
    text = raw.replace("\r\n", "\n").replace("\r", "\n").replace("﻿", "")
    text = re.sub(r"\A---\n.*?\n---\n", "", text, flags=re.S)   # YAML front matter
    text = re.sub(r"<!--.*?-->", "", text, flags=re.S)          # author notes
    if "```" in text:
        text = re.sub(r"```.*?(?:```|\Z)", "", text, flags=re.S)
        notes.append("Code blocks were skipped (not read aloud).")

    items: list = []
    unknown: set[str] = set()
    para: list[str] = []

    def flush() -> None:
        if para:
            _emit(" ".join(para), items, engine, max_chars, unknown, lexicon)
            items.append(Pause(PARAGRAPH_PAUSE))
            para.clear()

    for line in text.split("\n"):
        s = line.strip()
        if not s:
            flush()
            continue
        if _HR_RE.match(s):
            flush()
            items.append(Pause(SCENE_BREAK_PAUSE))
            continue
        if m := _HEADING_RE.match(s):
            flush()
            items.append(Pause(HEADING_PAUSE))
            _emit(_finish(m.group(1)), items, engine, max_chars, unknown, lexicon)
            items.append(Pause(HEADING_PAUSE))
            continue
        s = re.sub(r"^>\s?", "", s)
        if m := _LIST_RE.match(s):
            para.append(_finish(m.group(1)))
            continue
        para.append(s)
    flush()

    # Merge back-to-back pauses (keep the longest) and drop leading/trailing ones.
    cleaned: list = []
    for it in items:
        if isinstance(it, Pause):
            if not cleaned:
                continue
            if isinstance(cleaned[-1], Pause):
                cleaned[-1] = Pause(max(cleaned[-1].seconds, it.seconds))
                continue
        cleaned.append(it)
    while cleaned and isinstance(cleaned[-1], Pause):
        cleaned.pop()

    n = 0
    for it in cleaned:
        if isinstance(it, Speech):
            n += 1
            it.index = n
    if lexicon is not None:
        from .normalize import repeated_words

        repeats = sorted({r for it in cleaned if isinstance(it, Speech) for r in repeated_words(it.text)})
        if repeats:
            notes.append("Immediately repeated words can come out once: " + ", ".join(f'"{r}"' for r in repeats[:8])
                         + ". If one goes missing, rephrase or add a comma.")
    if unknown:
        notes.append("Removed bracketed text that isn't a supported tag: " + ", ".join(sorted(unknown)[:10]))
    if engine != "turbo" and any(f"[{t}]" in raw.lower() for t in TURBO_TAGS):
        notes.append("Expressive tags like [laugh] only work with the turbo engine; they were removed.")
    return cleaned, notes


# --------------------------------------------------------------------------- helpers

def word_count(text: str) -> float:
    plain = _TAG_RE.sub(" ", text)
    cjk = len(_CJK_RE.findall(plain))
    return len(_WORD_RE.findall(_CJK_RE.sub(" ", plain))) + 0.6 * cjk


def speech_bounds(text: str) -> tuple[float, float]:
    """Plausible (min, max) seconds for this chunk. Outside it, the take is probably glitched
    (truncated, or rambling/hallucinated audio)."""
    units = word_count(text)
    tags = len(_TAG_RE.findall(text))
    return units / 5.0, units / 1.5 + 1.5 + 1.5 * tags


def first_heading(raw: str) -> str | None:
    m = re.search(r"^#{1,2}\s+(.+?)\s*#*\s*$", raw, re.M)
    return _tidy(_clean_inline(m.group(1))) if m else None


def slugify(title: str, max_len: int = 50) -> str:
    s = unicodedata.normalize("NFKD", title).encode("ascii", "ignore").decode()
    s = re.sub(r"[^a-zA-Z0-9]+", "-", s).strip("-").lower()
    return s[:max_len].rstrip("-") or "episode"

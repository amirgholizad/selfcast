"""Text normalization before synthesis: turn script text into exactly what the voice model reads.

Applied to each paragraph after Markdown cleanup, in this order:
  1. spelled-out letters are joined back into the acronym ("C S S" -> "CSS"): Chatterbox swallows
     an immediately repeated letter, and reads common acronyms well when they're written as such
  2. the pronunciation file ("say" entries) replaces whole words
  3. numbers, money, percentages, ordinals, years and times are written out (inflect, MIT)
  4. colons and mid-sentence ellipses become a sentence break plus a short pause; the model's own
     text cleanup would otherwise turn them into commas and rush through them
"""
from __future__ import annotations

import re
import tomllib
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

DEFAULT_LEXICON = Path(__file__).with_name("pronunciations.toml")


# --------------------------------------------------------------------------- pronunciation file

@dataclass
class Lexicon:
    """Entries from pronunciations.toml: what to say instead of a word, and how Whisper may write it."""
    say: dict[str, str] = field(default_factory=dict)
    heard: dict[str, list[str]] = field(default_factory=dict)

    def accepted_variants(self) -> dict[str, list[str]]:
        """For the QA check: every way a listed word may legitimately come back from Whisper."""
        keys = set(self.say) | set(self.heard)
        return {k: [v for v in (self.say.get(k), *self.heard.get(k, [])) if v] for k in keys}


def _read_lexicon(path: Path, lex: Lexicon) -> None:
    data = tomllib.loads(path.read_text(encoding="utf-8"))
    for word, entry in data.items():
        if isinstance(entry, str):
            entry = {"say": entry}
        if not isinstance(entry, dict):
            raise ValueError(f"{path}: entry '{word}' should be a string or a table with say/heard")
        if entry.get("say"):
            lex.say[word] = str(entry["say"])
        else:
            lex.say.pop(word, None)  # a user entry with no "say" turns a default respelling off
        if "heard" in entry:
            lex.heard[word] = [str(h) for h in entry["heard"]]


def load_lexicon(*paths: Path | None) -> Lexicon:
    """The shipped defaults, then each existing file in `paths` (later entries win)."""
    lex = Lexicon()
    for path in (DEFAULT_LEXICON, *paths):
        if path is not None and Path(path).is_file():
            _read_lexicon(Path(path), lex)
    return lex


# --------------------------------------------------------------------------- letters and words

# Single capital letters written out one by one, with one consistent separator: "C S S",
# "C.S.S." / "C. S. S.", "H-T-M-L", and "C, S, S" (three or more; "A, B or C" and "A, B" read
# as lists and are left alone). Mixed separators ("vitamin D. A study") are not a run.
_LETTER_RUN_RE = re.compile(
    r"(?<![\w.])(?:[A-Z]\.(?: ?[A-Z]\.)+|[A-Z](?: [A-Z])+(?![\w])|[A-Z](?:-[A-Z])+(?![\w-])"
    r"|[A-Z](?:, [A-Z]){2,}+(?![\w]|,? (?:and|or) ))")


def join_letter_runs(text: str) -> str:
    def repl(m: re.Match) -> str:
        letters = re.sub(r"[^A-Z]", "", m.group(0))
        if re.fullmatch(r"[IA]+", letters):
            return m.group(0)  # "I A" is more likely two words than an acronym
        ends_sentence = m.group(0).endswith(".") and re.match(r"\s+[A-Z]|\s*$", text[m.end():])
        return letters + ("." if ends_sentence else "")
    return _LETTER_RUN_RE.sub(repl, text)


@lru_cache(maxsize=32)
def _say_pattern(keys: tuple[str, ...]) -> re.Pattern | None:
    if not keys:
        return None
    alts = "|".join(re.escape(k) for k in sorted(keys, key=len, reverse=True))
    return re.compile(rf"(?<![\w+#])(?:{alts})(?![\w+#])")


def apply_lexicon(text: str, lex: Lexicon) -> str:
    rx = _say_pattern(tuple(lex.say))
    return rx.sub(lambda m: lex.say[m.group(0)], text) if rx else text


# --------------------------------------------------------------------------- numbers

@lru_cache(maxsize=1)
def _inflect():
    import inflect

    return inflect.engine()


def _words(n: int | str) -> str:
    """Cardinal number in words, no commas: 120 -> 'one hundred and twenty', '2.5' -> 'two point five'."""
    return _inflect().number_to_words(str(n), group=0).replace(",", "")


def _year(n: int) -> str:
    if 2000 <= n <= 2009:
        return _words(n).replace(" and ", " ")  # "two thousand five"
    hi, lo = divmod(n, 100)
    if lo == 0:
        return f"{_words(hi)} hundred"
    return f"{_words(hi)} {'oh ' if lo < 10 else ''}{_words(lo)}"


def _ordinal(n: int) -> str:
    return _inflect().number_to_words(_inflect().ordinal(n)).replace(",", "")


_NUM = r"\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?"
_SCALE = r"(?:\s+(?P<scale>thousand|million|billion|trillion))?"
_CURRENCY = {"$": ("dollar", "dollars", "cent", "cents"), "£": ("pound", "pounds", "penny", "pence"),
             "€": ("euro", "euros", "cent", "cents")}

_MONEY_RE = re.compile(rf"(?P<sym>[$£€])(?P<num>{_NUM}){_SCALE}(?![\w])")
_PERCENT_RE = re.compile(rf"(?<![\w.])(?P<num>{_NUM})\s?%")
_ORDINAL_RE = re.compile(r"(?<![\w.,])(\d{1,3}(?:,\d{3})+|\d+)(st|nd|rd|th)\b", re.I)
_DECADE_RE = re.compile(r"(?<![\w.])(1[1-9]|20)(\d)0['’]?s\b")
_TIME_RE = re.compile(r"(?<![\w.:])([01]?\d|2[0-3]):([0-5]\d)(?![\d:])(?:\s*([ap])\.?m\.?(?![\w]))?", re.I)
_RATIO_RE = re.compile(r"(?<![\w.:])(\d+):(\d+)(?![\d:])")
_YEAR_RE = re.compile(r"(?<![\w.,$£€])(1[1-9]\d\d|20\d\d)(?![\w%]|[.,]\d)")
_RANGE_RE = re.compile(r"(?<![\w.,$£€-])(\d{1,4})\s?[-–]\s?(\d{1,4})(?![\w.,%-])")
_NUMBER_RE = re.compile(rf"(?<![\w.]){'(?P<num>' + _NUM + ')'}(?![\w]|\.\d)")


def _plain(num: str) -> str:
    return _words(num.replace(",", ""))


def _money(m: re.Match) -> str:
    one, many, sub_one, sub_many = _CURRENCY[m.group("sym")]
    num, scale = m.group("num").replace(",", ""), m.group("scale")
    if scale:
        return f"{_words(num)} {scale} {many}"
    whole, _, cents = num.partition(".")
    out = f"{_words(whole)} {one if whole == '1' else many}"
    if cents and len(cents) == 2 and int(cents):
        out += f" and {_words(int(cents))} {sub_one if cents == '01' else sub_many}"
    elif cents:
        out = f"{_words(num)} {many}"
    return out


def _time(m: re.Match) -> str:
    h, mm, ampm = int(m.group(1)), int(m.group(2)), m.group(3)
    if mm == 0:
        out = _words(h) + ("" if ampm else " o'clock")
    else:
        out = f"{_words(h)} {'oh ' if mm < 10 else ''}{_words(mm)}"
    return out + (f" {ampm.lower()}.m." if ampm else "")


def _decade(m: re.Match) -> str:
    year = _year(int(f"{m.group(1)}{m.group(2)}0"))
    head, _, last = year.rpartition(" ")
    plural = _inflect().plural(last)
    return f"{head} {plural}".strip()


def _range(m: re.Match) -> str:
    """'10-12' -> '10 to 12' (then written out). Only when the second number is larger."""
    a, b = m.group(1), m.group(2)
    if int(b) > int(a) or (len(a) == 4 and len(b) == 2):  # "10-12", "2024-25"
        return f"{a} to {b}"
    return m.group(0)


def normalize_numbers(text: str) -> str:
    text = _MONEY_RE.sub(_money, text)
    text = _PERCENT_RE.sub(lambda m: f"{_plain(m.group('num'))} percent", text)
    text = _ORDINAL_RE.sub(lambda m: _ordinal(int(m.group(1).replace(",", ""))), text)
    text = _DECADE_RE.sub(_decade, text)
    text = _TIME_RE.sub(_time, text)
    text = _RATIO_RE.sub(lambda m: f"{_words(m.group(1))} to {_words(m.group(2))}", text)
    text = _RANGE_RE.sub(_range, text)
    text = _YEAR_RE.sub(lambda m: _year(int(m.group(1))), text)
    return _NUMBER_RE.sub(lambda m: _plain(m.group("num")), text)


# --------------------------------------------------------------------------- punctuation

# Silence after a colon or a mid-sentence ellipsis. A full stop alone gave only ~0.25 s there in
# testing (as a comma, ~0.17 s), which still sounded rushed; a full stop plus 0.6 s fixed it.
COLON_PAUSE = 0.6


def _sentence_break(m: re.Match) -> str:
    return f". [pause {COLON_PAUSE}s] " + m.group(1).upper()


def break_colons(text: str) -> str:
    """'students too: what any program can' -> 'students too. [pause] What any program can'.
    Runs after the numbers are written out, so times and ratios (10:30, 16:9) are already words."""
    return re.sub(r"(?<=\S):\s+(\w)", _sentence_break, text)


def break_ellipses(text: str) -> str:
    """'runs almost anywhere... and that matters' -> 'runs almost anywhere. [pause] And that matters'."""
    text = re.sub(r"(?:\.\.\.|…)\s+(\w)", _sentence_break, text)
    return re.sub(r"(?:\.\.\.|…)", ".", text)


# --------------------------------------------------------------------------- entry point

def normalize(text: str, lex: Lexicon) -> str:
    text = join_letter_runs(text)
    text = apply_lexicon(text, lex)
    text = re.sub(r"\s&\s", " and ", text)
    text = normalize_numbers(text)
    text = break_colons(text)
    text = break_ellipses(text)
    return text


_REPEAT_RE = re.compile(r"\b(\w+)[\s,]+\1\b", re.I)


_NUMBER_WORDS = frozenset("""zero oh one two three four five six seven eight nine ten eleven twelve
    thirteen fourteen fifteen sixteen seventeen eighteen nineteen twenty thirty forty fifty sixty
    seventy eighty ninety hundred thousand million billion""".split())


def repeated_words(text: str) -> list[str]:
    """Immediately repeated words in the script ("that that"): Chatterbox may swallow the second
    one. Number words are left out: the QA check covers them."""
    return [m.group(0) for m in _REPEAT_RE.finditer(text) if m.group(1).lower() not in _NUMBER_WORDS]

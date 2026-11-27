"""Whisper QA: transcribe each synthesized chunk and compare it with the text that was meant.

Uses openai/whisper-base.en through transformers (already installed with Chatterbox), on the
CPU so it never competes with the model for GPU memory. The expected text is never passed to
Whisper as a prompt: that would bias it towards hearing what it was told and hide the errors.
"""
from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass

import numpy as np

WHISPER_REPO = "openai/whisper-base.en"  # Apache-2.0, ~290 MB, downloaded on first use
WHISPER_PATTERNS = ["*.json", "model.safetensors", "merges.txt", "vocab.json"]
WHISPER_REQUIRED = ["config.json", "model.safetensors", "preprocessor_config.json",
                    "tokenizer.json", "normalizer.json"]
WHISPER_SR = 16000
WHISPER_WINDOW = 30.0  # seconds; longer audio goes through Whisper's long-form decoding

# Words Whisper writes differently from the script (it predates ChatGPT and hears "Chachi PT")
# come from the "heard" lists in pronunciations.toml (see normalize.py). Spelled-out letters
# ("C S S", "C.S.S.") need no entry: they're matched against the acronym automatically.

_LETTER_RUN_RE = re.compile(r"(?<![A-Za-z])[A-Z](?:(?:\.\s*|[\s,\-]+)[A-Z])+(?![A-Za-z])\.?")


def collapse_letter_runs(text: str) -> str:
    """'C S S', 'C. S. S.', 'C, S, S', 'H-T-M-L' -> 'CSS', 'HTML'. A swallowed letter
    ('CS' for 'C S S') then shows up as a word error instead of being hidden."""
    return _LETTER_RUN_RE.sub(lambda m: re.sub(r"[^A-Z]", "", m.group(0)), text)


def _token(canonical: str) -> str:
    return "v" + re.sub(r"[^a-z0-9]", "", canonical.lower().replace("+", "plus"))


def _variant_patterns(variants: dict[str, list[str]]) -> list[tuple[re.Pattern, str]]:
    forms = []
    for canonical, heard in variants.items():
        for form in {canonical, *heard}:
            words = collapse_letter_runs(form).split()
            body = r"[\s\-]*".join(re.escape(w) for w in words)
            forms.append((len(form), re.compile(rf"(?<![A-Za-z0-9]){body}(?![A-Za-z0-9+])", re.I),
                          _token(canonical)))
    return [(rx, tok) for _, rx, tok in sorted(forms, key=lambda f: -f[0])]


@dataclass
class Check:
    """How well one take matched its text."""
    wer: float
    errors: int
    words: int
    heard: str
    ok: bool
    key_errors: int = 0   # errors on acronyms, numbers or listed words: where TTS really slips
    longest_run: int = 0  # most adjacent wrong words: a slur ("fewerly") rather than one misheard word
    diff: str = ""        # e.g. 'css -> cs; asked -> ask'

    def as_dict(self) -> dict:
        return asdict(self)


def describe(qa: dict) -> str:
    detail = qa.get("diff") or f'heard: "{qa["heard"]}"'
    return f"WER {qa['wer']:.0%} ({qa['errors']}/{qa['words']} words): {detail}"


def align(ref: list[str], hyp: list[str]) -> list[tuple[str, int | None, int | None]]:
    """Word alignment as (op, ref index, hyp index), op in ok / sub / del / ins. A word split or
    joined differently ("web page" vs "webpage") counts as ok: a spelling difference, not a misread."""
    n, m = len(ref), len(hyp)
    d = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        d[i][0] = i
    for j in range(m + 1):
        d[0][j] = j

    def split(i: int, j: int) -> bool:  # ref[i-1] == hyp[j-2] + hyp[j-1]
        return j > 1 and ref[i - 1] == hyp[j - 2] + hyp[j - 1]

    def joined(i: int, j: int) -> bool:  # ref[i-2] + ref[i-1] == hyp[j-1]
        return i > 1 and ref[i - 2] + ref[i - 1] == hyp[j - 1]

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            best = min(d[i - 1][j] + 1, d[i][j - 1] + 1, d[i - 1][j - 1] + (ref[i - 1] != hyp[j - 1]))
            if split(i, j):
                best = min(best, d[i - 1][j - 2])
            if joined(i, j):
                best = min(best, d[i - 2][j - 1])
            d[i][j] = best

    ops, i, j = [], n, m
    while i > 0 or j > 0:
        if i and j and d[i][j] == d[i - 1][j - 1] + (ref[i - 1] != hyp[j - 1]):
            ops.append(("ok" if ref[i - 1] == hyp[j - 1] else "sub", i - 1, j - 1))
            i, j = i - 1, j - 1
        elif i and split(i, j) and d[i][j] == d[i - 1][j - 2]:
            ops.append(("ok", i - 1, j - 2))
            i, j = i - 1, j - 2
        elif j and joined(i, j) and d[i][j] == d[i - 2][j - 1]:
            ops += [("ok", i - 1, j - 1), ("ok", i - 2, j - 1)]
            i, j = i - 2, j - 1
        elif i and d[i][j] == d[i - 1][j] + 1:
            ops.append(("del", i - 1, None))
            i -= 1
        else:
            ops.append(("ins", None, j - 1))
            j -= 1
    return ops[::-1]


def is_acceptable(qa: dict, threshold: float) -> bool:
    """Whether a take passes. Whisper itself mishears the odd ordinary word ("assistants" ->
    "assistance"), so one isolated slip in a long chunk is tolerated. What is not: a wrong
    acronym, number or listed word ("C S S" -> "CS"), two or more wrong words in a row (a slur),
    or a word error rate above `threshold`. Very short chunks (a heading) get one error of slack."""
    errors, words = qa["errors"], qa["words"]
    if errors == 0:
        return True
    if qa.get("key_errors", 0) or qa.get("longest_run", errors) >= 2:
        return False
    if words <= 4:
        return errors <= 1
    return errors / words <= threshold


class Judge:
    """Transcribes takes with Whisper and scores them against the expected text."""

    def __init__(self, threshold: float, variants: dict[str, list[str]] | None = None, log=print):
        self.threshold = threshold
        self.model_name = WHISPER_REPO
        self.log = log
        self.seconds = 0.0  # total time spent transcribing
        merged = dict(variants or {})
        self._variants = _variant_patterns(merged)
        self._variant_tokens = {_token(k) for k in merged}
        self._model = self._processor = self._normalizer = None
        self._path = None

    # ------------------------------------------------------------------ loading
    def _files(self):
        if self._path is None:
            from .engine import checkpoint_dir

            self._path = checkpoint_dir(WHISPER_REPO, WHISPER_PATTERNS, WHISPER_REQUIRED, log=self.log,
                                        what="the Whisper QA model (~290 MB)")
        return self._path

    def _load(self) -> None:
        if self._model is not None:
            return
        from transformers import WhisperForConditionalGeneration, WhisperProcessor

        from .engine import _quiet

        with _quiet():
            self._processor = WhisperProcessor.from_pretrained(self._files())
            self._model = WhisperForConditionalGeneration.from_pretrained(self._files()).to("cpu").eval()

    def normalize(self, text: str) -> list[str]:
        if self._normalizer is None:
            from transformers.models.whisper.english_normalizer import EnglishTextNormalizer

            mapping = json.loads((self._files() / "normalizer.json").read_text(encoding="utf-8"))
            self._normalizer = EnglishTextNormalizer(mapping)
        text = collapse_letter_runs(text)
        for rx, tok in self._variants:
            text = rx.sub(f" {tok} ", text)
        # "$5" and "5 dollars" both normalize to "$5"; Whisper also carries a "$" over to nearby
        # numbers ("$2.5 million requests"), so compare amounts without their currency symbol.
        return [re.sub(r"^[$£€¢]", "", w) for w in self._normalizer(text).split()]

    # ------------------------------------------------------------------ use
    def transcribe(self, wav: np.ndarray, sr: int) -> str:
        import time

        import librosa
        import torch

        self._load()
        t0 = time.time()
        if sr != WHISPER_SR:
            wav = librosa.resample(wav.astype(np.float32), orig_sr=sr, target_sr=WHISPER_SR)
        long_form = len(wav) > WHISPER_WINDOW * WHISPER_SR
        kwargs = {"truncation": False, "padding": "longest", "return_attention_mask": True} if long_form else {}
        feats = self._processor.feature_extractor(wav, sampling_rate=WHISPER_SR, return_tensors="pt", **kwargs)
        gen = {"attention_mask": feats.attention_mask, "return_timestamps": True} if long_form else {}
        with torch.inference_mode():
            ids = self._model.generate(feats.input_features, max_new_tokens=220, **gen)
        text = self._processor.tokenizer.batch_decode(ids, skip_special_tokens=True)[0].strip()
        self.seconds += time.time() - t0
        return text

    def key_words(self, text: str) -> set[str]:
        """Normalized words that TTS tends to get wrong and Whisper rarely does: acronyms,
        numbers and words with an accepted-variants entry."""
        acronyms = re.findall(r"\b[A-Z][A-Za-z]*[A-Z]\b", collapse_letter_runs(text))
        return {w for a in acronyms for w in self.normalize(a)} | self._variant_tokens

    def words(self, wav: np.ndarray, sr: int) -> list[tuple[str, float, float]]:
        """Transcribe audio of any length into (word, start, end) in seconds, punctuation kept
        on the words. Uses Whisper's cross-attention alignment, good to ~0.1 s."""
        import time

        import librosa
        from transformers import pipeline

        self._load()
        t0 = time.time()
        if sr != WHISPER_SR:
            wav = librosa.resample(wav.astype(np.float32), orig_sr=sr, target_sr=WHISPER_SR)
        asr = pipeline("automatic-speech-recognition", model=self._model, tokenizer=self._processor.tokenizer,
                       feature_extractor=self._processor.feature_extractor, device="cpu")
        # No chunk_length_s: Whisper's own sequential long-form decoding. The chunked pipeline
        # duplicated stretches of text on a 2-minute episode.
        out = asr({"raw": wav, "sampling_rate": WHISPER_SR}, return_timestamps="word", batch_size=1)
        self.seconds += time.time() - t0
        words = []
        for ch in out.get("chunks", []):
            start, end = ch["timestamp"]
            if ch["text"].strip() and start is not None:
                words.append((ch["text"].strip(), float(start), float(end if end is not None else start)))
        return words

    def score(self, expected: str, heard: str) -> Check:
        ref, hyp = self.normalize(expected), self.normalize(heard)
        keys = self.key_words(expected)
        ops = align(ref, hyp)
        errors = sum(op != "ok" for op, _, _ in ops)
        key_errors = sum(op in ("sub", "del") and (ref[i] in keys or any(ch.isdigit() for ch in ref[i]))
                         for op, i, _ in ops)
        run = longest = 0
        for op, _, _ in ops:
            run = run + 1 if op != "ok" else 0
            longest = max(longest, run)
        diff = "; ".join(f"{ref[i] if i is not None else '+'} -> {hyp[j] if j is not None else '(missing)'}"
                         for op, i, j in ops if op != "ok")
        qa = {"wer": round(errors / max(len(ref), 1), 3), "errors": errors, "words": len(ref), "heard": heard,
              "key_errors": key_errors, "longest_run": longest, "diff": diff}
        return Check(ok=is_acceptable(qa, self.threshold), **qa)

    def check(self, expected: str, wav: np.ndarray, sr: int) -> Check:
        return self.score(expected, self.transcribe(wav, sr))

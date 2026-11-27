from pathlib import Path

import numpy as np
import pytest

from selfcast.audio import shorten_pauses
from selfcast.qa import speech_windows
from selfcast.text import script_sections


def tone(seconds, sr):
    t = np.arange(int(seconds * sr)) / sr
    return (0.3 * np.sin(2 * np.pi * 200 * t)).astype(np.float32)


def test_shorten_pauses_only_touches_long_ones():
    sr = 16000
    wav = np.concatenate([tone(1, sr), np.zeros(int(0.5 * sr)), tone(1, sr), np.zeros(3 * sr), tone(1, sr)])
    segments = [(0.0, 1.0), (1.5, 2.5), (5.5, 6.5)]
    out, stats = shorten_pauses(wav, sr, segments, max_pause=0.8, keep=0.6, edge=0.0)
    assert stats["shortened"] == 1                        # the 0.5 s pause is kept as it is
    assert abs(len(out) / sr - (3 + 0.5 + 0.6)) < 0.01    # the 3 s pause became 0.6 s


def test_speech_windows_cut_in_pauses():
    sr = 16000
    parts = []
    for _ in range(8):                                    # 8 x (9 s of sound + 1 s pause) = 80 s
        parts += [tone(9, sr), np.zeros(sr, dtype=np.float32)]
    wav = np.concatenate(parts)
    windows = speech_windows(wav, sr, max_seconds=28)
    assert all((b - a) / sr <= 28 for a, b in windows)
    assert windows[0][0] == 0 and windows[-1][1] == len(wav)
    for a, _ in windows[1:]:                              # every cut falls inside a pause
        assert np.abs(wav[a - 100:a + 100]).max() == 0


def test_script_sections():
    raw = "# Title\n\nIntro.\n\n## First part\n\nOne two three four.\n\n## Second\n\nFive six."
    assert script_sections(raw) == [("First part", "One two three four."), ("Second", "Five six.")]


WHISPER = Path.home() / ".cache/huggingface/hub/models--openai--whisper-base.en"


@pytest.mark.skipif(not WHISPER.exists(), reason="Whisper normalizer files not downloaded")
def test_align_sections_finds_heading_or_opening_text():
    from selfcast.qa import Judge, align_sections

    said = ("welcome to the show today we talk about tools physics meets finance during my masters "
            "i met a professor and then something else entirely the field is called econophysics").split()
    words = [(w, i * 0.5, i * 0.5 + 0.4) for i, w in enumerate(said)]
    found = align_sections(Judge(0.1), [("Physics meets finance", "During my master's, I met a professor"),
                                        ("Not read aloud", "The field is called econophysics.")], words)
    assert found[0]["matched"] == "heading" and found[0]["start"] == said.index("physics") * 0.5
    assert found[1]["matched"] == "text" and found[1]["start"] == said.index("field") * 0.5 - 0.5

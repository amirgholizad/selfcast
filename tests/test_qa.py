from pathlib import Path

import pytest

from selfcast.qa import align, collapse_letter_runs, is_acceptable


def ops(ref, hyp):
    return [op for op, _, _ in align(ref.split(), hyp.split())]


def test_align_counts_split_and_joined_words_as_ok():
    assert ops("every web page", "every webpage") == ["ok", "ok", "ok"]
    assert ops("a webpage here", "a web page here") == ["ok", "ok", "ok"]


def test_align_slur_is_a_run():
    assert ops("far fewer leave a project", "fewerly a project").count("ok") == 2


def test_collapse_letter_runs():
    assert collapse_letter_runs("Why do we need C S S? H.T.M.L.") == "Why do we need CSS? HTML"


@pytest.mark.parametrize("qa, ok", [
    ({"errors": 0, "words": 30}, True),
    ({"errors": 1, "words": 30, "key_errors": 0, "longest_run": 1}, True),    # one isolated slip
    ({"errors": 1, "words": 30, "key_errors": 1, "longest_run": 1}, False),   # CSS -> CS
    ({"errors": 3, "words": 30, "key_errors": 0, "longest_run": 3}, False),   # a slur
    ({"errors": 4, "words": 30, "key_errors": 0, "longest_run": 1}, False),   # too many slips
    ({"errors": 1, "words": 4, "key_errors": 0, "longest_run": 1}, True),     # heading slack
])
def test_is_acceptable(qa, ok):
    assert is_acceptable(qa, 0.1) is ok


WHISPER = Path.home() / ".cache/huggingface/hub/models--openai--whisper-base.en"


@pytest.mark.skipif(not WHISPER.exists(), reason="Whisper normalizer files not downloaded")
def test_score_against_known_transcripts():
    from selfcast.normalize import load_lexicon
    from selfcast.qa import Judge

    judge = Judge(0.1, variants=load_lexicon().accepted_variants())
    assert not judge.score("why do we need C S S at all?", "Why do we need CS at all?").ok
    assert judge.score("why do we need CSS at all?", "Why do we need C.S.S. at all?").ok
    assert judge.score("ask ChatGPT for help", "ask Chachi PT for help").ok
    assert judge.score("about five dollars and two point five million requests",
                       "about $5 and $2.5 million requests").ok
    assert judge.score("in twenty twenty-four", "in 2024").ok
    assert not judge.score("Far fewer leave a project because the tools", "Fewerly a project because the tools").ok

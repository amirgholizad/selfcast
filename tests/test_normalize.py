import pytest

from selfcast.normalize import (Lexicon, apply_lexicon, break_colons, break_ellipses, join_letter_runs,
                                load_lexicon, normalize, normalize_numbers, repeated_words)


@pytest.mark.parametrize("text, expected", [
    ("why do we need C S S at all?", "why do we need CSS at all?"),
    ("H T M L, C, S, S", "HTML, CSS"),
    ("H-T-M-L and C.S.S.", "HTML and CSS."),
    ("the U.S. economy", "the US economy"),
    ("in the U.S. The end", "in the US. The end"),
    ("vitamin D. A study", "vitamin D. A study"),     # mixed separators: not a run
    ("plan A, B or C", "plan A, B or C"),             # a list
    ("A, B, C, D and E", "A, B, C, D and E"),
    ("Then I said", "Then I said"),
])
def test_join_letter_runs(text, expected):
    assert join_letter_runs(text) == expected


@pytest.mark.parametrize("text, expected", [
    ("I watched 120 students", "I watched one hundred and twenty students"),
    ("about $5 a month", "about five dollars a month"),
    ("$2.5 million", "two point five million dollars"),
    ("$5.50", "five dollars and fifty cents"),
    ("43% of sites", "forty-three percent of sites"),
    ("released in 1995", "released in nineteen ninety-five"),
    ("in 2005", "in two thousand five"),
    ("in 1900", "in nineteen hundred"),
    ("the 1st and 23rd", "the first and twenty-third"),
    ("the 1,000th visitor", "the one thousandth visitor"),
    ("the 1960s", "the nineteen sixties"),
    ("at 3:30 pm", "at three thirty p.m."),
    ("at 10:00", "at ten o'clock"),
    ("a 16:9 screen", "a sixteen to nine screen"),
    ("pages 10-12", "pages ten to twelve"),
    ("1,000 people", "one thousand people"),
    ("a 4K M4 Mac", "a 4K M4 Mac"),                   # numbers glued to letters stay
])
def test_numbers(text, expected):
    assert normalize_numbers(text) == expected


def test_colons_and_ellipses():
    assert break_colons("The lesson is simple: boring tools last.") == "The lesson is simple. [pause 0.6s] Boring tools last."
    assert break_ellipses("anywhere... and that matters...") == "anywhere. [pause 0.6s] And that matters."


def test_lexicon_whole_words_case_sensitive():
    lex = Lexicon(say={"US": "U S", "C++": "C plus plus"})
    assert apply_lexicon("the US and us, C++ and C", lex) == "the U S and us, C plus plus and C"


def test_lexicon_files_merge(tmp_path):
    user = tmp_path / "pronunciations.toml"
    user.write_text('Nguyen = "Win"\n[ChatGPT]\nheard = ["Chat Gee Pee Tee"]\n', encoding="utf-8")
    lex = load_lexicon(user)
    assert lex.say["Nguyen"] == "Win"
    assert lex.heard["ChatGPT"] == ["Chat Gee Pee Tee"]   # user entry replaces the default list
    assert "C++" in lex.accepted_variants()                 # defaults still there


def test_normalize_end_to_end():
    out = normalize("Learn C++ & H T M L too: what any program can do in 2.5 hours...", load_lexicon())
    assert out == "Learn C++ and HTML too. [pause 0.6s] What any program can do in two point five hours."


def test_repeated_words_ignores_numbers():
    assert repeated_words("I think that that is fine") == ["that that"]
    assert repeated_words("in twenty twenty-four") == []

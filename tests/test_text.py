from selfcast.text import GAP, TRIM_PADDING, Pause, Speech, parse_script

SCRIPT = """# Title

## Section one

A short sentence. Another short sentence that keeps going for a while, so it gets split
somewhere along the way because it is really far too long for one chunk of speech, isn't it?

[pause 2s]

Second paragraph here.

---

## Section two

Last words.
"""


def pauses(items):
    return [it for it in items if isinstance(it, Pause)]


def test_fixed_pacing_keeps_old_lengths():
    items, _ = parse_script(SCRIPT, max_chars=80)
    assert {round(p.seconds, 2) for p in pauses(items)} <= {0.75, 1.0, 1.5, 2.0}


def test_natural_pacing_is_deterministic_and_varied():
    a, _ = parse_script(SCRIPT, max_chars=80, pacing_seed=1)
    b, _ = parse_script(SCRIPT, max_chars=80, pacing_seed=1)
    c, _ = parse_script(SCRIPT, max_chars=80, pacing_seed=2)
    assert [p.seconds for p in pauses(a)] == [p.seconds for p in pauses(b)]
    assert [p.seconds for p in pauses(a)] != [p.seconds for p in pauses(c)]


def test_natural_pacing_rules():
    items, _ = parse_script(SCRIPT, max_chars=80, pacing_seed=1)
    # every two chunks in a row get a join
    for x, y in zip(items, items[1:]):
        assert not (isinstance(x, Speech) and isinstance(y, Speech))
    kinds = [(p.kind, p.seconds) for p in pauses(items)]
    assert ("explicit", 2.0) in kinds and ("scene", 1.5) in kinds   # written pauses are kept
    headings = [it for it in items if isinstance(it, Speech) and it.heading]
    assert [h.heading for h in headings] == [1, 2, 2]
    # a section starts with a longer pause than the one after its heading
    i = items.index(headings[1])
    assert items[i - 1].seconds > items[i + 1].seconds
    # a mid-sentence split is a shorter join than a sentence end
    joins = [(prev.text, p.seconds) for prev, p in zip(items, items[1:]) if isinstance(p, Pause) and p.kind == "chunk"]
    clause = [s for t, s in joins if t.rstrip().endswith(",")]
    assert clause and max(clause) < (GAP["sentence"] - TRIM_PADDING) * 0.85

import numpy as np

from selfcast.audio import best_window_first, condense_pauses, rank_windows, to_source_time


def test_best_window_first_order_and_length():
    sr = 100
    speech = np.arange(60 * sr, dtype=np.float32)          # sample value = position
    out = best_window_first(speech, sr, start=30.0, window=15.0, total=40.0)
    assert len(out) == 40 * sr
    assert out[0] == 30 * sr                               # starts with the chosen window
    assert out[15 * sr + 5] == 45 * sr + 5 + 2             # then what followed it (shifted by the 20 ms crossfade)
    assert out[-1] < 30 * sr                               # then what came before


def test_rank_windows_prefers_typical_pace_and_lively_pitch():
    feats = [{"snr_db": 40, "rate": 5.0, "pitch_var": 3.0},   # flat
             {"snr_db": 40, "rate": 5.0, "pitch_var": 4.5},   # lively
             {"snr_db": 40, "rate": 8.0, "pitch_var": 4.5}]   # rushed
    scores = rank_windows(feats, [0.98, 0.98, 0.98])
    assert scores.index(max(scores)) == 1


def test_condense_map_points_back_to_the_recording():
    sr = 16000
    t = np.arange(sr) / sr
    tone = (0.5 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
    wav = np.concatenate([np.zeros(sr), tone, np.zeros(3 * sr), tone]).astype(np.float32)   # 1 s gap, 3 s gap
    out, seg_map = condense_pauses(wav, sr, return_map=True)
    assert len(out) < len(wav) - 2 * sr                    # the 3 s pause was shortened
    second = len(out) / sr - 0.6                           # inside the second tone in the output
    assert 5.0 < to_source_time(second, seg_map, sr) < 6.0  # ...is inside it in the recording too

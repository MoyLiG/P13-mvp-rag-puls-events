"""summarize_delta : compare deux séries de hits par question (OFF vs ON)."""
from pulsevents_mvp.eval_utils import summarize_delta


def test_gain_net():
    off = [False, False, True, True]
    on = [True, True, True, True]
    d = summarize_delta(off, on)
    assert d["recall_off"] == 0.5
    assert d["recall_on"] == 1.0
    assert d["gain"] == 0.5
    assert d["improved"] == 2
    assert d["degraded"] == 0
    assert d["unchanged"] == 2
    assert d["n"] == 4


def test_echange_meme_recall():
    off = [True, False]
    on = [False, True]
    d = summarize_delta(off, on)
    assert d["gain"] == 0.0
    assert d["improved"] == 1
    assert d["degraded"] == 1
    assert d["unchanged"] == 0


def test_degradation():
    off = [True, True]
    on = [True, False]
    d = summarize_delta(off, on)
    assert d["gain"] == -0.5
    assert d["improved"] == 0
    assert d["degraded"] == 1


def test_longueurs_differentes_leve():
    import pytest
    with pytest.raises(ValueError):
        summarize_delta([True], [True, False])

from pulsevents_mvp import loader


def test_load_if_needed_no_op_si_deja_peuple(monkeypatch):
    appels = {"build": 0}
    monkeypatch.setattr(loader, "count_embeddings", lambda s: 2302)
    monkeypatch.setattr(loader, "build_index",
                        lambda *a, **k: appels.__setitem__("build", appels["build"] + 1))
    settings = object()  # non utilisé quand déjà peuplé
    assert loader.load_if_needed(settings) == 2302
    assert appels["build"] == 0  # build_index jamais appelé

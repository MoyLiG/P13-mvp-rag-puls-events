"""fr_preprocess : tokeniseur FR pour BM25 (dé-accent + lowercase + stopwords + stem).

On teste des INVARIANTS (deux formes -> même sortie), pas la valeur exacte du
stem (qui dépend de Snowball et serait fragile).
"""
from pulsevents_mvp.tokenizer_fr import fr_preprocess


def test_singulier_pluriel_meme_radical():
    assert fr_preprocess("gratuits") == fr_preprocess("gratuit")


def test_accents_ignores():
    assert fr_preprocess("évènement") == fr_preprocess("evenement")


def test_ponctuation_et_casse():
    assert fr_preprocess("Nantes,") == fr_preprocess("nantes")


def test_stopwords_retires():
    # "le"/"à"/"les" doivent disparaître -> même sortie que sans eux
    assert fr_preprocess("le concert à Nantes") == fr_preprocess("concert Nantes")


def test_renvoie_liste_de_tokens():
    out = fr_preprocess("concerts gratuits à Nantes")
    assert isinstance(out, list)
    assert all(isinstance(t, str) for t in out)
    assert out  # non vide


def test_symetrie_query_doc():
    # la même chaîne donne toujours la même sortie (doc et requête traités pareil)
    assert fr_preprocess("Expositions pour enfants") == fr_preprocess("Expositions pour enfants")


def test_chaine_vide_renvoie_liste_vide():
    assert fr_preprocess("") == []
    assert fr_preprocess("   ,;!  ") == []

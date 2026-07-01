"""Tests fumée (fonctions pures, sans infra)."""
import pandas as pd

from pulsevents_mvp.preprocessing import clean_html, normalize_text, to_documents


def test_clean_html():
    assert clean_html("<p>Bonjour <b>Nantes</b> !</p>") == "Bonjour Nantes !"


def test_normalize_text():
    assert normalize_text("a\n\n  b ") == "a b"


def test_to_documents_minimal():
    df = pd.DataFrame([{
        "uid": "X1",
        "title_fr": "Concert Jazz",
        "longdescription_fr": "Un grand concert de jazz à Nantes avec des invités.",
        "daterange_fr": "le 5 juin",
        "firstdate_begin": "2026-06-05",
        "lastdate_end": "2026-06-05",
        "location_name": "Stereolux",
        "location_city": "Nantes",
        "location_department": "Loire-Atlantique",
        "location_region": "Pays de la Loire",
        "keywords_fr": ["jazz"],
        "canonicalurl": "http://x",
    }])
    docs = to_documents(df, min_description_length=10)
    assert len(docs) == 1
    assert docs[0].metadata["uid"] == "X1"
    assert docs[0].metadata["city"] == "Nantes"
    assert "Titre : Concert Jazz" in docs[0].page_content


def test_to_documents_skips_short_description():
    df = pd.DataFrame([{
        "uid": "X2", "title_fr": "Bref", "longdescription_fr": "court",
        "location_city": "Nantes", "location_region": "Pays de la Loire",
    }])
    assert to_documents(df, min_description_length=30) == []

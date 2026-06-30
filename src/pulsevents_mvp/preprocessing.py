"""Pré-processing des événements Open Agenda (repris du POC P11).

Construit les Documents LangChain + métadonnées. Inchangé vs P11 pour
garantir que les textes de chunks correspondent au cache d'embeddings réutilisé.
"""
from __future__ import annotations

import logging
import re
import unicodedata
import warnings
from typing import Optional

import pandas as pd
from bs4 import BeautifulSoup, MarkupResemblesLocatorWarning
from langchain_core.documents import Document

logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore", category=MarkupResemblesLocatorWarning)


def clean_html(text: Optional[str]) -> str:
    if not text:
        return ""
    return BeautifulSoup(text, "html.parser").get_text(separator=" ", strip=True)


def normalize_text(text: str) -> str:
    if not text:
        return ""
    text = unicodedata.normalize("NFKC", text)
    return re.sub(r"\s+", " ", text).strip()


def _as_list(value) -> list:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return list(value)
    if hasattr(value, "tolist"):
        return list(value.tolist())
    try:
        if pd.isna(value):
            return []
    except (TypeError, ValueError):
        pass
    return [value]


def build_document_text(row: pd.Series) -> str:
    title = normalize_text(row.get("title_fr") or "")
    description = normalize_text(
        clean_html(row.get("longdescription_fr") or row.get("description_fr") or "")
    )
    daterange = normalize_text(row.get("daterange_fr") or "")
    fdb = str(row.get("firstdate_begin") or "")[:10]
    lde = str(row.get("lastdate_end") or "")[:10]
    place = normalize_text(row.get("location_name") or "")
    city = normalize_text(row.get("location_city") or "")
    dept = normalize_text(row.get("location_department") or "")
    keywords = _as_list(row.get("keywords_fr"))
    keywords_str = ", ".join(normalize_text(str(k)) for k in keywords if k)

    parts = [
        f"Titre : {title}",
        f"Quand : {daterange}" if daterange else "",
        f"Date debut (ISO) : {fdb}" if fdb else "",
        f"Date fin (ISO) : {lde}" if lde else "",
        f"Lieu : {place}" + (f" ({city}, {dept})" if city or dept else ""),
        f"Mots-cles : {keywords_str}" if keywords_str else "",
        "Description :",
        description,
    ]
    return "\n".join(p for p in parts if p)


def build_metadata(row: pd.Series) -> dict:
    return {
        "uid": row.get("uid"),
        "title": normalize_text(row.get("title_fr") or ""),
        "city": normalize_text(row.get("location_city") or ""),
        "department": normalize_text(row.get("location_department") or ""),
        "region": normalize_text(row.get("location_region") or ""),
        "firstdate_begin": str(row.get("firstdate_begin") or ""),
        "lastdate_end": str(row.get("lastdate_end") or ""),
        "url": row.get("canonicalurl") or "",
        "daterange": normalize_text(row.get("daterange_fr") or ""),
    }


def to_documents(df: pd.DataFrame, min_description_length: int = 30) -> list[Document]:
    documents: list[Document] = []
    skipped = 0
    for _, row in df.iterrows():
        text = build_document_text(row)
        desc_only = normalize_text(
            clean_html(row.get("longdescription_fr") or row.get("description_fr") or "")
        )
        if len(desc_only) < min_description_length:
            skipped += 1
            continue
        documents.append(Document(page_content=text, metadata=build_metadata(row)))
    logger.info("Documents : %d retenus, %d ignorés (desc. courte)", len(documents), skipped)
    return documents

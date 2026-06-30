#!/usr/bin/env python
"""Charge l'index pgvector depuis les données P11 (one-shot, idempotent).

Lancement : docker compose run --rm api python scripts/load_pgvector.py
"""
from __future__ import annotations

import logging

from pulsevents_mvp.config import load_settings
from pulsevents_mvp.loader import load_if_needed

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")


def main() -> None:
    load_if_needed(load_settings())


if __name__ == "__main__":
    main()

#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Pipeline de génération des livrables P13, de bout en bout :
  1. schéma d'architecture (PNG)
  2. livrables (PDF, DOCX, 2 XLSX, PPTX)
  3. vérification (fichiers présents + PDF lisible + sections attendues)

Usage : python scripts/pipeline.py
"""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = ROOT / "scripts"
DOCS = ROOT / "docs"
PY = sys.executable

EXPECTED = [
    "architecture_mvp_P13.png",
    "Rapport_gestion_projet_P13.docx",
    "Le_Gall_Morgan_1_rapport gestion projet_062026.pdf",
    "Le_Gall_Morgan_2_support presentation_062026.pptx",
    "Macro_backlog_P13.xlsx",
    "Estimation_couts_P13.xlsx",
]
PDF = DOCS / "Le_Gall_Morgan_1_rapport gestion projet_062026.pdf"
SECTIONS = ["Introduction", "Plan de projet", "Macro backlog", "Architecture technique",
            "Estimation des co", "Optimisation de l", "Bilan", "Conclusion", "Annexes"]


def step(title, script):
    print(f"\n=== {title} ===")
    r = subprocess.run([PY, str(SCRIPTS / script)], cwd=str(ROOT))
    if r.returncode != 0:
        sys.exit(f"ECHEC : {script} (code {r.returncode})")


def verify():
    print("\n=== Vérification ===")
    ok = True
    for f in EXPECTED:
        exists = (DOCS / f).exists()
        size = (DOCS / f).stat().st_size if exists else 0
        print(f"  {'OK ' if exists else 'MANQUE '}{f} ({size} o)")
        ok &= exists
    try:
        import pypdf
        reader = pypdf.PdfReader(str(PDF))
        full = "\n".join(p.extract_text() for p in reader.pages)
        print(f"  PDF : {len(reader.pages)} pages")
        for s in SECTIONS:
            present = s in full
            print(f"    {'OK ' if present else 'MANQUE '}section « {s} »")
            ok &= present
    except ImportError:
        print("  (pypdf absent : vérif contenu PDF sautée)")
    print("\n" + ("PIPELINE OK" if ok else "PIPELINE INCOMPLET"))
    return ok


if __name__ == "__main__":
    print("Pipeline livrables P13")
    step("1/3 Schéma d'architecture", "generate_archi_png.py")
    step("2/3 Livrables", "build_deliverables.py")
    sys.exit(0 if verify() else 1)

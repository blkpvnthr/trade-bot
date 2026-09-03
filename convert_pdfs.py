#!/usr/bin/env python3

import subprocess
from pathlib import Path

DOCS_DIR = Path("/Users/blkpvnthr/Documents/quantumquant/Repo/trader/docs")

pdf_files = list(DOCS_DIR.glob("*.pdf"))

if not pdf_files:
    print("No PDF files found.")
    exit(0)

for pdf_path in pdf_files:
    md_path = pdf_path.with_suffix(".md")

    print(f"Converting: {pdf_path.name} -> {md_path.name}")

    try:
        result = subprocess.run(
            [
                "markitdown",
                str(pdf_path),
                "-o",
                str(md_path),
            ],
            capture_output=True,
            text=True,
            check=True,
        )

        print(f"SUCCESS: {md_path.name}")

    except subprocess.CalledProcessError as e:
        print(f"FAILED: {pdf_path.name}")
        print(e.stderr)

print("Done.")


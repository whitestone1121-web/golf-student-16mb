"""
FineWeb-Edu Local Data Downloader
==================================
Downloads a slice of the FineWeb-Edu dataset from HuggingFace and saves it
as a local JSONL file so training doesn't depend on streaming.

FineWeb-Edu: curated educational text — better signal than raw FineWeb for
language modeling at small scale.

Usage:
    python3 golf/download_fineweb.py
    python3 golf/download_fineweb.py --docs 50000 --output golf/artifacts/fineweb_edu.jsonl

Output: golf/artifacts/fineweb_edu.jsonl  (~500MB for 50k docs)
"""

from __future__ import annotations
import argparse
import json
import os
import sys
from pathlib import Path

ARTIFACTS_DIR = Path(__file__).parent / "artifacts"
DEFAULT_OUTPUT = ARTIFACTS_DIR / "fineweb_edu.jsonl"

# HuggingFace dataset ID for FineWeb-Edu
DATASET_ID = "HuggingFaceFW/fineweb-edu"
SUBSET     = "sample-10BT"


def download(n_docs: int = 50_000, output_path: str | Path = DEFAULT_OUTPUT):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[FineWeb-Edu] Downloading {n_docs:,} documents → {output_path}")

    try:
        from datasets import load_dataset
    except ImportError:
        print("[FineWeb-Edu] ❌ 'datasets' not installed.")
        print("  Fix: pip install datasets --break-system-packages")
        print("  Or:  pip install datasets  (in a virtualenv)")
        sys.exit(1)

    print(f"[FineWeb-Edu] Streaming from HuggingFace ({DATASET_ID} / {SUBSET})...")
    ds = load_dataset(
        DATASET_ID,
        name=SUBSET,
        split="train",
        streaming=True,
        trust_remote_code=True,
    )

    written = 0
    total_chars = 0
    with open(output_path, "w", encoding="utf-8") as f:
        for row in ds:
            text = row.get("text", "")
            if len(text) < 100:
                continue
            # Write compact JSON line
            f.write(json.dumps({"text": text[:4000]}, ensure_ascii=False) + "\n")
            written += 1
            total_chars += len(text)

            if written % 5000 == 0:
                print(f"  {written:>8,} / {n_docs:,} docs  (~{total_chars/1e6:.1f}M chars)")

            if written >= n_docs:
                break

    size_mb = output_path.stat().st_size / 1024**2
    print(f"\n[FineWeb-Edu] ✅ Saved {written:,} docs → {output_path}  ({size_mb:.1f} MB)")
    print(f"[FineWeb-Edu] Avg doc length: {total_chars / max(1, written):.0f} chars")


def load_local(path: str | Path = DEFAULT_OUTPUT, limit: int | None = None) -> list[str]:
    """Load texts from a downloaded JSONL file."""
    path = Path(path)
    if not path.exists():
        return []
    texts = []
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if limit and i >= limit:
                break
            try:
                texts.append(json.loads(line)["text"])
            except Exception:
                pass
    return texts


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--docs",   type=int, default=50_000)
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    args = parser.parse_args()
    download(n_docs=args.docs, output_path=args.output)

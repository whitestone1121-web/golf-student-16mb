"""
Golf BPE Tokenizer Trainer
===========================
Trains a 4096-vocab BPE tokenizer on the FineWeb sample-10BT subset.
4096 vocab saves ~4MB vs standard 32k tokenizers (smaller embedding matrix).

Output: golf/artifacts/tokenizer.json  (~0.5MB)

Usage:
    python3 golf/tokenizer.py
    python3 golf/tokenizer.py --samples 50000 --vocab 4096
"""

from __future__ import annotations
import argparse
import os
import json
from pathlib import Path

ARTIFACTS_DIR = Path(__file__).parent / "artifacts"
TOKENIZER_PATH = ARTIFACTS_DIR / "tokenizer.json"


def train_bpe_tokenizer(
    n_samples: int = 100_000,
    vocab_size: int = 4096,
    output_path: str | Path = TOKENIZER_PATH,
) -> None:
    """Train and save BPE tokenizer on FineWeb text."""
    try:
        from tokenizers import Tokenizer
        from tokenizers.models import BPE
        from tokenizers.trainers import BpeTrainer
        from tokenizers.pre_tokenizers import ByteLevel
        from tokenizers.decoders import ByteLevel as ByteLevelDecoder
    except ImportError:
        raise ImportError("pip install tokenizers")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # ── Pull training text ────────────────────────────────────────────────────
    print(f"[Tokenizer] Loading {n_samples:,} FineWeb samples...")
    texts = _load_fineweb_texts(n_samples)
    print(f"[Tokenizer] Loaded {len(texts):,} documents  (~{sum(len(t) for t in texts) / 1e6:.1f}M chars)")

    # ── Train ─────────────────────────────────────────────────────────────────
    tokenizer = Tokenizer(BPE(unk_token="<unk>"))
    tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=True)
    tokenizer.decoder = ByteLevelDecoder()

    trainer = BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=5,
        special_tokens=["<pad>", "<unk>", "<bos>", "<eos>"],
        show_progress=True,
    )

    tokenizer.train_from_iterator(iter(texts), trainer=trainer, length=len(texts))

    tokenizer.save(str(output_path))
    size_kb = output_path.stat().st_size / 1024
    print(f"[Tokenizer] ✅ Saved → {output_path}  ({size_kb:.1f} KB, vocab={vocab_size})")

    # Sanity check
    enc = tokenizer.encode("The quick brown fox jumps over the lazy dog.")
    print(f"[Tokenizer] Sample encode: {len(enc.ids)} tokens → {enc.ids[:8]}...")


def _load_fineweb_texts(n_samples: int) -> list[str]:
    """Load FineWeb-Edu text: local JSONL → HuggingFace stream → DB fallback."""
    # 1. Prefer local snapshot (download_fineweb.py)
    local_path = Path(__file__).parent / "artifacts" / "fineweb_edu.jsonl"
    if local_path.exists():
        print(f"[Tokenizer] Using local FineWeb-Edu: {local_path}")
        texts = []
        with open(local_path, encoding="utf-8") as f:
            import json as _json
            for i, line in enumerate(f):
                if i >= n_samples:
                    break
                try:
                    texts.append(_json.loads(line)["text"])
                except Exception:
                    pass
        return texts

    # 2. Stream from HuggingFace
    try:
        from datasets import load_dataset
        ds = load_dataset(
            "HuggingFaceFW/fineweb-edu",
            name="sample-10BT",
            split="train",
            streaming=True,
            trust_remote_code=True,
        )
        texts = []
        for i, row in enumerate(ds):
            if i >= n_samples:
                break
            text = row.get("text", "")
            if len(text) > 50:
                texts.append(text[:2000])
        return texts
    except Exception as e:
        print(f"[Tokenizer] HuggingFace unavailable ({e}), using local fallback...")
        return _load_local_fallback(n_samples)


def _load_local_fallback(n_samples: int) -> list[str]:
    """
    Fallback: any local text DB (set LOCAL_TEXT_DB env var or provide fineweb_edu.jsonl)
    """
    import os as _os
    import sqlite3
    db_path = _os.environ.get("LOCAL_TEXT_DB", "memory.db")
    texts = []
    try:
        conn = sqlite3.connect(db_path, timeout=5)
        rows = conn.execute(
            "SELECT thought FROM episodes WHERE thought IS NOT NULL LIMIT ?",
            (n_samples,),
        ).fetchall()
        conn.close()
        texts = [r[0] for r in rows if r[0] and len(r[0]) > 10]
        print(f"[Tokenizer] Local DB fallback: {len(texts)} thoughts")
    except Exception as e:
        print(f"[Tokenizer] DB fallback failed: {e}")

    # Pad with basic English text if needed
    if len(texts) < 1000:
        print("[Tokenizer] WARNING: very few training texts — tokenizer quality will be low.")

    return texts


def load_tokenizer(path: str | Path = TOKENIZER_PATH):
    """Load a saved tokenizer for use in training/inference."""
    try:
        from tokenizers import Tokenizer
    except ImportError:
        raise ImportError("pip install tokenizers")
    return Tokenizer.from_file(str(path))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=100_000)
    parser.add_argument("--vocab", type=int, default=4096)
    parser.add_argument("--output", default=str(TOKENIZER_PATH))
    args = parser.parse_args()

    train_bpe_tokenizer(
        n_samples=args.samples,
        vocab_size=args.vocab,
        output_path=args.output,
    )

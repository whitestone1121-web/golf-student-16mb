"""
Golf Logit Cacher — pre-compute qwen2.5:32b soft targets
=========================================================
Pulls FineWeb text, tokenizes with the Golf 4096-vocab BPE,
then calls the local vLLM endpoint for top-K log-probabilities.

Run BEFORE the 10-minute training window (can run overnight).

Output: golf/artifacts/logit_cache.npz
  - tokens   : int16  [N, SEQ_LEN]   — token ids
  - lp_values: float16 [N, SEQ_LEN, K] — log-probs for top-K
  - lp_indices: int16  [N, SEQ_LEN, K] — token indices for top-K

Usage:
    python3 golf/cache_logits.py
    python3 golf/cache_logits.py --samples 50000 --topk 32 --seq-len 512
"""

from __future__ import annotations
import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import requests

ARTIFACTS_DIR = Path(__file__).parent / "artifacts"
TOKENIZER_PATH = ARTIFACTS_DIR / "tokenizer.json"
CACHE_PATH     = ARTIFACTS_DIR / "logit_cache.npz"

# Teacher LLM endpoint — set VLLM_URL env var to match your setup
VLLM_URL  = os.environ.get("VLLM_URL", "http://localhost:8000/v1/completions")
VLLM_MODEL = os.environ.get("VLLM_MODEL", "qwen2.5:32b")
TOP_K      = 32
SEQ_LEN    = 128
BATCH_SIZE = 4


def cache_logits(
    n_samples: int = 10_000,
    top_k: int = TOP_K,
    seq_len: int = SEQ_LEN,
    output_path: str | Path = CACHE_PATH,
    tokenizer_path: str | Path = TOKENIZER_PATH,
    hard_label_only: bool = False,
):
    output_path  = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if hard_label_only:
        print("[LogitCache] Hard-label mode: building cache from DB episodes (no LLM calls)")
        _build_hard_label_cache(output_path, tokenizer_path, n_samples, seq_len, top_k)
        return

    print(f"[LogitCache] Loading tokenizer from {tokenizer_path}")
    tokenizer = _load_tokenizer(tokenizer_path)

    print(f"[LogitCache] Loading {n_samples:,} FineWeb text chunks...")
    texts = _load_texts(n_samples)
    print(f"[LogitCache] Loaded {len(texts):,} chunks")

    all_tokens   = []
    all_lp_vals  = []
    all_lp_idx   = []

    t0    = time.time()
    done  = 0
    errs  = 0

    for b_start in range(0, len(texts), BATCH_SIZE):
        batch_texts = texts[b_start : b_start + BATCH_SIZE]

        # Tokenize
        for text in batch_texts:
            ids = tokenizer.encode(text).ids[:seq_len]
            if len(ids) < 4:
                continue
            # Pad to seq_len
            ids = ids + [0] * (seq_len - len(ids))

            # Call vLLM for soft logits
            lp_vals, lp_idx = _query_vllm(
                text=text[:1200],  # truncate text sent to vLLM
                top_k=top_k,
                seq_len=seq_len,
            )
            if lp_vals is None:
                errs += 1
                continue

            all_tokens.append(np.array(ids, dtype=np.int16))
            all_lp_vals.append(lp_vals.astype(np.float16))
            all_lp_idx.append(lp_idx.astype(np.int16))
            done += 1

        # Progress
        if done % 100 == 0 and done > 0:
            elapsed = time.time() - t0
            rate = done / elapsed
            eta  = (n_samples - done) / max(rate, 0.01)
            print(
                f"  [{done:>6}/{n_samples}] {rate:.1f} seq/s  "
                f"ETA: {eta/60:.1f}min  errors: {errs}"
            )

        if done >= n_samples:
            break

    if not all_tokens:
        print("[LogitCache] ❌ No samples cached. Check vLLM endpoint.")
        return

    np.savez_compressed(
        str(output_path),
        tokens    = np.stack(all_tokens),     # [N, SEQ_LEN]
        lp_values = np.stack(all_lp_vals),    # [N, SEQ_LEN, K]
        lp_indices= np.stack(all_lp_idx),     # [N, SEQ_LEN, K]
    )
    size_mb = output_path.stat().st_size / 1024**2
    print(f"\n[LogitCache] ✅ Saved {done:,} samples → {output_path}  ({size_mb:.1f} MB)")


def _query_vllm(text: str, top_k: int, seq_len: int):
    """
    Query local vLLM for top-K log-probabilities per position.
    Returns (lp_values [SEQ_LEN, K], lp_indices [SEQ_LEN, K]) or (None, None).
    """
    try:
        resp = requests.post(
            VLLM_URL,
            json={
                "model": VLLM_MODEL,
                "prompt": text,
                "max_tokens": 1,
                "logprobs": top_k,
                "echo": True,        # include prompt token logprobs
                "temperature": 0.0,
            },
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()

        # Extract logprobs from vLLM response
        choice = data["choices"][0]
        lp_raw = choice.get("logprobs", {})
        token_logprobs = lp_raw.get("top_logprobs", [])  # list of {token_id: logprob}

        if not token_logprobs:
            return None, None

        lp_vals = np.full((seq_len, top_k), -100.0, dtype=np.float32)
        lp_idx  = np.zeros((seq_len, top_k), dtype=np.int32)

        for t, tok_dict in enumerate(token_logprobs[:seq_len]):
            if tok_dict is None:
                continue
            # Sort by logprob descending
            sorted_items = sorted(tok_dict.items(), key=lambda x: -x[1])[:top_k]
            for k, (tid_str, lp) in enumerate(sorted_items):
                try:
                    tid = int(tid_str)
                except (ValueError, TypeError):
                    # vLLM sometimes returns token strings — use hash as fallback
                    tid = hash(tid_str) % 4096
                lp_vals[t, k] = lp
                lp_idx[t, k]  = tid

        return lp_vals, lp_idx

    except Exception as e:
        return None, None


def _load_texts(n_samples: int) -> list[str]:
    """Load FineWeb-Edu text: local JSONL → HuggingFace stream → DB fallback."""
    # 1. Local snapshot (fastest — no internet needed)
    local_path = Path(__file__).parent / "artifacts" / "fineweb_edu.jsonl"
    if local_path.exists():
        print(f"[LogitCache] Using local FineWeb-Edu: {local_path}")
        import json as _json
        texts = []
        with open(local_path, encoding="utf-8") as f:
            for i, line in enumerate(f):
                if i >= n_samples:
                    break
                try:
                    texts.append(_json.loads(line)["text"])
                except Exception:
                    pass
        return texts

    # 2. HuggingFace stream
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
            t = row.get("text", "")
            if len(t) > 100:
                texts.append(t)
        return texts
    except Exception:
        pass

    import sqlite3 as _sqlite3
    db = os.environ.get("LOCAL_TEXT_DB", "memory.db")
    try:
        conn = sqlite3.connect(db, timeout=5)
        rows = conn.execute(
            "SELECT thought FROM episodes WHERE thought IS NOT NULL LIMIT ?",
            (n_samples,),
        ).fetchall()
        conn.close()
        return [r[0] for r in rows if r[0] and len(r[0]) > 100]
    except Exception:
        return []


def _load_tokenizer(path: str | Path):
    try:
        from tokenizers import Tokenizer
    except ImportError:
        raise ImportError("pip install tokenizers")
    if not Path(path).exists():
        raise FileNotFoundError(
            f"Tokenizer not found at {path}. Run: python3 golf/tokenizer.py"
        )
    return Tokenizer.from_file(str(path))


def _build_hard_label_cache(
    output_path: Path,
    tokenizer_path: str | Path,
    n_samples: int,
    seq_len: int,
    top_k: int,
) -> None:
    """
    Fast fallback: build training cache from a local text DB.
    Uses text as input and the action label as the hard target.
    Saves in same npz format as the soft-logit cache so train.py is unchanged.

    Action → one-hot soft label: [0, 0, ..., 1.0, ..., 0] for the target token.
    This is equivalent to CE training but stored in the same format as KL distillation.
    """
    import sqlite3
    import json as _json
    import sys
    sys.path.insert(0, str(Path(__file__).parents[2] / "src"))

    tokenizer = _load_tokenizer(tokenizer_path)

    # Action vocab: maps action labels to token ids in our BPE vocab
    ACTIONS = [
        "explore_unknown", "observe_environment", "self_evolve",
        "respond_to_query", "alert_operator", "patrol", "analyze_threat", "idle",
    ]

    db_paths = [
        os.environ.get("LOCAL_TEXT_DB", "memory.db"),
    ]
    rows = []
    for db in db_paths:
        if not Path(db).exists():
            continue
        try:
            conn = sqlite3.connect(db, timeout=5)
            fetched = conn.execute(
                "SELECT thought, decision FROM episodes "
                "WHERE thought IS NOT NULL AND decision IS NOT NULL LIMIT ?",
                (n_samples,),
            ).fetchall()
            conn.close()
            rows = fetched
            print(f"[LogitCache] DB: {db} → {len(rows)} episodes")
            break
        except Exception as e:
            print(f"[LogitCache] DB error: {e}")

    if not rows:
        print("[LogitCache] ❌ No episodes found in DB.")
        return

    all_tokens   = []
    all_lp_vals  = []
    all_lp_idx   = []

    for thought, decision_json in rows:
        # Tokenize the teacher's inner monologue text
        ids = tokenizer.encode(str(thought)).ids[:seq_len]
        if len(ids) < 4:
            continue
        ids = ids + [0] * (seq_len - len(ids))  # pad

        # Parse action from decision
        try:
            dec = _json.loads(decision_json)
            action = dec.get("action", "idle")
        except Exception:
            action = "idle"

        # Build pseudo soft label: one-hot at action position → scattered into top_k
        # We use the action's token id from BPE as the top-1 prediction
        action_tok_id = tokenizer.encode(action).ids[0] if tokenizer.encode(action).ids else 0
        lp_vals = np.full((seq_len, top_k), -20.0, dtype=np.float32)  # low log-prob background
        lp_idx  = np.tile(np.arange(top_k, dtype=np.int32), (seq_len, 1))  # default indices
        # Place action token as top-1 everywhere (hard label)
        lp_vals[:, 0] = 0.0          # log-prob 0 = prob 1.0 for the correct token
        lp_idx[:, 0]  = action_tok_id

        all_tokens.append(np.array(ids, dtype=np.int16))
        all_lp_vals.append(lp_vals.astype(np.float16))
        all_lp_idx.append(lp_idx.astype(np.int32))

    if not all_tokens:
        print("[LogitCache] ❌ No valid episodes to cache.")
        return

    np.savez_compressed(
        str(output_path),
        tokens     = np.stack(all_tokens),
        lp_values  = np.stack(all_lp_vals),
        lp_indices = np.stack(all_lp_idx),
    )
    size_mb = output_path.stat().st_size / 1024**2
    print(f"[LogitCache] ✅ Hard-label cache: {len(all_tokens)} episodes → {output_path} ({size_mb:.2f} MB)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples",         type=int, default=10_000)
    parser.add_argument("--topk",            type=int, default=TOP_K)
    parser.add_argument("--seq-len",         type=int, default=SEQ_LEN)
    parser.add_argument("--output",          default=str(CACHE_PATH))
    parser.add_argument("--hard-label-only", action="store_true",
                        help="Skip LLM calls — build cache from DB hard labels (fast)")
    args = parser.parse_args()

    cache_logits(
        n_samples=args.samples,
        top_k=args.topk,
        seq_len=args.seq_len,
        output_path=args.output,
        hard_label_only=args.hard_label_only,
    )

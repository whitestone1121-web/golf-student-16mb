"""
Golf Training Script — OpenAI Parameter Golf Submission
=========================================================
Trains GolfStudent on FineWeb-Edu via next-token prediction.
Architecture: weight-tied hybrid (LinearRecurrence + Attention), 4096 BPE vocab.

Usage:
    python3 train_public.py                  # standard run
    python3 train_public.py --time-limit 600 # 10-minute contest window
"""
from __future__ import annotations
import argparse, math, signal, sys, time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent))
from model import GolfStudent, VOCAB_SIZE

TOKENIZER_PATH = Path(__file__).parent / "artifacts" / "tokenizer.json"
CHECKPOINT     = Path(__file__).parent / "artifacts" / "golf_best.pt"


class LazyFineWebDataset(torch.utils.data.IterableDataset):
    """Streams FineWeb JSONL lazily — no upfront load, starts training immediately."""
    def __init__(self, jsonl_path: str, seq_len: int = 128, max_seqs: int = 500_000):
        from tokenizers import Tokenizer
        self.tok = Tokenizer.from_file(str(TOKENIZER_PATH))
        self.path = jsonl_path
        self.seq_len = seq_len
        self.max_seqs = max_seqs

    def __iter__(self):
        import json
        seen = 0
        buf = []
        sl = self.seq_len
        with open(self.path, encoding="utf-8") as f:
            for line in f:
                if seen >= self.max_seqs:
                    break
                try:
                    ids = self.tok.encode(json.loads(line)["text"]).ids
                except Exception:
                    continue
                buf.extend(ids)
                while len(buf) >= sl + 1:
                    chunk = buf[:sl + 1]
                    buf = buf[sl + 1:]
                    yield torch.tensor(chunk, dtype=torch.long)
                    seen += 1
                    if seen >= self.max_seqs:
                        break


def train(time_limit: int = 600, batch_size: int = 64, max_lr: float = 8e-4,
          data: str = "fineweb_edu.jsonl"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Train] {device} | {time_limit}s budget")

    dataset = LazyFineWebDataset(data)
    loader  = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=True,
                         num_workers=0, pin_memory=(device.type == "cuda"))

    model = GolfStudent().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=max_lr, weight_decay=0.1)

    # IterableDataset has no len() — estimate steps from empirical ~10 steps/sec RTX 6000
    est_total_steps = time_limit * 10
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=est_total_steps, eta_min=max_lr * 0.01)
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    t0, best, step = time.time(), float("inf"), 0
    CHECKPOINT.parent.mkdir(parents=True, exist_ok=True)

    def _save(l):
        raw = model._orig_mod if hasattr(model, "_orig_mod") else model
        torch.save({"model": raw.state_dict(), "step": step, "loss": l}, str(CHECKPOINT))
        print(f"  → best {l:.4f}")

    signal.signal(signal.SIGINT, lambda s, f: (_save(best), sys.exit(0)))

    epoch = 0
    while True:
        epoch += 1
        el, es = 0.0, 0
        for batch in loader:
            if time.time() - t0 >= time_limit:
                break
            x, y = batch[:, :-1].to(device), batch[:, 1:].to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=(device.type == "cuda")):
                logits = model(x)
                loss = F.cross_entropy(logits.view(-1, VOCAB_SIZE), y.reshape(-1), ignore_index=0)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer); scaler.update()
            if step < est_total_steps:
                scheduler.step()
            step += 1; el += loss.item(); es += 1
            if loss.item() < best:
                best = loss.item(); _save(best)
        else:
            e = time.time() - t0
            ppl = math.exp(min(el / max(1, es), 20))
            print(f"  Epoch {epoch:3d} | loss {el/max(1,es):.4f} | ppl {ppl:.1f} | {e:.0f}s")
            continue
        break

    print(f"\n✅ Done | {step} steps | best loss {best:.4f}")
    if best == float("inf"):
        _save(best)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--time-limit", type=int, default=600)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--max-lr", type=float, default=8e-4)
    p.add_argument("--data", default="fineweb_edu.jsonl",
                   help="Path to local FineWeb-Edu JSONL file")
    a = p.parse_args()
    train(a.time_limit, a.batch_size, a.max_lr, a.data)

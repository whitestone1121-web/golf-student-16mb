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
          data: str = "fineweb_edu.jsonl", accum_steps: int = 4):
    # ── Device detection (CUDA / MPS / CPU) ──────────────────────────────────
    if torch.cuda.is_available():
        device = torch.device("cuda")
        amp_dtype = torch.bfloat16   # BF16: best on CUDA (Ampere+)
        use_amp   = True
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
        amp_dtype = torch.float16    # MPS: no BF16 support yet
        use_amp   = True
    else:
        device = torch.device("cpu")
        amp_dtype = torch.float32
        use_amp   = False

    print(f"[Train] {device} ({amp_dtype}) | {time_limit}s | bs={batch_size} accum={accum_steps} (eff={batch_size*accum_steps})")

    dataset = LazyFineWebDataset(data)
    loader  = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=True,
                         num_workers=0, pin_memory=(device.type == "cuda"))

    model = GolfStudent().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=max_lr, weight_decay=0.1)

    # ── EMA shadow model (same technique as leaderboard #1/#2/#3) ────────────
    # Keep EMA state on CPU float32 — avoids device/dtype conflicts in hot loop
    ema_decay = 0.999
    ema_state = {k: v.detach().cpu().float().clone()
                 for k, v in model.state_dict().items()}

    def update_ema():
        with torch.no_grad():
            for k, v in model.state_dict().items():
                ema_state[k].mul_(ema_decay).add_(v.detach().cpu().float(), alpha=1.0 - ema_decay)

    def save_ema():
        ema_fp = CHECKPOINT.parent / "golf_best_ema.pt"
        torch.save({"model": {k: v.half() for k, v in ema_state.items()}, "ema": True}, str(ema_fp))
        print(f"  [EMA] saved → {ema_fp.name}")

    # ── Scheduler: Cosine main + linear warmdown at 85% of budget ────────────
    # Warmdown = rapid LR → 0 in the last 15% of training steps
    # signalrush #1 used warmdown3500 — same concept
    warmdown_frac = 0.15
    est_total_steps = time_limit * 10
    warmdown_start  = int(est_total_steps * (1 - warmdown_frac))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=warmdown_start, eta_min=max_lr * 0.05)

    scaler = torch.amp.GradScaler(device.type, enabled=(use_amp and device.type == "cuda"))

    t0, best, step = time.time(), float("inf"), 0
    in_warmdown = False
    CHECKPOINT.parent.mkdir(parents=True, exist_ok=True)

    def _save(l):
        raw = model._orig_mod if hasattr(model, "_orig_mod") else model
        torch.save({"model": raw.state_dict(), "step": step, "loss": l}, str(CHECKPOINT))
        print(f"  → best {l:.4f}")

    signal.signal(signal.SIGINT, lambda s, f: (save_ema(), _save(best), sys.exit(0)))

    epoch = 0
    while True:
        epoch += 1
        el, es = 0.0, 0
        for batch in loader:
            elapsed = time.time() - t0
            if elapsed >= time_limit:
                break
            x, y = batch[:, :-1].to(device), batch[:, 1:].to(device)
            # Gradient accumulation: only zero_grad at start of accumulation window
            if step % accum_steps == 0:
                optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device.type, dtype=amp_dtype, enabled=use_amp):
                logits = model(x)
                loss = F.cross_entropy(logits.view(-1, VOCAB_SIZE), y.reshape(-1), ignore_index=0)
            # Scale loss by 1/accum_steps so gradient magnitude is batch-size invariant
            scaler.scale(loss / accum_steps).backward()

            is_update_step = (step + 1) % accum_steps == 0
            if is_update_step:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer); scaler.update()

            # ── LR schedule: cosine → warmdown ───────────────────────────────
            if step < warmdown_start:
                scheduler.step()
            else:
                # Linear warmdown: LR → 0 over remaining steps
                if not in_warmdown:
                    in_warmdown = True
                    print(f"  [Warmdown] step={step}, LR decaying to 0")
                remaining = max(1, est_total_steps - step)
                for pg in optimizer.param_groups:
                    pg["lr"] = max_lr * 0.05 * (remaining / (est_total_steps - warmdown_start))

            # ── EMA update (every step) ──────────────────────────────────────
            update_ema()

            step += 1; el += loss.item(); es += 1
            if loss.item() < best:
                best = loss.item(); _save(best)
        else:
            e = time.time() - t0
            ppl = math.exp(min(el / max(1, es), 20))
            lr_now = optimizer.param_groups[0]["lr"]
            print(f"  Epoch {epoch:3d} | loss {el/max(1,es):.4f} | ppl {ppl:.1f} | lr {lr_now:.2e} | {e:.0f}s")
            continue
        break

    save_ema()
    print(f"\n✅ Done | {step} steps | best loss {best:.4f}")
    if best == float("inf"):
        _save(best)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--time-limit", type=int, default=600)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--accum-steps", type=int, default=4,
                   help="Gradient accumulation steps (effective batch = batch_size * accum_steps)")
    p.add_argument("--max-lr", type=float, default=8e-4)
    p.add_argument("--data",
                   default=str(Path(__file__).parent / "artifacts" / "fineweb_edu.jsonl"),
                   help="Path to FineWeb-Edu JSONL (run download_fineweb.py first)")
    a = p.parse_args()
    train(a.time_limit, a.batch_size, a.max_lr, a.data, a.accum_steps)

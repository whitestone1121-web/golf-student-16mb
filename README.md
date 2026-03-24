# golf-student-16mb

**16MB Parameter Golf — 16.4M Parameter Weight-Tied Hybrid (d=288, L=14)**

A distillation student model for the [OpenAI Parameter Golf Challenge](https://github.com/openai/parameter-golf), optimized to maximize language modeling quality within a strict 16MB artifact limit.

## Architecture

| Component | Spec |
|---|---|
| Parameters | 16.4M |
| d_model | 288 |
| Layers | 14 (10 LinearRecurrence + 4 Attention) |
| Vocab | 4096 BPE (custom, saves ~4MB vs GPT-2) |
| Weight tying | ✅ (embedding = output projection) |
| FFN | SwiGLU, 3× expansion |
| Training | BF16, CosineAnnealingLR, 10-min budget |
| Compression | Manual absmax INT8 → 13.44 MB |
| Submission ZIP | ~14 MB ✅ under 16 MB limit |

## Highlights

- **Weight-tied hybrid**: LinearRecurrence (O(L) time) + Attention (Flash Attention 2 via SDPA) interleaved every 3rd layer. Recurrence handles long-range state efficiently; attention provides global context.
- **Custom 4k BPE**: Trained on FineWeb-Edu sample. Reclaims approx 4MB vs standard 32k tokenizer — used for more model depth instead.
- **Manual absmax INT8**: Replaces `torch.quantization.quantize_dynamic` (which adds approx 4MB of metadata overhead). Pure weight bytes stored as compressed `.npz`.

## Quick Start

```bash
# 1. Create and activate a virtual environment (avoids Python version conflicts on Mac)
python3 -m venv venv
source venv/bin/activate      # Mac/Linux
# venv\Scripts\activate       # Windows

# 2. Install dependencies
pip install -r requirements.txt

# 3. Download FineWeb-Edu slice (one-time, approx 500MB)
python3 download_fineweb.py --docs 50000

# 4. Train BPE tokenizer
python3 tokenizer.py --samples 50000

# 5. Run 10-minute training sprint
python3 train.py --time-limit 600

# 6. Package submission
python3 export.py
```

## File Structure

```
├── model.py              # GolfStudent architecture (pure PyTorch)
├── train.py              # Training loop (FineWeb-Edu, BF16, cosine LR)
├── tokenizer.py          # 4096 BPE tokenizer trainer
├── cache_logits.py       # Optional: pre-cache teacher logits
├── download_fineweb.py   # FineWeb-Edu JSONL downloader
├── export.py             # INT8 quantize + ZIP packaging
├── artifacts/
│   └── tokenizer.json    # Pre-trained 4096 BPE vocab
└── requirements.txt
```

## Requirements

```
torch>=2.2.0
tokenizers>=0.15.0
datasets>=2.0.0
numpy>=1.24.0
```

---

*Architecture and training methods are subject to pending patent applications. All rights reserved.*

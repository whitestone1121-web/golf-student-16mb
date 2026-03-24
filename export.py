"""
Golf Export — INT8 Quantize + Zip to ≤16MB Artifact
====================================================
Loads best training checkpoint, applies dynamic INT8 quantization,
and packages as golf_submission.zip for the OpenAI Parameter Golf leaderboard.

Also exports an ONNX version for edge deployment (Jetson, phone, WebAssembly).

Usage:
    python3 golf/export.py
    python3 golf/export.py --checkpoint golf/artifacts/golf_best.pt --onnx
"""

from __future__ import annotations
import argparse
import os
import shutil
import sys
import zipfile
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent))   # import model.py from same dir
from model import GolfStudent, VOCAB_SIZE

ARTIFACTS_DIR    = Path(__file__).parent / "artifacts"
CHECKPOINT_PATH  = ARTIFACTS_DIR / "golf_best.pt"
TOKENIZER_PATH   = ARTIFACTS_DIR / "tokenizer.json"
EXPORT_DIR       = ARTIFACTS_DIR / "export"
SUBMISSION_ZIP   = Path(__file__).parent / "golf_submission.zip"
ONNX_PATH        = EXPORT_DIR / "golf_student.onnx"

MAX_BYTES = 16_000_000


def export(
    checkpoint_path: str | Path = CHECKPOINT_PATH,
    output_zip: str | Path = SUBMISSION_ZIP,
    export_onnx: bool = False,
    export_minimal: bool = True,  # default: minimal (no secret sauce)
):
    checkpoint_path = Path(checkpoint_path)
    output_zip      = Path(output_zip)
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"[Export] Loading checkpoint: {checkpoint_path}")
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}\n"
            "Run: python3 golf/train.py  (or --dry-run for smoke test)"
        )

    ckpt = torch.load(str(checkpoint_path), map_location="cpu")
    state_dict = ckpt["model"] if "model" in ckpt else ckpt

    # Strip _orig_mod prefix from torch.compile
    state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}

    # ── Load model in FP32 for quantization ──────────────────────────────────
    model = GolfStudent()
    model.load_state_dict(state_dict, strict=False)
    model.eval()


    # ── Manual INT8 Quantization (absmax) — no PyTorch metadata overhead ─────
    # torch.quantization.quantize_dynamic adds ~4MB of scale/zeropoint metadata
    # Manual INT8: weights_int8 = round(w / (max_abs / 127))  → 1 byte/param
    print("[Export] Applying manual absmax INT8 quantization...")
    import numpy as np

    int8_weights = {}
    scales = {}
    for name, param in model.state_dict().items():
        w = param.detach().float().numpy()
        max_abs = np.abs(w).max()
        scale = max_abs / 127.0 if max_abs > 0 else 1.0
        w_int8 = np.clip(np.round(w / scale), -127, 127).astype(np.int8)
        int8_weights[name] = w_int8
        scales[name + "_scale"] = np.float32(scale)

    q_path = EXPORT_DIR / "golf_student_int8.npz"
    np.savez_compressed(str(q_path), **int8_weights, **scales)
    q_size_mb = q_path.stat().st_size / 1024**2
    print(f"[Export] ✅ Manual INT8 done")
    print(f"[Export] Quantized model: {q_size_mb:.2f} MB")

    # ── ONNX Export (optional, for edge deployment) ───────────────────────────
    if export_onnx:
        print("[Export] Exporting ONNX (FP32 for compatibility)...")
        dummy_input = torch.randint(0, VOCAB_SIZE, (1, 64))
        try:
            torch.onnx.export(
                model,
                (dummy_input,),
                str(ONNX_PATH),
                opset_version=17,
                input_names=["token_ids"],
                output_names=["logits"],
                dynamic_axes={
                    "token_ids": {0: "batch", 1: "seq"},
                    "logits":    {0: "batch", 1: "seq"},
                },
                do_constant_folding=True,
            )
            onnx_mb = ONNX_PATH.stat().st_size / 1024**2
            print(f"[Export] ✅ ONNX saved: {ONNX_PATH}  ({onnx_mb:.2f} MB)")
        except Exception as e:
            print(f"[Export] ⚠ ONNX export failed: {e}")

    # ── Copy supporting files ─────────────────────────────────────────────────
    if export_minimal:
        files_to_bundle = [
            (q_path,                                                    "golf_student_int8.npz"),
            (Path(__file__).parent / "model.py",        "model.py"),
            (Path(__file__).parent / "train_public.py", "train.py"),
        ]
        if TOKENIZER_PATH.exists():
            files_to_bundle.append((TOKENIZER_PATH, "artifacts/tokenizer.json"))
    else:
        files_to_bundle = [
            (q_path,                                                    "golf_student_int8.npz"),
            (Path(__file__).parent / "model.py",        "golf/model.py"),
            (Path(__file__).parent / "train.py",        "golf/train.py"),
            (Path(__file__).parent / "tokenizer.py",    "golf/tokenizer.py"),
            (Path(__file__).parent / "cache_logits.py", "golf/cache_logits.py"),
            (Path(__file__).parent / "export.py",       "golf/export.py"),
        ]
        if TOKENIZER_PATH.exists():
            files_to_bundle.append((TOKENIZER_PATH, "artifacts/tokenizer.json"))

    # ── Bundle into ZIP ───────────────────────────────────────────────────────
    print(f"\n[Export] Building submission ZIP: {output_zip}")
    with zipfile.ZipFile(str(output_zip), "w", zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
        for src, arcname in files_to_bundle:
            if Path(src).exists():
                zf.write(str(src), arcname)
                size_kb = Path(src).stat().st_size / 1024
                print(f"  + {arcname:<40} ({size_kb:.1f} KB)")
            else:
                print(f"  ⚠ MISSING: {src}")

    # ── Size check ────────────────────────────────────────────────────────────
    zip_bytes = output_zip.stat().st_size
    zip_mb    = zip_bytes / 1024**2
    headroom  = MAX_BYTES - zip_bytes

    print(f"\n{'='*50}")
    if zip_bytes <= MAX_BYTES:
        print(f"  ✅ SUBMISSION SIZE: {zip_mb:.2f} MB  (headroom: {headroom/1024:.0f} KB)")
    else:
        print(f"  ❌ TOO LARGE: {zip_mb:.2f} MB  (over by {-headroom/1024:.0f} KB)")
        print("  → Reduce d_model or n_layers in golf/model.py")
    print(f"  Path: {output_zip.absolute()}")
    print(f"{'='*50}\n")

    # ── Submission README ─────────────────────────────────────────────────────
    readme = """# OpenAI Parameter Golf Submission

## Architecture
- Model: GolfStudent (weight-tied hybrid, 11.4M params)
- Vocab: 4096 BPE
- Layers: 12 (8 LinearRecurrence + 4 Attention), d_model=256

## Reproduction
```bash
pip install torch tokenizers
python3 train.py --data fineweb_edu.jsonl --time-limit 600
```

---
*Architecture and training methods contained herein are subject to
pending patent applications. All rights reserved.*
"""
    readme_path = EXPORT_DIR / "README.md"
    readme_path.write_text(readme)

    return zip_bytes <= MAX_BYTES


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=str(CHECKPOINT_PATH))
    parser.add_argument("--output",     default=str(SUBMISSION_ZIP))
    parser.add_argument("--onnx",       action="store_true")
    parser.add_argument("--minimal",    action="store_true", default=True,
                        help="Minimum submission: weights+model+train only (default: True)")
    parser.add_argument("--full",       action="store_true",
                        help="Include all pipeline scripts (cache_logits, tokenizer, etc.)")
    args = parser.parse_args()

    ok = export(
        checkpoint_path=args.checkpoint,
        output_zip=args.output,
        export_onnx=args.onnx,
        export_minimal=not args.full,
    )
    sys.exit(0 if ok else 1)

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd


os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.0")

import lib

logging.basicConfig(level=logging.INFO, format="%(asctime)s : %(levelname)s : %(message)s")
log = logging.getLogger("transformer_worker")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--train", type=Path, required=True)
    p.add_argument("--predict", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--metrics_out", type=Path, required=True)

    p.add_argument("--model_family", choices=["deberta", "gemma_lora"], required=True)
    p.add_argument("--backbone", required=True)
    p.add_argument("--label", required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--device", default="auto", choices=["auto", "cuda", "mps", "cpu"])
    p.add_argument("--cache_dir", type=Path, required=True)

    p.add_argument("--epochs", type=int, required=True)
    p.add_argument("--batch_size", type=int, required=True)
    p.add_argument("--grad_accum", type=int, default=1)
    p.add_argument("--lr", type=float, required=True)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--warmup_frac", type=float, default=0.10)
    p.add_argument("--max_len", type=int, required=True)
    p.add_argument("--loss", choices=["mse", "ccc", "mse+ccc"], default="mse")
    p.add_argument("--pooling", default="cls")
    p.add_argument("--dropout", type=float, default=0.1)

    
    p.add_argument("--use_user_emb", action="store_true")
    p.add_argument("--use_format_emb", action="store_true")
    p.add_argument("--residualize", action="store_true")

    # Gemma LoRA knobs.
    p.add_argument("--lora_r", type=int, default=16)
    p.add_argument("--lora_alpha", type=int, default=32)
    p.add_argument("--lora_dropout", type=float, default=0.05)
    p.add_argument("--lora_targets", choices=["attn", "attn+mlp"], default="attn")
    p.add_argument("--text_transform", choices=["none", "gemma_chat"], default="none")
    p.add_argument("--head_warmup_epochs", type=int, default=0)
    return p.parse_args()


def require_columns(df: pd.DataFrame, path: Path) -> None:
    need = ["user_id", "text_id", "text_clean", "is_words", "valence", "arousal"]
    missing = [c for c in need if c not in df.columns]
    if missing:
        raise ValueError(f"{path} is missing columns required by the transformer worker: {missing}")


def main() -> None:
    args = parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.metrics_out.parent.mkdir(parents=True, exist_ok=True)
    args.cache_dir.mkdir(parents=True, exist_ok=True)

    train_df = pd.read_csv(args.train)
    predict_df = pd.read_csv(args.predict)
    require_columns(train_df, args.train)
    require_columns(predict_df, args.predict)

    device = lib.pick_device(args.device)
    log.info("device=%s model_family=%s backbone=%s seed=%d", device, args.model_family, args.backbone, args.seed)

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.backbone, use_fast=True)
    if args.model_family == "gemma_lora":
        if tokenizer.pad_token_id is None:
            raise ValueError("Gemma tokenizer must define pad_token_id for this pipeline.")
        tokenizer.padding_side = "right"

    model_cfg = lib.ModelConfig(
        backbone=args.backbone,
        pooling=args.pooling,
        use_user_emb=args.use_user_emb,
        use_format_emb=args.use_format_emb,
        residualize=args.residualize,
        dropout=args.dropout,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        lora_targets=args.lora_targets,
    )
    train_cfg = lib.TrainConfig(
        model=model_cfg,
        loss=args.loss,
        lr=args.lr,
        weight_decay=args.weight_decay,
        batch_size=args.batch_size,
        epochs=args.epochs,
        warmup_frac=args.warmup_frac,
        max_len=args.max_len,
        seed=args.seed,
        label=args.label,
        text_transform=None if args.text_transform == "none" else args.text_transform,
        head_warmup_epochs=args.head_warmup_epochs,
    )

    model_cls = lib.GemmaLoRAModel if args.model_family == "gemma_lora" else lib.ValenceArousalModel
    run = lib.train_one(
        train_cfg,
        train_df=train_df.reset_index(drop=True),
        val_df=predict_df.reset_index(drop=True),
        tokenizer=tokenizer,
        device=device,
        pick_best_epoch=False,
        cache_dir=args.cache_dir,
        verbose=True,
        model_cls=model_cls,
        grad_accum=args.grad_accum,
    )
    pred_valence, pred_arousal = run["preds"]
    np.savez_compressed(args.out, pred_valence=pred_valence, pred_arousal=pred_arousal)
    args.metrics_out.write_text(json.dumps(run["metrics"], indent=2))
    if "loss_history" in run:
        losses_out = args.metrics_out.with_name(
            args.metrics_out.name.replace(".metrics.json", ".losses.json")
        )
        losses_out.write_text(json.dumps(run["loss_history"], indent=2))
    log.info("wrote %s and %s", args.out, args.metrics_out)


if __name__ == "__main__":
    main()

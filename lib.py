from __future__ import annotations

import hashlib
import json
import logging
import pickle
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.stats import pearsonr
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset

log = logging.getLogger(__name__)


_CONTRACTION_PATTERNS = [
    (re.compile(r"\s+'\s*([a-zA-Z]{1,3})\b"), r"'\1"),
    (re.compile(r"\bn\s*'\s*t\b"),             r"n't"),
]
_PUNCT_AFTER     = re.compile(r"\s+([,.;:!?\)\]])")
_PUNCT_BEFORE    = re.compile(r"([\(\[])\s+")
_LITERAL_NEWLINE = re.compile(r"\\\s*n")
_MULTI_SPACE     = re.compile(r"\s{2,}")


def detokenize(s: str) -> str:
    """Undo upstream spaced-punctuation tokenization. Keeps anonymization tokens (ORG, PERSON)."""
    if not isinstance(s, str):
        return ""
    s = _LITERAL_NEWLINE.sub(" ", s)
    for pat, repl in _CONTRACTION_PATTERNS:
        s = pat.sub(repl, s)
    s = _PUNCT_AFTER.sub(r"\1", s)
    s = _PUNCT_BEFORE.sub(r"\1", s)
    return _MULTI_SPACE.sub(" ", s).strip()



def build_val_split(
    df: pd.DataFrame,
    seed: int = 13,
    unseen_frac: float = 0.20,
    within_user_holdout: float = 0.25,
) -> pd.Series:
    """Returns a boolean Series of the same length as `df` indicating val membership.

    """
    rng = np.random.default_rng(seed)
    users = df.user_id.unique().copy()
    rng.shuffle(users)
    n_unseen = int(round(len(users) * unseen_frac))
    unseen_users = set(users[:n_unseen])

    is_val = pd.Series(False, index=df.index)
    is_val |= df.user_id.isin(unseen_users)

    for u, sub in df.groupby("user_id"):
        if u in unseen_users or len(sub) < 4:
            continue
        n_hold = max(2, int(round(len(sub) * within_user_holdout)))
        essays = sub.index[~sub.is_words].tolist()
        wordsl = sub.index[ sub.is_words].tolist()
        rng.shuffle(essays); rng.shuffle(wordsl)
        ratio_words = len(wordsl) / max(1, len(sub))
        n_hold_w = int(round(n_hold * ratio_words))
        n_hold_e = n_hold - n_hold_w
        for idx in essays[:n_hold_e] + wordsl[:n_hold_w]:
            is_val.loc[idx] = True
    return is_val


_ZERO_VAR_TOL = 1e-12


def task1_correlation_robust(
    user_ids: Sequence,
    text_ids: Sequence,
    predictions: Sequence[float],
    labels: Sequence[float],
) -> dict:
    users = np.asarray(user_ids)
    preds = np.asarray(predictions, dtype=float)
    labs  = np.asarray(labels,      dtype=float)
    unique_users = np.unique(users)

    r_vals, p_vals = [], []
    for u in unique_users:
        mask = users == u
        if mask.sum() < 2: continue
        if np.var(labs[mask])  <= _ZERO_VAR_TOL: continue
        if np.var(preds[mask]) <= _ZERO_VAR_TOL:
            r_vals.append(0.0); p_vals.append(1e-10); continue
        r, p = pearsonr(preds[mask], labs[mask])
        r_vals.append(float(r)); p_vals.append(float(p))

    r_within = float(np.mean(r_vals)) if r_vals else float("nan")
    p_within = (float(len(p_vals) / sum(1.0 / max(pv, 1e-10) for pv in p_vals))
                if p_vals else float("nan"))

    means_p = [float(np.mean(preds[users == u])) for u in unique_users]
    means_l = [float(np.mean(labs[users == u]))  for u in unique_users]
    if np.var(means_p) <= _ZERO_VAR_TOL or np.var(means_l) <= _ZERO_VAR_TOL:
        r_between, p_between = float("nan"), float("nan")
    else:
        r_between, p_between = pearsonr(means_p, means_l)
        r_between, p_between = float(r_between), float(p_between)

    with np.errstate(invalid="ignore"):
        z = 0.5 * (np.arctanh(np.clip(r_within,  -0.999999, 0.999999))
                 + np.arctanh(np.clip(r_between, -0.999999, 0.999999)))
        r_composite = float(np.tanh(z))

    mae_within = float(np.nanmean([
        np.mean(np.abs(preds[users == u] - labs[users == u])) for u in unique_users
    ]))
    mae_between = float(np.mean(np.abs(np.asarray(means_p) - np.asarray(means_l))))

    return dict(
        r_within=r_within, p_within=p_within,
        r_between=r_between, p_between=p_between,
        r_composite=r_composite,
        mae_within=mae_within, mae_between=mae_between,
    )


def evaluate(df: pd.DataFrame, pred_valence, pred_arousal, label: str = "model") -> dict:
    out = {"model": label, "n": len(df)}
    for dim in ["valence", "arousal"]:
        preds = pred_valence if dim == "valence" else pred_arousal
        m = task1_correlation_robust(
            df["user_id"].tolist(), df["text_id"].tolist(),
            np.asarray(preds, dtype=float).tolist(), df[dim].tolist(),
        )
        for k in ["r_within", "r_between", "r_composite", "mae_within", "mae_between"]:
            out[f"{dim}_{k}"] = m[k]
    out["mean_r_composite"] = 0.5 * (out["valence_r_composite"] + out["arousal_r_composite"])
    return out



@dataclass
class ModelConfig:
    backbone: str = "microsoft/deberta-v3-base"
    pooling: str = "cls"              
    use_user_emb: bool = False
    n_users: int = 1                   # set by train_one
    user_emb_dim: int = 32
    use_format_emb: bool = False       
    format_emb_dim: int = 8
    dropout: float = 0.1
    residualize: bool = False
    # Gemma + LoRA 
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_targets: str = "attn"         


@dataclass
class TrainConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    loss: str = "mse"
    lr: float = 2e-5
    weight_decay: float = 0.01
    batch_size: int = 16
    epochs: int = 4
    warmup_frac: float = 0.10
    max_len: int = 192
    text_col: str = "text_clean"
    seed: int = 13
    label: str = "M6_no_format"
    target_dim: Optional[str] = None
    text_transform: Optional[str] = None
    head_warmup_epochs: int = 0


class ValenceArousalModel(nn.Module):
    """DeBERTa-v3 backbone + 2-output regression head, with optional user / format embeddings."""
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        from transformers import AutoModel
        self.cfg = cfg
        # IMPORTANT: force (transformers v5 honors the on-disk fp16 dtype by default. mixed with fp32 head this crashes with MPS 
        self.backbone = AutoModel.from_pretrained(cfg.backbone, dtype=torch.float32)
        hid = self.backbone.config.hidden_size
        extra = 0
        if cfg.use_user_emb:
            self.user_emb = nn.Embedding(cfg.n_users, cfg.user_emb_dim, padding_idx=0)
            extra += cfg.user_emb_dim
        if cfg.use_format_emb:
            self.format_emb = nn.Embedding(2, cfg.format_emb_dim)
            extra += cfg.format_emb_dim
        self.dropout = nn.Dropout(cfg.dropout)
        self.head    = nn.Linear(hid + extra, 2)

    def pool(self, last_hidden: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
        if self.cfg.pooling == "cls":
            return last_hidden[:, 0]
        mask = attn_mask.unsqueeze(-1).float()
        return (last_hidden * mask).sum(1) / mask.sum(1).clamp(min=1)

    def forward(self, input_ids, attention_mask, is_words=None, user_idx=None, **_):
        out = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        pooled = self.pool(out.last_hidden_state, attention_mask)
        feats = [pooled]
        if self.cfg.use_user_emb:
            feats.append(self.user_emb(user_idx))
        if self.cfg.use_format_emb:
            feats.append(self.format_emb(is_words))
        h = torch.cat(feats, dim=-1)
        return self.head(self.dropout(h))



_GEMMA_ATTN_TARGETS = (
    "q_proj.linear", "k_proj.linear",
    "v_proj.linear", "o_proj.linear",
)
_GEMMA_MLP_TARGETS = (
    "gate_proj.linear", "up_proj.linear", "down_proj.linear",
)
GEMMA_ATTN_ONLY     = _GEMMA_ATTN_TARGETS
GEMMA_ATTN_AND_MLP  = _GEMMA_ATTN_TARGETS + _GEMMA_MLP_TARGETS


class GemmaLoRAModel(nn.Module):


    def __init__(self, cfg: ModelConfig):
        super().__init__()
        from transformers import AutoModelForCausalLM
        from peft import LoraConfig, get_peft_model, TaskType
        self.cfg = cfg
        # Pooling can be set on either the ModelConfig (preferred) or fall back to "mean".
        self.pooling = cfg.pooling if cfg.pooling in ("mean", "eos", "last") else "mean"
        # LoRA target set
        if cfg.lora_targets == "attn":
            targets = GEMMA_ATTN_ONLY
        elif cfg.lora_targets == "attn+mlp":
            targets = GEMMA_ATTN_AND_MLP
        else:
            raise ValueError(f"unknown lora_targets: {cfg.lora_targets!r}")
        # Backbone in bf16 for memory; LoRA adapters and head are fp32 to keep
        # training numerics stable. Mixed-precision is supported by torch and peft.
        self.backbone = AutoModelForCausalLM.from_pretrained(
            cfg.backbone, dtype=torch.bfloat16, attn_implementation="eager",
        )
        peft_cfg = LoraConfig(
            r=cfg.lora_r, lora_alpha=cfg.lora_alpha, lora_dropout=cfg.lora_dropout,
            bias="none",
            target_modules=list(targets),
            task_type=TaskType.FEATURE_EXTRACTION,
        )
        self.backbone = get_peft_model(self.backbone, peft_cfg)
        cfg_obj = getattr(self.backbone, "config", None) or self.backbone.base_model.config
        hid = getattr(cfg_obj, "hidden_size", None)
        if hid is None:
            hid = cfg_obj.text_config.hidden_size
        self.hidden_size = hid
        self.dropout = nn.Dropout(cfg.dropout)
        self.head = nn.Linear(hid, 2)  # fp32 by default

    def _pool(self, last_hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        if self.pooling == "mean":
            mask = attention_mask.unsqueeze(-1).to(last_hidden.dtype)
            summed = (last_hidden * mask).sum(dim=1)
            denom  = mask.sum(dim=1).clamp(min=1)
            return (summed / denom).float()
        if self.pooling in ("last", "eos"):
            # Requires right-padding so the last real token is at attention_mask.sum() - 1.
            # The dataset / collator is responsible for that — see `EMADataset` callers.
            seq_lens = attention_mask.sum(dim=1) - 1                  # [B]
            idx = seq_lens.unsqueeze(1).unsqueeze(2).expand(-1, 1, last_hidden.size(-1))
            return last_hidden.gather(1, idx).squeeze(1).float()       # [B, H]
        raise ValueError(f"unknown pooling: {self.pooling!r}")

    def forward(self, input_ids, attention_mask, is_words=None, user_idx=None, **_):
        out = self.backbone(
            input_ids=input_ids, attention_mask=attention_mask,
            output_hidden_states=True, return_dict=True,
        )
        last_hidden = out.hidden_states[-1]  # [B, T, H], bf16
        pooled = self._pool(last_hidden, attention_mask)
        return self.head(self.dropout(pooled))

    # --- LoRA freeze/unfreeze helpers for staged training -----------------
    def freeze_backbone_adapters(self):
        """Freeze all LoRA-trainable params; only the head can train.
        Used for head-warmup at the start of training."""
        for n, p in self.backbone.named_parameters():
            if "lora_" in n:
                p.requires_grad_(False)

    def unfreeze_backbone_adapters(self):
        for n, p in self.backbone.named_parameters():
            if "lora_" in n:
                p.requires_grad_(True)



# Dataset & collator

class EMADataset(Dataset):
    def __init__(self, df: pd.DataFrame, tokenizer, text_col: str = "text_clean",
                 max_len: int = 192, include_user: bool = False,
                 user2idx: dict | None = None):
        self.df = df.reset_index(drop=True)
        self.tok = tokenizer
        self.text_col = text_col
        self.max_len = max_len
        self.include_user = include_user
        self.user2idx = user2idx or {}

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        row = self.df.iloc[i]
        enc = self.tok(row[self.text_col], truncation=True, max_length=self.max_len)
        item = {k: torch.tensor(v) for k, v in enc.items()}
        item["labels"] = torch.tensor(
            [float(row["valence"]), float(row["arousal"])], dtype=torch.float32
        )
        if self.include_user:
            item["user_idx"] = torch.tensor(self.user2idx.get(row["user_id"], 0), dtype=torch.long)
        item["is_words"] = torch.tensor(int(row["is_words"]), dtype=torch.long)
        return item


def collate(batch, pad_token_id: int):
    max_len = max(b["input_ids"].size(0) for b in batch)
    def pad(t, length, value=0):
        return torch.cat([t, torch.full((length - t.size(0),), value, dtype=t.dtype)])
    out = {
        "input_ids":      torch.stack([pad(b["input_ids"],      max_len, pad_token_id) for b in batch]),
        "attention_mask": torch.stack([pad(b["attention_mask"], max_len, 0)            for b in batch]),
        "labels":         torch.stack([b["labels"]   for b in batch]),
        "is_words":       torch.stack([b["is_words"] for b in batch]),
    }
    if "user_idx" in batch[0]:
        out["user_idx"] = torch.stack([b["user_idx"] for b in batch])
    return out


# Loss functions

def ccc_loss(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Lin's concordance correlation — sums (1-CCC) over output dimensions."""
    mu_p, mu_t = pred.mean(0), target.mean(0)
    var_p, var_t = pred.var(0, unbiased=False), target.var(0, unbiased=False)
    cov = ((pred - mu_p) * (target - mu_t)).mean(0)
    ccc = 2 * cov / (var_p + var_t + (mu_p - mu_t).pow(2) + eps)
    return (1 - ccc).sum()


def make_loss(kind: str, target_dim: Optional[str] = None):
    """If target_dim is given ('valence' or 'arousal'), the loss only includes that
    output column. Otherwise sums/averages over both. Used by single-task ablations."""
    if target_dim is None:
        if kind == "mse":     return lambda p, t: nn.functional.mse_loss(p, t)
        if kind == "ccc":     return lambda p, t: ccc_loss(p, t)
        if kind == "mse+ccc": return lambda p, t: nn.functional.mse_loss(p, t) + 0.5 * ccc_loss(p, t)
        raise ValueError(f"unknown loss: {kind!r}")

    idx = 0 if target_dim == "valence" else 1
    if target_dim not in ("valence", "arousal"):
        raise ValueError(f"target_dim must be 'valence' or 'arousal', got {target_dim!r}")
    if kind == "mse":
        return lambda p, t: nn.functional.mse_loss(p[:, idx], t[:, idx])
    if kind == "ccc":
        # CCC reduces to (pred[:,idx], target[:,idx]) — call ccc_loss on 1-D tensors reshaped to [B,1]
        return lambda p, t: ccc_loss(p[:, idx:idx+1], t[:, idx:idx+1])
    if kind == "mse+ccc":
        return lambda p, t: (nn.functional.mse_loss(p[:, idx], t[:, idx])
                              + 0.5 * ccc_loss(p[:, idx:idx+1], t[:, idx:idx+1]))
    raise ValueError(f"unknown loss: {kind!r}")



def set_seed(seed: int):
    import random
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


def pick_device(prefer: str | None = None) -> str:
    """Returns 'cuda' > 'mps' > 'cpu' by default. `prefer` overrides if available."""
    if prefer and prefer != "auto":
        return prefer
    if torch.cuda.is_available():            return "cuda"
    if torch.backends.mps.is_available():    return "mps"
    return "cpu"



# Training

def _linear_warmup_then_decay(step: int, total: int, warmup: int) -> float:
    if step < warmup:
        return step / max(1, warmup)
    return max(0.0, (total - step) / max(1, total - warmup))


def _predict(model, loader, device) -> np.ndarray:
    model.eval()
    outs = []
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outs.append(model(**batch).detach().float().cpu().numpy())
    return np.concatenate(outs, axis=0)



def _gemma_chat_wrap(tokenizer, text: str) -> str:
    """Wrap text as a Gemma chat-template user turn. The instruction prompt is fixed.
    Returns a string that's ready to feed to the tokenizer (no further chat templating).
    """
    msgs = [{
        "role": "user",
        "content": (
            "Rate the emotional valence (from -2 most negative to +2 most positive) "
            "and arousal (from 0 calm to 2 highly activated) implied by the following "
            f"short text:\n\n{text}"
        ),
    }]
    return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


def _apply_text_transform(df: pd.DataFrame, tokenizer, kind: str) -> pd.DataFrame:
    """Return a copy of df with text_clean re-written per `kind`. Identity if kind is None."""
    if kind is None:
        return df
    if kind == "gemma_chat":
        out = df.copy()
        out["text_clean"] = out["text_clean"].map(lambda t: _gemma_chat_wrap(tokenizer, t))
        return out
    raise ValueError(f"unknown text_transform: {kind!r}")


def _cache_key(cfg: TrainConfig, train_df: pd.DataFrame, val_df: pd.DataFrame,
               pick_best_epoch: bool) -> str:
    cfg_str = json.dumps(asdict(cfg), sort_keys=True)
    cols = ["user_id", "text_id", "text_clean", "valence", "arousal", "is_words"]
    def digest(df):
        h = pd.util.hash_pandas_object(df[cols], index=False).values.tobytes()
        return hashlib.sha256(h).hexdigest()[:12]
    h = hashlib.sha256(
        f"{cfg_str}|{digest(train_df)}|{digest(val_df)}|pick={pick_best_epoch}".encode()
    ).hexdigest()[:16]
    return f"{cfg.label}_{h}"


def train_one(
    cfg: TrainConfig,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    tokenizer,
    device: str,
    pick_best_epoch: bool = True,
    cache_dir: Path | None = None,
    verbose: bool = True,
    model_cls=None,
    grad_accum: int = 1,
) -> dict:
    """Train a single config. return {epoch, score, preds, metrics}.

    """
    if model_cls is None:
        model_cls = ValenceArousalModel
    if cache_dir is not None:
        cache_dir = Path(cache_dir); cache_dir.mkdir(parents=True, exist_ok=True)
        key = _cache_key(cfg, train_df, val_df, pick_best_epoch)
        path = cache_dir / f"{key}.pkl"
        if path.exists():
            if verbose: log.info(f"[cache hit] {path.name}")
            return pickle.loads(path.read_bytes())

    set_seed(cfg.seed)

    # User vocab spans train+val so val-seen users have an embedding.
    user_ids = sorted(set(train_df.user_id) | set(val_df.user_id))
    user2idx = {u: i + 1 for i, u in enumerate(user_ids)}
    cfg.model.n_users = len(user2idx) + 1

    # Optional residualization w.r.t. user-train means.
    train_user_mean = train_df.groupby("user_id")[["valence", "arousal"]].mean()
    gv, ga = float(train_df["valence"].mean()), float(train_df["arousal"].mean())
    if cfg.model.residualize:
        t_df = train_df.copy()
        t_df["valence"] = t_df["valence"] - t_df["user_id"].map(train_user_mean["valence"])
        t_df["arousal"] = t_df["arousal"] - t_df["user_id"].map(train_user_mean["arousal"])
    else:
        t_df = train_df

    # Optional text transform (e.g. wrap in Gemma chat template). Applied to both train and val.
    t_df  = _apply_text_transform(t_df,  tokenizer, cfg.text_transform)
    v_df  = _apply_text_transform(val_df, tokenizer, cfg.text_transform)

    ds_tr = EMADataset(t_df,  tokenizer, cfg.text_col, cfg.max_len,
                       include_user=cfg.model.use_user_emb, user2idx=user2idx)
    ds_va = EMADataset(v_df, tokenizer, cfg.text_col, cfg.max_len,
                       include_user=cfg.model.use_user_emb, user2idx=user2idx)
    coll  = lambda b: collate(b, tokenizer.pad_token_id)
    dl_tr = DataLoader(ds_tr, batch_size=cfg.batch_size, shuffle=True,  collate_fn=coll)
    dl_va = DataLoader(ds_va, batch_size=cfg.batch_size, shuffle=False, collate_fn=coll)

    model = model_cls(cfg.model).to(device)

    if cfg.head_warmup_epochs > 0 and hasattr(model, "freeze_backbone_adapters"):
        model.freeze_backbone_adapters()
        if verbose:
            log.info(f"  head-only warmup: backbone adapters frozen for the first "
                     f"{cfg.head_warmup_epochs} epoch(s)")
    # Only optimize parameters that require grad — important for LoRA, where the backbone is frozen.
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optim = AdamW(trainable_params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    loss_fn = make_loss(cfg.loss, target_dim=cfg.target_dim)
    if verbose:
        n_total = sum(p.numel() for p in model.parameters())
        n_train = sum(p.numel() for p in trainable_params)
        log.info(f"  params: total={n_total/1e6:.1f}M  trainable={n_train/1e6:.2f}M "
                 f"({100*n_train/max(1,n_total):.2f}%)")

    # total_steps counts OPTIMIZER steps, which equals micro-batches / grad_accum.
    import math
    total_steps = math.ceil(len(dl_tr) / grad_accum) * cfg.epochs
    warmup = int(total_steps * cfg.warmup_frac)
    step = 0

    best = {"epoch": -1, "score": -float("inf"), "preds": None, "metrics": None}
    last = {"epoch": -1, "score": -float("inf"), "preds": None, "metrics": None}
    train_losses: list[float] = []
    val_losses: list[float] = []

    for epoch in range(cfg.epochs):
        # Unfreeze adapters at the boundary of the warmup phase.
        if (epoch == cfg.head_warmup_epochs and cfg.head_warmup_epochs > 0
                and hasattr(model, "unfreeze_backbone_adapters")):
            model.unfreeze_backbone_adapters()
            # Add the newly-trainable LoRA params to the optimizer. They'll get fresh
            # Adam state (zero m/v), while the head's optimizer state is preserved.
            existing = {id(p) for g in optim.param_groups for p in g["params"]}
            new_params = [p for p in model.parameters()
                          if p.requires_grad and id(p) not in existing]
            if new_params:
                optim.add_param_group({"params": new_params,
                                       "lr": optim.param_groups[0]["lr"],
                                       "weight_decay": cfg.weight_decay})
            if verbose:
                log.info(f"  epoch {epoch+1}: unfreezing LoRA adapters "
                         f"(+{sum(p.numel() for p in new_params)/1e6:.2f}M trainable)")
        model.train()
        running = 0.0
        optim.zero_grad()
        micro_step = 0
        for batch in dl_tr:
            batch = {k: v.to(device) for k, v in batch.items()}
            labels = batch.pop("labels")
            pred = model(**batch)
            loss = loss_fn(pred, labels)
            (loss / grad_accum).backward()
            running += loss.item() * len(labels)
            micro_step += 1
            if micro_step % grad_accum == 0:
                for pg in optim.param_groups:
                    pg["lr"] = cfg.lr * _linear_warmup_then_decay(step, total_steps, warmup)
                optim.step()
                optim.zero_grad()
                step += 1
        # flush any leftover gradients from a partial accumulation window
        if micro_step % grad_accum != 0:
            for pg in optim.param_groups:
                pg["lr"] = cfg.lr * _linear_warmup_then_decay(step, total_steps, warmup)
            optim.step()
            optim.zero_grad()
            step += 1

        preds = _predict(model, dl_va, device)
        pv, pa = preds[:, 0], preds[:, 1]
        if cfg.model.residualize:
            prior_v = val_df["user_id"].map(train_user_mean["valence"]).fillna(gv).to_numpy()
            prior_a = val_df["user_id"].map(train_user_mean["arousal"]).fillna(ga).to_numpy()
            pv = pv + prior_v; pa = pa + prior_a

        if cfg.target_dim == "valence":
            pa = np.full_like(pa, ga)
        elif cfg.target_dim == "arousal":
            pv = np.full_like(pv, gv)

        res = evaluate(val_df, pv, pa, label=cfg.label)

        if cfg.target_dim == "valence":
            score = res["valence_r_composite"]
        elif cfg.target_dim == "arousal":
            score = res["arousal_r_composite"]
        else:
            score = res["mean_r_composite"]
        train_losses.append(running / len(t_df))
        val_losses.append(float(np.mean(
            (pv - val_df["valence"].to_numpy()) ** 2
            + (pa - val_df["arousal"].to_numpy()) ** 2
        )))
        if verbose:
            log.info(
                f"  epoch {epoch+1}/{cfg.epochs}  loss={running/len(t_df):.4f}  "
                f"val r_composite (V={res['valence_r_composite']:+.3f}  "
                f"A={res['arousal_r_composite']:+.3f}  mean={res['mean_r_composite']:+.3f})"
            )
        last = {"epoch": epoch, "score": score, "preds": (pv, pa), "metrics": res}
        if score > best["score"]:
            best = dict(last)

    chosen = best if pick_best_epoch else last
    chosen["metrics"]["best_val_epoch"] = best["epoch"]
    chosen["metrics"]["selection"]      = "best_val" if pick_best_epoch else "last"
    chosen["loss_history"] = {"train": train_losses, "val": val_losses}

    del model
    if torch.backends.mps.is_available(): torch.mps.empty_cache()
    if torch.cuda.is_available():         torch.cuda.empty_cache()

    if cache_dir is not None:
        path.write_bytes(pickle.dumps(chosen))
    return chosen

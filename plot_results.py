from __future__ import annotations
import glob
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

ARTIFACTS = Path("artifacts/paper_results")
FIGURES   = Path("figures")
FIGURES.mkdir(exist_ok=True)

plt.rcParams.update({
    "font.family":    "sans-serif",
    "font.size":      11,
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "figure.dpi":         150,
})


def load(model: str) -> dict:
    p = ARTIFACTS / model / "metrics.json"
    return json.loads(p.read_text())

def r(metrics: dict, split: str, variant: str, dim: str = "mean") -> float:
    key = f"{dim}_r_composite"
    return metrics[split][variant][key]


# Figure 1 — Main results bar chart

def fig1_main_results():
    models = [
        ("Gemma-4-E4B LoRA",   "gemma_lora",            "#f4a261"),
        ("DeBERTa-base\n(CCC loss, CLS)",   "deberta_base_ccc",      "#80b87a"),
        ("DeBERTa-base\n(MSE, mean pool)",  "deberta_base_mean_pool","#c0a87a"),
        ("DeBERTa-large\n(MSE, CLS)",       "deberta_large",         "#aaa0d4"),
        ("DeBERTa-base\n(MSE, CLS)",        "deberta_base",          "#e06c6c"),
    ]
    WINNER = 0.611

    labels = [m[0] for m in models]
    colors = [m[2] for m in models]
    mkeys  = [m[1] for m in models]

    text_vals  = []
    long_vals  = []
    for mk in mkeys:
        d = load(mk)
        text_vals.append(r(d, "test", "text_only"))
        long_vals.append(r(d, "test", "longitudinal_default"))

    x     = np.arange(len(models))
    width = 0.35

    fig, ax = plt.subplots(figsize=(6, 5))
    bars_t = ax.bar(x - width/2, text_vals, width, label="text only",         color=colors, alpha=0.55, edgecolor="white")
    bars_l = ax.bar(x + width/2, long_vals, width, label="+ longitudinal",    color=colors, alpha=1.0,  edgecolor="white")

    ax.axhline(WINNER, color="#cc2222", linewidth=1.8, linestyle="--", zorder=3,
               label=f"UKP_Psycontrol {WINNER:.3f}")
    ax.axhline(0.428, color="#888", linewidth=1.2, linestyle=":", zorder=3,
               label="UKP BERT baseline ≈0.428")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=12, rotation=40, ha="right")
    ax.set_ylabel("Test mean $r_{\\mathrm{composite}}$")
    ax.set_ylim(0.25, 0.75)
    ax.set_title("Text-only vs. longitudinal calibration", fontweight="bold")

    # value labels on top of longitudinal bars
    for bar in bars_l:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.004,
                f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=12, fontweight="bold")
    for bar in bars_t:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.004,
                f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=11, color="#555")

    patch_t = mpatches.Patch(facecolor="#aaa", alpha=0.55, edgecolor="white", label="text only")
    patch_l = mpatches.Patch(facecolor="#aaa", alpha=1.0,  edgecolor="white", label="+ longitudinal")
    winner_line = plt.Line2D([0],[0], color="#cc2222", linewidth=1.8, linestyle="--",
                              label=f"UKP_Psycontrol {WINNER:.3f}")
    baseline_line = plt.Line2D([0],[0], color="#888", linewidth=1.2, linestyle=":",
                                label="UKP BERT baseline ≈0.428")
    ax.legend(handles=[patch_t, patch_l, winner_line, baseline_line],
              loc="lower right", fontsize=12)

    plt.tight_layout()
    plt.savefig(FIGURES / "fig3_main_results.png", bbox_inches="tight")
    plt.close()
    print("fig1 done")



# Figure 2 — Loss curves (CCC vs mean_pool ablations)

def fig2_loss_curves():
    def avg_losses(model: str, pattern: str = "validation_oof_*.losses.json"):
        files = sorted(glob.glob(str(ARTIFACTS / model / "worker_outputs" / pattern)))
        if not files:
            return None, None
        trains, vals = [], []
        for f in files:
            d = json.loads(Path(f).read_text())
            trains.append(d["train"])
            vals.append(d["val"])
        return (np.mean(trains, axis=0), np.mean(vals, axis=0))

    ccc_tr,  ccc_val  = avg_losses("deberta_base_ccc")
    mean_tr, mean_val = avg_losses("deberta_base_mean_pool")

    epochs = [1, 2, 3, 4]
    fig, axes = plt.subplots(1, 2, figsize=(7, 3.2), sharey=False)

    # Left: CCC loss model
    ax = axes[0]
    ax.plot(epochs, ccc_tr,  "o-", color="#e06c6c", label="Train (CCC loss)")
    ax.plot(epochs, ccc_val, "s--", color="#4a90d4", label="Val (MSE)")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("CCC loss — CLS pool", fontweight="bold", fontsize=11)
    ax.legend(fontsize=10)
    ax.set_xticks(epochs)
    ax.set_ylim(bottom=0)
    #for e, (tv, vv) in enumerate(zip(ccc_tr, ccc_val)):
    #    ax.annotate(f"{tv:.3f}", (epochs[e], tv), textcoords="offset points",
    #                xytext=(0, 7), ha="center", fontsize=11, color="#e06c6c")
    #    ax.annotate(f"{vv:.3f}", (epochs[e], vv), textcoords="offset points",
    #                xytext=(0, -14), ha="center", fontsize=11, color="#4a90d4")

    # Right: mean pooling model
    
    ax = axes[1]
    ax.plot(epochs, mean_tr,  "o-",  color="g", label="Train (MSE)")
    ax.plot(epochs, mean_val, "s--", color="orange", label="Val (MSE)")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("MSE loss — mean pool", fontweight="bold", fontsize=11)
    ax.legend(fontsize=8)
    ax.set_xticks(epochs)
    ax.set_ylim(bottom=0)
    #for e, (tv, vv) in enumerate(zip(mean_tr, mean_val)):
    #    ax.annotate(f"{tv:.3f}", (epochs[e], tv), textcoords="offset points",
    #                xytext=(0, 7), ha="center", fontsize=11, color="g")
    #    ax.annotate(f"{vv:.3f}", (epochs[e], vv), textcoords="offset points",
    #                xytext=(0, -14), ha="center", fontsize=11, color="orange")

    plt.tight_layout()
    plt.savefig(FIGURES / "fig2_loss_curves.png", bbox_inches="tight")
    plt.close()
    print("fig2 done")



# Figure 3 — Feature weights (top 12)

def fig3_feature_weights():
    d = load("deberta_base")
    weights = d["feature_weights"]["longitudinal_default"][:12]

    features = [w["feature"] for w in weights]
    val_w    = [w["weight_valence"] for w in weights]
    aro_w    = [w["weight_arousal"] for w in weights]

    def shorten(f: str) -> str:
        return (f.replace("hist_mean_valence_all", "hist: mean val (all)")
                 .replace("hist_mean_arousal_all", "hist: mean aro (all)")
                 .replace("hist_mean_valence_last3", "hist: mean val (last-3)")
                 .replace("hist_mean_arousal_last3", "hist: mean aro (last-3)")
                 .replace("hist_days_since_first", "hist: days since first")
                 .replace("hist_n_words_prior", "hist: n word-entries prior")
                 .replace("hist_phase_gap", "hist: phase gap")
                 .replace("hist_n_prior", "hist: n prior entries")
                 .replace("hist_mean_arousal_all", "hist: mean aro (all)")
                 .replace("text_pred_valence_x_has_history", "text_pred_val × has_hist")
                 .replace("text_pred_arousal_x_has_history", "text_pred_aro × has_hist")
                 .replace("text_pred_valence", "text pred: valence")
                 .replace("text_pred_arousal", "text pred: arousal")
                 .replace("days_from_global_start", "days from study start")
                 .replace("collection_phase", "collection phase"))

    labels = [shorten(f) for f in features]
    y = np.arange(len(features))
    fig, ax = plt.subplots(figsize=(6, 6.5))

    ax.barh(y + 0.2, val_w, height=0.35, color="#e06c6c", label="Valence weight")
    ax.barh(y - 0.2, aro_w, height=0.35, color="#4a90d4", label="Arousal weight")
    ax.axvline(0, color="#333", linewidth=0.8)

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=12)
    ax.invert_yaxis()
    ax.set_xlabel("Standardised Ridge weight")
    ax.set_title("Ridge weights — DeBERTa-base\n(top 12, standardised features)",
                 fontweight="bold")
    ax.legend(fontsize=12)
    ax.set_xlim(-0.55, 1.0)

    plt.tight_layout()
    plt.savefig(FIGURES / "fig4_feature_weights.png", bbox_inches="tight")
    plt.close()
    print("fig3 done")



# Figure 5 Ablation: CCC vs MSE vs mean pool  

def fig6_ablation():
    rows = [
        ("MSE + CLS",  "deberta_base",          "#e06c6c"),
        ("CCC + CLS",  "deberta_base_ccc",      "#80b87a"),
        ("MSE + mean", "deberta_base_mean_pool","#f4a261"),
    ]
    labels = [r[0] for r in rows]
    colors = [r[2] for r in rows]
    mkeys  = [r[1] for r in rows]

    variants = ["text_only", "longitudinal_default"]
    var_labels = ["text only", "+ longitudinal"]
    dims = [("valence", "Valence $r_c$"), ("arousal", "Arousal $r_c$"), ("mean", "Mean $r_c$")]

    fig, axes = plt.subplots(1, 3, figsize=(6.5, 2.6), sharey=True)

    x     = np.arange(len(rows))
    width = 0.30

    for ax, (dk, dim_label) in zip(axes, dims):
        for vi, var in enumerate(variants):
            vals = []
            for mk in mkeys:
                d = load(mk)
                vals.append(r(d, "test", var, dim=dk))
            offset = (vi - 0.5) * width
            bars = ax.bar(x + offset, vals, width,
                          color=colors, alpha=[0.5, 1.0][vi], edgecolor="white",
                          label=var_labels[vi])
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=10, rotation=35, ha="right")
        ax.set_title(dim_label, fontweight="bold", fontsize=11)
        ax.set_ylim(0.40, 0.74)
        ax.yaxis.grid(True, linestyle="--", alpha=0.4, zorder=0)
        ax.set_axisbelow(True)
        ax.tick_params(axis="y", labelsize=10)
        ax.set_ylabel("$r_{composite}$", fontsize=10) if ax == axes[0] else ax.set_ylabel("")

    patch_t = mpatches.Patch(facecolor="#aaa", alpha=0.5, label="text only")
    patch_l = mpatches.Patch(facecolor="#aaa", alpha=1.0, label="+ longitudinal")
    fig.legend(handles=[patch_t, patch_l], loc="lower center", fontsize=10,
               ncols=2, bbox_to_anchor=(0.5, -0.02))
    plt.tight_layout(rect=[0, 0.12, 1, 1])
    plt.savefig(FIGURES / "fig5_ablation.png", bbox_inches="tight")
    plt.close()
    print("fig6 done")


if __name__ == "__main__":
    fig1_main_results()
    fig2_loss_curves()
    fig3_feature_weights()
    fig6_ablation()
    print("All figures saved to figures/")

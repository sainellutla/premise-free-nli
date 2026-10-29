"""Step 2: load SNLI (and MultiNLI eval splits), compute dataset statistics and figures.

Run from the repo root:  python src/data.py
"""
import re
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from datasets import load_dataset
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

SEED = 42
ROOT = Path(__file__).resolve().parent.parent
CACHE_DIR = str(ROOT / "data" / "hf_cache")
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"

LABEL_NAMES = {0: "entailment", 1: "neutral", 2: "contradiction"}
LABEL_ORDER = ["entailment", "neutral", "contradiction"]
SPLITS = ["train", "validation", "test"]

NEGATIONS = {"no", "not", "nobody", "never", "nothing", "none", "nor",
             "cannot", "neither", "nowhere"}
STOPWORDS = set(ENGLISH_STOP_WORDS) - NEGATIONS
TOKEN_RE = re.compile(r"\b\w+\b")


def load_snli(verbose=False):
    """Return {split: DataFrame(premise, hypothesis, label)} with label -1 dropped."""
    ds = load_dataset("stanfordnlp/snli", cache_dir=CACHE_DIR)
    out = {}
    for split in SPLITS:
        df = ds[split].to_pandas()
        n_before = len(df)
        df = df[df["label"] != -1].reset_index(drop=True)
        if verbose:
            print(f"SNLI {split}: {n_before} raw, dropped {n_before - len(df)} with label -1, kept {len(df)}")
        out[split] = df
    return out


def load_mnli_eval():
    """Return MultiNLI validation_matched and validation_mismatched as DataFrames."""
    ds = load_dataset("nyu-mll/multi_nli", cache_dir=CACHE_DIR)
    return {s: ds[s].to_pandas() for s in ["validation_matched", "validation_mismatched"]}


def setup_plot_style():
    sns.set_theme(style="whitegrid", palette="colorblind", font_scale=1.2)
    plt.rcParams.update({"savefig.dpi": 300, "figure.dpi": 100})


def main():
    RESULTS.mkdir(exist_ok=True)
    FIGURES.mkdir(exist_ok=True)

    snli = load_snli(verbose=True)

    mnli = load_mnli_eval()
    for name, df in mnli.items():
        print(f"MultiNLI {name}: {len(df)} examples loaded; label counts {df['label'].value_counts().sort_index().to_dict()}")

    # ---- dataset stats ----
    rows = []
    for split in SPLITS:
        df = snli[split].copy()
        df["label_name"] = df["label"].map(LABEL_NAMES)
        df["hyp_len"] = df["hypothesis"].str.split().str.len()
        for lab in LABEL_ORDER:
            sub = df[df["label_name"] == lab]
            rows.append({"split": split, "label": lab, "n_examples": len(sub),
                         "pct_of_split": round(100 * len(sub) / len(df), 2),
                         "hyp_len_mean": round(sub["hyp_len"].mean(), 3),
                         "hyp_len_median": sub["hyp_len"].median()})
        rows.append({"split": split, "label": "ALL", "n_examples": len(df),
                     "pct_of_split": 100.0,
                     "hyp_len_mean": round(df["hyp_len"].mean(), 3),
                     "hyp_len_median": df["hyp_len"].median()})
        snli[split] = df
    stats = pd.DataFrame(rows)
    stats.to_csv(RESULTS / "dataset_stats.csv", index=False)
    print("\n" + stats.to_string(index=False))

    # ---- top frequent words (train) ----
    train = snli["train"]
    top_rows = []
    for lab in LABEL_ORDER:
        counter = Counter()
        n_hyp = 0
        for h in train.loc[train["label_name"] == lab, "hypothesis"]:
            n_hyp += 1
            counter.update(t for t in TOKEN_RE.findall(h.lower()) if t not in STOPWORDS)
        for rank, (w, c) in enumerate(counter.most_common(20), start=1):
            top_rows.append({"label": lab, "rank": rank, "word": w, "count": c,
                             "per_1000_hypotheses": round(1000 * c / n_hyp, 2)})
    top = pd.DataFrame(top_rows)
    top.to_csv(RESULTS / "top_frequent_words.csv", index=False)
    print("\n" + top.pivot(index="rank", columns="label", values="word")[LABEL_ORDER].to_string())

    # ---- figures ----
    setup_plot_style()
    colors = sns.color_palette("colorblind", 3)

    # Fig 2: label distribution per split
    plot_df = stats[stats["label"] != "ALL"]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
    for ax, split in zip(axes, SPLITS):
        d = plot_df[plot_df["split"] == split]
        bars = ax.bar(d["label"], d["n_examples"], color=colors)
        for b, pct in zip(bars, d["pct_of_split"]):
            ax.annotate(f"{pct:.1f}%", (b.get_x() + b.get_width() / 2, b.get_height()),
                        ha="center", va="bottom", fontsize=11)
        ax.set_title(f"{split} (n = {d['n_examples'].sum():,})")
        ax.set_xlabel("Gold label")
        ax.set_ylabel("Number of examples")
        ax.set_ylim(0, d["n_examples"].max() * 1.12)
        ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
        ax.tick_params(axis="x", rotation=15)
    fig.suptitle("SNLI label distribution per split (label −1 removed)")
    fig.tight_layout()
    fig.savefig(FIGURES / "fig2_label_distribution.png", dpi=300)
    plt.close(fig)

    # Fig 3: hypothesis length by label (train)
    fig, ax = plt.subplots(figsize=(7.5, 5))
    sns.boxplot(data=train, x="label_name", y="hyp_len", order=LABEL_ORDER,
                hue="label_name", hue_order=LABEL_ORDER, palette=colors, legend=False,
                showfliers=True, flierprops={"marker": ".", "markersize": 2, "alpha": 0.3},
                ax=ax)
    ax.set_xlabel("Gold label")
    ax.set_ylabel("Hypothesis length (words)")
    ax.set_title(f"SNLI train: hypothesis length by label (n = {len(train):,})")
    fig.tight_layout()
    fig.savefig(FIGURES / "fig3_hypothesis_length.png", dpi=300)
    plt.close(fig)
    print("\nSaved figures/fig2_label_distribution.png and figures/fig3_hypothesis_length.png")


if __name__ == "__main__":
    np.random.seed(SEED)
    main()

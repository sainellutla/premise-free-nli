"""Step B2: easy/hard split of SNLI test.

"easy" = hypothesis-only LogisticRegression predicts the example correctly; otherwise "hard".
The full-input LogisticRegression is then evaluated on each subset. Uses saved models only.
Run from the repo root:  python src/easy_hard.py
"""
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import accuracy_score, f1_score

from data import LABEL_NAMES, LABEL_ORDER, load_snli, setup_plot_style
from features import transform_full

ROOT = Path(__file__).resolve().parent.parent
RESULTS, FIGURES, MODELS = ROOT / "results", ROOT / "figures", ROOT / "models"


def main():
    test = load_snli()["test"]
    y = test["label"].to_numpy()

    hyp_pred = joblib.load(MODELS / "logreg_hyp.joblib").predict(
        joblib.load(MODELS / "hyp_tfidf.joblib").transform(test["hypothesis"]))
    full_pred = joblib.load(MODELS / "logreg_full.joblib").predict(
        transform_full(joblib.load(MODELS / "full_transformers.joblib"), test))
    easy = hyp_pred == y

    rows = []
    for name, mask in [("easy", easy), ("hard", ~easy), ("all", np.ones_like(easy))]:
        yt, yp = y[mask], full_pred[mask]
        row = {"subset": name, "n_examples": int(mask.sum()),
               "pct_of_test": 100 * mask.mean()}
        for lab in [0, 1, 2]:
            n = int((yt == lab).sum())
            row[f"n_{LABEL_NAMES[lab]}"] = n
            row[f"pct_{LABEL_NAMES[lab]}"] = 100 * n / len(yt)
        row["full_input_accuracy"] = accuracy_score(yt, yp)
        row["full_input_macro_f1"] = f1_score(yt, yp, average="macro", labels=[0, 1, 2], zero_division=0)
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS / "easy_hard.csv", index=False, float_format="%.4f")
    print(df.to_string(index=False))

    # Figure 9
    setup_plot_style()
    colors = sns.color_palette("colorblind", 3)
    sub = df[df.subset != "all"].reset_index(drop=True)
    overall = df[df.subset == "all"]["full_input_accuracy"].iloc[0]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), gridspec_kw={"width_ratios": [1, 1.4]})

    bars = ax1.bar([f"{s}\n(n = {n:,})" for s, n in zip(sub.subset, sub.n_examples)],
                   sub.full_input_accuracy, color="#6e6e6e", width=0.6)
    for b, v in zip(bars, sub.full_input_accuracy):
        ax1.annotate(f"{v:.3f}", (b.get_x() + b.get_width() / 2, v), ha="center", va="bottom", fontsize=11)
    ax1.axhline(overall, ls="--", color="black", lw=1.2, label=f"All test ({overall:.3f})")
    ax1.set_ylim(0, 1)
    ax1.set_xlabel("Subset (by hypothesis-only LogReg correctness)")
    ax1.set_ylabel("Full-input LogReg accuracy")
    ax1.set_title("Full-input accuracy on easy vs. hard")
    ax1.legend(loc="upper right", fontsize=10)

    x = np.arange(len(sub))
    w = 0.26
    for i, lab in enumerate(LABEL_ORDER):
        vals = sub[f"pct_{lab}"].to_numpy()
        bb = ax2.bar(x + (i - 1) * w, vals, w, label=lab, color=colors[i])
        for b, v in zip(bb, vals):
            ax2.annotate(f"{v:.1f}%", (b.get_x() + b.get_width() / 2, v), ha="center", va="bottom", fontsize=10)
    ax2.set_xticks(x)
    ax2.set_xticklabels(sub.subset)
    ax2.set_ylim(0, 50)
    ax2.set_xlabel("Subset")
    ax2.set_ylabel("% of subset")
    ax2.set_title("Gold label distribution within each subset")
    ax2.legend(title="Gold label", loc="upper right", fontsize=10, ncol=3)
    fig.suptitle("SNLI test split by whether hypothesis-only LogReg is correct")
    fig.tight_layout()
    fig.savefig(FIGURES / "fig9_easy_hard.png", dpi=300)
    plt.close(fig)
    print("Saved figures/fig9_easy_hard.png")


if __name__ == "__main__":
    main()

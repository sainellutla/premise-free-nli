"""Step B1: apply SNLI-trained models, unchanged, to MultiNLI validation matched/mismatched.

Uses the saved models and the saved (train-fitted) vectorizers/scaler; nothing is refit.
Run from the repo root:  python src/transfer.py
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

from data import load_mnli_eval, load_snli, setup_plot_style
from features import transform_full

ROOT = Path(__file__).resolve().parent.parent
RESULTS, FIGURES, MODELS = ROOT / "results", ROOT / "figures", ROOT / "models"

HYP_MODELS = {"LogReg (hyp-only)": "logreg_hyp.joblib",
              "MultinomialNB (hyp-only)": "multinomialnb_hyp.joblib",
              "LinearSVC (hyp-only)": "linearsvc_hyp.joblib"}
FULL_NAME = "LogReg (full input)"
DUMMY_NAME = "Dummy (most frequent)"


def main():
    snli_test = load_snli()["test"]
    datasets = {"SNLI test": snli_test}
    for split, df in load_mnli_eval().items():
        n0 = len(df)
        df = df[df["label"] != -1].reset_index(drop=True)
        print(f"MultiNLI {split}: {n0} examples, dropped {n0 - len(df)} with label -1, kept {len(df)}")
        datasets["MNLI " + split.replace("validation_", "")] = df

    hyp_vec = joblib.load(MODELS / "hyp_tfidf.joblib")
    full_tf = joblib.load(MODELS / "full_transformers.joblib")
    models = {DUMMY_NAME: joblib.load(MODELS / "dummy_hyp.joblib"),
              **{n: joblib.load(MODELS / f) for n, f in HYP_MODELS.items()},
              FULL_NAME: joblib.load(MODELS / "logreg_full.joblib")}

    rows = []
    for ds_name, df in datasets.items():
        y = df["label"].to_numpy()
        Xh = hyp_vec.transform(df["hypothesis"])
        Xf = transform_full(full_tf, df)
        for m_name, model in models.items():
            pred = model.predict(Xf if m_name == FULL_NAME else Xh)
            rows.append({"model": m_name, "dataset": ds_name, "n_examples": len(y),
                         "accuracy": accuracy_score(y, pred),
                         "macro_f1": f1_score(y, pred, average="macro", labels=[0, 1, 2], zero_division=0)})
            print(f"{ds_name:<17} {m_name:<26} acc={rows[-1]['accuracy']:.4f} f1={rows[-1]['macro_f1']:.4f}")
    res = pd.DataFrame(rows)
    res.to_csv(RESULTS / "transfer.csv", index=False, float_format="%.4f")
    print("\n" + res.pivot(index="model", columns="dataset", values="accuracy")[list(datasets)].to_string())

    # Figure 8: grouped bars, dataset groups x model bars, dummy as a dashed line per group
    setup_plot_style()
    colors = sns.color_palette("colorblind", 4)
    model_names = list(HYP_MODELS) + [FULL_NAME]
    ds_names = list(datasets)
    x = np.arange(len(ds_names))
    w = 0.2
    fig, ax = plt.subplots(figsize=(11, 5.5))
    for i, m in enumerate(model_names):
        vals = [res[(res.model == m) & (res.dataset == d)]["accuracy"].iloc[0] for d in ds_names]
        bars = ax.bar(x + (i - 1.5) * w, vals, w, label=m, color=colors[i])
        for b, v in zip(bars, vals):
            ax.annotate(f"{v:.3f}", (b.get_x() + b.get_width() / 2, v), ha="center", va="bottom", fontsize=9)
    tick_labels = []
    for j, d in enumerate(ds_names):
        dacc = res[(res.model == DUMMY_NAME) & (res.dataset == d)]["accuracy"].iloc[0]
        ax.hlines(dacc, x[j] - 2.2 * w, x[j] + 2.2 * w, colors="black", linestyles="--", lw=1.4,
                  label="Dummy baseline (per dataset)" if j == 0 else None)
        tick_labels.append(f"{d}\n(n = {len(datasets[d]):,}; dummy = {dacc:.3f})")
    ax.set_xticks(x)
    ax.set_xticklabels(tick_labels)
    ax.set_ylim(0, 0.85)
    ax.set_xlabel("Evaluation set (models trained on SNLI train only)")
    ax.set_ylabel("Accuracy")
    ax.set_title("Transfer of SNLI-trained models to MultiNLI")
    ax.legend(loc="upper center", ncol=3, fontsize=9.5)
    fig.tight_layout()
    fig.savefig(FIGURES / "fig8_transfer.png", dpi=300)
    plt.close(fig)
    print("Saved figures/fig8_transfer.png")


if __name__ == "__main__":
    main()

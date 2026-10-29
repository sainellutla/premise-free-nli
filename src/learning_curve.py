"""Step A1: learning curve for hypothesis-only LogisticRegression.

Stratified training subsets of 1K, 5K, 10K, 50K, 100K and the full training set.
TF-IDF is refit on each subset (same settings). Subsets <=10K use seeds 42, 43, 44.
The full-training-set point reuses the saved model + vectorizer from evaluate.py
(identical settings and data, so refitting would reproduce it).

Run from the repo root:  python src/learning_curve.py
"""
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

from data import load_snli, setup_plot_style
from evaluate import timed_fit
from features import make_tfidf
from models import SEED, stratified_subsample_idx

ROOT = Path(__file__).resolve().parent.parent
RESULTS, FIGURES, MODELS = ROOT / "results", ROOT / "figures", ROOT / "models"
SIZES = [1_000, 5_000, 10_000, 50_000, 100_000]
MULTI_SEED_MAX = 10_000
SEEDS = [42, 43, 44]


def evaluate(y_true, y_pred):
    return accuracy_score(y_true, y_pred), f1_score(y_true, y_pred, average="macro")


def main():
    snli = load_snli()
    train, test = snli["train"], snli["test"]
    y_tr, y_te = train["label"].to_numpy(), test["label"].to_numpy()

    runs = []
    for n in SIZES:
        for seed in (SEEDS if n <= MULTI_SEED_MAX else [SEED]):
            idx = stratified_subsample_idx(y_tr, n=n, seed=seed)
            vec = make_tfidf()
            X = vec.fit_transform(train["hypothesis"].iloc[idx])
            model, secs, warns = timed_fit(LogisticRegression(max_iter=1000, random_state=SEED), X, y_tr[idx])
            acc, f1 = evaluate(y_te, model.predict(vec.transform(test["hypothesis"])))
            runs.append({"train_size": n, "seed": seed, "n_features": X.shape[1],
                         "accuracy": acc, "macro_f1": f1, "train_time_sec": secs,
                         "n_iter": int(model.n_iter_[0]), "fit_warnings": " | ".join(warns)})
            print(f"n={n:>6} seed={seed}: features={X.shape[1]} acc={acc:.4f} f1={f1:.4f} "
                  f"iter={model.n_iter_[0]} {secs:.1f}s" + (f" WARN {warns}" if warns else ""), flush=True)

    # Full training set: reuse the saved model and vectorizer.
    vec = joblib.load(MODELS / "hyp_tfidf.joblib")
    model = joblib.load(MODELS / "logreg_hyp.joblib")
    acc, f1 = evaluate(y_te, model.predict(vec.transform(test["hypothesis"])))
    runs.append({"train_size": len(y_tr), "seed": SEED, "n_features": len(vec.vocabulary_),
                 "accuracy": acc, "macro_f1": f1, "train_time_sec": np.nan,
                 "n_iter": int(model.n_iter_[0]), "fit_warnings": "reused saved model (models/logreg_hyp.joblib)"})
    print(f"n={len(y_tr)} (full, saved model): acc={acc:.4f} f1={f1:.4f}")

    runs_df = pd.DataFrame(runs)
    summary = (runs_df.groupby("train_size")
               .agg(n_seeds=("seed", "count"), seeds=("seed", lambda s: " ".join(map(str, s))),
                    n_features=("n_features", "mean"),
                    accuracy_mean=("accuracy", "mean"), accuracy_std=("accuracy", lambda s: s.std(ddof=0)),
                    macro_f1_mean=("macro_f1", "mean"), macro_f1_std=("macro_f1", lambda s: s.std(ddof=0)))
               .reset_index())
    # std is only meaningful where there are multiple seeds
    single = summary["n_seeds"] == 1
    summary.loc[single, ["accuracy_std", "macro_f1_std"]] = np.nan
    summary.to_csv(RESULTS / "learning_curve.csv", index=False, float_format="%.4f")
    runs_df.to_csv(RESULTS / "learning_curve_runs.csv", index=False, float_format="%.4f")
    print("\n" + summary.to_string(index=False))

    # Figure 6
    dummy_acc = accuracy_score(y_te, joblib.load(MODELS / "dummy_hyp.joblib").predict(
        vec.transform(test["hypothesis"])))
    setup_plot_style()
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.errorbar(summary["train_size"], summary["accuracy_mean"],
                yerr=summary["accuracy_std"].fillna(0), fmt="o-", capsize=5, lw=2, ms=7,
                label="Hypothesis-only LogReg (mean ± std over 3 seeds for ≤10K)")
    for x, y in zip(summary["train_size"], summary["accuracy_mean"]):
        ax.annotate(f"{y:.3f}", (x, y), textcoords="offset points", xytext=(0, 10), ha="center", fontsize=10)
    ax.axhline(dummy_acc, ls="--", color="black", lw=1.2, label=f"Dummy baseline ({dummy_acc:.3f})")
    ax.set_xscale("log")
    ax.set_xticks(summary["train_size"])
    ax.set_xticklabels(["1K", "5K", "10K", "50K", "100K", f"{len(y_tr)/1000:.0f}K\n(full)"])
    ax.set_ylim(0.3, 0.7)
    ax.set_xlabel("Training set size (log scale)")
    ax.set_ylabel("Test accuracy (SNLI test)")
    ax.set_title("Learning curve: hypothesis-only Logistic Regression")
    ax.legend(loc="center right", fontsize=10)
    fig.tight_layout()
    fig.savefig(FIGURES / "fig6_learning_curve.png", dpi=300)
    plt.close(fig)
    print("Saved figures/fig6_learning_curve.png")


if __name__ == "__main__":
    main()

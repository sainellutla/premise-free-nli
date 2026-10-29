"""Step A2: n-gram feature ablation for hypothesis-only LogisticRegression.

Three TF-IDF settings (max_features=5000, lowercase): unigrams (1,1), bigrams (2,2), both (1,2).
Each vectorizer is fit on the full SNLI train split. The (1,2) setting is identical to
evaluate.py, so it reuses the saved model + vectorizer.

Run from the repo root:  python src/feature_ablation.py
"""
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

from data import load_snli, setup_plot_style
from evaluate import timed_fit
from models import SEED

ROOT = Path(__file__).resolve().parent.parent
RESULTS, FIGURES, MODELS = ROOT / "results", ROOT / "figures", ROOT / "models"
SETTINGS = {"Unigrams (1,1)": (1, 1), "Bigrams (2,2)": (2, 2), "Uni + bigrams (1,2)": (1, 2)}


def main():
    snli = load_snli()
    train, test = snli["train"], snli["test"]
    y_tr, y_te = train["label"].to_numpy(), test["label"].to_numpy()

    rows = []
    for name, ngr in SETTINGS.items():
        if ngr == (1, 2):
            vec = joblib.load(MODELS / "hyp_tfidf.joblib")
            model = joblib.load(MODELS / "logreg_hyp.joblib")
            secs, warns, note = np.nan, [], "reused saved model (models/logreg_hyp.joblib)"
            X_te = vec.transform(test["hypothesis"])
        else:
            vec = TfidfVectorizer(ngram_range=ngr, max_features=5000, lowercase=True)
            X_tr = vec.fit_transform(train["hypothesis"])
            X_te = vec.transform(test["hypothesis"])
            model, secs, warns = timed_fit(LogisticRegression(max_iter=1000, random_state=SEED), X_tr, y_tr)
            note = ""
        pred = model.predict(X_te)
        # Test hypotheses with no in-vocabulary features (all-zero row) get only the intercept.
        empty = int((X_te.getnnz(axis=1) == 0).sum())
        rows.append({"setting": name, "ngram_range": str(ngr), "n_features": len(vec.vocabulary_),
                     "accuracy": accuracy_score(y_te, pred),
                     "macro_f1": f1_score(y_te, pred, average="macro"),
                     "n_test_all_zero_rows": empty, "train_time_sec": secs,
                     "n_iter": int(model.n_iter_[0]),
                     "notes": " | ".join(warns + ([note] if note else []))})
        print(f"{name}: acc={rows[-1]['accuracy']:.4f} f1={rows[-1]['macro_f1']:.4f} "
              f"all-zero test rows={empty} iter={model.n_iter_[0]}" + (f" WARN {warns}" if warns else ""), flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(RESULTS / "feature_ablation.csv", index=False, float_format="%.4f")
    print("\n" + df.to_string(index=False))

    # Figure 7: grouped bars (metric groups x settings)
    setup_plot_style()
    colors = sns.color_palette("colorblind", 3)
    metrics = [("accuracy", "Accuracy"), ("macro_f1", "Macro F1")]
    x = np.arange(len(metrics))
    w = 0.26
    fig, ax = plt.subplots(figsize=(8, 5))
    for i, r in enumerate(df.itertuples()):
        vals = [getattr(r, m) for m, _ in metrics]
        bars = ax.bar(x + (i - 1) * w, vals, w, label=r.setting, color=colors[i])
        for b, v in zip(bars, vals):
            ax.annotate(f"{v:.3f}", (b.get_x() + b.get_width() / 2, v), ha="center", va="bottom", fontsize=10)
    ax.set_xticks(x)
    ax.set_xticklabels([lab for _, lab in metrics])
    ax.set_ylim(0, 0.85)
    ax.set_xlabel("Metric (SNLI test)")
    ax.set_ylabel("Score")
    ax.set_title("Feature ablation: hypothesis-only LogReg by n-gram range\n(TF-IDF, max_features=5000)")
    ax.legend(loc="upper center", ncol=3, fontsize=10)
    fig.tight_layout()
    fig.savefig(FIGURES / "fig7_feature_ablation.png", dpi=300)
    plt.close(fig)
    print("Saved figures/fig7_feature_ablation.png")


if __name__ == "__main__":
    main()

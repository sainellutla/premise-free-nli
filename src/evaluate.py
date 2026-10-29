"""Step 5: train all models, evaluate on SNLI test, run 5-fold CV, save models and figures.

Run from the repo root:  python src/evaluate.py
"""
import time
import warnings
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_recall_fscore_support)
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from data import LABEL_NAMES, LABEL_ORDER, load_snli, setup_plot_style
from features import build_full_features, build_hyp_features, make_tfidf
from models import (SEED, SUBSAMPLE_SIZE, full_input_model, hyp_only_models,
                    stratified_subsample_idx)

ROOT = Path(__file__).resolve().parent.parent
RESULTS, FIGURES, MODELS = ROOT / "results", ROOT / "figures", ROOT / "models"
LABELS = [0, 1, 2]


def timed_fit(model, X, y):
    """Fit and return (fitted_model, wall_clock_seconds, list_of_warning_messages)."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        t0 = time.perf_counter()
        model.fit(X, y)
        secs = time.perf_counter() - t0
    msgs = sorted({f"{w.category.__name__}: {str(w.message).splitlines()[0]}" for w in caught})
    return model, secs, msgs


def score_row(name, features, y_true, y_pred, n_train, secs, warn_msgs):
    row = {"model": name, "features": features, "n_train": n_train,
           "train_time_sec": round(secs, 2),
           "accuracy": accuracy_score(y_true, y_pred),
           "macro_f1": f1_score(y_true, y_pred, average="macro", labels=LABELS, zero_division=0)}
    p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, labels=LABELS, zero_division=0)
    for i, lab in enumerate(LABELS):
        n = LABEL_NAMES[lab]
        row[f"{n}_precision"], row[f"{n}_recall"], row[f"{n}_f1"] = p[i], r[i], f[i]
    row["fit_warnings"] = " | ".join(warn_msgs)
    return row


def main():
    for d in (RESULTS, FIGURES, MODELS):
        d.mkdir(exist_ok=True)
    np.random.seed(SEED)

    snli = load_snli(verbose=True)
    train, val, test = snli["train"], snli["validation"], snli["test"]
    y_tr, y_te = train["label"].to_numpy(), test["label"].to_numpy()
    sub_idx = stratified_subsample_idx(y_tr)
    print(f"Stratified subsample: {len(sub_idx)} examples, label counts "
          f"{np.bincount(y_tr[sub_idx]).tolist()}")

    rows, preds = [], {}

    # ---- hypothesis-only ----
    t0 = time.perf_counter()
    hyp_vec, Xh_tr, Xh_va, Xh_te = build_hyp_features(train, val, test)
    print(f"Hyp-only TF-IDF: {Xh_tr.shape[1]} features, built in {time.perf_counter() - t0:.1f}s")
    joblib.dump(hyp_vec, MODELS / "hyp_tfidf.joblib")

    for name, (model, use_sub) in hyp_only_models().items():
        X, y = (Xh_tr[sub_idx], y_tr[sub_idx]) if use_sub else (Xh_tr, y_tr)
        print(f"Training {name} on {X.shape[0]} examples ...", flush=True)
        model, secs, w = timed_fit(model, X, y)
        pred = model.predict(Xh_te)
        preds[name] = pred
        rows.append(score_row(name, "hypothesis-only", y_te, pred, X.shape[0], secs, w))
        print(f"  {secs:.1f}s  test acc={rows[-1]['accuracy']:.4f}" + (f"  WARNINGS: {w}" if w else ""))
        joblib.dump(model, MODELS / (name.split(" ")[0].lower() + "_hyp.joblib"))

    # ---- full input ----
    t0 = time.perf_counter()
    full_tf, Xf_tr, Xf_va, Xf_te = build_full_features(train, val, test)
    print(f"Full-input features: {Xf_tr.shape[1]} columns, built in {time.perf_counter() - t0:.1f}s")
    joblib.dump(full_tf, MODELS / "full_transformers.joblib")
    name, model = full_input_model()
    print(f"Training {name} on {Xf_tr.shape[0]} examples ...", flush=True)
    model, secs, w = timed_fit(model, Xf_tr, y_tr)
    pred = model.predict(Xf_te)
    preds[name] = pred
    rows.append(score_row(name, "premise+hypothesis", y_te, pred, Xf_tr.shape[0], secs, w))
    print(f"  {secs:.1f}s  test acc={rows[-1]['accuracy']:.4f}" + (f"  WARNINGS: {w}" if w else ""))
    joblib.dump(model, MODELS / "logreg_full.joblib")

    results = pd.DataFrame(rows)
    results.to_csv(RESULTS / "model_results.csv", index=False, float_format="%.4f")
    print("\n" + results.drop(columns=["fit_warnings"]).to_string(index=False))

    # ---- 5-fold CV on the 100k subsample (vectorizer refit inside each fold) ----
    texts_sub = train["hypothesis"].iloc[sub_idx].to_numpy()
    y_sub = y_tr[sub_idx]
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    cv_models = {
        "LogReg (hyp-only)": LogisticRegression(max_iter=1000, random_state=SEED),
        "MultinomialNB (hyp-only)": MultinomialNB(),
        "LinearSVC (hyp-only)": LinearSVC(random_state=SEED),
    }
    cv_rows = []
    for name, est in cv_models.items():
        pipe = Pipeline([("tfidf", make_tfidf()), ("clf", clone(est))])
        print(f"5-fold CV: {name} ...", flush=True)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            cv = cross_validate(pipe, texts_sub, y_sub, cv=skf, scoring="accuracy", n_jobs=1)
        w = sorted({f"{c.category.__name__}: {str(c.message).splitlines()[0]}" for c in caught})
        accs = cv["test_score"]
        cv_rows.append({"model": name, "n_examples": len(y_sub), "n_folds": 5,
                        **{f"fold{i+1}_acc": a for i, a in enumerate(accs)},
                        "mean_acc": accs.mean(), "std_acc": accs.std(),
                        "mean_fold_fit_time_sec": cv["fit_time"].mean(),
                        "fit_warnings": " | ".join(w)})
        print(f"  {accs.mean():.4f} ± {accs.std():.4f}" + (f"  WARNINGS: {w}" if w else ""))
    cv_df = pd.DataFrame(cv_rows)
    cv_df.to_csv(RESULTS / "cv_results.csv", index=False, float_format="%.4f")
    print("\n" + cv_df.drop(columns=["fit_warnings"]).to_string(index=False))

    # ---- figures ----
    setup_plot_style()
    colors = sns.color_palette("colorblind", 8)

    # Fig 4: test accuracy for all models
    fig, ax = plt.subplots(figsize=(10, 5.5))
    bar_colors = [colors[7] if r.model.startswith("Dummy") else
                  colors[1] if r.features == "premise+hypothesis" else colors[0]
                  for r in results.itertuples()]
    bars = ax.bar(results["model"], results["accuracy"], color=bar_colors)
    for b, acc in zip(bars, results["accuracy"]):
        ax.annotate(f"{acc:.3f}", (b.get_x() + b.get_width() / 2, b.get_height()),
                    ha="center", va="bottom", fontsize=11)
    dummy_acc = results.loc[results["model"].str.startswith("Dummy"), "accuracy"].iloc[0]
    ax.axhline(dummy_acc, ls="--", color="black", lw=1.2,
               label=f"Dummy baseline ({dummy_acc:.3f})")
    ax.set_ylim(0, 1)
    ax.set_ylabel("Test accuracy (SNLI test)")
    ax.set_xlabel("Model")
    ax.set_title("SNLI test accuracy: hypothesis-only vs. full-input models")
    from matplotlib.patches import Patch
    handles = [Patch(color=colors[0], label="Hypothesis-only"),
               Patch(color=colors[1], label="Premise + hypothesis"),
               ax.get_lines()[0]]
    ax.legend(handles=handles, loc="upper left", fontsize=10)
    plt.setp(ax.get_xticklabels(), rotation=25, ha="right")
    fig.tight_layout()
    fig.savefig(FIGURES / "fig4_model_accuracy.png", dpi=300)
    plt.close(fig)

    # Fig 5: normalized confusion matrix for hyp-only LogReg
    cm = confusion_matrix(y_te, preds["LogReg (hyp-only)"], labels=LABELS, normalize="true")
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    sns.heatmap(cm, annot=True, fmt=".2f", cmap="Blues", vmin=0, vmax=1,
                xticklabels=LABEL_ORDER, yticklabels=LABEL_ORDER, ax=ax,
                cbar_kws={"label": "Fraction of true label"}, annot_kws={"size": 13})
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title("Hypothesis-only LogReg: confusion matrix\n(SNLI test, normalized by true label)")
    fig.tight_layout()
    fig.savefig(FIGURES / "fig5_confusion_matrix.png", dpi=300)
    plt.close(fig)
    print("\nSaved figures/fig4_model_accuracy.png and figures/fig5_confusion_matrix.png")


if __name__ == "__main__":
    main()

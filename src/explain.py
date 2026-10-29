"""Step 7: explanations for the hypothesis-only models (saved models, nothing refit).

- LR coefficients: top 20 features toward each label           -> results/top_words_lr.csv
- MultinomialNB: top 20 per label by log-prob ratio (class vs rest) -> results/top_words_nb.csv
- SHAP (LinearExplainer, LR) on a fixed random 1,000-example test sample -> results/shap_lr.csv
  (RandomForest TreeSHAP dropped as computationally infeasible: see PROBLEMS.md P16)
- Agreement: Spearman between method rankings per class over the union of top-100 lists
                                                                -> results/agreement.csv
- Figures 10-13.

Ranking direction (author decision, PROBLEMS.md D16): all methods rank features by their
contribution TOWARD the class:
  LR coef          : coef[c, j]
  NB log-ratio     : log P(j | c) - log P(j | not c)
  LR SHAP          : sign(mean SHAP when present) * mean |SHAP|
                     (falls back to sign(coef) for features never present in the sample)

Run from the repo root:  python src/explain.py
"""
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.sparse as sp
import seaborn as sns
import shap
from scipy.stats import spearmanr

from data import LABEL_NAMES, LABEL_ORDER, load_snli, setup_plot_style

ROOT = Path(__file__).resolve().parent.parent
RESULTS, FIGURES, MODELS = ROOT / "results", ROOT / "figures", ROOT / "models"
SEED = 42
N_SAMPLE = 1000
TOP_N = 20
UNION_TOP = 100
CLASSES = [0, 1, 2]
METHODS = ["LR coef", "LR SHAP", "NB log-ratio"]


def sample_idx(n_test):
    """Fixed random sample of test indices (seed 42), shared with removal.py / errors.py."""
    return np.sort(np.random.default_rng(SEED).choice(n_test, N_SAMPLE, replace=False))


def lr_shap_explainer(lr, X_train):
    """Interventional linear SHAP with the exact train feature means as background.
    For a linear model this equals coef * (x - train_mean)."""
    mean = np.asarray(X_train.mean(axis=0)).ravel()
    return shap.LinearExplainer(lr, (mean, sp.identity(len(mean), format="csr")),
                                feature_perturbation="interventional"), mean


def nb_log_ratio(nb):
    """log P(feature | class) - log P(feature | all other classes), Laplace/Lidstone
    smoothing with the model's alpha, computed from the fitted feature_count_."""
    fc, a = nb.feature_count_, nb.alpha
    V = fc.shape[1]
    out = np.zeros_like(fc)
    for c in range(fc.shape[0]):
        rest = fc[np.arange(fc.shape[0]) != c].sum(axis=0)
        log_p_c = np.log((fc[c] + a) / (fc[c].sum() + a * V))
        log_p_rest = np.log((rest + a) / (rest.sum() + a * V))
        out[c] = log_p_c - log_p_rest
    return out


def lr_shap_scores(sv, X_sample, coef):
    """Per class: mean |SHAP|, mean SHAP when present, and the signed toward-class score."""
    present = (X_sample > 0).toarray()
    n_present = present.sum(axis=0)
    rows = {}
    for c in CLASSES:
        s = sv[:, :, c]
        mean_abs = np.abs(s).mean(axis=0)
        with np.errstate(invalid="ignore", divide="ignore"):
            mean_present = np.where(n_present > 0, (s * present).sum(axis=0) / n_present, np.nan)
        direction = np.where(n_present > 0, np.sign(mean_present), np.sign(coef[c]))
        rows[c] = (mean_abs, mean_present, direction * mean_abs)
    return rows, n_present


def top_table(scores, names, value_name):
    rows = []
    for c in CLASSES:
        order = np.argsort(-scores[c])[:TOP_N]
        for r, j in enumerate(order, start=1):
            rows.append({"label": LABEL_NAMES[c], "rank": r, "feature": names[j], value_name: scores[c][j]})
    return pd.DataFrame(rows)


def main():
    snli = load_snli()
    train, test = snli["train"], snli["test"]
    vec = joblib.load(MODELS / "hyp_tfidf.joblib")
    lr = joblib.load(MODELS / "logreg_hyp.joblib")
    nb = joblib.load(MODELS / "multinomialnb_hyp.joblib")
    names = vec.get_feature_names_out()

    # ---- LR coefficients and NB log-ratios ----
    coef = lr.coef_
    nb_lr = nb_log_ratio(nb)
    top_table(coef, names, "coefficient").to_csv(RESULTS / "top_words_lr.csv", index=False, float_format="%.4f")
    top_table(nb_lr, names, "log_prob_ratio").to_csv(RESULTS / "top_words_nb.csv", index=False, float_format="%.4f")

    # ---- SHAP for LR on a fixed 1,000-example test sample ----
    X_train = vec.transform(train["hypothesis"])
    idx = sample_idx(len(test))
    X_s = vec.transform(test["hypothesis"].iloc[idx])
    explainer, train_mean = lr_shap_explainer(lr, X_train)
    sv = explainer.shap_values(X_s)  # (n, features, classes)
    manual = coef[2] * (X_s[0].toarray().ravel() - train_mean)
    print(f"SHAP sanity check (class 2, row 0): max |shap - coef*(x-mean)| = {np.abs(sv[0, :, 2] - manual).max():.2e}")
    shap_rows, n_present = lr_shap_scores(sv, X_s, coef)
    shap_df = pd.DataFrame({"feature": names, "n_present_in_sample": n_present})
    for c in CLASSES:
        mean_abs, mean_present, signed = shap_rows[c]
        shap_df[f"mean_abs_shap_{LABEL_NAMES[c]}"] = mean_abs
        shap_df[f"mean_shap_when_present_{LABEL_NAMES[c]}"] = mean_present
        shap_df[f"toward_score_{LABEL_NAMES[c]}"] = signed
    shap_df.to_csv(RESULTS / "shap_lr.csv", index=False, float_format="%.6f")
    shap_signed = {c: shap_rows[c][2] for c in CLASSES}
    top_table(shap_signed, names, "toward_score").to_csv(RESULTS / "top_words_shap_lr.csv", index=False, float_format="%.6f")

    # ---- Agreement ----
    method_scores = {"LR coef": coef, "LR SHAP": shap_signed, "NB log-ratio": nb_lr}
    agree_rows = []
    for c in CLASSES:
        union = set()
        for m in METHODS:
            union |= set(np.argsort(-np.asarray(method_scores[m][c]))[:UNION_TOP].tolist())
        union = np.array(sorted(union))
        for i, a in enumerate(METHODS):
            for b in METHODS[i + 1:]:
                rho, p = spearmanr(np.asarray(method_scores[a][c])[union], np.asarray(method_scores[b][c])[union])
                agree_rows.append({"label": LABEL_NAMES[c], "method_a": a, "method_b": b,
                                   "n_union_features": len(union), "spearman_rho": rho, "p_value": p})
    agree = pd.DataFrame(agree_rows)
    mean_rows = (agree.groupby(["method_a", "method_b"], sort=False)["spearman_rho"].mean().reset_index()
                 .assign(label="MEAN over classes", n_union_features=np.nan, p_value=np.nan))
    agree = pd.concat([agree, mean_rows[agree.columns]], ignore_index=True)

    # ---- RF Gini importance as a 4th, class-agnostic method (PROBLEMS.md D23) ----
    # Gini is an unsigned magnitude, so it is compared with each method's class-averaged
    # magnitude: mean |coef|, mean of the per-class mean |SHAP|, mean |NB log-ratio|.
    gini = joblib.load(MODELS / "randomforest_hyp.joblib").feature_importances_
    class_avg = {"LR coef": np.abs(coef).mean(axis=0),
                 "LR SHAP": np.mean([shap_rows[c][0] for c in CLASSES], axis=0),
                 "NB log-ratio": np.abs(nb_lr).mean(axis=0),
                 "RF Gini": gini}
    g_union = set()
    for m, s in class_avg.items():
        g_union |= set(np.argsort(-s)[:UNION_TOP].tolist())
    g_union = np.array(sorted(g_union))
    gini_rows = []
    for m in METHODS:
        rho, p = spearmanr(gini[g_union], class_avg[m][g_union])
        gini_rows.append({"label": "class-averaged magnitude", "method_a": "RF Gini", "method_b": m,
                          "n_union_features": len(g_union), "spearman_rho": rho, "p_value": p})
    agree = pd.concat([agree, pd.DataFrame(gini_rows)], ignore_index=True)
    agree.to_csv(RESULTS / "agreement.csv", index=False, float_format="%.4g")

    # ---- Frequency check: method scores vs train document frequency (D24) ----
    doc_freq = X_train.getnnz(axis=0)
    freq_rows = []
    for c in CLASSES:
        union = set()
        for m in METHODS:
            union |= set(np.argsort(-np.asarray(method_scores[m][c]))[:UNION_TOP].tolist())
        union = np.array(sorted(union))
        for m in METHODS:
            rho, p = spearmanr(np.asarray(method_scores[m][c])[union], doc_freq[union])
            freq_rows.append({"label": LABEL_NAMES[c], "method": m, "n_union_features": len(union),
                              "spearman_rho_vs_train_doc_freq": rho, "p_value": p})
    freq_df = pd.DataFrame(freq_rows)
    freq_df.to_csv(RESULTS / "ranking_vs_frequency.csv", index=False, float_format="%.4g")

    # ---- Length check: per-example neutral SHAP vs hypothesis length, top-10 overall (D25) ----
    overall_score = np.max(np.vstack([shap_signed[c] for c in CLASSES]), axis=0)
    overall_class = np.argmax(np.vstack([shap_signed[c] for c in CLASSES]), axis=0)
    hyp_len = test["hypothesis"].iloc[idx].str.split().str.len().to_numpy()
    len_rows = []
    for r, j in enumerate(np.argsort(-overall_score)[:10], start=1):
        rho, p = spearmanr(sv[:, j, 1], hyp_len)
        len_rows.append({"rank": r, "feature": names[j], "strongest_toward_label": LABEL_NAMES[overall_class[j]],
                         "n_present_in_sample": int(n_present[j]),
                         "spearman_rho_neutral_shap_vs_hyp_length": rho, "p_value": p})
    len_df = pd.DataFrame(len_rows)
    len_df.to_csv(RESULTS / "shap_vs_length.csv", index=False, float_format="%.4g")

    # ---- console summary ----
    for fname, col in [("top_words_lr.csv", "feature"), ("top_words_shap_lr.csv", "feature"), ("top_words_nb.csv", "feature")]:
        t = pd.read_csv(RESULTS / fname)
        print(f"\n{fname}:\n" + t.pivot(index="rank", columns="label", values=col)[LABEL_ORDER].to_string())
    print("\n" + agree.to_string(index=False))
    print("\nRanking vs train doc frequency:\n" + freq_df.to_string(index=False))
    print("\nNeutral SHAP vs hypothesis length (top-10 overall features):\n" + len_df.to_string(index=False))

    # ---- Figures ----
    setup_plot_style()
    colors = sns.color_palette("colorblind", 3)

    # Fig 10: top 15 toward-class features per label by LR SHAP
    fig, axes = plt.subplots(1, 3, figsize=(15, 6.5))
    for ax, c in zip(axes, CLASSES):
        order = np.argsort(-shap_signed[c])[:15][::-1]
        ax.barh([names[j] for j in order], shap_signed[c][order], color=colors[c])
        ax.set_title(LABEL_NAMES[c])
        ax.set_xlabel("Mean |SHAP| (features pushing toward label)")
        ax.tick_params(axis="y", labelsize=11)
    fig.suptitle(f"Top 15 hypothesis-only shortcut features per label (LR, LinearSHAP, {N_SAMPLE} SNLI test examples)")
    fig.tight_layout()
    fig.savefig(FIGURES / "fig10_top_shortcuts.png", dpi=300)
    plt.close(fig)

    # Fig 11: SHAP beeswarm for contradiction (LR). Seed the global RNG: the beeswarm jitters points.
    np.random.seed(SEED)
    plt.figure()
    shap.summary_plot(sv[:, :, 2], features=X_s.toarray(), feature_names=names, max_display=20,
                      show=False, plot_size=(9, 8))
    fig = plt.gcf()
    fig.axes[0].set_title(f"SHAP values for the contradiction class\n(hypothesis-only LR, {N_SAMPLE} SNLI test examples)")
    fig.axes[0].set_xlabel("SHAP value (impact on contradiction log-odds)")
    fig.tight_layout()
    fig.savefig(FIGURES / "fig11_shap_contradiction.png", dpi=300)
    plt.close(fig)

    # Fig 12: 3 strongest unigram shortcut words overall: SHAP toward their class, present vs absent
    is_unigram = np.array([" " not in n for n in names])
    best_score = np.max(np.vstack([shap_signed[c] for c in CLASSES]), axis=0)
    best_class = np.argmax(np.vstack([shap_signed[c] for c in CLASSES]), axis=0)
    cand = np.where(is_unigram)[0]
    top3 = cand[np.argsort(-best_score[cand])[:3]]
    np.random.seed(SEED)  # stripplot jitter
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    present = (X_s > 0).toarray()
    fig12_rows = []
    for ax, j in zip(axes, top3):
        c = best_class[j]
        vals = sv[:, j, c]
        d = pd.DataFrame({"status": np.where(present[:, j], "present", "absent"), "shap": vals})
        sns.stripplot(data=d, x="status", y="shap", order=["absent", "present"], ax=ax,
                      color=colors[c], alpha=0.5, size=4, jitter=0.25)
        n_p = int(present[:, j].sum())
        ax.set_title(f'"{names[j]}" → {LABEL_NAMES[c]}')
        ax.set_xticks([0, 1])
        ax.set_xticklabels([f"absent\n(n = {N_SAMPLE - n_p})", f"present\n(n = {n_p})"])
        ax.set_xlabel("Word in hypothesis")
        ax.set_ylabel(f"SHAP value toward {LABEL_NAMES[c]}")
        ax.axhline(0, color="black", lw=0.8)
        fig12_rows.append({"word": names[j], "label": LABEL_NAMES[c], "n_present": n_p,
                           "mean_shap_present": vals[present[:, j]].mean() if n_p else np.nan,
                           "mean_shap_absent": vals[~present[:, j]].mean()})
    fig.suptitle("Three strongest unigram shortcut words (LR SHAP): present vs. absent")
    fig.tight_layout()
    fig.savefig(FIGURES / "fig12_key_words.png", dpi=300)
    plt.close(fig)
    print("\nFig 12 words:\n" + pd.DataFrame(fig12_rows).to_string(index=False))

    # Fig 13: agreement heatmap. Pairs among the 3 class-specific methods = mean rho over classes;
    # pairs with RF Gini = class-averaged magnitude comparison.
    all_m = METHODS + ["RF Gini"]
    mat = pd.DataFrame(np.eye(len(all_m)), index=all_m, columns=all_m)
    for r in mean_rows.itertuples():
        mat.loc[r.method_a, r.method_b] = mat.loc[r.method_b, r.method_a] = r.spearman_rho
    for r in gini_rows:
        mat.loc[r["method_a"], r["method_b"]] = mat.loc[r["method_b"], r["method_a"]] = r["spearman_rho"]
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    sns.heatmap(mat, annot=True, fmt=".2f", cmap="viridis", vmin=-1, vmax=1, square=True, ax=ax,
                cbar_kws={"label": "Spearman ρ"}, annot_kws={"size": 13})
    ax.set_title("Agreement between explanation methods\n"
                 "(LR/NB pairs: toward-class, mean over 3 classes, union of top-100;\n"
                 "RF Gini pairs: class-averaged magnitude, union of top-100)", fontsize=11)
    ax.tick_params(axis="x", rotation=20)
    ax.tick_params(axis="y", rotation=0)
    fig.tight_layout()
    fig.savefig(FIGURES / "fig13_agreement.png", dpi=300)
    plt.close(fig)
    print("Saved figures 10-13")


if __name__ == "__main__":
    main()

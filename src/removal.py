"""Step 8: shortcut-removal experiment for hypothesis-only LogisticRegression (no retraining).

Two rankings of unigram shortcut words:
  (a) "LR SHAP": max over classes of the LR SHAP toward-class score (results/shap_lr.csv, Step 7)
  (b) "LR coef": max |coef| over classes, restricted to words in >= 0.1% of train hypotheses
For k = 5, 10, 20, 50, 100 the top-k words are deleted from the SNLI test hypotheses, which are
re-vectorized with the SAME fitted vectorizer and scored with the SAME saved model.

Random controls (per ranking; seeds 42-46; both exclude the ranking's top-100 words):
  - "nearest DF rank" (main, Fig 14; post-hoc, PROBLEMS.md P23): each shortcut word is replaced by
    the closest word in train-document-frequency rank within the ranking's candidate pool.
  - "bin-matched (20 DF bins)" (original, Fig 14b; poorly matched, P20): a random word from the same
    of 20 equal-count DF bins; if a bin runs out, the nearest bin with words left (n_fallback_draws).

Run from the repo root:  python src/removal.py
"""
import re
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import accuracy_score

from data import LABEL_NAMES, load_snli, setup_plot_style

ROOT = Path(__file__).resolve().parent.parent
RESULTS, FIGURES, MODELS = ROOT / "results", ROOT / "figures", ROOT / "models"
KS = [5, 10, 20, 50, 100]
SEEDS = [42, 43, 44, 45, 46]
N_BINS = 20
MIN_DF_FRAC = 0.001
EXCLUDE_TOP = 100


def remove_words(texts, words):
    """Delete whole-word, case-insensitive occurrences of `words` from each text."""
    if not words:
        return texts.copy(), 0
    pat = re.compile(r"(?u)\b(?:" + "|".join(re.escape(w) for w in sorted(words, key=len, reverse=True)) + r")\b",
                     re.IGNORECASE)
    new = texts.str.replace(pat, " ", regex=True)
    return new, int((new != texts).sum())


def matched_random(shortcut, pool, df, exclude, rng):
    """Draw one frequency-matched random word per shortcut word (same DF bin, without replacement)."""
    pool = sorted(pool, key=lambda w: (df[w], w))
    bin_of = {w: (i * N_BINS) // len(pool) for i, w in enumerate(pool)}
    available = {b: [w for w in pool if bin_of[w] == b and w not in exclude] for b in range(N_BINS)}
    chosen, fallbacks = [], 0
    for w in shortcut:
        b0 = bin_of[w]
        for dist in range(N_BINS):
            cands = [b for b in (b0 - dist, b0 + dist) if 0 <= b < N_BINS and available[b]]
            if cands:
                b = cands[0]
                break
        fallbacks += int(b != b0)
        pick = available[b].pop(rng.integers(len(available[b])))
        chosen.append(pick)
    return chosen, fallbacks


def nearest_rank_random(shortcut, pool, df, exclude, rng):
    """Post-hoc control (PROBLEMS.md P20/P23): for each shortcut word, take the available pool word
    closest to it in train-document-frequency rank (not in the ranking's top 100, not already taken).
    The seed sets the order in which shortcut words claim their match and breaks ties between
    equally close neighbours. Returns (words, mean absolute rank distance)."""
    pool_sorted = sorted(pool, key=lambda w: (df[w], w))
    rank = {w: i for i, w in enumerate(pool_sorted)}
    available = {w for w in pool_sorted if w not in exclude}
    chosen, dists = [None] * len(shortcut), []
    for i in rng.permutation(len(shortcut)):
        r0 = rank[shortcut[i]]
        for d in range(1, len(pool_sorted)):
            cands = [pool_sorted[r] for r in (r0 - d, r0 + d) if 0 <= r < len(pool_sorted) and pool_sorted[r] in available]
            if cands:
                pick = cands[rng.integers(len(cands))]
                break
        available.remove(pick)
        chosen[i] = pick
        dists.append(d)
    return chosen, float(np.mean(dists))


CTRL_BIN = "bin-matched (20 DF bins)"
CTRL_NN = "nearest DF rank"


def main():
    snli = load_snli()
    train, test = snli["train"], snli["test"]
    y_te = test["label"].to_numpy()
    vec = joblib.load(MODELS / "hyp_tfidf.joblib")
    lr = joblib.load(MODELS / "logreg_hyp.joblib")
    names = vec.get_feature_names_out()
    unigram = np.array([" " not in n for n in names])

    X_train = vec.transform(train["hypothesis"])
    df_counts = X_train.getnnz(axis=0)
    df = dict(zip(names, df_counts))
    n_train = X_train.shape[0]

    # Ranking (a): LR SHAP toward-class score from Step 7
    shap_df = pd.read_csv(RESULTS / "shap_lr.csv").set_index("feature").loc[names]
    shap_overall = shap_df[[f"toward_score_{LABEL_NAMES[c]}" for c in range(3)]].max(axis=1).to_numpy()
    pool_a = [n for n, u in zip(names, unigram) if u]
    rank_a = [names[j] for j in np.argsort(-shap_overall) if unigram[j]]

    # Ranking (b): max |coef|, unigrams in >= 0.1% of train hypotheses
    coef_max = np.abs(lr.coef_).max(axis=0)
    keep_b = unigram & (df_counts / n_train >= MIN_DF_FRAC)
    pool_b = [n for n, k in zip(names, keep_b) if k]
    rank_b = [names[j] for j in np.argsort(-coef_max) if keep_b[j]]
    print(f"Pool sizes: (a) {len(pool_a)} unigrams; (b) {len(pool_b)} unigrams with DF >= {MIN_DF_FRAC:.1%} "
          f"(>= {int(np.ceil(MIN_DF_FRAC * n_train))} of {n_train} train hypotheses)")

    hyps = test["hypothesis"]
    base_acc = accuracy_score(y_te, lr.predict(vec.transform(hyps)))
    print(f"No removal: accuracy {base_acc:.4f}")

    controls = [(CTRL_BIN, matched_random), (CTRL_NN, nearest_rank_random)]
    rows, runs = [], []
    for rname, ranking, pool in [("LR SHAP", rank_a, pool_a), ("LR coef (DF>=0.1%)", rank_b, pool_b)]:
        exclude = set(ranking[:EXCLUDE_TOP])
        print(f"\n{rname} top 20: {ranking[:20]}")
        rows.append({"ranking": rname, "condition": "none", "control_method": "", "k": 0,
                     "accuracy_mean": base_acc, "accuracy_std": np.nan, "n_seeds": 1, "n_test_hyp_changed": 0,
                     "mean_train_doc_freq": np.nan, "n_fallback_draws": 0, "mean_rank_distance": np.nan,
                     "n_distinct_word_sets": np.nan, "words": ""})
        for k in KS:
            words = ranking[:k]
            new, changed = remove_words(hyps, words)
            acc = accuracy_score(y_te, lr.predict(vec.transform(new)))
            rows.append({"ranking": rname, "condition": "shortcut", "control_method": "", "k": k,
                         "accuracy_mean": acc, "accuracy_std": np.nan, "n_seeds": 1, "n_test_hyp_changed": changed,
                         "mean_train_doc_freq": np.mean([df[w] for w in words]), "n_fallback_draws": 0,
                         "mean_rank_distance": np.nan, "n_distinct_word_sets": np.nan, "words": " ".join(words)})
            print(f"  k={k:>3}: shortcut acc={acc:.4f} (changed {changed}, mean DF {np.mean([df[w] for w in words]):.0f})")
            for cname, draw in controls:
                accs, changes, dfs, fbs, dists, sets = [], [], [], [], [], set()
                for seed in SEEDS:
                    rw, info = draw(words, pool, df, exclude, np.random.default_rng(seed))
                    new_r, ch_r = remove_words(hyps, rw)
                    a = accuracy_score(y_te, lr.predict(vec.transform(new_r)))
                    accs.append(a); changes.append(ch_r); dfs.append(np.mean([df[w] for w in rw]))
                    sets.add(frozenset(rw))
                    fb = info if cname == CTRL_BIN else 0
                    dist = info if cname == CTRL_NN else np.nan
                    fbs.append(fb); dists.append(dist)
                    runs.append({"ranking": rname, "control_method": cname, "k": k, "seed": seed, "accuracy": a,
                                 "n_test_hyp_changed": ch_r, "mean_train_doc_freq": dfs[-1],
                                 "n_fallback_draws": fb, "mean_rank_distance": dist, "words": " ".join(rw)})
                rows.append({"ranking": rname, "condition": "random", "control_method": cname, "k": k,
                             "accuracy_mean": np.mean(accs), "accuracy_std": np.std(accs, ddof=0),
                             "n_seeds": len(SEEDS), "n_test_hyp_changed": np.mean(changes),
                             "mean_train_doc_freq": np.mean(dfs), "n_fallback_draws": int(np.sum(fbs)),
                             "mean_rank_distance": np.mean(dists) if cname == CTRL_NN else np.nan,
                             "n_distinct_word_sets": len(sets), "words": ""})
                print(f"         {cname:<28} acc={np.mean(accs):.4f} ± {np.std(accs):.4f} "
                      f"(changed {np.mean(changes):.0f}, mean DF {np.mean(dfs):.0f}, distinct sets {len(sets)})")

    res = pd.DataFrame(rows)
    res.to_csv(RESULTS / "removal.csv", index=False, float_format="%.4f")
    pd.DataFrame(runs).to_csv(RESULTS / "removal_random_runs.csv", index=False, float_format="%.4f")
    print("\n" + res.drop(columns=["words"]).to_string(index=False))

    dummy_acc = accuracy_score(y_te, joblib.load(MODELS / "dummy_hyp.joblib").predict(vec.transform(hyps)))
    plot_removal(res, CTRL_NN, base_acc, dummy_acc, FIGURES / "fig14_removal.png",
                 "Removing shortcut words vs. frequency-matched random words\n"
                 "(control: nearest train-frequency rank; no retraining)")
    plot_removal(res, CTRL_BIN, base_acc, dummy_acc, FIGURES / "fig14b_removal_bin_control.png",
                 "Supplementary: original bin-matched random control (20 DF bins)\n"
                 "(poorly frequency-matched; see PROBLEMS.md P20)")
    print("Saved figures/fig14_removal.png and figures/fig14b_removal_bin_control.png")


def plot_removal(res, control_method, base_acc, dummy_acc, path, title):
    setup_plot_style()
    pal = sns.color_palette("colorblind", 4)
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for rname, color in [("LR SHAP", pal[0]), ("LR coef (DF>=0.1%)", pal[1])]:
        s = res[(res.ranking == rname) & (res.condition == "shortcut")]
        r = res[(res.ranking == rname) & (res.condition == "random") & (res.control_method == control_method)]
        ax.plot(s.k, s.accuracy_mean, "o-", color=color, lw=2.2, ms=7, label=f"{rname}: shortcut words removed")
        ax.errorbar(r.k, r.accuracy_mean, yerr=r.accuracy_std, fmt="s--", color=color, lw=1.6, ms=6,
                    capsize=4, alpha=0.8, label=f"{rname}: random control (mean ± std, 5 seeds)")
    ax.axhline(base_acc, ls=":", color="grey", lw=1.4, label=f"No removal ({base_acc:.3f})")
    ax.axhline(dummy_acc, ls="--", color="black", lw=1.2, label=f"Dummy baseline ({dummy_acc:.3f})")
    ax.set_xscale("log")
    ax.set_xticks(KS)
    ax.set_xticklabels([str(k) for k in KS])
    ax.set_ylim(0.3, 0.7)
    ax.set_xlabel("Number of words removed from test hypotheses (k, log scale)")
    ax.set_ylabel("Hypothesis-only LR accuracy (SNLI test)")
    ax.set_title(title, fontsize=12)
    ax.legend(loc="lower right", bbox_to_anchor=(0.995, 0.13), fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    main()

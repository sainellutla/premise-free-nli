# What Can a Model Learn Without the Premise? Explaining Hypothesis-Only Shortcuts in Natural Language Inference

Independent research project. This repo trains classical text classifiers on SNLI using **only the hypothesis**, compares them with a premise + hypothesis upper bound, and reports how far label prediction gets without the premise.

## Research questions
1. How accurately can simple models predict NLI labels from the hypothesis alone?
2. Which words drive those predictions, and do models and explanation methods agree?
3. How much do those shortcuts matter?

## Reproducing

Requirements: Python 3.12 (tested on Windows 11, CPU only, 16 GB RAM).

```powershell
python -m venv venv
.\venv\Scripts\pip install -r requirements.txt   # or requirements-lock.txt for the exact versions used
```

Run from the repo root, in this order:

```powershell
.\venv\Scripts\python.exe src\data.py             # Step 2: download SNLI + MultiNLI, dataset stats, Figs 2-3
.\venv\Scripts\python.exe src\evaluate.py         # Steps 3-5: features, train all models, test metrics, 5-fold CV, Figs 4-5
.\venv\Scripts\python.exe src\rerun_rf_timing.py  # Re-times the RandomForest fit with sleep blocked (see PROBLEMS.md P4/P9)
.\venv\Scripts\python.exe src\learning_curve.py   # Step A1: learning curve (1K-full), Fig 6
.\venv\Scripts\python.exe src\feature_ablation.py # Step A2: unigram / bigram / both ablation, Fig 7
.\venv\Scripts\python.exe src\transfer.py         # Step B1: SNLI-trained models on MultiNLI matched/mismatched, Fig 8
.\venv\Scripts\python.exe src\easy_hard.py        # Step B2: easy/hard split of SNLI test, Fig 9
.\venv\Scripts\python.exe src\explain.py          # Step 7: LR coef / NB / LR SHAP rankings, agreement (+ RF Gini),
                                                  #         frequency and length checks, Figs 10-13
.\venv\Scripts\python.exe src\removal.py          # Step 8: shortcut removal, both random controls, Figs 14 and 14b
.\venv\Scripts\python.exe src\errors.py           # Step 9: 15 error-analysis examples
```

`src/features.py` and `src/models.py` are imported by the other scripts; you don't run them directly. Every step after `evaluate.py` reuses the models and vectorizers it saves in `models/`, so run `evaluate.py` first. `removal.py` reads `results/shap_lr.csv`, so run `explain.py` before it.

How the reruns are reproduced:
- **RandomForest timing:** `rerun_rf_timing.py` refits only the RandomForest with sleep blocked and replaces its `train_time_sec` (PROBLEMS.md P4, P9).
- **Step 7 checks:** the RF Gini comparison, the frequency check and the length check are built into `explain.py`. One run regenerates every Step 7 output.
- **Step 8 controls:** `removal.py` computes both random controls in one run:
  - the original 20-bin frequency match (Fig 14b)
  - the post-hoc nearest-frequency-rank match (Fig 14; PROBLEMS.md P20, P23)

Runtime notes:
- `evaluate.py` takes roughly 10–20 minutes on a 12-thread laptop CPU. The RandomForest (200 trees) is the slowest part.
- Keep the machine awake while it runs, because sleep inflates the recorded wall-clock times.
- `explain.py` loads the 1.4 GB RandomForest for its Gini importances and needs a few GB of free RAM. The other steps after `evaluate.py` take a few minutes each.
- Trained models are saved to `models/`, which is gitignored. The RandomForest file is about 1.4 GB.

Random seed 42 is used everywhere. All vectorizers and scalers are fit on the training split only. In cross-validation, the TF-IDF vectorizer is refit inside each fold.

## Folder layout

```
data/        HuggingFace dataset cache (data/hf_cache/, gitignored)
src/
  data.py             loading SNLI / MultiNLI, dataset statistics, Figs 2-3
  features.py         hypothesis-only TF-IDF and full-input (premise + hypothesis + overlap) features
  models.py           model definitions and the stratified 100k subsample
  evaluate.py         training, test evaluation, 5-fold CV, Figs 4-5, saves models/
  rerun_rf_timing.py  RandomForest timing rerun
  learning_curve.py   A1: hypothesis-only LogReg vs. training-set size
  feature_ablation.py A2: n-gram range ablation
  transfer.py         B1: MultiNLI transfer of SNLI-trained models
  easy_hard.py        B2: easy/hard split by hypothesis-only correctness
  explain.py          7: explanation rankings, SHAP, agreement, frequency/length checks
  removal.py          8: shortcut-word removal vs. frequency-matched random controls
  errors.py           9: error-analysis examples
results/
  dataset_stats.csv       examples and hypothesis length (mean/median) per split and label
  top_frequent_words.csv  top-20 hypothesis words per label (train, stopwords removed except negations)
  model_results.csv       SNLI test accuracy, macro F1, per-class P/R/F1, training time per model
  cv_results.csv          5-fold stratified CV accuracy (100k subsample)
  learning_curve.csv      accuracy / macro F1 per training size (mean ± std over 3 seeds for ≤10K)
  learning_curve_runs.csv per-seed learning-curve runs
  feature_ablation.csv    accuracy / macro F1 for (1,1), (2,2), (1,2) n-grams
  transfer.csv            accuracy / macro F1 on SNLI test, MNLI matched, MNLI mismatched
  easy_hard.csv           easy/hard counts, label breakdown, full-input accuracy / macro F1 per subset
  top_words_lr.csv        top-20 features per label by LR coefficient
  top_words_nb.csv        top-20 features per label by NB log-probability ratio (class vs rest)
  top_words_shap_lr.csv   top-20 features per label by LR SHAP (toward-class)
  shap_lr.csv             per-feature mean |SHAP|, mean SHAP when present, toward score, per class
  agreement.csv           Spearman ρ between methods per class, plus RF Gini (class-averaged)
  ranking_vs_frequency.csv  Spearman ρ of each method's ranking vs train document frequency
  shap_vs_length.csv      neutral-class SHAP vs hypothesis length, top-10 features
  removal.csv             accuracy after removing top-k shortcut words vs random controls (control_method column)
  removal_random_runs.csv per-seed random-control runs and the words removed
  error_examples.csv      15 test examples with both models' predictions and top-3 SHAP features
figures/     fig2_label_distribution … fig14_removal, fig14b_removal_bin_control (300 dpi)
models/      saved models and vectorizers (gitignored)
PROBLEMS.md  log of every problem hit and every design choice
```

## Key design choices
The full list is in [PROBLEMS.md](PROBLEMS.md).
- **Data:** SNLI examples with label −1 are dropped.
- **Hypothesis-only features:** TF-IDF on the hypothesis, ngram_range=(1,2), max_features=5000, lowercase.
- **Full-input features:** premise TF-IDF and hypothesis TF-IDF (same settings, separate vectorizers), concatenated with 3 overlap features. The overlap features are:
  - fraction of hypothesis words found in the premise
  - number of shared words
  - hypothesis length minus premise length

  The overlap features are standardized, with the scaler fit on train.
- **RandomForest:** trained on a **stratified 100,000-example subsample** of train (seed 42) to keep training feasible on CPU. The same subsample is used for 5-fold CV of LogReg, MultinomialNB and LinearSVC.
- **Top-frequent-words table:** excludes scikit-learn's English stopwords but **keeps negation words** (no, not, nobody, never, nothing, …), since they're known contradiction cues.
- **Training time:** wall-clock time of `model.fit` only.
- **Explanations:** all per-class rankings rank features by how much they push *toward* the class (D16).
  - **LR SHAP:** LinearSHAP, with the exact training-set feature means as background, on a fixed random sample of 1,000 test examples (seed 42).
  - **RandomForest SHAP:** dropped. Exact TreeSHAP wasn't computationally feasible on this hardware (P16). RF Gini importance is used as a class-agnostic fourth method instead (D23).
- **Shortcut removal:** words are deleted from test hypotheses without retraining. Two rankings are compared:
  - LR SHAP
  - max |LR coefficient|, among words in ≥0.1% of train hypotheses

  The main random control matches each removed word to the word nearest it in train-frequency rank. This was a **post-hoc** change, made after the original 20-bin control turned out to be poorly matched (P20, P23). The original control is kept in `removal.csv` and in Fig 14b.

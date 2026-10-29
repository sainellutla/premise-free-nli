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
```

`src/features.py` and `src/models.py` are imported by `evaluate.py`; you don't run them directly.

Runtime notes:
- `evaluate.py` takes roughly 10–20 minutes on a 12-thread laptop CPU. The RandomForest (200 trees) is the slowest part.
- Keep the machine awake while it runs, because sleep inflates the recorded wall-clock times.
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
results/
  dataset_stats.csv       examples and hypothesis length (mean/median) per split and label
  top_frequent_words.csv  top-20 hypothesis words per label (train, stopwords removed except negations)
  model_results.csv       SNLI test accuracy, macro F1, per-class P/R/F1, training time per model
  cv_results.csv          5-fold stratified CV accuracy (100k subsample)
figures/     fig2_label_distribution, fig3_hypothesis_length, fig4_model_accuracy, fig5_confusion_matrix (300 dpi)
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

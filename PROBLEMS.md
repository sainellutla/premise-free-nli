# Problems and Design Choices Log

Problems hit during the project, and design choices that affect results.

## Problems

| # | Step | Problem | Cause | Fix |
|---|------|---------|-------|-----|
| P1 | 2 | HuggingFace warned "unauthenticated requests to the HF Hub". | No HF_TOKEN set. | No fix needed: the public datasets downloaded fine. |
| P2 | 2 | HuggingFace warned that symlinks aren't supported on this Windows machine. The raw downloaded files went to `C:\Users\nellu\.cache\huggingface\hub`, not only `data/hf_cache/`. | Windows without Developer Mode can't create symlinks. `huggingface_hub` keeps raw downloads in its own user-level cache; `cache_dir` only controls the processed Arrow cache. | No fix applied: caching still works (it just uses more disk), and results are unaffected. Noted for reproducibility. |
| P3 | 2 | The token "s" shows up at rank 18 of the neutral top-20 words. | The `\b\w+\b` regex splits possessives like "man's" into "man" + "s". | Left as-is to keep tokenization simple and transparent (D2). Read it as the possessive marker. |
| P4 | 5 | The RandomForest `train_time_sec` came out as 7864.7 s (~2 h 11 min), against an estimate of ~5 min from the timing test (20 trees on 100k = 29.1 s). | The Windows System event log shows the laptop in Modern Standby from 15:19:28 to 17:20:25 (with only a brief wake at 15:56:50), during the RF fit. Wall-clock time includes the sleep. | Fixed by the rerun in P9. Every other model's timing falls outside the standby window and was kept. |
| P9 | 5 | Rerun to fix P4. | — | `src/rerun_rf_timing.py` blocked idle sleep for its own process (`SetThreadExecutionState(ES_CONTINUOUS \| ES_SYSTEM_REQUIRED)`) and refit **only** the RandomForest, with identical settings and the same subsample. New fit time: **277.25 s**. Test accuracy 0.586523 and macro F1 0.585761 are identical to the first run (0.5865 / 0.5858 at 4 dp). Test predictions differ from the first run's saved model on 0 of 9,824 examples. Only the RF `train_time_sec` in `model_results.csv` was replaced (7864.72 → 277.25); every other value is untouched. The originally saved model file was kept. |
| P10 | 6 | The first `git commit` failed with "Author identity unknown". | No git user.name/user.email configured on this machine. | Set a repo-local identity only (`git config user.name sainellutla`, `git config user.email sainellutla.cs@gmail.com`; not global), then committed. |
| P5 | 5 | The LinearSVC full-train time (92.6 s) was ~4x the linear extrapolation from 100k (~21 s). | Training cost grows faster than linearly with n for liblinear on this data. | No fix needed; noted so time estimates aren't trusted blindly. |
| P6 | 5 | The saved RandomForest model is 1.37 GB (`models/randomforest_hyp.joblib`). | 200 fully grown trees (no max_depth, per the plan). | `models/` is gitignored, so it isn't pushed. |
| P7 | 5 | PowerShell showed the HF "unauthenticated" warning as a `NativeCommandError`. | Windows PowerShell 5.1 wraps native stderr in an ErrorRecord when it's redirected with `2>&1`. | Cosmetic only: the script exited with code 0 and all outputs were written. |
| P8 | 5 | The console log shows "Â±" instead of "±". | Encoding mismatch between the Python console output and PowerShell `Out-File`. | Cosmetic only; the CSVs store mean and std in separate numeric columns. |

## Design choices

| # | Step | Choice | Reason |
|---|------|--------|--------|
| D1 | 2 | Top frequent words exclude scikit-learn `ENGLISH_STOP_WORDS`, **except** negation words (no, not, nobody, never, nothing, none, nor, cannot, neither, nowhere), which are kept. | Negation words are known hypothesis-only cues for contradiction in SNLI; removing them would hide the effect being studied. |
| D2 | 2 | Top frequent words are computed on the SNLI **train** split, with lowercase and regex `\b\w+\b` tokenization. Hypothesis length = whitespace word count. | Train is the data the models learn from; simple, transparent tokenization. |
| D3 | 2 | Examples with label −1 (no annotator consensus) are dropped from all SNLI splits. | Specified in the project plan; these have no gold label. |
| D4 | 4 | The RandomForest trains on a stratified 100,000-example subsample of train (`train_test_split(train_size=100000, stratify=y, random_state=42)`; label counts 33,387 / 33,268 / 33,345). The same subsample is used for 5-fold CV. | Specified in the plan, to keep RF training feasible on a 16 GB / CPU-only machine. |
| D5 | 3 | The 3 overlap features in the full-input model are standardized with `StandardScaler`, fit on train only. | Puts the raw length difference on the same scale as the TF-IDF values for LogReg. |
| D6 | 3 | The overlap features use TF-IDF's default token pattern (`\b\w\w+\b`, lowercase), so single-character tokens like "a" are excluded. Length difference is in these tokens, not in whitespace words as in Step 2. Overlap counts are over word types (sets). | Overlap features see exactly the same words as the TF-IDF features. |
| D7 | 4 | `multi_class` isn't passed to LogisticRegression: it's deprecated in scikit-learn 1.9, and the default lbfgs solver is already multinomial for 3 classes. | Avoids a deprecation warning while matching the spec. |
| D8 | 5 | CV uses `Pipeline(TfidfVectorizer, model)`, so TF-IDF is refit on each fold's training part. CV std is the population std (numpy default, ddof=0). | No leakage across folds. |
| D9 | 5 | `train_time_sec` is wall-clock time (`time.perf_counter`) around `model.fit` only. It excludes feature extraction and prediction. | The paper reports training cost per model. |

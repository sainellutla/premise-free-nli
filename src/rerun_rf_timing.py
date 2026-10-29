"""Re-time the RandomForest fit with Windows sleep blocked (see PROBLEMS.md P4/P9).

The first run's RF fit time included ~2 h of laptop standby. This script:
  1. blocks idle sleep for this process (Windows SetThreadExecutionState),
  2. rebuilds the hypothesis-only features and the 100k subsample exactly as evaluate.py does,
  3. refits the RandomForest with the same settings and times model.fit only,
  4. checks test accuracy / macro F1 and predictions against the first run,
  5. replaces ONLY the RF train_time_sec in results/model_results.csv.

Run from the repo root:  python src/rerun_rf_timing.py
"""
import ctypes
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from data import load_snli
from evaluate import score_row, timed_fit
from features import build_hyp_features
from models import SEED, hyp_only_models, stratified_subsample_idx

ROOT = Path(__file__).resolve().parent.parent
RF_NAME = "RandomForest (hyp-only, 100k)"
ES_CONTINUOUS, ES_SYSTEM_REQUIRED = 0x80000000, 0x00000001


def main():
    if sys.platform == "win32":
        ok = ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED)
        print(f"Sleep blocked for this process: {bool(ok)}")
    np.random.seed(SEED)

    snli = load_snli()
    train, val, test = snli["train"], snli["validation"], snli["test"]
    y_tr, y_te = train["label"].to_numpy(), test["label"].to_numpy()
    sub_idx = stratified_subsample_idx(y_tr)
    _, Xh_tr, _, Xh_te = build_hyp_features(train, val, test)

    model, _ = hyp_only_models()[RF_NAME]
    print(f"Refitting {RF_NAME} on {len(sub_idx)} examples ...", flush=True)
    model, secs, warns = timed_fit(model, Xh_tr[sub_idx], y_tr[sub_idx])
    pred_new = model.predict(Xh_te)
    new = score_row(RF_NAME, "hypothesis-only", y_te, pred_new, len(sub_idx), secs, warns)
    print(f"New fit: {secs:.2f}s, accuracy={new['accuracy']:.6f}, macro_f1={new['macro_f1']:.6f}"
          + (f", warnings: {warns}" if warns else ""))
    del model

    # Compare against the first run: saved model predictions (exact) and stored CSV metrics.
    old_model = joblib.load(ROOT / "models" / "randomforest_hyp.joblib")
    pred_old = old_model.predict(Xh_te)
    del old_model
    n_diff = int((pred_old != pred_new).sum())
    print(f"Test predictions differing from first-run saved model: {n_diff} of {len(y_te)}")

    csv_path = ROOT / "results" / "model_results.csv"
    res = pd.read_csv(csv_path)
    row = res["model"] == RF_NAME
    old_acc, old_f1, old_time = (res.loc[row, c].iloc[0] for c in ("accuracy", "macro_f1", "train_time_sec"))
    acc_same = round(new["accuracy"], 4) == round(old_acc, 4)
    f1_same = round(new["macro_f1"], 4) == round(old_f1, 4)
    print(f"First run (CSV, 4 dp): accuracy={old_acc:.4f}, macro_f1={old_f1:.4f}, train_time_sec={old_time}")
    print(f"Accuracy identical: {acc_same}; macro F1 identical: {f1_same}")

    res.loc[row, "train_time_sec"] = round(secs, 2)
    res.to_csv(csv_path, index=False, float_format="%.4f")
    print(f"Replaced RF train_time_sec: {old_time} -> {round(secs, 2)}")


if __name__ == "__main__":
    main()

"""Step 9: error analysis on SNLI test (saved models only).

Three groups of 5 real test examples, each drawn at random (seed 42) from its eligible pool:
  A "confident_correct"  : hypothesis-only LR correct, confidence (max class prob) > 0.9
  B "confident_wrong"    : hypothesis-only LR wrong,   confidence > 0.9
  C "hyp_wrong_full_right": hypothesis-only LR wrong, full-input LR right
For each: premise, hypothesis, gold, both predictions and confidences, and the top 3 SHAP features
(LinearSHAP, train-mean background) pushing toward the hypothesis-only model's predicted class.

Run from the repo root:  python src/errors.py
"""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from data import LABEL_NAMES, load_snli
from explain import lr_shap_explainer
from features import transform_full

ROOT = Path(__file__).resolve().parent.parent
RESULTS, MODELS = ROOT / "results", ROOT / "models"
SEED = 42
N_PER_GROUP = 5
CONF = 0.9


def main():
    snli = load_snli()
    train, test = snli["train"], snli["test"]
    y = test["label"].to_numpy()
    vec = joblib.load(MODELS / "hyp_tfidf.joblib")
    lr = joblib.load(MODELS / "logreg_hyp.joblib")
    lr_full = joblib.load(MODELS / "logreg_full.joblib")
    names = vec.get_feature_names_out()

    X_te = vec.transform(test["hypothesis"])
    p_hyp = lr.predict_proba(X_te)
    pred_hyp, conf_hyp = p_hyp.argmax(axis=1), p_hyp.max(axis=1)
    p_full = lr_full.predict_proba(transform_full(joblib.load(MODELS / "full_transformers.joblib"), test))
    pred_full, conf_full = p_full.argmax(axis=1), p_full.max(axis=1)
    assert (pred_hyp == lr.predict(X_te)).all()

    pools = {
        "A_confident_correct": np.where((pred_hyp == y) & (conf_hyp > CONF))[0],
        "B_confident_wrong": np.where((pred_hyp != y) & (conf_hyp > CONF))[0],
        "C_hyp_wrong_full_right": np.where((pred_hyp != y) & (pred_full == y))[0],
    }
    rng = np.random.default_rng(SEED)
    chosen = {g: np.sort(rng.choice(p, size=min(N_PER_GROUP, len(p)), replace=False)) for g, p in pools.items()}
    for g, p in pools.items():
        print(f"{g}: pool size {len(p)}, selected {len(chosen[g])}")

    explainer, _ = lr_shap_explainer(lr, vec.transform(train["hypothesis"]))
    rows = []
    for g, idxs in chosen.items():
        sv = explainer.shap_values(X_te[idxs])  # (n, features, classes)
        for r, i in enumerate(idxs):
            c = pred_hyp[i]
            present = X_te[i].indices
            vals = sv[r, present, c]
            order = present[np.argsort(-vals)][:3]
            top3 = "; ".join(f"{names[j]} ({sv[r, j, c]:+.3f})" for j in order)
            rows.append({"group": g, "test_index": int(i), "premise": test["premise"].iloc[i],
                         "hypothesis": test["hypothesis"].iloc[i], "gold": LABEL_NAMES[y[i]],
                         "hyp_only_pred": LABEL_NAMES[c], "hyp_only_conf": conf_hyp[i],
                         "full_input_pred": LABEL_NAMES[pred_full[i]], "full_input_conf": conf_full[i],
                         "top3_shap_toward_hyp_pred": top3})
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS / "error_examples.csv", index=False, float_format="%.3f")
    pd.set_option("display.max_colwidth", 80)
    pd.set_option("display.width", 250)
    print(df.drop(columns=["premise"]).to_string(index=False))


if __name__ == "__main__":
    main()

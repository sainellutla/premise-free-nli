"""Step 3: feature extraction. Every vectorizer/scaler is fit on TRAIN only."""
import re

import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler

TFIDF_PARAMS = dict(ngram_range=(1, 2), max_features=5000, lowercase=True)
# Same token pattern TfidfVectorizer uses by default, so overlap features
# see the same words as the TF-IDF features.
TOKEN_RE = re.compile(r"(?u)\b\w\w+\b")
OVERLAP_NAMES = ["overlap_frac_hyp_in_prem", "overlap_shared_count", "len_diff_hyp_minus_prem"]


def make_tfidf():
    return TfidfVectorizer(**TFIDF_PARAMS)


def build_hyp_features(train, val, test):
    """Hypothesis-only TF-IDF. Returns (vectorizer, X_train, X_val, X_test)."""
    vec = make_tfidf()
    X_tr = vec.fit_transform(train["hypothesis"])
    return vec, X_tr, vec.transform(val["hypothesis"]), vec.transform(test["hypothesis"])


def overlap_features(premises, hypotheses):
    """3 columns: fraction of hypothesis word types found in the premise,
    number of shared word types, hypothesis length minus premise length (tokens)."""
    out = np.zeros((len(premises), 3), dtype=np.float64)
    for i, (p, h) in enumerate(zip(premises, hypotheses)):
        p_tok = TOKEN_RE.findall(p.lower())
        h_tok = TOKEN_RE.findall(h.lower())
        p_set, h_set = set(p_tok), set(h_tok)
        shared = len(h_set & p_set)
        out[i, 0] = shared / len(h_set) if h_set else 0.0
        out[i, 1] = shared
        out[i, 2] = len(h_tok) - len(p_tok)
    return out


def transform_full(transformers, df):
    """Apply already-fitted full-input transformers (from build_full_features) to new data."""
    P = transformers["premise_vec"].transform(df["premise"])
    H = transformers["hypothesis_vec"].transform(df["hypothesis"])
    O = transformers["overlap_scaler"].transform(overlap_features(df["premise"], df["hypothesis"]))
    return sparse.hstack([P, H, sparse.csr_matrix(O)], format="csr")


def build_full_features(train, val, test):
    """Premise TF-IDF + hypothesis TF-IDF + 3 standardized overlap features.
    Returns (transformers_dict, X_train, X_val, X_test)."""
    vec_p, vec_h = make_tfidf(), make_tfidf()
    scaler = StandardScaler()

    def transform(df, fit=False):
        if fit:
            P = vec_p.fit_transform(df["premise"])
            H = vec_h.fit_transform(df["hypothesis"])
            O = scaler.fit_transform(overlap_features(df["premise"], df["hypothesis"]))
        else:
            P = vec_p.transform(df["premise"])
            H = vec_h.transform(df["hypothesis"])
            O = scaler.transform(overlap_features(df["premise"], df["hypothesis"]))
        return sparse.hstack([P, H, sparse.csr_matrix(O)], format="csr")

    X_tr = transform(train, fit=True)
    X_va = transform(val)
    X_te = transform(test)
    return {"premise_vec": vec_p, "hypothesis_vec": vec_h, "overlap_scaler": scaler}, X_tr, X_va, X_te

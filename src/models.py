"""Step 4: model definitions."""
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC

SEED = 42
SUBSAMPLE_SIZE = 100_000


def hyp_only_models():
    """name -> (estimator, uses_100k_subsample). LogisticRegression uses the default
    lbfgs solver, which is multinomial for 3 classes (multi_class is deprecated)."""
    return {
        "Dummy (most frequent)": (DummyClassifier(strategy="most_frequent"), False),
        "LogReg (hyp-only)": (LogisticRegression(max_iter=1000, random_state=SEED), False),
        "MultinomialNB (hyp-only)": (MultinomialNB(), False),
        "LinearSVC (hyp-only)": (LinearSVC(random_state=SEED), False),
        "RandomForest (hyp-only, 100k)": (
            RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=SEED), True),
    }


def full_input_model():
    return "LogReg (full input)", LogisticRegression(max_iter=1000, random_state=SEED)


def stratified_subsample_idx(y, n=SUBSAMPLE_SIZE, seed=SEED):
    """Indices of a stratified n-example subsample of the training set."""
    idx, _ = train_test_split(range(len(y)), train_size=n, stratify=y, random_state=seed)
    return sorted(idx)

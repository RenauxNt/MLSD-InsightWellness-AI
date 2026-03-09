import json
import logging
from pathlib import Path

import pandas as pd
import yaml
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths & config
# ---------------------------------------------------------------------------
REPORTS_DIR = Path("reports")
REPORTS_DIR.mkdir(parents=True, exist_ok=True)
COMPARISON_PATH = REPORTS_DIR / "model_comparison.json"

params = yaml.safe_load(open("params.yaml", "r", encoding="utf-8"))
D, S, T = params["data"], params["split"], params["train"]
target = S["target"]


def load_xy(fp: str):
    """Load features and target from a CSV file."""
    df = pd.read_csv(fp)
    y = df[target]
    X = df.drop(columns=[target])
    return X, y


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
logger.info("Loading datasets …")
Xtr, ytr = load_xy(D["train"])
Xte, yte = load_xy(D["test"])
logger.info("Train: %d rows | Test: %d rows", Xtr.shape[0], Xte.shape[0])

# ---------------------------------------------------------------------------
# Models to compare
# ---------------------------------------------------------------------------
models = {
    "DecisionTree": DecisionTreeClassifier(
        max_depth=15,
        random_state=T["random_state"],
    ),
    "RandomForest": RandomForestClassifier(
        n_estimators=T["n_estimators"],
        max_depth=T["max_depth"],
        random_state=T["random_state"],
        n_jobs=-1,
    ),
    "GradientBoosting": GradientBoostingClassifier(
        n_estimators=100,
        max_depth=5,
        random_state=T["random_state"],
    ),
}

# ---------------------------------------------------------------------------
# Training & evaluation
# ---------------------------------------------------------------------------
results = {}

for name, model in models.items():
    logger.info("Training %s …", name)

    model.fit(Xtr, ytr)
    pte = model.predict(Xte)

    cv_scores = cross_val_score(model, Xtr, ytr, cv=5, scoring="f1_macro", n_jobs=-1)

    results[name] = {
        "test_accuracy": round(float(accuracy_score(yte, pte)), 4),
        "test_f1_macro": round(float(f1_score(yte, pte, average="macro")), 4),
        "test_precision_macro": round(
            float(precision_score(yte, pte, average="macro")), 4
        ),
        "test_recall_macro": round(float(recall_score(yte, pte, average="macro")), 4),
        "cv_f1_macro_mean": round(float(cv_scores.mean()), 4),
        "cv_f1_macro_std": round(float(cv_scores.std()), 4),
    }

    logger.info(
        "%s — Accuracy: %.4f | F1-macro: %.4f | CV F1: %.4f ± %.4f",
        name,
        results[name]["test_accuracy"],
        results[name]["test_f1_macro"],
        results[name]["cv_f1_macro_mean"],
        results[name]["cv_f1_macro_std"],
    )

# ---------------------------------------------------------------------------
# Save comparison
# ---------------------------------------------------------------------------
with open(COMPARISON_PATH, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=4)

logger.info("Comparison saved -> %s", COMPARISON_PATH)
logger.info("Done.")

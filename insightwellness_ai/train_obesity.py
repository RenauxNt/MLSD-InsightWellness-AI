import json
import logging
import time
from pathlib import Path

from codecarbon import EmissionsTracker
import altair as alt
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import cross_val_score
import yaml

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
METRICS_PATH = REPORTS_DIR / "metrics.json"

params = yaml.safe_load(open("params.yaml", "r", encoding="utf-8"))
D, S, T = params["data"], params["split"], params["train"]
target = S["target"]


def load_xy(fp: str):
    """Load features and target from a CSV file."""
    df = pd.read_csv(fp)
    y = df[target]
    X = df.drop(columns=[target])
    return X, y


def plot_confusion_matrix(cm, classes, out_path: Path):
    """Save a confusion-matrix heatmap to *out_path* using Altair."""
    rows = [
        {"True label": classes[i], "Predicted label": classes[j], "Count": int(cm[i][j])}
        for i in range(len(classes))
        for j in range(len(classes))
    ]

    base = alt.Chart(alt.Data(values=rows)).encode(
        x=alt.X("Predicted label:N", sort=classes, title="Predicted label"),
        y=alt.Y("True label:N", sort=classes, title="True label"),
    )

    heatmap = base.mark_rect().encode(
        color=alt.Color("Count:Q", scale=alt.Scale(scheme="blues"), title="Count"),
    )

    text = base.mark_text(fontSize=12).encode(
        text="Count:Q",
        color=alt.condition(
            alt.datum.Count > int(cm.max() / 2),
            alt.value("white"),
            alt.value("black"),
        ),
    )

    chart = (heatmap + text).properties(
        width=max(300, len(classes) * 60),
        height=max(300, len(classes) * 60),
        title="Confusion Matrix",
    )

    chart.save(str(out_path))
    logger.info("Confusion matrix saved -> %s", out_path)


# ---------------------------------------------------------------------------
# Carbon tracking
# ---------------------------------------------------------------------------
tracker = EmissionsTracker(
    output_dir=str(REPORTS_DIR),
    output_file="emissions.csv",
    save_to_file=True,
    gpu_ids=[],
    measure_power_secs=0.1,
)
tracker.start()

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
logger.info("Loading datasets …")
Xtr, ytr = load_xy(D["train"])
Xva, yva = load_xy(D["validation"])

Xte, yte = None, None
if D.get("test"):
    Xte, yte = load_xy(D["test"])
    logger.info("Test set loaded: %d rows", Xte.shape[0])

logger.info("Train: %d rows, %d features | Val: %d rows", Xtr.shape[0], Xtr.shape[1], Xva.shape[0])

# ---------------------------------------------------------------------------
# Model training
# ---------------------------------------------------------------------------
logger.info("Training RandomForestClassifier …")
t0 = time.perf_counter()

clf = RandomForestClassifier(
    n_estimators=T["n_estimators"],
    max_depth=T["max_depth"],
    random_state=T["random_state"],
    n_jobs=-1,
)
clf.fit(Xtr, ytr)

train_duration_s = time.perf_counter() - t0
logger.info("Training completed in %.2f s", train_duration_s)

# ---------------------------------------------------------------------------
# Cross-validation (stability check)
# ---------------------------------------------------------------------------
logger.info("Running 5-fold cross-validation on training set …")
cv_scores = cross_val_score(clf, Xtr, ytr, cv=5, scoring="f1_macro", n_jobs=-1)
logger.info("CV F1-macro: %.4f ± %.4f", cv_scores.mean(), cv_scores.std())

# ---------------------------------------------------------------------------
# Evaluation — validation set
# ---------------------------------------------------------------------------
pva = clf.predict(Xva)

metrics: dict = {
    "val_accuracy":        float(accuracy_score(yva, pva)),
    "val_f1_macro":        float(f1_score(yva, pva, average="macro")),
    "val_precision_macro": float(precision_score(yva, pva, average="macro")),
    "val_recall_macro":    float(recall_score(yva, pva, average="macro")),
    "cv_f1_macro_mean":    float(cv_scores.mean()),
    "cv_f1_macro_std":     float(cv_scores.std()),
    "train_duration_s":    round(train_duration_s, 2),
}

# ---------------------------------------------------------------------------
# Evaluation — test set (if available)
# ---------------------------------------------------------------------------
if Xte is not None:
    pte = clf.predict(Xte)
    metrics["test_accuracy"]        = float(accuracy_score(yte, pte))
    metrics["test_f1_macro"]        = float(f1_score(yte, pte, average="macro"))
    metrics["test_precision_macro"] = float(precision_score(yte, pte, average="macro"))
    metrics["test_recall_macro"]    = float(recall_score(yte, pte, average="macro"))
    logger.info("Test accuracy: %.4f | Test F1-macro: %.4f", metrics["test_accuracy"], metrics["test_f1_macro"])

# ---------------------------------------------------------------------------
# Classification report & confusion matrix
# ---------------------------------------------------------------------------
classes = clf.classes_.tolist()

# Validation
val_report = classification_report(yva, pva, output_dict=True)
val_report_path = REPORTS_DIR / "val_classification_report.json"
with open(val_report_path, "w", encoding="utf-8") as f:
    json.dump(val_report, f, indent=2)

cm = confusion_matrix(yva, pva, labels=classes)
plot_confusion_matrix(cm, classes, REPORTS_DIR / "val_confusion_matrix.html")

# Test (if available)
if Xte is not None:
    test_report = classification_report(yte, pte, output_dict=True)
    with open(REPORTS_DIR / "test_classification_report.json", "w", encoding="utf-8") as f:
        json.dump(test_report, f, indent=2)

    cm_te = confusion_matrix(yte, pte, labels=classes)
    plot_confusion_matrix(cm_te, classes, REPORTS_DIR / "test_confusion_matrix.html")

# ---------------------------------------------------------------------------
# Save model artifacts
# ---------------------------------------------------------------------------
model_dir = Path(T["model_dir"])
model_dir.mkdir(parents=True, exist_ok=True)

# 1) Model
model_path = model_dir / "model.joblib"
joblib.dump(clf, model_path)
logger.info("Model saved -> %s", model_path)

# 2) Feature order
features_out = model_dir / "features.json"
with open(features_out, "w", encoding="utf-8") as f:
    json.dump({"feature_order": list(Xtr.columns)}, f, indent=2)

# 3) Classes
classes_path = model_dir / "classes.json"
with open(classes_path, "w", encoding="utf-8") as f:
    json.dump({"classes": classes}, f, indent=2)

# 4) Feature importances
fi_path = model_dir / "feature_importances.json"
try:
    importances = clf.feature_importances_
    fi = sorted(
        [{"feature": feat, "importance": float(imp)} for feat, imp in zip(Xtr.columns, importances)],
        key=lambda x: x["importance"],
        reverse=True,
    )
    with open(fi_path, "w", encoding="utf-8") as f:
        json.dump(fi, f, indent=2)
except Exception:
    fi_path = None

# 5) Metrics
with open(METRICS_PATH, "w", encoding="utf-8") as f:
    json.dump(metrics, f, indent=4)

logger.info("Metrics: %s", json.dumps(metrics, indent=2))

# ---------------------------------------------------------------------------
# Stop carbon tracker
# ---------------------------------------------------------------------------
emissions = tracker.stop()
logger.info("Estimated emissions: %.6f kg CO₂eq", emissions if emissions else 0)

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
logger.info("Model         -> %s", model_path)
logger.info("Features      -> %s", features_out)
logger.info("Classes       -> %s", classes_path)
logger.info("Importances   -> %s", fi_path)
logger.info("Metrics       -> %s", METRICS_PATH)
logger.info("Emissions     -> %s", REPORTS_DIR / "emissions.csv")
logger.info("Done.")

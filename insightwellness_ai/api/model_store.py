"""Loaded model artifacts. Access as `model_store.model`, never
`from model_store import model` — that freezes the value at import time."""

import os

import gcsfs
import joblib
import shap

from insightwellness_ai.config import MODEL_URI

MODEL_PATH = os.environ.get("MODEL_PATH") or MODEL_URI

model = None
FEATURE_ORDER = None
explainer = None


def load_model_artifacts():
    global model, FEATURE_ORDER, explainer
    print(f"Loading model from: {MODEL_PATH}")
    try:
        fs = gcsfs.GCSFileSystem()

        with fs.open(MODEL_PATH, "rb") as f:
            model = joblib.load(f)

        FEATURE_ORDER = model.feature_names_in_.tolist()
        explainer = shap.TreeExplainer(model)
        print("Model loaded successfully from Cloud Storage!")
    except Exception as e:
        print(f"CRITICAL ERROR: Could not load model: {e}")

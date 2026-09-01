"""Holds the loaded model artifacts (model, feature order, SHAP explainer).

Always access these through the module (`model_store.model`), never via
`from model_store import model`: the from-import would freeze the value at
import time and miss both the startup load and test monkeypatching.
"""

import os

import gcsfs
import joblib
import shap

GCS_BUCKET = os.environ.get("GCS_BUCKET") or "mlops-2026-ramzan1"
MODEL_PATH = os.environ.get("MODEL_PATH") or f"gs://{GCS_BUCKET}/models/model.joblib"

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

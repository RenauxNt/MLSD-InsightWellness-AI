"""Shared fixtures for the API tests.

The app module is imported here, before any test module, with GCS blocked:
loading the real model from Cloud Storage at import time would hang (or
require credentials) in CI. Tests get a small trained stand-in model instead.
"""

from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
import shap
from sklearn.ensemble import HistGradientBoostingClassifier

with patch("gcsfs.GCSFileSystem", side_effect=RuntimeError("gcs disabled in tests")):
    from insightwellness_ai.api import app as app_module

FEATURE_ORDER = [
    "Gender",
    "Age",
    "family_history_with_overweight",
    "FAVC",
    "FCVC",
    "NCP",
    "CAEC",
    "SMOKE",
    "CH2O",
    "SCC",
    "FAF",
    "TUE",
    "CALC",
    "MTRANS_automobile",
    "MTRANS_motorbike",
    "MTRANS_bike",
    "MTRANS_walking",
]

VALID_PAYLOAD = {
    "Gender": 1,
    "Age": 25.5,
    "family_history_with_overweight": 1,
    "FAVC": 1,
    "FCVC": 2,
    "NCP": 3,
    "CAEC": 1,
    "SMOKE": 0,
    "CH2O": 2,
    "SCC": 0,
    "FAF": 1,
    "TUE": 1,
    "CALC": 1,
    "MTRANS_automobile": 1,
    "MTRANS_motorbike": 0,
    "MTRANS_bike": 0,
    "MTRANS_walking": 0,
}


@pytest.fixture
def feature_order():
    return list(FEATURE_ORDER)


@pytest.fixture
def valid_payload():
    return dict(VALID_PAYLOAD)


@pytest.fixture
def client(monkeypatch):
    rng = np.random.default_rng(42)
    n = 200
    X = pd.DataFrame({col: rng.integers(0, 4, size=n) for col in FEATURE_ORDER})
    X["Age"] = rng.uniform(15, 70, size=n)
    X = X[FEATURE_ORDER]
    y = rng.integers(0, 7, size=n)

    model = HistGradientBoostingClassifier(max_iter=5, max_depth=2, random_state=0)
    model.fit(X, y)

    monkeypatch.setattr(app_module, "model", model)
    monkeypatch.setattr(app_module, "FEATURE_ORDER", FEATURE_ORDER)
    monkeypatch.setattr(app_module, "explainer", shap.TreeExplainer(model))

    app_module.app.config["TESTING"] = True
    with app_module.app.test_client() as c:
        yield c

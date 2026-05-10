from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
import shap
from sklearn.ensemble import HistGradientBoostingClassifier

# Block the GCS call that runs at app-module import time.
# Without this the import hangs trying to authenticate to Cloud Storage.
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


def test_explain_happy_path(client):
    resp = client.post("/explain", json=VALID_PAYLOAD)
    assert resp.status_code == 200
    body = resp.get_json()
    assert body["status"] == "success"
    assert "prediction" in body
    assert isinstance(body["prediction_code"], int)
    assert isinstance(body["probabilities"], dict)
    assert len(body["probabilities"]) == 7
    assert abs(sum(body["probabilities"].values()) - 1.0) < 1e-3
    expl = body["explanation"]
    assert isinstance(expl["base_value"], float)
    assert len(expl["top_drivers"]) == 5  # new default top_n
    for d in expl["top_drivers"]:
        assert {"feature", "impact_score", "input_value"} <= d.keys()


def test_explain_top_n_query_param(client):
    resp = client.post("/explain?top_n=2", json=VALID_PAYLOAD)
    assert resp.status_code == 200
    assert len(resp.get_json()["explanation"]["top_drivers"]) == 2


def test_explain_top_n_clamped(client):
    resp = client.post("/explain?top_n=999", json=VALID_PAYLOAD)
    assert resp.status_code == 200
    assert len(resp.get_json()["explanation"]["top_drivers"]) == len(FEATURE_ORDER)


def test_explain_missing_features(client):
    resp = client.post("/explain", json={"Gender": 1, "Age": 25.5})
    assert resp.status_code == 400
    assert "Missing features" in resp.get_json()["error"]


def test_explain_invalid_enum_value(client):
    bad = dict(VALID_PAYLOAD)
    bad["Gender"] = 5  # not in valid_values [0, 1]
    resp = client.post("/explain", json=bad)
    assert resp.status_code == 400
    assert "Gender" in resp.get_json()["error"]


def test_explain_invalid_numeric_type(client):
    bad = dict(VALID_PAYLOAD)
    bad["Age"] = "twenty"
    resp = client.post("/explain", json=bad)
    assert resp.status_code == 400
    assert "Age" in resp.get_json()["error"]


def test_explain_mtrans_conflict(client):
    bad = dict(VALID_PAYLOAD)
    bad["MTRANS_automobile"] = 1
    bad["MTRANS_walking"] = 1
    resp = client.post("/explain", json=bad)
    assert resp.status_code == 400
    assert "MTRANS" in resp.get_json()["error"]


def test_explain_model_not_loaded(client, monkeypatch):
    monkeypatch.setattr(app_module, "model", None)
    resp = client.post("/explain", json=VALID_PAYLOAD)
    assert resp.status_code == 500

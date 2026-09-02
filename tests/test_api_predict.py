"""Tests for /predict and the info endpoints (/, /status, /features)."""

from insightwellness_ai.api import model_store
from insightwellness_ai.api.schema import CLASS_MAPPING


def test_predict_happy_path(client, valid_payload):
    resp = client.post("/predict", json=valid_payload)
    assert resp.status_code == 200
    body = resp.get_json()
    assert body["status"] == "success"
    assert body["prediction"] in CLASS_MAPPING.values()
    assert isinstance(body["prediction_code"], int)
    assert 0 <= body["prediction_code"] <= 6


def test_predict_missing_features(client):
    resp = client.post("/predict", json={"Gender": 1, "Age": 25.5})
    assert resp.status_code == 400
    assert "Missing features" in resp.get_json()["error"]


def test_predict_rejects_non_dict_payload(client):
    resp = client.post("/predict", json=[1, 2, 3])
    assert resp.status_code == 400


def test_predict_invalid_enum_value(client, valid_payload):
    valid_payload["FCVC"] = 9  # valid values are [1, 2, 3]
    resp = client.post("/predict", json=valid_payload)
    assert resp.status_code == 400
    assert "FCVC" in resp.get_json()["error"]


def test_predict_invalid_numeric_type(client, valid_payload):
    valid_payload["Age"] = "twenty"
    resp = client.post("/predict", json=valid_payload)
    assert resp.status_code == 400
    assert "Age" in resp.get_json()["error"]


def test_predict_mtrans_conflict(client, valid_payload):
    valid_payload["MTRANS_automobile"] = 1
    valid_payload["MTRANS_walking"] = 1
    resp = client.post("/predict", json=valid_payload)
    assert resp.status_code == 400
    assert "MTRANS" in resp.get_json()["error"]


def test_predict_model_not_loaded(client, valid_payload, monkeypatch):
    monkeypatch.setattr(model_store, "model", None)
    resp = client.post("/predict", json=valid_payload)
    assert resp.status_code == 500


def test_status_available(client):
    resp = client.get("/status")
    assert resp.status_code == 200
    assert resp.get_json() == {"status": "available"}


def test_status_unavailable_without_model(client, monkeypatch):
    monkeypatch.setattr(model_store, "model", None)
    resp = client.get("/status")
    assert resp.status_code == 500
    assert resp.get_json()["status"] == "unavailable"


def test_features_returns_full_schema(client, feature_order):
    resp = client.get("/features")
    assert resp.status_code == 200
    expected_inputs = resp.get_json()["expected_inputs"]
    assert set(expected_inputs.keys()) == set(feature_order)


def test_index_lists_all_endpoints(client):
    resp = client.get("/")
    assert resp.status_code == 200
    endpoints = resp.get_json()["endpoints"]
    for route in ("GET /status", "POST /predict", "POST /explain", "POST /chat"):
        assert route in endpoints

"""Tests for /explain (SHAP-based explanation endpoint)."""

from insightwellness_ai.api import app as app_module


def test_explain_happy_path(client, valid_payload):
    resp = client.post("/explain", json=valid_payload)
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
    assert len(expl["top_drivers"]) == 5  # default top_n
    for d in expl["top_drivers"]:
        assert {"feature", "impact_score", "input_value"} <= d.keys()


def test_explain_top_n_query_param(client, valid_payload):
    resp = client.post("/explain?top_n=2", json=valid_payload)
    assert resp.status_code == 200
    assert len(resp.get_json()["explanation"]["top_drivers"]) == 2


def test_explain_top_n_clamped(client, valid_payload, feature_order):
    resp = client.post("/explain?top_n=999", json=valid_payload)
    assert resp.status_code == 200
    assert len(resp.get_json()["explanation"]["top_drivers"]) == len(feature_order)


def test_explain_missing_features(client):
    resp = client.post("/explain", json={"Gender": 1, "Age": 25.5})
    assert resp.status_code == 400
    assert "Missing features" in resp.get_json()["error"]


def test_explain_invalid_enum_value(client, valid_payload):
    valid_payload["Gender"] = 5  # not in valid_values [0, 1]
    resp = client.post("/explain", json=valid_payload)
    assert resp.status_code == 400
    assert "Gender" in resp.get_json()["error"]


def test_explain_invalid_numeric_type(client, valid_payload):
    valid_payload["Age"] = "twenty"
    resp = client.post("/explain", json=valid_payload)
    assert resp.status_code == 400
    assert "Age" in resp.get_json()["error"]


def test_explain_mtrans_conflict(client, valid_payload):
    valid_payload["MTRANS_automobile"] = 1
    valid_payload["MTRANS_walking"] = 1
    resp = client.post("/explain", json=valid_payload)
    assert resp.status_code == 400
    assert "MTRANS" in resp.get_json()["error"]


def test_explain_model_not_loaded(client, valid_payload, monkeypatch):
    monkeypatch.setattr(app_module, "model", None)
    resp = client.post("/explain", json=valid_payload)
    assert resp.status_code == 500

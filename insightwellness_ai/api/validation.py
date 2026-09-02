"""Validation of /predict and /explain JSON payloads."""

from flask import jsonify

from insightwellness_ai.api import model_store
from insightwellness_ai.api.schema import EXPECTED_MODEL_SCHEMA, MTRANS_FEATURES


def validate_payload(data):
    """Validate a /predict or /explain JSON payload.

    Returns a Flask response tuple on failure, or None on success.
    """
    if not data or not isinstance(data, dict):
        return jsonify({"error": "Payload must be a JSON dictionary."}), 400

    missing_features = [f for f in model_store.FEATURE_ORDER if f not in data]
    if missing_features:
        return jsonify({"error": f"Missing features: {missing_features}"}), 400

    for key, value in data.items():
        if key in EXPECTED_MODEL_SCHEMA:
            valid_values = EXPECTED_MODEL_SCHEMA[key]["valid_values"]
            if isinstance(valid_values, list) and value not in valid_values:
                return jsonify(
                    {
                        "error": f"Invalid input for '{key}'. Expected one of {valid_values}, got {value}."
                    }
                ), 400
            elif isinstance(valid_values, str) and "Numeric" in valid_values:
                if not isinstance(value, (int, float)):
                    return jsonify(
                        {
                            "error": f"Invalid type for '{key}'. Expected number, got {type(value).__name__}."
                        }
                    ), 400

    mtrans_sum = sum(data.get(col, 0) for col in MTRANS_FEATURES if col in data)
    if mtrans_sum > 1:
        return jsonify(
            {"error": "Invalid transportation data. Max ONE main MTRANS mode allowed."}
        ), 400

    return None

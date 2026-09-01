"""Info endpoints: API root, health, and expected input schema."""

from flask import Blueprint, jsonify

from insightwellness_ai.api import model_store
from insightwellness_ai.api.schema import EXPECTED_MODEL_SCHEMA

info_bp = Blueprint("info", __name__)


@info_bp.route("/", methods=["GET"])
def index():
    """
    API Root endpoint.
    Returns the API name and a list of available endpoints.
    ---
    tags:
      - Info
    responses:
      200:
        description: A JSON dictionary with API details.
    """
    return jsonify(
        {
            "name": "InsightWellness API (Cloud Run Ready)",
            "endpoints": {
                "GET /status": "Check API health.",
                "GET /features": "View expected schema.",
                "POST /predict": "Send patient data JSON for prediction.",
                "POST /explain": "Send patient data JSON for prediction and SHAP feature impact explanation.",
                "POST /chat": "Ask the multi-agent team a question about a prediction, the dataset, or healthy habits.",
                "GET /apidocs": "View interactive Swagger documentation.",
            },
        }
    ), 200


@info_bp.route("/status", methods=["GET"])
def status():
    """
    Check API health and model status.
    Verifies if the machine learning model has been successfully loaded into memory.
    ---
    tags:
      - Health
    responses:
      200:
        description: API is running and model is loaded.
      500:
        description: Model failed to load.
    """
    if model_store.model is not None and model_store.FEATURE_ORDER is not None:
        return jsonify({"status": "available"}), 200
    return jsonify({"status": "unavailable", "error": "Model failed to load."}), 500


@info_bp.route("/features", methods=["GET"])
def features():
    """
    View expected model input schema.
    Returns the dictionary of expected features, descriptions, and valid values.
    ---
    tags:
      - Info
    responses:
      200:
        description: A dictionary of expected features.
    """
    return jsonify(
        {
            "status": "success",
            "expected_inputs": EXPECTED_MODEL_SCHEMA,
        }
    ), 200

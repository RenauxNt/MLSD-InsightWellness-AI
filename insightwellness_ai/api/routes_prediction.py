"""Prediction endpoints: /predict and its SHAP-explained variant /explain."""

import numpy as np
import pandas as pd
from flask import Blueprint, jsonify, request

from insightwellness_ai.api import model_store
from insightwellness_ai.api.schema import CLASS_MAPPING
from insightwellness_ai.api.validation import validate_payload

prediction_bp = Blueprint("prediction", __name__)


def _prepare_input(data):
    """Returns (error, None, None) on bad payload, else (None, ordered_input, input_df)."""
    error = validate_payload(data)
    if error is not None:
        return error, None, None

    feature_order = model_store.FEATURE_ORDER
    ordered_input = {f: data[f] for f in feature_order}
    input_df = pd.DataFrame([ordered_input], columns=feature_order)
    return None, ordered_input, input_df


@prediction_bp.route("/predict", methods=["POST"])
def predict():
    """
    Predict obesity risk level based on patient data.
    Takes a JSON payload of patient statistics and returns the predicted weight classification.
    ---
    tags:
      - Prediction
    parameters:
      - in: body
        name: patient_data
        description: JSON dictionary containing patient metrics.
        required: true
        schema:
          $ref: '#/definitions/PatientData'
    responses:
      200:
        description: Successful prediction.
      400:
        description: Invalid payload (missing features, invalid types, or bad values).
      500:
        description: Model not loaded or internal server error.
    """
    if model_store.model is None or model_store.FEATURE_ORDER is None:
        return jsonify({"error": "Model not loaded."}), 500

    try:
        error, _, input_df = _prepare_input(request.get_json())
        if error is not None:
            return error

        pred_index = int(model_store.model.predict(input_df).tolist()[0])
        predicted_class = CLASS_MAPPING.get(pred_index, "Unknown Class")

        return jsonify(
            {
                "status": "success",
                "prediction": predicted_class,
                "prediction_code": pred_index,
            }
        )

    except Exception as e:
        return jsonify({"error": str(e)}), 400


@prediction_bp.route("/explain", methods=["POST"])
def explain():
    """
    Explain a prediction using SHAP feature impact scores.
    Takes the same patient JSON payload as /predict and returns the predicted class
    plus the most influential features for that prediction.
    ---
    tags:
      - Explanation
    parameters:
      - in: query
        name: top_n
        type: integer
        required: false
        default: 5
        description: Number of top driver features to return (clamped to the feature count).
      - in: body
        name: patient_data
        description: JSON dictionary containing patient metrics (same schema as /predict).
        required: true
        schema:
          $ref: '#/definitions/PatientData'
    responses:
      200:
        description: Successful prediction with SHAP explanation.
      400:
        description: Invalid payload or explanation failure.
      500:
        description: Model not loaded or internal server error.
    """
    if (
        model_store.model is None
        or model_store.FEATURE_ORDER is None
        or model_store.explainer is None
    ):
        return jsonify({"error": "Model not loaded."}), 500

    try:
        error, ordered_input, input_df = _prepare_input(request.get_json())
        if error is not None:
            return error

        feature_order = model_store.FEATURE_ORDER
        top_n = request.args.get("top_n", default=5, type=int) or 5
        top_n = max(1, min(top_n, len(feature_order)))

        # argmax(proba) == predict(); avoids a second ensemble traversal
        proba = model_store.model.predict_proba(input_df)[0]
        pred_index = int(np.argmax(proba))
        predicted_class = CLASS_MAPPING.get(pred_index, "Unknown Class")

        probabilities = {
            CLASS_MAPPING.get(i, str(i)): round(float(p), 4)
            for i, p in enumerate(proba)
        }

        explanation = model_store.explainer(input_df)

        if explanation.values.ndim == 3:
            class_shap_values = explanation.values[0, :, pred_index]
            base_value = float(explanation.base_values[0, pred_index])
        else:
            class_shap_values = explanation.values[0]
            base_value = float(explanation.base_values[0])

        feature_impacts = [
            {
                "feature": feature,
                "impact_score": round(float(impact), 4),
                "input_value": ordered_input[feature],
            }
            for feature, impact in zip(feature_order, class_shap_values, strict=True)
        ]

        feature_impacts.sort(key=lambda x: abs(x["impact_score"]), reverse=True)

        return jsonify(
            {
                "status": "success",
                "prediction": predicted_class,
                "prediction_code": pred_index,
                "probabilities": probabilities,
                "explanation": {
                    "description": (
                        "Impact scores show how much each feature pushed the prediction toward this specific class. "
                        "Positive scores pushed toward this class, negative scores pushed away."
                    ),
                    "base_value": round(base_value, 4),
                    "top_drivers": feature_impacts[:top_n],
                },
            }
        ), 200

    except Exception as e:
        return jsonify({"error": f"Explanation failed: {str(e)}"}), 400

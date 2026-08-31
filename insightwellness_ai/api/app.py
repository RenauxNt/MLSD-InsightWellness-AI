import os
import joblib
import pandas as pd
import gcsfs
import shap
from flask import Flask, request, jsonify
from flasgger import Swagger

app = Flask(__name__)

# Initialize Flasgger
# This automatically creates a Swagger UI at /apidocs
swagger = Swagger(
    app,
    template={
        "info": {
            "title": "InsightWellness API",
            "description": "API for predicting obesity risk levels based on patient data.",
            "version": "1.0.0",
        }
    },
)

FEATURE_ORDER = None

# --- Model Feature Schema ---
EXPECTED_MODEL_SCHEMA = {
    "Gender": {"description": "Gender (0 = Male, 1 = Female)", "valid_values": [0, 1]},
    "Age": {"description": "Age in years", "valid_values": "Numeric (e.g., 25.0)"},
    "family_history_with_overweight": {
        "description": "Family history of overweight (0 = False, 1 = True)",
        "valid_values": [0, 1],
    },
    "FAVC": {
        "description": "Frequent consumption of high caloric food (0 = False, 1 = True)",
        "valid_values": [0, 1],
    },
    "FCVC": {
        "description": "Frequency of consumption of vegetables (1 = Never, 2 = Sometimes, 3 = Always)",
        "valid_values": [1, 2, 3],
    },
    "NCP": {
        "description": "Number of main meals per day (1, 2, 3, or 4+ meals)",
        "valid_values": [1, 2, 3, 4],
    },
    "CAEC": {
        "description": "Consumption of food between meals (0=No, 1=Sometimes, 2=Frequently, 3=Always)",
        "valid_values": [0, 1, 2, 3],
    },
    "SMOKE": {
        "description": "Does the patient smoke? (0 = False, 1 = True)",
        "valid_values": [0, 1],
    },
    "CH2O": {
        "description": "Consumption of water daily (1 = Less than 1L, 2 = 1 to 2L, 3 = More than 2L)",
        "valid_values": [1, 2, 3],
    },
    "SCC": {
        "description": "Calories consumption monitoring (0 = False, 1 = True)",
        "valid_values": [0, 1],
    },
    "FAF": {
        "description": "Physical activity frequency (0 = None, 1 = 1 to 2 days, 2 = 2 to 4 days, 3 = 4 to 5 days)",
        "valid_values": [0, 1, 2, 3],
    },
    "TUE": {
        "description": "Time using technology devices (0 = 0 to 2 hours, 1 = 3 to 5 hours, 2 = More than 5 hours)",
        "valid_values": [0, 1, 2],
    },
    "CALC": {
        "description": "Consumption of alcohol (0=No, 1=Sometimes, 2=Frequently, 3=Always)",
        "valid_values": [0, 1, 2, 3],
    },
    "MTRANS_automobile": {
        "description": "Main transport is automobile (0=No, 1=Yes). Max ONE MTRANS variable can be 1.",
        "valid_values": [0, 1],
    },
    "MTRANS_motorbike": {
        "description": "Main transport is motorbike (0=No, 1=Yes). Max ONE MTRANS variable can be 1",
        "valid_values": [0, 1],
    },
    "MTRANS_bike": {
        "description": "Main transport is bike (0=No, 1=Yes). Max ONE MTRANS variable can be 1.",
        "valid_values": [0, 1],
    },
    "MTRANS_walking": {
        "description": "Main transport is walking (0=No, 1=Yes). Max ONE MTRANS variable can be 1.",
        "valid_values": [0, 1],
    },
}

# Translates the model's numeric output back to readable labels
CLASS_MAPPING = {
    0: "Insufficient_Weight",
    1: "Normal_Weight",
    2: "Overweight_Level_I",
    3: "Overweight_Level_II",
    4: "Obesity_Type_I",
    5: "Obesity_Type_II",
    6: "Obesity_Type_III",
}

# --- Load Model ---
GCS_BUCKET = os.environ.get("GCS_BUCKET") or "mlops-2026-ramzan1"
MODEL_PATH = os.environ.get("MODEL_PATH") or f"gs://{GCS_BUCKET}/models/model.joblib"

model = None
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


load_model_artifacts()


def validate_payload(data):
    """Validate a /predict or /explain JSON payload.

    Returns a Flask response tuple on failure, or None on success.
    """
    if not data or not isinstance(data, dict):
        return jsonify({"error": "Payload must be a JSON dictionary."}), 400

    missing_features = [f for f in FEATURE_ORDER if f not in data]
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

    mtrans_columns = [
        "MTRANS_automobile",
        "MTRANS_motorbike",
        "MTRANS_bike",
        "MTRANS_walking",
    ]
    mtrans_sum = sum(data.get(col, 0) for col in mtrans_columns if col in data)
    if mtrans_sum > 1:
        return jsonify(
            {"error": "Invalid transportation data. Max ONE main MTRANS mode allowed."}
        ), 400

    return None


# --- Endpoints ---


@app.route("/", methods=["GET"])
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
                "GET /apidocs": "View interactive Swagger documentation.",
            },
        }
    ), 200


@app.route("/status", methods=["GET"])
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
    if model is not None and FEATURE_ORDER is not None:
        return jsonify({"status": "available"}), 200
    return jsonify({"status": "unavailable", "error": "Model failed to load."}), 500


@app.route("/features", methods=["GET"])
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


@app.route("/predict", methods=["POST"])
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
          type: object
          properties:
            Gender:
              type: integer
              example: 1
            Age:
              type: number
              example: 25.5
            family_history_with_overweight:
              type: integer
              example: 1
            FAVC:
              type: integer
              example: 1
            FCVC:
              type: integer
              example: 2
            NCP:
              type: integer
              example: 3
            CAEC:
              type: integer
              example: 1
            SMOKE:
              type: integer
              example: 0
            CH2O:
              type: integer
              example: 2
            SCC:
              type: integer
              example: 0
            FAF:
              type: integer
              example: 1
            TUE:
              type: integer
              example: 1
            CALC:
              type: integer
              example: 1
            MTRANS_automobile:
              type: integer
              example: 1
            MTRANS_motorbike:
              type: integer
              example: 0
            MTRANS_bike:
              type: integer
              example: 0
            MTRANS_walking:
              type: integer
              example: 0
    responses:
      200:
        description: Successful prediction.
      400:
        description: Invalid payload (missing features, invalid types, or bad values).
      500:
        description: Model not loaded or internal server error.
    """
    if model is None or FEATURE_ORDER is None:
        return jsonify({"error": "Model not loaded."}), 500

    try:
        data = request.get_json()

        error = validate_payload(data)
        if error is not None:
            return error

        ordered_input = {f: data[f] for f in FEATURE_ORDER}
        input_df = pd.DataFrame([ordered_input], columns=FEATURE_ORDER)

        pred_index = int(model.predict(input_df).tolist()[0])
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


@app.route("/explain", methods=["POST"])
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
        description: Number of top driver features to return (clamped to [1, 17]).
      - in: body
        name: patient_data
        description: JSON dictionary containing patient metrics (same schema as /predict).
        required: true
        schema:
          type: object
          properties:
            Gender:
              type: integer
              example: 1
            Age:
              type: number
              example: 25.5
            family_history_with_overweight:
              type: integer
              example: 1
            FAVC:
              type: integer
              example: 1
            FCVC:
              type: integer
              example: 2
            NCP:
              type: integer
              example: 3
            CAEC:
              type: integer
              example: 1
            SMOKE:
              type: integer
              example: 0
            CH2O:
              type: integer
              example: 2
            SCC:
              type: integer
              example: 0
            FAF:
              type: integer
              example: 1
            TUE:
              type: integer
              example: 1
            CALC:
              type: integer
              example: 1
            MTRANS_automobile:
              type: integer
              example: 1
            MTRANS_motorbike:
              type: integer
              example: 0
            MTRANS_bike:
              type: integer
              example: 0
            MTRANS_walking:
              type: integer
              example: 0
    responses:
      200:
        description: Successful prediction with SHAP explanation.
      400:
        description: Invalid payload or explanation failure.
      500:
        description: Model not loaded or internal server error.
    """
    if model is None or FEATURE_ORDER is None or explainer is None:
        return jsonify({"error": "Model not loaded."}), 500

    try:
        data = request.get_json()

        error = validate_payload(data)
        if error is not None:
            return error

        top_n = request.args.get("top_n", default=5, type=int) or 5
        top_n = max(1, min(top_n, len(FEATURE_ORDER)))

        ordered_input = {f: data[f] for f in FEATURE_ORDER}
        input_df = pd.DataFrame([ordered_input], columns=FEATURE_ORDER)

        pred_index = int(model.predict(input_df).tolist()[0])
        predicted_class = CLASS_MAPPING.get(pred_index, "Unknown Class")

        proba = model.predict_proba(input_df)[0].tolist()
        probabilities = {
            CLASS_MAPPING.get(i, str(i)): round(float(p), 4)
            for i, p in enumerate(proba)
        }

        explanation = explainer(input_df)

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
            for feature, impact in zip(FEATURE_ORDER, class_shap_values)
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


_agent_team = None


def get_agent_team():
    # Lazy import: loading the team pulls in agno/chromadb and indexes the
    # knowledge base, which must not slow down or break API startup.
    global _agent_team
    if _agent_team is None:
        from insightwellness_ai.agents.team import team

        _agent_team = team
    return _agent_team


@app.route("/chat", methods=["POST"])
def chat():
    """
    Ask the multi-agent team a question about a prediction, the dataset, or healthy habits.
    The team coordinates a predictor agent (calls /predict), a SHAP explainability agent
    (calls /explain), a RAG agent grounded in the project knowledge base, and a web
    research agent.
    ---
    tags:
      - Chat
    parameters:
      - in: body
        name: chat_request
        required: true
        schema:
          type: object
          properties:
            question:
              type: string
              example: Why was I classified as Overweight_Level_I?
    responses:
      200:
        description: Answer from the agent team.
      400:
        description: Missing question.
      500:
        description: Agent team failure.
    """
    data = request.get_json(silent=True) or {}
    question = data.get("question", "")
    if not question:
        return jsonify({"error": "No question provided"}), 400

    try:
        team = get_agent_team()
        response = team.run(question, stream=False)
        answer = response.content if hasattr(response, "content") else str(response)
        return jsonify({"answer": answer}), 200
    except Exception as e:
        return jsonify({"error": f"Chat failed: {str(e)}"}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port, debug=True)

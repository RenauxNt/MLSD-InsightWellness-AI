import os
import joblib
import pandas as pd
import gcsfs
from flask import Flask, request, jsonify

app = Flask(__name__)

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
MODEL_PATH = "gs://mlops-2026-ramzan1/models/model.joblib"

model = None

def load_model_artifacts():
    global model, FEATURE_ORDER
    print(f"Loading model from: {MODEL_PATH}")
    try:
        fs = gcsfs.GCSFileSystem()
        
        with fs.open(MODEL_PATH, 'rb') as f:
            model = joblib.load(f)
            
        FEATURE_ORDER = model.feature_names_in_.tolist()
        print("Model loaded successfully from Cloud Storage!")
    except Exception as e:
        print(f"CRITICAL ERROR: Could not load model: {e}")

load_model_artifacts()

# --- Endpoints ---


@app.route("/", methods=["GET"])
def index():
    return jsonify(
        {
            "name": "InsightWellness API (Cloud Run Ready)",
            "endpoints": {
                "GET /status": "Check API health.",
                "GET /features": "View expected schema.",
                "POST /predict": "Send patient data JSON for prediction.",
            },
        }
    ), 200


@app.route("/status", methods=["GET"])
def status():
    if model is not None and FEATURE_ORDER is not None:
        return jsonify({"status": "available"}), 200
    return jsonify({"status": "unavailable", "error": "Model failed to load."}), 500


@app.route("/features", methods=["GET"])
def features():
    return jsonify(
        {
            "status": "success",
            "expected_inputs": EXPECTED_MODEL_SCHEMA,
        }
    ), 200


@app.route("/predict", methods=["POST"])
def predict():
    if model is None or FEATURE_ORDER is None:
        return jsonify({"error": "Model not loaded."}), 500

    try:
        data = request.get_json()

        if not data or not isinstance(data, dict):
            return jsonify({"error": "Payload must be a JSON dictionary."}), 400

        # Check for missing features
        missing_features = [f for f in FEATURE_ORDER if f not in data]
        if missing_features:
            return jsonify({"error": f"Missing features: {missing_features}"}), 400

        # Check input values
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

        # Check that only one MTRANS variable is 1
        mtrans_columns = [
            "MTRANS_automobile",
            "MTRANS_motorbike",
            "MTRANS_bike",
            "MTRANS_walking",
        ]
        mtrans_sum = sum(data.get(col, 0) for col in mtrans_columns if col in data)
        if mtrans_sum > 1:
            return jsonify(
                {
                    "error": "Invalid transportation data. Max ONE main MTRANS mode allowed."
                }
            ), 400

        # Reorder input data to match model's expected feature order
        ordered_input = {f: data[f] for f in FEATURE_ORDER}
        input_df = pd.DataFrame([ordered_input], columns=FEATURE_ORDER)

        # Get the numeric prediction
        pred_index = int(model.predict(input_df).tolist()[0])

        # Use the dictionary to translate the number to the readable class name
        predicted_class = CLASS_MAPPING.get(pred_index, "Unknown Class")

        # Return the beautifully translated data
        return jsonify(
            {
                "status": "success",
                "prediction": predicted_class,
                "prediction_code": pred_index,
            }
        )

    except Exception as e:
        return jsonify({"error": str(e)}), 400


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port, debug=True)

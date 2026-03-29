import json
import os
from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

# --- Model Feature Schema ---
EXPECTED_MODEL_SCHEMA = {
    "Gender": {"description": "Gender (0 = Male, 1 = Female)", "valid_values": [0, 1]},
    "Age": {"description": "Age in years", "valid_values": "Numeric (e.g., 25.0)"},
    "family_history_with_overweight": {"description": "Family history of overweight (0 = False, 1 = True)", "valid_values": [0, 1]},
    "FAVC": {"description": "Frequent consumption of high caloric food (0 = False, 1 = True)", "valid_values": [0, 1]},
    "FCVC": {"description": "Frequency of consumption of vegetables (1 = Never, 2 = Sometimes, 3 = Always)", "valid_values": [1, 2, 3]},
    "NCP": {"description": "Number of main meals per day (1, 2, 3, or 4+ meals)", "valid_values": [1, 2, 3, 4]},
    "CAEC": {"description": "Consumption of food between meals (0=No, 1=Sometimes, 2=Frequently, 3=Always)", "valid_values": [0, 1, 2, 3]},
    "SMOKE": {"description": "Does the patient smoke? (0 = False, 1 = True)", "valid_values": [0, 1]},
    "CH2O": {"description": "Consumption of water daily (1 = Less than 1L, 2 = 1 to 2L, 3 = More than 2L)", "valid_values": [1, 2, 3]},
    "SCC": {"description": "Calories consumption monitoring (0 = False, 1 = True)", "valid_values": [0, 1]},
    "FAF": {"description": "Physical activity frequency (0 = None, 1 = 1 to 2 days, 2 = 2 to 4 days, 3 = 4 to 5 days)", "valid_values": [0, 1, 2, 3]},
    "TUE": {"description": "Time using technology devices (0 = 0 to 2 hours, 1 = 3 to 5 hours, 2 = More than 5 hours)", "valid_values": [0, 1, 2]},
    "CALC": {"description": "Consumption of alcohol (0=No, 1=Sometimes, 2=Frequently, 3=Always)", "valid_values": [0, 1, 2, 3]},
    "MTRANS_automobile": {"description": "Main transport is automobile (0=No, 1=Yes). Max ONE MTRANS variable can be 1.", "valid_values": [0, 1]},
    "MTRANS_motorbike": {"description": "Main transport is motorbike (0=No, 1=Yes). Max ONE MTRANS variable can be 1", "valid_values": [0, 1]},
    "MTRANS_bike": {"description": "Main transport is bike (0=No, 1=Yes). Max ONE MTRANS variable can be 1.", "valid_values": [0, 1]},
    "MTRANS_walking": {"description": "Main transport is walking (0=No, 1=Yes). Max ONE MTRANS variable can be 1. (Note: If all MTRANS are 0, it implies Public Transportation).", "valid_values": [0, 1]}
}

MODEL_DIR = "models" 
MODEL_PATH = os.path.join(MODEL_DIR, "model.joblib")
FEATURES_PATH = os.path.join(MODEL_DIR, "features.json")
CLASSES_PATH = os.path.join(MODEL_DIR, "classes.json")

model = None
feature_order = None
class_labels = None

def load_model_artifacts():
    """Load model and features into global memory."""
    global model, feature_order, class_labels
    print("Loading model and artifacts...")
    try:
        model = joblib.load(MODEL_PATH)
        
        with open(FEATURES_PATH, "r", encoding="utf-8") as f:
            feature_order = json.load(f)["feature_order"]
            
        try:
            with open(CLASSES_PATH, "r", encoding="utf-8") as f:
                class_labels = json.load(f)["classes"]
        except FileNotFoundError:
            class_labels = getattr(model, "classes_", [])
            
        print("Model and artifacts loaded successfully!")
    except Exception as e:
        print(f"CRITICAL ERROR: Could not load model artifacts: {e}")

# Runs when the server starts
load_model_artifacts()

# --- Endpoints ---

@app.route('/', methods=['GET'])
def index():
    """Root endpoint providing basic API documentation."""
    return jsonify({
        "name": "InsightWellness API",
        "description": "Machine Learning API to predict Obesity levels.",
        "endpoints": {
            "GET /status": "Check if API is running and model is loaded.",
            "GET /features": "View the feature names and data format required for prediction.",
            "POST /predict": "Send JSON data here to get a model prediction."
        }
    }), 200

@app.route('/features', methods=['GET'])
def features():
    """Endpoint to expose the expected input schema and valid values."""
    if feature_order is None:
        return jsonify({"error": "Features not loaded."}), 500
        
    # Build a detailed dictionary for the features expected by the model
    detailed_features = {}
    for feature in feature_order:
        # If the feature matches our defined schema exactly
        if feature in EXPECTED_MODEL_SCHEMA:
            detailed_features[feature] = EXPECTED_MODEL_SCHEMA[feature]
        else:
            # Fallback (keeps the same dictionary structure as the rest)
            detailed_features[feature] = {
                "description": f"Encoded or derived feature for {feature}",
                "valid_values": "Unknown. Check model preprocessing pipeline."
            }
            
    return jsonify({
        "status": "success",
        "message": "Send a POST request to /predict with a JSON body containing these exact keys.",
        "expected_inputs": detailed_features,
        "possible_prediction_classes": class_labels
    }), 200


@app.route('/status', methods=['GET'])
def status():
    """Endpoint to check if the API is running and the model is loaded."""
    if model is not None and feature_order is not None:
        return jsonify({
            "status": "success",
            "message": "InsightWellness API is running and ready!"
        }), 200
    else:
        return jsonify({
            "status": "error",
            "message": "InsightWellness API is running, but the model failed to load."
        }), 500

@app.route('/predict', methods=['POST'])
def predict():
    """Endpoint to generate predictions."""
    if model is None or feature_order is None:
         return jsonify({"error": "Model artifacts not loaded properly."}), 500

    try:
        data = request.get_json()
        
        # Validate that all required features are present
        missing_features = [f for f in feature_order if f not in data]
        if missing_features:
            return jsonify({
                "error": "Missing required features in the JSON payload.",
                "missing_features": missing_features,
                "hint": "Make a GET request to /features to see the full list of required keys."
            }), 400

        # Validate data types and values against the schema
        for key, value in data.items():
            if key in EXPECTED_MODEL_SCHEMA:
                valid_values = EXPECTED_MODEL_SCHEMA[key]["valid_values"]
                
                if isinstance(valid_values, list) and value not in valid_values:
                    return jsonify({
                        "error": f"Invalid input for '{key}'",
                        "message": f"Expected one of {valid_values}, but got {value}."
                    }), 400
                
                elif isinstance(valid_values, str) and "Numeric" in valid_values:
                    if not isinstance(value, (int, float)):
                        return jsonify({
                            "error": f"Invalid type for '{key}'",
                            "message": f"Expected a number, but got {type(value).__name__}."
                        }), 400

        # Validate mutually exclusive MTRANS columns
        mtrans_columns = ["MTRANS_automobile", "MTRANS_motorbike", "MTRANS_bike", "MTRANS_walking"]
        mtrans_sum = sum(data.get(col, 0) for col in mtrans_columns if col in data)
        
        if mtrans_sum > 1:
            return jsonify({
                "error": "Invalid transportation data.",
                "message": "You can only select a maximum of ONE main mode of transportation. "
                "Please ensure only one MTRANS field is set to 1, or set all to 0 for Public Transportation."
            }), 400

        # Create a DataFrame using the exact feature order expected by the model
        input_df = pd.DataFrame([data], columns=feature_order)
        
        # Generate prediction
        prediction = model.predict(input_df)
        
        # (Optional) Get prediction probabilities if your model supports it
        probabilities = model.predict_proba(input_df).tolist()[0]
        
        return jsonify({
            "status": "success",
            "prediction": prediction.tolist()[0],
            "probabilities": probabilities
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
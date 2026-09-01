"""InsightWellness API: Flask app assembly.

Routes live in the routes_* modules, the input schema in schema.py, payload
validation in validation.py, and model artifact loading in model_store.py.
"""

import os

from flasgger import Swagger
from flask import Flask

from insightwellness_ai.api import model_store
from insightwellness_ai.api.routes_chat import chat_bp
from insightwellness_ai.api.routes_info import info_bp
from insightwellness_ai.api.routes_prediction import prediction_bp

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

app.register_blueprint(info_bp)
app.register_blueprint(prediction_bp)
app.register_blueprint(chat_bp)

model_store.load_model_artifacts()


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port, debug=True)

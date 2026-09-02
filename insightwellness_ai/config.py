import os

# `or` (not a get() default) so an empty env var also falls back
PROJECT_ID = os.environ.get("GCP_PROJECT_ID") or "mlsd-487610"
BUCKET_NAME = os.environ.get("GCS_BUCKET") or "mlops-2026-ramzan1"
LOCATION = os.environ.get("GCP_REGION") or "europe-west1"

RAW_DATA = f"gs://{BUCKET_NAME}/raw/data_raw.csv"

# written by vertex/training.py, read by api/model_store.py
MODEL_BLOB = "models/model.joblib"
MODEL_URI = f"gs://{BUCKET_NAME}/{MODEL_BLOB}"

API_BASE_URL = (
    os.environ.get("INSIGHTWELLNESS_API_URL")
    or "https://insightwellness-api-545205658175.europe-west1.run.app"
)

TARGET = "Obesity"
TEST_SIZE = 0.2
RANDOM_STATE = 42
STRATIFY = True

TRAIN_CONFIG = {
    "n_estimators": 500,
    "random_state": 257,
    "param_grid": {
        "max_depth": [3, 5, 7],
        "learning_rate": [0.02, 0.05, 0.1, 0.2],
    },
}

PIPELINE_ROOT = f"gs://{BUCKET_NAME}/pipeline-root"

BQ_DATASET = "obesity_dataset"
BQ_RAW_TABLE = "raw"

BASE_IMAGE = f"{LOCATION}-docker.pkg.dev/{PROJECT_ID}/vertex-ai-pipeline-example/pipeline-base:latest"

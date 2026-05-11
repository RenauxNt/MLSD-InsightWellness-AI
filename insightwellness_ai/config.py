PROJECT_ID = "mlsd-487610"
BUCKET_NAME = "mlops-2026-ramzan1"

RAW_DATA = "gs://mlops-2026-ramzan1/raw/data_raw.csv"

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
LOCATION = "europe-west1"

BQ_DATASET = "obesity_dataset"
BQ_TRAIN_TABLE = "train_table"
BQ_TEST_TABLE = "test_table"
BQ_RAW_TABLE = "raw"

BASE_IMAGE = f"{LOCATION}-docker.pkg.dev/{PROJECT_ID}/vertex-ai-pipeline-example/pipeline-base:latest"

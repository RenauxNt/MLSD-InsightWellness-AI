from kfp.dsl import Artifact, Dataset, Input, Model, Output, component

from insightwellness_ai.config import BASE_IMAGE


@component(base_image=BASE_IMAGE)
def training(
    train_dataset: Input[Dataset],
    best_params: Input[Artifact],
    model: Output[Model],
):
    import json
    import os

    import joblib
    import pandas as pd
    from google.cloud import storage

    from insightwellness_ai.config import BUCKET_NAME, MODEL_BLOB, TARGET, TRAIN_CONFIG
    from insightwellness_ai.pipeline.training import train_model

    N_ESTIMATORS = TRAIN_CONFIG["n_estimators"]
    RANDOM_STATE = TRAIN_CONFIG["random_state"]

    df = pd.read_parquet(train_dataset.path)
    X_train = df.drop(columns=[TARGET])
    y_train = df[TARGET]

    with open(best_params.path) as f:
        params = json.load(f)

    clf = train_model(X_train, y_train, params, N_ESTIMATORS, RANDOM_STATE)

    os.makedirs(model.path, exist_ok=True)

    local_model_file = os.path.join(model.path, "model.joblib")
    joblib.dump(clf, local_model_file)

    client = storage.Client()
    bucket = client.bucket(BUCKET_NAME)
    blob = bucket.blob(MODEL_BLOB)
    blob.upload_from_filename(local_model_file)

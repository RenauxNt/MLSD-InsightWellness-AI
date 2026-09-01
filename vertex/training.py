from kfp.dsl import component, Input, Output, Dataset, Artifact, Model
from insightwellness_ai.config import BASE_IMAGE


@component(base_image=BASE_IMAGE)
def training(
    train_dataset: Input[Dataset],
    best_params: Input[Artifact],
    model: Output[Model],
):
    import pandas as pd
    import joblib
    import json
    import os
    from google.cloud import storage

    from insightwellness_ai.config import TRAIN_CONFIG, TARGET, BUCKET_NAME
    from insightwellness_ai.pipeline.training import train_model

    N_ESTIMATORS = TRAIN_CONFIG["n_estimators"]
    RANDOM_STATE = TRAIN_CONFIG["random_state"]

    df = pd.read_parquet(train_dataset.path)
    X_train = df.drop(columns=[TARGET])
    y_train = df[TARGET]

    with open(best_params.path, "r") as f:
        params = json.load(f)

    clf = train_model(X_train, y_train, params, N_ESTIMATORS, RANDOM_STATE)

    os.makedirs(model.path, exist_ok=True)

    local_model_file = os.path.join(model.path, "model.joblib")
    joblib.dump(clf, local_model_file)

    client = storage.Client()
    bucket = client.bucket(BUCKET_NAME)
    blob = bucket.blob("models/model.joblib")
    blob.upload_from_filename(local_model_file)

from kfp.dsl import Artifact, Dataset, Input, Metrics, Output, component

from insightwellness_ai.config import BASE_IMAGE


@component(base_image=BASE_IMAGE)
def model_selection(
    train_dataset: Input[Dataset],
    best_params: Output[Artifact],
    metrics: Output[Metrics],
):
    import json

    import pandas as pd

    from insightwellness_ai.config import TARGET, TRAIN_CONFIG
    from insightwellness_ai.pipeline.model_selection import select_model

    N_ESTIMATORS = TRAIN_CONFIG["n_estimators"]
    RANDOM_STATE = TRAIN_CONFIG["random_state"]
    PARAM_GRID = TRAIN_CONFIG["param_grid"]

    df = pd.read_parquet(train_dataset.path)

    X = df.drop(columns=[TARGET])
    y = df[TARGET]

    params, score = select_model(X, y, PARAM_GRID, N_ESTIMATORS, RANDOM_STATE)

    with open(best_params.path, "w") as f:
        json.dump(params, f)

    metrics.log_metric("best_cv_f1", float(score))

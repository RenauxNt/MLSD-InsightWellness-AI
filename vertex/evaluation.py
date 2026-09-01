from kfp.dsl import component, Input, Output, Dataset, Model, Metrics, Artifact
from insightwellness_ai.config import BASE_IMAGE


@component(base_image=BASE_IMAGE)
def evaluation(
    test_dataset: Input[Dataset],
    model: Input[Model],
    metrics: Output[Metrics],
    confusion_matrix_artifact: Output[Artifact],
    report_artifact: Output[Artifact],
):
    import pandas as pd
    import joblib
    import matplotlib.pyplot as plt
    import json
    import os

    from insightwellness_ai.config import TARGET
    from insightwellness_ai.pipeline.evaluation import evaluate_model

    df = pd.read_parquet(test_dataset.path)
    X_test = df.drop(columns=[TARGET])
    y_test = df[TARGET]

    clf = joblib.load(os.path.join(model.path, "model.joblib"))
    eval_metrics, report, cm = evaluate_model(clf, X_test, y_test)

    for k, v in eval_metrics.items():
        metrics.log_metric(k, v)

    plt.figure()
    plt.imshow(cm)
    plt.title("Confusion Matrix")
    plt.colorbar()
    plt.xlabel("Predicted")
    plt.ylabel("True")

    plt.savefig(confusion_matrix_artifact.path)
    plt.close()

    with open(report_artifact.path, "w") as f:
        json.dump(report, f, indent=2)

from kfp import dsl, compiler
from google.cloud import aiplatform

from insightwellness_ai.config import (
    PIPELINE_ROOT,
    PROJECT_ID,
    BQ_DATASET,
    BQ_RAW_TABLE,
    LOCATION,
)
from vertex.data_ingestion import data_ingestion
from vertex.preprocess import preprocess
from vertex.model_selection import model_selection
from vertex.training import training
from vertex.evaluation import evaluation


@dsl.pipeline(
    name="insightwellness-ml-pipeline",
    pipeline_root=PIPELINE_ROOT,
)
def pipeline(
    bq_project: str = PROJECT_ID,
    bq_dataset: str = BQ_DATASET,
    bq_table: str = BQ_RAW_TABLE,
):
    ingestion_task = data_ingestion(
        bq_project=bq_project,
        bq_dataset=bq_dataset,
        bq_table=bq_table,
    )

    preprocess_task = preprocess(input_dataset=ingestion_task.outputs["dataset"])

    model_selection_task = model_selection(
        train_dataset=preprocess_task.outputs["train_dataset"]
    )

    training_task = training(
        train_dataset=preprocess_task.outputs["train_dataset"],
        best_params=model_selection_task.outputs["best_params"],
    )

    evaluation(
        model=training_task.outputs["model"],
        test_dataset=preprocess_task.outputs["test_dataset"],
    )


if __name__ == "__main__":
    compiler.Compiler().compile(
        pipeline_func=pipeline,
        package_path="pipeline.json",
    )

    aiplatform.init(
        project=PROJECT_ID,
        location=LOCATION,
    )

    job = aiplatform.PipelineJob(
        display_name="insightwellness-ml-pipeline",
        template_path="pipeline.json",
        pipeline_root=PIPELINE_ROOT,
    )

    job.run()

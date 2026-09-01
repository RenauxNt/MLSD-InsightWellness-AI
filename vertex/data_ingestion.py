# src/data_ingestion.py
from kfp.dsl import Dataset, Output, component

from insightwellness_ai.config import BASE_IMAGE


@component(base_image=BASE_IMAGE)
def data_ingestion(
    bq_project: str, bq_dataset: str, bq_table: str, dataset: Output[Dataset]
):
    from google.cloud import bigquery

    client = bigquery.Client(project=bq_project)
    extract_job = client.extract_table(
        f"{bq_project}.{bq_dataset}.{bq_table}",
        destination_uris=[f"{dataset.uri}/*.parquet"],
        job_config=bigquery.ExtractJobConfig(
            destination_format=bigquery.DestinationFormat.PARQUET
        ),
    )
    extract_job.result()
    print(f"Exported to {dataset.uri}")

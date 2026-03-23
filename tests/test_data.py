from insightwellness_ai.preprocess import preprocess_data
from insightwellness_ai.config import (
    PROJECT_ID,
    BQ_DATASET,
    BQ_RAW_TABLE,
    TEST_SIZE,
    RANDOM_STATE,
    STRATIFY,
)
from google.cloud import bigquery

client = bigquery.Client()


def load_bq_table(table_id: str):
    query = f"SELECT * FROM `{table_id}`"
    return client.query(query).to_dataframe()


def test_NaNs():
    df = load_bq_table(f"{PROJECT_ID}.{BQ_DATASET}.{BQ_RAW_TABLE}")

    total_nans = df.isna().sum().sum()

    train_df, test_df = preprocess_data(df, TEST_SIZE, RANDOM_STATE, STRATIFY)

    total_nans_train = train_df.isna().sum().sum()
    total_nans_test = test_df.isna().sum().sum()

    assert total_nans == 0, "NaNs found in raw data"
    assert total_nans_train == 0, "NaNs found in training data"
    assert total_nans_test == 0, "NaNs found in test data"

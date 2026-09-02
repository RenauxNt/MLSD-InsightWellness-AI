from kfp.dsl import Dataset, Input, Output, component

from insightwellness_ai.config import BASE_IMAGE


@component(base_image=BASE_IMAGE)
def preprocess(
    input_dataset: Input[Dataset],
    train_dataset: Output[Dataset],
    test_dataset: Output[Dataset],
):
    import logging
    import os

    import pandas as pd

    from insightwellness_ai.config import RANDOM_STATE, STRATIFY, TEST_SIZE
    from insightwellness_ai.pipeline.preprocess import preprocess_data

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    logger = logging.getLogger(__name__)

    logger.info("Loading dataset from %s", input_dataset.path)
    df = pd.read_parquet(input_dataset.path)

    train_df, test_df = preprocess_data(df, TEST_SIZE, RANDOM_STATE, STRATIFY)

    logger.info("Train shape: %s", train_df.shape)
    logger.info("Test shape: %s", test_df.shape)

    logger.info("Saving outputs...")
    os.makedirs(train_dataset.path, exist_ok=True)
    os.makedirs(test_dataset.path, exist_ok=True)
    train_df.to_parquet(train_dataset.path + "/data.parquet")
    test_df.to_parquet(test_dataset.path + "/data.parquet")

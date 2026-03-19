import logging

import pandas as pd
import yaml

from google.cloud import bigquery
from sklearn.model_selection import train_test_split

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
params = yaml.safe_load(open("params.yaml", "r", encoding="utf-8"))

# ---------------------------------------------------------------------------
# Load raw data
# ---------------------------------------------------------------------------
logger.info("Loading raw data from %s", params["data"]["raw"])
raw_df = pd.read_csv(params["data"]["raw"])
raw_df = raw_df.drop(columns=["Unnamed: 0"], errors="ignore")
logger.info("Raw dataset: %d rows, %d columns", raw_df.shape[0], raw_df.shape[1])

raw_df = raw_df.drop(columns=["Weight", "Height"], errors="ignore")
logger.info("Dropped columns: %s", ["Weight", "Height"])

# ---------------------------------------------------------------------------
# Text normalisation
# ---------------------------------------------------------------------------
text_columns = [
    "CAEC",
    "CALC",
    "Gender",
    "NObeyesdad",
    "family_history_with_overweight",
    "FAVC",
    "SMOKE",
    "SCC",
    "MTRANS",
]

for col in text_columns:
    if col in raw_df.columns:
        raw_df[col] = (
            raw_df[col].astype("string").str.normalize("NFKC").str.strip().str.lower()
        )

# ---------------------------------------------------------------------------
# Ordinal encoding
# ---------------------------------------------------------------------------
caec_calc_mapping = {"no": 0, "sometimes": 1, "frequently": 2, "always": 3}
for col in ["CAEC", "CALC"]:
    unmapped = set(raw_df[col].dropna().unique()) - set(caec_calc_mapping.keys())
    if unmapped:
        logger.warning("Unmapped values in %s: %s", col, unmapped)
    raw_df[col] = raw_df[col].map(caec_calc_mapping).astype("Int64")

# ---------------------------------------------------------------------------
# Target encoding
# ---------------------------------------------------------------------------
obesity_mapping = {
    "insufficient_weight": 0,
    "normal_weight": 1,
    "overweight_level_i": 2,
    "overweight_level_ii": 3,
    "obesity_type_i": 4,
    "obesity_type_ii": 5,
    "obesity_type_iii": 6,
}
unmapped = set(raw_df["NObeyesdad"].dropna().unique()) - set(obesity_mapping.keys())
if unmapped:
    logger.warning("Unmapped values in NObeyesdad: %s", unmapped)
raw_df["Obesity"] = raw_df["NObeyesdad"].map(obesity_mapping).astype("Int64")
raw_df.drop(columns=["NObeyesdad"], inplace=True)

# ---------------------------------------------------------------------------
# Binary encoding
# ---------------------------------------------------------------------------
gender_mapping = {"male": 0, "female": 1}
raw_df["Gender"] = raw_df["Gender"].map(gender_mapping).astype("Int64")

binary_mapping = {"yes": 1, "no": 0}
for col in ["family_history_with_overweight", "FAVC", "SMOKE", "SCC"]:
    raw_df[col] = raw_df[col].map(binary_mapping).astype("Int64")

# ---------------------------------------------------------------------------
# One-hot encoding (MTRANS)
# ---------------------------------------------------------------------------
raw_df = pd.get_dummies(raw_df, columns=["MTRANS"], drop_first=False, dtype=int)
raw_df.drop(columns=["MTRANS_public_transportation"], inplace=True)

# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------
logger.info(
    "Preprocessed dataset: %d rows, %d columns", raw_df.shape[0], raw_df.shape[1]
)
raw_df.to_csv(params["data"]["preprocessed"], index=False)
logger.info("Saved -> %s", params["data"]["preprocessed"])
logger.info("Done.")


logger.info("Splitting dataset...")

bq_client = bigquery.Client(project=params["bq"]["project_id"])


def upload_to_bq(df, table_name):
    job_config = bigquery.LoadJobConfig(write_disposition="WRITE_TRUNCATE")
    table_id = f"{params['bq']['bq_dataset']}.{table_name}"

    job = bq_client.load_table_from_dataframe(df, table_id, job_config=job_config)
    job.result()

    logger.info("Uploaded to %s", table_id)


X = raw_df.drop(columns=["Obesity"])
y = raw_df["Obesity"]

logger.info(
    "Dataset: %d rows, %d features, target=%s", len(raw_df), X.shape[1], "Obesity"
)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=params["split"]["test_size"],
    random_state=params["split"]["random_state"],
    stratify=y if params["split"]["stratify"] else None,
)

train_df = pd.concat([X_train, y_train], axis=1)
test_df = pd.concat([X_test, y_test], axis=1)

logger.info("Train: %s | Test: %s", train_df.shape, test_df.shape)

logger.info("Uploading datasets to BigQuery...")

upload_to_bq(train_df, params["bq"]["train_table"])
upload_to_bq(test_df, params["bq"]["test_table"])

logger.info("Preprocessing completed successfully.")

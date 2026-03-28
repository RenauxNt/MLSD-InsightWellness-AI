import pandas as pd
import logging
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)


def preprocess_data(raw_df, test_size, random_state, stratify):
    raw_df = raw_df.drop(columns=["Unnamed: 0"], errors="ignore")
    raw_df = raw_df.drop(columns=["Weight", "Height"], errors="ignore")

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
            raw_df[col] = raw_df[col].astype("string").str.strip().str.lower()

    caec_calc_mapping = {"no": 0, "sometimes": 1, "frequently": 2, "always": 3}
    for col in ["CAEC", "CALC"]:
        unmapped = set(raw_df[col].dropna().unique()) - set(caec_calc_mapping.keys())
        if unmapped:
            logger.warning("Unmapped values in %s: %s", col, unmapped)
        raw_df[col] = raw_df[col].map(caec_calc_mapping).astype("Int64")

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

    gender_mapping = {"male": 0, "female": 1}
    raw_df["Gender"] = raw_df["Gender"].map(gender_mapping).astype("Int64")

    binary_mapping = {"true": 1, "false": 0}
    for col in ["family_history_with_overweight", "FAVC", "SMOKE", "SCC"]:
        raw_df[col] = raw_df[col].map(binary_mapping).astype("Int64")

    raw_df = pd.get_dummies(raw_df, columns=["MTRANS"], drop_first=False, dtype=int)
    raw_df.drop(columns=["MTRANS_public_transportation"], inplace=True)

    logger.info(
        "Preprocessed dataset: %d rows, %d columns", raw_df.shape[0], raw_df.shape[1]
    )

    logger.info("Splitting dataset...")

    X = raw_df.drop(columns=["Obesity"])
    y = raw_df["Obesity"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y if stratify else None,
    )

    train_df = pd.concat([X_train, y_train], axis=1)
    test_df = pd.concat([X_test, y_test], axis=1)

    logger.info("Train: %s | Test: %s", train_df.shape, test_df.shape)
    logger.info("Preprocessing completed successfully.")

    return train_df, test_df

"""Unit tests for the preprocessing step shared by the Vertex pipeline.

Runs preprocess_data on a synthetic raw dataframe shaped like the raw
obesity CSV and checks the encodings, the dropped columns, the split, and
the contract with the API's expected input schema.
"""

import numpy as np
import pandas as pd
import pytest

from insightwellness_ai.api.app import EXPECTED_MODEL_SCHEMA
from insightwellness_ai.preprocess import preprocess_data

RAW_CLASSES = [
    "Insufficient_Weight",
    "Normal_Weight",
    "Overweight_Level_I",
    "Overweight_Level_II",
    "Obesity_Type_I",
    "Obesity_Type_II",
    "Obesity_Type_III",
]

MTRANS_VALUES = [
    "Public_Transportation",
    "Automobile",
    "Motorbike",
    "Bike",
    "Walking",
]

ROWS_PER_CLASS = 10


@pytest.fixture(scope="module")
def raw_df():
    rng = np.random.default_rng(0)
    rows = []
    i = 0
    for label in RAW_CLASSES:
        for _ in range(ROWS_PER_CLASS):
            rows.append(
                {
                    "Unnamed: 0": i,
                    # Mixed case and stray whitespace on purpose: the raw
                    # CSV is dirty and preprocessing must normalize it.
                    "Gender": " Male " if i % 2 == 0 else "female",
                    "Age": float(rng.uniform(15, 70)),
                    "Height": 1.70,
                    "Weight": 70.0,
                    "family_history_with_overweight": "True" if i % 2 else "False",
                    "FAVC": "False",
                    "FCVC": int(rng.integers(1, 4)),
                    "NCP": 3,
                    "CAEC": "Sometimes",
                    "SMOKE": "False",
                    "CH2O": 2,
                    "SCC": "False",
                    "FAF": 1,
                    "TUE": 1,
                    "CALC": "no",
                    "MTRANS": MTRANS_VALUES[i % len(MTRANS_VALUES)],
                    "NObeyesdad": label,
                }
            )
            i += 1
    return pd.DataFrame(rows)


@pytest.fixture(scope="module")
def split(raw_df):
    return preprocess_data(raw_df.copy(), test_size=0.2, random_state=42, stratify=True)


def test_drops_index_and_leakage_columns(split):
    train_df, _ = split
    # Weight and Height directly determine BMI/obesity: leakage if kept.
    assert not {"Unnamed: 0", "Weight", "Height"} & set(train_df.columns)


def test_target_is_encoded(split):
    train_df, test_df = split
    full = pd.concat([train_df, test_df])
    assert "NObeyesdad" not in full.columns
    assert set(full["Obesity"].unique()) == set(range(7))


def test_categorical_encodings_and_normalization(split):
    train_df, test_df = split
    full = pd.concat([train_df, test_df])
    # " Male "/"female" mixed case must map cleanly, leaving no NaN behind
    assert not full["Gender"].isna().any()
    assert set(full["Gender"].unique()) <= {0, 1}
    assert set(full["family_history_with_overweight"].unique()) <= {0, 1}
    assert set(full["CAEC"].unique()) == {1}  # every row was "Sometimes"
    assert set(full["CALC"].unique()) == {0}  # every row was "no"


def test_mtrans_one_hot_encoding(split):
    train_df, test_df = split
    full = pd.concat([train_df, test_df])
    dummies = ["MTRANS_automobile", "MTRANS_motorbike", "MTRANS_bike", "MTRANS_walking"]
    assert set(dummies) <= set(full.columns)
    # public transportation is the dropped reference category
    assert "MTRANS" not in full.columns
    assert "MTRANS_public_transportation" not in full.columns
    # a row has at most one main transport mode
    assert full[dummies].sum(axis=1).isin([0, 1]).all()


def test_split_sizes_and_stratification(split):
    train_df, test_df = split
    total = len(RAW_CLASSES) * ROWS_PER_CLASS
    assert len(train_df) + len(test_df) == total
    assert len(test_df) == int(total * 0.2)
    # stratify=True: every class keeps its proportion in the test split
    assert (test_df["Obesity"].value_counts() == 2).all()


def test_output_schema_matches_api_contract(split):
    """The pipeline's output columns must be exactly what the API expects."""
    train_df, _ = split
    assert set(train_df.columns) == set(EXPECTED_MODEL_SCHEMA) | {"Obesity"}

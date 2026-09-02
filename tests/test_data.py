"""Data and configuration contract tests.

Validates the artifacts the running system depends on: the RAG knowledge
base documents, params.yaml, config sanity, and the class-mapping contract
between the pipeline and the API.
"""

from pathlib import Path

import yaml

from insightwellness_ai import config
from insightwellness_ai.api.schema import CLASS_MAPPING

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_knowledge_base_documents_exist():
    """/chat's RAG agent indexes data/*.md at startup; an empty or missing
    knowledge base would silently degrade every answer."""
    md_files = sorted((REPO_ROOT / "data").glob("*.md"))
    assert md_files, "no markdown documents found in data/"
    for path in md_files:
        content = path.read_text(encoding="utf-8")
        chunks = [c.strip() for c in content.split("\n\n") if c.strip()]
        assert chunks, f"{path.name} produces no indexable chunks"


def test_params_yaml_matches_config():
    """params.yaml mirrors config.py for the notebooks; pin the shared values."""
    params = yaml.safe_load((REPO_ROOT / "params.yaml").read_text(encoding="utf-8"))
    assert isinstance(params, dict) and params

    assert params["split"]["target"] == config.TARGET
    assert params["split"]["test_size"] == config.TEST_SIZE
    assert params["split"]["random_state"] == config.RANDOM_STATE
    assert params["train"]["n_estimators"] == config.TRAIN_CONFIG["n_estimators"]
    assert params["train"]["random_state"] == config.TRAIN_CONFIG["random_state"]
    assert params["train"]["param_grid"] == config.TRAIN_CONFIG["param_grid"]


def test_config_values_are_sane():
    assert 0 < config.TEST_SIZE < 1
    assert isinstance(config.RANDOM_STATE, int)
    assert config.TARGET == "Obesity"
    assert config.RAW_DATA.startswith("gs://")


def test_class_mapping_is_complete():
    """The API translates model output codes 0-6; all 7 classes must exist."""
    assert sorted(CLASS_MAPPING.keys()) == list(range(7))
    assert len(set(CLASS_MAPPING.values())) == 7


def test_dashboard_constants_match_api_schema():
    """The dashboard container has no installed package, so it carries
    copies of the API constants — this test keeps them honest."""
    from insightwellness_ai.api import schema
    from insightwellness_ai.dashboard import streamlit_app

    assert streamlit_app.CLASS_MAPPING == schema.CLASS_MAPPING
    assert streamlit_app.FEATURE_ORDER == list(schema.EXPECTED_MODEL_SCHEMA)
    assert streamlit_app.MTRANS_FEATURES == schema.MTRANS_FEATURES
    for feature, spec in schema.EXPECTED_MODEL_SCHEMA.items():
        if isinstance(spec["valid_values"], list):
            assert (
                list(streamlit_app.CATEGORICAL_OPTIONS[feature]) == spec["valid_values"]
            ), f"dashboard options for {feature} drifted from the API schema"

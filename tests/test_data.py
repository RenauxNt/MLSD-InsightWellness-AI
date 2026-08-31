"""Data and configuration contract tests.

Validates the artifacts the running system depends on: the RAG knowledge
base documents, params.yaml, config sanity, and the class-mapping contract
between the pipeline and the API.
"""

from pathlib import Path

import yaml

from insightwellness_ai import config
from insightwellness_ai.api.app import CLASS_MAPPING

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


def test_params_yaml_parses():
    params = yaml.safe_load((REPO_ROOT / "params.yaml").read_text(encoding="utf-8"))
    assert isinstance(params, dict) and params


def test_config_values_are_sane():
    assert 0 < config.TEST_SIZE < 1
    assert isinstance(config.RANDOM_STATE, int)
    assert config.TARGET == "Obesity"
    assert config.RAW_DATA.startswith("gs://")


def test_class_mapping_is_complete():
    """The API translates model output codes 0-6; all 7 classes must exist."""
    assert sorted(CLASS_MAPPING.keys()) == list(range(7))
    assert len(set(CLASS_MAPPING.values())) == 7

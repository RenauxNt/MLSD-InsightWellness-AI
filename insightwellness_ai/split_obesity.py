import logging
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
import yaml

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
cfgd, cfgs = params["data"], params["split"]

# ---------------------------------------------------------------------------
# Load preprocessed dataset
# ---------------------------------------------------------------------------
logger.info("Loading preprocessed data from %s", cfgd["preprocessed"])
df = pd.read_csv(cfgd["preprocessed"])

target = cfgs["target"]
y = df[target]
X = df.drop(columns=[target])

logger.info("Dataset: %d rows, %d features, target=%s", len(df), X.shape[1], target)

# ---------------------------------------------------------------------------
# First: test split
# ---------------------------------------------------------------------------
X_trv, X_test, y_trv, y_test = train_test_split(
    X,
    y,
    test_size=cfgs["test_size"],
    random_state=cfgs["random_state"],
    stratify=y if cfgs["stratify"] else None,
)

# ---------------------------------------------------------------------------
# Then: validation split from remaining
# ---------------------------------------------------------------------------
val_fraction = cfgs["validation_size"] / (1 - cfgs["test_size"])
X_train, X_val, y_train, y_val = train_test_split(
    X_trv,
    y_trv,
    test_size=val_fraction,
    random_state=cfgs["random_state"],
    stratify=y_trv if cfgs["stratify"] else None,
)

logger.info("Train: %d | Validation: %d | Test: %d", len(X_train), len(X_val), len(X_test))

# ---------------------------------------------------------------------------
# Save outputs (works with both local paths and gs:// paths)
# ---------------------------------------------------------------------------
# Create local dir only if paths are local (not GCS)
if not cfgd["train"].startswith("gs://"):
    Path(cfgd["train"]).parent.mkdir(parents=True, exist_ok=True)

pd.concat([X_train, y_train], axis=1).to_csv(cfgd["train"], index=False)
pd.concat([X_val, y_val], axis=1).to_csv(cfgd["validation"], index=False)
pd.concat([X_test, y_test], axis=1).to_csv(cfgd["test"], index=False)

logger.info("Train      -> %s", cfgd["train"])
logger.info("Validation -> %s", cfgd["validation"])
logger.info("Test       -> %s", cfgd["test"])
logger.info("Done.")

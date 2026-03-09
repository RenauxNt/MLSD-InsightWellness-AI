# InsightWellness AI

InsightWellness AI is a project developed for the INFO9023 Machine Learning Systems Design course at ULiège, designed to put MLOps concepts into practice. It combines a Random Forest model that predicts a user's obesity level from health and lifestyle data. A REST API to serve the model is planned as the next step.

## Project Structure

```
InsightWellness-AI/
├── insightwellness_ai/         # Main package
│   ├── data/
│   │   ├── preprocess.py       # Data cleaning and feature engineering
│   │   └── split_obesity.py    # Train/test split
│   └── train_obesity.py        # Model training and evaluation
├── models/                     # Saved model artifacts (joblib, feature/class metadata)
├── notebooks/
│   └── exploratory_data_analysis.ipynb
├── reports/                    # Metrics, classification reports, confusion matrices
│   ├── metrics.json
│   ├── test_classification_report.json
│   └── figures/
├── docs/                       # Component documentation
│   ├── EXPERIMENTATION.md      # Model experimentation and results
│   └── exploratory_data_analysis_documentation.md
├── tests/                      # Unit tests
├── .github/workflows/ci.yml    # CI/CD pipeline (pre-commit + pytest)
├── .pre-commit-config.yaml
├── params.yaml                 # Centralised config (data paths, hyperparameters)
└── pyproject.toml
```

## Documentation

- [Dataset Card](docs/DATASET_CARD.md) — Dataset description, sources, bias and limitations
- [EDA](docs/exploratory_data_analysis_documentation.md) — Dataset analysis and feature descriptions
- [Experimentation](docs/EXPERIMENTATION.md) — Model choice, training results and feature importances

## Data

Data is stored on Google Cloud Storage and referenced via `params.yaml`. The training script reads directly from GCS, so no local data files are required.

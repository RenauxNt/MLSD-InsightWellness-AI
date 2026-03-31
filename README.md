# InsightWellness AI

InsightWellness AI is a project developed for the INFO9023 Machine Learning Systems Design course at ULiège, designed to put MLOps concepts into practice. It combines a Random Forest model that predicts a user's obesity level from lifestyle data.

## Project Structure

```
InsightWellness-AI/
├── insightwellness_ai/                 # Main package
│   └── api/                            # API
├── vertex/                             # Vertex AI
├── notebooks/
│   ├── models_experimentation.ipynb    # Exploration/testing different tree-based models and hyperparameter tuning.
│   └── exploratory_data_analysis.ipynb
├── docs/                               # Documentation
│   ├── models_experimentation.md       # Documentation of the testing and exploration of different tree-based models based on models_experimentation.ipynb
│   └── dataset_card.md                 # Documentation based on exploratory_data_analysis.ipynb
├── tests/                              # Unit tests
├── milestones/                         # Slides for milestone presentation
├── .github/workflows/ci.yml            # CI/CD pipeline
├── .pre-commit-config.yaml
├── Dockerfile                          # Dockerfile for API
├── Dockerfile.vertex                   # Dockerfile for Vertex AI pipeline
├── params.yaml                         # Centralised config (data paths, hyperparameters)
├── uv.lock
└── pyproject.toml
```

## Vertex AI pipeline

- Data ingestion: Collect data from BigQuery​
- Preprocess: Cleaning, feature engineering
and train/test split. ​
- Model selection: Hyperparameter tuning.​
- Training: Train on train data.​
- Evaluation: Produces performance metrics using test
data.


# InsightWellness AI

InsightWellness AI is a project developed for the INFO9023 Machine Learning Systems Design course at ULiège, designed to put MLOps concepts into practice. It combines a Random Forest model that predicts a user's obesity level from lifestyle data. A REST API to serve the model is planned as the next step.

## Project Structure

```
InsightWellness-AI/
├── insightwellness_ai/                 # Main package
│   ├── data/
│   │   └── preprocess.py               # Data cleaning, feature engineering and train/test split.
│   └── train.py                        # Model training
├── notebooks/
│   ├── models_experimentation.ipynb    # Exploration/testing different tree-based models and hyperparameter tuning.
│   └── exploratory_data_analysis.ipynb
├── docs/                               # Documentation
│   ├── models_experimentation.md       # Documentation of the testing and exploration of different tree-based models based on models_experimentation.ipynb
│   └── exploratory_data_analysis_documentation.md # Documentation based on exploratory_data_analysis.ipynb
├── tests/                              # Unit tests
├── milestones/                         # Slides for milestone presentation
├── .github/workflows/ci.yml            # CI/CD pipeline
├── .pre-commit-config.yaml
├── params.yaml                         # Centralised config (data paths, hyperparameters)
├── uv.lock
└── pyproject.toml
```

## Data storage

Raw data is stored on Google Cloud Storage. Training and testing data on Big Query. Everything is referenced via `params.yaml`. No local data files are required.

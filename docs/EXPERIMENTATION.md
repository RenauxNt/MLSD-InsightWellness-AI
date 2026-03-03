# Model Experimentation

## 1. Task Definition

Multi-class classification of obesity levels (7 classes) based on individuals' eating habits and physical condition.
Dataset: 2,111 records, 16 features, 0 missing values. ~77% synthetically generated via SMOTE.

Target variable: `Obesity` (7 classes: Insufficient_Weight, Normal_Weight, Overweight_Level_I, Overweight_Level_II, Obesity_Type_I, Obesity_Type_II, Obesity_Type_III)

---

## 2. Data Split

| Set   | Size | Strategy |
|-------|------|----------|
| Train | 80%  | Stratified split (random_state=42) |
| Test  | 20%  | Stratified split (random_state=42) |

Stratification was used to preserve class balance across splits.
Data is stored and read from Google Cloud Storage (`gs://mlops-2026-ramzan1/`).

---

## 3. Model Choice

**Selected model: RandomForestClassifier**

Random Forest was chosen for the following reasons:
- Handles mixed feature types (numerical + categorical) naturally after encoding
- Robust to outliers and noisy features
- Provides feature importance scores out of the box
- Little hyperparameter sensitivity compared to boosting methods
- Fast to train on a dataset of this size

### Hyperparameters

| Parameter     | Value |
|---------------|-------|
| n_estimators  | 100   |
| max_depth     | 15    |
| random_state  | 42    |

> **Note:** A comparison against additional models (e.g. Logistic Regression, Gradient Boosting) is planned to further justify this choice.

---

## 4. Results

### Overall Metrics (Test Set)

| Metric              | Value  |
|---------------------|--------|
| Accuracy            | 93.62% |
| F1-score (macro)    | 93.53% |
| Precision (macro)   | 93.99% |
| Recall (macro)      | 93.45% |

### Cross-Validation (5-fold, on training set)

| Metric            | Value           |
|-------------------|-----------------|
| F1-macro mean     | 94.26%          |
| F1-macro std      | ± 0.73%         |

The low standard deviation across folds confirms the model is stable and not overfitting to a particular split.

### Training Time

| Metric         | Value  |
|----------------|--------|
| Training time  | 0.37 s |

### Per-Class Performance (Test Set)

| Class               | Precision | Recall | F1-score | Support |
|---------------------|-----------|--------|----------|---------|
| Insufficient_Weight | 100.00%   | 92.59% | 96.15%   | 54      |
| Normal_Weight       | 78.87%    | 96.55% | 86.82%   | 58      |
| Overweight_Level_I  | 90.38%    | 81.03% | 85.45%   | 58      |
| Overweight_Level_II | 96.43%    | 93.10% | 94.74%   | 58      |
| Obesity_Type_I      | 97.10%    | 95.71% | 96.40%   | 70      |
| Obesity_Type_II     | 96.67%    | 96.67% | 96.67%   | 60      |
| Obesity_Type_III    | 98.46%    | 98.46% | 98.46%   | 65      |

The weakest class is **Normal_Weight** (F1: 86.82%), which is expected given the overlap with adjacent weight categories (Insufficient_Weight and Overweight_Level_I). All other classes exceed 94% F1.

---

## 5. Feature Importances

| Rank | Feature                        | Importance |
|------|--------------------------------|------------|
| 1    | Weight                         | 31.85%     |
| 2    | FCVC (vegetable consumption)   | 10.22%     |
| 3    | Height                         | 9.63%      |
| 4    | Age                            | 9.45%      |
| 5    | NCP (number of main meals)     | 5.23%      |
| 6    | TUE (technology use time)      | 4.95%      |
| 7    | Gender                         | 4.93%      |
| 8    | FAF (physical activity freq.)  | 4.67%      |
| 9    | CH2O (water intake)            | 4.66%      |
| 10   | CAEC (food between meals)      | 3.68%      |

Weight is by far the most predictive feature (31.85%), which is expected for an obesity classification task. Lifestyle features (vegetable consumption, physical activity, water intake) collectively contribute significantly.

---

## 6. Carbon Emissions

Training emissions were tracked using [CodeCarbon](https://codecarbon.io/). Results are stored in `reports/emissions.csv`.

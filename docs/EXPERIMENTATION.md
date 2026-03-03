# Model Experimentation

## 1. Task Definition

Multi-class classification of obesity levels (7 classes) based on individuals' eating habits and physical condition.
Dataset: 2,111 records, 16 features, 0 missing values. ~77% synthetically generated via SMOTE.

Target variable: `Obesity` (7 classes: Insufficient_Weight, Normal_Weight, Overweight_Level_I, Overweight_Level_II, Obesity_Type_I, Obesity_Type_II, Obesity_Type_III)

---

## 2. Data Split

| Set   | Size | Strategy |
|-------|------|----------|
| Train | 80%  | Stratified split (random_state=257) |
| Test  | 20%  | Stratified split (random_state=257) |

Stratification was used to preserve class balance across splits.
Data is stored and read from Google Cloud Storage (`gs://mlops-2026-ramzan1/`).

---

## 3. Model Comparison

Three models were evaluated to select the best approach. Each model was trained with 5-fold cross-validation on the training set. The comparison was run with two different random states (42 and 257) to ensure the results are consistent and not dependent on a particular seed.

### Results (random_state=42)

| Model             | Accuracy | F1-macro | CV F1 mean | CV F1 std |
|-------------------|----------|----------|------------|-----------|
| Decision Tree     | 92.43%   | 92.23%   | 91.81%     | ±1.32%    |
| Random Forest     | 93.62%   | 93.53%   | 94.26%     | ±0.73%    |
| Gradient Boosting | **95.98%** | **95.84%** | **96.07%** | ±1.36% |

### Results (random_state=257)

| Model             | Accuracy | F1-macro | CV F1 mean | CV F1 std |
|-------------------|----------|----------|------------|-----------|
| Decision Tree     | 92.20%   | 91.91%   | 91.12%     | ±0.85%    |
| Random Forest     | 94.56%   | 94.49%   | 94.33%     | ±0.43%    |
| Gradient Boosting | **96.45%** | **96.33%** | **96.18%** | ±0.93% |

**Gradient Boosting consistently outperforms the other two models across both random states**, with a ~2% improvement in F1-macro over Random Forest and ~4% over Decision Tree.

---

## 4. Final Model: GradientBoostingClassifier

Gradient Boosting was selected as the final model. Unlike Random Forest (bagging), Gradient Boosting builds trees sequentially, each correcting the errors of the previous one, which leads to better performance on structured tabular data.

### Hyperparameter Tuning (GridSearchCV)

A grid search with 5-fold cross-validation was run over the following grid:

| Parameter      | Values tested  |
|----------------|----------------|
| n_estimators   | 500 (fixed)    |
| max_depth      | [3, 5, 7]      |
| learning_rate  | [0.05, 0.1, 0.2] |

**Best parameters found:**

| Parameter     | Value |
|---------------|-------|
| n_estimators  | 500   |
| max_depth     | 5     |
| learning_rate | 0.05  |

---

## 5. Final Model Results

### Overall Metrics (Test Set)

| Metric              | Value  |
|---------------------|--------|
| Accuracy            | 95.51% |
| F1-score (macro)    | 95.38% |
| Precision (macro)   | 95.70% |
| Recall (macro)      | 95.33% |

### Cross-Validation (5-fold, on training set)

| Metric            | Value      |
|-------------------|------------|
| F1-macro mean     | 96.73%     |
| F1-macro std      | ±0.70%     |

The low standard deviation confirms the model generalises well across different data splits.

### Training Time

| Metric         | Value   |
|----------------|---------|
| Training time  | 14.45 s |

---

## 6. Feature Importances

| Rank | Feature                        | Importance |
|------|--------------------------------|------------|
| 1    | Weight                         | 51.94%     |
| 2    | Height                         | 17.04%     |
| 3    | FCVC (vegetable consumption)   | 9.09%      |
| 4    | Gender                         | 7.20%      |
| 5    | Age                            | 3.81%      |
| 6    | CALC (alcohol consumption)     | 2.66%      |
| 7    | CH2O (water intake)            | 2.32%      |
| 8    | CAEC (food between meals)      | 1.15%      |
| 9    | NCP (number of main meals)     | 1.11%      |
| 10   | FAF (physical activity freq.)  | 1.08%      |

Weight and Height together account for ~69% of the model's decisions, which is expected for obesity classification. Lifestyle features contribute the remaining importance.

---

## 7. Carbon Emissions

Training emissions were tracked using [CodeCarbon](https://codecarbon.io/). Results are stored in `reports/emissions.csv`.

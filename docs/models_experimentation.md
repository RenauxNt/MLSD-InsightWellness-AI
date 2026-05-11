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
Data is stored and read from Big Query.

---

## 3. Model Comparison

Three models were evaluated to select the best approach. Each model was trained with 5-fold cross-validation on the training set. The comparison was run with two different random states (42 and 257) to ensure the results are consistent and not dependent on a particular seed.

### Results (random_state=42)

| Model             | Accuracy | F1-macro | CV F1 mean | CV F1 std |
|-------------------|----------|----------|------------|-----------|
| Decision Tree     | 76.83%   | 76.51%   | 68.73%     | ±5.45%    |
| Random Forest     | 68.79%   | 66.78%   | 63.73%     | ±4.46%    |
| Gradient Boosting | **81.56%** | **81.50%** | **76.88%** | ±5.57% |

### Results (random_state=257)

| Model             | Accuracy | F1-macro | CV F1 mean | CV F1 std |
|-------------------|----------|----------|------------|-----------|
| Decision Tree     | 77.07%   | 76.62%   | 68.54%     | ±4.76%    |
| Random Forest     | 68.32%   | 66.05%   | 63.48%     | ±3.94%    |
| Gradient Boosting | **81.80%** | **81.74%** | **77.64%** | ±5.45% |

**Gradient Boosting consistently outperforms the other two models across both random states**.

---

## 4. Final Model: HistGradientBoostingClassifier

Gradient boosting was selected over Random Forest and Decision Tree based on the benchmark above — sequential trees that correct the errors of the previous one outperform bagging and single-tree baselines on this kind of structured tabular data.

For the production pipeline we then switched from the standard `GradientBoostingClassifier` to `HistGradientBoostingClassifier`. The deal-maker was **native handling of categorical features**: our lifestyle data is heavily categorical (transportation mode, food between meals, family history, gender, snacking habits…), and HGB processes them directly through its `categorical_features` argument instead of forcing one-hot expansion. As a side benefit, histogram-based binning makes training noticeably faster.

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
| max_depth     | 7     |
| learning_rate | 0.02  |

---

## 5. Final Model Results

### Overall Metrics (Test Set)

| Metric              | Value  |
|---------------------|--------|
| Accuracy            | 82.03% |
| F1-score (macro)    | 81.80% |
| Precision (macro)   | 81.96% |
| Recall (macro)      | 81.85% |

### Cross-Validation (5-fold, on training set)

| Metric            | Value   |
|-------------------|---------|
| F1-macro mean     | 81.71%  |
| F1-macro std      | ±2.31%  |

---

## 6. Feature Importances

| Rank | Feature                              | Importance |
|------|--------------------------------------|------------|
| 1    | Age                                  | 17.71%     |
| 2    | FCVC (vegetable consumption)         | 15.32%     |
| 3    | TUE (technology use)                 | 13.11%     |
| 4    | NCP (number of main meals)           | 8.74%      |
| 5    | FAF (physical activity frequency)    | 8.53%      |
| 6    | CH2O (water intake)                  | 7.75%      |
| 7    | Gender                               | 7.36%      |
| 8    | CAEC (food between meals)            | 6.45%      |
| 9    | family_history_with_overweight       | 4.67%      |
| 10   | CALC (alcohol consumption)           | 4.08%      |
| 11   | FAVC (high-calorie food consumption) | 2.55%      |
| 12   | MTRANS_automobile                    | 1.45%      |
| 13   | SCC (calorie monitoring)             | 0.84%      |
| 14   | MTRANS_walking                       | 0.69%      |
| 15   | SMOKE                                | 0.49%      |
| 16   | MTRANS_bike                          | 0.15%      |
| 17   | MTRANS_motorbike                     | 0.11%      |

## 7. Permutation Feature Importances

| Rank | Feature                              | Importance |
|------|--------------------------------------|------------|
| 1    | Age                                  | 23.33%     |
| 2    | FCVC (vegetable consumption)         | 18.37%     |
| 3    | Gender                               | 17.54%     |
| 4    | TUE (technology use)                 | 12.67%     |
| 5    | NCP (number of main meals)           | 10.09%     |
| 6    | FAF (physical activity frequency)    | 6.88%      |
| 7    | family_history_with_overweight       | 6.62%      |
| 8    | CH2O (water intake)                  | 5.98%      |
| 9    | CALC (alcohol consumption)           | 4.82%      |
| 10   | CAEC (food between meals)            | 3.22%      |
| 11   | FAVC (high-calorie food consumption) | 2.36%      |
| 12   | MTRANS_automobile                    | 2.29%      |
| 13   | SCC (calorie monitoring)             | 1.04%      |
| 14   | MTRANS_motorbike                     | 0.24%      |
| 15   | SMOKE                                | 0.07%      |
| 16   | MTRANS_walking                       | 0.02%      |
| 17   | MTRANS_bike                          | 0.00%      |

Compared to the tree-based feature importance gender is much higher now. Overall age, vegetable consumption, technology use, gender and number of main meals are the most impactful features. After that, physical activity frequency, family history with overweight, food between meals and water intake have all importance above 5%. The rest all have importance below 5%.

---

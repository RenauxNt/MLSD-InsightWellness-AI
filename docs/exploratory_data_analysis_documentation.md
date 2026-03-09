# Exploratory Data Analysis

> For full dataset description, sources, and limitations see [DATASET_CARD.md](DATASET_CARD.md).

## 1. Data Quality

### Missing Values

No missing values were detected across any column.

### Duplicate Records

24 duplicate rows were detected (1.14%). These were kept as they are likely a result of the SMOTE oversampling process rather than data entry errors.

### Numeric Sanity Check

All numerical values fall within plausible ranges. No negative values detected.

---

## 2. Variable Description

| Variable | Type | Description |
|----------|------|-------------|
| Gender | Categorical (binary) | Male / Female |
| Age | Continuous | Age in years |
| Height | Continuous | Height in meters |
| Weight | Continuous | Weight in kilograms |
| family_history_with_overweight | Binary | Family member with overweight history |
| FAVC | Binary | Frequent consumption of high-caloric food |
| FCVC | Integer | Frequency of vegetable consumption (1–3) |
| NCP | Continuous | Number of main meals per day |
| CAEC | Ordinal | Food consumption between meals (No / Sometimes / Frequently / Always) |
| SMOKE | Binary | Smoker |
| CH2O | Continuous | Daily water intake (1–3) |
| SCC | Binary | Monitors daily calorie intake |
| FAF | Continuous | Physical activity frequency (0–3) |
| TUE | Integer | Daily technology use in hours (0–2) |
| CALC | Ordinal | Alcohol consumption frequency (No / Sometimes / Frequently / Always) |
| MTRANS | Categorical | Transportation method |
| NObeyesdad | Target | Obesity level (7 classes) |

---

## 3. Target Variable

The dataset is approximately balanced across the 7 classes, which is a result of the SMOTE oversampling applied during dataset creation.

```
Insufficient_Weight    272
Normal_Weight          287
Overweight_Level_I     290
Overweight_Level_II    290
Obesity_Type_I         351
Obesity_Type_II        297
Obesity_Type_III       324
```

---

## 4. Correlation

![Correlation Matrix](../reports/figures/correlation.png)

Weight is strongly correlated with the target, as expected. Height and Age show moderate correlations. Lifestyle features (FCVC, FAF, CH2O) show weaker but meaningful correlations with obesity level.

---

## 5. Pairplot

![Pairplot](../reports/figures/pairplot.png)

The pairplot confirms that Weight and Height are the most visually separating features across obesity classes. Lifestyle features show more overlap between classes, suggesting they contribute incrementally rather than being individually decisive.

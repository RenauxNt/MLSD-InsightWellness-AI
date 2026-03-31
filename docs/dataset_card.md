# Exploratory Data Analysis

## 1. Dataset card

### Overview

This dataset contains survey data collected from individuals in Mexico, Peru, and Colombia to estimate obesity levels based on eating habits and physical condition. It includes 2,111 rows and 17 columns with demographic, dietary, and lifestyle features, along with a categorical target variable for obesity levels.

You can see more details of the EDA and more plots in notebooks/exploratory_data_analysis.ipynb.

### Dataset Details

This dataset was introduced by **Palechor & de la Hoz Manotas (2019)** and published in *Data in Brief*.
It contains information collected from individuals in **Colombia, Peru, and Mexico** (ages 14–61) to estimate obesity levels based on eating habits and physical condition.

- **Curated by:** Fabio Mendoza Palechor & Alexis de la Hoz Manotas (Universidad de la Costa, Colombia)
- **License:** Creative Commons Attribution 4.0 International (CC BY 4.0)
- **Size:** 2,111 records, 17 attributes
- **Shared by:** UCI Machine Learning Repository
- **Language(s):** Original survey in Spanish; released dataset provided in English.

**Target Variable:** `NObeyesdad`

Classes:

- Insufficient Weight
- Normal Weight
- Overweight Level I
- Overweight Level II
- Obesity Type I
- Obesity Type II
- Obesity Type III

### Dataset Sources

- **Repository:** [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/544/estimation+of+obesity+levels+based+on+eating+habits+and+physical+condition)
- **Paper:** [Palechor & de la Hoz Manotas, 2019](https://doi.org/10.1016/j.dib.2019.104344)

---

## 2. Dataset Structure

- **Rows:** 2,111
- **Columns:** 17
- **Target Variable:** `NObeyesdad`

### Class Distribution

```
Insufficient_Weight    272
Normal_Weight          287
Overweight_Level_I     290
Overweight_Level_II    290
Obesity_Type_I         351
Obesity_Type_II        297
Obesity_Type_III       324
```

### Milestone 1 feedback.

The model relied almost entirely on weight (See milestone1.pdf in milestone/). The model's utility was unclear because a simple BMI rule could categorize a person. We took that feedback into account and decided to remove the weight and the height features. The project switch to an early warning system for obesity based only on lifestyle. The BMI rule only detect obesity after the weight gain has occurred. The goal of our approach is to detect persons at risk before the weight gain.

### Feature overview

| Variable | Type | Description |
|----------|------|-------------|
| Gender | Categorical (binary) | Male / Female |
| Age | Continuous | Age in years |
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

## 3. Data Quality

### Missing Values

No missing values were detected across any column.

### Duplicate Records

24 duplicate rows were detected (1.14%). These were kept as they are likely a result of the SMOTE oversampling process rather than data entry errors.

### Numeric Sanity Check

All numerical values fall within plausible ranges. No negative values detected.

---


## 4. Correlation

![Correlation Matrix](./figures/correlation.png)

After removing weight and height, family history of overweight (0.51) has become the predominant factor.
---

## 5. Pairplot

![Pairplot](./figures/pairplot.png)


## 6. Bias, Risks, and Limitations

- Data is **self-reported**, which may introduce inaccuracies.
- Dataset only covers **Mexico, Peru, and Colombia**, limiting geographic diversity.
- **77% of the data is synthetic (SMOTE)**.
- The dataset ignores other important determinants of obesity such as genetics or socioeconomic factors.

### Recommendations

This dataset should primarily be used for **educational or research purposes**. It is **not suitable for clinical decision-making or healthcare deployment**.

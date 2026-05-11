# Dataset Feature Glossary

The model uses 17 features after one-hot encoding of `MTRANS`. All values
are stored as numeric (integers or floats) in the preprocessed dataset and
in the API request payload. This document is the source of truth for what
each value means.

## Demographics

### Gender — binary (0 / 1)
- 0 = Male
- 1 = Female

Stored as integer in the API; the underlying survey values were "Male" /
"Female".

### Age — continuous (years)
Original survey range: 14–61 years. The Streamlit form accepts 10–100 to
allow exploration but predictions outside the training range should be
treated with caution.

## Family history

### family_history_with_overweight — binary (0 / 1)
- 0 = No family history of overweight
- 1 = At least one family member with a history of overweight

Strongest single correlation with the target in the dataset (≈0.51).

## Eating habits

### FAVC — Frequent consumption of high-caloric food — binary (0 / 1)
- 0 = No
- 1 = Yes

### FCVC — Frequency of vegetable consumption — ordinal 1–3
- 1 = Never
- 2 = Sometimes
- 3 = Always

Stored as a continuous float because of SMOTE oversampling, but the three
values above are the only meaningful buckets.

### NCP — Number of main meals per day — ordinal 1–4
- 1 = One meal
- 2 = Two meals
- 3 = Three meals (most common)
- 4 = Four or more meals

Stored as continuous due to SMOTE.

### CAEC — Food consumption between meals — ordinal 0–3
- 0 = No
- 1 = Sometimes
- 2 = Frequently
- 3 = Always

### SCC — Monitors daily calorie intake — binary (0 / 1)
- 0 = No
- 1 = Yes

## Lifestyle

### SMOKE — binary (0 / 1)
- 0 = No
- 1 = Yes

Low feature importance in the trained model.

### CH2O — Daily water intake — ordinal 1–3
- 1 = Less than 1 L
- 2 = 1–2 L
- 3 = More than 2 L

### FAF — Physical activity frequency — ordinal 0–3
- 0 = None
- 1 = 1–2 days per week
- 2 = 2–4 days per week
- 3 = 4–5 days per week

### TUE — Daily technology use — ordinal 0–2
- 0 = 0–2 hours
- 1 = 3–5 hours
- 2 = More than 5 hours

Used as a proxy for sedentary behavior.

### CALC — Alcohol consumption frequency — ordinal 0–3
- 0 = No
- 1 = Sometimes
- 2 = Frequently
- 3 = Always

## Transportation

### MTRANS_* — one-hot encoded transport method
The original `MTRANS` categorical column was expanded into four binary
columns. Exactly one of them is 1, the others are 0. If all four are 0 it
implicitly means "Public transportation" (the dropped reference category).

- `MTRANS_automobile` (0/1)
- `MTRANS_motorbike` (0/1)
- `MTRANS_bike` (0/1)
- `MTRANS_walking` (0/1)

All four MTRANS columns have low individual feature importance.

## Target variable

### Obesity (also called `NObeyesdad` in the raw dataset) — categorical (0–6)
Seven obesity classes, encoded as integers in the preprocessed table:

- 0 = Insufficient_Weight
- 1 = Normal_Weight
- 2 = Overweight_Level_I
- 3 = Overweight_Level_II
- 4 = Obesity_Type_I
- 5 = Obesity_Type_II
- 6 = Obesity_Type_III

Class definitions and BMI mapping live in `bmi_and_classes.md`.

## Excluded features

The original dataset included `Height` (m) and `Weight` (kg). Both were
intentionally removed before training. Reason: the project is positioned
as an **early-warning system based on lifestyle**, while height and weight
already determine BMI directly and would let the model "cheat" instead of
learning lifestyle signals. See `dataset_card.md` section 2 for the full
rationale.

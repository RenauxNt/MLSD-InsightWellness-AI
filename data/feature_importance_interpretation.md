# Feature Importance & SHAP Interpretation

This document explains how to read the SHAP `impact_score` returned by the
`/explain` endpoint and what direction each feature usually pushes the
prediction. It is the companion to the global feature importances reported
in `models_experimentation.md`.

## How to read a SHAP impact score

A SHAP impact score quantifies how much a single feature value moved the
model's output for the predicted class, relative to a neutral baseline.

- **Positive impact_score** → the feature value pushed the model **toward**
  the predicted class. The user "matches the profile" of that class on this
  feature.
- **Negative impact_score** → the feature value pushed the model **away
  from** the predicted class. The user does not match the profile on this
  feature, but the prediction was still made because other features
  outweighed it.
- **Magnitude** matters more than rank: a feature with `+0.40` is a much
  stronger driver than one with `+0.05`.

SHAP scores are **per-class and per-individual**. The same feature may push
toward `Obesity_Type_I` for one user and away from it for another, depending
on context.

## Top global drivers (from training)

Across the test set, the most influential features (permutation importance)
are: Age, FCVC (vegetable consumption), Gender, TUE (technology use), NCP
(meals per day), FAF (physical activity), and family_history_with_overweight.
These are the features most likely to dominate any individual SHAP
explanation.

## Direction each feature usually pushes risk

These are typical directions observed in the trained model. Individual SHAP
values can deviate when features interact.

### Age
Risk generally **increases with age** up to mid-adult years, then plateaus.
Younger users (teens, early 20s) tend to push toward `Insufficient_Weight`
or `Normal_Weight`; older users push toward overweight and obesity classes.

### FCVC — vegetable consumption (1=Never, 2=Sometimes, 3=Always)
**Higher FCVC → lower obesity risk.** A user reporting "Always" on
vegetables typically has negative SHAP for obesity classes (pushes away)
and positive SHAP for normal-weight classes.

### TUE — daily technology use (0=0-2 h, 1=3-5 h, 2=>5 h)
**Higher TUE → higher risk**, as a proxy for sedentary behavior. TUE=2 is
a strong push toward overweight/obesity classes.

### NCP — main meals per day (1–4)
Both extremes can push toward risk classes. Very low (1) is sometimes
linked to skipped meals followed by overeating; very high (4+) can indicate
calorie surplus. Mid-range (3) is the most common normal-weight profile.

### FAF — physical activity frequency (0=None, 1=1-2 days, 2=2-4 days, 3=4-5 days)
**Higher FAF → lower risk.** FAF=0 is a strong push toward obesity classes;
FAF=3 pushes toward normal weight.

### CH2O — daily water intake (1=<1 L, 2=1-2 L, 3=>2 L)
**Higher CH2O → modestly lower risk**, with smaller magnitude than FAF or
FCVC.

### Gender (0=Male, 1=Female)
Gender contributes asymmetrically across classes. The model has learned
distinct distributions per gender, so gender often appears among top
drivers even though its direction depends on the predicted class.

### CAEC — food between meals (0=No, 1=Sometimes, 2=Frequently, 3=Always)
**Higher CAEC → higher risk** for overweight/obesity classes. Frequent
between-meal eating is one of the cleaner positive risk signals in the
dataset.

### family_history_with_overweight (0=No, 1=Yes)
**Yes → higher risk.** This is the single strongest correlation with
obesity in the dataset (≈0.51), so a positive value almost always pushes
toward obesity classes.

### CALC — alcohol consumption (0=No, 1=Sometimes, 2=Frequently, 3=Always)
**Higher CALC → higher risk**, contributing extra calories. Effect is
moderate.

### FAVC — frequent high-caloric food (0=No, 1=Yes)
**Yes → higher risk**, but the model relies on it less than expected
because it correlates with other diet features (FCVC, CAEC).

### SCC — monitors daily calorie intake (0=No, 1=Yes)
**Yes → lower risk.** Self-monitoring is associated with normal-weight
profiles. Low overall importance in the model.

### MTRANS — transportation method (one-hot: automobile / motorbike / bike / walking)
Active transport (walking, bike) pushes toward lower-risk classes;
automobile and motorbike push slightly toward higher-risk classes. All
MTRANS features have low individual importance in the model.

### SMOKE (0=No, 1=Yes)
Very low importance in this model. SHAP values are usually near zero and
should not be over-interpreted.

## When SHAP and the RAG agent disagree

If the `/explain` output names a feature as a strong driver but the RAG
agent's general advice points elsewhere, **trust SHAP for what the model
actually used**, and treat the RAG output as context for *why* that feature
matters in obesity research. The model's decision is grounded in the
training data; the knowledge base provides the human-readable rationale.

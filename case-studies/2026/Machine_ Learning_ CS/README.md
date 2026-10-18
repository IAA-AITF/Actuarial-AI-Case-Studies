# Underwriting Rating Model – Machine Learning Case Study

## Overview

This repository contains an **underwriting machine-learning case study** built with synthetic claim and in-force/exposure data.

The study investigates whether historical disease information—starting with **respiratory disease history**—can help predict future circulatory disease claims and support underwriting risk segmentation.

The workflow begins with a simple and interpretable logistic-regression benchmark and then expands to multivariate and machine-learning models for comparison.

> **No LLM API is required.**  
> The entire workflow is designed to run locally, so the synthetic dataset and modeling pipeline can be tested and reviewed in a local Python/Jupyter environment.

---

## Main Notebook

The primary analysis is contained in:

```text
260820_UW rating Model_ML_final - 복사본.ipynb
```

The notebook is organized as a step-by-step case study and includes explanatory Markdown, model code, outputs, and evaluation plots.

---

## Study Workflow

The analysis follows the process below:

1. Load raw claim and exposure data
2. Preprocess and validate the datasets
3. Map claim records to **KCD-9** disease classifications
4. Review claim distribution by KCD chapter
5. Build a fixed **5-year historical disease window**
6. Construct **3-, 4-, and 5-year future follow-up cohorts**
7. Run a single-variable logistic-regression benchmark
8. Build additional underwriting history features
9. Compare multivariate and machine-learning models using the same Train/Test split
10. Evaluate model discrimination and probability quality
11. Review calibration
12. Provide frameworks for SHAP interpretation and Out-of-Time validation

---

## ICD and KCD

The case study uses **KCD-9 (Korean Standard Classification of Diseases)** codes.

- **ICD** is the international disease classification system maintained by the World Health Organization.
- **KCD** is the Korean adaptation of the ICD framework.
- The overall classification structure is similar, although some detailed codes and subcategories may differ.
- KCD-9 is used in this project because the underlying claim data are structured using Korean disease codes.

The notebook includes English explanations and translated disease-category names to make the analysis easier to review internationally.

---

## Cohort Design

The historical disease period is fixed at **5 years**.

Example index date used in the notebook:

```text
Index date       : 2020-05-04
History period   : 2015-05-04 ~ 2020-05-03
```

Future follow-up periods are compared across:

```text
3 years
4 years
5 years
```

Observed eligible insured counts in the current analysis are approximately:

| Follow-up period | Eligible insureds |
|---|---:|
| 3 years | 13,796 |
| 4 years | 13,566 |
| 5 years | 13,310 |

---

## Future Disease Targets

The study evaluates the following future circulatory-disease outcomes:

| Target | KCD range |
|---|---|
| Hypertension | `I10-I15` |
| Ischemic heart disease | `I20-I25` |
| Pulmonary circulation disease | `I26-I28` |
| Arrhythmia | `I48-I49` |
| Heart failure | `I50` |

Each target is converted to an insured-level binary outcome for each follow-up period.

---

## Benchmark Model

The baseline model is intentionally simple:

```text
Logistic Regression
Feature: Respiratory_History
```

`Respiratory_History` indicates whether an insured had at least one respiratory disease claim (`J00-J99`) during the 5-year history period.

The benchmark produces:

- AUC
- Odds Ratio
- Predicted risk without respiratory history
- Predicted risk with respiratory history
- Model-based Relative Risk

This benchmark is retained so that more complex models can be compared against a simple and interpretable underwriting baseline.

---

## Extended Underwriting Features

The extended model creates additional insured-level features from the same 5-year historical claim window.

### Respiratory history

- `Respiratory_History`
- `Chronic_Respiratory_History` (`J40-J47`)
- `Respiratory_Claim_Count`
- `Years_Since_Last_Respiratory`

### Healthcare utilization

- `History_Claim_Count`
- `Distinct_KCD_Count`

### Prior circulatory history

- `Cardiovascular_History`
- `Hypertension_History`
- `Ischemic_Heart_History`
- `Arrhythmia_History`
- `Heart_Failure_History`

### Other medical history

- `Diabetes_History`
- `Metabolic_History`

This design allows the case study to test whether respiratory history adds predictive information beyond broader historical morbidity and healthcare utilization.

---

## Models Compared

The extended pipeline compares the following models on the **same Train/Test split**:

- `Logistic_Benchmark`
- `Logistic_Multivariate`
- `RandomForest`
- `HistGradientBoosting`
- `XGBoost` *(optional; included when installed)*

Using the same split ensures that performance differences mainly reflect model/feature differences rather than different random samples.

---

## Evaluation Metrics

The main validation metrics are:

### ROC-AUC

Measures overall discrimination between event and non-event cases.

### PR-AUC

Useful for evaluating low-frequency disease events where class imbalance is substantial.

### Brier Score

Measures the accuracy of predicted probabilities.

Lower values indicate better probability accuracy.

### Calibration

Calibration plots compare predicted probability with the observed event rate.

The notebook also includes frameworks for:

- **SHAP** model interpretation
- **Out-of-Time (OOT) validation**

`SHAP` and `XGBoost` are optional packages, and the rest of the modeling workflow can run without them.

---

## Example Results

The simple respiratory-history benchmark shows limited discrimination for several future circulatory targets.

When broader historical features are included, performance improves materially for some targets.

Selected examples from the current notebook run:

| Follow-up | Target | Best-performing model by ROC-AUC | ROC-AUC |
|---|---|---|---:|
| 3 years | Hypertension | Logistic Multivariate | ~0.81 |
| 4 years | Hypertension | Random Forest | ~0.74 |
| 5 years | Hypertension | Random Forest | ~0.74 |
| 3 years | Ischemic Heart Disease | Random Forest | ~0.79 |
| 4 years | Ischemic Heart Disease | HistGradientBoosting | ~0.80 |
| 5 years | Ischemic Heart Disease | Logistic Multivariate | ~0.86 |

These results are intended for **case-study and model-comparison purposes**, not as production underwriting rates.

---

## Local Environment

A typical local environment requires:

```text
Python
Jupyter Notebook / JupyterLab
pandas
numpy
matplotlib
scikit-learn
openpyxl
```

Optional packages:

```text
xgboost
shap
```

Example installation:

```bash
pip install pandas numpy matplotlib scikit-learn openpyxl jupyter
pip install xgboost shap
```

---

## Input Files

The notebook expects input/reference files such as:

```text
Raw_exposure_data.pickle
Raw_claim_data.pickle
KCD-9_mst.xlsx
KCD9_대분류별_분류요건.csv
```

If the synthetic data files are distributed separately because of file-size limits, place them in the working directory or update the file paths in the notebook before execution.

---

## Running the Case Study

1. Clone or download this repository.
2. Create a local Python environment.
3. Install the required packages.
4. Place the synthetic claim/exposure data and KCD reference files in the expected location.
5. Open the notebook:

```bash
jupyter notebook
```

6. Run the notebook cells sequentially from top to bottom.

The benchmark results can be exported to Excel as:

```text
Respiratory_Association_Result.xlsx
```

---

## Reports

PDF reports are provided as supporting review material and summarize the purpose, methodology, model comparison, and major results of the case study.

For detailed methodology and code execution, refer to the Jupyter Notebook.

---

## Project Scope

This repository is intended as an **internal actuarial / underwriting machine-learning case study**.

The current work is designed to demonstrate:

- cohort construction from claim and exposure data,
- KCD-based disease feature engineering,
- underwriting risk-factor exploration,
- comparison of interpretable and machine-learning models,
- relative-risk and model-performance analysis,
- and a locally executable workflow without reliance on an LLM API.

It is **not intended to represent a production underwriting or pricing model** without additional validation, governance, and business review.

---

## Review

Comments and feedback on the following areas are especially welcome:

- cohort and target definitions,
- underwriting feature design,
- KCD disease grouping,
- model comparison methodology,
- interpretation of relative risk,
- and the overall case-study structure.

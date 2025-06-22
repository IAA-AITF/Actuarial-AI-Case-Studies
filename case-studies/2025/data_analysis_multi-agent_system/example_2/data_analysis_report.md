# Introduction
This dataset, commonly referred to as the “Diabetes 130-US hospitals” dataset, comes from a large pool of patient admissions spanning the years 1999–2008 at 130 U.S. hospitals. Each record represents a hospital stay for a diabetic patient, including demographic information, diagnoses, length of stay, lab results, and medications used. This dataset is valuable for analyzing factors associated with readmissions, medication changes, and general outcomes in diabetic care.

---

## Column Descriptions
Below is an overview of each column’s meaning and context:

- **encounter_id** – Unique identifier for the encounter.  
- **patient_nbr** – Unique identifier for the patient (multiple encounters possible).  
- **race** – Patient-reported race (Caucasian, AfricanAmerican, etc.).  
- **gender** – Gender of the patient (Male, Female, or Unknown).  
- **age** – Patient’s age in 10-year intervals (e.g., [0-10), [10-20), [90-100)).  
- **weight** – Weight recorded in specific ranges (mostly missing).  
- **admission_type_id** – Numeric ID describing the admission category (e.g., emergency, urgent).  
- **discharge_disposition_id** – Numeric ID describing discharge status (home, transfer, etc.).  
- **admission_source_id** – Numeric ID describing the referral source (e.g., physician referral, emergency).  
- **time_in_hospital** – Number of days between admission and discharge (1–14).  
- **payer_code** – Payment source (e.g., Medicare, Medicaid, private insurance).  
- **medical_specialty** – Primary specialty of the attending physician (e.g., InternalMedicine).  
- **num_lab_procedures** – Number of lab tests performed during the encounter.  
- **num_procedures** – Number of non-lab procedures performed.  
- **num_medications** – Number of distinct medications administered.  
- **number_outpatient** – Count of prior outpatient visits within one year.  
- **number_emergency** – Count of prior emergency visits within one year.  
- **number_inpatient** – Count of prior inpatient visits within one year.  
- **diag_1** – Primary diagnosis (ICD-9 code).  
- **diag_2** – Secondary diagnosis (ICD-9 code).  
- **diag_3** – Tertiary diagnosis (ICD-9 code).  
- **number_diagnoses** – Number of diagnoses on record for the encounter.  
- **max_glu_serum** – Indicates glucose serum measurement range (>200, >300, Normal, or none).  
- **A1Cresult** – Shows A1C test results (>7, >8, Normal, or none).  
- **metformin, repaglinide, …, insulin** – Prescription status (No, Steady, Up, Down) of various diabetic medications.  
- **change** – Indicates whether the diabetes medication dosage was changed during the encounter.  
- **diabetesMed** – Indicates whether any diabetes medication was prescribed during the encounter.  
- **readmitted** – Readmission status: “<30” (within 30 days), “>30” (after 30 days), or “NO” (no readmission).

---

## Data Preview
Below are the first 10 rows from the dataset:

| encounter_id | patient_nbr | race             | gender | age      | weight | admission_type_id | discharge_disposition_id | admission_source_id | time_in_hospital | payer_code | medical_specialty         | num_lab_procedures | num_procedures | num_medications | number_outpatient | number_emergency | number_inpatient | diag_1   | diag_2    | diag_3   | number_diagnoses | max_glu_serum | A1Cresult | metformin | repaglinide | nateglinide | chlorpropamide | glimepiride | acetohexamide | glipizide  | glyburide | tolbutamide | pioglitazone | rosiglitazone | acarbose | miglitol | troglitazone | tolazamide | examide | citoglipton | insulin | glyburide-metformin | glipizide-metformin | glimepiride-pioglitazone | metformin-rosiglitazone | metformin-pioglitazone | change | diabetesMed | readmitted |
|--------------|-------------|------------------|--------|----------|--------|-------------------|-------------------------|---------------------|------------------|-----------|---------------------------|--------------------|---------------|-----------------|------------------|-----------------|-----------------|----------|-----------|----------|------------------|--------------|----------|-----------|------------|------------|----------------|------------|---------------|------------|-----------|-------------|-------------|---------------|---------|----------|--------------|-----------|--------|-------------|---------|---------------------|--------------------|---------------------------|-------------------------|-------------------------|--------|------------|-----------|
| 2278392      | 8222157     | Caucasian        | Female | [0-10)   | ?      | 6                 | 25                      | 1                   | 1                | ?         | Pediatrics-Endocrinology | 41                 | 0             | 1               | 0                | 0               | 0               | 250.83   | ?         | ?        | 1                |              |          | No        | No         | No         | No             | No         | No            | No         | No        | No          | No          | No            | No       | No       | No           | No        | No     | No          | No      | No                  | No                 | No                        | No                      | No                      | No     | No         | NO        |
| 149190       | 55629189    | Caucasian        | Female | [10-20)  | ?      | 1                 | 1                       | 7                   | 3                | ?         | ?                         | 59                 | 0             | 18              | 0                | 0               | 0               | 276      | 250.01    | 255      | 9                |              |          | No        | No         | No         | No             | No         | No            | No         | No        | No          | No          | No            | No       | No       | No           | No        | No     | No          | Up      | No                  | No                 | No                        | No                      | No                      | Ch     | Yes        | >30       |
| 64410        | 86047875    | AfricanAmerican  | Female | [20-30)  | ?      | 1                 | 1                       | 7                   | 2                | ?         | ?                         | 11                 | 5             | 13              | 2                | 0               | 1               | 648      | 250       | V27      | 6                |              |          | No        | No         | No         | No             | No         | No            | Steady    | No        | No          | No          | No            | No       | No       | No           | No        | No     | No          | No      | No                  | No                 | No                        | No                      | No                      | No     | Yes        | NO        |
| 500364       | 82442376    | Caucasian        | Male   | [30-40)  | ?      | 1                 | 1                       | 7                   | 2                | ?         | ?                         | 44                 | 1             | 16              | 0                | 0               | 0               | 8        | 250.43    | 403      | 7                |              |          | No        | No         | No         | No             | No         | No            | No         | No        | No          | No          | No            | No       | No       | No           | No        | No     | No          | Up      | No                  | No                 | No                        | No                      | No                      | Ch     | Yes        | NO        |
| 16680        | 42519267    | Caucasian        | Male   | [40-50)  | ?      | 1                 | 1                       | 7                   | 1                | ?         | ?                         | 51                 | 0             | 8               | 0                | 0               | 0               | 197      | 157       | 250      | 5                |              |          | No        | No         | No         | No             | No         | No            | Steady    | No        | No          | No          | No            | No       | No       | No           | No        | No     | No          | Steady  | No                  | No                 | No                        | No                      | No                      | Ch     | Yes        | NO        |
| 35754        | 82637451    | Caucasian        | Male   | [50-60)  | ?      | 2                 | 1                       | 2                   | 3                | ?         | ?                         | 31                 | 6             | 16              | 0                | 0               | 0               | 414      | 411       | 250      | 9                |              |          | No        | No         | No         | No             | No         | No            | No         | No        | No          | No          | No            | No       | No       | No           | No        | No     | No          | Steady  | No                  | No                 | No                        | No                      | No                      | No     | Yes        | >30       |
| 55842        | 84259809    | Caucasian        | Male   | [60-70)  | ?      | 3                 | 1                       | 2                   | 4                | ?         | ?                         | 70                 | 1             | 21              | 0                | 0               | 0               | 414      | 411       | V45      | 7                |              |          | Steady    | No         | No         | No             | Steady    | No            | No         | No        | No          | No          | No            | No       | No       | No           | No        | No     | No          | Steady  | No                  | No                 | No                        | No                      | No                      | Ch     | Yes        | NO        |
| 63768        | 114882984   | Caucasian        | Male   | [70-80)  | ?      | 1                 | 1                       | 7                   | 5                | ?         | ?                         | 73                 | 0             | 12              | 0                | 0               | 0               | 428      | 492       | 250      | 8                |              |          | No        | No         | No         | No             | No         | No            | No         | Steady   | No          | No          | No            | No       | No       | No           | No        | No     | No          | No      | No                  | No                 | No                        | No                      | No                      | No     | Yes        | >30       |
| 12522        | 48330783    | Caucasian        | Female | [80-90)  | ?      | 2                 | 1                       | 4                   | 13               | ?         | ?                         | 68                 | 2             | 28              | 0                | 0               | 0               | 398      | 427       | 38       | 8                |              |          | No        | No         | No         | No             | No         | No            | Steady    | No        | No          | No          | No            | No       | No       | No           | No        | No     | No          | Steady  | No                  | No                 | No                        | No                      | No                      | Ch     | Yes        | NO        |
| 15738        | 63555939    | Caucasian        | Female | [90-100) | ?      | 3                 | 3                       | 4                   | 12               | ?         | InternalMedicine          | 33                 | 3             | 18              | 0                | 0               | 0               | 434      | 198       | 486      | 8                |              |          | No        | No         | No         | No             | No         | No            | No         | No        | No          | No          | Steady        | No       | No       | No           | No        | No     | No          | Steady  | No                  | No                 | No                        | No                      | No                      | Ch     | Yes        | NO        |

*(Note: “?” indicates a missing or unknown value; a null entry under max_glu_serum or A1Cresult means not measured.)*

---

## Missing Values Summary
Below is the count of missing values among selected columns:

- **max_glu_serum**: 96,420 missing  
- **A1Cresult**: 84,748 missing  
- **weight**: 98,569 missing  

Most essential demographic and numeric columns have complete data. However, lab-related fields (max_glu_serum, A1Cresult) and weight are frequently missing, which could impact models needing these predictors.

---

## Descriptive Statistics

### Numerical Columns

Below is a summary of the numeric features:

| Column               | Count     | Mean      | Std Dev  | Min   | 25%       | 50%    | 75%       | Max   |
|----------------------|-----------|----------:|---------:|------:|----------:|-------:|----------:|------:|
| time_in_hospital     | 101,766   | 4.40      | 2.99     | 1     | 2         | 4      | 6         | 14    |
| num_lab_procedures   | 101,766   | 43.10     | 19.67    | 1     | 31        | 44     | 57        | 132   |
| num_procedures       | 101,766   | 1.34      | 1.70     | 0     | 0         | 1      | 2         | 6     |
| num_medications      | 101,766   | 16.02     | 8.13     | 1     | 10        | 15     | 20        | 81    |
| number_outpatient    | 101,766   | 0.37      | 1.27     | 0     | 0         | 0      | 0         | 42    |
| number_emergency     | 101,766   | 0.20      | 0.93     | 0     | 0         | 0      | 0         | 76    |
| number_inpatient     | 101,766   | 0.64      | 1.26     | 0     | 0         | 0      | 1         | 21    |
| number_diagnoses     | 101,766   | 7.42      | 1.93     | 1     | 6         | 8      | 9         | 16    |

Immediately below are the boxplots for each of these numerical columns, which help visualize their distributions and potential outliers:

- ![time_in_hospital](example_2/plots/time_in_hospital_boxplot.png)  
  Most visits are short (1–6 days) with a maximum of 14 days, so we see a relatively narrow distribution.

- ![num_lab_procedures](example_2/plots/num_lab_procedures_boxplot.png)  
  The majority of patients have between about 31 and 57 lab tests. The mean is around 43, suggesting moderate testing for most admissions.

- ![num_procedures](example_2/plots/num_procedures_boxplot.png)  
  Typically, 0–2 procedures; only a minority have more than 2, which indicates some admissions required more intensive intervention.

- ![num_medications](example_2/plots/num_medications_boxplot.png)  
  The average is 16 medications, but many cluster between 10 and 20. A small subset receives notably more, reflecting complex cases or multiple comorbidities.

- ![number_outpatient](example_2/plots/number_outpatient_boxplot.png), ![number_emergency](example_2/plots/number_emergency_boxplot.png), ![number_inpatient](example_2/plots/number_inpatient_boxplot.png)  
  Most patients have no prior outpatient, emergency, or inpatient visits, indicating many single or first-time admissions. A few frequent healthcare utilizers have high counts.

- ![number_diagnoses](example_2/plots/number_diagnoses_boxplot.png)  
  Patients commonly have around 6–9 diagnoses, reflecting the complexity of diabetic patients often presenting multiple comorbidities.

### Interpretation
Most admissions in the dataset are relatively short hospital stays, with moderate lab testing and medication usage. The outliers in admissions, lab procedures, or prior visits highlight patients who have substantially more complex care paths, possibly due to multiple comorbidities or repeated hospital use.

---

## Categorical Columns

Below is a look at a few key categorical columns. These are the largest categories:

- **race**: 76,099 Caucasian, 19,210 AfricanAmerican, followed by smaller groups (Hispanic, Asian, Other).  
- **gender**: 54,708 Female, 47,055 Male, and 3 unknown/invalid.  
- **age**: The most common brackets are [70-80), [60-70), [50-60), and [80-90).  
- **weight**: Over 98,000 unknown (“?”), with only a small fraction recorded.  
- **change**: 54,755 records show “No” change, while 47,011 indicate a medication dosage change.  
- **diabetesMed**: 78,363 indicate “Yes” for diabetes medication prescriptions, 23,403 are “No.”  
- **readmitted**: 54,864 “NO,” 35,545 “>30,” 11,357 “<30.”

Below are bar charts illustrating distributions of several major categorical features:

- ![race](example_2/plots/race_barchart.png)  
  Caucasian is clearly the largest group, followed by AfricanAmerican. Gaps may reflect underlying population demographics or differences in healthcare access.

- ![gender](example_2/plots/gender_barchart.png)  
  Slightly more females than males in the dataset.

- ![age](example_2/plots/age_barchart.png)  
  Older age brackets dominate hospital admissions, consistent with the chronic nature of diabetes.

- ![weight](example_2/plots/weight_barchart.png)  
  Virtually all are unknown (“?”), underscoring the scarcity of weight data.

- ![change](example_2/plots/change_barchart.png)  
  A substantial subset of patients (surprisingly close to half) had medication dosage changes during their stay.

- ![diabetesMed](example_2/plots/diabetesMed_barchart.png)  
  Most patients did receive a diabetes medication, typical in a diabetic cohort.

- ![readmitted](example_2/plots/readmitted_barchart.png)  
  “NO” is the most common outcome, though a significant portion is readmitted (>30 or <30 days), warranting analysis into potential drivers of readmission.

### Interpretation
Demographically, the dataset is skewed toward older populations, primarily Caucasian, and includes many patients who consistently require diabetes management. The high proportion of medication changes indicates dynamic treatment approaches. The readmission patterns highlight potential needs for better follow-up and care coordination.

---

## Conclusion
Overall, this dataset offers a detailed snapshot of diabetic patients hospitalized in various U.S. hospitals. Numerical columns show diverse lengths of stay, varying intensities of lab and medical procedures, and the frequent presence of multiple comorbidities. Categorical data reveals an older patient distribution, with race and gender distributions reflective of broader demographic patterns. Missing data is notable primarily for weight and certain lab measures (A1C, max glucose serum), so analyses involving these features require careful imputation or restricted inclusion. The dataset thus provides a rich foundation for investigating diabetes care outcomes and the factors that contribute to readmissions.
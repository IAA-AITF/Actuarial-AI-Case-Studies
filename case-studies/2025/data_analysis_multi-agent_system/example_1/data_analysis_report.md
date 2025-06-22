# Introduction
This dataset is commonly referred to as the “Medical Cost Personal Dataset,” containing information about individuals’ demographics, health characteristics, and medical insurance charges. It is frequently analyzed to understand healthcare costs and risk factors in insurance domains.

Before delving into descriptive statistics, let us outline each column’s meaning, preview the first ten records, discuss missing values, and then interpret the statistical findings along with the associated plots.

---

## Column Descriptions
• age – Age of the primary beneficiary  
• sex – Gender of the beneficiary (male/female)  
• bmi – Body Mass Index, offering insight into weight status  
• children – Number of children or dependents covered  
• smoker – Indicates if the person smokes (yes/no)  
• region – Residential region in the US (northeast, northwest, southeast, southwest)  
• charges – Annual medical insurance charges (USD)  

---

## Data Preview
Below are the first ten rows of the dataset for a quick look:

| age | sex    | bmi   | children | smoker | region    | charges     |
|-----|--------|-------|----------|--------|-----------|-------------|
| 19  | female | 27.9  | 0        | yes    | southwest | 16884.924   |
| 18  | male   | 33.77 | 1        | no     | southeast | 1725.5523   |
| 28  | male   | 33.0  | 3        | no     | southeast | 4449.462    |
| 33  | male   | 22.705| 0        | no     | northwest | 21984.47061 |
| 32  | male   | 28.88 | 0        | no     | northwest | 3866.8552   |
| 31  | female | 25.74 | 0        | no     | southeast | 3756.6216   |
| 46  | female | 33.44 | 1        | no     | southeast | 8240.5896   |
| 37  | female | 27.74 | 3        | no     | northwest | 7281.5056   |
| 37  | male   | 29.83 | 2        | no     | northeast | 6406.4107   |
| 60  | female | 25.84 | 0        | no     | northwest | 28923.13692 |

---

## Missing Values Summary
All columns in this dataset are fully populated, and there are no missing entries:  
• age: 0  
• sex: 0  
• bmi: 0  
• children: 0  
• smoker: 0  
• region: 0  
• charges: 0  

This completeness is ideal for thorough and reliable analyses.

---

## Descriptive Statistics

### Numerical Columns
Below are the key statistical metrics for the numerical columns. Accompanying each table is a boxplot that illustrates the respective distribution:

1) age  
   - Count: 1338.0  
   - Mean: 39.21  
   - Std: 14.05  
   - Min: 18.0  
   - 25%: 27.0  
   - 50%: 39.0  
   - 75%: 51.0  
   - Max: 64.0  

   Boxplot:  
   ![Age Boxplot](example_1/plots/age_boxplot.png)

   Discussion:  
   The mean age sits at around 39, and the data extend from 18 to 64, showing a wide adult age range. The boxplot appears quite balanced with no significant outliers, indicating a fairly uniform coverage across different adult age groups.

2) bmi  
   - Count: 1338.0  
   - Mean: 30.66  
   - Std: 6.10  
   - Min: 15.96  
   - 25%: 26.30  
   - 50%: 30.40  
   - 75%: 34.69  
   - Max: 53.13  

   Boxplot:  
   ![BMI Boxplot](example_1/plots/bmi_boxplot.png)

   Discussion:  
   The average BMI is roughly 30.66, which borders the clinical definition of obesity. The boxplot indicates some individuals in the higher BMI range (over 40), representing severe obesity cases. Overall, the distribution skews slightly with these higher outliers.

3) children  
   - Count: 1338.0  
   - Mean: 1.09  
   - Std: 1.21  
   - Min: 0.0  
   - 25%: 0.0  
   - 50%: 1.0  
   - 75%: 2.0  
   - Max: 5.0  

   Boxplot:  
   ![Children Boxplot](example_1/plots/children_boxplot.png)

   Discussion:  
   Many policyholders have either no children, one, or two. There are fewer records with three or more children, but up to five in total. The boxplot confirms a right-skewed distribution, indicating that larger family sizes are less frequent.

4) charges  
   - Count: 1338.0  
   - Mean: 13270.42  
   - Std: 12110.01  
   - Min: 1121.87  
   - 25%: 4740.29  
   - 50%: 9382.03  
   - 75%: 16639.91  
   - Max: 63770.43  

   Boxplot:  
   ![Charges Boxplot](example_1/plots/charges_boxplot.png)

   Discussion:  
   The cost distribution has a long tail toward high values. The boxplot shows a cluster of outliers in the upper range, reflecting expensive medical interventions for some policyholders. Most charges range below $20,000, with a median close to $9,400.

---

### Categorical Columns
Next, we look at the counts of each category. Right after each table, a bar chart shows the distribution visually:

1) sex  
   - male: 676  
   - female: 662  

   Bar Chart:  
   ![Sex Bar Chart](example_1/plots/sex_barchart.png)

   Discussion:  
   The dataset has nearly equal numbers of males and females, suggesting a balanced sample for gender-based analysis and modeling.

2) smoker  
   - no: 1064  
   - yes: 274  

   Bar Chart:  
   ![Smoker Bar Chart](example_1/plots/smoker_barchart.png)

   Discussion:  
   Smokers comprise a smaller portion of the data (about 20%), which is typical of population samples. In health cost datasets, this group is often crucial because their medical expenses can be substantially higher.

3) region  
   - southeast: 364  
   - southwest: 325  
   - northwest: 325  
   - northeast: 324  

   Bar Chart:  
   ![Region Bar Chart](example_1/plots/region_barchart.png)

   Discussion:  
   Individuals come from four primary regions, each with a comparable representation. This distribution reduces geographic bias and allows the analysis of regional differences in insurance charges or health traits.

---

## Conclusion
Overall, the dataset is thorough and well-suited for modeling and analysis. It covers a wide adult age range, presents a broad spread of BMI values, and captures medical costs with clear outliers on the high end. Categorical forests show a nearly even gender split, a modest proportion of smokers, and balanced regional samples. Such features make this dataset an excellent candidate for exploring how various demographic and health-related factors influence insurance charges.
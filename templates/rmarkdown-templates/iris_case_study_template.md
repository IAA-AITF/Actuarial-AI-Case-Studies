Iris Dataset Case Study
================
Your Name
YYYY-MM-DD

- [1 Introduction](#1-introduction)
- [2 Data Loading & Preview](#2-data-loading--preview)
- [3 Exploratory Visualization](#3-exploratory-visualization)
- [4 Minimal Working ML Example](#4-minimal-working-ml-example)
- [5 How‑To Examples](#5-howto-examples)
- [6 External Resources](#6-external-resources)

# 1 Introduction

This RMarkdown provides a minimal case study on the Iris dataset.  
We’ll cover loading the data, exploratory visualization, building a
simple classifier, and referencing external resources.

# 2 Data Loading & Preview

Below we load the Iris dataset, convert it to a data frame, and preview
the first five rows.

``` r
# Load necessary libraries
library(datasets)
library(dplyr)

# Load the iris dataset
df <- as_tibble(iris)

# Display the first 5 rows
head(df, 5)
```

    ## # A tibble: 5 × 5
    ##   Sepal.Length Sepal.Width Petal.Length Petal.Width Species
    ##          <dbl>       <dbl>        <dbl>       <dbl> <fct>  
    ## 1          5.1         3.5          1.4         0.2 setosa 
    ## 2          4.9         3            1.4         0.2 setosa 
    ## 3          4.7         3.2          1.3         0.2 setosa 
    ## 4          4.6         3.1          1.5         0.2 setosa 
    ## 5          5           3.6          1.4         0.2 setosa

The code above:  
- Loads the built‑in `iris` dataset.  
- Converts it to a tibble for convenient printing.  
- Shows the first five rows of the data frame.

# 3 Exploratory Visualization

We create a simple scatter plot of two features to visualize class
separation.

``` r
library(ggplot2)

# Scatter plot of Sepal Length vs. Sepal Width
ggplot(df, aes(x = Sepal.Length, y = Sepal.Width, color = Species)) +
  geom_point() +
  labs(
    title = "Sepal Length vs. Sepal Width",
    x = "Sepal Length (cm)",
    y = "Sepal Width (cm)"
  )
```

![](iris_case_study_template_files/figure-gfm/exploratory-plot-1.png)<!-- -->

The plot above shows the relationship between sepal length and width,
colored by species.

# 4 Minimal Working ML Example

We’ll split the data into training and testing sets, train a multinomial
logistic regression classifier, and evaluate its performance.

``` r
library(caret)
library(nnet)

set.seed(42)
# Split into train/test
train_index <- createDataPartition(df$Species, p = 0.8, list = FALSE)
train <- df[train_index, ]
test  <- df[-train_index, ]

# Train a multinomial logistic regression model
model <- multinom(Species ~ ., data = train)
```

    ## # weights:  18 (10 variable)
    ## initial  value 131.833475 
    ## iter  10 value 12.601102
    ## iter  20 value 1.478352
    ## iter  30 value 0.501860
    ## iter  40 value 0.383960
    ## iter  50 value 0.295028
    ## iter  60 value 0.262898
    ## iter  70 value 0.238648
    ## iter  80 value 0.221312
    ## iter  90 value 0.214986
    ## iter 100 value 0.195268
    ## final  value 0.195268 
    ## stopped after 100 iterations

``` r
# Predict and evaluate
pred <- predict(model, test)
conf_mat <- confusionMatrix(pred, test$Species)

# Show results
print(conf_mat)
```

    ## Confusion Matrix and Statistics
    ## 
    ##             Reference
    ## Prediction   setosa versicolor virginica
    ##   setosa         10          0         0
    ##   versicolor      0          9         1
    ##   virginica       0          1         9
    ## 
    ## Overall Statistics
    ##                                           
    ##                Accuracy : 0.9333          
    ##                  95% CI : (0.7793, 0.9918)
    ##     No Information Rate : 0.3333          
    ##     P-Value [Acc > NIR] : 8.747e-12       
    ##                                           
    ##                   Kappa : 0.9             
    ##                                           
    ##  Mcnemar's Test P-Value : NA              
    ## 
    ## Statistics by Class:
    ## 
    ##                      Class: setosa Class: versicolor Class: virginica
    ## Sensitivity                 1.0000            0.9000           0.9000
    ## Specificity                 1.0000            0.9500           0.9500
    ## Pos Pred Value              1.0000            0.9000           0.9000
    ## Neg Pred Value              1.0000            0.9500           0.9500
    ## Prevalence                  0.3333            0.3333           0.3333
    ## Detection Rate              0.3333            0.3000           0.3000
    ## Detection Prevalence        0.3333            0.3333           0.3333
    ## Balanced Accuracy           1.0000            0.9250           0.9250

We chose a multinomial logistic regression (`nnet::multinom`) for its
simplicity in handling multi‑class problems.  
The confusion matrix above summarizes the model’s performance.

# 5 How‑To Examples

- [Go to Minimal Working ML Example](#4-minimal-working-ml-example)  
- To learn more about R’s built‑in datasets, see the [iris
  documentation](https://stat.ethz.ch/R-manual/R-devel/library/datasets/html/iris.html).

# 6 External Resources

Further reading: [Scikit‑learn Iris dataset
documentation](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_iris.html)

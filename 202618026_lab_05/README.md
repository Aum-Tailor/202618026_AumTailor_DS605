# Machine Learning with Scikit-learn and From Scratch

## Project Overview

This project implements and compares **Linear Regression** and
**Logistic Regression** using two approaches:

1.  **Scikit-learn**
2.  **Manual implementation using NumPy and Pandas**

The manual models were then optimized to reduce computational time while
maintaining comparable predictive performance.

The project follows the complete workflow:

``` text
Raw Dataset
    ↓
Data Exploration
    ↓
Data Cleaning
    ↓
Target Construction
    ↓
Feature Preparation
    ↓
Fixed Train-Test Split
    ↓
Preprocessing
    ↓
Scikit-learn Models
    ↓
Manual Models
    ↓
Evaluation
    ↓
Comparison
    ↓
Optimization
    ↓
Final Comparison
```

------------------------------------------------------------------------

# 1. Assignment Objectives

The main objectives of the assignment were to:

-   Apply machine-learning preprocessing using Scikit-learn.
-   Build a Linear Regression model for continuous productivity
    prediction.
-   Build a Logistic Regression model for target-achievement
    classification.
-   Reproduce the same workflow manually using NumPy and Pandas.
-   Use the same train-test observations for fair comparison.
-   Compare model performance and computational time.
-   Optimize the manual implementations.
-   Analyze why the optimized implementations perform differently from
    Scikit-learn.

------------------------------------------------------------------------

# 2. Dataset

The project uses the **UCI Productivity Prediction of Garment
Employees** dataset.

The dataset contains information related to garment production,
including:

-   Date
-   Quarter
-   Department
-   Day
-   Team
-   Targeted productivity
-   SMV
-   WIP
-   Overtime
-   Incentive
-   Idle time
-   Idle workers
-   Number of style changes
-   Number of workers
-   Actual productivity

### Dataset Shape

``` text
1197 observations × 15 original columns
```

------------------------------------------------------------------------

# 3. Data Exploration and Cleaning

A general preprocessing notebook was used to inspect the raw dataset
before model implementation.

The following were examined:

-   Dataset shape
-   Data types
-   Missing values
-   Unique values
-   Duplicate observations
-   Numerical and categorical variables
-   Variable distributions
-   Target distribution
-   Potential data-entry errors
-   Extreme observations

## 3.1 Actual Productivity Errors

The target variable `actual_productivity` is expected to represent
productivity on a 0--1 scale.

During exploration, values greater than 1 were identified and
investigated as data-entry errors. These were corrected before saving
the cleaned dataset.

The cleaned dataset was saved as:

``` text
cleaned_productivity.csv
```

## 3.2 Extreme Values

Potential IQR-based extreme observations were observed in variables such
as:

-   `idle_men`
-   `idle_time`
-   `wip`
-   `incentive`
-   `over_time`
-   `no_of_style_change`

These observations were retained rather than automatically removed.

The reason was that an IQR outlier is not automatically a data error.
The dataset is also relatively small, so removing valid extreme
observations could unnecessarily reduce useful information.

------------------------------------------------------------------------

# 4. Target Variables

## 4.1 Regression Target

For Linear Regression:

``` text
Target = actual_productivity
```

The objective is to predict the continuous value of actual productivity.

------------------------------------------------------------------------

## 4.2 Classification Target

For Logistic Regression, a binary variable called `MeetsTarget` was
created:

``` python
df["MeetsTarget"] = (
    df["actual_productivity"] >= df["targeted_productivity"]
).astype(int)
```

Therefore:

``` text
MeetsTarget = 1
→ Actual productivity >= Targeted productivity

MeetsTarget = 0
→ Actual productivity < Targeted productivity
```

### Class Distribution

  Class                  Count   Proportion
  ----------------- ---------- ------------
  MeetsTarget = 1          875       73.10%
  MeetsTarget = 0          322       26.90%
  **Total**           **1197**     **100%**

For classification, `actual_productivity` was removed from the predictor
variables so that the continuous target was not directly supplied to the
classifier.

------------------------------------------------------------------------

# 5. Feature Preparation

The predictors were divided into numerical and categorical features.

## Numerical Features

``` text
team
targeted_productivity
smv
wip
over_time
incentive
idle_time
idle_men
no_of_style_change
no_of_workers
```

## Categorical Features

``` text
date
quarter
department
day
```

------------------------------------------------------------------------

# 6. Train-Test Split

A single fixed 80/20 split was created:

``` python
train_test_split(
    df.index,
    test_size=0.20,
    random_state=42
)
```

The resulting data contained:

``` text
Training observations: 957
Testing observations : 240
```

The **same train and test indices** were reused for every
implementation.

This was important for a fair comparison because differences in
performance should come from the implementation/model rather than from
different samples.

------------------------------------------------------------------------

# 7. Preprocessing

## Numerical Features

For the Scikit-learn implementation:

-   Missing values were handled using median imputation.
-   Numerical variables were standardized using `StandardScaler`.

For the manual implementation, the equivalent operations were
implemented using Pandas and NumPy.

The training-set statistics were used to transform the test data.

## Categorical Features

Categorical variables were:

-   Missing-value handled using the training-set mode.
-   One-hot encoded.

The manual implementation reproduced one-hot encoding using Pandas.

### Final Feature Matrix

Before adding the intercept:

``` text
Training matrix: 957 × 82
Testing matrix : 240 × 82
```

After adding the intercept column for the manual models:

``` text
Training matrix: 957 × 83
Testing matrix : 240 × 83
```

------------------------------------------------------------------------

# 8. Part A --- Scikit-learn Implementation

## 8.1 Linear Regression

Scikit-learn's `LinearRegression` was used to predict
`actual_productivity`.

### Evaluation Metrics

-   Mean Absolute Error (MAE)
-   Root Mean Squared Error (RMSE)
-   R²

### Results

  Metric                  Result
  ----------------- ------------
  Training Time       0.155152 s
  Prediction Time     0.007354 s
  MAE                   0.112381
  RMSE                  0.150048
  R²                    0.152084

------------------------------------------------------------------------

# 9. Logistic Regression

Scikit-learn's `LogisticRegression` was used to predict `MeetsTarget`.

The initial classification predictions used the default probability
threshold of 0.5.

### Results at Threshold 0.5

  Metric        Result
  ----------- --------
  Accuracy      0.7417
  Precision     0.7700
  Recall        0.9266
  F1 Score      0.8410

------------------------------------------------------------------------

# 10. ROC-AUC and Classification Threshold

ROC analysis was performed using predicted probabilities.

The ROC-AUC was:

``` text
0.6945
```

A threshold was selected using Youden's J statistic:

``` text
Youden's J = TPR - FPR
```

The selected threshold was:

``` text
0.7329
```

At this threshold:

  Metric        Result
  ----------- --------
  Accuracy      0.6917
  Precision     0.8552
  Recall        0.7006
  F1 Score      0.7702

The threshold of **0.7329** was then kept fixed for the
manual-vs-Scikit-learn Logistic Regression comparison.

> **Methodological note:** In a production modeling workflow, the
> classification threshold should ideally be selected using validation
> data or cross-validation and then evaluated on an untouched test set.
> In this assignment, the ROC-derived threshold was used consistently
> for the required comparison.

------------------------------------------------------------------------

# 11. Part B --- Manual Implementation

The same workflow was reproduced without using Scikit-learn
model-fitting or metric functions.

The manual implementation used:

``` text
NumPy
Pandas
```

The manual workflow included:

-   Missing-value handling
-   Standardization
-   One-hot encoding
-   Matrix construction
-   Intercept addition
-   Linear Regression
-   Logistic Regression
-   Prediction
-   Evaluation metrics

------------------------------------------------------------------------

# 12. Manual Linear Regression

The Linear Regression coefficients were calculated using the
normal-equation approach.

Conceptually:

``` text
β = (XᵀX)⁻¹Xᵀy
```

The initial implementation used:

``` python
beta_manual = np.linalg.pinv(XtX) @ Xty
```

### Results

  Metric                  Manual
  ----------------- ------------
  Training Time       0.005508 s
  Prediction Time     0.000185 s
  MAE                   0.112381
  RMSE                  0.150048
  R²                    0.152076

The manual implementation reproduced the Scikit-learn predictive results
very closely.

------------------------------------------------------------------------

# 13. Manual Logistic Regression

Logistic Regression was implemented from scratch using:

-   Sigmoid function
-   Matrix multiplication
-   Gradient calculation
-   Gradient descent
-   Manual probability prediction
-   Manual classification
-   Manual confusion matrix
-   Manual Accuracy
-   Manual Precision
-   Manual Recall
-   Manual F1 Score

The original implementation used:

``` text
Learning rate = 0.01
Epochs = 10,000
```

The same classification threshold of `0.7329` was used.

### Results

  Metric                  Manual
  ----------------- ------------
  Training Time       0.366106 s
  Prediction Time     0.000030 s
  Accuracy              0.675000
  Precision             0.841379
  Recall                0.689266
  F1 Score              0.757764

------------------------------------------------------------------------

# 14. Part C --- Optimization

The objective of Part C was to improve the computational efficiency of
the manual implementations while maintaining predictive performance.

The optimization process was iterative.

The important principle followed was:

``` text
Implement
   ↓
Measure
   ↓
Identify bottleneck
   ↓
Optimize
   ↓
Measure again
   ↓
Keep or reject the optimization
```

Not every attempted optimization improved runtime.

------------------------------------------------------------------------

# 15. Linear Regression Optimization

## Original Manual Approach

The original implementation used:

``` python
np.linalg.pinv(XtX) @ Xty
```

The pseudoinverse calculation was replaced by directly solving the
linear system:

``` python
np.linalg.solve(XtX, Xty)
```

This avoided explicitly calculating the pseudoinverse.

### Final Linear Regression Comparison

  -------------------------------------------------------------------------------------------
  Implementation   Train Time (s)     Prediction            MAE           RMSE             R²
                                        Time (s)                               
  ---------------- -------------- -------------- -------------- -------------- --------------
  Scikit-learn           0.155152       0.007354       0.112381       0.150048       0.152084

  Manual                 0.005508       0.000185       0.112381       0.150048       0.152076

  **Optimized        **0.001409**   **0.000116**   **0.112381**   **0.150048**   **0.152076**
  Manual**                                                                     
  -------------------------------------------------------------------------------------------

### Optimization Result

Compared with the original manual implementation:

``` text
Training-time reduction   = 74.43%
Prediction-time reduction = 37.07%
Training speedup          = 3.91×
Prediction speedup        = 1.59×
```

The predictive metrics remained effectively unchanged.

------------------------------------------------------------------------

# 16. Logistic Regression Optimization

Logistic Regression required several optimization attempts.

## Stage 1 --- Original Manual Model

``` text
Gradient Descent
10,000 epochs
```

Training time:

``` text
0.366106 s
```

------------------------------------------------------------------------

## Stage 2 --- Early Stopping

Early stopping was introduced to avoid unnecessary iterations.

However, the measured training time increased to:

``` text
0.819451 s
```

Therefore, this optimization was **not retained**.

This demonstrated that reducing the number of optimization iterations
does not necessarily reduce total runtime if additional calculations,
such as repeated loss evaluation, introduce significant overhead.

------------------------------------------------------------------------

## Stage 3 --- Momentum

Momentum-based gradient descent was introduced along with a higher
learning rate.

Training time improved to:

``` text
0.060443 s
```

This was substantially faster than the original manual implementation,
but it was still slower than the measured Scikit-learn training time:

``` text
Scikit-learn = 0.043161 s
```

Therefore, optimization continued.

------------------------------------------------------------------------

## Stage 4 --- Final Optimization

The final implementation used momentum-based gradient descent with a
reduced number of epochs:

``` text
Momentum + 500 epochs
```

Training time became:

``` text
0.027837 s
```

This was below the measured Scikit-learn training time of:

``` text
0.043161 s
```

### Final Logistic Regression Comparison

  ----------------------------------------------------------------------------------------------------------
  Implementation   Train Time (s)     Prediction       Accuracy      Precision         Recall       F1 Score
                                        Time (s)                                              
  ---------------- -------------- -------------- -------------- -------------- -------------- --------------
  Scikit-learn           0.043161       0.008512       0.691667       0.855172       0.700565       0.770186

  Manual                 0.366106       0.000030       0.675000       0.841379       0.689266       0.757764

  **Optimized        **0.027837**   **0.000028**   **0.691667**   **0.855172**   **0.700565**   **0.770186**
  Manual**                                                                                    
  ----------------------------------------------------------------------------------------------------------

### Optimization Result

Compared with the original manual implementation:

``` text
Training-time reduction   = 92.40%
Training speedup          = 13.15×
Prediction-time reduction = 4.35%
Prediction speedup        = 1.05×
```

The optimized manual model achieved the same reported classification
metrics as Scikit-learn at the fixed threshold of `0.7329` in this run.

------------------------------------------------------------------------

# 17. Optimization History

  ------------------------------------------------------------------------
  Model            Stage                Training Time (s) Optimization
  ---------------- ---------------- --------------------- ----------------
  Linear           Original Manual               0.005508 Pseudoinverse
  Regression                                              

  Logistic         Original Manual               0.366106 Gradient Descent
  Regression                                              --- 10,000
                                                          epochs

  Logistic         Early Stopping                0.819451 Early stopping
  Regression                                              

  Logistic         Momentum                      0.060443 Momentum +
  Regression                                              higher learning
                                                          rate

  Logistic         Final Optimized           **0.027837** Momentum + 500
  Regression                                              epochs
  ------------------------------------------------------------------------

This history was retained to show the actual optimization process,
including approaches that did not produce the desired runtime
improvement.

------------------------------------------------------------------------

# 18. Final Combined Comparison

## Linear Regression

  -------------------------------------------------------------------------------------------
  Implementation       Train Time     Prediction            MAE           RMSE             R²
                                            Time                               
  ---------------- -------------- -------------- -------------- -------------- --------------
  Scikit-learn           0.155152       0.007354       0.112381       0.150048       0.152084

  Manual                 0.005508       0.000185       0.112381       0.150048       0.152076

  **Optimized        **0.001409**   **0.000116**   **0.112381**   **0.150048**   **0.152076**
  Manual**                                                                     
  -------------------------------------------------------------------------------------------

## Logistic Regression

  ----------------------------------------------------------------------------------------------------------
  Implementation       Train Time     Prediction       Accuracy      Precision         Recall             F1
                                            Time                                              
  ---------------- -------------- -------------- -------------- -------------- -------------- --------------
  Scikit-learn           0.043161       0.008512       0.691667       0.855172       0.700565       0.770186

  Manual                 0.366106       0.000030       0.675000       0.841379       0.689266       0.757764

  **Optimized        **0.027837**   **0.000028**   **0.691667**   **0.855172**   **0.700565**   **0.770186**
  Manual**                                                                                    
  ----------------------------------------------------------------------------------------------------------

------------------------------------------------------------------------

# 19. Overall Findings

### Linear Regression

The manual Linear Regression implementation produced almost identical
predictive metrics to Scikit-learn.

Replacing the pseudoinverse operation with `np.linalg.solve()`
substantially reduced the training and prediction times while preserving
model performance.

### Logistic Regression

The original manual Logistic Regression implementation was considerably
slower because it required many gradient-descent iterations.

Several optimization strategies were evaluated. Early stopping increased
runtime because of additional computation. Momentum-based optimization
substantially reduced training time, but the first momentum
configuration was still slower than Scikit-learn.

Reducing the number of epochs while retaining momentum produced the
final optimized implementation.

The final optimized model:

-   Reduced manual training time by **92.40%**
-   Was approximately **13.15× faster than the original manual
    implementation**
-   Had a measured training time below the Scikit-learn implementation
-   Matched the reported classification metrics of Scikit-learn at the
    selected threshold in this run

------------------------------------------------------------------------

# 20. Important Implementation Insight

The project demonstrates that an algorithm being mathematically
equivalent does not guarantee identical computational performance.

For example:

``` text
Same model
    +
Same data
    +
Same features
```

can still produce different runtimes because of:

-   Numerical linear algebra routines
-   Optimization algorithms
-   Number of iterations
-   Convergence criteria
-   Vectorization
-   Repeated calculations
-   Implementation overhead

The optimization experiments therefore focused not only on model
accuracy but also on how the algorithm was computationally implemented.

------------------------------------------------------------------------

# 21. Limitations

The measured runtime values are hardware-, Python-version-,
library-version-, and system-load-dependent. Therefore, the exact timing
values should not be interpreted as universal benchmarks.

The classification threshold was selected from ROC analysis for this
assignment and then fixed for the comparison. In a production workflow,
threshold selection should normally be performed using a validation
procedure rather than the final test set.

The relatively low Linear Regression R² (`≈ 0.152`) also indicates that
the linear model explains only a limited amount of variation in the
continuous productivity target. This is a property of the fitted model
and dataset and was not artificially changed through standardization.

------------------------------------------------------------------------

# 22. Conclusion

This project demonstrated the complete machine-learning workflow from
raw data exploration to model implementation, evaluation, comparison,
and optimization.

Both Linear Regression and Logistic Regression were implemented using
Scikit-learn and reproduced manually using NumPy and Pandas.

The manual implementations successfully reproduced the main predictive
behavior of the library-based models. Computational optimization was
then performed iteratively.

For Linear Regression, replacing the pseudoinverse with a direct
linear-system solver reduced the manual training time by **74.43%**.

For Logistic Regression, several optimization strategies were tested.
The final momentum-based implementation with 500 epochs reduced manual
training time by **92.40%**, achieving a measured training time below
the Scikit-learn implementation while retaining the reported
classification performance.

The project therefore demonstrates both the **mathematical
implementation of machine-learning algorithms** and the practical
importance of **computational optimization and benchmarking**.

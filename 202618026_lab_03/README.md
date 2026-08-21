# Scikit-learn Preprocessing and Model Evaluation

Name: Aum Tailor 
Student ID: 202618026
Course: DS605 — Fundamentals of Machine Learning

# Project Overview
This repository contains my work for Lab 3, where I built and compared Scikit-learn preprocessing pipelines and evaluated two classification models on the task of predicting hotel booking cancellations.

- Part A (Preprocessing): Loaded and explored the Hotel Booking Demand dataset, handled missing values, removed data-leakage columns, checked for outliers, and built two separate preprocessing pipelines using `ColumnTransformer` and `Pipeline`.
- Part B (Model Training & Evaluation): Trained Logistic Regression and Decision Tree classifiers on both pipelines, evaluated all four combinations, and compared their performance to identify signs of overfitting and the best overall approach.

# Dataset
Kaggle Hotel Booking Demand** (`hotel_bookings.csv`)
Link: https://www.kaggle.com/datasets/jessemostipak/hotel-booking-demand

Goal: Predict whether a booking will be cancelled (`is_canceled` = 1) or not (0), based on features like lead time, deposit type, room type, and guest history.

Target variable: `is_canceled`

# Preprocessing Choices
- Missing values: Calculated missing count/percentage for every column. Dropped `company` due to very high missingness (~94%), since imputing it would mostly be guessing. Numerical missing values were handled with `KNNImputer(n_neighbors=5)`; categorical missing values with `SimpleImputer(strategy="most_frequent")`.
- Leakage columns: Removed `reservation_status` and `reservation_status_date`, as they directly reveal the final booking outcome and would let the model "cheat."
- Outliers: Checked numerical features (`lead_time`, `adr`, stay-length columns) using boxplots and the IQR method. Removed only clear, extreme outliers (e.g., negative or unrealistically high `adr` values) rather than aggressively trimming the data, and reported the number of rows removed.
- Encoding: Categorical features encoded with `OneHotEncoder(handle_unknown="ignore")` to safely handle unseen categories at test time.
- Scaling — two pipelines compared:
  - Pipeline A: `KNNImputer` + `StandardScaler` for numerical features.
  - Pipeline B: `KNNImputer` + `MinMaxScaler` for numerical features.
- Train-test split: `train_test_split(test_size=0.2, stratify=y, random_state=42)`, used consistently across all four experiments. All preprocessing was fit only on the training data to avoid data leakage.

# Observations
1. Overfitting pattern: Decision Tree models reach near-perfect training accuracy but show a noticeably larger drop on test accuracy compared to Logistic Regression, indicating overfitting. Logistic Regression's smaller train-test gap suggests it generalizes better.
2. Effect of scaling on Logistic Regression: StandardScaler and MinMaxScaler produce very similar results for Logistic Regression, with only marginal differences in accuracy/F1 — scaling method has a limited effect since it's a coefficient-based linear model relying on relative feature magnitudes.
3. Effect of scaling on Decision Tree: Decision Tree performance is essentially unchanged between Pipeline A and Pipeline B, confirming that tree-based splits are scale-invariant.
4. Best overall combination: The combination with the highest test F1-score and the smallest train-test accuracy gap gives the best overall result — typically Logistic Regression, since it trades a bit of raw training accuracy for better generalization.
5. Confusion matrix insight: The Decision Tree captures more true positives for the cancellation class during training, but this advantage shrinks noticeably on test data — reinforcing that its apparent strength is partly due to overfitting rather than true predictive power.

# Tech Stack
Python, Pandas, Numpy, Matplotlib/Seaborn, Scikit-learn (`Pipeline`, `ColumnTransformer`, `KNNImputer`, `SimpleImputer`, `StandardScaler`, `MinMaxScaler`, `OneHotEncoder`, `LogisticRegression`, `DecisionTreeClassifier`).
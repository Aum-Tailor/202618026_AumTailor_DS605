Assignment: Vectorized Programming and Data Wrangling


Name: Aum Tailor
Student ID: 202618026
Course: DS605 Fundamentals of machine learning 

Project Overview

This repository contains my work for a two-part assignment where I practiced working with arrays and real-world data using Python.
Part A (NumPy): I used NumPy to generate random numbers, manipulate arrays, and perform linear algebra operations like matrix multiplication and finding inverses without using slow Python loops. I also simulated a normal distribution and visualized it with a histogram.
Part B (Pandas and Data Wrangling): I analyzed the famous Titanic dataset. I cleaned up missing data, searched for extreme ticket prices, created new features, and built some charts to figure out what actually helped passengers survive.

What is in this Repository?

1. Assignment_Notebook.ipynb: My Jupyter Notebook containing all the Python code, outputs, and visualizations for both Part A and Part B.


2. train.csv: The original, raw Titanic dataset I started with.


3. cleaned_titanic.csv: The final version of the dataset after I filled in the missing ages and added my new custom features.


4. README text: This file, summarizing the project, key observations, and assignment deliverables.



About the Dataset

I used the Titanic: Machine Learning from Disaster dataset (train.csv).
The Goal: To explore the data and understand the factors that influenced whether a passenger survived (1) or did not (0).
Key Info Available: Passenger Class, Gender, Age, Ticket Fare, Siblings/Spouses aboard, Parents/Children aboard, and the Port of Embarkation.
New Features Added: I combined the family-related columns to create a FamilySize feature, and an IsAlone feature to flag people traveling by themselves.

Observations

1. Multicollinearity Risk: 'FamilySize' has a nearly perfect positive correlation with 'SibSp' (0.89) and 'Parch' (0.78). For linear models (like Logistic Regression), dropping the original 'SibSp'/'Parch' in favor of the engineered feature is recommended to satisfy feature independence assumptions.

2. Skewness and Scaling: The scatter plot reveals a massive right-skew in the 'Fare' distribution with extreme outliers. A log transformation (e.g., np.log1p) is highly recommended before feeding this into distance-based models (KNN, SVM) to prevent 'Fare' magnitude from dominating the loss function.

3. Non-linear Relationships: 'Age' shows a near-zero Pearson correlation (-0.08) with 'Survived', implying a weak global linear relationship. However, tree-based models might still extract value by creating localized splits (e.g., children < 10 having higher survival odds despite the overall flat trend).

4. Feature Importance Signals: 'Pclass' has a strong negative correlation (-0.34) with 'Survived' and 'Fare' (-0.55). This proxy relationship indicates socioeconomic status is a primary signal. Tree-based feature importance will likely rank Sex, Pclass, and Fare at the very top.

5. Interaction Potential: Given that survival heavily depends on being Female AND upper class, creating explicit interaction terms (e.g., Sex_mapped * Pclass) could provide deterministic linear boundaries for simpler baseline models.
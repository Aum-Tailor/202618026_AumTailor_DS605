# Airbnb Price Prediction — NYC

A machine learning regression project for predicting Airbnb listing prices using the **AB_NYC_2019** dataset. The project covers exploratory data analysis, data cleaning, feature engineering, preprocessing, baseline model comparison, hyperparameter tuning with RandomizedSearchCV and Optuna, final model evaluation, feature importance, and model serialization.

## Project Overview

**Objective:** Predict the price of an Airbnb listing from its location, room type, availability, review activity, minimum-night requirements, and host/listing characteristics.

The target variable is transformed using:

```python
log_price = np.log1p(price)
```

Predictions are converted back to the original dollar scale using:

```python
np.expm1(prediction)
```

## Dataset

The notebook uses the **AB_NYC_2019.csv** dataset.

The dataset contains Airbnb listings from New York City, including variables related to:

- Neighbourhood and neighbourhood group
- Room type
- Latitude and longitude
- Minimum nights
- Number of reviews
- Reviews per month
- Host listing count
- Availability
- Price
- Last review date

> **Note:** The dataset itself is not included in this repository. Update the CSV path in the notebook according to your local environment.

## Workflow

The project follows this pipeline:

```text
Raw Dataset
     ↓
Exploratory Data Analysis
     ↓
Data Cleaning
     ↓
Outlier Handling
     ↓
Feature Engineering
     ↓
Train/Test Split
     ↓
Preprocessing
     ↓
Baseline Model Comparison
     ↓
Hyperparameter Tuning
     ↓
Optuna Optimization
     ↓
Final Gradient Boosting Model
     ↓
Evaluation
     ↓
Feature Importance
     ↓
Model Serialization
```

## 1. Exploratory Data Analysis

The notebook performs:

- Dataset shape and structure inspection
- Missing-value analysis
- Descriptive statistics for price
- Price distribution visualization
- Price comparison across neighbourhood groups
- Price comparison across room types
- Geographic visualization using latitude and longitude
- Correlation heatmap for selected numerical variables

Generated visualizations are stored in the `assets/` directory.

## 2. Data Cleaning and Feature Engineering

The following preprocessing steps are applied:

### Price

Listings with `price = 0` are removed because they do not represent valid listing prices.

### Missing review information

`reviews_per_month` is filled with `0`, representing listings with no recorded reviews.

Missing `name` and `host_name` values are filled with `"Unknown"`.

### Last review

`last_review` is converted to datetime and transformed into:

```text
days_since_last_review
```

The reference date is the latest observed review date in the dataset. Listings without a review are assigned a value beyond the maximum observed number of days.

### Outlier handling

- Price is capped using the **99th percentile**.
- `minimum_nights` is capped at **30 days**.

### Target transformation

Because price is highly skewed, the target is transformed using:

```python
df["log_price"] = np.log1p(df["price"])
```

## 3. Features

The model uses the following predictors:

### Categorical features

- `neighbourhood_group`
- `neighbourhood`
- `room_type`

### Numerical features

- `latitude`
- `longitude`
- `minimum_nights`
- `number_of_reviews`
- `reviews_per_month`
- `calculated_host_listings_count`
- `availability_365`
- `days_since_last_review`

### Target

- `log_price`

## 4. Preprocessing

A `ColumnTransformer` is used to apply different transformations to numerical and categorical variables.

### Numerical pipeline

```text
Median Imputation
      ↓
StandardScaler
```

### Categorical pipeline

```text
Most-Frequent Imputation
      ↓
One-Hot Encoding
```

The preprocessing and model are combined using a Scikit-learn `Pipeline`, preventing preprocessing steps from being separated from model training.

## 5. Train/Test Split

The data is divided using:

```python
train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)
```

Thus, **80%** of the data is used for training and **20%** for testing.

## 6. Baseline Models

The initial model comparison includes:

- Linear Regression
- Ridge Regression
- Random Forest Regressor
- Gradient Boosting Regressor

Models are evaluated using:

- R²
- RMSE
- MAE

The notebook compares both training and testing performance to examine generalization and the train/test performance gap.

## 7. Hyperparameter Tuning

Random Forest and Gradient Boosting are further tuned using `RandomizedSearchCV`.

The project then uses **Optuna** with 5-fold cross-validation to search a broader hyperparameter space.

### Random Forest search space

Optuna tunes:

- `n_estimators`
- `max_depth`
- `min_samples_split`
- `min_samples_leaf`
- `max_features`

### Gradient Boosting search space

Optuna tunes:

- `n_estimators`
- `max_depth`
- `learning_rate`
- `subsample`
- `min_samples_leaf`

Each Optuna study uses **30 trials** and maximizes mean 5-fold cross-validated R².

## 8. Final Model

The final selected model is an **Optuna-tuned Gradient Boosting Regressor**.

The final hyperparameters used in the notebook are:

| Parameter | Value |
|---|---:|
| `n_estimators` | 396 |
| `max_depth` | 6 |
| `learning_rate` | 0.07955102802742788 |
| `subsample` | 0.9084758819863101 |
| `min_samples_leaf` | 10 |
| `random_state` | 42 |

The best Optuna Gradient Boosting trial achieved a cross-validation R² of approximately **0.6463**.

## 9. Final Model Performance

The final model was evaluated on the held-out test set.

### Log-price scale

| Metric | Result |
|---|---:|
| Train R² | 0.7213 |
| Test R² | 0.6561 |
| Train-Test R² Gap | 0.0652 |

### Original dollar scale

| Metric | Result |
|---|---:|
| RMSE | $73.17 |
| MAE | $41.94 |

The final evaluation converts the model predictions from log-price back to the original price scale using `np.expm1()`.

## 10. Model Comparison

The final Optuna comparison reported:

| Model | Train R² | Test R² | Gap |
|---|---:|---:|---:|
| Random Forest (Optuna) | 0.7456 | 0.6516 | 0.0940 |
| Gradient Boosting (Optuna) | 0.7213 | 0.6561 | 0.0652 |

The notebook selected the Optuna-tuned Gradient Boosting model for the final pipeline.

## 11. Visual Outputs

The notebook generates and saves:

```text
assets/
├── eda_overview.png
├── correlation_heatmap.png
├── actual_vs_predicted.png
└── feature_importance.png
```

### Actual vs Predicted

The plot compares actual Airbnb prices with prices predicted by the final Gradient Boosting model.

### Feature Importance

The notebook extracts feature importances from the final Gradient Boosting model and displays the top 15 transformed features.

## 12. Saved Model Files

The notebook saves the trained model using `joblib`:

```text
models/
├── airbnb_price_model.pkl
└── category_options.pkl
```

### `airbnb_price_model.pkl`

Contains the complete final pipeline, including:

- Numerical preprocessing
- Categorical preprocessing
- One-hot encoding
- Final Gradient Boosting model

### `category_options.pkl`

Stores available categorical values for:

- `neighbourhood_group`
- `neighbourhood`
- `room_type`

## 13. Installation

Install the required Python packages:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn joblib optuna
```

## 14. How to Run

1. Place `AB_NYC_2019.csv` in an accessible location.
2. Update the dataset path in the notebook:

```python
df = pd.read_csv("path/to/AB_NYC_2019.csv")
```

3. Open the notebook in Jupyter Notebook, JupyterLab, or VS Code.
4. Run the cells sequentially.
5. The model and generated visualizations will be saved automatically.

## 15. Project Structure

```text
airbnb-price-prediction/
│
├── README.md
├── notebook.ipynb
│
├── assets/
│   ├── eda_overview.png
│   ├── correlation_heatmap.png
│   ├── actual_vs_predicted.png
│   └── feature_importance.png
│
└── models/
    ├── airbnb_price_model.pkl
    └── category_options.pkl
```

## 16. Technologies Used

- **Python**
- **Pandas** — data manipulation
- **NumPy** — numerical operations
- **Matplotlib** — visualization
- **Seaborn** — statistical visualization
- **Scikit-learn** — preprocessing, modeling and evaluation
- **Optuna** — hyperparameter optimization
- **Joblib** — model serialization

## 17. Key Takeaways

- Airbnb price is highly skewed, so a log transformation is applied to the target.
- Categorical variables are handled through one-hot encoding.
- Numerical variables are imputed and standardized within a preprocessing pipeline.
- Multiple regression models are compared before tuning.
- Randomized search is used for initial hyperparameter tuning.
- Optuna is then used for a broader 30-trial search with 5-fold cross-validation.
- The final model is an Optuna-tuned Gradient Boosting Regressor.
- The final held-out test R² is **0.6561**, with an R² train/test gap of **0.0652**.
- On the original dollar scale, the final model has an RMSE of approximately **$73.17** and MAE of approximately **$41.94**.

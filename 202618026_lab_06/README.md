# Image and Text Classification using Machine Learning

This project focuses on applying traditional machine-learning techniques to two different types of data: **images** and **emails**.

The main idea was to convert the original data into useful numerical representations and then compare different classification models based on their predictive performance and computation time.

## Project Structure

The project is divided into three parts:

- **Part A:** Image Feature Extraction and Classification
- **Part B:** Text Representation and Email Classification
- **Part C:** Improving the Text Representation

---

## Part A – Image Classification

The image dataset contains **crack and non-crack images**.

The images were kept in their original form without resizing. Instead, numerical features were extracted from each image, including:

- Mean brightness
- Contrast
- Minimum intensity
- Maximum intensity
- Median intensity
- Dark pixel ratio
- Bright pixel ratio
- Edge count
- Edge density

These features were then standardized and used with **Logistic Regression** and **K-Nearest Neighbors (KNN)**.

### Results

| Model | Accuracy | Precision | Recall | F1 Score |
|---|---:|---:|---:|---:|
| Logistic Regression | 86.25% | 82.22% | 92.50% | 87.06% |
| KNN | 92.50% | 94.74% | 90.00% | 92.31% |

KNN performed better overall on the extracted image features, while Logistic Regression had slightly higher recall and faster prediction.

---

## Part B – Email Classification

The email dataset was provided as a **word-frequency matrix** containing 3,000 word features. The email identifier was removed and the `Prediction` column was used as the target.

Two text representations were compared:

1. Word Frequency
2. TF-IDF

Both representations were tested using Logistic Regression and KNN.

### Results

| Representation | Model | Accuracy | Precision | Recall | F1 Score |
|---|---|---:|---:|---:|---:|
| Word Frequency | Logistic Regression | 96.62% | 92.06% | 96.67% | 94.31% |
| Word Frequency | KNN | 83.38% | 64.35% | 95.67% | 76.94% |
| TF-IDF | Logistic Regression | 95.07% | 93.08% | 89.67% | 91.34% |
| TF-IDF | KNN | 89.47% | 77.84% | 89.00% | 83.05% |

Logistic Regression performed better than KNN for both text representations. Word Frequency with Logistic Regression gave the highest accuracy and F1 score.

---

## Part C – Improving the Representation

To reduce the size of the text representation, words appearing in fewer than **50 training emails** were removed before applying TF-IDF.

This reduced the number of features from **3,000 to 1,551**, removing **1,449 features**.

### Results

| Approach | Features | Accuracy | F1 Score | Training Time | Prediction Time |
|---|---:|---:|---:|---:|---:|
| Original TF-IDF | 3,000 | 95.07% | 91.34% | 0.0755s | 0.0013s |
| Reduced TF-IDF | 1,551 | 94.59% | 90.48% | 0.0626s | 0.0006s |

The reduced representation was **48.3% smaller** and reduced both training and prediction time. However, there was a small decrease in accuracy and F1 score.

This shows the trade-off between **reducing feature dimensionality and retaining predictive information**.

---

## Overall

The results show that the choice of **feature representation and classification model** has a noticeable effect on both performance and computation time.

For the image data, KNN performed better with the extracted image features. For the email data, Logistic Regression performed better with both word-frequency and TF-IDF representations.

Reducing the text features in Part C made the model faster and smaller, but resulted in a small loss in predictive performance.

Overall, the project provided practical experience with **feature extraction, text representation, classification, model evaluation, and dimensionality reduction**.
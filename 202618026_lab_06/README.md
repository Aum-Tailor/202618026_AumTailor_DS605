Machine Learning – Image & Text Classification
Overview

This project focuses on working with image and email data, converting them into numerical features, and using traditional machine-learning models for classification. The main models used were Logistic Regression and KNN.

Part A – Image Classification

The image dataset contains crack and non-crack images.

Instead of resizing the images, I kept them in their original form and extracted numerical features from each image:

Mean brightness
Contrast
Minimum and maximum intensity
Median intensity
Dark and bright pixel ratios
Edge count
Edge density

These features were then standardized and used with Logistic Regression and KNN.

Results
Model	Accuracy	Precision	Recall	F1
Logistic Regression	86.25%	82.22%	92.50%	87.06%
KNN	92.50%	94.74%	90.00%	92.31%

KNN performed better overall on the extracted image features, while Logistic Regression had slightly higher recall.

Part B – Email Classification

The email dataset was already provided as a word-frequency matrix with 3,000 word features. The Email No. column was removed and Prediction was used as the target.

Two representations were compared:

Word Frequency
TF-IDF

For each representation, Logistic Regression and KNN were trained and evaluated.

Results
Representation	Model	Accuracy	Precision	Recall	F1
Word Frequency	Logistic Regression	96.62%	92.06%	96.67%	94.31%
Word Frequency	KNN	83.38%	64.35%	95.67%	76.94%
TF-IDF	Logistic Regression	95.07%	93.08%	89.67%	91.34%
TF-IDF	KNN	89.47%	77.84%	89.00%	83.05%

Logistic Regression performed better than KNN with both representations. Word Frequency with Logistic Regression gave the highest accuracy and F1-score.

Part C – Improving the Representation

For this part, I reduced the number of text features by removing words that appeared in fewer than 50 training emails.

This reduced the feature count:

3,000 → 1,551 features

Results
Approach	Features	Accuracy	F1	Train Time	Prediction Time
Original TF-IDF	3,000	95.07%	91.34%	0.0755s	0.0013s
Reduced TF-IDF	1,551	94.59%	90.48%	0.0626s	0.0006s

The reduced representation was almost 48.3% smaller and made both training and prediction faster, but there was a small drop in classification performance.

Overall

Working with both datasets showed that feature representation and model choice can noticeably change the results. For the image data, KNN worked better with the extracted features, while for the email data, Logistic Regression performed better. Reducing the text features also showed the trade-off between model performance and computational efficiency.
# Eigenfaces Face Recognition — Results

## 1. PCA and Explained Variance

| Metric | Result |
|---|---:|
| Total images | 400 |
| Training images | 320 |
| Testing images | 80 |
| Unique subjects | 40 |
| Original features per image | 10,000 |
| Components for at least 90% variance | 91 |
| Variance retained by 91 components | 90.08% |
| Components for at least 95% variance | 153 |
| Variance retained by 153 components | 95.03% |
| Relative error: eigenvalues vs. squared singular values | \(9.34 \times 10^{-16}\) |
| Mean absolute cosine similarity of principal directions | Approximately 1.0000 |

**Observations**
- Gram-matrix eigendecomposition and SVD produce consistent principal directions and variance values.
- The first 91 principal components retain 90.08% of the training variance.

## 2. Reconstruction Quality and Component Selection

Test reconstruction MSE is the average squared difference between original test-image pixel intensities and their reconstructed pixel intensities, averaged across all test images and pixels.

\[
\operatorname{MSE}=\frac{1}{ND}\sum_{i=1}^{N}\sum_{j=1}^{D}(X_{ij}-\hat X_{ij})^2
\]

Here, \(N=80\) test images and \(D=10{,}000\) pixels per image. Lower MSE indicates better reconstruction quality.

| Components | Variance Retained | Test Reconstruction MSE | 1-NN Test Accuracy | Feature Reduction |
|---:|---:|---:|---:|---:|
| 50 | 83.41% | 362.7177 | 98.75% | 99.50% |
| 91 | 90.08% | 299.8171 | 98.75% | 99.09% |
| 153 | 95.03% | 258.6437 | 98.75% | 98.47% |

- Reconstruction RMSE using 91 components: 17.3152.

**Observations**
- Reconstruction MSE decreases as the number of components increases.
- Recognition accuracy remains unchanged across the three tested component counts.
- Ninety-one components are selected for the main representation because they retain at least 90% of the training variance.

## 3. Storage Comparison

| Representation | Storage |
|---|---:|
| Original image matrix | 32.0000 MB |
| PCA coefficients | 0.2912 MB |
| Mean face | 0.0800 MB |
| Eigenface basis | 7.2800 MB |
| Total PCA representation | 7.6512 MB |

*Storage estimates use float64 values and include the shared mean face and eigenface basis.*

**Observations**
- Feature-count reduction: 99.09%.
- Total storage reduction, including PCA parameters: 76.09%.
- PCA coefficients require substantially less storage than the original image matrix.

## 4. Raw-Pixel vs. PCA-Space Recognition

A 1-Nearest Neighbor (1-NN) classifier assigns each test image the identity of its closest training image using Euclidean distance.

The same classifier is applied to raw pixels and PCA features to compare the representations while keeping the recognition method fixed. \(k=1\) provides a direct similarity-based baseline because each prediction depends on the single closest training image. This comparison does not establish that \(k=1\) is the best KNN hyperparameter; that is evaluated separately in Section 6.

| Representation | Features per Image | Correct Predictions | Test Accuracy |
|---|---:|---:|---:|
| Raw pixels, 1-NN | 10,000 | 79/80 | 98.75% |
| PCA features, 1-NN | 91 | 79/80 | 98.75% |

**Observations**
- Both representations achieve 98.75% test accuracy.
- Both misclassify test image index 9.
- Actual identity: `s40`.
- Predicted identity: `s5`.
- PCA reduces the feature count by 99.09% without reducing the observed test accuracy.

## 5. Traditional Classifier Comparison

KNN, Logistic Regression, and an RBF-kernel SVM are compared using the same 91-component PCA representation. Five-fold stratified cross-validation is performed on the training set. PCA is fitted separately within each fold to prevent validation-data leakage.

| Model | Mean CV Accuracy | CV Standard Deviation |
|---|---:|---:|
| KNN (\(k=1\)) | 95.94% | 2.34% |
| Logistic Regression | 95.94% | 2.54% |
| SVM (RBF) | 95.31% | 1.71% |

**Observations**
- KNN and Logistic Regression tie for the highest mean cross-validation accuracy.
- KNN is carried forward for further investigation as a similarity-based classifier.
- The results do not show that KNN outperforms Logistic Regression.

## 6. KNN Hyperparameter Comparison

After selecting KNN as the similarity-based classifier, the number of neighbors is varied to evaluate its effect on performance. The same PCA representation and five-fold stratified cross-validation procedure are used.

| Number of Neighbors (\(k\)) | Mean CV Accuracy | CV Standard Deviation |
|---:|---:|---:|
| 1 | 95.94% | 2.34% |
| 3 | 92.19% | 2.21% |
| 5 | 89.69% | 3.64% |

**Observations**
- \(k=1\) achieves the highest mean cross-validation accuracy among the tested settings.
- Mean cross-validation accuracy decreases as \(k\) increases from 1 to 5.
- Therefore, \(k=1\) is selected for the final KNN model based on cross-validation results.

## 7. Final Model Evaluation

The selected KNN model with \(k=1\) is fitted on the complete training set using 91 PCA components and evaluated on the held-out test set.

| Metric | Result |
|---|---:|
| Selected model | KNN (\(k=1\)) |
| PCA components | 91 |
| Correct predictions | 79/80 |
| Test accuracy | 98.75% |

The test set is reserved for final evaluation rather than classifier or hyperparameter selection.

## 8. Conclusions

- PCA reduces the feature count from 10,000 to 91 while retaining 90.08% of the training variance.
- Gram-matrix eigendecomposition and SVD produce consistent numerical results.
- Reconstruction quality improves as more principal components are retained.
- Raw-pixel and PCA-space 1-NN both achieve 98.75% test accuracy.
- PCA achieves the same observed recognition accuracy with 99.09% fewer features.
- KNN and Logistic Regression tie at 95.94% mean cross-validation accuracy.
- Among the tested KNN settings, \(k=1\) achieves the highest mean cross-validation accuracy.
- The final PCA-based KNN model achieves 98.75% test accuracy.

## 9. Output Figures

- `sample_faces.png` — sample face images.
- `explained_variance.png` — explained and cumulative explained variance.
- `eigenfaces.png` — mean face and leading eigenfaces.
- `reconstruction_comparison.png` — original and reconstructed test images.
- `recognition_examples.png` — recognition predictions and misclassification example.
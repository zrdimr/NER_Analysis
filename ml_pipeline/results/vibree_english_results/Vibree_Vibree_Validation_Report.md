# Stress Potential Analysis and Machine Learning Pipeline Report

## 0. Dataset Flags
**Sentiment data flags** mapping in this dataset:
- `1`: **Stress Potential (Positif)**
- `0`: **Not Potential**

## 1. Sentiment Distribution
The distribution of text sentiment polarities analyzed via TextBlob.
![Sentiment Distribution](./1_sentiment_dist.png)

## 2. Dataset Distribution
Class distribution of Stress Potential vs Not Potential before balancing:
![Dataset Distribution](./2_dataset_dist.png)

## 3. Heatmap After Dimensionality Reduction (PCA)
Using PCA to extract 10 components from the numerical metadata features:
![PCA Heatmap](./3_pca_heatmap.png)

## 4. WordCloud
Most frequent terms found across all text samples:
![WordCloud](./4_wordcloud.png)

## 5. Correlation
Pearsons Correlation between TextBlob Sentiment Score and Actual Stress Potential Label: **-0.1334**

## 6. Shape of Combined Features
- Original feature shape (TF-IDF 100 + Sentiment 1): **(5000, 101)**
- Training, Validation, and Test Split performed: **70% / 15% / 15%**
- Shape of the Training Set *after* applying SMOTE for class balancing: **(3902, 101)**
![SMOTE Balance Distribution](./smote_dist.png)

## 7. Confusion Matrix (Logistic Regression)
Performance on the 15% Test set:
![Confusion Matrix](./7_conf_matrix.png)

## 8. Learning Rate & Pipeline Params
- **Vectorization**: TF-IDF (100 features)
- **Oversampling**: SMOTE (on combined continuous features)
- **SMOTE Random State**: 42
- **Model**: Logistic Regression
- **Optimization Strategy**: Default LBFGS solver, max_iterations=1000.
*(Note: standard Logistic Regression does not use a direct 'learning rate' parameter like deep learning models, but converges via exact gradient optimization).*

## 9. Classification Report (Logistic Regression Model)
**Accuracy**: 0.6400

```text
                  precision    recall  f1-score   support

   Not Potential       0.70      0.63      0.66       418
Stress Potential       0.58      0.65      0.62       332

        accuracy                           0.64       750
       macro avg       0.64      0.64      0.64       750
    weighted avg       0.65      0.64      0.64       750

```

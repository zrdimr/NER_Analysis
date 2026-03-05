import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from textblob import TextBlob
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Paths
DATA_PATH = "data/raw/dreaddit_StressAnalysis.csv"
RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

def preprocess_basic(df):
    df['text'] = df['text'].astype(str)
    return df

def generate_report():
    print("Loading data...")
    df = pd.read_csv(DATA_PATH)
    df = preprocess_basic(df)

    img_paths = {}

    # 0. Set Sentiment / Stress Status
    # In the user prompt: "Sentiment data Flags are stress potensial (positif) and not potensial"
    # Actually Dreaddit dataset label 1 = Stress, label 0 = Not Stress.
    df['label_name'] = df['label'].map({1: "Stress Potential", 0: "Not Potential"})

    # 1. Sentiment Distribution
    print("Calculating Sentiment Distribution...")
    df['sentiment'] = df['text'].apply(lambda x: TextBlob(x).sentiment.polarity)
    plt.figure(figsize=(10, 6))
    df['sentiment'].hist(bins=50, color='skyblue', edgecolor='black')
    plt.title("Sentiment Distribution")
    plt.xlabel("Sentiment Polarity")
    plt.ylabel("Frequency")
    img_paths['sentiment_dist'] = os.path.join(RESULTS_DIR, "1_sentiment_dist.png")
    plt.savefig(img_paths['sentiment_dist'])
    plt.close()

    # 2. Dataset Distribution
    print("Calculating Dataset Distribution...")
    plt.figure(figsize=(8, 6))
    sns.countplot(x='label_name', data=df, palette='coolwarm')
    plt.title("Initial Dataset Distribution")
    img_paths['dataset_dist'] = os.path.join(RESULTS_DIR, "2_dataset_dist.png")
    plt.savefig(img_paths['dataset_dist'])
    plt.close()

    # 3. Heatmap After PCA
    print("Generating PCA Heatmap...")
    numeric_df = df.select_dtypes(include=['float64', 'int64']).drop(columns=['label', 'id', 'post_id'], errors='ignore')
    pca = PCA(n_components=10)
    reduced_data = pca.fit_transform(numeric_df.fillna(0))
    reduced_corr = pd.DataFrame(reduced_data).corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(reduced_corr, cmap="coolwarm", annot=True, fmt=".2f")
    plt.title("Heatmap After Dimensionality Reduction (PCA)")
    img_paths['pca_heatmap'] = os.path.join(RESULTS_DIR, "3_pca_heatmap.png")
    plt.savefig(img_paths['pca_heatmap'])
    plt.close()

    # 4. WordCloud
    print("Generating WordCloud...")
    plt.figure(figsize=(12, 10))
    wordcloud = WordCloud(background_color='white', width=800, height=800).generate(' '.join(df['text']))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    img_paths['wordcloud'] = os.path.join(RESULTS_DIR, "4_wordcloud.png")
    plt.savefig(img_paths['wordcloud'])
    plt.close()

    # 5. Correlation 
    print("Calculating correlations...")
    corr_val = df['sentiment'].corr(df['label'])
    
    # 6. Shape of combined features
    print("Building features...")
    tfidf = TfidfVectorizer(max_features=100)
    X_tfidf = tfidf.fit_transform(df['text']).toarray()
    X_combined = np.hstack([X_tfidf, df[['sentiment']].values])
    initial_shape = X_combined.shape
    y = df['label'].values

    # Splitting 70, 15, 15
    print("Splitting dataset 70% / 15% / 15%...")
    X_train, X_temp, y_train, y_temp = train_test_split(X_combined, y, test_size=0.30, random_state=42, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp)

    # Apply SMOTE
    print("Applying SMOTE...")
    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
    smote_shape = X_train_res.shape

    # Plot SMOTE distribution
    plt.figure(figsize=(8, 6))
    sns.countplot(x=y_train_res, palette='coolwarm')
    plt.title("Training Set Distribution After SMOTE")
    plt.xticks([0, 1], ["Not Potential", "Stress Potential"])
    img_paths['smote_dist'] = os.path.join(RESULTS_DIR, "smote_dist.png")
    plt.savefig(img_paths['smote_dist'])
    plt.close()

    # Train Logistic Regression
    print("Training Logistic Regression...")
    lr_model = LogisticRegression(random_state=42, max_iter=1000)
    lr_model.fit(X_train_res, y_train_res)

    # 7. Confusion Matrix
    y_pred = lr_model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=["Not Potential", "Stress Potential"], yticklabels=["Not Potential", "Stress Potential"])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix (Logistic Regression)')
    img_paths['conf_matrix'] = os.path.join(RESULTS_DIR, "7_conf_matrix.png")
    plt.savefig(img_paths['conf_matrix'])
    plt.close()

    # 8-9 Classification report
    acc = accuracy_score(y_test, y_pred)
    cr = classification_report(y_test, y_pred, target_names=["Not Potential", "Stress Potential"])

    # Create Markdown Report
    print("Writing markdown report...")
    md_content = f"""# Stress Potential Analysis and Machine Learning Pipeline Report

## 0. Dataset Flags
**Sentiment data flags** mapping in this dataset:
- `1`: **Stress Potential (Positif)**
- `0`: **Not Potential**

## 1. Sentiment Distribution
The distribution of text sentiment polarities analyzed via TextBlob.
![Sentiment Distribution]({img_paths['sentiment_dist']})

## 2. Dataset Distribution
Class distribution of Stress Potential vs Not Potential before balancing:
![Dataset Distribution]({img_paths['dataset_dist']})

## 3. Heatmap After Dimensionality Reduction (PCA)
Using PCA to extract 10 components from the numerical metadata features:
![PCA Heatmap]({img_paths['pca_heatmap']})

## 4. WordCloud
Most frequent terms found across all text samples:
![WordCloud]({img_paths['wordcloud']})

## 5. Correlation
Pearsons Correlation between TextBlob Sentiment Score and Actual Stress Potential Label: **{corr_val:.4f}**

## 6. Shape of Combined Features
- Original feature shape (TF-IDF 100 + Sentiment 1): **{initial_shape}**
- Training, Validation, and Test Split performed: **70% / 15% / 15%**
- Shape of the Training Set *after* applying SMOTE for class balancing: **{smote_shape}**
![SMOTE Balance Distribution]({img_paths['smote_dist']})

## 7. Confusion Matrix (Logistic Regression)
Performance on the 15% Test set:
![Confusion Matrix]({img_paths['conf_matrix']})

## 8. Learning Rate & Pipeline Params
- **Vectorization**: TF-IDF (100 features)
- **Oversampling**: SMOTE (on combined continuous features)
- **SMOTE Random State**: 42
- **Model**: Logistic Regression
- **Optimization Strategy**: Default LBFGS solver, max_iterations=1000.
*(Note: standard Logistic Regression does not use a direct 'learning rate' parameter like deep learning models, but converges via exact gradient optimization).*

## 9. Classification Report (Logistic Regression Model)
**Accuracy**: {acc:.4f}

```text
{cr}
```
"""

    with open("Validation_Report.md", "w") as f:
        f.write(md_content)
    
    print("Done! Report saved to Validation_Report.md")

if __name__ == "__main__":
    generate_report()

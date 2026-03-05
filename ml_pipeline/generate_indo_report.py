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
from src.data.entda_augmentation import augment_with_entda
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Paths
DATA_PATH = "data/raw/Vibree_indonesian_stress_dataset.csv"
RESULTS_DIR = "indo_results"
MODEL_DIR = "models/inference_models"
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

def preprocess_basic(df):
    df['text'] = df['text'].astype(str)
    return df

def generate_report():
    print("Loading data...")
    df = pd.read_csv(DATA_PATH)
    df = preprocess_basic(df)

    img_paths = {}

    df['label_name'] = df['label'].map({1: "Stress Potential", 0: "Not Potential"})

    # 1. Sentiment Distribution
    print("Calculating Sentiment Distribution...")
    df['sentiment'] = df['text'].apply(lambda x: TextBlob(x).sentiment.polarity)
    plt.figure(figsize=(10, 6))
    df['sentiment'].hist(bins=50, color='skyblue', edgecolor='black')
    plt.title("Sentiment Distribution")
    plt.xlabel("Sentiment Polarity")
    plt.ylabel("Frequency")
    img_paths['sentiment_dist'] = os.path.join(RESULTS_DIR, "indo_1_sentiment_dist.png")
    plt.savefig(img_paths['sentiment_dist'])
    plt.close()

    # 2. Dataset Distribution
    print("Calculating Dataset Distribution...")
    plt.figure(figsize=(8, 6))
    sns.countplot(x=df['label_name'], palette='coolwarm')
    plt.title("Initial Dataset Distribution")
    img_paths['dataset_dist'] = os.path.join(RESULTS_DIR, "indo_2_dataset_dist.png")
    plt.savefig(img_paths['dataset_dist'])
    plt.close()

    # 3. Heatmap After PCA
    print("Generating PCA Heatmap...")
    numeric_df = df.select_dtypes(include=['float64', 'int64']).drop(columns=['label', 'id', 'post_id'], errors='ignore')
    pca = PCA(n_components=min(10, numeric_df.shape[1]))
    reduced_data = pca.fit_transform(numeric_df.fillna(0))
    reduced_corr = pd.DataFrame(reduced_data).corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(reduced_corr, cmap="coolwarm", annot=True, fmt=".2f")
    plt.title("Heatmap After Dimensionality Reduction (PCA)")
    img_paths['pca_heatmap'] = os.path.join(RESULTS_DIR, "indo_3_pca_heatmap.png")
    plt.savefig(img_paths['pca_heatmap'])
    plt.close()

    # 4. WordCloud
    print("Generating WordCloud...")
    plt.figure(figsize=(12, 10))
    wordcloud = WordCloud(background_color='white', width=800, height=800).generate(' '.join(df['text']))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    img_paths['wordcloud'] = os.path.join(RESULTS_DIR, "indo_4_wordcloud.png")
    plt.savefig(img_paths['wordcloud'])
    plt.close()

    # 5. Correlation 
    print("Calculating correlations...")
    corr_val = df['sentiment'].corr(df['label'])
    
    # Splitting 70, 15, 15 Before EnTDA and TF-IDF
    print("Splitting dataset 70% / 15% / 15%...")
    train_texts, temp_texts, y_train, y_temp = train_test_split(df['text'].tolist(), df['label'].values, test_size=0.30, random_state=42, stratify=df['label'].values)
    val_texts, test_texts, y_val, y_test = train_test_split(temp_texts, y_temp, test_size=0.50, random_state=42, stratify=y_temp)

    # 6. Apply EnTDA
    print("Applying EnTDA (Entity-to-Text Based Augmentation)...")
    aug_train_texts, y_train_res = augment_with_entda(train_texts, y_train, target_balance=True)

    print("Building features...")
    tfidf = TfidfVectorizer(max_features=100)
    
    X_train_tfidf = tfidf.fit_transform(aug_train_texts).toarray()
    X_test_tfidf = tfidf.transform(test_texts).toarray()
    
    # Calculate sentiment for new texts (train only as needed to match shape, but simpler is just extracting again)
    train_sentiment = np.array([TextBlob(t).sentiment.polarity for t in aug_train_texts]).reshape(-1, 1)
    test_sentiment = np.array([TextBlob(t).sentiment.polarity for t in test_texts]).reshape(-1, 1)
    
    X_train_res = np.hstack([X_train_tfidf, train_sentiment])
    X_test_res = np.hstack([X_test_tfidf, test_sentiment])
    
    entda_shape = X_train_res.shape

    # Plot EnTDA distribution
    plt.figure(figsize=(8, 6))
    sns.countplot(x=y_train_res, palette='coolwarm')
    plt.title("Training Set Distribution After EnTDA")
    plt.xticks([0, 1], ["Not Potential", "Stress Potential"])
    img_paths['entda_dist'] = os.path.join(RESULTS_DIR, "indo_entda_dist.png")
    plt.savefig(img_paths['entda_dist'])
    plt.close()

    # Train Logistic Regression
    print("Training Logistic Regression...")
    lr_model = LogisticRegression(random_state=42, max_iter=1000)
    lr_model.fit(X_train_res, y_train_res)

    # Save trained LR model & TFIDF
    import joblib
    joblib.dump(lr_model, os.path.join(MODEL_DIR, "indo_logistic_regression.joblib"))
    joblib.dump(tfidf, os.path.join(MODEL_DIR, "indo_tfidf_vectorizer.joblib"))

    # 7. Confusion Matrix
    y_pred = lr_model.predict(X_test_res)
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=["Not Potential", "Stress Potential"], yticklabels=["Not Potential", "Stress Potential"])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix (Logistic Regression)')
    img_paths['conf_matrix'] = os.path.join(RESULTS_DIR, "indo_7_conf_matrix.png")
    plt.savefig(img_paths['conf_matrix'])
    plt.close()

    # 8-9 Classification report
    acc = accuracy_score(y_test, y_pred)
    cr = classification_report(y_test, y_pred, target_names=["Not Potential", "Stress Potential"])

    # Create Markdown Report
    print("Writing markdown report...")
    md_content = f"""# Indonesian Stress Potential Analysis (EnTDA Augmented)
    
## 0. Dataset Flags
**Sentiment data flags** mapping in this dataset:
- `1`: **Stress Potential (Positif)**
- `0`: **Not Potential**

## 1. Sentiment Distribution
The distribution of text sentiment polarities analyzed via TextBlob.
![Sentiment Distribution](/Users/riadizur/.gemini/antigravity/brain/a5cf41f8-124c-482a-9adc-7c67d004bf97/indo_1_sentiment_dist.png)

## 2. Dataset Distribution
Class distribution of Stress Potential vs Not Potential before balancing:
![Dataset Distribution](/Users/riadizur/.gemini/antigravity/brain/a5cf41f8-124c-482a-9adc-7c67d004bf97/indo_2_dataset_dist.png)

## 3. Heatmap After Dimensionality Reduction (PCA)
Using PCA to extract components from the numerical metadata features:
![PCA Heatmap](/Users/riadizur/.gemini/antigravity/brain/a5cf41f8-124c-482a-9adc-7c67d004bf97/indo_3_pca_heatmap.png)

## 4. WordCloud
Most frequent terms found across all text samples:
![WordCloud](/Users/riadizur/.gemini/antigravity/brain/a5cf41f8-124c-482a-9adc-7c67d004bf97/indo_4_wordcloud.png)

## 5. Correlation
Pearsons Correlation between TextBlob Sentiment Score and Actual Stress Potential Label: **{corr_val:.4f}**

## 6. Shape of Combined Features
- Training, Validation, and Test Split performed: **70% / 15% / 15%**
- Shape of the Training Set *after* applying **EnTDA (Entity-to-Text)** class balancing via Contextual Embeddings: **{entda_shape}**
![EnTDA Balance Distribution](/Users/riadizur/.gemini/antigravity/brain/a5cf41f8-124c-482a-9adc-7c67d004bf97/indo_entda_dist.png)

## 7. Confusion Matrix (Logistic Regression)
Performance on the 15% Test set:
![Confusion Matrix](/Users/riadizur/.gemini/antigravity/brain/a5cf41f8-124c-482a-9adc-7c67d004bf97/indo_7_conf_matrix.png)

## 8. Learning Rate & Pipeline Params
- **Vectorization**: TF-IDF (100 features) + Sentiment
- **Oversampling**: Entity-to-Text Based Augmentation (EnTDA) on Raw Texts
- **Model**: Logistic Regression
- **Optimization Strategy**: Default LBFGS solver, max_iterations=1000.
*(All inference models including Transformers point to `<pipeline>/models/inference_models/` folder).*

## 9. Classification Report (Logistic Regression Model)
**Accuracy**: {acc:.4f}

```text
{cr}
```
"""

    with open(os.path.join(RESULTS_DIR, "Indo_Validation_Report.md"), "w") as f:
        f.write(md_content)
    
    print("Done! Report saved to Indo_Validation_Report.md")

if __name__ == "__main__":
    generate_report()

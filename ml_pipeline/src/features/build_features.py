import numpy as np
import pandas as pd
from textblob import TextBlob
from sklearn.feature_extraction.text import TfidfVectorizer

def get_sentiment(text):
    if not isinstance(text, str) or not text.strip():
        return 0.0
    return TextBlob(text).sentiment.polarity

def build_features(df, max_features=100):
    """
    Generates combined feature matrix using TF-IDF and Sentiment Analysis,
    and returns sentiment-based labels to replicate the notebook.
    """
    # 1. TF-IDF feature extraction
    tfidf = TfidfVectorizer(max_features=max_features)
    text_features = tfidf.fit_transform(df['text']).toarray()
    
    # 2. Sentiment calculation
    sentiments = df['text'].apply(get_sentiment).values.reshape(-1, 1)
    
    # 3. Combine TF-IDF and Sentiment features
    combined_features = np.hstack([text_features, sentiments])
    
    # 4. Target variable (Sentiment labels as defined in notebook logic)
    # Notebook: labels negative sentiments (<0) as 0, positive as 1
    labels = np.where(sentiments.flatten() >= 0, 1, 0)
    
    return combined_features, labels, tfidf

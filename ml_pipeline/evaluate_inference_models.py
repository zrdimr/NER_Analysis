import os
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import classification_report, accuracy_score
import warnings
warnings.filterwarnings('ignore')

from src.models.predict_transformer import TransformerPredictor

# Data
DATA_PATH = "data/raw/Vibree_indonesian_stress_dataset.csv"
MODEL_DIR = "models/inference_models"

print("Loading test data (Using last 500 rows as mock evaluation set)...")
df = pd.read_csv(DATA_PATH)
df['text'] = df['text'].astype(str)
test_df = df.tail(500)
texts = test_df['text'].tolist()
true_labels = test_df['label'].values

print(f"\n--- EVALUATING LOGISTIC REGRESSION ---")
lr_path = os.path.join(MODEL_DIR, "indo_logistic_regression.joblib")
tfidf_path = os.path.join(MODEL_DIR, "indo_tfidf_vectorizer.joblib")

if os.path.exists(lr_path) and os.path.exists(tfidf_path):
    lr_model = joblib.load(lr_path)
    tfidf = joblib.load(tfidf_path)
    
    from textblob import TextBlob
    X_tfidf = tfidf.transform(texts).toarray()
    sentiments = np.array([TextBlob(t).sentiment.polarity for t in texts]).reshape(-1, 1)
    X_res = np.hstack([X_tfidf, sentiments])
    
    y_pred = lr_model.predict(X_res)
    print(f"Accuracy: {accuracy_score(true_labels, y_pred):.4f}")
    print(classification_report(true_labels, y_pred, target_names=["Not Potential", "Stress Potential"]))
else:
    print("Logistic Regression model non-existent.")

# Transformers
models = ['mobilebert', 'bert', 'indobert', 'mentalbert']
for m in models:
    print(f"\n--- EVALUATING {m.upper()} ---")
    model_path = os.path.join(MODEL_DIR, m)
    if os.path.exists(model_path):
        import torch
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        # To speed up inference:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForSequenceClassification.from_pretrained(model_path)
        model.eval()
        
        preds = []
        batch_size = 16
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            encodings = tokenizer(batch_texts, truncation=True, padding=True, max_length=128, return_tensors="pt")
            with torch.no_grad():
                outputs = model(**encodings)
                batch_preds = outputs.logits.argmax(dim=-1).cpu().numpy()
                preds.extend(batch_preds)
                
        print(f"Accuracy: {accuracy_score(true_labels, preds):.4f}")
        print(classification_report(true_labels, preds, target_names=["Not Potential", "Stress Potential"]))
    else:
        print(f"Model {m} largely not trained yet or not saved in {model_path}.")

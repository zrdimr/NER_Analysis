import joblib
import numpy as np
from src.data.preprocess import preprocess_text
from src.features.build_features import get_sentiment

class StressPredictor:
    def __init__(self, model_path, tfidf_path, scaler_path):
        """
        Loads the trained model, TF-IDF vectorizer, and scaler.
        """
        self.model = joblib.load(model_path)
        self.tfidf = joblib.load(tfidf_path)
        self.scaler = joblib.load(scaler_path)
        
    def predict(self, texts):
        if isinstance(texts, str):
            texts = [texts]
            
        predictions = []
        for text in texts:
            # 1. Preprocess
            clean_text = preprocess_text(text)
            
            # 2. Extract features
            text_features = self.tfidf.transform([clean_text]).toarray()
            sentiment = get_sentiment(clean_text)
            combined_features = np.hstack([text_features, np.array([[sentiment]])])
            
            # 3. Scale features
            combined_features_scaled = self.scaler.transform(combined_features)
            
            # 4. Predict
            pred = self.model.predict(combined_features_scaled)[0]
            label = "Positive (Not Stressed/Healthy)" if pred == 1 else "Negative (Stressed/At Risk)"
            predictions.append((label, pred))
            
        return predictions

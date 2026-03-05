import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

class TransformerPredictor:
    def __init__(self, model_dir):
        """
        Loads the trained transformer model and tokenizer.
        """
        self.device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_dir)
        self.model.to(self.device)
        self.model.eval()

    def predict(self, texts, max_length=128):
        if isinstance(texts, str):
            texts = [texts]
            
        encodings = self.tokenizer(texts, truncation=True, padding=True, max_length=max_length, return_tensors='pt')
        encodings = {k: v.to(self.device) for k, v in encodings.items()}
        
        with torch.no_grad():
            outputs = self.model(**encodings)
            logits = outputs.logits
            preds = torch.argmax(logits, dim=-1).cpu().numpy()
            
        predictions = []
        for pred in preds:
            label = "Positive (Not Stressed/Healthy)" if pred == 1 else "Negative (Stressed/At Risk)"
            predictions.append((label, pred))
            
        return predictions

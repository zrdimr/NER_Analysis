import os
import torch
import numpy as np
import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer, Trainer, TrainingArguments
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from src.models.custom_architectures import TransformerBasic, TransformerLSTM, TransformerLSTMCRF
from src.data.entda_augmentation import augment_with_entda

MODEL_MAP = {
    "bert": "bert-base-uncased",
    "mobilebert": "google/mobilebert-uncased",
    "indobert": "indobenchmark/indobert-base-p1",
    "mentalbert": "mental/mental-bert-base-uncased"
}

def compute_metrics(pred):
    labels = pred.label_ids
    # For custom custom architecture return tuple, `pred.predictions` might be a tuple. 
    preds = pred.predictions[0].argmax(-1) if isinstance(pred.predictions, tuple) else pred.predictions.argmax(-1)
    
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary', zero_division=0)
    acc = accuracy_score(labels, preds)
    return {'accuracy': float(acc), 'f1': float(f1), 'precision': float(precision), 'recall': float(recall)}


class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.get("labels")
        input_ids = inputs.get("input_ids")
        attention_mask = inputs.get("attention_mask")
        
        if hasattr(model, 'crf'):
            outputs = model(input_ids, attention_mask, labels=labels)
            loss = outputs[0]
            logits = outputs[1]
        else:
            outputs = model(input_ids, attention_mask)
            
            # Huggingface models return SequenceClassifierOutput by default. Custom models return raw tensor.
            if hasattr(outputs, "logits"):
                logits = outputs.logits
            elif isinstance(outputs, tuple):
                logits = outputs[0]
            else:
                logits = outputs
                
            loss_fct = torch.nn.CrossEntropyLoss()
            loss_feature_dims = model.classifier.out_features if hasattr(model, 'classifier') else 2
            
            # Use contiguous array to avoid NoneType shape errors on View
            loss = loss_fct(logits.contiguous().view(-1, loss_feature_dims), labels.contiguous().view(-1))
            
            # Wrap standard output logic back to tuple format so Huggingface Trainer `outputs[1:]` doesnt truncate tensor data length
            outputs = (loss, logits)
            
        return (loss, outputs) if return_outputs else loss


def train_and_eval_generic(dataset_name, model_base, architecture, apply_entda=False, epochs=3, batch_size=8, lr=2e-5):
    # Paths based on dataset wrapper
    base_data = "data/processed"
    train_path = os.path.join(base_data, f"{dataset_name}_train_processed.csv")
    test_path = os.path.join(base_data, f"{dataset_name}_test_processed.csv")
    
    if not os.path.exists(train_path):
        raise ValueError(f"Training dataset {train_path} does not exist.")
        
    df_train = pd.read_csv(train_path).dropna()
    df_test = pd.read_csv(test_path).dropna()
    
    train_texts, train_labels = df_train['text'].astype(str).tolist(), df_train['label'].astype(int).tolist()
    test_texts, test_labels = df_test['text'].astype(str).tolist(), df_test['label'].astype(int).tolist()
    
    if apply_entda:
        train_texts, train_labels = augment_with_entda(train_texts, train_labels, target_balance=True)
        
    hf_model_name = MODEL_MAP.get(model_base.lower())
    if not hf_model_name:
        raise ValueError(f"Unknown model_base {model_base}.")
        
    tokenizer = AutoTokenizer.from_pretrained(hf_model_name)
    max_length = 128
    
    train_encodings = tokenizer(train_texts, truncation=True, padding='max_length', max_length=max_length)
    test_encodings = tokenizer(test_texts, truncation=True, padding='max_length', max_length=max_length)
    
    class CustomTorchDataset(torch.utils.data.Dataset):
        def __init__(self, encodings, labels):
            self.encodings = encodings
            self.labels = labels

        def __getitem__(self, idx):
            item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
            item['labels'] = torch.tensor(self.labels[idx])
            return item

        def __len__(self):
            return len(self.labels)

    train_ds = CustomTorchDataset(train_encodings, train_labels)
    test_ds = CustomTorchDataset(test_encodings, test_labels)
    
    # Init Model
    if architecture == "transformer":
        model = TransformerBasic(hf_model_name)
    elif architecture == "transformer_lstm":
        model = TransformerLSTM(hf_model_name)
    elif architecture == "transformer_lstm_crf":
        model = TransformerLSTMCRF(hf_model_name)
    else:
        raise ValueError(f"Unknown Custom Architecture {architecture}")
        
    save_dir = f"models/inference_models/{dataset_name}/{'entda' if apply_entda else 'no_entda'}/{model_base}_{architecture}"
    os.makedirs(save_dir, exist_ok=True)
    
    training_args = TrainingArguments(
        output_dir=save_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=lr,
        warmup_steps=10,
        weight_decay=0.01,
        logging_dir='./logs',
        eval_strategy="no",
        save_strategy="no",
        report_to="none"
    )
    
    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=test_ds,
        compute_metrics=compute_metrics
    )
    
    print(f"Training {model_base} with {architecture} on {dataset_name} (EnTDA={apply_entda})...")
    train_results = trainer.train()
    training_time = train_results.metrics.get("train_runtime", 0) if hasattr(train_results, 'metrics') else 0
    
    eval_results = trainer.evaluate()
    
    # Save the custom model explicitly
    torch.save(model.state_dict(), os.path.join(save_dir, "model_weights.pth"))
    tokenizer.save_pretrained(save_dir)
    
    # Calculate folder size
    size_mb = sum(os.path.getsize(os.path.join(dirpath, f)) for dirpath, _, files in os.walk(save_dir) for f in files) / (1024 * 1024)
    
    eval_results["training_time"] = training_time
    eval_results["model_size_mb"] = size_mb
    eval_results["jumlah_dataset"] = len(train_labels)
    eval_results["positif_dataset"] = sum(1 for x in train_labels if x == 1)
    eval_results["negatif_dataset"] = sum(1 for x in train_labels if x == 0)
    
    return eval_results

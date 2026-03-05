import os
import torch
import numpy as np
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

MODEL_MAP = {
    "bert": "bert-base-uncased",
    "mobilebert": "google/mobilebert-uncased",
    "indobert": "indobenchmark/indobert-base-p1",
    "mentalbert": "mental/mental-bert-base-uncased"
}

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

def train_transformer(texts, labels, model_type, config):
    hf_model_name = MODEL_MAP.get(model_type.lower())
    if not hf_model_name:
        raise ValueError(f"Unknown model type {model_type}. Options: {list(MODEL_MAP.keys())}")

    # Split dataset
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        texts, labels,
        test_size=config['model'].get('test_size', 0.2),
        random_state=config['model'].get('random_state', 42)
    )

    from src.data.entda_augmentation import augment_with_entda
    train_texts, train_labels = augment_with_entda(train_texts, train_labels, target_balance=True)


    tokenizer = AutoTokenizer.from_pretrained(hf_model_name)
    model = AutoModelForSequenceClassification.from_pretrained(hf_model_name, num_labels=2)

    # Tokenizer settings
    t_config = config.get('transformer', {})
    max_length = t_config.get('max_length', 128)
    
    train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=max_length)
    val_encodings = tokenizer(val_texts, truncation=True, padding=True, max_length=max_length)

    class CustomDataset(torch.utils.data.Dataset):
        def __init__(self, encodings, labels):
            self.encodings = encodings
            self.labels = labels

        def __getitem__(self, idx):
            item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
            item['labels'] = torch.tensor(self.labels[idx])
            return item

        def __len__(self):
            return len(self.labels)

    train_dataset = CustomDataset(train_encodings, train_labels)
    val_dataset = CustomDataset(val_encodings, val_labels)

    # Training arguments
    epochs = t_config.get('epochs', 3)
    batch_size = t_config.get('batch_size', 8)
    lr = t_config.get('learning_rate', 2e-5)
    save_dir = t_config.get('model_save_dir', 'models/transformer')

    training_args = TrainingArguments(
        output_dir=save_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=lr,
        warmup_steps=10,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=10,
        eval_strategy="epoch",  # `evaluation_strategy` is deprecated, use `eval_strategy`
        save_strategy="no",
        load_best_model_at_end=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics
    )

    trainer.train()
    
    # Evaluate
    eval_results = trainer.evaluate()

    # Save best model to distinct folders per model_type inside inference_models
    final_save_path = os.path.join(save_dir, model_type.lower())
    os.makedirs(final_save_path, exist_ok=True)
    model.save_pretrained(final_save_path)
    tokenizer.save_pretrained(final_save_path)

    return model, tokenizer, eval_results

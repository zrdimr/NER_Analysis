import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig
from torchcrf import CRF

class TransformerBasic(nn.Module):
    def __init__(self, model_name, num_labels=2):
        super().__init__()
        self.config = AutoConfig.from_pretrained(model_name)
        self.transformer = AutoModel.from_pretrained(model_name, config=self.config)
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(self.config.hidden_size, num_labels)
        
    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.transformer(input_ids=input_ids, attention_mask=attention_mask)
        # Use simple CLS token pooler output or mean pooling
        if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
            pooled_output = outputs.pooler_output
        else:
            pooled_output = outputs.last_hidden_state[:, 0]
            
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return logits

class TransformerLSTM(nn.Module):
    def __init__(self, model_name, num_labels=2, lstm_hidden_size=256):
        super().__init__()
        self.config = AutoConfig.from_pretrained(model_name)
        self.transformer = AutoModel.from_pretrained(model_name, config=self.config)
        
        self.lstm = nn.LSTM(
            input_size=self.config.hidden_size, 
            hidden_size=lstm_hidden_size, 
            num_layers=1, 
            batch_first=True, 
            bidirectional=True
        )
        
        self.dropout = nn.Dropout(0.1)
        # Bidirectional means output size is 2 * lstm_hidden_size
        self.classifier = nn.Linear(lstm_hidden_size * 2, num_labels)
        
    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.transformer(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state
        
        lstm_output, (h_n, c_n) = self.lstm(sequence_output)
        
        # Get the representation of the CLS token / first timestep after LSTM
        lstm_pooled = lstm_output[:, 0, :]
        lstm_pooled = self.dropout(lstm_pooled)
        
        logits = self.classifier(lstm_pooled)
        return logits

class TransformerLSTMCRF(nn.Module):
    def __init__(self, model_name, num_labels=2, lstm_hidden_size=256):
        super().__init__()
        self.num_labels = num_labels
        self.config = AutoConfig.from_pretrained(model_name)
        self.transformer = AutoModel.from_pretrained(model_name, config=self.config)
        
        self.lstm = nn.LSTM(
            input_size=self.config.hidden_size, 
            hidden_size=lstm_hidden_size, 
            num_layers=1, 
            batch_first=True, 
            bidirectional=True
        )
        self.dropout = nn.Dropout(0.1)
        self.linear = nn.Linear(lstm_hidden_size * 2, num_labels)
        
        # CRF layer designed for token classification, but here we enforce sequence-level tags
        # by replicating the sentence label across all tokens during training for consistency,
        # and aggregating them.
        self.crf = CRF(num_labels, batch_first=True)
        
    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.transformer(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state
        
        lstm_output, _ = self.lstm(sequence_output)
        lstm_output = self.dropout(lstm_output)
        
        emissions = self.linear(lstm_output)
        
        if labels is not None:
            # Replicate the sentence-level label across the sequence to train the CRF
            # Labels shape: (batch_size,) -> (batch_size, seq_len)
            seq_len = emissions.size(1)
            expanded_labels = labels.unsqueeze(1).expand(-1, seq_len).clone()
            
            # CRF needs byte-type or bool-type attention mask for PyTorch > 1.0 depending on version
            mask = attention_mask.bool()
            
            # Loss is negative log-likelihood
            log_likelihood = self.crf(emissions, expanded_labels, mask=mask, reduction='mean')
            # Return tuple for Trainer compatibility (loss, logits)
            
            # For logits, simply return the emissions of the CLS token to match SequenceClassification format
            return -log_likelihood, emissions[:, 0, :]
        else:
            mask = attention_mask.bool()
            best_paths = self.crf.decode(emissions, mask=mask)
            
            # To interface cleanly with classification logic, we reconstruct 2D tensor logits.
            # We take the predicted label at the first token (CLS) across the batch.
            # Convert list of lists to a tensor of predictions
            preds = [path[0] for path in best_paths]
            pred_tensor = torch.tensor(preds, device=emissions.device)
            
            # Create dummy logits that enforce the predicted argmax
            dummy_logits = torch.zeros((emissions.size(0), self.num_labels), device=emissions.device)
            dummy_logits.scatter_(1, pred_tensor.unsqueeze(1), 10.0)
            
            return dummy_logits

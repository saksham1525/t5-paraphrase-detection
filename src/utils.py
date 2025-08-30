import torch
from torch.utils.data import Dataset, DataLoader
from transformers import T5Tokenizer, T5ForConditionalGeneration, get_scheduler
from torch.optim import AdamW
from datasets import load_dataset
from sklearn.metrics import accuracy_score, f1_score
import json


class TextDataset(Dataset):
    """Dataset class for MRPC text equivalence classification tasks"""
    
    def __init__(self, data, tokenizer, max_input_length=128, max_target_length=16):
        self.data = data
        self.tokenizer = tokenizer
        self.max_input_length = max_input_length
        self.max_target_length = max_target_length
        
    def __len__(self):
        return len(self.data)
    
    def tokenize(self, text, max_length):
        """Tokenize text with specified maximum length and padding"""
        return self.tokenizer(text, max_length=max_length, truncation=True,
                              padding='max_length', return_tensors='pt')
    
    def __getitem__(self, idx):
        item = self.data[idx]        
        prompt_text = f"are these sentences equivalent? answer 1 for yes, 0 for no: sentence1: {item['sentence1']} sentence2: {item['sentence2']}"
        target_text = "1" if item['label'] == 1 else "0"        
        tokenized_inputs = self.tokenize(prompt_text, self.max_input_length)
        tokenized_targets = self.tokenize(target_text, self.max_target_length)        
        return {
            'input_ids': tokenized_inputs['input_ids'].squeeze(),
            'attention_mask': tokenized_inputs['attention_mask'].squeeze(),
            'labels': tokenized_targets['input_ids'].squeeze()
        }


def classify_output(tokenizer, prediction_ids):
    """Convert T5 model output tokens to binary classification (1, 0, or -1 for ambiguous)"""
    output_text = tokenizer.decode(prediction_ids, skip_special_tokens=True).strip()
    return 1 if output_text == "1" else 0 if output_text == "0" else -1


def load_mrpc_dataset():
    """Load MRPC dataset from GLUE benchmark and return train/validation/test splits"""
    dataset = load_dataset("glue", "mrpc")
    return dataset["train"], dataset["validation"], dataset["test"]


def create_dataloaders(train_data, validation_data, tokenizer, batch_size=8):
    """Create PyTorch DataLoaders for training and validation datasets"""
    train_dataset = TextDataset(train_data, tokenizer)
    validation_dataset = TextDataset(validation_data, tokenizer)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    validation_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, validation_loader


def initialize_model_and_tokenizer(device):
    """Initialize T5-small model and tokenizer, move model to specified device"""
    tokenizer = T5Tokenizer.from_pretrained("t5-small", legacy=True)
    model = T5ForConditionalGeneration.from_pretrained("t5-small")
    return model.to(device), tokenizer


def train_single_epoch(model, dataloader, optimizer, scheduler, device):
    """Train model for one epoch and return average loss"""
    model.train()
    total_epoch_loss = 0
    
    for batch in dataloader:
        batch = {key: value.to(device) for key, value in batch.items()}
        
        model_outputs = model(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask'],
            labels=batch['labels']
        )
        
        batch_loss = model_outputs.loss
        total_epoch_loss += batch_loss.item()
        
        # Backward pass with gradient clipping
        batch_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
    
    return total_epoch_loss / len(dataloader)


def validate_model(model, dataloader, tokenizer, device):
    """Validate model and return average loss, accuracy, and F1 score"""
    model.eval()
    all_predictions, all_references = [], []
    total_validation_loss = 0
    
    with torch.no_grad():
        for batch in dataloader:
            batch = {key: value.to(device) for key, value in batch.items()}
            
            model_outputs = model(**batch)
            total_validation_loss += model_outputs.loss.item()
            
            # Generate predictions (single token outputs)
            generated_tokens = model.generate(
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask'],
                max_new_tokens=1, do_sample=False,
                pad_token_id=tokenizer.pad_token_id
            )
            
            predicted_texts = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
            reference_texts = tokenizer.batch_decode(batch['labels'], skip_special_tokens=True)
            
            # Convert text outputs to binary labels
            predicted_labels = [classify_output(tokenizer, 
                    tokenizer(pred_text, return_tensors='pt')['input_ids'][0])
                   for pred_text in predicted_texts]
            reference_labels = [1 if ref_text.strip() == "1" else 0 for ref_text in reference_texts]
            
            # Filter out ambiguous predictions (where model output is neither "0" nor "1")
            valid_prediction_pairs = [(pred, ref) for pred, ref in zip(predicted_labels, reference_labels) if pred != -1]
            if valid_prediction_pairs:
                valid_predictions, valid_references = zip(*valid_prediction_pairs)
                all_predictions.extend(valid_predictions)
                all_references.extend(valid_references)
    
    average_loss = total_validation_loss / len(dataloader)
    accuracy = accuracy_score(all_references, all_predictions)
    f1_score_value = f1_score(all_references, all_predictions, zero_division='warn')
    
    return average_loss, accuracy, f1_score_value


def save_results(results_data, output_file_path):
    """Save training results dictionary to JSON file"""
    with open(output_file_path, 'w') as file_handle:
        json.dump(results_data, file_handle, indent=2)
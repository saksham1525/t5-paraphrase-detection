import torch
from torch.utils.data import DataLoader, Dataset
from transformers import T5Tokenizer, T5ForConditionalGeneration, get_scheduler
from torch.optim import AdamW
from datasets import load_dataset
from sklearn.metrics import accuracy_score, f1_score

from tqdm import tqdm
import json
import warnings
from training_logger import TrainingLogger
from training_plots import generate_training_curves
from benchmark_comparison import compare_with_bert_benchmark

# Suppress known warnings for cleaner output
warnings.filterwarnings("ignore", category=RuntimeWarning, message="Mean of empty slice")
warnings.filterwarnings("ignore", category=RuntimeWarning, message="invalid value encountered in scalar divide")
warnings.filterwarnings("ignore", message="F-score is ill-defined and being set to 0.0 due to no true nor predicted samples")

class TextDataset(Dataset):
    def __init__(self, data, tokenizer, max_input_length=128, max_target_length=16):
        self.data = data
        self.tokenizer = tokenizer
        self.max_input_length = max_input_length
        self.max_target_length = max_target_length
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        input_text = f"are these sentences equivalent? answer 1 for yes, 0 for no: sentence1: {item['sentence1']} sentence2: {item['sentence2']}"
        target_text = "1" if item['label'] == 1 else "0"
        
        # Tokenize input
        inputs = self.tokenizer(
            input_text,
            max_length=self.max_input_length,
            truncation=True,
            padding='max_length',
            return_tensors='pt'
        )
        
        # Tokenize target
        targets = self.tokenizer(
            target_text,
            max_length=self.max_target_length,
            truncation=True,
            padding='max_length',
            return_tensors='pt'
        )
        
        return {
            'input_ids': inputs['input_ids'].squeeze(),
            'attention_mask': inputs['attention_mask'].squeeze(),
            'labels': targets['input_ids'].squeeze()
        }

def classify_output(tokenizer, prediction_ids):
    """Strict classification - only accepts exact '0' or '1' outputs"""
    output_text = tokenizer.decode(prediction_ids, skip_special_tokens=True).strip()
    
    if output_text == "1":
        return 1
    elif output_text == "0":
        return 0
    
    return -1  # unknown/ambiguous

class Trainer:
    def __init__(self):
        self.logger = TrainingLogger()
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.logger.info(f"Device: {self.device}")
        
        self.logger.info("Loading T5 model...")
        self.tokenizer = T5Tokenizer.from_pretrained("t5-small", legacy=True)
        self.model = T5ForConditionalGeneration.from_pretrained("t5-small")
        self.model.to(self.device)
        
        self.logger.info("Loading MRPC dataset...")
        dataset = load_dataset("glue", "mrpc")
        
        # Store full datasets separately
        self.full_train_data = dataset["train"]
        self.full_val_data = dataset["validation"] 
        self.test_data = dataset["test"]  # For final evaluation if needed
        
        self.logger.info(f"Dataset loaded - Train: {len(self.full_train_data)}, Val: {len(self.full_val_data)}, Test: {len(self.test_data)} samples")
    
    def create_data_splits(self):
        """Create data splits - 10% for hyperparameter search, full dataset for final training"""
        
        total_train, total_val = len(self.full_train_data), len(self.full_val_data)
        
        # Use 10% for hyperparameter search
        search_train_samples = int(total_train * 0.10)
        search_val_samples = int(total_val * 0.10)
        
        self.logger.info(f"Data splits - Search: {search_train_samples}/{search_val_samples} (10%), Final: {total_train}/{total_val} (100%)")
        
        # Create search datasets (10% of data)
        self.search_train_data = self.full_train_data.select(range(search_train_samples))
        self.search_val_data = self.full_val_data.select(range(search_val_samples))
        
        return {
            'search_train': len(self.search_train_data),
            'search_val': len(self.search_val_data), 
            'final_train': len(self.full_train_data),
            'final_val': len(self.full_val_data)
        }
    
    def set_data_for_search(self):
        """Set current data to hyperparameter search datasets"""
        self.train_data = self.search_train_data
        self.val_data = self.search_val_data
        
    def set_data_for_final(self):
        """Set current data to final training datasets"""
        self.train_data = self.full_train_data
        self.val_data = self.full_val_data
    
    def create_dataloaders(self, batch_size=8):
        """Create dataloaders with custom dataset"""
        train_dataset = TextDataset(self.train_data, self.tokenizer)
        val_dataset = TextDataset(self.val_data, self.tokenizer)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        return train_loader, val_loader
    
    def train_epoch(self, dataloader, optimizer, scheduler):
        self.model.train()
        total_loss = 0
        
        for batch in dataloader:
            batch = {k: v.to(self.device) for k, v in batch.items()}
            
            # Forward pass
            outputs = self.model(
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask'],
                labels=batch['labels']
            )
            
            loss = outputs.loss
            total_loss += loss.item()
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping to prevent explosion
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
        
        return total_loss / len(dataloader)
    
    def validate(self, dataloader):
        self.model.eval()
        predictions = []
        references = []
        total_loss = 0
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader):
                batch = {k: v.to(self.device) for k, v in batch.items()}
                
                outputs = self.model(**batch)
                total_loss += outputs.loss.item()
                
                generated = self.model.generate(
                    input_ids=batch['input_ids'],
                    attention_mask=batch['attention_mask'],
                    max_new_tokens=1,  # Force single token only
                    do_sample=False,
                    pad_token_id=self.tokenizer.pad_token_id
                )
                
                pred_texts = self.tokenizer.batch_decode(generated, skip_special_tokens=True)
                ref_texts = self.tokenizer.batch_decode(batch['labels'], skip_special_tokens=True)
                
                # Convert to binary using safer classification
                batch_valid_count = 0
                for pred, ref in zip(pred_texts, ref_texts):
                    # Use tokenizer to get prediction IDs for classify_output
                    pred_tokens = self.tokenizer(pred, return_tensors='pt')['input_ids'][0]
                    pred_binary = classify_output(self.tokenizer, pred_tokens)
                    
                    ref_clean = ref.lower().strip()
                    ref_binary = 1 if ref_clean == "1" else 0
                    
                    # Only include predictions that are not ambiguous (-1)
                    if pred_binary != -1:
                        predictions.append(pred_binary)
                        references.append(ref_binary)
                        batch_valid_count += 1
        
        avg_loss = total_loss / len(dataloader)
        accuracy = accuracy_score(references, predictions)
        f1 = f1_score(references, predictions, zero_division='warn')
        
        # Log validation summary
        self.logger.info(f"Validation complete: {len(predictions)} valid predictions, F1={f1:.4f}, Acc={accuracy:.4f}")
        
        return avg_loss, accuracy, f1
    
    def train(self, lr=3e-5, batch_size=8, epochs=3, patience=1, min_delta=0.01, is_final=False):
        """Main training function with early stopping"""
        # Log training configuration to file
        self.logger.info(f"Training Configuration: LR={lr}, Batch={batch_size}, Epochs={epochs}, Patience={patience}, Min_delta={min_delta}")
        
        # Setup optimizer and scheduler
        optimizer = AdamW(self.model.parameters(), lr=lr, weight_decay=0.01)
        train_loader, val_loader = self.create_dataloaders(batch_size)
        
        total_steps = len(train_loader) * epochs
        scheduler = get_scheduler("linear", optimizer=optimizer, 
                                num_warmup_steps=100, num_training_steps=total_steps)
        
        self.logger.info(f"Training steps: {total_steps}, Batches per epoch: {len(train_loader)}")
        
        # Display batch info for final training
        if is_final:
            print(f"Batches per epoch: {len(train_loader)}")
        
        history = {"train_loss": [], "val_loss": [], "val_accuracy": [], "val_f1": []}
        best_f1 = 0
        best_val_loss = float('inf')
        patience_counter = 0
        
        # Training loop
        for epoch in range(epochs):
            train_loss = self.train_epoch(train_loader, optimizer, scheduler)
            val_loss, accuracy, f1 = self.validate(val_loader)
            
            # Show epoch results for final training
            if is_final:
                print(f"Epoch {epoch + 1}: F1={f1:.4f}, Accuracy={accuracy:.4f}")
            
            # Update history
            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history["val_accuracy"].append(accuracy)
            history["val_f1"].append(f1)
            
            # Log detailed epoch information
            self.logger.log_training_epoch(epoch + 1, epochs, train_loss, val_loss, f1, accuracy)
            
            if f1 > best_f1:
                best_f1 = f1
                self.logger.info(f"New best F1: {best_f1:.4f}")
            
            # Early stopping logic
            if val_loss < best_val_loss - min_delta:
                best_val_loss = val_loss
                patience_counter = 0
                self.logger.info(f"Validation loss improved to {best_val_loss:.4f}")
            else:
                patience_counter += 1
                self.logger.info(f"No improvement in validation loss (patience: {patience_counter}/{patience})")
                
                if patience_counter >= patience:
                    self.logger.log_early_stopping(epoch + 1, "No improvement in validation loss")
                    break
        
        actual_epochs = len(history["train_loss"])
        
        # Show final training completion
        if is_final:
            if actual_epochs < epochs:
                print(f"Training stopped early at epoch {actual_epochs}")
            print(f"Model training completed - Final F1: {best_f1:.4f}")
        
        self.logger.info(f"Training completed: {actual_epochs}/{epochs} epochs, F1: {best_f1:.4f}, Val loss: {best_val_loss:.4f}")
        
        return history, best_f1
    
    def hyperparameter_search(self, search_epochs=2, learning_rates=None, batch_sizes=None):
        """Search for best hyperparameters on separated dataset"""
        
        # Use search dataset (already separated)
        self.set_data_for_search()
        
        # Hyperparameter ranges
        if learning_rates is None:
            learning_rates = [2e-5, 3e-5, 4e-5]  
        if batch_sizes is None:
            batch_sizes = [8, 12, 16]
        
        self.logger.terminal_checkpoint("Starting hyperparameter search")
        self.logger.info(f"Search dataset: {len(self.train_data)} training, {len(self.val_data)} validation")
        self.logger.info(f"Learning rates: {learning_rates}, Batch sizes: {batch_sizes}, Epochs: {search_epochs}")
        
        total_combinations = len(learning_rates) * len(batch_sizes)
        self.logger.info(f"Total combinations: {total_combinations}")
        
        best_f1 = -0.1  # Accept any non-negative F1, even 0.0
        best_config = None
        search_results = []
        
        combination = 0
        for lr in learning_rates:
            for batch_size in batch_sizes:
                combination += 1
                
                # Simple counter update
                self.logger.log_hyperparameter_progress(combination, total_combinations, lr, batch_size)
                
                # Reinitialize model
                self.model = T5ForConditionalGeneration.from_pretrained("t5-small").to(self.device)
                
                # Limited training for hyperparameter search
                history, f1_score = self.train(lr=lr, batch_size=batch_size, epochs=search_epochs)
                
                final_accuracy = history['val_accuracy'][-1] if history.get('val_accuracy') else None
                result = {
                    'lr': lr, 'batch_size': batch_size,
                    'f1_score': f1_score, 'final_accuracy': final_accuracy
                }
                search_results.append(result)
                
                is_best = f1_score > best_f1
                if is_best:
                    best_f1 = f1_score
                    best_config = result
                
                self.logger.log_hyperparameter_result(combination, f1_score, is_best)
        
        # Complete the counter line and show best results
        print(f" Hyperparameter tuning done!")
        print(f"Best hyperparameters: LR={best_config['lr']}, Batch={best_config['batch_size']}, F1={best_f1:.4f}")
        
        self.logger.info(f"Hyperparameter search completed - Best F1: {best_f1:.4f} (LR={best_config['lr'] if best_config else None}, Batch={best_config['batch_size'] if best_config else None})")
        self.logger.info(f"All F1 scores: {[r['f1_score'] for r in search_results]}")
        
        return best_config, search_results
    
    def final_training(self, best_lr, best_batch_size, max_epochs=20):
        """Final training with best hyperparameters and early stopping"""
        
        # Use final training dataset (already separated)
        self.set_data_for_final()
        
        print(f"Final training parameters: LR={best_lr}, Batch={best_batch_size}, Max epochs={max_epochs}")
        print(f"Model training started...")
        self.logger.info(f"Final dataset: {len(self.train_data)} training, {len(self.val_data)} validation")
        self.logger.info(f"Best hyperparameters: LR={best_lr}, Batch={best_batch_size}, Max epochs={max_epochs}")
        
        # Reinitialize model for final training with selected hyperparameters
        self.model = T5ForConditionalGeneration.from_pretrained("t5-small").to(self.device)
        
        # Use aggressive early stopping for stable training
        final_history, final_f1 = self.train(
            lr=best_lr, 
            batch_size=best_batch_size, 
            epochs=max_epochs,
            patience=2,  # More aggressive patience for final training
            min_delta=0.01,  # Higher threshold for improvement
            is_final=True  # Enable final training display mode
        )
        
        self.logger.info(f"Final training completed - F1: {final_f1:.4f}")
        
        final_accuracy = final_history['val_accuracy'][-1] if final_history.get('val_accuracy') else None
        self.logger.info(f"Final accuracy: {final_accuracy:.4f}" if final_accuracy else "Final accuracy: Not available")
        
        return final_history, final_f1

def main():
    print("T5 Text Equivalence Model Training")
    
    trainer = Trainer()
    
    # Create data splits (10% for search, full for final training)
    data_config = trainer.create_data_splits()
    
    # Phase 1: Hyperparameter Search (10% of data)
    best_config, search_results = trainer.hyperparameter_search(search_epochs=6)
    
    if best_config is None:
        print("FAILED - No valid hyperparameter configuration found")
        trainer.logger.close()
        return
    
    # Phase 2: Final Training (full dataset with best hyperparameters) 
    final_history, final_f1 = trainer.final_training(
        best_lr=best_config['lr'], 
        best_batch_size=best_config['batch_size'], 
        max_epochs=20
    )
    
    # Final summary - minimal terminal output
    final_accuracy = final_history['val_accuracy'][-1] if final_history.get('val_accuracy') else None
    print(f"Training Complete - F1: {final_f1:.4f}, Accuracy: {(final_accuracy if final_accuracy else 0.0):.4f}")
    
    # Log detailed summary to file
    trainer.logger.info(f"Dataset usage - Search: {data_config['search_train']}/{data_config['search_val']}, Final: {data_config['final_train']}/{data_config['final_val']}")
    trainer.logger.info(f"Best hyperparameters: LR={best_config['lr']}, Batch={best_config['batch_size']}")
    trainer.logger.info(f"Final results: F1={final_f1:.4f}, Accuracy={(final_accuracy if final_accuracy else 0.0):.4f}, Train Loss={final_history['train_loss'][-1]:.4f}, Val Loss={final_history['val_loss'][-1]:.4f}")
    
    # Generate timestamped filenames for structured organization
    results_file = trainer.logger.get_timestamped_filename("training_results", "json")
    plot_file = trainer.logger.get_timestamped_filename("training_curves", "png")
    
    # Save results with timestamp
    with open(results_file, 'w') as f:
        json.dump({
            "data_configuration": data_config,
            "best_hyperparameters": best_config, 
            "final_history": final_history,
            "final_f1": final_f1,
            "search_results": search_results
        }, f, indent=2)
    
    # Generate training curves with timestamp
    generate_training_curves(final_history, plot_file)
    
    # Compare with BERT benchmark
    compare_with_bert_benchmark(final_f1, final_accuracy, trainer.logger)
    
    trainer.logger.info(f"Results saved to {results_file}, plots saved to {plot_file}")
    trainer.logger.close()

if __name__ == "__main__":
    main()
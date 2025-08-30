import torch
import warnings
from torch.optim import AdamW
from transformers import get_scheduler

from outputs import TrainingLogger, generate_training_curves, compare_with_bert_benchmark
from utils import (
    load_mrpc_dataset, create_dataloaders, initialize_model_and_tokenizer,
    train_single_epoch, validate_model, save_results
)

# Suppress known warnings for cleaner output
warnings.filterwarnings("ignore", category=RuntimeWarning, message="Mean of empty slice")
warnings.filterwarnings("ignore", category=RuntimeWarning, message="invalid value encountered in scalar divide")
warnings.filterwarnings("ignore", message="F-score is ill-defined and being set to 0.0 due to no true nor predicted samples")


class Trainer:
    def __init__(self):
        self.logger = TrainingLogger()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")        
        self.model, self.tokenizer = initialize_model_and_tokenizer(self.device)
        
        # Load and split dataset
        self.full_train_data, self.full_val_data, self.test_data = load_mrpc_dataset()
        self.search_train_data = self.full_train_data.select(range(int(len(self.full_train_data) * 0.10)))
        self.search_val_data = self.full_val_data.select(range(int(len(self.full_val_data) * 0.10)))
    
    def _set_data(self, use_search=True):
        if use_search:
            self.train_data, self.val_data = self.search_train_data, self.search_val_data
        else:
            self.train_data, self.val_data = self.full_train_data, self.full_val_data
    
    def train(self, lr=3e-5, batch_size=8, epochs=3, patience=1, min_delta=0.01, is_final=False):
        train_loader, val_loader = create_dataloaders(self.train_data, self.val_data, self.tokenizer, batch_size)
        
        optimizer = AdamW(self.model.parameters(), lr=lr, weight_decay=0.01)
        total_steps = len(train_loader) * epochs
        scheduler = get_scheduler("linear", optimizer=optimizer, 
                                num_warmup_steps=100, num_training_steps=total_steps)
        
        if is_final:
            print(f"Batches per epoch: {len(train_loader)}")
        
        history = {"train_loss": [], "val_loss": [], "val_accuracy": [], "val_f1": []}
        best_f1 = 0
        best_val_loss = float('inf')
        patience_counter = 0
        
        # Training loop
        for epoch in range(epochs):
            train_loss = train_single_epoch(self.model, train_loader, optimizer, scheduler, self.device)
            val_loss, accuracy, f1 = validate_model(self.model, val_loader, self.tokenizer, self.device)
            
            if is_final:
                print(f"Epoch {epoch + 1}: F1={f1:.4f}, Accuracy={accuracy:.4f}")
            
            # Update history
            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history["val_accuracy"].append(accuracy)
            history["val_f1"].append(f1)
            
            if f1 > best_f1:
                best_f1 = f1
            
            # Early stopping logic
            if val_loss < best_val_loss - min_delta:
                best_val_loss, patience_counter = val_loss, 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    break
        
        actual_epochs = len(history["train_loss"])
        
        if is_final:
            if actual_epochs < epochs:
                print(f"Training stopped early at epoch {actual_epochs}")
            print(f"Model training completed - Final F1: {best_f1:.4f}")
        
        return history, best_f1
    
    def hyperparameter_search(self, search_epochs=2, learning_rates=None, batch_sizes=None):
        self._set_data(use_search=True)
        
        learning_rates = learning_rates or [2e-5, 3e-5, 4e-5]
        batch_sizes = batch_sizes or [8, 12, 16]
        
        print("Starting hyperparameter search")
        
        best_f1 = -0.1
        best_config = None
        search_results = []
        
        combination = 0
        for lr in learning_rates:
            for batch_size in batch_sizes:
                combination += 1
                print(f"{combination}", end=".", flush=True)
                
                self.model, _ = initialize_model_and_tokenizer(self.device)
                history, f1_score = self.train(lr=lr, batch_size=batch_size, epochs=search_epochs)
                
                final_accuracy = history['val_accuracy'][-1] if history.get('val_accuracy') else None
                result = {
                    'lr': lr, 'batch_size': batch_size,
                    'f1_score': f1_score, 'final_accuracy': final_accuracy
                }
                search_results.append(result)
                
                if f1_score > best_f1:
                    best_f1 = f1_score
                    best_config = result
        
        print(f" Hyperparameter tuning done!")
        if best_config:
            print(f"Best hyperparameters: LR={best_config['lr']}, Batch={best_config['batch_size']}, F1={best_f1:.4f}")
        
        return best_config, search_results
    
    def final_training(self, best_lr, best_batch_size, max_epochs=20):
        self._set_data(use_search=False)
        
        print(f"Final training parameters: LR={best_lr}, Batch={best_batch_size}, Max epochs={max_epochs}")
        self.model, _ = initialize_model_and_tokenizer(self.device)
        
        final_history, final_f1 = self.train(
            lr=best_lr, 
            batch_size=best_batch_size, 
            epochs=max_epochs,
            patience=2,
            min_delta=0.01,
            is_final=True
        )
        
        return final_history, final_f1


def main():
    print("T5 Text Equivalence Model Training")
    
    trainer = Trainer()
    
    # Phase 1: Hyperparameter Search
    best_config, search_results = trainer.hyperparameter_search(search_epochs=6)
    
    if best_config is None:
        print("FAILED - No valid hyperparameter configuration found")
        trainer.logger.close()
        return
    
    # Phase 2: Final Training
    final_history, final_f1 = trainer.final_training(
        best_lr=best_config['lr'], 
        best_batch_size=best_config['batch_size'], 
        max_epochs=20
    )
    
    # Display and save results
    final_accuracy = final_history['val_accuracy'][-1] if final_history.get('val_accuracy') else None
    print(f"Training Complete - F1: {final_f1:.4f}, Accuracy: {(final_accuracy or 0.0):.4f}")
    
    # Save results
    results_file = trainer.logger.get_timestamped_filename("training_results", "json")
    plot_file = trainer.logger.get_timestamped_filename("training_curves", "png")
    
    results_data = {
        "best_hyperparameters": best_config, 
        "final_history": final_history,
        "final_f1": final_f1,
        "search_results": search_results
    }
    save_results(results_data, results_file)
    
    generate_training_curves(final_history, plot_file)
    compare_with_bert_benchmark(final_f1, final_accuracy, trainer.logger)    
    trainer.logger.close()


if __name__ == "__main__":
    main()
import logging
import os
from datetime import datetime

class TrainingLogger:
    def __init__(self, log_dir="logs"):
        # Create logs directory if it doesn't exist
        os.makedirs(log_dir, exist_ok=True)
        
        # Generate unique log filename with timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_filename = os.path.join(log_dir, f"training_log_{timestamp}.log")
        
        # Setup file logger
        self.logger = logging.getLogger('T5Training')
        self.logger.setLevel(logging.INFO)
        
        # Remove any existing handlers
        self.logger.handlers.clear()
        
        # File handler for detailed logging
        file_handler = logging.FileHandler(log_filename)
        file_handler.setLevel(logging.INFO)
        
        # Formatter for detailed logs
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        
        self.logger.addHandler(file_handler)
        
        # Store log filename and timestamp for reference
        self.log_filename = log_filename
        self.timestamp = timestamp
        
        # Initialize session
        self.logger.info("=== T5 TRAINING SESSION STARTED ===")
        
        # Terminal notification
        print(f"Training session started. Detailed logs: {log_filename}")
    
    def info(self, message):
        """Log info message to file only"""
        self.logger.info(message)
    
    def debug(self, message):
        """Log debug message to file only"""
        self.logger.debug(message)
    
    def terminal_checkpoint(self, message):
        """Display important checkpoint in terminal and log to file"""
        print(message)
        self.logger.info(f"TERMINAL: {message}")
    
    def log_hyperparameter_progress(self, config_num, total_configs, lr, batch_size):
        """Simple counter progress for hyperparameter search"""
        print(f"{config_num}", end="", flush=True)
        if config_num < total_configs:
            print(".", end="", flush=True)  # Add dot separator
        self.info(f"Starting hyperparameter configuration {config_num}/{total_configs} - LR={lr}, Batch={batch_size}")
    
    def log_hyperparameter_result(self, config_num, f1_score, is_best=False):
        """Log hyperparameter result"""
        if is_best:
            terminal_msg = f"Configuration {config_num} - NEW BEST F1: {f1_score:.4f}"
        else:
            terminal_msg = f"Configuration {config_num} - F1: {f1_score:.4f}"
        
        print(terminal_msg)
        self.info(f"TERMINAL: {terminal_msg}")
        self.info(f"Configuration {config_num} completed - F1: {f1_score:.4f}, Best: {is_best}")
    
    def log_training_epoch(self, epoch, total_epochs, train_loss, val_loss, f1_score, accuracy):
        """Log detailed epoch information to file only"""
        self.info(f"Epoch {epoch}/{total_epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, F1: {f1_score:.4f}, Accuracy: {accuracy:.4f}")
    
    def log_debug_outputs(self, generated_texts, expected_texts, valid_count, total_count):
        """Log debug output information to file only"""
        self.info(f"Generated samples: {generated_texts[:3]}")
        self.info(f"Expected samples: {expected_texts[:3]}")
        self.info(f"Valid predictions: {valid_count}/{total_count}")
    
    def log_early_stopping(self, epoch, reason):
        """Log early stopping event"""
        self.info(f"Early stopping triggered at epoch {epoch} - Reason: {reason}")
    
    def log_final_results(self, phase, best_f1, best_lr=None, best_batch=None):
        """Log final results for each phase"""
        if phase == "hyperparameter_search":
            terminal_msg = f"Hyperparameter search completed. Best F1: {best_f1:.4f} (LR={best_lr}, Batch={best_batch})"
            self.terminal_checkpoint(terminal_msg)
        elif phase == "final_training":
            terminal_msg = f"Final training completed. F1: {best_f1:.4f}"
            self.terminal_checkpoint(terminal_msg)
        
        self.info(f"{phase} phase completed - Best F1: {best_f1:.4f}")
    
    def get_timestamped_filename(self, base_name, extension, subfolder="results"):
        """Generate timestamped filename with organized directory structure"""
        os.makedirs(subfolder, exist_ok=True)
        return os.path.join(subfolder, f"{base_name}_{self.timestamp}.{extension}")
    
    def close(self):
        """Close logger and handlers"""
        for handler in self.logger.handlers:
            handler.close()
        self.logger.handlers.clear()
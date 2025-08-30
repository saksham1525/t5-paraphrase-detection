import logging
import os
from datetime import datetime
import matplotlib.pyplot as plt


class TrainingLogger:
    def __init__(self, log_dir="logs"):
        os.makedirs(log_dir, exist_ok=True)
        self.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_filename = os.path.join(log_dir, f"training_log_{self.timestamp}.log")
        
        self.logger = logging.getLogger('T5Training')
        self.logger.setLevel(logging.INFO)
        self.logger.handlers.clear()
        
        handler = logging.FileHandler(log_filename)
        handler.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
        self.logger.addHandler(handler)
        
        print(f"Training session started. Logs: {log_filename}")
    
    def info(self, message):
        self.logger.info(message)
    
    def get_timestamped_filename(self, base_name, extension, subfolder="results"):
        os.makedirs(subfolder, exist_ok=True)
        return os.path.join(subfolder, f"{base_name}_{self.timestamp}.{extension}")
    
    def close(self):
        for handler in self.logger.handlers:
            handler.close()
        self.logger.handlers.clear()


def generate_training_curves(final_history, plot_file):
    epochs = range(1, len(final_history["train_loss"]) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3))
    
    ax1.plot(epochs, final_history["train_loss"], "b-", label="Train")
    ax1.plot(epochs, final_history["val_loss"], "r-", label="Val")
    ax1.set_title("Final Training Loss")
    ax1.legend()
    
    ax2.plot(epochs, final_history["val_f1"], "g-", label="F1")
    ax2.plot(epochs, final_history["val_accuracy"], "m-", label="Accuracy")
    ax2.set_title("Final Training Metrics")
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    plt.show()


def compare_with_bert_benchmark(final_f1, final_accuracy, logger):
    bert_f1, bert_acc = 88.9, 84.8
    t5_f1 = final_f1 * 100
    t5_acc = (final_accuracy or 0.0) * 100
    
    print(f"Benchmark: BERT F1={bert_f1:.1f}% Acc={bert_acc:.1f}% | T5 F1={t5_f1:.1f}% Acc={t5_acc:.1f}%")
    logger.info(f"Benchmark - BERT: F1={bert_f1:.1f}% Acc={bert_acc:.1f}% | T5: F1={t5_f1:.1f}% Acc={t5_acc:.1f}%")

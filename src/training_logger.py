import logging
import os
from datetime import datetime

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
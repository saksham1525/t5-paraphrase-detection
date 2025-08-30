# T5 Text Equivalence Classification

A fine-tuned T5-small model for text equivalence classification on the MRPC dataset with hyperparameter search and fine tuning comparision with BERT benchmark.

## Features

- Automated hyperparameter search across 9 configurations
- Real-time training progress with comprehensive logging with timestamped files
- Performance benchmarking against BERT-base
- Training visualization and structured result organization
- **Modular codebase**: Clean separation of data handling, model operations, and utilities

## Installation

```bash
git clone https://github.com/yourusername/t5-paraphrase-detection.git
cd t5-paraphrase-detection
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Usage

```bash
cd src
python main.py
```

This executes hyperparameter search, final training, BERT comparison, and generates timestamped results in `logs/` and `results/` directories.

## Configuration

- **Model**: T5-small with AdamW optimizer
- **Hyperparameters**: Learning rates [2e-5, 3e-5, 4e-5], Batch sizes [8, 12, 16]  
- **Training**: 6 search epochs, up to 20 final epochs with early stopping
- **Output**: Binary classification ("0" or "1")

## Dependencies

Main requirements: `torch`, `transformers`, `datasets`, `scikit-learn`, `matplotlib`. Refer `requirements.txt`

## Project Structure

```
├── src/                       # Source code directory
│   ├── main.py               # Main training pipeline orchestration (182 lines)
│   ├── utils.py              # Core utilities: dataset, model, training functions (129 lines)
│   └── outputs.py            # Output handling: logging, plotting, benchmarking (61 lines)
├── requirements.txt          # Python dependencies
├── logs/                     # Training logs (timestamped)
├── results/                  # Results and plots (timestamped)
└── README.md                 # This file
```

## Module Overview

- **`main.py`**: High-level training orchestration and pipeline management
- **`utils.py`**: Essential utilities including TextDataset, model operations, training/validation functions
- **`outputs.py`**: All output operations including logging, plotting, and BERT benchmark comparison

## Performance

The model achieves competitive performance on MRPC:
- Typical F1 scores: 85-90%
- Typical accuracy: 82-87%
- Comparison against BERT-base benchmark provided automatically


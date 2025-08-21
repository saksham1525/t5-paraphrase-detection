# T5 Text Equivalence Classification

A fine-tuned T5-small model for text equivalence classification on the MRPC dataset with hyperparameter search and fine tuning comparision with BERT benchmark.

## Features

- Automated hyperparameter search across 9 configurations
- Real-time training progress with comprehensive logging with timestamped files
- Performance benchmarking against BERT-base
- Training visualization and structured result organization

## Installation

```bash
git clone https://github.com/yourusername/t5-text-equivalence.git
cd t5-text-equivalence
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Usage

```bash
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
├── main.py                    # Main training pipeline
├── training_logger.py         # Logging utilities
├── training_plots.py          # Visualization utilities
├── benchmark_comparison.py    # BERT benchmark comparison
├── requirements.txt           # Python dependencies
├── logs/                      # Training logs (timestamped)
├── results/                   # Results and plots (timestamped)
└── README.md                  # This file
```

## Performance

The model achieves competitive performance on MRPC:
- Typical F1 scores: 85-90%
- Typical accuracy: 82-87%
- Comparison against BERT-base benchmark provided automatically


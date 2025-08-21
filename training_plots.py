import matplotlib.pyplot as plt

def generate_training_curves(final_history, plot_file):
    """Generate and save training curve plots"""
    epochs = range(1, len(final_history["train_loss"]) + 1)
    plt.figure(figsize=(10, 3))
    
    plt.subplot(1, 2, 1)
    plt.plot(epochs, final_history["train_loss"], "b-", label="Train")
    plt.plot(epochs, final_history["val_loss"], "r-", label="Val")
    plt.title("Final Training Loss")
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(epochs, final_history["val_f1"], "g-", label="F1")
    plt.plot(epochs, final_history["val_accuracy"], "m-", label="Accuracy")
    plt.title("Final Training Metrics")
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    plt.show()

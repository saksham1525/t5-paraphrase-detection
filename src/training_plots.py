import matplotlib.pyplot as plt

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

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def plot_history(history, title_prefix="", save_path=None):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(history['train_losses'], label='Train Loss')
    ax1.plot(history['test_losses'], label='Test Loss')
    ax1.set_title(f'{title_prefix} Loss')
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.legend()

    ax2.plot(history['train_accs'], label='Train Accuracy')
    ax2.plot(history['test_accs'], label='Test Accuracy')
    ax2.set_title(f'{title_prefix} Accuracy')
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.legend()

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.close()

def plot_heatmap(matrix, row_labels, col_labels, title="Accuracy Heatmap", annot=True):
    plt.figure(figsize=(10, 6))
    sns.heatmap(matrix, xticklabels=col_labels, yticklabels=row_labels, annot=annot, fmt=".2f", cmap="viridis")
    plt.title(title)
    plt.xlabel("Width Settings")
    plt.ylabel("Depth Settings")
    plt.tight_layout()
    plt.show()
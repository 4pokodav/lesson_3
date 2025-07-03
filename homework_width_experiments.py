import torch
import logging
import itertools
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from datasets import get_mnist_loaders
from trainer import train_model
from utils.experiment_utils import setup_logging, run_experiment
from utils.model_utils import build_model
from utils.visualization_utils import plot_history

def run_width_experiments(width_settings, input_size, num_classes, device, epochs=5):
    train_loader, test_loader = get_mnist_loaders(batch_size=64)

    for name, layer_sizes in width_settings.items():
        print(f"\nЗапускаем эксперимент: {name} - Слои: {layer_sizes}")

        model = build_model(
            input_size=input_size,
            num_classes=num_classes,
            layer_sizes=layer_sizes
        ).to(device)

        logging.info(f"Эксперимент: {name} - {layer_sizes}")

        history, _, duration = run_experiment(
            model=model,
            train_loader=train_loader,
            test_loader=test_loader,
            epochs=epochs,
            device=device,
            label=f"width_{name}"
        )

        print(f"Время обучения: {duration:.2f} секунд")
        plot_history(history, title_prefix=f"Width {name}", save_path=f"homework_3/plots/width_{name}.png")

def run_grid_search(width_ranges, input_size, num_classes, device, epochs=3):
    train_loader, test_loader = get_mnist_loaders(batch_size=64)
    results = {}

    combos = list(itertools.product(*width_ranges))
    for combo in combos:
        model_id = f"{'-'.join(map(str, combo))}"
        model = build_model(
            input_size=input_size,
            num_classes=num_classes,
            layer_sizes=list(combo)
        ).to(device)

        history, final_acc, _ = run_experiment(
            model=model,
            train_loader=train_loader,
            test_loader=test_loader,
            epochs=epochs,
            device=device,
            label=f"grid_{model_id}"
        )

        results[combo] = final_acc
        print(f"{combo}: Test Accuracy = {final_acc:.4f}")

    return results

def plot_grid_search_heatmap(results, title="Grid Search Accuracy"):
    sizes = sorted(set([s[0] for s in results]))
    values = sorted(set([s[1] for s in results]))

    heatmap = np.zeros((len(sizes), len(values)))

    for (s1, s2, _), acc in results.items():
        i, j = sizes.index(s1), values.index(s2)
        heatmap[i, j] = acc

    plt.figure(figsize=(8, 6))
    sns.heatmap(heatmap, xticklabels=values, yticklabels=sizes, annot=True, fmt=".2f")
    plt.xlabel("Layer 2 Width")
    plt.ylabel("Layer 1 Width")
    plt.title(title)
    plt.tight_layout()
    plt.savefig("homework_3/plots/grid_search_heatmap.png")
    plt.close()

if __name__ == "__main__":
    setup_logging("homework_3/results/width_experiments/width_experiment.log")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    width_variants = {
        "narrow": [64, 32, 16],
        "medium": [256, 128, 64],
        "wide": [1024, 512, 256],
        "very_wide": [2048, 1024, 512]
    }

    run_width_experiments(
        width_settings=width_variants,
        input_size=784,
        num_classes=10,
        device=device,
        epochs=5
    )

    width_ranges = [
        [64, 128, 256],
        [32, 64, 128],
        [16]      
    ]

    search_results = run_grid_search(width_ranges, input_size=784, num_classes=10, device=device, epochs=3)
    plot_grid_search_heatmap(search_results, title="Width Grid Search Accuracy")
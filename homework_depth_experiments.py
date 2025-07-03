import torch
import time
import logging
from datasets import get_mnist_loaders
from trainer import train_model
from utils.experiment_utils import setup_logging, run_experiment
from utils.model_utils import build_model
from utils.visualization_utils import plot_history

def run_depth_experiments(depth_settings, input_size, num_classes, device, epochs=5):
    train_loader, test_loader = get_mnist_loaders(batch_size=64)

    for depth in depth_settings:
        layer_sizes = [128] * depth
        model = build_model(
            input_size=input_size,
            num_classes=num_classes,
            layer_sizes=layer_sizes
        ).to(device)

        model_name = f"depth_{depth}"
        print(f"\nЗапускаем эксперимент: {model_name}")
        logging.info(f"Запускаем эксперимент с глубиной {depth}")

        history, _, duration = run_experiment(
            model=model,
            train_loader=train_loader,
            test_loader=test_loader,
            epochs=epochs,
            device=device,
            label=model_name
        )

        print(f"Время обучения с глубиной{depth}: {duration:.2f} секунд")
        plot_history(history, title_prefix=f"Depth {depth}", save_path=f"homework_3/plots/depth_{depth}.png")

def run_overfitting_analysis(depth_settings, input_size, num_classes, device, epochs=10):
    train_loader, test_loader = get_mnist_loaders(batch_size=64)

    for depth in depth_settings:
        print(f"\nАнализ переобучения для глубины {depth} (Dropout + BatchNorm)")

        model = build_model(
            input_size=input_size,
            num_classes=num_classes,
            layer_sizes=[128] * depth,
            use_batchnorm=True,
            use_dropout=True,
            dropout_rate=0.3
        ).to(device)

        model_name = f"overfit_depth_{depth}"
        logging.info(f"Анализ переробучения для глубины {depth}")

        history, _, duration = run_experiment(
            model=model,
            train_loader=train_loader,
            test_loader=test_loader,
            epochs=epochs,
            device=device,
            label=model_name
        )

        print(f"Время анализа: {duration:.2f} секунд")
        plot_history(history, title_prefix=f"Overfit Depth {depth}", save_path=f"homework_3/plots/overfitting_depth_{depth}")

if __name__ == "__main__":
    setup_logging("homework_3/results/depth_experiments/depth_experiment.log")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    depth_list = [1, 2, 3, 5, 7]
    run_depth_experiments(
        depth_settings=depth_list,
        input_size=784,
        num_classes=10,
        device=device,
        epochs=5
    )

    run_overfitting_analysis(
        depth_settings=depth_list,
        input_size=784,
        num_classes=10,
        device=device,
        epochs=10
    )
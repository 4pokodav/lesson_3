import os
import logging
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import numpy as np

from datasets import get_mnist_loaders
from models import FullyConnectedModel
from trainer import train_model
from utils.visualization_utils import plot_history
from utils.model_utils import build_model

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("homework_3/results/regularization_experiments/experiment.log"),
        logging.StreamHandler()
    ]
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RESULTS_DIR = "homework_3/results/regularization_experiments"
os.makedirs(RESULTS_DIR, exist_ok=True)

def create_model_with_regularization(dropout_rate=0, use_batchnorm=False):
    """
    Создает модель с выбранной регуляризацией:
    - dropout_rate: коэффициент Dropout (0 - отключено)
    - use_batchnorm: включить BatchNorm или нет
    Используем фиксированную архитектуру для сравнения
    """
    input_size = 28 * 28
    num_classes = 10
    layer_sizes = [256, 128, 64]

    model = build_model(
        input_size=input_size,
        num_classes=num_classes,
        layer_sizes=layer_sizes,
        use_batchnorm=use_batchnorm,
        use_dropout=(dropout_rate > 0),
        dropout_rate=dropout_rate
    )

    return model.to(DEVICE)

def train_and_evaluate(model, l2_lambda=0):
    """
    Тренируем модель с L2 регуляризацией (weight_decay=l2_lambda)
    """
    train_loader, test_loader = get_mnist_loaders(batch_size=64)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=l2_lambda)

    train_losses, train_accs, test_losses, test_accs = [], [], [], []
    epochs = 15

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        for data, target in train_loader:
            data, target = data.to(DEVICE), target.to(DEVICE)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)

        train_loss = total_loss / len(train_loader)
        train_acc = correct / total

        model.eval()
        total_loss, correct, total = 0, 0, 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(DEVICE), target.to(DEVICE)
                output = model(data)
                loss = criterion(output, target)

                total_loss += loss.item()
                pred = output.argmax(dim=1)
                correct += pred.eq(target).sum().item()
                total += target.size(0)

        test_loss = total_loss / len(test_loader)
        test_acc = correct / total

        train_losses.append(train_loss)
        train_accs.append(train_acc)
        test_losses.append(test_loss)
        test_accs.append(test_acc)

        logging.info(
            f"Epoch {epoch+1}/{epochs} - Train loss: {train_loss:.4f}, Train acc: {train_acc:.4f}, "
            f"Test loss: {test_loss:.4f}, Test acc: {test_acc:.4f}"
        )

    history = {
        'train_losses': train_losses,
        'train_accs': train_accs,
        'test_losses': test_losses,
        'test_accs': test_accs,
    }

    return history

def plot_weights_distribution(model, title):
    """
    Визуализация распределения весов всех линейных слоев модели
    """
    weights = []
    for name, param in model.named_parameters():
        if 'weight' in name and len(param.data.size()) > 1:
            weights.append(param.data.cpu().numpy().flatten())

    if not weights:
        logging.warning("Не найдены веса для построения графика")
        return

    all_weights = np.concatenate(weights)

    plt.figure(figsize=(8, 5))
    plt.hist(all_weights, bins=100, alpha=0.75)
    plt.title(f'Weight distribution: {title}')
    plt.xlabel('Weight value')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"homework_3/plots/{title.replace(' ', '_')}_weights_hist.png")
    plt.close()


def experiment_regularization():
    """
    Основной эксперимент 3.1: сравнение техник регуляризации
    """
    logging.info("Запуск экспериментов по регуляризации")

    configs = [
        {'name': 'No Regularization', 'dropout': 0, 'batchnorm': False, 'weight_decay': 0},
        {'name': 'Dropout 0.1', 'dropout': 0.1, 'batchnorm': False, 'weight_decay': 0},
        {'name': 'Dropout 0.3', 'dropout': 0.3, 'batchnorm': False, 'weight_decay': 0},
        {'name': 'Dropout 0.5', 'dropout': 0.5, 'batchnorm': False, 'weight_decay': 0},
        {'name': 'BatchNorm only', 'dropout': 0, 'batchnorm': True, 'weight_decay': 0},
        {'name': 'Dropout 0.5 + BatchNorm', 'dropout': 0.5, 'batchnorm': True, 'weight_decay': 0},
        {'name': 'L2 regularization (1e-4)', 'dropout': 0, 'batchnorm': False, 'weight_decay': 1e-4},
    ]

    results = {}

    for cfg in configs:
        logging.info(f"Обучение модели с конфигурацией: {cfg['name']}")

        model = create_model_with_regularization(
            dropout_rate=cfg['dropout'],
            use_batchnorm=cfg['batchnorm']
        )

        history = train_and_evaluate(model, l2_lambda=cfg['weight_decay'])
        results[cfg['name']] = history

        # Визуализация графиков обучения
        plot_history(history, title_prefix=cfg['name'], save_path=f"homework_3/plots/{cfg['name'].replace(' ', '_')}_history.png")

        # Визуализация распределения весов после обучения
        plot_weights_distribution(model, cfg['name'])

    logging.info("Эксперименты по регуляризации завершены")
    return results


def experiment_adaptive_regularization():
    """
    Эксперимент 3.2: адаптивные техники регуляризации
    - Dropout с изменяющимся коэффициентом (с ростом эпох)
    - BatchNorm с разным momentum
    - Комбинация этих техник
    """
    logging.info("Запуск экспериментов по адаптивной регуляризации")

    train_loader, test_loader = get_mnist_loaders(batch_size=64)
    criterion = nn.CrossEntropyLoss()
    epochs = 15

    def train_model_with_adaptive_dropout(momentum=None):
        """
        Пример адаптивного Dropout: коэффициент растет от 0 до 0.5 по эпохам
        BatchNorm с заданным momentum
        """
        input_size = 28 * 28
        num_classes = 10
        layer_sizes = [256, 128, 64]

        # Собираем слои вручную с изменяющимся Dropout
        class AdaptiveDropoutModel(nn.Module):
            def __init__(self, dropout_start=0.0, dropout_end=0.5, epochs=epochs, momentum=momentum):
                super().__init__()
                layers = []
                prev_size = input_size
                self.epochs = epochs
                self.current_epoch = 0
                self.dropout_start = dropout_start
                self.dropout_end = dropout_end

                for size in layer_sizes:
                    layers.append(nn.Linear(prev_size, size))
                    if momentum is not None:
                        layers.append(nn.BatchNorm1d(size, momentum=momentum))
                    layers.append(nn.ReLU())
                    # Dropout rate будет изменяться в forward
                    layers.append(nn.Dropout(p=dropout_start))
                    prev_size = size

                layers.append(nn.Linear(prev_size, num_classes))
                self.layers = nn.Sequential(*layers)

            def forward(self, x):
                x = x.view(x.size(0), -1)
                # Перед forward обновим Dropout для всех Dropout слоев
                dropout_layers = [layer for layer in self.layers if isinstance(layer, nn.Dropout)]
                current_p = self.dropout_start + (self.dropout_end - self.dropout_start) * self.current_epoch / max(1, self.epochs - 1)
                for layer in dropout_layers:
                    layer.p = current_p
                output = self.layers(x)
                return output

            def set_epoch(self, epoch):
                self.current_epoch = epoch

        model = AdaptiveDropoutModel()
        model.to(DEVICE)

        optimizer = optim.Adam(model.parameters(), lr=0.001)
        train_losses, train_accs, test_losses, test_accs = [], [], [], []

        for epoch in range(epochs):
            model.set_epoch(epoch)
            model.train()
            total_loss, correct, total = 0, 0, 0
            for data, target in train_loader:
                data, target = data.to(DEVICE), target.to(DEVICE)
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                pred = output.argmax(dim=1)
                correct += pred.eq(target).sum().item()
                total += target.size(0)

            train_loss = total_loss / len(train_loader)
            train_acc = correct / total

            model.eval()
            total_loss, correct, total = 0, 0, 0
            with torch.no_grad():
                for data, target in test_loader:
                    data, target = data.to(DEVICE), target.to(DEVICE)
                    output = model(data)
                    loss = criterion(output, target)

                    total_loss += loss.item()
                    pred = output.argmax(dim=1)
                    correct += pred.eq(target).sum().item()
                    total += target.size(0)

            test_loss = total_loss / len(test_loader)
            test_acc = correct / total

            train_losses.append(train_loss)
            train_accs.append(train_acc)
            test_losses.append(test_loss)
            test_accs.append(test_acc)

            logging.info(f"Epoch {epoch+1}/{epochs} - Dropout p={model.layers[3].p:.3f} - "
                         f"Train acc: {train_acc:.4f}, Test acc: {test_acc:.4f}")

        history = {
            'train_losses': train_losses,
            'train_accs': train_accs,
            'test_losses': test_losses,
            'test_accs': test_accs,
        }
        return model, history

    # Пробуем разные momentums для BatchNorm
    momentums = [0.1, 0.5, 0.9]
    results = {}

    for momentum in momentums:
        logging.info(f"Обучение с BatchNorm (momentum={momentum}) и адаптивным Dropout")
        model, history = train_model_with_adaptive_dropout(momentum=momentum)
        name = f"Adaptive Dropout + BatchNorm momentum={momentum}"
        results[name] = history
        plot_history(history, title_prefix=name, save_path=f"homework_3/plots/{name.replace(' ', '_')}_history.png")
        plot_weights_distribution(model, name)

    logging.info("Эксперименты по адаптивной регуляризации завершены")
    return results


if __name__ == '__main__':
    logging.info("Начало экспериментов по регуляризации")
    results_3_1 = experiment_regularization()

    logging.info("Начало экспериментов по адаптивной регуляризации")
    results_3_2 = experiment_adaptive_regularization()

    logging.info("Все эксперименты завершены")
import torch
import os
import time
import logging
from trainer import train_model
from utils.utils import count_parameters

def setup_logging(log_file):
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    logging.basicConfig(
        filename=log_file,
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

def run_experiment(model, train_loader, test_loader, label, device, epochs=5, lr=0.001):
    model = model.to(device)
    start_time = time.time()

    history = train_model(
        model, train_loader, test_loader,
        epochs=epochs, lr=lr, device=str(device)
    )

    end_time = time.time()
    duration = end_time - start_time

    num_params = count_parameters(model)
    final_train_acc = history['train_accs'][-1]
    final_test_acc = history['test_accs'][-1]

    logging.info(f"Эксперимент: {label}")
    logging.info(f"Параметры: {num_params}, Время: {duration:.2f}s")
    logging.info(f"Итоговая точность - Train: {final_train_acc:.4f}, Test: {final_test_acc:.4f}\n")

    return history, num_params, duration
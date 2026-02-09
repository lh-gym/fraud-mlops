from __future__ import annotations

import os
import random
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from fraud_mlops.model import build_model


@dataclass
class TrainingResult:
    model_state: dict
    model_config: dict
    metrics: dict[str, float]


def multiprocessing_backend_available() -> bool:
    shm_manager = Path(torch.__file__).resolve().parent / "bin" / "torch_shm_manager"
    return shm_manager.exists() and os.access(shm_manager, os.X_OK)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def compute_classification_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> dict[str, float]:
    y_true = y_true.astype(np.int64)
    y_pred = (y_prob >= threshold).astype(np.int64)

    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))

    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2.0 * precision * recall / max(precision + recall, 1e-8)
    accuracy = (tp + tn) / max(tp + tn + fp + fn, 1)

    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "accuracy": float(accuracy),
    }


def predict_scores(model: nn.Module, x_data: np.ndarray, device: torch.device, batch_size: int = 2048) -> np.ndarray:
    dataset = TensorDataset(torch.from_numpy(x_data).float())
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    outputs = []
    model.eval()
    with torch.no_grad():
        for (features,) in loader:
            logits = model(features.to(device))
            probs = torch.sigmoid(logits).cpu().numpy()
            outputs.append(probs)
    return np.concatenate(outputs, axis=0) if outputs else np.empty((0,), dtype=np.float32)


def measure_latency_ms(model: nn.Module, x_data: np.ndarray, device: torch.device, repeats: int = 20) -> tuple[float, float]:
    if len(x_data) == 0:
        return 0.0, 0.0
    sample = x_data[: min(len(x_data), 2048)]
    timings = []
    for _ in range(repeats):
        start = time.perf_counter()
        _ = predict_scores(model, sample, device=device, batch_size=len(sample))
        elapsed = (time.perf_counter() - start) * 1000.0
        timings.append(elapsed)
    avg_ms = float(np.mean(timings))
    p95_ms = float(np.percentile(timings, 95))
    return avg_ms, p95_ms


def train_model(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    hidden_dims: list[int],
    lr: float,
    epochs: int,
    batch_size: int,
    dropout: float,
    backend: str = "multiprocessing",
    seed: int = 42,
) -> TrainingResult:
    seed_everything(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if backend == "multiprocessing":
        if multiprocessing_backend_available():
            workers = max(1, min(8, (os.cpu_count() or 2) // 2))
        else:
            workers = 0
    else:
        workers = 0
        torch.set_num_threads(max(1, (os.cpu_count() or 2) - 1))

    train_dataset = TensorDataset(
        torch.from_numpy(x_train).float(),
        torch.from_numpy(y_train).float(),
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=workers > 0,
    )

    model = build_model(input_dim=x_train.shape[1], hidden_dims=hidden_dims, dropout=dropout).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_f1 = -1.0
    best_state = None

    for _ in range(epochs):
        model.train()
        for features, target in train_loader:
            features = features.to(device)
            target = target.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(features)
            loss = criterion(logits, target)
            loss.backward()
            optimizer.step()

        val_prob = predict_scores(model, x_val, device=device)
        val_metrics = compute_classification_metrics(y_val, val_prob)
        if val_metrics["f1"] > best_f1:
            best_f1 = val_metrics["f1"]
            best_state = {name: tensor.detach().cpu() for name, tensor in model.state_dict().items()}

    if best_state is None:
        best_state = {name: tensor.detach().cpu() for name, tensor in model.state_dict().items()}

    model.load_state_dict(best_state)
    val_prob = predict_scores(model, x_val, device=device)
    metrics = compute_classification_metrics(y_val, val_prob)
    avg_latency_ms, p95_latency_ms = measure_latency_ms(model, x_val, device=device)
    metrics.update(
        {
            "latency_avg_ms": avg_latency_ms,
            "latency_p95_ms": p95_latency_ms,
        }
    )

    return TrainingResult(
        model_state=best_state,
        model_config={
            "input_dim": int(x_train.shape[1]),
            "hidden_dims": list(hidden_dims),
            "dropout": float(dropout),
        },
        metrics=metrics,
    )

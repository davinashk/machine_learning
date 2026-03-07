from pathlib import Path

import torch
from torch import nn

from .data_processing import (
    compute_normalization_stats,
    load_raw_data,
    normalize_features,
    normalize_target,
    preprocess_features,
    preprocess_target,
    save_normalization_stats,
)
from .model import build_model, save_model


def train(
    data_path: str | Path = "data/used_cars.csv",
    model_dir: str | Path = "model",
    epochs: int = 2500,
    lr: float = 1e-4,
) -> nn.Module:
    """End-to-end training pipeline."""
    data_path = Path(data_path)
    model_dir = Path(model_dir)

    df = load_raw_data(data_path)
    X = preprocess_features(df)
    y = preprocess_target(df)

    X_mean, X_std, y_mean, y_std = compute_normalization_stats(X, y)
    save_normalization_stats(model_dir, X_mean, X_std, y_mean, y_std)

    X_norm = normalize_features(X, X_mean, X_std)
    y_norm = normalize_target(y, y_mean, y_std)

    model = build_model(input_dim=X_norm.shape[1], output_dim=1)
    loss_func = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    for _ in range(epochs):
        y_pred = model(X_norm)
        loss = loss_func(y_pred, y_norm)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    save_model(model, model_dir)
    return model


if __name__ == "__main__":
    train()


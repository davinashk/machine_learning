from pathlib import Path
from typing import Iterable, Sequence

import torch

from .data_processing import (
    load_normalization_stats,
    normalize_features,
)
from .model import load_model


def predict(
    inputs: Sequence[Sequence[float]] | torch.Tensor,
    model_dir: str | Path = "model",
) -> torch.Tensor:
    """
    Run predictions for one or more input rows.

    Each row should be [age, mileage, accident_free_flag].
    """
    model_dir = Path(model_dir)

    if not isinstance(inputs, torch.Tensor):
        inputs = torch.tensor(inputs, dtype=torch.float32)

    X_mean, X_std, y_mean, y_std = load_normalization_stats(model_dir)
    X_norm = normalize_features(inputs, X_mean, X_std)

    model = load_model(model_dir, input_dim=inputs.shape[1], output_dim=1)

    with torch.no_grad():
        y_norm_pred = model(X_norm)

    return y_norm_pred * y_std + y_mean


if __name__ == "__main__":
    # Example usage
    example_inputs = [
        [5, 2000, 1],
        [10, 4000, 1],
    ]
    preds = predict(example_inputs)
    print(preds)


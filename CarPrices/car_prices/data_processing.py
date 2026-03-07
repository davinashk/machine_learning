from pathlib import Path
from typing import Tuple

import pandas as pd
import torch


def load_raw_data(data_path: str | Path) -> pd.DataFrame:
    """Load the raw used-cars CSV."""
    return pd.read_csv(data_path)


def preprocess_features(df: pd.DataFrame) -> torch.Tensor:
    """Turn raw dataframe columns into a feature tensor X."""
    age = df["model_year"].max() - df["model_year"]

    mileage = (
        df["milage"]
        .str.replace(",", "")
        .str.replace(" mi.", "")
        .astype(int)
    )

    accident_free = (df["accident"] == "None reported").astype(int)

    X = torch.column_stack(
        [
            torch.tensor(age, dtype=torch.float32),
            torch.tensor(mileage, dtype=torch.float32),
            torch.tensor(accident_free, dtype=torch.float32),
        ]
    )
    return X


def preprocess_target(df: pd.DataFrame) -> torch.Tensor:
    """Turn raw price column into a target tensor y."""
    price = (
        df["price"]
        .str.replace(",", "")
        .str.replace("$", "")
        .astype(int)
    )

    y = torch.tensor(price, dtype=torch.float32).reshape(-1, 1)
    return y


def compute_normalization_stats(
    X: torch.Tensor, y: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute mean/std for features and target."""
    X_mean = X.mean(dim=0)
    X_std = X.std(dim=0)
    y_mean = y.mean()
    y_std = y.std()
    return X_mean, X_std, y_mean, y_std


def normalize_features(
    X: torch.Tensor, X_mean: torch.Tensor, X_std: torch.Tensor
) -> torch.Tensor:
    return (X - X_mean) / X_std


def normalize_target(
    y: torch.Tensor, y_mean: torch.Tensor, y_std: torch.Tensor
) -> torch.Tensor:
    return (y - y_mean) / y_std


def save_normalization_stats(
    model_dir: str | Path,
    X_mean: torch.Tensor,
    X_std: torch.Tensor,
    y_mean: torch.Tensor,
    y_std: torch.Tensor,
) -> None:
    """Persist normalization statistics to disk."""
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    torch.save(X_mean, model_dir / "X_mean.pt")
    torch.save(X_std, model_dir / "X_std.pt")
    torch.save(y_mean, model_dir / "y_mean.pt")
    torch.save(y_std, model_dir / "y_std.pt")


def load_normalization_stats(
    model_dir: str | Path,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Load normalization statistics from disk."""
    model_dir = Path(model_dir)
    X_mean = torch.load(model_dir / "X_mean.pt")
    X_std = torch.load(model_dir / "X_std.pt")
    y_mean = torch.load(model_dir / "y_mean.pt")
    y_std = torch.load(model_dir / "y_std.pt")
    return X_mean, X_std, y_mean, y_std


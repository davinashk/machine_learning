from pathlib import Path

import torch
from torch import nn


def build_model(input_dim: int = 3, output_dim: int = 1) -> nn.Module:
    """Create the core regression model."""
    return nn.Linear(input_dim, output_dim)


def save_model(model: nn.Module, model_dir: str | Path) -> None:
    """Save model parameters to disk."""
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), model_dir / "model.pt")


def load_model(
    model_dir: str | Path, input_dim: int = 3, output_dim: int = 1
) -> nn.Module:
    """Load a trained model from disk."""
    model_dir = Path(model_dir)
    model = build_model(input_dim=input_dim, output_dim=output_dim)
    state_dict = torch.load(model_dir / "model.pt", map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()
    return model


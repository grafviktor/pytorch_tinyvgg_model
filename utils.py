"""
Contains various utility functions for PyTorch model training and saving.
"""

import torch
from pathlib import Path
import model_builder


def save_model(model: torch.nn.Module,
               target_dir: str,
               model_name: str,
               model_args: dict):
    """Saves a PyTorch model to a target directory.

    Args:
      model: A target PyTorch model to save.
      target_dir: A directory for saving the model to.
      model_name: A filename for the saved model. Should include
        either ".pth" or ".pt" as the file extension.
      model_args: model constructor arguments

    Example usage:
      save_model(model=model_0,
                 target_dir="models",
                 model_name="05_going_modular_tingvgg_model.pth")
    """
    # Create target directory
    target_dir_path = Path(target_dir)
    target_dir_path.mkdir(parents=True,
                          exist_ok=True)

    # Create model save path
    assert model_name.endswith(".pth") or model_name.endswith(".pt"), "model_name should end with '.pt' or '.pth'"
    model_save_path = target_dir_path / model_name

    # Save the model state_dict() and constructor args
    print(f"[INFO] Saving model to: {model_save_path}")
    torch.save({
        "model_state": model.state_dict(),
        "model_args": model_args,
    }, f=model_save_path)


def load_model(target_dir: str,
               model_name: str) -> torch.nn.Module:

    model_load_path = Path(target_dir) / model_name
    print(f"[INFO] Loading model from: {model_load_path}")
    checkpoint = torch.load(model_load_path)
    model = model_builder.TinyVGG(**checkpoint["model_args"])  # rebuild with args
    model.load_state_dict(checkpoint["model_state"])
    return model
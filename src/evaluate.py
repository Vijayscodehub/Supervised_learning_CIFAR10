"""Evaluate the selected CIFAR-10 model on the untouched test set."""
import json
from pathlib import Path

import torch
from torch import nn

from src.data import build_dataloaders
from src.model import SmallCNN
from src.train import get_device, validate_one_epoch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT_PATH = (
    PROJECT_ROOT
    / "checkpoints"
    / "augmented_dropout0_0_1e-3_seed42"
    / "best_model.pt"
)
RESULT_PATH = PROJECT_ROOT / "reports" / "final_test_result.json"


def main() -> None:
    # The test loader has no random crop or flip, so every test image is fixed.
    _, _, test_loader = build_dataloaders()

    device = get_device()
    model = SmallCNN().to(device)

    # Load the saved weights from the selected validation checkpoint.
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])

    # Reuse the validation loop because it evaluates a loader without learning.
    test_metrics = validate_one_epoch(
        model=model,
        validation_loader=test_loader,
        loss_function=nn.CrossEntropyLoss(),
        device=device,
    )

    # Save a readable record of the final test result.
    RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
    result = {
        "checkpoint": str(CHECKPOINT_PATH),
        "checkpoint_epoch": checkpoint["epoch"],
        "checkpoint_validation_accuracy": checkpoint["validation_accuracy"],
        "test": test_metrics,
    }
    RESULT_PATH.write_text(json.dumps(result, indent=2) + "\n")

    print(f"Checkpoint epoch: {checkpoint['epoch']}")
    print(f"Test loss: {test_metrics['loss']:.4f}")
    print(f"Test accuracy: {test_metrics['accuracy']:.2%}")


if __name__ == "__main__":
    main()
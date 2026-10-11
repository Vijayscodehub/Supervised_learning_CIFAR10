"""Train the CIFAR-10 CNN for multiple epochs and save the best model."""
import json
from pathlib import Path

import torch

# `nn` contains neural-network layers and loss functions.
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# These imports use our own project files.
from src.data import build_dataloaders
from src.model import SmallCNN

NUM_EPOCHS = 20
LEARNING_RATE = 3e-4
EXPERIMENT_NAME = "augmented_lr_3e-4_seed42"
USE_AUGMENTATION = True
SEED = 42
PROJECT_ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT_PATH = PROJECT_ROOT / "checkpoints" / EXPERIMENT_NAME / "best_model.pt"
RESULT_PATH = CHECKPOINT_PATH.with_name("best_result.json")
RUNS_DIR = PROJECT_ROOT / "runs" / EXPERIMENT_NAME


def get_device() -> torch.device:
    # `def` creates a reusable function. `-> torch.device` is a type hint:
    # it tells readers and editors that this function returns a device object.
    # This project intentionally trains on CPU. `torch.device("cpu")` tells
    # PyTorch where model weights, images, and labels should live.
    return torch.device("cpu")


def train_one_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    loss_function: nn.Module,
    optimizer: Optimizer,
    device: torch.device,
) -> dict[str, float]:
    # The names after `:` are type hints only; Python does not enforce them.
    # This function trains once on every batch, which is one epoch.

    # `.train()` switches layers such as Dropout into training behavior.
    model.train()

    # These variables collect values from all batches for epoch-level metrics.
    total_loss = 0.0
    total_correct = 0
    total_examples = 0

    # A DataLoader gives one batch at a time. Python unpacks each batch into
    # `images` and their matching ground-truth `labels`.
    for images, labels in train_loader:
        # `.to(device)` returns tensors located on the same device as the model.
        images = images.to(device)
        labels = labels.to(device)

        # Gradients accumulate by default in PyTorch, so clear the gradients
        # left from the previous batch before calculating new ones.
        optimizer.zero_grad()

        # Calling `model(images)` runs the model's `forward` method.
        # It returns 10 raw class scores, called logits, for each image.
        logits = model(images)

        # `loss_function(...)` compares the logits with the correct integer
        # labels and returns one number describing how wrong this batch is.
        loss = loss_function(logits, labels)

        # `.backward()` applies backpropagation and stores gradients on every
        # trainable weight and bias. `.step()` uses those gradients to update
        # the parameters in a direction that should reduce future loss.
        loss.backward()
        optimizer.step()

        # `size(0)` means the size of dimension 0: the number of images here.
        batch_size = labels.size(0)

        # CrossEntropyLoss is already an average for this batch. Multiply by
        # its size so the final epoch average treats every image equally.
        total_loss += loss.item() * batch_size

        # `argmax(dim=1)` finds the position of the largest score among the
        # 10 class scores. That position is the model's predicted class ID.
        predictions = torch.argmax(logits, dim=1)

        # `==` compares predictions and labels item by item. `sum()` counts
        # True values, and `.item()` converts the one-value tensor to Python.
        total_correct += (predictions == labels).sum().item()
        total_examples += batch_size

    # `return { ... }` creates and returns a dictionary with named metrics.
    # Dividing by all examples gives average loss and fraction correct.
    return {
        "loss": total_loss / total_examples,
        "accuracy": total_correct / total_examples,
    }


def validate_one_epoch(
    model: nn.Module,
    validation_loader: DataLoader,
    loss_function: nn.Module,
    device: torch.device,
) -> dict[str, float]:
    # Evaluate on validation data without changing the model's weights.
    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_examples = 0

    with torch.no_grad():
        for images, labels in validation_loader:
            images = images.to(device)
            labels = labels.to(device)

            logits = model(images)
            loss = loss_function(logits, labels)

            batch_size = labels.size(0)
            total_loss += loss.item() * batch_size

            predictions = torch.argmax(logits, dim=1)
            total_correct += (predictions == labels).sum().item()
            total_examples += batch_size

    # Return average loss and accuracy across every validation image.
    return {
        "loss": total_loss / total_examples,
        "accuracy": total_correct / total_examples,
    }


def save_checkpoint(
    model: nn.Module,
    epoch: int,
    training_metrics: dict[str, float],
    validation_metrics: dict[str, float],
) -> None:
    # Save the best weights and a readable summary of that epoch's results.
    CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "validation_accuracy": validation_metrics["accuracy"],
        },
        CHECKPOINT_PATH,
    )
    result = {
        "experiment_name": EXPERIMENT_NAME,
        "seed": SEED,
        "learning_rate": LEARNING_RATE,
        "use_augmentation": USE_AUGMENTATION,
        "epoch": epoch,
        "training": training_metrics,
        "validation": validation_metrics,
    }
    RESULT_PATH.write_text(json.dumps(result, indent=2) + "\n")


def main() -> None:
    # Train and validate repeatedly, keeping the best validation model.
    device = get_device()
    torch.manual_seed(SEED)

    # Keep crop and flip enabled while comparing learning rates.
    train_loader, validation_loader, _ = build_dataloaders(
        use_augmentation=USE_AUGMENTATION
    )

    # Create a new CNN with random initial weights and biases, then move all
    # its parameters to the selected device with `.to(device)`.
    model = SmallCNN().to(device)

    # CrossEntropyLoss is the standard loss for one-label, 10-class problems.
    loss_function = nn.CrossEntropyLoss()

    # `model.parameters()` supplies every learnable weight and bias to AdamW.
    # `lr=1e-3` is a keyword argument; 1e-3 means 0.001, the learning rate and 3e-4 means 0.0003.
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    # TensorBoard writes one point for each metric after every epoch.
    writer = SummaryWriter(log_dir=RUNS_DIR)

    print(f"Experiment: {EXPERIMENT_NAME}")
    print(f"Using augmentation: {USE_AUGMENTATION}")
    print(f"Using device: {device}")
    best_validation_accuracy = 0.0

    for epoch in range(1, NUM_EPOCHS + 1):
        training_metrics = train_one_epoch(
            model, train_loader, loss_function, optimizer, device
        )
        validation_metrics = validate_one_epoch(
            model, validation_loader, loss_function, device
        )

        print(f"Epoch {epoch}/{NUM_EPOCHS}")
        print(f"Training loss: {training_metrics['loss']:.4f}")
        print(f"Training accuracy: {training_metrics['accuracy']:.2%}")
        print(f"Validation loss: {validation_metrics['loss']:.4f}")
        print(f"Validation accuracy: {validation_metrics['accuracy']:.2%}")

        writer.add_scalar("Loss/train", training_metrics["loss"], epoch)
        writer.add_scalar("Loss/validation", validation_metrics["loss"], epoch)
        writer.add_scalar("Accuracy/train", training_metrics["accuracy"], epoch)
        writer.add_scalar("Accuracy/validation", validation_metrics["accuracy"], epoch)
        writer.add_scalar("Learning rate", optimizer.param_groups[0]["lr"], epoch)

        if validation_metrics["accuracy"] > best_validation_accuracy:
            best_validation_accuracy = validation_metrics["accuracy"]
            save_checkpoint(model, epoch, training_metrics, validation_metrics)
            print(f"Saved best model: {CHECKPOINT_PATH}")

    writer.close()


if __name__ == "__main__":
    # `__name__` is "__main__" only when this file is run directly with
    # `python -m src.train`. Importing it from another file will not train.
    main()
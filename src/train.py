"""Train the CIFAR-10 CNN and validate after one epoch."""
import torch

# `nn` contains neural-network layers and loss functions.
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader

# These imports use our own project files.
from src.data import build_dataloaders
from src.model import SmallCNN


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


def main() -> None:
    # Train once, then measure performance on separate validation images.
    device = get_device()

    train_loader, validation_loader, _ = build_dataloaders()

    # Create a new CNN with random initial weights and biases, then move all
    # its parameters to the selected device with `.to(device)`.
    model = SmallCNN().to(device)

    # CrossEntropyLoss is the standard loss for one-label, 10-class problems.
    loss_function = nn.CrossEntropyLoss()

    # `model.parameters()` supplies every learnable weight and bias to AdamW.
    # `lr=1e-3` is a keyword argument; 1e-3 means 0.001, the learning rate.
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    training_metrics = train_one_epoch(
        model=model,
        train_loader=train_loader,
        loss_function=loss_function,
        optimizer=optimizer,
        device=device,
    )

    validation_metrics = validate_one_epoch(
        model=model,
        validation_loader=validation_loader,
        loss_function=loss_function,
        device=device,
    )

    # An f-string starts with `f` and inserts values inside `{...}`.
    # `:.4f` shows loss with four decimal places; `:.2%` shows accuracy as a
    # percentage with two decimal places. Dictionary values use `metrics[...]`.
    print(f"Using device: {device}")
    print(f"Training loss: {training_metrics['loss']:.4f}")
    print(f"Training accuracy: {training_metrics['accuracy']:.2%}")
    print(f"Validation loss: {validation_metrics['loss']:.4f}")
    print(f"Validation accuracy: {validation_metrics['accuracy']:.2%}")


if __name__ == "__main__":
    # `__name__` is "__main__" only when this file is run directly with
    # `python -m src.train`. Importing it from another file will not train.
    main()
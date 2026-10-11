"""Analyze mistakes made by the selected CIFAR-10 model on test images."""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, confusion_matrix

from src.data import CIFAR10_MEAN, CIFAR10_STD, build_dataloaders
from src.evaluate import CHECKPOINT_PATH
from src.model import SmallCNN
from src.train import get_device


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = PROJECT_ROOT / "reports"
FIGURE_DIR = REPORT_DIR / "figures"


def collect_predictions(model: SmallCNN, test_loader, device: torch.device):
    # Store every prediction once so all reports describe the same test run.
    all_images = []
    all_labels = []
    all_predictions = []
    all_confidences = []

    model.eval()
    with torch.no_grad():
        for images, labels in test_loader:
            logits = model(images.to(device))
            probabilities = torch.softmax(logits, dim=1)
            confidences, predictions = torch.max(probabilities, dim=1)

            all_images.append(images)
            all_labels.append(labels)
            all_predictions.append(predictions.cpu())
            all_confidences.append(confidences.cpu())

    return (
        torch.cat(all_images),
        torch.cat(all_labels),
        torch.cat(all_predictions),
        torch.cat(all_confidences),
    )


def show_image_grid(images, labels, predictions, confidences, class_names, title, path):
    # Undo normalization so the saved images have natural colors.
    mean = torch.tensor(CIFAR10_MEAN).view(3, 1, 1)
    standard_deviation = torch.tensor(CIFAR10_STD).view(3, 1, 1)

    figure, axes = plt.subplots(4, 4, figsize=(10, 10))
    figure.suptitle(title)

    for axis, image, label, prediction, confidence in zip(
        axes.flat, images[:16], labels[:16], predictions[:16], confidences[:16]
    ):
        display_image = (image * standard_deviation + mean).clamp(0, 1)
        axis.imshow(display_image.permute(1, 2, 0))
        axis.set_title(
            f"True: {class_names[label]}\n"
            f"Pred: {class_names[prediction]} ({confidence:.0%})",
            fontsize=8,
        )
        axis.axis("off")

    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def main() -> None:
    # The test loader has fixed preprocessing and never changes model weights.
    _, _, test_loader = build_dataloaders()
    class_names = test_loader.dataset.classes

    device = get_device()
    model = SmallCNN().to(device)
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])

    images, labels, predictions, confidences = collect_predictions(
        model, test_loader, device
    )

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    report = classification_report(
        labels, predictions, target_names=class_names, output_dict=True, zero_division=0
    )
    (REPORT_DIR / "per_class_metrics.json").write_text(json.dumps(report, indent=2) + "\n")

    matrix = confusion_matrix(labels, predictions, labels=range(len(class_names)))
    display = ConfusionMatrixDisplay(matrix, display_labels=class_names)
    display.plot(xticks_rotation=45, values_format="d")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "confusion_matrix.png", dpi=150)
    plt.close()

    # Sort wrong predictions by confidence to find the model's strongest mistakes.
    wrong_indices = torch.where(predictions != labels)[0]
    confident_wrong_indices = wrong_indices[
        torch.argsort(confidences[wrong_indices], descending=True)
    ]
    show_image_grid(
        images[confident_wrong_indices],
        labels[confident_wrong_indices],
        predictions[confident_wrong_indices],
        confidences[confident_wrong_indices],
        class_names,
        "Confidently Wrong Predictions",
        FIGURE_DIR / "confidently_wrong_predictions.png",
    )

    # Low confidence means the model was least certain about its chosen class.
    low_confidence_indices = torch.argsort(confidences)
    show_image_grid(
        images[low_confidence_indices],
        labels[low_confidence_indices],
        predictions[low_confidence_indices],
        confidences[low_confidence_indices],
        class_names,
        "Low-Confidence Predictions",
        FIGURE_DIR / "low_confidence_predictions.png",
    )

    print(f"Saved per-class metrics: {REPORT_DIR / 'per_class_metrics.json'}")
    print(f"Saved figures: {FIGURE_DIR}")


if __name__ == "__main__":
    main()
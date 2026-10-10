from pathlib import Path

import numpy as np
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from torchvision.datasets import CIFAR10


CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
SPLIT_PATH = PROJECT_ROOT / "splits" / "cifar10_train_val_seed42.npz"

#Train transform      = random augmentation + conversion + normalization
#Evaluation transform = conversion + normalization only

def build_train_transform(use_augmentation: bool = True) -> transforms.Compose:
    # This experiment can turn crop and flip off while keeping normalization.
    if not use_augmentation:
        return build_evaluation_transform()

    return transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        ]
    )


def build_evaluation_transform() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        ]
    )


def load_split_indices() -> tuple[np.ndarray, np.ndarray]:
    with np.load(SPLIT_PATH) as split_data:
        train_indices = split_data["train_indices"]
        validation_indices = split_data["validation_indices"]

    return train_indices, validation_indices


def build_datasets(use_augmentation: bool = True) -> tuple[Subset, Subset, CIFAR10]:
    train_indices, validation_indices = load_split_indices()

    train_base_dataset = CIFAR10(
        root=DATA_DIR,
        train=True,
        transform=build_train_transform(use_augmentation),
        download=False,
    )
    validation_base_dataset = CIFAR10(
        root=DATA_DIR,
        train=True,
        transform=build_evaluation_transform(),
        download=False,
    )
    test_dataset = CIFAR10(
        root=DATA_DIR,
        train=False,
        transform=build_evaluation_transform(),
        download=False,
    )

    train_dataset = Subset(train_base_dataset, train_indices)
    validation_dataset = Subset(validation_base_dataset, validation_indices)

    return train_dataset, validation_dataset, test_dataset


def build_dataloaders(
    train_batch_size: int = 128,
    evaluation_batch_size: int = 256,
    num_workers: int = 0,
    use_augmentation: bool = True,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    train_dataset, validation_dataset, test_dataset = build_datasets(use_augmentation)

    train_loader = DataLoader(
        train_dataset,
        batch_size=train_batch_size,
        shuffle=True,
        num_workers=num_workers,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=evaluation_batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=evaluation_batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    return train_loader, validation_loader, test_loader
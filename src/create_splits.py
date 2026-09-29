import json
from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from torchvision.datasets import CIFAR10


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
SPLITS_DIR = PROJECT_ROOT / "splits"
SEED = 42
VALIDATION_SIZE = 5_000


def main() -> None:
    dataset = CIFAR10(root=DATA_DIR, train=True, download=False)
    labels = np.asarray(dataset.targets)

    splitter = StratifiedShuffleSplit(
        n_splits=1,
        test_size=VALIDATION_SIZE,
        random_state=SEED,
    )
    train_indices, validation_indices = next(
        splitter.split(np.zeros(len(labels)), labels)
    )

    if len(train_indices) != 45_000 or len(validation_indices) != VALIDATION_SIZE:
        raise ValueError("The split does not have the expected 45,000/5,000 sizes.")
    if np.intersect1d(train_indices, validation_indices).size != 0:
        raise ValueError("Training and validation indices overlap.")

    train_counts = np.bincount(labels[train_indices], minlength=len(dataset.classes))
    validation_counts = np.bincount(
        labels[validation_indices], minlength=len(dataset.classes)
    )
    if not np.array_equal(train_counts, np.full(len(dataset.classes), 4_500)):
        raise ValueError("The training split is not class-balanced.")
    if not np.array_equal(validation_counts, np.full(len(dataset.classes), 500)):
        raise ValueError("The validation split is not class-balanced.")

    SPLITS_DIR.mkdir(exist_ok=True)
    split_path = SPLITS_DIR / f"cifar10_train_val_seed{SEED}.npz"
    metadata_path = SPLITS_DIR / f"cifar10_train_val_seed{SEED}.json"

    np.savez_compressed(
        split_path,
        train_indices=train_indices,
        validation_indices=validation_indices,
    )
    metadata_path.write_text(
        json.dumps(
            {
                "dataset": "CIFAR-10 official training split",
                "seed": SEED,
                "train_size": len(train_indices),
                "validation_size": len(validation_indices),
                "class_names": dataset.classes,
                "train_class_counts": train_counts.tolist(),
                "validation_class_counts": validation_counts.tolist(),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    print(f"Saved training indices: {split_path}")
    print(f"Saved split metadata: {metadata_path}")
    print(f"Training examples: {len(train_indices)}")
    print(f"Validation examples: {len(validation_indices)}")
    print(f"Training examples per class: {train_counts.tolist()}")
    print(f"Validation examples per class: {validation_counts.tolist()}")


if __name__ == "__main__":
    main()
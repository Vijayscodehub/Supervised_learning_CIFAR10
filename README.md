# CIFAR-10 Supervised Learning Project

## Goal

Build a complete, reproducible image-classification experiment using CIFAR-10 and PyTorch. The goal is not only to train a CNN, but to understand the entire supervised-learning workflow:

```text
Data -> split -> preprocessing -> model -> training -> validation
     -> hyperparameter selection -> final test -> error analysis
```

At the end, you should be able to explain:

> I defined the problem, created a reproducible train/validation/test protocol, trained a CNN that I designed, selected decisions using validation data, evaluated once on untouched test data, and analyzed where the model failed.

## What This Project Is

- A PyTorch project.
- A CNN that you define yourself with convolution, pooling, dropout, and fully connected layers.
- A manually written training loop and validation loop.
- A reproducible experiment with TensorBoard tracking.

## What This Project Is Not

- Not a pretrained model, YOLO, ResNet, or transfer-learning project.
- Not an Ultralytics-style one-command training workflow.
- Not a TensorFlow/Keras `model.fit()` project.
- Not a fully manual backpropagation implementation.

PyTorch will calculate gradients and update parameters when you call:

```python
loss.backward()
optimizer.step()
```

A later NumPy MLP project will manually create weights and biases, calculate gradients, and implement gradient descent. This project teaches the reliable ML workflow around a neural network.

## Dataset

CIFAR-10 contains 60,000 color images of shape `3 x 32 x 32` across 10 classes:

```text
airplane, automobile, bird, cat, deer,
dog, frog, horse, ship, truck
```

The official dataset has:

```text
50,000 training images
10,000 test images
```

Use the supplied training images to make a reproducible split:

```text
50,000 official training images
    -> 45,000 training images
    ->  5,000 validation images

10,000 official test images
    -> untouched until final evaluation
```

CIFAR-10 images are independent examples, so a random stratified split is appropriate. For video or driving data, you would split by scene or sequence instead to prevent leakage.

## Learning Objectives

By the end, you should understand:

- The distinct purposes of training, validation, and test data.
- Why the test set must not guide architecture or hyperparameter choices.
- How to create a PyTorch `Dataset` and `DataLoader`.
- How images are normalized and augmented.
- How a CNN converts an image into class scores.
- What cross-entropy loss, an optimizer, batches, epochs, and learning rate do.
- How to read training and validation curves.
- How to detect underfitting and overfitting.
- How to evaluate a classifier beyond one accuracy number.

## Project Milestones

### 1. Set Up the Environment

Create a Python environment with PyTorch, Torchvision, TensorBoard, Matplotlib, Scikit-learn, and Pandas.

Confirm whether CUDA is available, but make the project work on CPU too.

### 2. Explore the Dataset

Before training, create a notebook that:

- Displays a grid of sample images and labels.
- Confirms image shape and label format.
- Counts the examples in every class.
- Shows why the problem is challenging: images are small and several classes are visually similar, such as cats versus dogs and automobiles versus trucks.

### 3. Create Reproducible Splits

- Download the official CIFAR-10 training and test sets with Torchvision.
- Split the official training set into 45,000 training and 5,000 validation images.
- Use a fixed random seed.
- Use a stratified split so all classes remain similarly represented.
- Save the train and validation indices so future runs use exactly the same split.
- Do not inspect test metrics while choosing the model or hyperparameters.

### 4. Preprocess and Augment Data

Use separate transformations for training and evaluation.

Training transformation:

- Convert image to a tensor.
- Apply small random crop with padding.
- Apply random horizontal flip.
- Normalize with CIFAR-10 channel mean and standard deviation.

Validation and test transformation:

- Convert image to a tensor.
- Normalize with the same training-data statistics.
- Do not apply random augmentations.

Avoid vertical flips because upside-down objects are not representative of normal CIFAR-10 images. Start with only crop and horizontal flip; add or remove one augmentation at a time during experiments.

With PyTorch `nn.CrossEntropyLoss`, labels should remain class integers from `0` to `9`. Do not one-hot encode labels.

The training and evaluation transforms are defined in `src/data.py`.

### 5. Build Datasets and DataLoaders

Create PyTorch `Dataset` objects for training, validation, and test data.

Create `DataLoader` objects with:

- `shuffle=True` for training.
- `shuffle=False` for validation and test.
- A configurable batch size, initially `128` for training and `256` for evaluation.
- A configurable number of workers.

The split-aware datasets and loaders are defined in `src/data.py`.

### 6. Establish a Baseline

Build a small CNN yourself. Begin with two convolution blocks:

```text
Input image: 3 x 32 x 32
    -> convolution + ReLU + max pooling
    -> convolution + ReLU + max pooling
    -> flatten
    -> fully connected layer + ReLU + dropout
    -> fully connected output layer with 10 logits
```

Your model will use PyTorch layers such as:

```python
nn.Conv2d
nn.ReLU
nn.MaxPool2d
nn.Dropout
nn.Linear
```

Do not apply softmax inside the model when using `nn.CrossEntropyLoss`; it expects raw logits.

The CNN baseline is defined in `src/model.py`.

### 7. Write the Training Loop

Write the loop yourself rather than using a high-level trainer:

```python
model.train()
for images, labels in train_loader:
    optimizer.zero_grad()
    logits = model(images)
    loss = criterion(logits, labels)
    loss.backward()
    optimizer.step()
```

This step teaches the full optimization process behind a CNN classifier:

- Set the model to training mode so layers like dropout behave correctly.
- Pull the next batch from the `DataLoader`.
- Zero the gradients so old values do not accumulate across batches.
- Run the forward pass to compute logits.
- Compute the loss between predictions and the true class labels.
- Call `loss.backward()` to compute gradients for each weight and bias.
- Update parameters with `optimizer.step()`.

For the baseline, use:

- Loss: `nn.CrossEntropyLoss()`.
- Optimizer: `AdamW`.
- Initial learning rate: `1e-3`.
- Training length: start with 20 to 30 epochs.
- Checkpoint: save the model with the best validation accuracy or lowest validation loss.

The training loop for this project is implemented in `src/train.py`.

### 8. Write the Validation Loop

At the end of every epoch:

- Set `model.eval()`.
- Disable gradient calculation with `torch.no_grad()`.
- Calculate validation loss and validation accuracy.
- Save the best checkpoint based on the selected validation metric.

Validation data is for making decisions such as:

- Which CNN architecture to use.
- Whether augmentation helps.
- Which learning rate works better.
- Whether dropout helps.
- When to stop training.

### 9. Track Experiments in TensorBoard

Log these values every epoch:

- Training loss.
- Validation loss.
- Training accuracy.
- Validation accuracy.
- Learning rate.

Use TensorBoard to inspect whether the model is:

```text
Underfitting:  training and validation results are both poor.
Overfitting:   training improves while validation stagnates or worsens.
Generalizing:  validation improves and remains reasonably close to training.
```

### 10. Tune One Question at a Time

Use validation results only. Create a small experiment table like this:

| ID | Change | Validation result | What it tells us |
| --- | --- | --- | --- |
| 0 | Crop and horizontal-flip augmentation | 77.66% accuracy, Epoch 20 | Augmented baseline |
| 1 | No augmentation | 74.22% accuracy, Epoch 8 | Augmentation improved validation accuracy |
| 2 | Seeded crop and horizontal-flip baseline, learning rate 0.001 | 77.74% accuracy, Epoch 20 | Reference run for learning-rate comparison |
| 3 | Change learning rate | lr 0.0003 - 74.18% accuracy, Epoch	18 | Is optimization more stable?  - worsened|
| 4 | Add or change dropout | | Does regularization help? |
| 5 | Compare AdamW and SGD with momentum | | Does optimizer choice matter? |

Do not change many things in one experiment. Each run should answer one clear question.

### 11. Perform One Final Test Evaluation

After selecting the final configuration using validation data:

1. Load the best saved checkpoint.
2. Evaluate on the untouched 10,000-image official test set.
3. Record the test loss and accuracy.
4. Do not keep changing the model based on test results.

The test result estimates how well the final chosen model generalizes to unseen CIFAR-10 data.

### 12. Analyze Errors

Accuracy is not enough. Create:

- A confusion matrix.
- Per-class precision, recall, and F1 score.
- A grid of confidently wrong predictions.
- A grid of low-confidence predictions.
- Notes on common confusions, such as cat versus dog or automobile versus truck.

Answer questions such as:

- Which classes are hardest?
- Are errors caused by small object size, background clutter, pose, or class similarity?
- Does augmentation improve every class or only some classes?
- Is validation performance consistent with test performance?

## Suggested Project Structure

```text
Supervised_learning_CIFAR10/
├── README.md
├── requirements.txt
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   └── 02_error_analysis.ipynb
├── src/
│   ├── data.py
│   ├── model.py
│   ├── train.py
│   ├── evaluate.py
│   └── utils.py
├── configs/
│   └── baseline.yaml
├── splits/
│   ├── cifar10_train_val_seed42.json
│   └── cifar10_train_val_seed42.npz
├── checkpoints/
├── runs/
└── reports/
    ├── figures/
    └── results.md
```

Use notebooks for data exploration and result visualization. Keep the actual training, validation, checkpointing, and final testing code in Python files so runs are repeatable.

## Definition of Done

The project is complete when you can show:

- A fixed, reproducible 45k/5k/10k split.
- A CNN that you defined yourself.
- Your own PyTorch training and validation loops.
- TensorBoard learning curves.
- A small validation-based experiment table.
- A final test result evaluated once after selecting the model.
- A confusion matrix, per-class metrics, and qualitative error analysis.
- A README or report that explains the results and limitations honestly.

## Next Step

Start with Milestone 1: set up the environment, download CIFAR-10 using Torchvision, and verify the dataset shapes and labels before writing the CNN.

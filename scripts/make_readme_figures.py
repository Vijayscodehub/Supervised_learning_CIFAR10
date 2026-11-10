"""Generate the README figures for the CIFAR-10 project.

Reads tracked reports, local TensorBoard logs in runs/ and the local CIFAR-10
copy in data/. Writes PNGs to reports/figures/readme/.

    pip install matplotlib numpy pillow tensorboard
    python scripts/make_readme_figures.py
"""
import glob
import json
import pickle
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports" / "figures" / "readme"
OUT.mkdir(parents=True, exist_ok=True)

CLASSES = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
INK, MUTED, GRID = "#1f2933", "#6b7785", "#e4e7eb"
BLUE, TEAL, AMBER, RED, GREY, PURPLE = "#2f6fdf", "#1a9e8f", "#e0a030", "#d64545", "#aab2bd", "#7a5cc7"

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10.5,
    "axes.edgecolor": GRID, "axes.labelcolor": INK, "axes.titlecolor": INK,
    "axes.titleweight": "bold", "axes.titlesize": 12.5, "axes.titlelocation": "left",
    "axes.spines.top": False, "axes.spines.right": False,
    "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "figure.facecolor": "white", "axes.facecolor": "white",
    "savefig.dpi": 160, "savefig.bbox": "tight",
})


def save(fig, name):
    fig.savefig(OUT / name)
    plt.close(fig)
    print("wrote", name)


# --------------------------------------------------------------------- experiments
EXPERIMENTS = [  # (run dir, label, colour)
    ("baseline", "0 · augmented baseline", GREY),
    ("no_augmentation", "1 · no augmentation", RED),
    ("augmented_lr_1e-3_seed42", "2 · lr 1e-3, seed 42", BLUE),
    ("augmented_lr_3e-4_seed42", "3 · lr 3e-4, seed 42", AMBER),
    ("augmented_dropout0_0_1e-3_seed42", "4 · dropout 0.0 (selected)", TEAL),
]


def load_run(name):
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    # Use the newest event file (lr 3e-4 has an earlier aborted 5-epoch attempt).
    path = sorted(glob.glob(str(ROOT / "runs" / name / "events.*")))[-1]
    acc = EventAccumulator(path)
    acc.Reload()
    return {tag: np.array([s.value for s in acc.Scalars(tag)]) for tag in acc.Tags()["scalars"]}


runs = {name: load_run(name) for name, _, _ in EXPERIMENTS}

# 1. Learning curves of the selected run
sel = runs["augmented_dropout0_0_1e-3_seed42"]
ep = np.arange(1, len(sel["Loss/train"]) + 1)
best = int(np.argmax(sel["Accuracy/validation"])) + 1
fig, axes = plt.subplots(1, 2, figsize=(11.5, 3.9))
for ax, key, title, fmt in [(axes[0], "Loss", "Loss", "{:.3f}"), (axes[1], "Accuracy", "Accuracy", "{:.1%}")]:
    tr, va = sel[f"{key}/train"], sel[f"{key}/validation"]
    if key == "Accuracy":
        tr, va = tr * 100, va * 100
    ax.plot(ep, tr, color=BLUE, lw=2.2, marker="o", ms=3.5, label="training")
    ax.plot(ep, va, color=TEAL, lw=2.2, marker="o", ms=3.5, label="validation")
    ax.axvline(best, color=MUTED, ls="--", lw=1)
    lo, hi = ax.get_ylim()
    ax.text(best - 0.3, lo + (hi - lo) * (0.92 if key == "Loss" else 0.06),
            f"best val\nepoch {best}", color=MUTED, fontsize=9, ha="right",
            va="top" if key == "Loss" else "bottom")
    ax.set_xlabel("Epoch")
    ax.set_ylabel(title + (" (%)" if key == "Accuracy" else ""))
    ax.set_title(f"{title}: selected model")
    ax.set_xticks([1, 5, 10, 15, 20])
    ax.legend(frameon=False, loc="center right")
save(fig, "learning_curves.png")

# 2. Validation accuracy of every experiment + best-accuracy bars
fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.2), gridspec_kw={"width_ratios": [1.45, 1]})
ax = axes[0]
for name, label, col in EXPERIMENTS:
    va = runs[name]["Accuracy/validation"] * 100
    ax.plot(np.arange(1, len(va) + 1), va, color=col, lw=2.6 if "selected" in label else 1.6,
            label=label, zorder=3 if "selected" in label else 2)
ax.set_xlabel("Epoch")
ax.set_ylabel("Validation accuracy (%)")
ax.set_title("Validation accuracy per epoch")
ax.set_xticks([1, 5, 10, 15, 20])
ax.legend(frameon=False, fontsize=9, loc="lower right")

ax = axes[1]
bests = [(label, runs[name]["Accuracy/validation"].max() * 100, int(np.argmax(runs[name]["Accuracy/validation"])) + 1, col)
         for name, label, col in EXPERIMENTS]
y = np.arange(len(bests))[::-1]
ax.barh(y, [b[1] for b in bests], color=[b[3] for b in bests], height=0.6)
for yi, (label, v, e, _) in zip(y, bests):
    ax.text(v + 0.15, yi, f"{v:.2f}%  (ep {e})", va="center", color=INK, fontsize=9.5)
ax.set_yticks(y, [b[0] for b in bests])
ax.set_xlim(72, 80.5)
ax.set_xlabel("Best validation accuracy (%)")
ax.set_title("Best epoch per experiment")
ax.grid(axis="y", visible=False)
fig.tight_layout()
save(fig, "experiment_comparison.png")

# 3. Overfitting check: train vs val loss with and without augmentation
fig, axes = plt.subplots(1, 2, figsize=(11.5, 3.6), sharey=True)
for ax, name, title in [(axes[0], "no_augmentation", "No augmentation: overfits after epoch 8"),
                        (axes[1], "augmented_dropout0_0_1e-3_seed42", "Crop + flip: train and val stay close")]:
    r = runs[name]
    e = np.arange(1, len(r["Loss/train"]) + 1)
    ax.plot(e, r["Loss/train"], color=BLUE, lw=2.2, label="training loss")
    ax.plot(e, r["Loss/validation"], color=RED if name == "no_augmentation" else TEAL, lw=2.2, label="validation loss")
    ax.fill_between(e, r["Loss/train"], r["Loss/validation"], where=r["Loss/validation"] > r["Loss/train"],
                    color=RED, alpha=0.08)
    ax.set_title(title, fontsize=11.5)
    ax.set_xlabel("Epoch")
    ax.set_xticks([1, 5, 10, 15, 20])
    ax.legend(frameon=False)
axes[0].set_ylabel("Cross-entropy loss")
fig.tight_layout()
save(fig, "augmentation_overfitting.png")

# --------------------------------------------------------------------- test metrics
metrics = json.load(open(ROOT / "reports" / "per_class_metrics.json"))
fig, ax = plt.subplots(figsize=(12, 4))
order = sorted(CLASSES, key=lambda c: metrics[c]["f1-score"])
x = np.arange(len(order))
w = 0.27
for k, (key, col, lab) in enumerate([("precision", BLUE, "precision"), ("recall", AMBER, "recall"), ("f1-score", TEAL, "F1")]):
    ax.bar(x + (k - 1) * w, [metrics[c][key] * 100 for c in order], w, color=col, label=lab)
for i, c in enumerate(order):
    ax.text(i + w, metrics[c]["f1-score"] * 100 + 1, f"{metrics[c]['f1-score'] * 100:.0f}", ha="center",
            fontsize=8.5, color=TEAL, fontweight="bold")
ax.axhline(metrics["accuracy"] * 100, color=INK, ls="--", lw=1)
ax.text(-0.55, metrics["accuracy"] * 100 + 0.8, f"overall accuracy {metrics['accuracy']:.2%}",
        ha="left", color=INK, fontsize=9)
ax.set_xticks(x, order)
ax.set_ylim(50, 100)
ax.set_ylabel("%")
ax.set_title("Per-class test metrics, sorted by F1 (10,000 test images)")
ax.grid(axis="x", visible=False)
ax.legend(frameon=False, ncols=3, loc="upper left")
save(fig, "per_class_metrics.png")

# Confusion matrix counts transcribed from reports/figures/confusion_matrix.png
# (produced by src/analyze_errors.py). Checked below against per-class recall.
CM = np.array([
    [804, 15, 25, 10, 11, 3, 12, 6, 62, 52],
    [11, 877, 2, 5, 1, 3, 1, 2, 21, 77],
    [72, 3, 610, 50, 75, 72, 69, 20, 15, 14],
    [25, 7, 50, 575, 46, 160, 74, 28, 10, 25],
    [24, 3, 37, 48, 775, 33, 32, 32, 11, 5],
    [9, 4, 25, 164, 34, 700, 14, 33, 8, 9],
    [7, 4, 24, 41, 38, 21, 852, 2, 7, 4],
    [17, 3, 24, 36, 43, 64, 4, 789, 5, 15],
    [39, 17, 5, 11, 3, 3, 2, 2, 897, 21],
    [21, 56, 3, 7, 4, 0, 4, 2, 25, 878],
])
assert (CM.sum(1) == 1000).all(), CM.sum(1)
for i, c in enumerate(CLASSES):
    assert CM[i, i] == round(metrics[c]["recall"] * 1000), c
    assert abs(CM[i, i] / CM[:, i].sum() - metrics[c]["precision"]) < 1e-9, c

# 4. Row-normalised confusion matrix
fig, ax = plt.subplots(figsize=(7.6, 6.4))
pct = CM / CM.sum(1, keepdims=True) * 100
im = ax.imshow(pct, cmap="Blues", vmin=0, vmax=100)
for i in range(10):
    for j in range(10):
        v = pct[i, j]
        if v >= 0.5:
            ax.text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=9,
                    color="white" if v > 55 else (RED if i != j and v >= 10 else INK),
                    fontweight="bold" if i != j and v >= 10 else "normal")
ax.set_xticks(range(10), CLASSES, rotation=40, ha="right")
ax.set_yticks(range(10), CLASSES)
ax.set_xlabel("Predicted class")
ax.set_ylabel("True class")
ax.grid(False)
ax.set_title("Confusion matrix (% of each true class)")
for s in ax.spines.values():
    s.set_visible(False)
cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
cb.outline.set_visible(False)
save(fig, "confusion_matrix_normalized.png")

# 5. Top confusions
pairs = sorted(((CM[i, j], CLASSES[i], CLASSES[j]) for i in range(10) for j in range(10) if i != j), reverse=True)[:10]
fig, ax = plt.subplots(figsize=(8, 4.2))
y = np.arange(len(pairs))[::-1]
animals = {"bird", "cat", "deer", "dog", "frog", "horse"}
cols = [PURPLE if (t in animals and p in animals) else (BLUE if (t not in animals and p not in animals) else AMBER)
        for _, t, p in pairs]
ax.barh(y, [p[0] for p in pairs], color=cols, height=0.62)
for yi, (n, t, p) in zip(y, pairs):
    ax.text(n + 2, yi, str(n), va="center", color=INK, fontsize=9.5)
ax.set_yticks(y, [f"{t} → {p}" for _, t, p in pairs])
ax.set_xlabel("Test images (true → predicted)")
ax.set_title("Ten most frequent mistakes")
ax.grid(axis="y", visible=False)
from matplotlib.patches import Patch
ax.legend(handles=[Patch(color=PURPLE, label="animal ↔ animal"), Patch(color=BLUE, label="vehicle ↔ vehicle"),
                   Patch(color=AMBER, label="animal ↔ vehicle")], frameon=False, loc="lower right")
save(fig, "top_confusions.png")

# --------------------------------------------------------------------- dataset images
def load_batch(name):
    with open(ROOT / "data" / "cifar-10-batches-py" / name, "rb") as f:
        d = pickle.load(f, encoding="bytes")
    return d[b"data"].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1), np.array(d[b"labels"])


images, labels = load_batch("data_batch_1")
rng = np.random.default_rng(42)

# 6. Class sample grid
per = 10
fig, axes = plt.subplots(10, per + 1, figsize=(per * 0.95 + 1.6, 10 * 0.95),
                         gridspec_kw={"width_ratios": [2.2] + [1] * per})
for r, c in enumerate(CLASSES):
    idx = rng.choice(np.flatnonzero(labels == r), per, replace=False)
    axes[r, 0].axis("off")
    axes[r, 0].text(1, 0.5, c, ha="right", va="center", fontsize=12, color=INK, fontweight="bold",
                    transform=axes[r, 0].transAxes)
    for k, i in enumerate(idx):
        axes[r, k + 1].imshow(images[i], interpolation="nearest")
        axes[r, k + 1].axis("off")
fig.subplots_adjust(wspace=0.06, hspace=0.06)
save(fig, "cifar10_samples.png")

# 7. Augmentation demo: original vs random crop (pad 4) + horizontal flip
picks = [np.flatnonzero(labels == CLASSES.index(c))[3] for c in ["automobile", "cat", "horse", "ship"]]
fig, axes = plt.subplots(len(picks), 6, figsize=(8.2, 5.8))
for r, i in enumerate(picks):
    img = images[i]
    axes[r, 0].imshow(img, interpolation="nearest")
    axes[r, 0].set_title("original" if r == 0 else "", fontsize=10, color=INK)
    padded = np.pad(img, ((4, 4), (4, 4), (0, 0)))
    for k in range(1, 6):
        dy, dx = rng.integers(0, 9, 2)
        aug = padded[dy:dy + 32, dx:dx + 32]
        if rng.random() < 0.5:
            aug = aug[:, ::-1]
        axes[r, k].imshow(aug, interpolation="nearest")
        if r == 0:
            axes[r, k].set_title(f"augmented {k}", fontsize=10, color=MUTED)
for a in axes.flat:
    a.set_xticks([]); a.set_yticks([]); a.grid(False)
    for s in a.spines.values():
        s.set_visible(False)
for r in range(len(picks)):
    for s in axes[r, 0].spines.values():
        s.set_visible(True); s.set_color(TEAL); s.set_linewidth(2)
fig.suptitle("Training augmentation: RandomCrop(32, padding=4) + RandomHorizontalFlip",
             x=0.02, ha="left", fontweight="bold", color=INK, fontsize=11.5)
fig.tight_layout()
save(fig, "augmentation_examples.png")

for name, label, _ in EXPERIMENTS:
    r = runs[name]
    i = int(np.argmax(r["Accuracy/validation"]))
    print(f"{label:32s} best ep {i + 1:2d} val {r['Accuracy/validation'][i]:.4f} "
          f"train {r['Accuracy/train'][i]:.4f} final-ep train {r['Accuracy/train'][-1]:.4f} "
          f"val {r['Accuracy/validation'][-1]:.4f} | val loss min ep {int(np.argmin(r['Loss/validation'])) + 1}")

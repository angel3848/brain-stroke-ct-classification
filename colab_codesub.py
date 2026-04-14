"""
Brain Stroke CT Classification — Google Colab Version
Combined: ResNet-50 Fine-tuning + Traditional ML Classifiers (Logistic Regression, Random Forest, SVM)
Bleeding vs Ischemia vs Normal

Usage: Upload/download your dataset to /content/data/Brain_Stroke_CT_Dataset/ then run all cells.
"""

# ============================================================
# Cell 1: Install & Setup
# ============================================================
# !pip install -q tqdm seaborn scikit-learn
#
# If your dataset is on Kaggle, uncomment and set your credentials:
# import os
# os.environ['KAGGLE_USERNAME'] = 'your_username'
# os.environ['KAGGLE_KEY'] = 'your_key'
# !kaggle datasets download -d <dataset-path> -p /content/data --unzip

# ============================================================
# Cell 2: Imports
# ============================================================
import os
import json
import time
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from PIL import Image
from collections import Counter

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms, models
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_curve, auc,
    precision_recall_curve, average_precision_score, f1_score,
    accuracy_score, cohen_kappa_score
)
from tqdm import tqdm

# ============================================================
# Cell 3: Configuration — UPDATE THESE PATHS FOR YOUR SETUP
# ============================================================
SEED = 42
DATA_ROOT = Path("/content/data/Brain_Stroke_CT_Dataset")  # <-- Update to your upload path
OUTPUT_DIR = Path("/content/results")
CLF_OUTPUT_DIR = Path("/content/results/classifiers")
OUTPUT_DIR.mkdir(exist_ok=True)
CLF_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CLASS_NAMES = ["Bleeding", "Ischemia", "Normal"]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASS_NAMES)}
NUM_CLASSES = 3

BATCH_SIZE = 32
NUM_EPOCHS = 30
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
IMG_SIZE = 224
PATIENCE = 7

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


set_seed(SEED)

# ============================================================
# Cell 4: Dataset Class
# ============================================================
class StrokeCTDataset(Dataset):
    def __init__(self, file_paths, labels, transform=None):
        self.file_paths = file_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        img = Image.open(self.file_paths[idx]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, self.labels[idx]


def load_dataset():
    """Load PNG file paths and labels for the 3 classes."""
    paths, labels = [], []
    for cls_name in CLASS_NAMES:
        cls_dir = DATA_ROOT / cls_name / "PNG"
        for f in sorted(cls_dir.glob("*.png")):
            paths.append(str(f))
            labels.append(CLASS_TO_IDX[cls_name])
    return paths, labels


# ============================================================
# Cell 5: Transforms
# ============================================================
train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(15),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# ============================================================
# Cell 6: Model Builder
# ============================================================
def build_model():
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(num_ftrs, NUM_CLASSES)
    )
    return model.to(DEVICE)


# ============================================================
# Cell 7: Training & Evaluation Functions
# ============================================================
def train_one_epoch(model, loader, criterion, optimizer):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []

    for images, labels in tqdm(loader, desc="  Train", leave=False):
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    epoch_f1 = f1_score(all_labels, all_preds, average='macro')
    return epoch_loss, epoch_acc, epoch_f1


@torch.no_grad()
def evaluate(model, loader, criterion):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels, all_probs = [], [], []

    for images, labels in tqdm(loader, desc="  Val  ", leave=False):
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        outputs = model(images)
        loss = criterion(outputs, labels)

        running_loss += loss.item() * images.size(0)
        probs = torch.softmax(outputs, dim=1)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    epoch_f1 = f1_score(all_labels, all_preds, average='macro')
    return epoch_loss, epoch_acc, epoch_f1, np.array(all_preds), np.array(all_labels), np.array(all_probs)


# ============================================================
# Cell 8: Plotting Functions
# ============================================================
def plot_training_curves(history):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    epochs = range(1, len(history['train_loss']) + 1)

    axes[0].plot(epochs, history['train_loss'], 'b-o', label='Train', markersize=3)
    axes[0].plot(epochs, history['val_loss'], 'r-o', label='Validation', markersize=3)
    axes[0].set_title('Loss', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('Loss')
    axes[0].legend(); axes[0].grid(True, alpha=0.3)

    axes[1].plot(epochs, history['train_acc'], 'b-o', label='Train', markersize=3)
    axes[1].plot(epochs, history['val_acc'], 'r-o', label='Validation', markersize=3)
    axes[1].set_title('Accuracy', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Epoch'); axes[1].set_ylabel('Accuracy')
    axes[1].legend(); axes[1].grid(True, alpha=0.3)

    axes[2].plot(epochs, history['train_f1'], 'b-o', label='Train', markersize=3)
    axes[2].plot(epochs, history['val_f1'], 'r-o', label='Validation', markersize=3)
    axes[2].set_title('Macro F1-Score', fontsize=14, fontweight='bold')
    axes[2].set_xlabel('Epoch'); axes[2].set_ylabel('F1-Score')
    axes[2].legend(); axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "training_curves.png", dpi=300, bbox_inches='tight')
    plt.show()


def plot_confusion_matrix(y_true, y_pred, title="Confusion Matrix", filename="confusion_matrix.png", save_dir=None):
    if save_dir is None:
        save_dir = OUTPUT_DIR
    cm = confusion_matrix(y_true, y_pred)
    cm_pct = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=CLASS_NAMES,
                yticklabels=CLASS_NAMES, ax=axes[0])
    axes[0].set_title(f'{title} (Counts)', fontsize=13, fontweight='bold')
    axes[0].set_ylabel('True Label'); axes[0].set_xlabel('Predicted Label')

    sns.heatmap(cm_pct, annot=True, fmt='.1f', cmap='Blues', xticklabels=CLASS_NAMES,
                yticklabels=CLASS_NAMES, ax=axes[1])
    axes[1].set_title(f'{title} (Percentages %)', fontsize=13, fontweight='bold')
    axes[1].set_ylabel('True Label'); axes[1].set_xlabel('Predicted Label')

    plt.tight_layout()
    plt.savefig(save_dir / filename, dpi=300, bbox_inches='tight')
    plt.show()
    return cm


def plot_roc_curves(y_true, y_probs, title="ROC Curves", filename="roc_curves.png", save_dir=None):
    if save_dir is None:
        save_dir = OUTPUT_DIR
    y_bin = label_binarize(y_true, classes=[0, 1, 2])
    fig, ax = plt.subplots(figsize=(8, 7))
    colors = ['#e74c3c', '#3498db', '#2ecc71']

    auc_scores = {}
    for i, (cls, color) in enumerate(zip(CLASS_NAMES, colors)):
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_probs[:, i])
        roc_auc = auc(fpr, tpr)
        auc_scores[cls] = roc_auc
        ax.plot(fpr, tpr, color=color, lw=2, label=f'{cls} (AUC = {roc_auc:.4f})')

    fpr_micro, tpr_micro, _ = roc_curve(y_bin.ravel(), y_probs.ravel())
    auc_micro = auc(fpr_micro, tpr_micro)
    ax.plot(fpr_micro, tpr_micro, color='black', lw=2, linestyle='--',
            label=f'Micro-avg (AUC = {auc_micro:.4f})')

    ax.plot([0, 1], [0, 1], 'k--', alpha=0.3)
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_dir / filename, dpi=300, bbox_inches='tight')
    plt.show()
    return auc_scores, auc_micro


def plot_precision_recall_curves(y_true, y_probs, filename="pr_curves.png", save_dir=None):
    if save_dir is None:
        save_dir = OUTPUT_DIR
    y_bin = label_binarize(y_true, classes=[0, 1, 2])
    fig, ax = plt.subplots(figsize=(8, 7))
    colors = ['#e74c3c', '#3498db', '#2ecc71']

    ap_scores = {}
    for i, (cls, color) in enumerate(zip(CLASS_NAMES, colors)):
        precision, recall, _ = precision_recall_curve(y_bin[:, i], y_probs[:, i])
        ap = average_precision_score(y_bin[:, i], y_probs[:, i])
        ap_scores[cls] = ap
        ax.plot(recall, precision, color=color, lw=2, label=f'{cls} (AP = {ap:.4f})')

    ax.set_xlabel('Recall', fontsize=12)
    ax.set_ylabel('Precision', fontsize=12)
    ax.set_title('Precision-Recall Curves', fontsize=14, fontweight='bold')
    ax.legend(loc='lower left', fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_dir / filename, dpi=300, bbox_inches='tight')
    plt.show()
    return ap_scores


def plot_class_distribution(train_labels, val_labels, test_labels):
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for ax, labels, title in zip(axes,
                                  [train_labels, val_labels, test_labels],
                                  ['Training Set', 'Validation Set', 'Test Set']):
        counts = Counter(labels)
        bars = ax.bar(CLASS_NAMES, [counts.get(i, 0) for i in range(NUM_CLASSES)],
                       color=['#e74c3c', '#3498db', '#2ecc71'])
        ax.set_title(title, fontsize=13, fontweight='bold')
        ax.set_ylabel('Count')
        for bar, count in zip(bars, [counts.get(i, 0) for i in range(NUM_CLASSES)]):
            ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 20,
                    str(count), ha='center', fontsize=11, fontweight='bold')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "class_distribution.png", dpi=300, bbox_inches='tight')
    plt.show()


def plot_sample_predictions(model, dataset, indices, filename="sample_predictions.png"):
    model.eval()
    fig, axes = plt.subplots(3, 6, figsize=(20, 10))
    inv_normalize = transforms.Normalize(
        mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
        std=[1/0.229, 1/0.224, 1/0.225]
    )

    for ax_idx, data_idx in enumerate(indices[:18]):
        row, col = ax_idx // 6, ax_idx % 6
        img_tensor, true_label = dataset[data_idx]
        img_show = inv_normalize(img_tensor).permute(1, 2, 0).numpy().clip(0, 1)

        with torch.no_grad():
            output = model(img_tensor.unsqueeze(0).to(DEVICE))
            prob = torch.softmax(output, dim=1)
            pred_label = prob.argmax(1).item()
            confidence = prob[0, pred_label].item()

        axes[row][col].imshow(img_show)
        color = 'green' if pred_label == true_label else 'red'
        axes[row][col].set_title(
            f'True: {CLASS_NAMES[true_label]}\nPred: {CLASS_NAMES[pred_label]} ({confidence:.1%})',
            fontsize=9, color=color, fontweight='bold'
        )
        axes[row][col].axis('off')

    plt.suptitle('Sample Predictions (Green=Correct, Red=Incorrect)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / filename, dpi=300, bbox_inches='tight')
    plt.show()


def plot_model_comparison(all_results):
    models_list = list(all_results.keys())
    metrics = ['accuracy', 'macro_f1', 'cohen_kappa']
    labels = ['Accuracy', 'Macro F1', "Cohen's Kappa"]
    colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12']

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(metrics))
    width = 0.2

    for i, model_name in enumerate(models_list):
        vals = [all_results[model_name][m] for m in metrics]
        bars = ax.bar(x + i * width, vals, width, label=model_name, color=colors[i])
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + 0.005,
                    f'{val:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Model Comparison on Test Set', fontsize=14, fontweight='bold')
    ax.set_xticks(x + width * (len(models_list) - 1) / 2)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylim(0, 1.1)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(CLF_OUTPUT_DIR / "model_comparison.png", dpi=300, bbox_inches='tight')
    plt.show()


# ============================================================
# Cell 9: Load Data & Split
# ============================================================
print("Loading dataset...")
all_paths, all_labels = load_dataset()
print(f"Total images: {len(all_paths)}")
print(f"Class distribution: {Counter(all_labels)}")

# Stratified split: 70% train, 15% val, 15% test
train_paths, temp_paths, train_labels, temp_labels = train_test_split(
    all_paths, all_labels, test_size=0.30, random_state=SEED, stratify=all_labels
)
val_paths, test_paths, val_labels, test_labels = train_test_split(
    temp_paths, temp_labels, test_size=0.50, random_state=SEED, stratify=temp_labels
)

print(f"Train: {len(train_paths)} | Val: {len(val_paths)} | Test: {len(test_paths)}")
plot_class_distribution(train_labels, val_labels, test_labels)

# ============================================================
# Cell 10: Create Datasets & Loaders
# ============================================================
train_dataset = StrokeCTDataset(train_paths, train_labels, train_transform)
val_dataset = StrokeCTDataset(val_paths, val_labels, val_transform)
test_dataset = StrokeCTDataset(test_paths, test_labels, val_transform)

class_counts = Counter(train_labels)
weights = [1.0 / class_counts[label] for label in train_labels]
sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=sampler, num_workers=2)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

# ============================================================
# Cell 11: Build Model & Train ResNet-50
# ============================================================
print("Building ResNet-50 (pretrained ImageNet)...")
model = build_model()
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total params: {total_params:,} | Trainable: {trainable_params:,}")

# Class-weighted loss
class_weights = torch.tensor([
    len(train_labels) / (NUM_CLASSES * class_counts[i]) for i in range(NUM_CLASSES)
], dtype=torch.float).to(DEVICE)
print(f"Class weights: {class_weights.cpu().numpy()}")

criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)

# Training loop
history = {k: [] for k in ['train_loss', 'train_acc', 'train_f1',
                            'val_loss', 'val_acc', 'val_f1', 'lr']}
best_val_f1 = 0.0
patience_counter = 0
best_model_path = OUTPUT_DIR / "best_model.pth"

print(f"\n{'='*60}")
print(f"Training for up to {NUM_EPOCHS} epochs (patience={PATIENCE})")
print(f"{'='*60}\n")

start_time = time.time()

for epoch in range(1, NUM_EPOCHS + 1):
    epoch_start = time.time()
    current_lr = optimizer.param_groups[0]['lr']

    train_loss, train_acc, train_f1 = train_one_epoch(model, train_loader, criterion, optimizer)
    val_loss, val_acc, val_f1, _, _, _ = evaluate(model, val_loader, criterion)
    scheduler.step()

    for key, val in [('train_loss', train_loss), ('train_acc', train_acc),
                     ('train_f1', train_f1), ('val_loss', val_loss),
                     ('val_acc', val_acc), ('val_f1', val_f1), ('lr', current_lr)]:
        history[key].append(val)

    elapsed = time.time() - epoch_start
    improved = ""
    if val_f1 > best_val_f1:
        best_val_f1 = val_f1
        torch.save(model.state_dict(), best_model_path)
        patience_counter = 0
        improved = " *BEST*"
    else:
        patience_counter += 1

    print(f"Epoch {epoch:02d}/{NUM_EPOCHS} ({elapsed:.0f}s) | "
          f"Train Loss={train_loss:.4f} Acc={train_acc:.4f} F1={train_f1:.4f} | "
          f"Val Loss={val_loss:.4f} Acc={val_acc:.4f} F1={val_f1:.4f} | "
          f"LR={current_lr:.6f}{improved}")

    if patience_counter >= PATIENCE:
        print(f"\nEarly stopping at epoch {epoch} (no improvement for {PATIENCE} epochs)")
        break

total_time = time.time() - start_time
print(f"\nTraining complete in {total_time:.1f}s ({total_time/60:.1f} min)")
plot_training_curves(history)

# ============================================================
# Cell 12: Evaluate ResNet-50 on Test Set
# ============================================================
print(f"\n{'='*60}")
print("Evaluating best model on TEST set...")
model.load_state_dict(torch.load(best_model_path, weights_only=True))

test_loss, test_acc, test_f1, test_preds, test_true, test_probs = evaluate(
    model, test_loader, criterion
)

print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_acc:.4f}")
print(f"Test Macro F1: {test_f1:.4f}")
print(f"Cohen's Kappa: {cohen_kappa_score(test_true, test_preds):.4f}")

report = classification_report(test_true, test_preds, target_names=CLASS_NAMES, digits=4)
print(f"\nClassification Report:\n{report}")
report_dict = classification_report(test_true, test_preds, target_names=CLASS_NAMES,
                                    digits=4, output_dict=True)

cm = plot_confusion_matrix(test_true, test_preds, "Test Set Confusion Matrix", "confusion_matrix_test.png")
auc_scores, auc_micro = plot_roc_curves(test_true, test_probs, "Test Set ROC Curves", "roc_curves_test.png")
ap_scores = plot_precision_recall_curves(test_true, test_probs, "pr_curves_test.png")

sample_indices = random.sample(range(len(test_dataset)), min(18, len(test_dataset)))
plot_sample_predictions(model, test_dataset, sample_indices)

# ============================================================
# Cell 13: Evaluate ResNet-50 on Validation Set
# ============================================================
print(f"\n{'='*60}")
print("Evaluating best model on VALIDATION set...")
val_loss2, val_acc2, val_f12, val_preds, val_true, val_probs = evaluate(model, val_loader, criterion)
val_report = classification_report(val_true, val_preds, target_names=CLASS_NAMES, digits=4)
val_report_dict = classification_report(val_true, val_preds, target_names=CLASS_NAMES,
                                        digits=4, output_dict=True)
print(f"Val Accuracy: {val_acc2:.4f} | Val F1: {val_f12:.4f}")
print(val_report)
plot_confusion_matrix(val_true, val_preds, "Validation Set Confusion Matrix", "confusion_matrix_val.png")
plot_roc_curves(val_true, val_probs, "Validation Set ROC Curves", "roc_curves_val.png")

# Save ResNet-50 results
resnet_results = {
    "config": {
        "model": "ResNet-50", "pretrained": "ImageNet-V2",
        "img_size": IMG_SIZE, "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE, "weight_decay": WEIGHT_DECAY,
        "optimizer": "AdamW", "scheduler": "CosineAnnealingLR",
        "epochs_trained": len(history['train_loss']),
        "max_epochs": NUM_EPOCHS, "patience": PATIENCE,
        "seed": SEED, "device": str(DEVICE),
        "total_params": total_params, "trainable_params": trainable_params,
    },
    "dataset": {
        "total_images": len(all_paths),
        "train_size": len(train_paths), "val_size": len(val_paths), "test_size": len(test_paths),
        "class_distribution": {CLASS_NAMES[k]: v for k, v in Counter(all_labels).items()},
        "train_distribution": {CLASS_NAMES[k]: v for k, v in Counter(train_labels).items()},
    },
    "test_results": {
        "accuracy": float(test_acc), "macro_f1": float(test_f1),
        "cohen_kappa": float(cohen_kappa_score(test_true, test_preds)),
        "loss": float(test_loss),
        "per_class": report_dict,
        "auc_per_class": auc_scores, "auc_micro": float(auc_micro),
        "ap_per_class": ap_scores,
        "confusion_matrix": cm.tolist(),
    },
    "val_results": {
        "accuracy": float(val_acc2), "macro_f1": float(val_f12),
        "per_class": val_report_dict,
    },
    "training_history": history,
    "training_time_seconds": total_time,
}

with open(OUTPUT_DIR / "results.json", 'w') as f:
    json.dump(resnet_results, f, indent=2)

print(f"\nResNet-50 results saved to {OUTPUT_DIR}")

# ============================================================
# Cell 14: Traditional ML Classifiers — Feature Extraction
# ============================================================
print(f"\n{'='*60}")
print("PART 2: Traditional ML Classifiers")
print(f"{'='*60}")

print("\nBuilding ResNet-50 feature extractor (ImageNet weights)...")
feature_extractor = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
feature_extractor.fc = nn.Identity()
feature_extractor = feature_extractor.to(DEVICE)
feature_extractor.eval()
print("Feature dimension: 2048")

clf_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

clf_train_dataset = StrokeCTDataset(train_paths, train_labels, clf_transform)
clf_test_dataset = StrokeCTDataset(test_paths, test_labels, clf_transform)
clf_train_loader = DataLoader(clf_train_dataset, batch_size=64, shuffle=False, num_workers=2)
clf_test_loader = DataLoader(clf_test_dataset, batch_size=64, shuffle=False, num_workers=2)


@torch.no_grad()
def extract_features(feat_model, loader):
    features, labels_list = [], []
    for images, lbls in tqdm(loader, desc="  Extracting features"):
        images = images.to(DEVICE)
        feats = feat_model(images)
        features.append(feats.cpu().numpy())
        labels_list.append(lbls.numpy())
    return np.concatenate(features), np.concatenate(labels_list)


print("\nExtracting training features...")
X_train, y_train = extract_features(feature_extractor, clf_train_loader)
print(f"  Shape: {X_train.shape}")

print("Extracting test features...")
X_test, y_test = extract_features(feature_extractor, clf_test_loader)
print(f"  Shape: {X_test.shape}")

del feature_extractor
if torch.cuda.is_available():
    torch.cuda.empty_cache()

print("Scaling features...")
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ============================================================
# Cell 15: Train & Evaluate Traditional Classifiers
# ============================================================
classifiers = {
    'Logistic Regression': LogisticRegression(
        C=1.0, max_iter=2000, solver='lbfgs',
        class_weight='balanced', random_state=SEED, n_jobs=-1,
    ),
    'Random Forest': RandomForestClassifier(
        n_estimators=500, max_depth=None, min_samples_split=5,
        class_weight='balanced', random_state=SEED, n_jobs=-1,
    ),
    'SVM': SVC(
        C=10.0, kernel='rbf', class_weight='balanced',
        probability=True, random_state=SEED,
    ),
}

all_clf_results = {}
for name, clf in classifiers.items():
    print(f"\n{'='*60}")
    print(f"Training: {name}")
    start = time.time()
    clf.fit(X_train, y_train)
    train_time = time.time() - start
    print(f"Training time: {train_time:.1f}s")

    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average='macro')
    kappa = cohen_kappa_score(y_test, y_pred)

    clf_report = classification_report(y_test, y_pred, target_names=CLASS_NAMES, digits=4)
    clf_report_dict = classification_report(y_test, y_pred, target_names=CLASS_NAMES,
                                            digits=4, output_dict=True)

    print(f"Accuracy: {acc:.4f} | Macro F1: {macro_f1:.4f} | Kappa: {kappa:.4f}")
    print(clf_report)

    safe_name = name.lower().replace(' ', '_').replace("'", "")
    clf_cm = plot_confusion_matrix(y_test, y_pred, f"{name} — Confusion Matrix",
                                   f"cm_{safe_name}.png", save_dir=CLF_OUTPUT_DIR)

    y_probs_clf = clf.predict_proba(X_test)
    clf_auc_scores, clf_auc_micro = plot_roc_curves(y_test, y_probs_clf, f"{name} — ROC Curves",
                                                     f"roc_{safe_name}.png", save_dir=CLF_OUTPUT_DIR)
    clf_ap_scores = plot_precision_recall_curves(y_test, y_probs_clf, f"pr_{safe_name}.png",
                                                 save_dir=CLF_OUTPUT_DIR)

    per_class_f1 = [clf_report_dict[c]['f1-score'] for c in CLASS_NAMES]

    all_clf_results[name] = {
        'accuracy': float(acc), 'macro_f1': float(macro_f1), 'cohen_kappa': float(kappa),
        'per_class': clf_report_dict, 'per_class_f1': per_class_f1,
        'auc_per_class': {k: float(v) for k, v in clf_auc_scores.items()},
        'auc_micro': float(clf_auc_micro),
        'ap_per_class': {k: float(v) for k, v in clf_ap_scores.items()},
        'confusion_matrix': clf_cm.tolist(),
        'training_time_seconds': train_time,
    }

# ============================================================
# Cell 16: Classifier Comparison & Summary
# ============================================================
plot_model_comparison(all_clf_results)

# Summary table
print(f"\n{'='*60}")
print(f"{'Model':<25} {'Accuracy':>10} {'Macro F1':>10} {'Kappa':>10} {'Train Time':>12}")
print(f"{'-'*67}")
for name, res in all_clf_results.items():
    print(f"{name:<25} {res['accuracy']:>10.4f} {res['macro_f1']:>10.4f} "
          f"{res['cohen_kappa']:>10.4f} {res['training_time_seconds']:>10.1f}s")
print(f"{'='*60}")

# Save classifier results
save_clf_results = {}
for name, res in all_clf_results.items():
    save_clf_results[name] = {k: v for k, v in res.items() if k != 'per_class_f1'}

save_clf_results['_meta'] = {
    'feature_extractor': 'ResNet-50 (ImageNet-V2)',
    'feature_dim': 2048,
    'scaler': 'StandardScaler',
    'train_size': len(train_paths),
    'val_size': len(val_paths),
    'test_size': len(test_paths),
    'seed': SEED,
}

with open(CLF_OUTPUT_DIR / "classifier_results.json", 'w') as f:
    json.dump(save_clf_results, f, indent=2)

print(f"\nAll classifier results saved to {CLF_OUTPUT_DIR}")
print("\nDone! All models trained and evaluated.")

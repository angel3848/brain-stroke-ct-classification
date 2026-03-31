"""
Traditional ML Classifiers for Brain Stroke CT Classification
Logistic Regression, Random Forest, SVM
Uses ResNet-50 (ImageNet) as feature extractor.
"""

import json
import time
import random
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from PIL import Image
from collections import Counter

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
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
# Configuration
# ============================================================
SEED = 42
DATA_ROOT = Path("/Users/mpcr/Desktop/Medical Image Projects/Stroke/Brain_Stroke_CT_Dataset")
OUTPUT_DIR = Path("/Users/mpcr/Desktop/Medical Image Projects/Stroke/results/classifiers")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CLASS_NAMES = ["Bleeding", "Ischemia", "Normal"]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASS_NAMES)}
NUM_CLASSES = 3

BATCH_SIZE = 64
IMG_SIZE = 224

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# ============================================================
# Dataset
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
    paths, labels = [], []
    for cls_name in CLASS_NAMES:
        cls_dir = DATA_ROOT / cls_name / "PNG"
        for f in sorted(cls_dir.glob("*.png")):
            paths.append(str(f))
            labels.append(CLASS_TO_IDX[cls_name])
    return paths, labels


# ============================================================
# Feature Extraction
# ============================================================
def build_feature_extractor():
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    model.fc = nn.Identity()
    model = model.to(DEVICE)
    model.eval()
    return model


@torch.no_grad()
def extract_features(model, loader):
    features, labels = [], []
    for images, lbls in tqdm(loader, desc="  Extracting features"):
        images = images.to(DEVICE)
        feats = model(images)
        features.append(feats.cpu().numpy())
        labels.append(lbls.numpy())
    return np.concatenate(features), np.concatenate(labels)


# ============================================================
# Plotting
# ============================================================
def plot_confusion_matrix(y_true, y_pred, title, filename):
    cm = confusion_matrix(y_true, y_pred)
    cm_pct = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=CLASS_NAMES,
                yticklabels=CLASS_NAMES, ax=axes[0])
    axes[0].set_title(f'{title} (Counts)', fontsize=13, fontweight='bold')
    axes[0].set_ylabel('True Label')
    axes[0].set_xlabel('Predicted Label')

    sns.heatmap(cm_pct, annot=True, fmt='.1f', cmap='Blues', xticklabels=CLASS_NAMES,
                yticklabels=CLASS_NAMES, ax=axes[1])
    axes[1].set_title(f'{title} (Percentages %)', fontsize=13, fontweight='bold')
    axes[1].set_ylabel('True Label')
    axes[1].set_xlabel('Predicted Label')

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / filename, dpi=300, bbox_inches='tight')
    plt.close()
    return cm


def plot_roc_curves(y_true, y_probs, title, filename):
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
    plt.savefig(OUTPUT_DIR / filename, dpi=300, bbox_inches='tight')
    plt.close()
    return auc_scores, auc_micro


def plot_pr_curves(y_true, y_probs, filename):
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
    plt.savefig(OUTPUT_DIR / filename, dpi=300, bbox_inches='tight')
    plt.close()
    return ap_scores


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
    plt.savefig(OUTPUT_DIR / "model_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()


def plot_per_class_f1_comparison(all_results):
    models_list = list(all_results.keys())
    colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12']

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(NUM_CLASSES)
    width = 0.2

    for i, model_name in enumerate(models_list):
        f1s = all_results[model_name]['per_class_f1']
        bars = ax.bar(x + i * width, f1s, width, label=model_name, color=colors[i])
        for bar, val in zip(bars, f1s):
            ax.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + 0.005,
                    f'{val:.3f}', ha='center', va='bottom', fontsize=8, fontweight='bold')

    ax.set_ylabel('F1-Score', fontsize=12)
    ax.set_title('Per-Class F1-Score Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x + width * (len(models_list) - 1) / 2)
    ax.set_xticklabels(CLASS_NAMES, fontsize=11)
    ax.set_ylim(0, 1.1)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "per_class_f1_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()


# ============================================================
# Evaluate a single classifier
# ============================================================
def evaluate_classifier(name, clf, X_test, y_test, has_proba=True):
    print(f"\n{'='*60}")
    print(f"Evaluating: {name}")
    print(f"{'='*60}")

    start = time.time()
    y_pred = clf.predict(X_test)
    pred_time = time.time() - start

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='macro')
    kappa = cohen_kappa_score(y_test, y_pred)

    report = classification_report(y_test, y_pred, target_names=CLASS_NAMES, digits=4)
    report_dict = classification_report(y_test, y_pred, target_names=CLASS_NAMES,
                                        digits=4, output_dict=True)

    print(f"Accuracy:      {acc:.4f}")
    print(f"Macro F1:      {f1:.4f}")
    print(f"Cohen's Kappa: {kappa:.4f}")
    print(f"\n{report}")

    # Confusion matrix
    safe_name = name.lower().replace(' ', '_').replace("'", "")
    cm = plot_confusion_matrix(y_test, y_pred,
                               f"{name} — Confusion Matrix",
                               f"cm_{safe_name}.png")

    # Probabilities for ROC/PR
    auc_scores, auc_micro, ap_scores = {}, 0.0, {}
    if has_proba:
        y_probs = clf.predict_proba(X_test)
        auc_scores, auc_micro = plot_roc_curves(y_test, y_probs,
                                                 f"{name} — ROC Curves",
                                                 f"roc_{safe_name}.png")
        ap_scores = plot_pr_curves(y_test, y_probs, f"pr_{safe_name}.png")

        print(f"AUC per class: { {k: round(v, 4) for k, v in auc_scores.items()} }")
        print(f"Micro AUC:     {auc_micro:.4f}")

    per_class_f1 = [report_dict[c]['f1-score'] for c in CLASS_NAMES]

    return {
        'accuracy': float(acc),
        'macro_f1': float(f1),
        'cohen_kappa': float(kappa),
        'per_class': report_dict,
        'per_class_f1': per_class_f1,
        'auc_per_class': {k: float(v) for k, v in auc_scores.items()},
        'auc_micro': float(auc_micro),
        'ap_per_class': {k: float(v) for k, v in ap_scores.items()},
        'confusion_matrix': cm.tolist(),
        'prediction_time_seconds': pred_time,
    }


# ============================================================
# Main
# ============================================================
def main():
    print(f"Device: {DEVICE}")
    print(f"{'='*60}")

    # ------ Load data ------
    print("Loading dataset...")
    all_paths, all_labels = load_dataset()
    print(f"Total images: {len(all_paths)}")

    # Same split as train_resnet.py
    train_paths, temp_paths, train_labels, temp_labels = train_test_split(
        all_paths, all_labels, test_size=0.30, random_state=SEED, stratify=all_labels
    )
    val_paths, test_paths, val_labels, test_labels = train_test_split(
        temp_paths, temp_labels, test_size=0.50, random_state=SEED, stratify=temp_labels
    )

    print(f"Train: {len(train_paths)} | Val: {len(val_paths)} | Test: {len(test_paths)}")

    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    train_dataset = StrokeCTDataset(train_paths, train_labels, transform)
    val_dataset = StrokeCTDataset(val_paths, val_labels, transform)
    test_dataset = StrokeCTDataset(test_paths, test_labels, transform)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # ------ Extract features ------
    print("\nBuilding ResNet-50 feature extractor (ImageNet weights)...")
    feature_extractor = build_feature_extractor()
    print("Feature dimension: 2048")

    print("\nExtracting training features...")
    X_train, y_train = extract_features(feature_extractor, train_loader)
    print(f"  Shape: {X_train.shape}")

    print("Extracting validation features...")
    X_val, y_val = extract_features(feature_extractor, val_loader)

    print("Extracting test features...")
    X_test, y_test = extract_features(feature_extractor, test_loader)
    print(f"  Shape: {X_test.shape}")

    # Free GPU memory
    del feature_extractor
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ------ Scale features ------
    print("\nScaling features...")
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    # ------ Define classifiers ------
    classifiers = {
        'Logistic Regression': LogisticRegression(
            C=1.0,
            max_iter=2000,
            solver='lbfgs',
            multi_class='multinomial',
            class_weight='balanced',
            random_state=SEED,
            n_jobs=-1,
        ),
        'Random Forest': RandomForestClassifier(
            n_estimators=500,
            max_depth=None,
            min_samples_split=5,
            class_weight='balanced',
            random_state=SEED,
            n_jobs=-1,
        ),
        'SVM': SVC(
            C=10.0,
            kernel='rbf',
            class_weight='balanced',
            probability=True,
            random_state=SEED,
        ),
    }

    # ------ Train & evaluate ------
    all_results = {}
    for name, clf in classifiers.items():
        print(f"\n{'='*60}")
        print(f"Training: {name}")
        start = time.time()
        clf.fit(X_train, y_train)
        train_time = time.time() - start
        print(f"Training time: {train_time:.1f}s")

        results = evaluate_classifier(name, clf, X_test, y_test)
        results['training_time_seconds'] = train_time
        all_results[name] = results

    # ------ Comparison plots ------
    print(f"\n{'='*60}")
    print("Generating comparison plots...")
    plot_model_comparison(all_results)
    plot_per_class_f1_comparison(all_results)

    # ------ Save results ------
    # Strip per_class_f1 list (not JSON-friendly label)
    save_results = {}
    for name, res in all_results.items():
        save_results[name] = {k: v for k, v in res.items() if k != 'per_class_f1'}

    save_results['_meta'] = {
        'feature_extractor': 'ResNet-50 (ImageNet-V2)',
        'feature_dim': 2048,
        'scaler': 'StandardScaler',
        'train_size': len(train_paths),
        'val_size': len(val_paths),
        'test_size': len(test_paths),
        'seed': SEED,
    }

    with open(OUTPUT_DIR / "classifier_results.json", 'w') as f:
        json.dump(save_results, f, indent=2)

    # ------ Summary table ------
    print(f"\n{'='*60}")
    print(f"{'Model':<25} {'Accuracy':>10} {'Macro F1':>10} {'Kappa':>10} {'Train Time':>12}")
    print(f"{'-'*67}")
    for name, res in all_results.items():
        print(f"{name:<25} {res['accuracy']:>10.4f} {res['macro_f1']:>10.4f} "
              f"{res['cohen_kappa']:>10.4f} {res['training_time_seconds']:>10.1f}s")
    print(f"{'='*60}")

    print(f"\nAll results saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()

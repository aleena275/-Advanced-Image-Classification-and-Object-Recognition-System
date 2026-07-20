"""
train.py
--------
Training pipeline covering the project's key phases:
  - Dataset loading & augmentation (Phase 1 & 2)
  - From-scratch CNN training (Phase 3)
  - Transfer learning (Phase 4)
  - LR scheduling, early stopping, checkpoints, gradient clipping (Phase 6)
  - Evaluation metrics (Phase 8)

Usage:
    python train.py --model scratch --epochs 20
    python train.py --model transfer --backbone resnet18 --epochs 10
"""

import argparse
import copy
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.metrics import classification_report, confusion_matrix

from model import build_model

CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]


def get_dataloaders(batch_size: int = 64, data_dir: str = "./data"):
    """Phase 1 & 2: dataset loading, preprocessing, augmentation."""
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                              (0.2470, 0.2435, 0.2616)),
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                              (0.2470, 0.2435, 0.2616)),
    ])

    train_set = datasets.CIFAR10(root=data_dir, train=True,
                                  download=True, transform=train_transform)
    test_set = datasets.CIFAR10(root=data_dir, train=False,
                                 download=True, transform=test_transform)

    train_loader = DataLoader(train_set, batch_size=batch_size,
                               shuffle=True, num_workers=2)
    test_loader = DataLoader(test_set, batch_size=batch_size,
                              shuffle=False, num_workers=2)
    return train_loader, test_loader


def train_one_epoch(model, loader, criterion, optimizer, device, clip_norm=5.0):
    model.train()
    running_loss, correct, total = 0.0, 0, 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()

        # Phase 6: gradient clipping
        nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)

    return running_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)

        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)

        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    avg_loss = running_loss / total
    accuracy = correct / total
    return avg_loss, accuracy, all_preds, all_labels


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["scratch", "transfer"], default="scratch")
    parser.add_argument("--backbone", choices=["resnet18", "vgg16", "efficientnet_b0"],
                         default="resnet18")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=5, help="early stopping patience")
    parser.add_argument("--out", default="best_model.pt")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_loader, test_loader = get_dataloaders(args.batch_size)

    if args.model == "scratch":
        model = build_model("scratch", num_classes=10).to(device)
    else:
        model = build_model("transfer", backbone=args.backbone,
                             num_classes=10, mode="fine_tune").to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                            lr=args.lr, weight_decay=1e-4)  # L2 regularization
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=2)

    best_acc = 0.0
    best_state = None
    epochs_no_improve = 0

    for epoch in range(args.epochs):
        start = time.time()
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, _, _ = evaluate(model, test_loader, criterion, device)
        scheduler.step(val_loss)
        elapsed = time.time() - start

        print(f"Epoch {epoch+1}/{args.epochs} | "
              f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
              f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} | {elapsed:.1f}s")

        # Phase 6: checkpointing + early stopping
        if val_acc > best_acc:
            best_acc = val_acc
            best_state = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
            torch.save(best_state, args.out)
            print(f"  ✔ New best model saved (val_acc={val_acc:.4f})")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= args.patience:
                print("Early stopping triggered.")
                break

    # Phase 8: final evaluation report
    model.load_state_dict(best_state)
    _, _, preds, labels = evaluate(model, test_loader, criterion, device)
    print("\nClassification report:")
    print(classification_report(labels, preds, target_names=CIFAR10_CLASSES))
    print("Confusion matrix:")
    print(confusion_matrix(labels, preds))


if __name__ == "__main__":
    main()

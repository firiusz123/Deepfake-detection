import os

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve
from tqdm import tqdm

from .model import SimpleCNN
from .dataset import get_dataloaders
from .utils import EarlyStopping


def _format_table(header_title, row_titles, matrix, cell_formatter):
    col_width = 20
    header_parts = [f"{header_title:<10}"] + [f"{title:^{col_width}}" for title in row_titles]
    header = "|".join(header_parts) + "|"
    separator = "-" * len(header)
    lines = [header, separator]
    for title, row in zip(row_titles, matrix):
        row_cells = "|".join(f"{cell_formatter(val):^{col_width}}" for val in row)
        lines.append(f"{title:<10}|{row_cells}|")
    return "\n".join(lines)


def _print_confusion_matrix(y_true, y_pred, classes):
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(classes))))
    with np.errstate(all="ignore"):
        row_sums = cm.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        normalized = cm.astype(float) / row_sums

    print("\nConfusion matrix counts (rows=true, columns=predicted):")
    counts_table = _format_table("True\\Pred", classes, cm, lambda val: f"{val}")
    print(counts_table)

    print("\nConfusion matrix percentages:")
    perc_table = _format_table(
        "True\\Pred",
        classes,
        normalized * 100.0,
        lambda val: f"{val:5.1f}%"
    )
    print(perc_table)

def run_pipeline(args, mode='train'):
    train_loader, val_loader, test_loader = get_dataloaders(
        args.archive_path, args.img_size, args.batch_size
    )

    model = SimpleCNN(img_size=args.img_size)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    model_path = 'model_best.pth'

    if mode == 'train':
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=2)
        early_stopping = EarlyStopping(patience=5)

        for epoch in range(args.epochs):
            model.train()
            train_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Train]")
            for images, labels in train_bar:
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                train_bar.set_postfix(loss=loss.item())

            model.eval()
            val_loss = 0
            with torch.no_grad():
                for images, labels in val_loader:
                    images, labels = images.to(device), labels.to(device)
                    outputs = model(images)
                    val_loss += criterion(outputs, labels).item()
            
            val_loss /= len(val_loader)
            print(f'Epoch {epoch+1}, Val Loss: {val_loss:.4f}')
            
            scheduler.step(val_loss)
            early_stopping(val_loss)
            if early_stopping.early_stop:
                print("Early stopping triggered")
                break
        
        torch.save(model.state_dict(), model_path)
        print(f"Model saved to {model_path}")

    elif mode == 'test':
        if os.path.exists(model_path):
            model.load_state_dict(torch.load(model_path))
            model.eval()
            correct = 0
            total = 0
            all_labels = []
            all_scores = []
            all_predictions = []
            with torch.no_grad():
                for images, labels in tqdm(test_loader, desc="Testing"):
                    images, labels = images.to(device), labels.to(device)
                    outputs = model(images)
                    _, predicted = torch.max(outputs.data, 1)

                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()

                    all_labels.append(labels.cpu())
                    all_scores.append(outputs[:, 1].cpu())
                    all_predictions.append(predicted.cpu())

            accuracy = 100 * correct / total if total else 0.0
            print(f'Accuracy: {accuracy:.2f}%')

            if all_labels and all_scores:
                labels_tensor = torch.cat(all_labels)
                scores_tensor = torch.cat(all_scores)
                preds_tensor = torch.cat(all_predictions)
                if labels_tensor.numel() and torch.unique(labels_tensor).numel() == 2:
                    y_true = labels_tensor.numpy()
                    y_scores = scores_tensor.numpy()
                    y_preds = preds_tensor.numpy()
                    roc_auc = roc_auc_score(y_true, y_scores)
                    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
                    print(f"Test ROC AUC: {roc_auc * 100:.2f}%")
                    print(
                        "ROC curve points: "
                        f"FPR range [{fpr.min():.3f}, {fpr.max():.3f}], "
                        f"TPR range [{tpr.min():.3f}, {tpr.max():.3f}], "
                        f"{len(fpr)} thresholds"
                    )
                    _print_confusion_matrix(y_true, y_preds, classes=["real", "fake"])
                else:
                    print("ROC curve skipped because test split lacks both classes.")
        else:
            print("Model file not found.")

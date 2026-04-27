import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, confusion_matrix
from torch.utils.data import DataLoader, Subset

import config.config as cfg
from models.wavelet_model import WaveletHybridNet
from data.dataset import WaveletDeepfakeDataset
from data.transforms import get_transforms

import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc



# =====================================================
# SETTINGS
# =====================================================
MAX_SAMPLES = 5000   # None = full dataset
RANDOM_SEED = 42
CHECKPOINT_PATH = "best_fold_0.pt"


# =====================================================
# CONFUSION MATRIX (FORMATTED)
# =====================================================
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

    print("\nConfusion matrix counts:")
    print(_format_table("True\\Pred", classes, cm, lambda x: f"{x}"))

    print("\nConfusion matrix %:")
    print(_format_table("True\\Pred", classes, normalized * 100, lambda x: f"{x:5.1f}%"))


# =====================================================
# TEST LOOP
# =====================================================
def test(model, loader, device):
    model.eval()

    all_preds, all_labels, all_probs = [], [], []

    loop = tqdm(loader, desc="Testing")

    with torch.no_grad():
        for x_rgb, wav, y in loop:

            x_rgb = x_rgb.to(device)
            y = y.to(device)
            wav = {k: v.to(device) for k, v in wav.items()}

            out = model(x_rgb, wav)

            probs = torch.softmax(out, dim=1)[:, 1]
            preds = torch.argmax(out, dim=1)

            all_probs.extend(probs.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())

    return all_preds, all_labels, all_probs

# =====================================================
# ROC CURVE PLOT
# =====================================================
def plot_roc_curve(y_true, y_probs, save_path="roc_curve.png"):
    fpr, tpr, _ = roc_curve(y_true, y_probs)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.4f}")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend(loc="lower right")

    plt.savefig(save_path)
    print(f"[INFO] ROC curve saved to {save_path}")

    plt.close()


# =====================================================
# CONFUSION MATRIX PLOT
# =====================================================
def plot_confusion_matrix(y_true, y_pred, classes, normalize=False, save_path="cm.png"):
    cm = confusion_matrix(y_true, y_pred)

    if normalize:
        cm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    plt.figure()
    plt.imshow(cm, interpolation="nearest")
    plt.title("Confusion Matrix" + (" (Normalized)" if normalize else ""))
    plt.colorbar()

    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)

    # annotate cells
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            val = cm[i, j]
            if normalize:
                text = f"{val:.2f}"
            else:
                text = f"{int(val)}"

            plt.text(j, i, text, ha="center", va="center")

    plt.ylabel("True label")
    plt.xlabel("Predicted label")

    plt.tight_layout()
    plt.savefig(save_path)
    print(f"[INFO] Confusion matrix saved to {save_path}")

    plt.close()



# =====================================================
# MAIN
# =====================================================
if __name__ == "__main__":

    device = cfg.DEVICE

    # -------------------------------------------------
    # TRANSFORMS (NO AUGMENTATION)
    # -------------------------------------------------
    transform = get_transforms(
        img_size=cfg.IMG_SIZE,
        normalize=True,
        augment=False
    )

    # -------------------------------------------------
    # DATASET
    # -------------------------------------------------
    dataset = WaveletDeepfakeDataset(
        root_dir=cfg.DATA_ROOT,
        split=cfg.TEST_SPLIT,
        transform=transform
    )

    print(f"[INFO] Full test dataset size: {len(dataset)}")

    # -------------------------------------------------
    # SUBSET (OPTIONAL)
    # -------------------------------------------------
    if MAX_SAMPLES is not None:
        np.random.seed(RANDOM_SEED)

        indices = np.random.choice(
            len(dataset),
            size=min(MAX_SAMPLES, len(dataset)),
            replace=False
        )

        dataset = Subset(dataset, indices)
        print(f"[INFO] Using subset: {len(dataset)} samples")

    # -------------------------------------------------
    # DATALOADER
    # -------------------------------------------------
    loader = DataLoader(
        dataset,
        batch_size=cfg.BATCH_SIZE,
        shuffle=False,
        num_workers=cfg.NUM_WORKERS,
        pin_memory=cfg.PIN_MEMORY
    )

    # -------------------------------------------------
    # MODEL
    # -------------------------------------------------
    model = WaveletHybridNet().to(device)

    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)

    # -------------------------------------------------
    # ROBUST CHECKPOINT LOADER
    # -------------------------------------------------
    if isinstance(checkpoint, dict):
        if "model_state" in checkpoint:
            state_dict = checkpoint["model_state"]
            print("[INFO] Loaded key: model_state")
        elif "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
            print("[INFO] Loaded key: state_dict")
        else:
            state_dict = checkpoint
            print("[INFO] Loaded raw state_dict (dict without key)")
    else:
        state_dict = checkpoint
        print("[INFO] Loaded raw state_dict (not dict)")

    model.load_state_dict(state_dict)

    # -------------------------------------------------
    # OPTIONAL INFO PRINT
    # -------------------------------------------------
    if isinstance(checkpoint, dict):
        print(f"Checkpoint AUC: {checkpoint.get('auc', 'N/A')}")
        print(f"Checkpoint Epoch: {checkpoint.get('epoch', 'N/A')}")

    print(f"\nLoaded model from: {CHECKPOINT_PATH}")

    # -------------------------------------------------
    # TEST
    # -------------------------------------------------
    preds, labels, probs = test(model, loader, device)

    # -------------------------------------------------
    # METRICS
    # -------------------------------------------------
    acc = (np.array(preds) == np.array(labels)).mean()
    auc_score = roc_auc_score(labels, probs)

    print("\n================ TEST RESULTS ================\n")
    print(f"Samples:  {len(labels)}")
    print(f"Accuracy: {acc:.4f}")
    print(f"AUC:      {auc_score:.4f}")

    # -------------------------------------------------
    # CONFUSION MATRIX
    # -------------------------------------------------
    _print_confusion_matrix(labels, preds, ["real", "fake"])

    # -------------------------------------------------
    # PLOTS
    # -------------------------------------------------
    plot_roc_curve(labels, probs, save_path="roc_curve.png")

    plot_confusion_matrix(labels, preds, ["real", "fake"],
                        normalize=False,
                        save_path="cm_counts.png")

    plot_confusion_matrix(labels, preds, ["real", "fake"],
                        normalize=True,
                        save_path="cm_normalized.png")

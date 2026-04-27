import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Subset

import config.config as cfg

from models.wavelet_model import WaveletHybridNet
from data.dataset import WaveletDeepfakeDataset
from data.transforms import get_transforms


# =====================================================
# EARLY STOPPING
# =====================================================
class EarlyStopping:
    def __init__(self, patience=3):
        self.patience = patience
        self.best = None
        self.counter = 0
        self.stop = False

    def step(self, score):
        if self.best is None or score > self.best:
            self.best = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.stop = True


# =====================================================
def accuracy_fn(y_true, y_pred):
    return (np.array(y_true) == np.array(y_pred)).mean()


# =====================================================
def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()

    total_loss = 0
    preds, labels, probs = [], [], []

    loop = tqdm(loader, desc="Train")

    for x_rgb, wav, y in loop:
        x_rgb, y = x_rgb.to(device), y.to(device)
        wav = {k: v.to(device) for k, v in wav.items()}

        optimizer.zero_grad()
        out = model(x_rgb, wav)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        p = torch.softmax(out, dim=1)[:, 1].detach().cpu().numpy()
        preds.extend(torch.argmax(out, 1).cpu().numpy())
        probs.extend(p)
        labels.extend(y.cpu().numpy())

    return total_loss / len(loader), preds, labels, probs


# =====================================================
def validate(model, loader, criterion, device):
    model.eval()

    total_loss = 0
    preds, labels, probs = [], [], []

    loop = tqdm(loader, desc="Val")

    with torch.no_grad():
        for x_rgb, wav, y in loop:
            x_rgb, y = x_rgb.to(device), y.to(device)
            wav = {k: v.to(device) for k, v in wav.items()}

            out = model(x_rgb, wav)
            loss = criterion(out, y)

            total_loss += loss.item()

            p = torch.softmax(out, dim=1)[:, 1].cpu().numpy()
            preds.extend(torch.argmax(out, 1).cpu().numpy())
            probs.extend(p)
            labels.extend(y.cpu().numpy())

    return total_loss / len(loader), preds, labels, probs


# =====================================================
def train_kfold(dataset, device):

    kf = KFold(n_splits=cfg.N_FOLDS, shuffle=True, random_state=42)

    for fold, (tr, va) in enumerate(kf.split(dataset)):
        print(f"\nFOLD {fold}")

        train_set = Subset(dataset, tr)
        val_set = Subset(dataset, va)

        train_loader = DataLoader(train_set, batch_size=cfg.BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_set, batch_size=cfg.BATCH_SIZE)

        model = WaveletHybridNet().to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=cfg.LEARNING_RATE)
        crit = nn.CrossEntropyLoss()

        early = EarlyStopping(patience=cfg.PATIENCE)

        best_auc = 0

        for epoch in range(cfg.EPOCHS):

            tr_loss, _, _, _ = train_one_epoch(model, train_loader, opt, crit, device)
            va_loss, v_p, v_l, v_pr = validate(model, val_loader, crit, device)

            auc = roc_auc_score(v_l, v_pr)

            print(f"Epoch {epoch} | AUC {auc:.4f}")

            if auc > best_auc:
                best_auc = auc
                torch.save(model.state_dict(), f"best_fold_{fold}.pt")

            early.step(auc)
            if early.stop:
                print("Early stop")
                break


# =====================================================
# MAIN ENTRY POINT (IMPORTANT)
# =====================================================
if __name__ == "__main__":

    device = cfg.DEVICE

    transform = get_transforms(
        img_size=cfg.IMG_SIZE,
        augment=cfg.USE_AUGMENTATION
    )

    dataset = WaveletDeepfakeDataset(
        root_dir=cfg.DATA_ROOT,
        split=cfg.TRAIN_SPLIT,
        transform=transform
    )

    train_kfold(dataset, device)

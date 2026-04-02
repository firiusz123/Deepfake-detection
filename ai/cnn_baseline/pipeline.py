import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from .model import SimpleCNN
from .dataset import get_dataloaders
from .utils import EarlyStopping
import os

def run_pipeline(args, mode='train'):
    train_loader, val_loader, test_loader = get_dataloaders(args.archive_path, args.img_size, args.batch_size)
    
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
            with torch.no_grad():
                for images, labels in tqdm(test_loader, desc="Testing"):
                    images, labels = images.to(device), labels.to(device)
                    outputs = model(images)
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
            print(f'Accuracy: {100 * correct / total:.2f}%')
        else:
            print("Model file not found.")

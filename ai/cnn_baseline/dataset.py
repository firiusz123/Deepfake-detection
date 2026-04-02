import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, ConcatDataset

def get_dataloaders(archive_path, img_size, batch_size):
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Expects archive/[train, val, test] structure directly
    def load_split(split):
        split_path = os.path.join(archive_path, split)
        if os.path.exists(split_path):
            return datasets.ImageFolder(split_path, transform=transform)
        return None

    train_dataset = load_split('train')
    val_dataset = load_split('val')
    test_dataset = load_split('test')

    return (
        DataLoader(train_dataset, batch_size=batch_size, shuffle=True) if train_dataset else None,
        DataLoader(val_dataset, batch_size=batch_size, shuffle=False) if val_dataset else None,
        DataLoader(test_dataset, batch_size=batch_size, shuffle=False) if test_dataset else None
    )

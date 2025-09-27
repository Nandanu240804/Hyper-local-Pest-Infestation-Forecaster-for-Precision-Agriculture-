# data_loader.py
import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import pandas as pd
from PIL import Image

class PestDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        csv_file = os.path.normpath(csv_file)
        if not os.path.exists(csv_file):
            raise FileNotFoundError(f"CSV file not found: {csv_file}")

        self.annotations = pd.read_csv(csv_file)
        if self.annotations.shape[1] < 2:
            raise ValueError(f"CSV file must have at least 2 columns (image_path,label). Got {self.annotations.shape[1]} columns.")

        self.img_dir = os.path.normpath(img_dir)
        self.transform = transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        img_name_or_path = str(self.annotations.iloc[idx, 0]).strip()
        if os.path.isabs(img_name_or_path):
            img_path = img_name_or_path
        else:
            img_path = os.path.join(self.img_dir, img_name_or_path)

        img_path = os.path.normpath(img_path)
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Image not found: {img_path} (row {idx})")

        image = Image.open(img_path).convert('RGB')

        # convert label to int if possible, then to torch.long tensor
        raw_label = self.annotations.iloc[idx, 1]
        try:
            label_int = int(raw_label)
        except Exception:
            # if it's not convertible, raise early to avoid silent issues
            raise ValueError(f"Label at row {idx} is not an integer: {raw_label}")

        label = torch.tensor(label_int, dtype=torch.long)

        if self.transform:
            image = self.transform(image)

        return image, label

def get_data_loaders(batch_size=8, num_workers=0, img_size=224):
    BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    df_dir = os.path.join(BASE_DIR, 'ip102_v1.1')

    train_csv = os.path.join(df_dir, 'train.csv')
    val_csv = os.path.join(df_dir, 'val.csv')

    images_dir = os.path.join(df_dir, 'images')

    # Use same resize for train and val (use smaller size to save memory)
    train_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_dataset = PestDataset(csv_file=train_csv, img_dir=images_dir, transform=train_transform)
    val_dataset = PestDataset(csv_file=val_csv, img_dir=images_dir, transform=val_transform)

    try:
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=False)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=False)
    except Exception as e:
        print(f"Warning: DataLoader creation with num_workers={num_workers} failed: {e}")
        print("Falling back to num_workers=0")
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    return train_loader, val_loader, train_dataset, val_dataset

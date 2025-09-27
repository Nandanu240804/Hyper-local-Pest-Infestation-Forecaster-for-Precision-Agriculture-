#!/usr/bin/env python3
"""
Main script to train the pest detection model
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR

from data_loader import get_data_loaders
from model import create_model
from train import train_model, plot_training_history

def main():
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Data loader parameters (tweak batch_size if you hit OOM)
    batch_size = 8
    num_workers = 0   # keep 0 on Windows; raise on Linux if you want
    img_size = 224    # smaller size reduces memory use

    # Get data loaders
    print("Loading data...")
    train_loader, val_loader, train_dataset, val_dataset = get_data_loaders(
        batch_size=batch_size, num_workers=num_workers, img_size=img_size
    )

    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    # faster and safer way to compute number of classes:
    try:
        num_classes = int(train_dataset.annotations.iloc[:, 1].nunique())
    except Exception:
        # fallback to scanning labels (will be slower)
        labels = []
        for _, lbl in train_dataset:
            # labels are torch.long tensors in the dataset
            labels.append(int(lbl.item() if hasattr(lbl, "item") else lbl))
        num_classes = len(set(labels))

    print(f"Number of classes: {num_classes}")

    # Create model (create_model should return an nn.Module)
    print("Creating model...")
    model = create_model(num_classes=num_classes, device=device)
    model.to(device)  # ensure model is on the right device

    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = StepLR(optimizer, step_size=10, gamma=0.1)

    # Train model
    print("Starting training...")
    trained_model, train_losses, val_losses, val_accuracies = train_model(
        model, train_loader, val_loader, criterion, optimizer, scheduler,
        num_epochs=30, device=device
    )

    # Plot training history
    plot_training_history(train_losses, val_losses, val_accuracies)

    print("Training completed!")

if __name__ == "__main__":
    main()


"""
This file contains functions for setting up and loading data for a modular PyTorch project.

Functions:
    create_dataloader(train_dir, test_dir, transform, batch_size, num_workers=NUM_WORKERS):
        Creates data loaders for training and testing datasets.

Globals:
    NUM_WORKERS: The number of CPU cores available on the system.
"""

import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

NUM_WORKERS = 2

def create_dataloader(train_dir: str,
                      test_dir: str,
                      transform: transforms.Compose,
                      batch_size: int,
                      num_workers: int = NUM_WORKERS):
    """
    Creates DataLoader objects for training and testing datasets.

    Args:
        train_dir (str): The directory containing training data.
        test_dir (str): The directory containing testing data.
        transform (transforms.Compose): Transformations to apply to the data.
        batch_size (int): The number of samples per batch.
        num_workers (int, optional): Number of subprocesses to use for data loading.
                                      Defaults to the number of CPU cores available.

    Returns:
        tuple: A tuple containing train DataLoader, test DataLoader, and class names.
    """
    train_data = datasets.ImageFolder(root=train_dir,
                                      transform=transform)
    test_data = datasets.ImageFolder(root=test_dir,
                                     transform=transform)

    train_dataloader = DataLoader(dataset=train_data,
                                  batch_size=batch_size,
                                  shuffle=True,
                                  num_workers=num_workers,
                                  pin_memory=True)
    test_dataloader = DataLoader(dataset=test_data,
                                  batch_size=batch_size,
                                  shuffle=False,
                                  num_workers=num_workers,
                                  pin_memory=True)

    class_names = train_data.classes

    return train_dataloader, test_dataloader, class_names

import os
from typing import Tuple

import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms


CIFAR10_CLASSES = [
    'airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck'
]


def get_transforms(train: bool = True):
    if train:
        return transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
    else:
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])


def get_dataloaders(data_dir: str = './data', batch_size: int = 128, val_split: float = 0.1, num_workers: int = 4) -> Tuple[DataLoader, DataLoader, DataLoader]:
    os.makedirs(data_dir, exist_ok=True)
    train_transform = get_transforms(train=True)
    test_transform = get_transforms(train=False)

    full_train = datasets.CIFAR10(root=data_dir, train=True, download=True, transform=train_transform)
    test_set = datasets.CIFAR10(root=data_dir, train=False, download=True, transform=test_transform)

    if val_split > 0:
        val_len = int(len(full_train) * val_split)
        train_len = len(full_train) - val_len
        train_set, val_set = random_split(full_train, [train_len, val_len])
        # ensure val uses test transforms
        val_set.dataset.transform = test_transform
    else:
        train_set = full_train
        val_set = None

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=num_workers) if val_set is not None else None
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader


from PIL import Image


def load_image_for_predict(image_path: str):
    img = Image.open(image_path).convert('RGB')
    transform = get_transforms(train=False)
    return transform(img).unsqueeze(0)

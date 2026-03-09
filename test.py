import argparse
import os

import torch
import torch.nn as nn

from models.cnn_model import get_model
from utils.dataset import get_dataloaders, CIFAR10_CLASSES


def evaluate(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, preds = outputs.max(1)
            correct += preds.eq(labels).sum().item()
            total += labels.size(0)
    return correct / total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=str, default='./data')
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--num-workers', type=int, default=4)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    _, _, test_loader = get_dataloaders(data_dir=args.data_dir, batch_size=args.batch_size, num_workers=args.num_workers)

    model = get_model(len(CIFAR10_CLASSES))
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)

    acc = evaluate(model, test_loader, device)
    print(f'Test Accuracy: {acc:.4f}')


if __name__ == '__main__':
    main()

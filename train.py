import argparse
import os
import time

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from models.cnn_model import get_model
from utils.dataset import get_dataloaders, CIFAR10_CLASSES


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for images, labels in tqdm(loader, desc='Train', leave=False):
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, preds = outputs.max(1)
        correct += preds.eq(labels).sum().item()
        total += labels.size(0)
    return running_loss / total, correct / total


def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in tqdm(loader, desc='Eval', leave=False):
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * images.size(0)
            _, preds = outputs.max(1)
            correct += preds.eq(labels).sum().item()
            total += labels.size(0)
    return running_loss / total, correct / total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=str, default='./data')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--save-dir', type=str, default='./checkpoints')
    parser.add_argument('--num-workers', type=int, default=4)
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_loader, val_loader, test_loader = get_dataloaders(data_dir=args.data_dir, batch_size=args.batch_size, num_workers=args.num_workers)

    model = get_model(len(CIFAR10_CLASSES)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

    best_acc = 0.0
    for epoch in range(1, args.epochs + 1):
        start = time.time()
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = (None, None)
        if val_loader is not None:
            val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        scheduler.step()
        elapsed = time.time() - start

        print(f'Epoch {epoch}/{args.epochs} - {elapsed:.1f}s - train_loss: {train_loss:.4f} train_acc: {train_acc:.4f} val_loss: {val_loss} val_acc: {val_acc}')

        # save best
        if val_acc is not None and val_acc > best_acc:
            best_acc = val_acc
            path = os.path.join(args.save_dir, f'best_epoch_{epoch}.pth')
            torch.save({'model_state_dict': model.state_dict(), 'epoch': epoch, 'acc': best_acc}, path)
            print('Saved', path)

    # final test
    if test_loader is not None:
        _, test_acc = evaluate(model, test_loader, criterion, device)
        print('Final test accuracy:', test_acc)


if __name__ == '__main__':
    main()

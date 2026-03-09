import argparse
import torch
import torch.nn.functional as F

from models.cnn_model import get_model
from utils.dataset import load_image_for_predict, CIFAR10_CLASSES


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', type=str, required=True, help='Path to input image')
    parser.add_argument('--checkpoint', type=str, required=True, help='Model checkpoint .pth')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = get_model(len(CIFAR10_CLASSES))
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    model = model.to(device)
    model.eval()

    img = load_image_for_predict(args.image).to(device)
    with torch.no_grad():
        logits = model(img)
        probs = F.softmax(logits, dim=1)[0]
        top_prob, top_idx = probs.max(0)

    print(f'Predicted: {CIFAR10_CLASSES[top_idx]} ({top_prob:.4f})')


if __name__ == '__main__':
    main()

# CIFAR10_Project

Minimal PyTorch project scaffold for CIFAR-10 classification.

Structure:
- `data/` - raw/downloaded dataset
- `models/cnn_model.py` - CNN model definition
- `utils/dataset.py` - data loading and transforms
- `checkpoints/` - saved model weights
- `train.py` - training script
- `test.py` - evaluation script
- `predict.py` - single-image prediction
- `requirements.txt` - python dependencies

Quick start:

1. Create a virtual env and install requirements:

```bash
pip install -r requirements.txt
```

2. Train:

```bash
python train.py --epochs 20 --batch-size 128
```

3. Test:

```bash
python test.py --checkpoint checkpoints/best_epoch_10.pth
```

4. Predict a single image:

```bash
python predict.py --image path/to/img.jpg --checkpoint checkpoints/best_epoch_10.pth
```

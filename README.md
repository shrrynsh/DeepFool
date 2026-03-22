# DeepFool (PyTorch)

Minimal DeepFool attack demo on a pretrained ResNet-34.

Paper: https://arxiv.org/abs/1511.04599

## Files
- `deepfool.py`: DeepFool implementation
- `test.py`: run attack + visualize perturbed image

## Install
```bash
pip install torch torchvision numpy matplotlib pillow
```

## Run
```bash
python test.py
```

Expected output:
- Original and perturbed class labels in terminal
- Perturbed image plot

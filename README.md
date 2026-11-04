# wikiart-dcgan

Minimal PyTorch implementation of a DCGAN trained on WikiArt-style images, with
evaluation using FID and Inception Score.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Data layout

Point the training/evaluation scripts to a directory containing images:

```text
data/wikiart/
  image1.jpg
  image2.png
  ...
```

Nested folders also work.

## Train

```bash
python train.py \
  --data-dir /path/to/wikiart \
  --output-dir outputs/run1 \
  --epochs 50 \
  --batch-size 64 \
  --image-size 64
```

This saves:

- model checkpoints (`generator.pt`, `discriminator.pt`)
- generated sample grid (`samples.png`)

## Evaluate (FID + Inception Score)

```bash
python evaluate.py \
  --data-dir /path/to/wikiart \
  --generator-checkpoint outputs/run1/generator.pt \
  --num-samples 5000 \
  --batch-size 64 \
  --image-size 64
```

The script prints FID and Inception Score for generated samples against real
images from the dataset.
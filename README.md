# wikiart-dcgan-pytorch

DCGAN in PyTorch trained on WikiArt to generate paintings, evaluated with FID and Inception Score.

## Setup

```bash
pip install -r requirements.txt
```

## 1. Prepare the data

Downloads [steubk/wikiart](https://www.kaggle.com/datasets/steubk/wikiart) with kagglehub (~31 GB),
applies the tutorial's `Resize(64)` + `CenterCrop(64)` to every image once and caches everything as
a single uint8 tensor (~1 GB) in `~/.wikiart-data/`.

```bash
python prepare_data.py                              # all 27 styles
python prepare_data.py --styles Impressionism       # a single style
python prepare_data.py --data-root /path/to/wikiart # already downloaded / extracted
```

## 2. Train

```bash
python train.py --epochs 5 --name baseline
```

Model, weight initialization, loss and training loop are those of the official
[PyTorch DCGAN tutorial](https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html)
(DCGAN of Radford et al., 2015): G maps z ~ N(0, I) of size 100 through 5 transposed convolutions to
64x64x3 (Tanh), D mirrors it with strided convolutions, LeakyReLU(0.2) and a sigmoid output, BatchNorm
in both, weights from N(0, 0.02), `BCELoss`, Adam (lr 0.0002, beta1 0.5), batch size 128, seed 999,
5 epochs by default. The only difference is that the images come from the cache above instead of an
`ImageFolder` that decodes the JPEGs every epoch.

Output in `~/.wikiart-runs/<name>/`: `samples/epoch_XXX.png` (fixed noise, every epoch),
`losses.png`, `history.json` and `G_epochXXX.pt` generator snapshots.

## 3. Evaluate

```bash
python evaluate.py --run baseline          # newest generator snapshot
python evaluate.py --run baseline --all    # every snapshot, e.g. FID over training
```

Inception Score and FID with pytorch-ignite following the
[ignite GAN evaluation blog post](https://pytorch-ignite.ai/blog/gan-evaluation-with-fid-and-is/)
(10k generated vs. 10k real images by default). Two FIDs are reported:

- `fid`: standard 2048-d pool3 FID (pytorch-fid weights), comparable with the literature
- `fid_ignite`: ignite's default FID on the 1000-d softmax output, as in the blog post

## Files

| file              | content                                        |
|-------------------|------------------------------------------------|
| `models.py`       | DCGAN generator, discriminator, weight init    |
| `prepare_data.py` | download, resize and cache WikiArt             |
| `train.py`        | training loop, samples, loss curves            |
| `evaluate.py`     | IS and FID with pytorch-ignite                 |

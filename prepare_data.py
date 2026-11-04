"""Download WikiArt and cache it as one uint8 tensor of 64x64 images.

Decoding ~80k full-size JPEGs every epoch would be far too slow, so every image is
resized once with the tutorial's Resize(64) + CenterCrop(64) and stored as
N x 3 x 64 x 64 uint8 together with its style label. The cache is ~1 GB and is
loaded straight onto the GPU during training.

  python prepare_data.py                          # all 27 styles
  python prepare_data.py --styles Impressionism   # just one style
  python prepare_data.py --data-root /path/to/wikiart  # skip the kagglehub download
"""

import os

# the DataHub container has 8 CPUs but shows 64, so torch/numpy would start 64 threads
# each and slow everything down (same fix as in hw1); must be set before importing them
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")

import argparse
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

DEFAULT_CACHE_DIR = Path.home() / ".wikiart-data"  # outside ~/private (quota)


def find_style_dirs(root):
    # the Kaggle archive has one folder per style, possibly inside an extra "wikiart" folder
    root = Path(root)
    if (root / "wikiart").is_dir():
        root = root / "wikiart"
    return sorted(d for d in root.iterdir() if d.is_dir())


def load_image(path, size):
    # the tutorial's transforms.Resize(image_size) + transforms.CenterCrop(image_size),
    # stored as uint8 (ToTensor + Normalize are applied in train.py)
    try:
        img = Image.open(path).convert("RGB")
    except Exception as e:  # a few files in the dump are broken
        print(f"skipping {path}: {e}")
        return None
    img = transforms.CenterCrop(size)(transforms.Resize(size)(img))
    return np.asarray(img, dtype=np.uint8).transpose(2, 0, 1)


def _load(job):
    return load_image(*job)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-root", type=Path, default=None, help="extracted WikiArt folder (skips kagglehub)")
    p.add_argument("--styles", nargs="*", default=None, help="style folder names to keep (default: all)")
    p.add_argument("--image-size", type=int, default=64)
    p.add_argument("--out", type=Path, default=None, help="cache file (default: ~/.wikiart-data/wikiart_<size>[_styles].pt)")
    p.add_argument("--workers", type=int, default=8)
    args = p.parse_args()

    if args.data_root is None:
        import kagglehub
        args.data_root = Path(kagglehub.dataset_download("steubk/wikiart"))
    style_dirs = find_style_dirs(args.data_root)
    if args.styles:
        style_dirs = [d for d in style_dirs if d.name in args.styles]
        assert style_dirs, f"none of {args.styles} found in {args.data_root}"
    style_names = [d.name for d in style_dirs]

    jobs, labels = [], []
    for label, d in enumerate(style_dirs):
        files = sorted(f for f in d.iterdir() if f.suffix.lower() in (".jpg", ".jpeg", ".png"))
        print(f"{d.name:35s} {len(files):6d} images")
        jobs += [(f, args.image_size) for f in files]
        labels += [label] * len(files)

    with Pool(args.workers) as pool:
        imgs = pool.map(_load, jobs, chunksize=64)
    keep = [i for i, im in enumerate(imgs) if im is not None]
    images = torch.from_numpy(np.stack([imgs[i] for i in keep]))
    labels = torch.tensor([labels[i] for i in keep])

    if args.out is None:
        tag = "_" + "-".join(args.styles) if args.styles else ""
        args.out = DEFAULT_CACHE_DIR / f"wikiart_{args.image_size}{tag}.pt"
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"images": images, "labels": labels, "styles": style_names}, args.out)
    print(f"saved {tuple(images.shape)} to {args.out}")


if __name__ == "__main__":
    main()

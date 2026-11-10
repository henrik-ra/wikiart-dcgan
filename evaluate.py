"""Inception Score and FID of generated paintings with pytorch-ignite.

Follows https://pytorch-ignite.ai/blog/gan-evaluation-with-fid-and-is/ : both image
sets are resized to 299x299 with PIL (bilinear) and fed through ignite's FID and
InceptionScore metrics inside an ignite Engine.

ignite's default FID uses the 1000-d softmax output of Inception-v3 as features
("fid_ignite"), which gives small numbers that are not comparable with papers. So
the standard 2048-d pool3 FID with the pytorch-fid weights is computed as well
("fid"), through the feature-extractor wrapper from the ignite FID docs.

  python evaluate.py --run baseline                    # newest G snapshot of the run
  python evaluate.py --run baseline --all              # every G_epochXXX.pt (FID over training)
  python evaluate.py --run baseline --n 10000
  python evaluate.py --run baseline --all --every 5      # every 5th snapshot
  python evaluate.py --run cond --style Impressionism    # FID against one style only

A snapshot with an EMA generator (train.py --ema) is evaluated twice, as "G" and "G_ema".
For a conditional G the generated images get the labels of the real images they are
compared with, so they follow the same style distribution.
"""

import os

# the DataHub container has 8 CPUs but shows 64, so torch/numpy would start 64 threads
# each and slow everything down (same fix as in hw1); must be set before importing them
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")

import argparse
import json
import re
import time
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, TensorDataset
from torchvision import transforms
from torchvision.utils import save_image

from models import Generator, ResGenerator
from train import DEFAULT_RUN_DIR


def interpolate(batch, device):
    # batch: float images in [0, 1]; PIL resize to the Inception input size as in the blog
    arr = []
    for img in batch.cpu():
        pil = transforms.ToPILImage()(img).resize((299, 299), Image.BILINEAR)
        arr.append(transforms.ToTensor()(pil))
    return torch.stack(arr).to(device)


class WrapperInceptionV3(nn.Module):
    # pool3 features of the pytorch-fid Inception network (2048-d)
    def __init__(self, fid_incv3):
        super().__init__()
        self.fid_incv3 = fid_incv3

    @torch.no_grad()
    def forward(self, x):
        return self.fid_incv3(x)[0][:, :, 0, 0]


def build_evaluator(step_fn, device, with_real_is):
    from ignite.engine import Engine, Events
    from ignite.metrics import FID, InceptionScore
    from pytorch_fid.inception import InceptionV3

    pool3 = WrapperInceptionV3(InceptionV3([InceptionV3.BLOCK_INDEX_BY_DIM[2048]]).to(device)).eval()
    ev = Engine(step_fn)
    FID(num_features=2048, feature_extractor=pool3, device=device).attach(ev, "fid")
    FID(device=device).attach(ev, "fid_ignite")
    InceptionScore(device=device, output_transform=lambda o: o[0]).attach(ev, "is")
    if with_real_is:
        InceptionScore(device=device, output_transform=lambda o: o[1]).attach(ev, "is_real")
    ev.add_event_handler(Events.ITERATION_COMPLETED(every=50),
                         lambda e: print(f"  batch {e.state.iteration}/{e.state.epoch_length}", flush=True))
    return ev


def evaluate_checkpoint(path, images, labels, args, device, with_real_is):
    ck = torch.load(path, map_location=device)
    n_classes = ck.get("n_classes", 0)
    results = []
    for key in ("G", "G_ema"):
        if key not in ck:
            continue
        if ck.get("arch", "dcgan") == "resnet":
            G = ResGenerator(ck["nz"], ck["ngf"]).to(device)
        else:
            G = Generator(ck["nz"], ck["ngf"], n_classes=n_classes,
                          image_size=ck.get("image_size", 64)).to(device)
        G.load_state_dict(ck[key])
        # like the tutorial, images are generated with G in train mode (BatchNorm uses the
        # statistics of the current batch), the snapshot itself is not changed

        @torch.no_grad()
        def step(engine, batch):
            real = batch[0].to(device).float().div(255)
            y = batch[1].to(device) if n_classes else None
            z = torch.randn(len(real), ck["nz"], 1, 1, device=device)
            fake = G(z, y).add(1).div(2).clamp(0, 1)  # [-1, 1] -> [0, 1]
            return interpolate(fake, device), interpolate(real, device)

        sub = torch.randperm(len(images))[:args.n]
        t0 = time.time()
        m = build_evaluator(step, device, with_real_is and key == "G").run(
            DataLoader(TensorDataset(images[sub], labels[sub]), args.batch_size), max_epochs=1).metrics
        res = {"checkpoint": path.name, "G": key, "epoch": ck["epoch"], "iteration": ck.get("iteration"),
               "style": args.style, "n": len(sub), **m, "eval_time_s": round(time.time() - t0)}
        print(json.dumps(res), flush=True)
        results.append(res)

        with torch.no_grad():  # a fresh 8x8 grid for the report
            z = torch.randn(64, ck["nz"], 1, 1, device=device)
            y = labels[torch.randint(len(labels), (64,))].to(device) if n_classes else None
            tag = ("_ema" if key == "G_ema" else "") + (f"_{args.style}" if args.style else "")
            save_image(G(z, y), path.with_name(f"{path.stem}{tag}_grid.png"), nrow=8, padding=2, normalize=True)
    return results


def epoch_of(path):
    return int(re.search(r"G_epoch(\d+)\.pt$", path.name).group(1))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", required=True, help="run name in runs/")
    p.add_argument("--data", type=Path, default=None, help="real images to compare with (default: the run's data)")
    p.add_argument("--n", type=int, default=10000, help="number of generated and of real images")
    p.add_argument("--batch-size", type=int, default=100)
    p.add_argument("--all", action="store_true", help="evaluate every G snapshot, not only the newest")
    p.add_argument("--every", type=int, default=1, help="with --all: only every N-th snapshot (and the last)")
    p.add_argument("--style", default=None, help="compare with the real images of this style only")
    p.add_argument("--epoch", type=int, default=None, help="evaluate only the snapshot of this epoch")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run = DEFAULT_RUN_DIR / args.run
    run_args = json.loads((run / "args.json").read_text())
    if args.data is None:
        args.data = Path(run_args["data"])
    data = torch.load(args.data.expanduser())
    images, labels, styles = data["images"], data["labels"], data["styles"]
    keep_styles = [args.style] if args.style else run_args.get("styles")  # the run's training styles
    if keep_styles:
        keep = torch.isin(labels, torch.tensor([styles.index(s) for s in keep_styles]))
        images, labels = images[keep], labels[keep]

    ckpts = sorted(run.glob("G_epoch*.pt"), key=epoch_of)
    assert ckpts, f"no G_epochXXX.pt in {run}"
    if args.epoch is not None:
        ckpts = [c for c in ckpts if epoch_of(c) == args.epoch]
        assert ckpts, f"no snapshot of epoch {args.epoch} in {run}"
    elif args.all:
        ckpts = [c for i, c in enumerate(ckpts) if (i + 1) % args.every == 0 or c == ckpts[-1]]
    else:
        ckpts = ckpts[-1:]
    results = []
    for i, c in enumerate(ckpts):
        results += evaluate_checkpoint(c, images, labels, args, device, with_real_is=(i == 0))

    name = f"metrics_epoch{args.epoch}" if args.epoch is not None else ("metrics_all" if args.all else "metrics")
    out = run / (f"{name}_{args.style}.json" if args.style else f"{name}.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"saved {out}")


if __name__ == "__main__":
    main()

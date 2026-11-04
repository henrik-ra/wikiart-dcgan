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
"""

import os

# the DataHub container has 8 CPUs but shows 64, so torch/numpy would start 64 threads
# each and slow everything down (same fix as in hw1); must be set before importing them
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, TensorDataset
from torchvision import transforms
from torchvision.utils import save_image

from models import Generator
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


def evaluate_checkpoint(path, images, args, device, with_real_is):
    ck = torch.load(path, map_location=device)
    G = Generator(ck["nz"], ck["ngf"]).to(device)
    G.load_state_dict(ck["G"])
    # like the tutorial, images are generated with G in train mode (BatchNorm uses the
    # statistics of the current batch), the snapshot itself is not changed

    @torch.no_grad()
    def step(engine, batch):
        real = batch[0].to(device).float().div(255)
        z = torch.randn(len(real), ck["nz"], 1, 1, device=device)
        fake = G(z).add(1).div(2).clamp(0, 1)  # [-1, 1] -> [0, 1]
        return interpolate(fake, device), interpolate(real, device)

    real_subset = images[torch.randperm(len(images))[:args.n]]
    t0 = time.time()
    m = build_evaluator(step, device, with_real_is).run(DataLoader(TensorDataset(real_subset), args.batch_size),
                                                       max_epochs=1).metrics
    res = {"checkpoint": path.name, "epoch": ck["epoch"], "n": len(real_subset), **m,
           "eval_time_s": round(time.time() - t0)}
    print(json.dumps(res))

    with torch.no_grad():  # a fresh 8x8 grid for the report
        z = torch.randn(64, ck["nz"], 1, 1, device=device)
        save_image(G(z), path.with_name(path.stem + "_grid.png"), nrow=8, padding=2, normalize=True)
    return res


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", required=True, help="run name in ~/.wikiart-runs")
    p.add_argument("--data", type=Path, default=None, help="real images to compare with (default: the run's data)")
    p.add_argument("--n", type=int, default=10000, help="number of generated and of real images")
    p.add_argument("--batch-size", type=int, default=100)
    p.add_argument("--all", action="store_true", help="evaluate every G snapshot, not only the newest")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run = DEFAULT_RUN_DIR / args.run
    if args.data is None:
        args.data = Path(json.loads((run / "args.json").read_text())["data"])
    images = torch.load(args.data.expanduser())["images"]

    ckpts = sorted(run.glob("G_epoch*.pt"))
    assert ckpts, f"no G_epochXXX.pt in {run}"
    if not args.all:
        ckpts = ckpts[-1:]
    results = [evaluate_checkpoint(c, images, args, device, with_real_is=(i == 0)) for i, c in enumerate(ckpts)]

    out = run / ("metrics_all.json" if args.all else "metrics.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"saved {out}")


if __name__ == "__main__":
    main()

"""Train a DCGAN on WikiArt, following the PyTorch DCGAN tutorial:
https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html

Model, initialization, loss, optimizers, hyperparameters and the training loop are the
tutorial's. The only practical difference: instead of decoding the JPEGs with
ImageFolder every epoch, the images are loaded from the cache of prepare_data.py,
which applied the tutorial's Resize(64) + CenterCrop(64) once.

  python train.py --epochs 5 --name baseline

Everything a run produces goes to ~/.wikiart-runs/<name>/ (outside ~/private):
  samples/epoch_XXX.png  64 images from the fixed noise after every epoch
  history.json           G and D loss and D outputs of every iteration
  losses.png             training-loss curves (as in the tutorial)
  G_epochXXX.pt          generator after every --save-every epochs (for evaluate.py)
"""

import os

# the DataHub container has 8 CPUs but shows 64, so torch/numpy would start 64 threads
# each and slow everything down (same fix as in hw1); must be set before importing them
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")

import argparse
import json
import random
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.utils import save_image

from models import Discriminator, Generator, weights_init

DEFAULT_RUN_DIR = Path.home() / ".wikiart-runs"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", type=Path, default=Path.home() / ".wikiart-data" / "wikiart_64.pt")
    p.add_argument("--name", default="baseline", help="run name, output goes to ~/.wikiart-runs/<name>")
    # hyperparameters of the tutorial
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--nz", type=int, default=100, help="size of z latent vector")
    p.add_argument("--ngf", type=int, default=64, help="size of feature maps in generator")
    p.add_argument("--ndf", type=int, default=64, help="size of feature maps in discriminator")
    p.add_argument("--lr", type=float, default=0.0002)
    p.add_argument("--beta1", type=float, default=0.5)
    p.add_argument("--seed", type=int, default=999)
    p.add_argument("--save-every", type=int, default=1, help="save G every N epochs")
    return p.parse_args()


def plot_losses(G_losses, D_losses, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 5))
    plt.title("Generator and Discriminator Loss During Training")
    plt.plot(G_losses, label="G")
    plt.plot(D_losses, label="D")
    plt.xlabel("iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()


def main():
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    out = DEFAULT_RUN_DIR / args.name
    (out / "samples").mkdir(parents=True, exist_ok=True)
    with open(out / "args.json", "w") as f:
        json.dump({k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}, f, indent=2)

    # uint8 images, already Resize(64) + CenterCrop(64); the tutorial's ToTensor() and
    # Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) map them to [-1, 1], done per batch below
    data = torch.load(args.data.expanduser())
    images = data["images"].to(device)
    n = len(images)
    num_batches = (n + args.batch_size - 1) // args.batch_size  # DataLoader keeps the last batch
    print(f"{n} images from {len(data['styles'])} styles, {num_batches} batches per epoch")

    netG = Generator(args.nz, args.ngf).to(device)
    netG.apply(weights_init)
    netD = Discriminator(args.ndf).to(device)
    netD.apply(weights_init)
    print(netG)
    print(netD)

    criterion = nn.BCELoss()
    # batch of latent vectors to visualize the progression of the generator
    fixed_noise = torch.randn(64, args.nz, 1, 1, device=device)
    real_label = 1.0
    fake_label = 0.0
    optimizerD = optim.Adam(netD.parameters(), lr=args.lr, betas=(args.beta1, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=args.lr, betas=(args.beta1, 0.999))

    G_losses, D_losses, D_x_list, D_G_z1_list, D_G_z2_list = [], [], [], [], []
    print("Starting Training Loop...")
    for epoch in range(args.epochs):
        t0 = time.time()
        perm = torch.randperm(n, device=device)  # shuffle=True
        for i in range(num_batches):
            batch = images[perm[i * args.batch_size:(i + 1) * args.batch_size]]
            real_cpu = batch.float().div(255).sub(0.5).div(0.5)  # ToTensor + Normalize

            ############################
            # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
            ###########################
            ## Train with all-real batch
            netD.zero_grad()
            b_size = real_cpu.size(0)
            label = torch.full((b_size,), real_label, dtype=torch.float, device=device)
            output = netD(real_cpu).view(-1)
            errD_real = criterion(output, label)
            errD_real.backward()
            D_x = output.mean().item()

            ## Train with all-fake batch
            noise = torch.randn(b_size, args.nz, 1, 1, device=device)
            fake = netG(noise)
            label.fill_(fake_label)
            output = netD(fake.detach()).view(-1)
            errD_fake = criterion(output, label)
            errD_fake.backward()
            D_G_z1 = output.mean().item()
            errD = errD_real + errD_fake
            optimizerD.step()

            ############################
            # (2) Update G network: maximize log(D(G(z)))
            ###########################
            netG.zero_grad()
            label.fill_(real_label)  # fake labels are real for generator cost
            output = netD(fake).view(-1)
            errG = criterion(output, label)
            errG.backward()
            D_G_z2 = output.mean().item()
            optimizerG.step()

            if i % 50 == 0:
                print("[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f"
                      % (epoch, args.epochs, i, num_batches, errD.item(), errG.item(), D_x, D_G_z1, D_G_z2),
                      flush=True)

            G_losses.append(errG.item())
            D_losses.append(errD.item())
            D_x_list.append(D_x)
            D_G_z1_list.append(D_G_z1)
            D_G_z2_list.append(D_G_z2)

        # after every epoch: images from the fixed noise and a generator snapshot
        e = epoch + 1
        with torch.no_grad():
            fake = netG(fixed_noise).detach().cpu()
        save_image(fake, out / "samples" / f"epoch_{e:03d}.png", nrow=8, padding=2, normalize=True)
        if e % args.save_every == 0 or e == args.epochs:
            torch.save({"G": netG.state_dict(), "epoch": e, "nz": args.nz, "ngf": args.ngf},
                       out / f"G_epoch{e:03d}.pt")
        print(f"epoch {e} done in {time.time() - t0:.0f} s", flush=True)

    with open(out / "history.json", "w") as f:
        json.dump({"G_losses": G_losses, "D_losses": D_losses, "D_x": D_x_list,
                   "D_G_z1": D_G_z1_list, "D_G_z2": D_G_z2_list, "batches_per_epoch": num_batches}, f)
    plot_losses(G_losses, D_losses, out / "losses.png")
    print(f"done, results in {out}")


if __name__ == "__main__":
    main()

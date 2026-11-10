"""Train a DCGAN on WikiArt, following the PyTorch DCGAN tutorial:
https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html

With the default arguments, model, initialization, loss, optimizers, hyperparameters
and the training loop are the tutorial's. The only practical difference: instead of
decoding the JPEGs with ImageFolder every epoch, the images are loaded from the cache
of prepare_data.py, which applied the tutorial's Resize(64) + CenterCrop(64) once.

  python train.py --epochs 5 --name baseline

Optional changes for the experiments (all off by default):
  --styles Impressionism       train on a subset of the styles
  --real-label 0.9             one-sided label smoothing (Salimans et al., 2016)
  --lr-d 4e-4 --lr-g 1e-4      two time-scale update rule (Heusel et al., 2017)
  --sn                         spectral norm in D instead of BatchNorm (Miyato et al., 2018)
  --loss hinge                 hinge loss (as in SNGAN)
  --loss wgan-gp               WGAN-GP (Gulrajani et al., 2017), use with --n-critic 5
  --ema 0.999                  also keep an exponential moving average of G
  --diffaug                    DiffAugment on all D inputs (Zhao et al., 2020)
  --cond                       conditional DCGAN on the style labels
  --ngf 128 --ndf 128          wider networks
  --arch resnet                ResNet GAN with self-attention (SAGAN), use with --loss hinge

Everything a run produces goes to runs/<name>/ (ignored by git):
  samples/epoch_XXX.png  64 images from the fixed noise after every --sample-every epochs
  history.json           G and D loss and D outputs of every iteration
  losses.png             training-loss curves (as in the tutorial)
  G_epochXXX.pt          generator after every --save-every epochs (for evaluate.py)
  last.pt                full training state after every epoch, for --resume
"""

import os

# the DataHub container has 8 CPUs but shows 64, so torch/numpy would start 64 threads
# each and slow everything down (same fix as in hw1); must be set before importing them
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")

import argparse
import copy
import json
import random
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision.utils import save_image

from models import Discriminator, Generator, ResDiscriminator, ResGenerator, diff_augment, weights_init

DEFAULT_RUN_DIR = Path(__file__).resolve().parent / "runs"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", type=Path, default=Path.home() / ".wikiart-data" / "wikiart_64.pt")
    p.add_argument("--name", default="baseline", help="run name, output goes to runs/<name>")
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
    p.add_argument("--sample-every", type=int, default=1, help="save the fixed-noise grid every N epochs")
    # experiments, the defaults are the tutorial's setting
    p.add_argument("--styles", nargs="*", default=None, help="train only on these styles")
    p.add_argument("--beta2", type=float, default=0.999)
    p.add_argument("--lr-g", type=float, default=None, help="generator lr (default: --lr)")
    p.add_argument("--lr-d", type=float, default=None, help="discriminator lr (default: --lr)")
    p.add_argument("--real-label", type=float, default=1.0, help="target of real images for D, e.g. 0.9")
    p.add_argument("--loss", choices=["bce", "hinge", "wgan-gp"], default="bce")
    p.add_argument("--n-critic", type=int, default=1, help="D updates per G update")
    p.add_argument("--gp-weight", type=float, default=10.0, help="gradient penalty weight for wgan-gp")
    p.add_argument("--sn", action="store_true", help="spectral norm in D (BatchNorm in D is dropped)")
    p.add_argument("--ema", type=float, default=0.0, help="decay of an EMA copy of G, 0 = off")
    p.add_argument("--diffaug", action="store_true", help="DiffAugment color,translation,cutout")
    p.add_argument("--cond", action="store_true", help="conditional on the style label")
    p.add_argument("--resume", action="store_true", help="continue from runs/<name>/last.pt if it exists")
    p.add_argument("--arch", choices=["dcgan", "resnet"], default="dcgan",
                   help="resnet: SAGAN-style ResNet G and D with spectral norm and self-attention")
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


def gradient_penalty(netD, real, fake, labels):
    a = torch.rand(real.size(0), 1, 1, 1, device=real.device)
    x = (a * real + (1 - a) * fake).requires_grad_(True)
    out = netD(x, labels)
    (g,) = torch.autograd.grad(out.sum(), x, create_graph=True)
    return ((g.flatten(1).norm(2, dim=1) - 1) ** 2).mean()


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
    images, labels, styles = data["images"], data["labels"], data["styles"]
    if args.styles:
        unknown = set(args.styles) - set(styles)
        assert not unknown, f"unknown styles {unknown}, available: {styles}"
        keep = torch.isin(labels, torch.tensor([styles.index(s) for s in args.styles]))
        images, labels = images[keep], labels[keep]
    images, labels = images.to(device), labels.to(device)
    n = len(images)
    num_batches = (n + args.batch_size - 1) // args.batch_size  # DataLoader keeps the last batch
    print(f"{n} images from {len(args.styles or styles)} styles, {num_batches} batches per epoch")

    n_classes = len(styles) if args.cond else 0
    image_size = images.shape[-1]  # 64 or 128, given by the cache
    if args.arch == "resnet":
        assert image_size == 64, "the resnet GAN is built for 64x64"
        assert args.loss != "bce" and not args.cond, "the resnet D returns raw scores, use --loss hinge"
        netG = ResGenerator(args.nz, args.ngf).to(device)  # Xavier init inside, as in SNGAN/SAGAN
        netD = ResDiscriminator(args.ndf).to(device)
    else:
        netG = Generator(args.nz, args.ngf, n_classes=n_classes, image_size=image_size).to(device)
        netG.apply(weights_init)
        netD = Discriminator(args.ndf, n_classes=n_classes,
                             norm="none" if (args.sn or args.loss == "wgan-gp") else "bn",
                             sn=args.sn, sigmoid=(args.loss == "bce"), image_size=image_size).to(device)
        netD.apply(weights_init)
    print(netG)
    print(netD)
    netG_ema = copy.deepcopy(netG).requires_grad_(False) if args.ema else None

    criterion = nn.BCELoss()
    # batch of latent vectors to visualize the progression of the generator
    fixed_noise = torch.randn(64, args.nz, 1, 1, device=device)
    present = labels.unique()
    fixed_labels = present[torch.arange(64, device=device) % len(present)] if args.cond else None
    real_label = args.real_label
    fake_label = 0.0
    optimizerD = optim.Adam(netD.parameters(), lr=args.lr_d or args.lr, betas=(args.beta1, args.beta2))
    optimizerG = optim.Adam(netG.parameters(), lr=args.lr_g or args.lr, betas=(args.beta1, args.beta2))
    aug = diff_augment if args.diffaug else (lambda x: x)

    # D loss on real / fake scores and G loss on fake scores
    if args.loss == "bce":
        d_real = lambda o: criterion(o, torch.full_like(o, real_label))
        d_fake = lambda o: criterion(o, torch.full_like(o, fake_label))
        g_loss = lambda o: criterion(o, torch.ones_like(o))  # fake labels are real for generator cost
    elif args.loss == "hinge":
        d_real = lambda o: F.relu(1 - o).mean()
        d_fake = lambda o: F.relu(1 + o).mean()
        g_loss = lambda o: -o.mean()
    else:  # wgan-gp
        d_real = lambda o: -o.mean()
        d_fake = lambda o: o.mean()
        g_loss = lambda o: -o.mean()

    G_losses, D_losses, D_x_list, D_G_z1_list, D_G_z2_list = [], [], [], [], []
    errG, D_G_z2 = torch.tensor(float("nan")), float("nan")
    start_epoch = 0
    if args.resume and (out / "last.pt").exists():
        st = torch.load(out / "last.pt", map_location=device)
        netG.load_state_dict(st["G"]); netD.load_state_dict(st["D"])
        optimizerG.load_state_dict(st["optG"]); optimizerD.load_state_dict(st["optD"])
        if netG_ema is not None:
            netG_ema.load_state_dict(st["G_ema"])
        fixed_noise, fixed_labels = st["fixed_noise"], st["fixed_labels"]
        G_losses, D_losses, D_x_list, D_G_z1_list, D_G_z2_list = (st["history"][k] for k in
                                                                  ("G", "D", "D_x", "D_G_z1", "D_G_z2"))
        torch.set_rng_state(st["rng"].cpu())  # map_location moved it to the GPU
        if torch.cuda.is_available() and st.get("rng_cuda") is not None:
            torch.cuda.set_rng_state(st["rng_cuda"].cpu())
        start_epoch = st["epoch"]
        print(f"resumed from {out / 'last.pt'} after epoch {start_epoch}")
    print("Starting Training Loop...")
    for epoch in range(start_epoch, args.epochs):
        t0 = time.time()
        perm = torch.randperm(n, device=device)  # shuffle=True
        for i in range(num_batches):
            idx = perm[i * args.batch_size:(i + 1) * args.batch_size]
            real_cpu = images[idx].float().div(255).sub(0.5).div(0.5)  # ToTensor + Normalize
            y = labels[idx] if args.cond else None

            ############################
            # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
            ###########################
            ## Train with all-real batch
            netD.zero_grad()
            b_size = real_cpu.size(0)
            output = netD(aug(real_cpu), y).view(-1)
            errD_real = d_real(output)
            errD_real.backward()
            D_x = output.mean().item()

            ## Train with all-fake batch
            noise = torch.randn(b_size, args.nz, 1, 1, device=device)
            fake = netG(noise, y)
            output = netD(aug(fake.detach()), y).view(-1)
            errD_fake = d_fake(output)
            if args.loss == "wgan-gp":
                errD_fake = errD_fake + args.gp_weight * gradient_penalty(netD, real_cpu, fake.detach(), y)
            errD_fake.backward()
            D_G_z1 = output.mean().item()
            errD = errD_real + errD_fake
            optimizerD.step()

            ############################
            # (2) Update G network: maximize log(D(G(z)))
            ###########################
            if i % args.n_critic == 0:
                netG.zero_grad()
                output = netD(aug(fake), y).view(-1)
                errG = g_loss(output)
                errG.backward()
                D_G_z2 = output.mean().item()
                optimizerG.step()
                if netG_ema is not None:
                    with torch.no_grad():
                        for pe, p in zip(netG_ema.parameters(), netG.parameters()):
                            pe.lerp_(p, 1 - args.ema)
                        for be, b in zip(netG_ema.buffers(), netG.buffers()):
                            be.copy_(b)

            if i % 50 == 0:
                print("[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f"
                      % (epoch, args.epochs, i, num_batches, errD.item(), errG.item(), D_x, D_G_z1, D_G_z2),
                      flush=True)

            G_losses.append(errG.item())
            D_losses.append(errD.item())
            D_x_list.append(D_x)
            D_G_z1_list.append(D_G_z1)
            D_G_z2_list.append(D_G_z2)

        # images from the fixed noise and a generator snapshot
        e = epoch + 1
        last = e == args.epochs
        if e % args.sample_every == 0 or last:
            with torch.no_grad():
                for tag, net in (("", netG), ("_ema", netG_ema)):
                    if net is not None:
                        fake = net(fixed_noise, fixed_labels).detach().cpu()
                        save_image(fake, out / "samples" / f"epoch_{e:03d}{tag}.png",
                                   nrow=8, padding=2, normalize=True)
        if e % args.save_every == 0 or last:
            ck = {"G": netG.state_dict(), "epoch": e, "iteration": e * num_batches, "nz": args.nz,
                  "ngf": args.ngf, "n_classes": n_classes, "styles": styles, "arch": args.arch,
                  "image_size": image_size}
            if netG_ema is not None:
                ck["G_ema"] = netG_ema.state_dict()
            torch.save(ck, out / f"G_epoch{e:03d}.pt")
        # full state for --resume, written to a temporary file first so that a crash
        # while saving never leaves a broken last.pt behind
        st = {"G": netG.state_dict(), "D": netD.state_dict(), "optG": optimizerG.state_dict(),
              "optD": optimizerD.state_dict(), "G_ema": netG_ema.state_dict() if netG_ema else None,
              "epoch": e, "fixed_noise": fixed_noise, "fixed_labels": fixed_labels,
              "history": {"G": G_losses, "D": D_losses, "D_x": D_x_list, "D_G_z1": D_G_z1_list,
                          "D_G_z2": D_G_z2_list},
              "rng": torch.get_rng_state(),
              "rng_cuda": torch.cuda.get_rng_state() if torch.cuda.is_available() else None}
        torch.save(st, out / "last.pt.tmp")
        os.replace(out / "last.pt.tmp", out / "last.pt")
        print(f"epoch {e} done in {time.time() - t0:.0f} s", flush=True)

    with open(out / "history.json", "w") as f:
        json.dump({"G_losses": G_losses, "D_losses": D_losses, "D_x": D_x_list,
                   "D_G_z1": D_G_z1_list, "D_G_z2": D_G_z2_list, "batches_per_epoch": num_batches}, f)
    plot_losses(G_losses, D_losses, out / "losses.png")
    print(f"done, results in {out}")


if __name__ == "__main__":
    main()

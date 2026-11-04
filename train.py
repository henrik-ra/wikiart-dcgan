import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.utils import make_grid, save_image
from tqdm import tqdm

from dcgan.data import build_dataset
from dcgan.model import Discriminator, Generator


def weights_init(module):
    name = module.__class__.__name__
    if name.startswith("Conv"):
        nn.init.normal_(module.weight.data, 0.0, 0.02)
    elif name.startswith("BatchNorm"):
        nn.init.normal_(module.weight.data, 1.0, 0.02)
        nn.init.constant_(module.bias.data, 0)


def parse_args():
    parser = argparse.ArgumentParser(description="Train DCGAN on WikiArt-style images.")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--latent-dim", type=int, default=100)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--beta1", type=float, default=0.5)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = build_dataset(args.data_dir, args.image_size)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=torch.cuda.is_available(),
    )

    generator = Generator(latent_dim=args.latent_dim).to(device)
    discriminator = Discriminator().to(device)
    generator.apply(weights_init)
    discriminator.apply(weights_init)

    criterion = nn.BCELoss()
    optimizer_d = optim.Adam(discriminator.parameters(), lr=args.lr, betas=(args.beta1, 0.999))
    optimizer_g = optim.Adam(generator.parameters(), lr=args.lr, betas=(args.beta1, 0.999))

    fixed_noise = torch.randn(64, args.latent_dim, 1, 1, device=device)

    for epoch in range(args.epochs):
        pbar = tqdm(dataloader, desc=f"epoch {epoch + 1}/{args.epochs}")
        for real, _ in pbar:
            real = real.to(device)
            batch_size = real.size(0)
            label_real = torch.ones(batch_size, device=device)
            label_fake = torch.zeros(batch_size, device=device)

            discriminator.zero_grad(set_to_none=True)
            out_real = discriminator(real)
            loss_real = criterion(out_real, label_real)

            noise = torch.randn(batch_size, args.latent_dim, 1, 1, device=device)
            fake = generator(noise)
            out_fake = discriminator(fake.detach())
            loss_fake = criterion(out_fake, label_fake)
            loss_d = loss_real + loss_fake
            loss_d.backward()
            optimizer_d.step()

            generator.zero_grad(set_to_none=True)
            out_fake_for_g = discriminator(fake)
            loss_g = criterion(out_fake_for_g, label_real)
            loss_g.backward()
            optimizer_g.step()

            pbar.set_postfix(
                {
                    "loss_d": f"{loss_d.item():.4f}",
                    "loss_g": f"{loss_g.item():.4f}",
                }
            )

    generator.eval()
    with torch.no_grad():
        samples = generator(fixed_noise).cpu()
        samples = (samples + 1) / 2
        grid = make_grid(samples, nrow=8)
        save_image(grid, output_dir / "samples.png")

    torch.save(generator.state_dict(), output_dir / "generator.pt")
    torch.save(discriminator.state_dict(), output_dir / "discriminator.pt")
    print(f"Saved outputs to {output_dir}")


if __name__ == "__main__":
    main()

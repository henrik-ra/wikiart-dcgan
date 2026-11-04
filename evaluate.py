import argparse

import torch
from torch.utils.data import DataLoader
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.inception import InceptionScore

from dcgan.data import build_dataset
from dcgan.model import Generator


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate DCGAN with FID and Inception Score.")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--generator-checkpoint", required=True)
    parser.add_argument("--num-samples", type=int, default=5000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--latent-dim", type=int, default=100)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def to_uint8(images: torch.Tensor) -> torch.Tensor:
    images = (images + 1) / 2
    images = images.clamp(0, 1)
    return (images * 255).to(torch.uint8)


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = build_dataset(args.data_dir, args.image_size)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=torch.cuda.is_available(),
    )

    generator = Generator(latent_dim=args.latent_dim).to(device)
    state_dict = torch.load(args.generator_checkpoint, map_location=device)
    generator.load_state_dict(state_dict)
    generator.eval()

    fid = FrechetInceptionDistance(feature=2048, normalize=False).to(device)
    inception = InceptionScore(normalize=False).to(device)

    seen = 0
    loader_iter = iter(dataloader)
    while seen < args.num_samples:
        try:
            real, _ = next(loader_iter)
        except StopIteration:
            loader_iter = iter(dataloader)
            real, _ = next(loader_iter)

        batch = min(real.size(0), args.num_samples - seen)
        real = real[:batch].to(device)
        noise = torch.randn(batch, args.latent_dim, 1, 1, device=device)
        with torch.no_grad():
            fake = generator(noise)

        real_uint8 = to_uint8(real)
        fake_uint8 = to_uint8(fake)
        fid.update(real_uint8, real=True)
        fid.update(fake_uint8, real=False)
        inception.update(fake_uint8)
        seen += batch

    fid_value = fid.compute().item()
    is_mean, is_std = inception.compute()

    print(f"FID: {fid_value:.4f}")
    print(f"Inception Score: {is_mean.item():.4f} ± {is_std.item():.4f}")


if __name__ == "__main__":
    main()

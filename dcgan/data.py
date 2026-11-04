from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


def build_transform(image_size: int):
    return transforms.Compose(
        [
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )


def build_dataset(data_dir: str, image_size: int):
    root = Path(data_dir)
    if not root.exists():
        raise FileNotFoundError(f"Data directory does not exist: {data_dir}")

    return RecursiveImageDataset(root=root, transform=build_transform(image_size))


class RecursiveImageDataset(Dataset):
    EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    def __init__(self, root: Path, transform):
        self.root = root
        self.transform = transform
        self.image_paths = sorted(
            p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in self.EXTENSIONS
        )
        if not self.image_paths:
            raise ValueError(f"No supported images found in: {root}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index: int):
        image_path = self.image_paths[index]
        with Image.open(image_path) as img:
            image = img.convert("RGB")
        return self.transform(image), 0

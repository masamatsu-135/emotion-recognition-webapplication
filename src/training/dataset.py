import csv
from pathlib import Path
from typing import Callable, Optional, List, Tuple

import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset
import torchvision.transforms as T


class FER2013ResNetDataset(Dataset):

    def __init__(self, csv_path, usage="Training", transform=None):
        self.csv_path = Path(csv_path)
        self.usage = usage
        self.transform = transform

        self.samples = []
        self.load_csv()

    def load_csv(self):
        if not self.csv_path.exists():
            raise FileNotFoundError(
                f"CSV not found: {self.csv_path}\n"
                f"Resolved absolute path: {self.csv_path.resolve()}"
    )
        with self.csv_path.open("r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row["Usage"] != self.usage:
                    continue
                
                label = int(row["emotion"])

                pixels_str = row["pixels"]
                pixels = np.fromstring(pixels_str, dtype=np.uint8, sep=" ")

                pixels = pixels.reshape(48, 48)

                self.samples.append((pixels, label))

        if len(self.samples) == 0:
            raise ValueError(
                f"No samples found for Usage='{self.usage}'."
                "Check csv_path or usage string."
            )
        
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        pixels, label = self.samples[idx]

        img = Image.fromarray(pixels)

        if self.transform is not None:
            img = self.transform(img)

        return img, label
    

def get_resnet_transforms(train=True):

    base_transforms = [
        T.Grayscale(num_output_channels=3),
        T.Resize((224, 224)),
    ]

    if train:
        aug = [
            T.RandomHorizontalFlip(p=0.5),
            T.RandomRotation(10),
        ]
        base_transforms = aug + base_transforms
    
    base_transforms += [
        T.ToTensor(),
        T.Normalize(
            mean=[0.485, 0.456, 0.406],
            std =[0.229, 0.224, 0.225],
        ),
    ]

    return T.Compose(base_transforms)

if __name__ == "__main__":
    this_file = Path(__file__).resolve()
    project_root = this_file.parents[2]

    csv_path = project_root / "data/raw/fer2013.csv"

    train_ds = FER2013ResNetDataset(
        csv_path=csv_path,
        usage="Training",
        transform=get_resnet_transforms(train=True),
    )

    val_ds = FER2013ResNetDataset(
        csv_path=csv_path,
        usage="PublicTest",
        transform=get_resnet_transforms(train=False),
    )

    print("Train size :", len(train_ds))
    print("Val size   :", len(val_ds))

    img, label = train_ds[0]
    print("Image shape:", img.shape)
    print("Label      :", label)
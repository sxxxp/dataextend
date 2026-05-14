from torch.utils.data import Dataset
from torchvision import datasets, transforms
import torch
from torchvision import transforms

# CIFAR10 augmentation
train_transform = transforms.Compose(
    [
        transforms.RandomCrop(
            32,
            padding=4,
        ),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(
            (0.5, 0.5, 0.5),
            (0.5, 0.5, 0.5),
        ),
    ]
)


# Test transform
test_transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize(
            (0.5, 0.5, 0.5),
            (0.5, 0.5, 0.5),
        ),
    ]
)


class FixedNoiseDatasets(Dataset):

    def __init__(
        self,
        stage=0,
        dataset: datasets = datasets.CIFAR10,
        train=True,
        noise_levels=[0.0, 0.05, 0.08, 0.1, 0.15],
        root="./data",
    ):
        self.stage = stage
        self.noise_levels = noise_levels

        self.base_dataset = dataset(
            root=root,
            train=train,
            download=True,
            transform=train_transform if train else test_transform,
        )

        self.noise_std = self.noise_levels[stage]

        self.cached_images = []

        for i in range(len(self.base_dataset)):

            x, y = self.base_dataset[i]

            if stage > 0:
                noise = torch.randn_like(x) * self.noise_std
                x = torch.clamp(x + noise, 0, 1)

            self.cached_images.append((x, y))

    def __len__(self):
        return len(self.cached_images)

    def __getitem__(self, idx):

        x, y = self.cached_images[idx]

        return {
            "image": x,
            "label": y,
            "index": idx,
            "stage": self.stage,
        }

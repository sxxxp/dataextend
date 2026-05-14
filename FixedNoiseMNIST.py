import torch

from torch.utils.data import Dataset
from torchvision import datasets, transforms


class FixedNoiseMNIST(Dataset):

    def __init__(
        self,
        stage=0,
        train=True,
        root="./data",
    ):
        self.stage = stage

        self.base_dataset = datasets.MNIST(
            root=root,
            train=train,
            download=True,
            transform=transforms.ToTensor(),
        )

        # 단계별 noise 강도
        self.noise_levels = [
            0.0,  # stage0
            0.1,  # stage1
            0.2,  # stage2
            0.35,  # stage3
            0.5,  # stage4
        ]

        self.noise_std = self.noise_levels[stage]

        # noise를 미리 생성해서 고정
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

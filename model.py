import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader
from tqdm import tqdm


class SimpleCNN(nn.Module):

    def __init__(self):
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Linear(128, 10),
        )

    def forward(self, x):
        return self.net(x)


def train_model(
    model,
    loader,
    epochs=3,
    lr=1e-3,
    device="cuda",
):

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(
        model.parameters(),
        lr=lr,
    )

    model.to(device)

    for epoch in range(epochs):

        model.train()

        total_loss = 0
        total_correct = 0
        total_samples = 0
        progress_bar = tqdm(
            loader,
            desc=f"Epoch {epoch+1}/{epochs}",
        )

        for batch in progress_bar:

            x = batch["image"].to(device)
            y = batch["label"].to(device)

            optimizer.zero_grad()

            pred = model(x)

            loss = criterion(pred, y)

            loss.backward()

            optimizer.step()

            total_loss += loss.item()

            predicted = pred.argmax(dim=1)

            correct = (predicted == y).sum().item()

            total_correct += correct
            total_samples += len(y)

            accuracy = total_correct / total_samples

            # tqdm에 표시
            progress_bar.set_postfix(
                {
                    "loss": f"{loss.item():.4f}",
                    "acc": f"{accuracy:.4f}",
                }
            )

        avg_loss = total_loss / len(loader)

        print(
            f"\nEpoch {epoch+1} Finished | "
            f"Avg Loss: {avg_loss:.4f} | "
            f"Accuracy: {accuracy:.4f}"
        )


def extract_success_samples(
    model,
    dataset,
    batch_size=128,
    device="cuda",
):

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
    )

    model.eval()

    success_indices = []

    correct_count = 0
    total_count = 0

    with torch.no_grad():

        for batch in loader:

            x = batch["image"].to(device)
            y = batch["label"].to(device)
            idx = batch["index"]

            pred = model(x).argmax(dim=1)

            correct = pred == y

            success_indices.extend(idx[correct].tolist())

            correct_count += correct.sum().item()
            total_count += len(y)

    accuracy = correct_count / total_count

    print(f"Success Rate: " f"{correct_count}/{total_count} " f"({accuracy:.4f})")

    return success_indices

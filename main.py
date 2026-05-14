import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import Dataset, DataLoader, ConcatDataset, Subset

from FixedNoiseMNIST import FixedNoiseMNIST
from model import SimpleCNN, train_model, extract_success_samples


def main():

    device = "cuda" if torch.cuda.is_available() else "cpu"

    batch_size = 128

    # -----------------------------------------------------
    # Stage Dataset 생성
    # -----------------------------------------------------

    stage_datasets = {}
    test_datasets = {}
    for stage in range(5):

        stage_datasets[stage] = FixedNoiseMNIST(
            stage=stage,
            train=True,
        )

        test_datasets[stage] = FixedNoiseMNIST(
            stage=stage,
            train=False,
        )

    # =====================================================
    # Model0
    # 전체 stage 학습
    # =====================================================

    print("\n===== Train Model0 =====")

    all_dataset = ConcatDataset(
        [
            stage_datasets[0],
            stage_datasets[1],
            stage_datasets[2],
            stage_datasets[3],
            stage_datasets[4],
        ]
    )

    all_loader = DataLoader(
        all_dataset,
        batch_size=batch_size,
        shuffle=True,
    )

    model0 = SimpleCNN()

    train_model(
        model0,
        all_loader,
        epochs=3,
        device=device,
    )

    # =====================================================
    # Model1
    # stage0 + stage1 학습
    # =====================================================

    print("\n===== Train Model1 =====")

    model1_dataset = ConcatDataset(
        [
            stage_datasets[0],
            stage_datasets[1],
        ]
    )

    model1_loader = DataLoader(
        model1_dataset, batch_size=batch_size, shuffle=True, num_workers=0
    )

    model1 = SimpleCNN()

    train_model(
        model1,
        model1_loader,
        epochs=3,
        device=device,
    )

    current_model = model1

    # =====================================================
    # Stage2 ~ Stage4 반복
    # =====================================================

    accumulated_datasets = [
        stage_datasets[0],
        stage_datasets[1],
    ]

    for stage in range(2, 5):

        print(f"\n===== Evaluate Stage{stage} =====")

        # 현재 stage에서 성공 샘플 추출
        success_indices = extract_success_samples(
            current_model,
            stage_datasets[stage],
            device=device,
        )

        success_dataset = Subset(
            stage_datasets[stage],
            success_indices,
        )

        print(f"Selected Samples: " f"{len(success_dataset)}")

        # 누적
        accumulated_datasets.append(success_dataset)

        # 다음 모델 학습
        next_dataset = ConcatDataset(accumulated_datasets)

        next_loader = DataLoader(
            next_dataset,
            batch_size=batch_size,
            shuffle=True,
        )

        next_model = SimpleCNN()

        print(f"\n===== Train Model{stage} =====")

        train_model(
            next_model,
            next_loader,
            epochs=3,
            device=device,
        )

        current_model = next_model
    print("\nExperiment Finished")

    for i in range(5):
        print("\n====== Evaluate on Stage{} Test Set ======".format(i))
        acc0 = extract_success_samples(model0, test_datasets[i], device=device)
        acc4 = extract_success_samples(current_model, test_datasets[i], device=device)
        print(f"Model0 Success Samples: {len(acc0)}")
        print(f"Model4 Success Samples: {len(acc4)}")


if __name__ == "__main__":
    main()

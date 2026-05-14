# Incremental Noise Curriculum Learning

## 연구 개요

본 연구는 제한된 데이터 환경에서 데이터 증강(Data Augmentation)의 방식이 모델의 일반화 성능과 학습 효율에 어떤 영향을 미치는지를 분석하기 위한 실험이다.

일반적인 데이터 증강 방식은 생성된 모든 데이터를 한 번에 학습시키는 형태를 사용한다. 그러나 이러한 방식은 난이도가 높은 노이즈 데이터까지 동일한 비중으로 포함하게 되어 학습 안정성을 저하시킬 가능성이 있다.

본 연구에서는 노이즈 강도에 따라 데이터를 단계적으로 선별하며 학습하는 **Incremental Noise Curriculum Learning** 방식을 제안하고, 기존의 단순 데이터 확장 방식과 비교한다.

---

# 연구 목표

다음 두 가지 학습 전략을 비교한다.

## 대조군 (Baseline)

전체 노이즈 데이터를 한 번에 학습시키는 방식.

- 모든 단계(Stage)의 노이즈 데이터를 통합
- 모델이 처음부터 고난도 노이즈를 함께 학습
- 일반적인 Naive Augmentation 방식

---

## 실험군 (Proposed)

노이즈 난이도에 따라 데이터를 점진적으로 확장하는 방식.

- 낮은 단계의 노이즈부터 학습
- 현재 모델이 성공적으로 분류한 샘플만 다음 단계 학습에 사용
- Curriculum Learning과 Self-paced Learning 개념을 결합

---

# 연구 가설

> 데이터가 부족한 환경에서 난이도 기반의 점진적 데이터 확장은 단순 무작위 데이터 확장보다 더 빠른 수렴 속도와 낮은 최종 오류율(Error Rate)을 보일 것이다.

---

# Dataset

본 연구에서는 CIFAR-10 데이터셋을 사용한다.

CIFAR-10은 다양한 객체(Class)를 포함하는 RGB 자연 이미지 데이터셋으로,
MNIST보다 복잡한 시각적 특징과 높은 분류 난이도를 가진다.
따라서 노이즈 환경에서의 일반화 성능과 robustness 분석에 더욱 적합하다.

Dataset 정보:

- Train Dataset: 50,000 images
- Test Dataset: 10,000 images
- Image Size: 32×32
- Color Channel: RGB (3 Channels)
- Classes: 10 classes

클래스 구성:

- Airplane
- Automobile
- Bird
- Cat
- Deer
- Dog
- Frog
- Horse
- Ship
- Truck

데이터 증강(Data Augmentation):

학습 데이터에는 일반화 성능 향상을 위해
다음 augmentation을 적용하였다.

- Random Crop
- Random Horizontal Flip
- Normalize

---

# Noise Augmentation

각 이미지에 Gaussian Noise를 추가하여 총 5단계의 데이터셋을 생성한다.

| Stage   | Noise Strength |
| ------- | -------------- |
| Stage 0 | Clean Image    |
| Stage 1 | Weak Noise     |
| Stage 2 | Moderate Noise |
| Stage 3 | Strong Noise   |
| Stage 4 | Extreme Noise  |

노이즈는 각 단계별로 고정된 확률 분포를 사용하여 생성하며, 모든 실험에서 동일한 노이즈 샘플을 유지한다.

---

# 모델 구조

실험에는 CIFAR-10 분류를 위한 CNN 기반 모델을 사용한다.

모델은 다중 Convolution Block과 Batch Normalization,
Dropout을 포함하여 노이즈 환경에서의 학습 안정성과
일반화 성능을 향상시키도록 구성하였다.

구조:

Conv2D(3 → 32)
BatchNorm
ReLU

Conv2D(32 → 32)
BatchNorm
ReLU

MaxPool
Dropout(0.25)

---

Conv2D(32 → 64)
BatchNorm
ReLU

Conv2D(64 → 64)
BatchNorm
ReLU

MaxPool
Dropout(0.25)

---

Conv2D(64 → 128)
BatchNorm
ReLU

MaxPool

---

Flatten

Linear(2048 → 256)
ReLU

Dropout(0.5)

Linear(256 → 10)

입력 이미지 크기:

32 × 32 × 3 (RGB)

출력 클래스:

10 Classes

총 파라미터 수:

약 750,000 parameters

특징:

- Batch Normalization을 통한 학습 안정화
- Dropout을 통한 노이즈 환경 regularization
- Deep Convolution Block 기반 feature extraction
- CIFAR-10 환경에 최적화된 구조

---

# 실험 방법

각 모델은 동일 epoch로 학습되었으나,
데이터셋 크기 차이로 인해
실제 optimization step 수는 상이할 수 있다.

## 1. Baseline Model

전체 노이즈 데이터(Stage0~4)를 한 번에 통합하여 학습한다.

```text
Stage0 + Stage1 + Stage2 + Stage3 + Stage4
```

---

## 2. Incremental Curriculum Model

### Step 1

초기 모델은 낮은 난이도의 데이터만 사용하여 학습한다.

```text
Stage0 + Stage1
```

---

### Step 2

현재 모델을 이용해 다음 단계(Stage2)를 평가한다.

```text
Model1 → Evaluate Stage2
```

정확하게 분류된 샘플만 선택한다.

```text
Correctly Classified Samples Only
```

---

### Step 3

선택된 샘플을 기존 데이터에 누적하여 새로운 모델을 학습한다.

```text
Stage0 + Stage1 + Selected(Stage2)
```

---

### Step 4

동일한 과정을 Stage4까지 반복한다.

```text
Model2 → Evaluate Stage3
Model3 → Evaluate Stage4
```

---

# 평가 지표

다음 지표를 사용하여 모델 성능을 비교한다.

## Accuracy

```text
Accuracy = Correct / Total
```

---

## Error Rate

```text
Error Rate = 1 - Accuracy
```

---

## Success Rate

각 Stage에서 모델이 성공적으로 분류한 샘플 비율.

---

## Noise Robustness

노이즈 강도 증가에 따른 성능 감소율을 측정한다.

---

# 기대 효과

본 연구를 통해 다음과 같은 결과를 기대한다.

- 고난도 노이즈 환경에서의 일반화 성능 향상
- 제한된 데이터 환경에서의 효율적인 데이터 활용
- Curriculum Learning 기반 데이터 증강 전략의 효과 검증
- 노이즈 강도에 따른 학습 가능 경계(learnable boundary) 분석

---

# 향후 확장 가능성

향후 다음과 같은 방향으로 연구를 확장할 수 있다.

- CIFAR-10 데이터셋 실험
- Label Noise 환경 분석
- Pseudo Labeling 결합
- Hard Example Mining 적용
- Self-training 기반 확장
- Dynamic Curriculum Scheduling
- Transformer 기반 모델 비교

---

# 실행 환경

- Python 3.x
- PyTorch
- TorchVision
- tqdm

---

# 실행 예시

```bash
python main.py
```

---

# 참고 개념

- Curriculum Learning
- Self-paced Learning
- Data Augmentation
- Robust Classification
- Noise Robustness
- Incremental Learning

# 실행 결과

선별적으로 데이터를 사용했을 때 일반적인 데이터 패턴을 더 잘 찾아낼 것이라고 예상했지만
오히려 모든 데이터를 한번에 학습한 쪽이 결과가 더 좋았다. 성공한 데이터만을 학습해 버리니
모델이 오히려 특정한 패턴만을 학습해버려 일반화 성능은 더 떨어지는 결과를 낳았다.

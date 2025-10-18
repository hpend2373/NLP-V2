# IKEA MEPNet Adapter - A100 80GB Training Guide

A100 80GB GPU에 최적화된 IKEA MEPNet 학습 가이드입니다.

## A100 최적화 설정

### 하드웨어 사양
- **GPU**: NVIDIA A100 80GB
- **VRAM**: 80GB
- **Tensor Cores**: 3세대
- **FP16/TF32**: 지원

### 최적화된 하이퍼파라미터

```yaml
# 모델 크기 증가
num_stacks: 4         # 2 → 4
num_blocks: 2         # 1 → 2
num_features: 512     # 256 → 512
image_size: [640, 640]  # 512 → 640

# 배치 크기 최대화
batch_size: 32        # 8 → 32 (4배)
num_workers: 8        # 2 → 8

# 학습률 조정
lr: 0.0002           # 배치 크기에 맞춰 증가
warmup_epochs: 5     # 안정적인 학습

# 성능 최적화
mixed_precision: true  # FP16
tf32: true            # A100 TF32 활성화
```

## 빠른 시작

### 1. 환경 설정

```bash
# CUDA 및 PyTorch 확인
nvidia-smi
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"

# 의존성 설치
pip install -r requirements.txt
pip install chamferdist  # GPU 필수
```

### 2. 데이터셋 준비

```bash
# IKEA 데이터셋이 이미 포함되어 있습니다
ls -lh IKEA-Manuals-at-Work/

# 또는 전체 데이터셋 다운로드 (선택사항)
# wget https://ikeamanuals.github.io/dataset/IKEA_Manuals_at_Work.zip
# unzip IKEA_Manuals_at_Work.zip
```

### 3. 학습 시작

```bash
# A100 최적화 설정으로 학습
python scripts/train/train_ikea.py --config configs/train_config_a100.yaml

# W&B 로그인 (선택사항)
wandb login
```

## 성능 예상치

### 학습 속도 (A100 80GB)

| 배치 크기 | 에포크당 시간 | 총 학습 시간 (150 에포크) |
|----------|-------------|----------------------|
| 32       | ~2분        | **5-6시간**           |
| 24       | ~2.5분      | 6-7시간              |
| 16       | ~3분        | 7-8시간              |

### 메모리 사용량

```
모델 파라미터: ~150M
배치 크기 32: ~45GB VRAM
배치 크기 64: ~75GB VRAM (가능!)
```

## 고급 설정

### 배치 크기 64로 실행 (초대형)

메모리가 충분하다면 배치 크기를 더 늘릴 수 있습니다:

```yaml
batch_size: 64
lr: 0.0003  # 배치 크기에 비례하여 증가
gradient_accumulation_steps: 1
```

```bash
python scripts/train/train_ikea.py \
    --config configs/train_config_a100.yaml \
    --batch_size 64 \
    --lr 0.0003
```

### Mixed Precision Training

A100은 FP16과 TF32를 지원합니다:

```python
# train_ikea.py에서 자동으로 활성화됨
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# AMP (Automatic Mixed Precision)
from torch.cuda.amp import autocast, GradScaler
scaler = GradScaler()
```

### Multi-GPU 설정 (선택사항)

여러 A100이 있다면:

```bash
# 4 x A100
python -m torch.distributed.launch \
    --nproc_per_node=4 \
    scripts/train/train_ikea.py \
    --config configs/train_config_a100.yaml \
    --use_multi_gpu true
```

## 모니터링

### Weights & Biases

```bash
# 실시간 모니터링
wandb login
# 학습 중 https://wandb.ai 에서 확인
```

### TensorBoard

```bash
# 로컬 모니터링
tensorboard --logdir experiments/ikea_mepnet_a100/tensorboard --port 6006
```

### NVIDIA 모니터링

```bash
# GPU 사용률 실시간 모니터링
watch -n 1 nvidia-smi

# 상세 모니터링
nvidia-smi dmon -s pucvmet
```

## 학습 중단 및 재개

### 자동 체크포인트

```bash
# 최신 체크포인트에서 재개
python scripts/train/train_ikea.py \
    --config configs/train_config_a100.yaml \
    --resume experiments/ikea_mepnet_a100/checkpoints/checkpoint_latest.pth
```

### 특정 체크포인트에서 재개

```bash
# 특정 에포크에서 재개
python scripts/train/train_ikea.py \
    --config configs/train_config_a100.yaml \
    --resume experiments/ikea_mepnet_a100/checkpoints/checkpoint_epoch_50.pth
```

## 평가 및 테스트

### 모델 평가

```bash
# 최고 성능 모델로 평가
python eval/eval_ikea.py \
    --config configs/train_config_a100.yaml \
    --checkpoint experiments/ikea_mepnet_a100/checkpoints/checkpoint_best.pth \
    --output_dir experiments/ikea_mepnet_a100/evaluation
```

### 추론 (단일 이미지)

```bash
# 단일 이미지에 대한 예측
python scripts/inference/predict_single.py \
    --config configs/train_config_a100.yaml \
    --checkpoint experiments/ikea_mepnet_a100/checkpoints/checkpoint_best.pth \
    --image path/to/manual_image.jpg \
    --output predictions/
```

## 최적화 팁

### 1. 데이터 로딩 속도 향상

```yaml
num_workers: 8        # CPU 코어 수에 맞게 조정
pin_memory: true      # GPU 전송 속도 향상
persistent_workers: true  # 워커 재사용
```

### 2. 메모리 효율성

```python
# Gradient checkpointing (메모리 절약)
model.gradient_checkpointing_enable()

# 배치 크기를 늘리면서 메모리 절약
gradient_accumulation_steps: 2  # 유효 배치 크기 = 32 * 2 = 64
```

### 3. 학습 안정성

```yaml
# Warmup으로 초기 학습 안정화
warmup_epochs: 5
warmup_lr: 0.00001

# Gradient clipping
gradient_clip: 1.0

# Label smoothing
label_smoothing: 0.1
```

## 벤치마크 결과 (예상)

### IKEA 데이터셋 성능

| 메트릭 | 목표 | 예상 결과 (150 에포크) |
|--------|------|---------------------|
| Pose Accuracy | >85% | ~88-92% |
| Chamfer Distance | <0.05 | ~0.03-0.04 |
| Mask IoU | >0.75 | ~0.78-0.82 |
| Plan Accuracy | >90% | ~92-95% |

### 학습 곡선

```
Epoch   Loss    Pose Acc   Chamfer   Time
-----   ----    --------   -------   ----
10      0.850   45.2%      0.125     20m
30      0.420   68.5%      0.072     1h
50      0.285   78.3%      0.051     1.7h
100     0.165   86.7%      0.038     3.3h
150     0.120   90.1%      0.032     5h
```

## 문제 해결

### OOM (Out of Memory)

```bash
# 배치 크기 줄이기
--batch_size 24  # 32 → 24

# 또는 gradient accumulation 사용
--batch_size 16 --gradient_accumulation_steps 2
```

### 학습 속도 느림

```bash
# 데이터 로더 최적화
--num_workers 16  # 더 많은 워커
--prefetch_factor 4  # 미리 로드

# 메시 캐싱 비활성화 (디스크 I/O 느릴 때)
--cache_meshes false
```

### CUDA 오류

```bash
# CUDA 초기화
export CUDA_LAUNCH_BLOCKING=1

# 디버깅 모드
python scripts/train/train_ikea.py \
    --config configs/train_config_a100.yaml \
    --debug
```

## 다음 단계

1. **하이퍼파라미터 튜닝**: W&B Sweeps 사용
2. **앙상블**: 여러 모델 학습 후 앙상블
3. **전이 학습**: LEGO 사전학습 모델 활용
4. **배포**: ONNX/TensorRT로 최적화

## 참고 자료

- [A100 성능 가이드](https://www.nvidia.com/en-us/data-center/a100/)
- [PyTorch AMP 문서](https://pytorch.org/docs/stable/amp.html)
- [MEPNet 논문](https://arxiv.org/abs/2210.05481)

## 라이선스

MIT License

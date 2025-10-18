# IKEA MEPNet Adapter - Colab Training Guide

이 가이드는 Google Colab에서 IKEA MEPNet 모델을 학습하는 방법을 설명합니다.

## 빠른 시작

### 1. Colab 노트북 열기

**Option A: 직접 업로드**
1. [Google Colab](https://colab.research.google.com/) 접속
2. `File` → `Upload notebook` 클릭
3. `notebooks/train_ikea_colab.ipynb` 파일 업로드

**Option B: GitHub에서 직접 열기**
1. [Google Colab](https://colab.research.google.com/) 접속
2. `File` → `Open notebook` → `GitHub` 탭
3. Repository 주소 입력: `hpend2373/NLP`
4. `ikea_mepnet_adapter/notebooks/train_ikea_colab.ipynb` 선택

### 2. GPU 런타임 설정

1. 상단 메뉴: `Runtime` → `Change runtime type`
2. Hardware accelerator: **T4 GPU** 선택 (또는 사용 가능한 GPU)
3. `Save` 클릭

### 3. 노트북 실행

노트북의 셀을 순서대로 실행하세요:

1. **환경 설정**: GPU 확인 및 패키지 설치
2. **데이터셋 다운로드**: IKEA dataset 다운로드 (~15GB)
3. **학습 설정**: 하이퍼파라미터 구성
4. **학습 시작**: 100 에포크 학습 (~3-5시간)
5. **결과 확인**: 모델 평가 및 시각화

## 주요 설정

### 데이터셋

```yaml
data:
  root_dir: "./IKEA-Manuals-at-Work"
  dataset_args:
    load_meshes: true  # 3D 메시 로딩
    furniture_categories: null  # 모든 가구 카테고리
```

### 학습 파라미터

```yaml
batch_size: 8  # GPU 메모리에 따라 조정 가능
epochs: 100
lr: 0.0001
device: "cuda"
```

### GPU 메모리 부족 시

batch_size를 줄이세요:
- **16GB GPU**: batch_size = 8
- **8GB GPU**: batch_size = 4
- **4GB GPU**: batch_size = 2

## 학습 모니터링

### Weights & Biases (권장)

1. [wandb.ai](https://wandb.ai) 가입
2. API key 획득
3. 노트북에서 로그인:
   ```python
   wandb.login(key="YOUR_API_KEY")
   ```
4. 대시보드에서 실시간 모니터링

### TensorBoard

노트북에서 실행:
```python
%load_ext tensorboard
%tensorboard --logdir experiments/ikea_mepnet_colab/tensorboard
```

## 학습 시간 예상

- **T4 GPU**: 약 3-5시간 (100 에포크)
- **V100 GPU**: 약 2-3시간
- **A100 GPU**: 약 1-2시간

## 중단된 학습 재개

Colab 세션이 끊겼을 경우:

```python
!python scripts/train/train_ikea.py \
    --config configs/train_config_colab.yaml \
    --resume experiments/ikea_mepnet_colab/checkpoints/checkpoint_latest.pth
```

## 결과 저장

### Google Drive에 저장 (권장)

```python
from google.colab import drive
drive.mount('/content/drive')

!cp -r experiments/ikea_mepnet_colab /content/drive/MyDrive/
```

### 로컬로 다운로드

```python
!zip -r ikea_mepnet_trained.zip experiments/ikea_mepnet_colab/
from google.colab import files
files.download('ikea_mepnet_trained.zip')
```

## 모델 평가

학습 완료 후:

```python
!python eval/eval_ikea.py \
    --config configs/train_config_colab.yaml \
    --checkpoint experiments/ikea_mepnet_colab/checkpoints/checkpoint_best.pth \
    --output_dir experiments/ikea_mepnet_colab/evaluation
```

## 주요 메트릭

- **Pose Accuracy**: 6D 자세 추정 정확도
- **Chamfer Distance**: 3D 형상 재구성 오차
- **Mask IoU**: 2D 세그멘테이션 정확도
- **Plan Accuracy**: 조립 순서 예측 정확도

## 문제 해결

### 데이터셋 다운로드 실패

Google Drive를 사용하세요:
1. IKEA dataset을 수동으로 다운로드
2. Google Drive에 업로드
3. Colab에서 Drive mount 후 복사

### GPU 메모리 부족

`configs/train_config_colab.yaml` 수정:
```yaml
batch_size: 4  # 8에서 4로 감소
num_features: 128  # 256에서 128로 감소
```

### ChamferDistance 설치 실패

GPU가 없으면 설치 실패할 수 있습니다. 런타임 설정에서 GPU가 활성화되어 있는지 확인하세요.

## 다음 단계

학습 완료 후:
1. 평가 결과 분석
2. 테스트 세트에서 추론 실행
3. 시각화 및 오류 분석
4. 하이퍼파라미터 튜닝

## 참고 자료

- [MEPNet 논문](https://arxiv.org/abs/2210.05481)
- [IKEA Dataset](https://ikeamanuals.github.io/)
- [Weights & Biases Docs](https://docs.wandb.ai/)

## 라이선스

MIT License

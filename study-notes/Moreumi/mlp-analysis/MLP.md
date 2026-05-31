# MLP 모델 분석

## 1. MLP 모델 학습 설정

### 1-1. MLP (Multi-Layer Perceptron) 모델 설명

MLP는 가장 기본적인 인공 신경망 모델로, 입력층(input layer), 은닉층(hidden layer), 출력층(output layer)으로 구성된다. 각 층은 완전 연결(fully connected)되어 있으며, 비선형 활성화 함수(ReLU 등)를 통해 복잡한 패턴을 학습한다.

**MLP의 특성**:
- 1차원 벡터를 입력으로 받음 (이미지/시계열 같은 구조적 정보 활용 불가)
- 비교적 간단한 구조라 학습이 빠름
- 과적합(overfitting) 위험이 큰 모델 → Dropout, Early Stopping 등 규제 필요
- 음향 분류에서는 MFCC, log-mel 같은 압축된 feature와 함께 사용

**한계**:
- 시간 정보를 그대로 학습하기 어려움 (CNN, RNN이 더 유리)
- Feature engineering 품질에 성능이 크게 좌우됨

### 1-2. 본 분석의 전처리 설정 (팀 합의)

| 항목 | 설정 | 비고 |
| --- | --- | --- |
| Sample Rate | 22050 Hz | 학계 표준 |
| Channels | mono | 단일 채널 |
| Bit Depth | 32-float | librosa 기본 |
| Duration | 4초 (repeat padding) | 짧은 신호는 반복 |
| 정규화 | log-mel 추출 후 train set mean/std | feature 단계 표준화 |

### 1-3. Feature 추출

각 wav 파일에서 log-mel spectrogram을 추출한 뒤, 시간축 평균/표준편차로 압축:

```
원본 wav (88,200 샘플)
   ↓ Mel spectrogram (n_mels=128, n_fft=2048, hop_length=512)
log_mel (128 × 173)
   ↓ 시간축 압축
mean (128차원) + std (128차원)
   ↓
256차원 feature 벡터
```

총 8,732개 파일 × 256차원 = `features/mlp_features.csv`

### 1-4. MLP 구조 (Hidden Layer 2개, 팀 합의)

```
입력 (256차원)
   ↓ Dense Linear(256 → 128) + ReLU + Dropout(0.3)   ← Hidden Layer 1
   ↓ Dense Linear(128 → 64) + ReLU + Dropout(0.3)    ← Hidden Layer 2
   ↓ Dense Linear(64 → 10)                            ← 출력 Layer
출력 (10 클래스)
```

- 활성화 함수: ReLU
- 정규화: Dropout 0.3 (각 hidden layer 뒤)
- Hidden Layer 개수 2개 (팀 합의)

### 1-5. 학습 설정

| 항목 | 값 |
| --- | --- |
| Optimizer | Adam |
| Learning rate | 0.001 |
| Batch size | 64 |
| Loss function | CrossEntropyLoss |
| Max Epochs | 100 |
| Early Stopping patience | 10 (Val 사용 시) |
| Random Seed | 42 |
| Device | CPU |

### 1-6. 분석 구성

본 분석은 총 3가지로 구성된다:

| 분석 | Train | Validation | Test | 목적 |
| --- | --- | --- | --- | --- |
| **1차-A** | fold 1~9 | 없음 | fold 10 | 팀 결정 (Val 없음) |
| **1차-B** | fold 1~8 | fold 9 | fold 10 | 팀 결정 (Val=fold9) |
| **2차** | 각 회차 8 folds | 각 회차 1 fold | 각 회차 1 fold | 10-fold CV (일반화 검증) |

---

## 2. Validation 사용 여부 비교 (1차-A vs 1차-B)

### 2-1. FACT — 동일 Test 조건에서 Validation 사용 여부에 따른 차이를 비교한다

1차-A와 1차-B는 Test fold(10)가 동일하다. 차이는 **Validation 사용 여부와 Early Stopping 적용 여부**뿐이다.

| 항목 | 1차-A | 1차-B |
| --- | --- | --- |
| Train | fold 1~9 (전체) | fold 1~8 |
| Validation | 없음 | fold 9 |
| Test | fold 10 | fold 10 |
| Epoch | 100 (고정) | Early stopping (patience=10) |
| 종료 시점 | 100 (강제) | 45 (자동) |

### 2-2. 객관적 결과

#### 전체 정확도

| 분석 | Test Accuracy | Macro F1 | Train 종료 Epoch |
| --- | --- | --- | --- |
| 1차-A (Val 없음) | 0.6284 | 0.6462 | 100 |
| 1차-B (Val=fold9) | 0.6858 | 0.7045 | 45 (early stop) |

**정확도 차이: +5.74%p (1차-B 우세)**

#### 학습 곡선 비교

![학습 곡선 - 1차-A](https://github.com/JIWOO1113/AIX_DEEP_Project/raw/main/study-notes/Moreumi/mlp-analysis/mlp_v1a_no_val_training_curve.png)

1차-A: Train accuracy가 epoch 100까지 0.37 → 0.93으로 계속 상승. Train loss도 1.7 → 0.2까지 계속 감소. **학습이 끝나는 시점을 판단할 기준이 없음.**

![학습 곡선 - 1차-B](https://github.com/JIWOO1113/AIX_DEEP_Project/raw/main/study-notes/Moreumi/mlp-analysis/mlp_v1b_with_val_training_curve.png)

1차-B: Train loss는 계속 감소하지만 **Val loss는 epoch 약 10 부근에서 최저점(약 1.2) 후 상승**. Val accuracy도 약 0.68 부근에서 정체. Early stopping이 epoch 45에서 학습 중단.

#### 클래스별 정확도 비교

| 클래스 | 1차-A | 1차-B | 차이 (%p) |
| --- | --- | --- | --- |
| jackhammer | 0.438 | 0.677 | **+24.0** |
| air_conditioner | 0.560 | 0.780 | **+22.0** |
| street_music | 0.730 | 0.790 | +6.0 |
| siren | 0.518 | 0.542 | +2.4 |
| drilling | 0.490 | 0.510 | +2.0 |
| dog_bark | 0.640 | 0.650 | +1.0 |
| car_horn | 0.788 | 0.788 | 0.0 |
| gun_shot | 0.812 | 0.812 | 0.0 |
| engine_idling | 0.710 | 0.688 | -2.2 |
| children_playing | 0.810 | 0.750 | -6.0 |

- 개선 6개, 변화 없음 2개, 악화 2개
- 최대 개선: jackhammer +24.0%p
- 최대 악화: children_playing -6.0%p

#### 시각화

![1차-A vs 1차-B 비교](https://github.com/JIWOO1113/AIX_DEEP_Project/raw/main/study-notes/Moreumi/mlp-analysis/mlp_v1a_vs_v1b_class_acc.png)

### 2-3. 결과 해석

#### 학습 패턴

- 1차-A의 학습 곡선은 Train 성능이 계속 상승하는 패턴을 보임
- 1차-B의 Val loss는 학습 초기(epoch 약 10)에 최저점을 기록한 뒤 다시 상승
- 1차-A는 종료 시점 판단 기준이 없어 100 epoch까지 진행, 1차-B는 Val 기준 epoch 45에서 자동 종료

#### 클래스별 변동

- jackhammer와 air_conditioner에서 가장 큰 변동 발생 (+22~24%p)
- children_playing은 1차-A에서 0.810, 1차-B에서 0.750으로 -6%p
- car_horn과 gun_shot은 두 분석에서 동일한 정확도

### 2-4. 얻은 정보

- Validation 사용 여부에 따라 동일 Train/Test 조건에서도 전체 정확도가 5.74%p 차이가 발생
- 학습 곡선상 1차-A는 Train loss가 계속 감소하는 반면 1차-B의 Val loss는 학습 중반 이후 다시 증가하는 패턴 관찰
- 클래스별로 변동 방향이 다름 (개선/악화/변화 없음 모두 존재)

---

## 3. 채택한 방식의 분석 (1차-B: Val=fold9, Test=fold10)

### 3-1. FACT — 팀이 채택한 1차-B 결과를 자세히 분석한다

1차-B는 팀이 최종 채택한 분석 방식이다. Train fold 1~8로 학습, Val fold 9로 학습 중 모니터링 (Early Stopping), Test fold 10으로 최종 평가.

### 3-2. 전체 성능

| 지표 | 값 |
| --- | --- |
| Test Accuracy | 0.6858 |
| Test Macro F1 | 0.7045 |
| Test Loss | 1.2745 |
| Best Val Accuracy | 0.6789 |
| 종료 Epoch | 45 (Early Stopping) |

- Test (fold 10) 데이터 수: 837개
- Train 데이터 수: 7,079개 (fold 1~8)
- Validation 데이터 수: 816개 (fold 9)

### 3-3. 클래스별 정확도

| 클래스 | n | 정확도 |
| --- | --- | --- |
| gun_shot | 32 | 0.8125 |
| street_music | 100 | 0.7900 |
| car_horn | 33 | 0.7879 |
| air_conditioner | 100 | 0.7800 |
| children_playing | 100 | 0.7500 |
| engine_idling | 93 | 0.6882 |
| jackhammer | 96 | 0.6771 |
| dog_bark | 100 | 0.6500 |
| siren | 83 | 0.5422 |
| drilling | 100 | 0.5100 |

### 3-4. Confusion Matrix

![1차-B Confusion Matrix](https://github.com/JIWOO1113/AIX_DEEP_Project/raw/main/study-notes/Moreumi/mlp-analysis/mlp_v1b_with_val_confusion.png)

#### 주요 혼동 패턴 (5건 이상)

| True 클래스 | Predicted 클래스 | 횟수 |
| --- | --- | --- |
| siren | children_playing | 20 |
| jackhammer | drilling | 18 |
| drilling | jackhammer | 18 |
| drilling | children_playing | 15 |
| children_playing | dog_bark | 13 |
| dog_bark | children_playing | 13 |
| siren | dog_bark | 12 |
| drilling | siren | 12 |
| air_conditioner | jackhammer | 12 |
| engine_idling | jackhammer | 12 |
| jackhammer | air_conditioner | 9 |
| siren | air_conditioner | 9 |

### 3-5. 결과 해석

#### 클래스 성능 순위

- 정확도 최고: gun_shot (0.81), street_music (0.79), car_horn (0.79), air_conditioner (0.78)
- 정확도 중간: children_playing (0.75), engine_idling (0.69), jackhammer (0.68), dog_bark (0.65)
- 정확도 최저: siren (0.54), drilling (0.51)

#### 주요 혼동 그룹

본 모델이 혼동하는 클래스 그룹:

1. **기계음 그룹**: jackhammer ↔ drilling (서로 18번씩), drilling ↔ siren (12번)
2. **인간 활동 그룹**: children_playing ↔ dog_bark (서로 13번)
3. **음향 패턴 유사 그룹**: siren → children_playing (20번)

### 3-6. 얻은 정보

- 본 모델은 gun_shot, street_music, car_horn 같은 특징적 신호를 잘 분류 (0.78 이상)
- drilling과 siren에서 가장 낮은 정확도 (0.51, 0.54)
- jackhammer와 drilling은 양방향으로 서로 혼동 (각 18번)
- siren은 children_playing으로 잘못 예측되는 경우가 가장 많음 (20번)

---

## 4. 일반화 성능 검증 (2차: 10-fold CV)

### 4-1. FACT — 모든 fold를 한 번씩 test로 사용하여 일반화 성능을 측정한다

2차 분석은 fold 1~10 각각을 한 번씩 test로 사용하는 10-fold Cross Validation이다. Test fold를 fold 10으로 고정한 1차 분석과 달리, **fold 선택에 따른 변동성과 모델의 일반화 성능**을 측정한다.

### 4-2. Fold별 정확도

![Fold별 정확도](https://github.com/JIWOO1113/AIX_DEEP_Project/raw/main/study-notes/Moreumi/mlp-analysis/mlp_v2_fold_accuracy.png)

| Fold | 정확도 | Macro F1 | 종료 Epoch |
| --- | --- | --- | --- |
| 1 | 0.5452 | 0.5773 | 38 |
| 2 | 0.6610 | 0.6582 | 31 |
| 3 | 0.5719 | 0.5957 | 28 |
| 4 | 0.6222 | 0.6206 | 33 |
| 5 | 0.7350 | 0.7400 | 32 |
| 6 | 0.5772 | 0.5965 | 23 |
| 7 | 0.6492 | 0.6654 | 30 |
| 8 | 0.6762 | 0.6975 | 31 |
| 9 | 0.6385 | 0.6632 | 16 |
| 10 | 0.6822 | 0.6997 | 45 |
| **평균** | **0.6359 ± 0.0581** | **0.6514** | - |

- 최고: fold 5 (0.7350)
- 최저: fold 1 (0.5452)
- 최고-최저 차이: 18.98%p

### 4-3. 클래스별 정확도 (전체 8,732개 합계 기준)

| 클래스 | n | 정확도 |
| --- | --- | --- |
| gun_shot | 374 | 0.9251 |
| dog_bark | 1000 | 0.7200 |
| car_horn | 429 | 0.7249 |
| street_music | 1000 | 0.6910 |
| siren | 929 | 0.6846 |
| children_playing | 1000 | 0.6120 |
| drilling | 1000 | 0.6070 |
| engine_idling | 1000 | 0.6060 |
| jackhammer | 1000 | 0.5150 |
| air_conditioner | 1000 | 0.5080 |

### 4-4. Confusion Matrix (10-fold CV 합계)

![2차 Confusion Matrix](https://github.com/JIWOO1113/AIX_DEEP_Project/raw/main/study-notes/Moreumi/mlp-analysis/mlp_v2_confusion.png)

#### 주요 혼동 패턴 (100건 이상)

| True 클래스 | Predicted 클래스 | 횟수 |
| --- | --- | --- |
| jackhammer | drilling | 159 |
| air_conditioner | engine_idling | 153 |
| street_music | children_playing | 134 |
| siren | dog_bark | 132 |
| engine_idling | air_conditioner | 118 |
| jackhammer | air_conditioner | 110 |
| air_conditioner | jackhammer | 102 |
| engine_idling | jackhammer | 101 |

### 4-5. 결과 해석

#### 전체 성능

- 10-fold CV 평균 정확도 0.6359 (표준편차 0.0581)
- Fold별 정확도 범위 0.545 ~ 0.735 (18.98%p 변동)
- Fold 1, 3, 6이 평균보다 낮음 (0.545, 0.572, 0.577)
- Fold 5가 가장 높음 (0.735)

#### 클래스별 강점

- gun_shot이 0.9251로 가장 높음 (전체 374개 중 346개 정답)
- car_horn, dog_bark는 0.72 수준
- air_conditioner, jackhammer는 0.50 수준에서 정체

#### 주요 혼동 그룹

1. **기계음 4종 상호 혼동**: air_conditioner, engine_idling, jackhammer, drilling이 서로 빈번하게 혼동 (각 100건 이상)
2. **인간 활동 혼동**: street_music ↔ children_playing (134건)
3. **소리 패턴 유사**: siren → dog_bark (132건)

### 4-6. 얻은 정보

- Fold별 정확도 표준편차 0.058 (변동성 큼)
- 전체 평균 0.6359는 1차-B의 0.6858보다 5%p 낮음
- gun_shot은 모든 fold에서 일관되게 높은 정확도 (0.9251)
- 기계음 4종(air_conditioner, engine_idling, jackhammer, drilling)이 서로 가장 빈번하게 혼동

---

## 5. 이전 전처리 분석과의 연결

### 5-1. 클래스별 패턴 비교

이전 전처리 분석들(SR, Bit Depth, Channels, Duration, Loudness)에서 식별된 클래스별 특성과 본 MLP 모델 결과의 일치 여부:

| 클래스 | 사전 인지 사항 | MLP 2차 정확도 | 일치 여부 |
| --- | --- | --- | --- |
| gun_shot | 클리핑 62%, 짧은 폭발음, 정확도 높음 | 0.925 (최고) | 일치 |
| drilling | 모든 분석에서 일관 식별, 광대역 신호 | 0.607 (중하위) | 일치 (어려운 클래스) |
| jackhammer | 반복적 타격, 0.5 수준 | 0.515 (하위) | 일치 |
| car_horn | 최저 정확도, 변환에 민감 | 0.725 (중상위) | 불일치 (예상보다 높음) |
| children_playing | 평균 Peak 0.27, 가장 조용 | 0.612 (중위) | 변동 |
| air_conditioner | 평균 Peak 0.30, 조용한 환경음 | 0.508 (최하위) | 변동 |

### 5-2. Fold별 어려움 패턴

이전 분석에서 fold 3, fold 6, fold 8이 일관되게 낮은 정확도를 보였다. 본 MLP 분석에서도 확인:

| Fold | 이전 분석 평균 | MLP 2차 정확도 |
| --- | --- | --- |
| fold 1 | (분석 따라 변동) | **0.545 (최저)** |
| fold 3 | 일관되게 낮음 | 0.572 |
| fold 5 | 일관되게 높음 | **0.735 (최고)** |
| fold 6 | 일관되게 낮음 | 0.577 |
| fold 8 | 일관되게 낮음 | 0.676 |
| fold 10 | Duration 분석 최고 | 0.682 |

본 MLP 분석에서 fold 1이 가장 낮게 나옴 (이전 분석에서는 두드러지지 않았던 fold).

### 5-3. 3가지 분석 종합 비교

![3가지 분석 비교](https://github.com/JIWOO1113/AIX_DEEP_Project/raw/main/study-notes/Moreumi/mlp-analysis/mlp_all_three_class_acc.png)

| 클래스 | 1차-A | 1차-B | 2차 (CV 평균) |
| --- | --- | --- | --- |
| gun_shot | 0.81 | 0.81 | 0.93 |
| car_horn | 0.79 | 0.79 | 0.72 |
| children_playing | 0.81 | 0.75 | 0.61 |
| dog_bark | 0.64 | 0.65 | 0.72 |
| drilling | 0.49 | 0.51 | 0.61 |
| engine_idling | 0.71 | 0.69 | 0.61 |
| jackhammer | 0.44 | 0.68 | 0.52 |
| siren | 0.52 | 0.54 | 0.68 |
| street_music | 0.73 | 0.79 | 0.69 |
| air_conditioner | 0.56 | 0.78 | 0.51 |

---

## 6. 인지하고 가야 할 사항

### 6-1. Validation 사용 여부에 따른 학습 패턴 차이

본 MLP 모델은 Validation 사용 여부에 따라 학습 종료 시점과 클래스별 정확도가 달라진다.

- 1차-A (Val 없음, epoch 100 고정): Train accuracy 0.93까지 도달, Test accuracy 0.628
- 1차-B (Val 사용, Early Stopping): epoch 45에서 자동 종료, Test accuracy 0.686
- 동일 Train/Test 조건에서 전체 정확도 5.74%p 차이

### 6-2. 1차-B와 2차의 정확도 차이

1차-B(0.686)와 2차(0.636)의 정확도 차이 5%p는 **Test fold 선택의 영향**일 수 있다.

- 1차-B의 Test는 fold 10으로 고정
- 2차의 Test fold 10 단일 결과는 0.682 (1차-B와 유사)
- 2차의 평균 0.636은 모든 fold(특히 fold 1=0.545)의 평균
- fold 10이 상대적으로 정확도가 높은 fold라는 점이 1차-B 결과에 반영됨

### 6-3. 클래스별 성능 격차

| 그룹 | 클래스 | 2차 정확도 |
| --- | --- | --- |
| 상위 | gun_shot, dog_bark, car_horn | 0.72 ~ 0.93 |
| 중위 | street_music, siren, children_playing | 0.61 ~ 0.69 |
| 하위 | drilling, engine_idling, jackhammer, air_conditioner | 0.51 ~ 0.61 |

상위-하위 그룹 간 정확도 격차 약 40%p.

### 6-4. 기계음 4종의 상호 혼동

air_conditioner, engine_idling, jackhammer, drilling이 서로 가장 빈번하게 혼동된다. 각 쌍에서 100건 이상의 혼동 발생.

- jackhammer ↔ drilling: 각 159건
- air_conditioner ↔ engine_idling: 각 118~153건
- jackhammer ↔ air_conditioner: 각 102~110건

이 4개 클래스가 본 MLP 모델에서 분류 난이도가 가장 높은 그룹.

### 6-5. Fold 1의 낮은 정확도

본 MLP 2차 분석에서 fold 1이 0.5452로 가장 낮음. 이전 전처리 분석에서는 fold 1이 두드러지지 않았으나 본 모델에서는 가장 어려운 fold로 나타남.

### 6-6. MLP의 256차원 feature 한계

본 분석은 log-mel spectrogram(128 × 173)을 시간축 평균/표준편차로 압축한 256차원 feature를 사용. 시간축 정보가 압축 과정에서 손실됨. 시간 패턴이 중요한 클래스(예: 반복적인 jackhammer 타격, drilling의 진동 패턴)에서 정확도가 낮은 이유 중 하나일 수 있음.

---

## 7. 분석 환경

| 항목 | 값 |
| --- | --- |
| Python | 3.10.20 |
| PyTorch | 2.5.1 (CUDA: False, CPU 사용) |
| 주요 라이브러리 | librosa 0.11.0, numpy, pandas, scikit-learn, matplotlib |
| 분석 스크립트 | code/extract_features.py, code/mlp_model.py, code/mlp_comparison_plots.py |
| Feature 데이터 | features/mlp_features.csv (8,732행 × 260열) |
| 결과 데이터 | code/output/mlp_v1a_no_val.csv, mlp_v1a_no_val_class_acc.csv, mlp_v1b_with_val.csv, mlp_v1b_with_val_class_acc.csv, mlp_v2_cv.csv, mlp_v2_class_acc.csv, mlp_comparison_summary.csv |
| 시각화 | mlp_v1a_no_val_confusion.png, mlp_v1a_no_val_training_curve.png, mlp_v1b_with_val_confusion.png, mlp_v1b_with_val_training_curve.png, mlp_v2_confusion.png, mlp_v2_fold_accuracy.png, mlp_v1a_vs_v1b_class_acc.png, mlp_all_three_class_acc.png |
| 전처리 파라미터 | SR=22050, mono, 32-float, 4초 (repeat padding), n_mels=128, n_fft=2048, hop_length=512 |
| MLP 구조 | Hidden Layer 2개 (256 → 128 → 64 → 10), Dropout 0.3 |
| 학습 설정 | Adam (lr=0.001), CrossEntropyLoss, batch=64, max epoch=100, early stopping patience=10 |
| 재현 가능성 | random_state=42 |

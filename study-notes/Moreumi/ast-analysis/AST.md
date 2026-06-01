# AST 모델 분석

## 1. AST 모델 학습 설정

### 1-1. AST (Audio Spectrogram Transformer) 모델 설명

AST는 오디오의 log-mel spectrogram을 이미지처럼 패치(patch) 단위로 분할하여 Transformer로 처리하는 음향 분류 모델이다. CNN을 사용하지 않고 순수 self-attention만으로 spectrogram의 시간-주파수 구조를 학습한다. 본 분석에서 사용한 모델은 AudioSet(527 클래스, 약 200만 개 오디오)으로 사전학습된 `MIT/ast-finetuned-audioset-10-10-0.4593`이다.

**AST의 특성**:
- spectrogram을 16×16 패치로 분할하여 시간축 정보를 보존 (MLP의 시간축 압축과 대비)
- Self-attention으로 패치 간 장거리 의존성 학습
- 대규모 사전학습(AudioSet)으로 일반적 음향 표현(representation) 보유
- 입력은 16,000Hz mono audio를 1024 frame × 128 mel bins spectrogram으로 변환

**MLP와의 핵심 차이**:
- MLP는 log-mel을 시간축 평균/표준편차로 압축한 256차원 벡터를 사용 → 시간 정보 손실
- AST는 1024 frame을 그대로 입력 → 반복 타격, 진동 패턴 등 시간 구조 학습 가능

### 1-2. 본 분석의 입력 설정

| 항목 | 설정 | 비고 |
| --- | --- | --- |
| Sample Rate | 16,000 Hz | AST 모델 요구 사양 (MLP는 22,050Hz) |
| Channels | mono | 단일 채널 |
| 입력 형태 | 1024 frame × 128 mel bins | Feature Extractor 자동 변환 |
| 길이 처리 | 1024 frame로 자동 padding | 짧은 클립(≤4초)은 zero-padding |
| 정규화 | ASTFeatureExtractor 자동 처리 | AudioSet 통계 기반 정규화 |

UrbanSound8K 원본은 다양한 sample rate(최대 44.1kHz)를 가지므로, librosa로 16,000Hz mono로 resample한 뒤 ASTFeatureExtractor에 입력했다. 검증 결과 입력 shape는 `(1024, 128)`, 값 범위는 약 -1.28 ~ 1.10으로 정규화가 정상 작동함을 확인했다.

### 1-3. 분석 구성 (2가지 fine-tuning 방식)

본 분석은 사전학습된 AST를 UrbanSound8K에 적용하는 2가지 방식을 비교한다.

| 방식 | 학습 범위 | 본체(transformer) | classifier head | 목적 |
| --- | --- | --- | --- | --- |
| **Feature Extraction (FE)** | head만 | 동결(freeze) | 학습 | AudioSet pretrained representation 평가 |
| **Full Fine-tuning (Full)** | 전체 | 학습 | 학습 | UrbanSound8K 특화 학습 |

**Zero-shot 제외 사유**: AudioSet의 527개 라벨과 UrbanSound8K의 10개 라벨이 1:1로 대응되지 않아(air_conditioner, jackhammer 등은 대응 라벨이 모호함), 라벨 매핑에 주관이 개입된다. 결과 신뢰성과 MLP 분석(zero-shot 미포함)과의 일관성을 위해 순수 zero-shot은 측정에서 제외했다.

### 1-4. classifier head 교체

AudioSet 사전학습 모델의 마지막 classifier(527 클래스)를 UrbanSound8K용 10 클래스로 교체했다.

```
ASTForAudioClassification.from_pretrained(
    "MIT/ast-finetuned-audioset-10-10-0.4593",
    num_labels=10,
    ignore_mismatched_sizes=True   # 527 → 10 클래스 변경 허용
)
```

모델 로드 시 `classifier.dense`(527→10)만 랜덤 초기화되고, transformer 본체(201개 파라미터)는 AudioSet 가중치를 그대로 유지했다.

### 1-5. Feature Extraction 구현 (embedding 캐싱)

FE는 본체가 학습 중 변하지 않으므로, 전체 데이터를 본체에 1회만 통과시켜 768차원 embedding을 추출한 뒤 저장하고, 그 위에 head만 반복 학습했다.

```
원본 wav → 16kHz resample → ASTFeatureExtractor → (1024, 128)
   ↓ AST 본체 1회 통과 (동결)
pooler_output (768차원 embedding)  ← 추출 후 저장
   ↓ head 학습 (50 epoch)
LayerNorm(768) + Linear(768 → 10)
   ↓
출력 (10 클래스)
```

| split | embedding shape |
| --- | --- |
| Train (fold 1~8) | (7079, 768) |
| Val (fold 9) | (816, 768) |
| Test (fold 10) | (837, 768) |

embedding 추출은 본체 1회 통과로 약 10~15분 소요, 이후 head 학습은 epoch당 1초 미만으로 50 epoch 전체가 약 1~2분이다.

### 1-6. 학습 설정

| 항목 | Feature Extraction | Full Fine-tuning |
| --- | --- | --- |
| 학습 대상 | head (LayerNorm + Linear) | 전체 모델 |
| Optimizer | AdamW | AdamW |
| Learning rate | 1e-3 | 1e-5 |
| Batch size | 전체 batch (7079) | 16 |
| Max Epochs | 50 | 5 |
| Early Stopping | best val 저장 | patience=2 |
| Loss function | CrossEntropyLoss | CrossEntropyLoss |
| Random Seed | 42 | 42 |
| Device | T4 GPU (CUDA) | T4 GPU (CUDA) |

### 1-7. 평가 방식 (MLP 1차-B와 동일)

| 항목 | 설정 |
| --- | --- |
| Train | fold 1~8 (7,079개) |
| Validation | fold 9 (816개) |
| Test | fold 10 (837개) |

fold 분할은 MLP 1차-B와 완전히 동일하다(7079/816/837). 동일한 Test fold(10)를 사용하므로 MLP와 직접 비교가 가능하다.

### 1-8. 평가 지표 (팀 합의 12개 항목)

본 프로젝트는 모든 모델에 대해 동일한 12개 평가 항목을 사용한다. UrbanSound8K는 클래스 불균형(gun_shot 374, car_horn 429 vs 나머지 ~1000)이 있으므로, accuracy 단일 지표가 아닌 balanced accuracy와 macro 평균 지표를 함께 본다.

| 통계값 | 설명 |
| --- | --- |
| accuracy | 전체 sample 중 모델이 정답을 맞힌 비율 |
| balanced accuracy | class별 recall을 평균낸 값. class imbalance가 있을 때 accuracy보다 공정하게 볼 수 있음 |
| macro precision | class별 precision을 단순 평균한 값. 모든 class를 같은 비중으로 반영 |
| macro recall | class별 recall을 단순 평균한 값. 작은 class의 탐지 성능까지 반영 |
| macro F1 | class별 F1을 단순 평균한 값. class imbalance가 있는 모델 비교에서 핵심 지표로 사용 |
| weighted precision | class별 precision을 sample 수에 따라 가중 평균한 값 |
| weighted recall | class별 recall을 sample 수에 따라 가중 평균한 값 |
| weighted F1 | class별 F1을 sample 수에 따라 가중 평균한 값. 실제 데이터 분포 기준 성능을 반영 |
| class별 precision | 특정 class라고 예측한 sample 중 실제로 그 class인 비율. 오탐이 많은지 확인 |
| class별 recall | 실제 특정 class sample 중 모델이 제대로 맞힌 비율. 미탐이 많은지 확인 |
| class별 F1 | class별 precision과 recall의 조화평균. 각 class의 종합 성능 확인 |
| confusion matrix | 실제 class와 예측 class의 대응표. 어떤 class끼리 헷갈리는지 확인 |

---

## 2. Feature Extraction 분석

### 2-1. FACT — AST 본체를 동결하고 head만 학습한 결과를 측정한다

AudioSet pretrained 본체를 동결한 채 768차원 embedding을 추출하고, classifier head(LayerNorm + Linear 768→10)만 50 epoch 학습했다. Best Val Accuracy 기준으로 best model을 선택하여 Test(fold 10)를 평가했다.

### 2-2. 전체 성능

| 지표 | 값 |
| --- | --- |
| accuracy | 0.8829 |
| balanced accuracy | 0.8861 |
| macro precision | 0.9006 |
| macro recall | 0.8861 |
| macro F1 | 0.8903 |
| weighted precision | 0.8881 |
| weighted recall | 0.8829 |
| weighted F1 | 0.8824 |
| Best Val Accuracy | 0.8750 |

학습 과정에서 Val Accuracy는 epoch 1의 0.2341에서 시작하여 epoch 15에서 0.8431, epoch 50에서 0.8750으로 점진 상승했다.

### 2-3. 클래스별 성능 (F1 정확도순 정렬)

| 클래스 | precision | recall | F1-score | n |
| --- | --- | --- | --- | --- |
| jackhammer | 0.9895 | 0.9792 | 0.9843 | 96 |
| gun_shot | 1.0000 | 0.9688 | 0.9841 | 32 |
| engine_idling | 0.9158 | 0.9355 | 0.9255 | 93 |
| drilling | 0.9468 | 0.8900 | 0.9175 | 100 |
| car_horn | 0.9355 | 0.8788 | 0.9062 | 33 |
| street_music | 0.8911 | 0.9000 | 0.8955 | 100 |
| children_playing | 0.8099 | 0.9800 | 0.8869 | 100 |
| dog_bark | 0.8286 | 0.8700 | 0.8488 | 100 |
| siren | 0.9508 | 0.6988 | 0.8056 | 83 |
| air_conditioner | 0.7379 | 0.7600 | 0.7488 | 100 |

### 2-4. Confusion Matrix

![FE Confusion Matrix](https://github.com/JIWOO1113/AIX_DEEP_Project/raw/main/study-notes/Moreumi/ast-analysis/ast_feature_ext_confusion.png)

#### 주요 혼동 패턴 (5건 이상)

| True 클래스 | Predicted 클래스 | 횟수 |
| --- | --- | --- |
| siren | dog_bark | 14 |
| air_conditioner | children_playing | 12 |
| siren | air_conditioner | 7 |
| drilling | street_music | 7 |
| dog_bark | air_conditioner | 6 |
| engine_idling | air_conditioner | 6 |
| street_music | children_playing | 5 |

### 2-5. 결과 해석

#### 클래스 성능 순위
- 정확도 최고: gun_shot(0.9840), jackhammer(0.9840), engine_idling(0.9260)
- 정확도 중간: drilling, car_horn, street_music, children_playing (0.88 ~ 0.92)
- 정확도 최저: siren(0.8060), air_conditioner(0.7490)

#### siren의 낮은 recall
siren은 precision 0.9510으로 높으나 recall 0.6990으로 낮다. 즉 siren으로 예측한 것은 대부분 정확하나, 실제 siren의 약 30%를 다른 클래스(주로 dog_bark 14건)로 놓쳤다.

#### air_conditioner의 분산 오분류
air_conditioner는 children_playing(12), engine_idling(6) 등으로 분산 오분류되어 가장 낮은 F1(0.7490)을 기록했다.

### 2-6. 얻은 정보
- AST 본체를 동결한 FE만으로 Test Accuracy 0.8829, Macro F1 0.8903 달성
- gun_shot, jackhammer는 0.98 이상으로 매우 높은 정확도
- siren은 recall이 낮으며(0.6990), 주로 dog_bark로 오분류됨
- air_conditioner가 가장 어려운 클래스(F1 0.7490)

---

## 3. Full Fine-tuning 분석

### 3-1. FACT — AST 본체 전체를 학습한 결과를 측정한다

AudioSet pretrained 모델을 fresh하게 재로드한 뒤(FE의 head 학습 영향 제거), 본체와 head 전체를 lr=1e-5로 최대 5 epoch 학습했다. Validation 기준 early stopping(patience=2)을 적용했다.

### 3-2. 전체 성능

| 지표 | 값 |
| --- | --- |
| accuracy | 0.8901 |
| balanced accuracy | 0.8999 |
| macro precision | 0.9102 |
| macro recall | 0.8999 |
| macro F1 | 0.8993 |
| weighted precision | 0.9001 |
| weighted recall | 0.8901 |
| weighted F1 | 0.8890 |
| Best Val Accuracy | 0.9007 (epoch 3) |
| 종료 Epoch | 5 (Early Stopping, patience=2) |

### 3-3. 학습 곡선

![Full FT 학습 곡선](https://github.com/JIWOO1113/AIX_DEEP_Project/raw/main/study-notes/Moreumi/ast-analysis/ast_training_curve.png)

| Epoch | Train Loss | Val Loss | Val Accuracy |
| --- | --- | --- | --- |
| 1 | 0.2790 | 0.4694 | 0.8995 |
| 2 | 0.0448 | 0.5285 | 0.8860 |
| 3 | 0.0155 | 0.5870 | 0.9007 |
| 4 | 0.0080 | 0.5987 | 0.8958 |
| 5 | 0.0077 | 0.6685 | 0.8995 |

Train Loss는 0.2790 → 0.0077로 급감(거의 0)한 반면, Val Loss는 0.4694 → 0.6685로 지속 상승했다. Val Accuracy는 0.886 ~ 0.901 범위에서 진동하며 추세적 상승을 보이지 않았다.

### 3-4. 클래스별 성능 (F1 정확도순 정렬)

| 클래스 | precision | recall | F1-score | n |
| --- | --- | --- | --- | --- |
| gun_shot | 1.0000 | 1.0000 | 1.0000 | 32 |
| jackhammer | 0.9600 | 1.0000 | 0.9796 | 96 |
| car_horn | 0.9143 | 0.9697 | 0.9412 | 33 |
| engine_idling | 0.9556 | 0.9247 | 0.9399 | 93 |
| street_music | 0.8857 | 0.9300 | 0.9073 | 100 |
| drilling | 0.9767 | 0.8400 | 0.9032 | 100 |
| children_playing | 0.8151 | 0.9700 | 0.8858 | 100 |
| dog_bark | 0.7500 | 0.9300 | 0.8304 | 100 |
| siren | 1.0000 | 0.6747 | 0.8058 | 83 |
| air_conditioner | 0.8444 | 0.7600 | 0.8000 | 100 |

### 3-5. Confusion Matrix

![Full FT Confusion Matrix](https://github.com/JIWOO1113/AIX_DEEP_Project/raw/main/study-notes/Moreumi/ast-analysis/ast_full_ft_confusion.png)

#### 주요 혼동 패턴 (5건 이상)

| True 클래스 | Predicted 클래스 | 횟수 |
| --- | --- | --- |
| siren | dog_bark | 22 |
| drilling | street_music | 9 |
| air_conditioner | children_playing | 9 |
| air_conditioner | dog_bark | 7 |
| engine_idling | air_conditioner | 7 |
| street_music | children_playing | 6 |

### 3-6. 결과 해석

#### 과적합 발생
Train Loss가 거의 0(0.0077)에 도달하는 동안 Val Loss는 지속 상승(0.4694 → 0.6685)했다. 이는 7,079개 학습 데이터로 약 86M 파라미터의 전체 모델을 학습할 때 발생하는 과적합 패턴이다. Val Accuracy는 epoch 1에서 이미 0.8995에 도달했고 이후 추세 상승이 없었다.

#### siren 오분류 증가
siren → dog_bark 오분류가 FE의 14건에서 Full FT의 22건으로 증가했다. 본체 전체를 학습했음에도 siren의 recall은 FE(0.6990)보다 낮은 0.6750을 기록했다.

#### air_conditioner 개선
air_conditioner는 F1이 FE의 0.7490에서 0.8000으로 +0.0510 개선되어, Full FT에서 가장 큰 폭으로 향상된 클래스다.

### 3-7. 얻은 정보
- Full FT의 Test Accuracy 0.8901은 FE(0.8829)보다 +0.0072 높음
- Val Loss 지속 상승으로 과적합 발생, Val Accuracy는 epoch 1에 이미 천장 도달
- siren → dog_bark 오분류는 본체 학습 후에도 줄지 않고 오히려 증가(14 → 22)
- air_conditioner만 FE 대비 뚜렷한 개선(+0.0510)

---

## 4. Feature Extraction vs Full Fine-tuning 비교

### 4-1. FACT — 동일 Test 조건에서 두 fine-tuning 방식의 차이를 비교한다

FE와 Full FT는 fold 분할(train 1~8, val 9, test 10)과 평가 방식이 동일하다. 차이는 본체(transformer)의 학습 여부뿐이다.

### 4-2. 전체 성능 비교

| 지표 | Feature Extraction | Full Fine-tuning | 차이 |
| --- | --- | --- | --- |
| accuracy | 0.8829 | 0.8901 | +0.0072 |
| balanced accuracy | 0.8861 | 0.8999 | +0.0138 |
| macro precision | 0.9006 | 0.9102 | +0.0096 |
| macro recall | 0.8861 | 0.8999 | +0.0138 |
| macro F1 | 0.8903 | 0.8993 | +0.0090 |
| weighted precision | 0.8881 | 0.9001 | +0.0120 |
| weighted recall | 0.8829 | 0.8901 | +0.0072 |
| weighted F1 | 0.8824 | 0.8890 | +0.0066 |
| Best Val Accuracy | 0.8750 | 0.9007 | +0.0257 |
| 학습 시간 (T4) | 추출 10~15분 + head 1~2분 | 약 40~60분 | Full이 약 3배 |

### 4-3. 클래스별 F1 비교 (정확도순 정렬, FE 기준)

| 클래스 | FE F1 | Full FT F1 | 차이 |
| --- | --- | --- | --- |
| gun_shot | 0.9840 | 1.0000 | +0.0160 |
| jackhammer | 0.9840 | 0.9800 | -0.0040 |
| engine_idling | 0.9260 | 0.9400 | +0.0140 |
| drilling | 0.9180 | 0.9030 | -0.0150 |
| car_horn | 0.9060 | 0.9410 | +0.0350 |
| street_music | 0.8960 | 0.9070 | +0.0110 |
| children_playing | 0.8870 | 0.8860 | -0.0010 |
| dog_bark | 0.8490 | 0.8300 | -0.0190 |
| siren | 0.8060 | 0.8060 | 0.0000 |
| air_conditioner | 0.7490 | 0.8000 | +0.0510 |

### 4-4. 결과 해석

#### 전체 성능 차이
Full FT가 FE보다 Test Accuracy +0.0072, Macro F1 +0.0090로 소폭 높으나, 학습 시간은 약 3배 더 소요됐다.

#### 클래스별 변동
- 개선 5개, 악화 4개, 변화 없음 1개로 방향이 혼재
- 최대 개선: air_conditioner +0.0510
- 최대 악화: dog_bark -0.0190
- 대부분 클래스의 변동폭이 0.02 이내로 미미함

#### 과적합 대비 이득
Full FT는 약 3배의 학습 시간과 과적합(Val Loss 상승)을 감수하고도 FE 대비 1%p 이내의 개선만 보였다.

### 4-5. 얻은 정보
- Full FT는 FE 대비 Test Accuracy +0.0072로 미미한 차이
- Full FT는 학습 시간 약 3배, 과적합 발생
- 본 데이터 규모(7,079개)에서는 본체를 동결한 FE가 비용 대비 효율적
- air_conditioner만 본체 학습으로 뚜렷한 이득(+0.0510)

---

## 5. MLP 모델과의 비교

### 5-1. FACT — 동일 데이터·동일 fold 분할에서 MLP와 AST를 비교한다

MLP 1차-B와 AST는 fold 분할(train 1~8, val 9, test 10)과 Test fold(10)가 동일하다. 입력 feature와 모델 구조만 다르다(MLP: 256차원 압축 feature / AST: 1024 frame spectrogram).

### 5-2. 전체 성능 비교

| 지표 | MLP 1차-B | AST FE | AST Full FT |
| --- | --- | --- | --- |
| accuracy | 0.6858 | 0.8829 | 0.8901 |
| balanced accuracy | 0.6988 | 0.8861 | 0.8999 |
| macro precision | 0.7223 | 0.9006 | 0.9102 |
| macro recall | 0.6988 | 0.8861 | 0.8999 |
| macro F1 | 0.7045 | 0.8903 | 0.8993 |
| weighted precision | 0.7004 | 0.8881 | 0.9001 |
| weighted recall | 0.6858 | 0.8829 | 0.8901 |
| weighted F1 | 0.6868 | 0.8824 | 0.8890 |

AST FE는 MLP 1차-B 대비 accuracy +0.1971, macro F1 +0.1858 개선되었다. 모든 12개 지표에서 AST가 MLP를 큰 폭으로 상회한다.

### 5-3. 클래스별 정확도(recall) 비교

![MLP vs AST 비교](https://github.com/JIWOO1113/AIX_DEEP_Project/raw/main/study-notes/Moreumi/ast-analysis/ast_vs_mlp_comparison.png)

| 클래스 | MLP 2차 (CV) | AST FE | AST Full FT |
| --- | --- | --- | --- |
| air_conditioner | 0.5080 | 0.7600 | 0.7600 |
| car_horn | 0.7249 | 0.8790 | 0.9700 |
| children_playing | 0.6120 | 0.9800 | 0.9700 |
| dog_bark | 0.7200 | 0.8700 | 0.9300 |
| drilling | 0.6070 | 0.8900 | 0.8400 |
| engine_idling | 0.6060 | 0.9350 | 0.9250 |
| gun_shot | 0.9251 | 0.9690 | 1.0000 |
| jackhammer | 0.5150 | 0.9790 | 1.0000 |
| siren | 0.6846 | 0.6990 | 0.6750 |
| street_music | 0.6910 | 0.9000 | 0.9300 |

(MLP는 2차 10-fold CV의 클래스별 정확도, AST는 Test fold 10 기준 recall)

### 5-4. 결과 해석

#### 기계음 4종의 혼동 해소 (핵심 발견)
MLP 분석에서 가장 큰 약점은 기계음 4종(air_conditioner, engine_idling, jackhammer, drilling)의 상호 혼동이었다. 특히 jackhammer ↔ drilling은 MLP 2차에서 양방향으로 각 159건 혼동되었다.

AST에서는 이 혼동이 크게 해소되었다.

| 클래스 | MLP 2차 정확도 | AST FE | AST Full FT |
| --- | --- | --- | --- |
| jackhammer | 0.5150 | 0.9790 | 1.0000 |
| engine_idling | 0.6060 | 0.9350 | 0.9250 |
| drilling | 0.6070 | 0.8900 | 0.8400 |
| air_conditioner | 0.5080 | 0.7600 | 0.7600 |

AST Full FT의 confusion matrix에서 jackhammer는 96건 전부 정답(drilling 오분류 0건), drilling도 jackhammer로의 오분류가 0건이다. MLP에서 양방향 159건씩 혼동되던 두 클래스가 완전히 분리되었다.

MLP 분석은 이 혼동의 원인으로 시간축 정보 손실(log-mel을 시간축 평균/표준편차로 압축)을 지목했다. AST는 1024 frame을 그대로 입력하여 반복 타격 주기, 진동 패턴 등 시간 구조를 보존하므로, 시간 정보가 중요한 기계음 분류에서 큰 폭의 개선을 보였다. 이는 시간 정보를 보존하는 모델 구조의 효과를 보여주는 결과다.

#### siren의 일관된 약점
siren은 MLP(0.6846), AST FE(0.6990), AST Full FT(0.6750) 세 모델 모두에서 0.68 ~ 0.70 수준에 머물렀다. MLP와 AST 모두에서 siren → dog_bark 오분류가 주요 패턴으로 나타났다(MLP 2차 132건, AST FE 14건, AST Full FT 22건). 모델 구조와 학습 범위를 바꿔도 동일한 혼동이 반복된다.

#### gun_shot의 일관된 강점
gun_shot은 MLP(0.9251)에서도 이미 높았고 AST에서 0.97 ~ 1.00으로 더 향상되었다. 짧고 특징적인 폭발음은 모든 모델에서 잘 분류된다.

### 5-5. 얻은 정보
- AST FE는 MLP 1차-B 대비 Test Accuracy +0.1971 개선
- MLP의 최대 약점이던 jackhammer ↔ drilling 상호 혼동이 AST에서 완전히 분리됨
- siren은 세 모델 모두에서 dog_bark로 오분류되는 약점이 반복됨
- 기계음 4종의 정확도 개선이 시간 정보 보존 구조의 효과를 보여줌

---

## 6. 인지하고 가야 할 사항

### 6-1. Zero-shot을 분석에서 제외한 이유

AudioSet의 527개 라벨과 UrbanSound8K의 10개 라벨이 1:1로 대응되지 않는다. dog_bark, siren, gun_shot 등은 AudioSet에 대응 라벨이 있으나 air_conditioner, jackhammer 등은 대응 라벨이 모호하다. 라벨 매핑에 주관이 개입되면 결과 신뢰성이 떨어지고, MLP 분석(zero-shot 미포함)과의 일관성도 깨진다. 이에 순수 zero-shot은 측정에서 제외하고 FE와 Full FT만 비교했다.

### 6-2. Feature Extraction으로 충분함

Full FT는 FE 대비 Test Accuracy +0.0072의 미미한 개선만 보였고, 학습 시간은 약 3배, 과적합(Val Loss 0.4694 → 0.6685 상승)을 동반했다. 본 데이터 규모(7,079개)에서 약 86M 파라미터의 전체 모델을 학습하는 것은 비용 대비 효율이 낮으며, 본체를 동결한 FE가 실용적이다.

### 6-3. Full Fine-tuning의 과적합

Full FT에서 Train Loss는 epoch 5에 0.0077로 거의 0에 도달했으나 Val Loss는 지속 상승했다. Val Accuracy는 epoch 1에서 이미 0.8995로 천장에 도달하여 추가 학습의 이득이 없었다. 데이터 양 대비 모델 파라미터가 많을 때 발생하는 전형적 과적합이며, early stopping(patience=2)으로 epoch 5에서 학습을 종료했다.

### 6-4. siren → dog_bark 혼동은 데이터 자체의 모호성

siren은 MLP, AST FE, AST Full FT 세 모델 모두에서 dog_bark로 오분류되는 패턴이 반복되었다. 모델 구조(MLP → Transformer)와 학습 범위(head만 → 전체)를 모두 바꿔도 동일한 혼동이 나타나며, Full FT에서는 오히려 오분류가 증가(14 → 22건)했다. 이는 특정 모델의 한계가 아니라 사이렌의 변조음과 개 짖음의 음높이 변화가 음향적으로 유사한, 데이터 자체의 모호성에서 비롯된 것으로 판단된다.

### 6-5. 기계음 4종 혼동 해소가 보여주는 시간 정보의 가치

MLP의 최대 약점이던 기계음 4종(특히 jackhammer ↔ drilling 양방향 159건 혼동)이 AST에서 거의 완전히 해소되었다. AST Full FT에서 jackhammer는 100% 정답률을 기록했다. MLP가 시간축을 평균/표준편차로 압축하여 반복 패턴 정보를 잃은 반면, AST는 1024 frame을 그대로 입력하여 시간 구조를 학습한다. 이 차이가 기계음 분류 성능의 차이로 직결되었으며, 음향 분류에서 시간 정보를 보존하는 모델 구조의 가치를 정량적으로 보여준다.

### 6-6. air_conditioner의 잔존 약점

air_conditioner는 AST에서도 가장 낮은 정확도(0.7600)에 머물렀다. children_playing, dog_bark 등으로 분산 오분류된다. 뚜렷한 시간/주파수 특징이 없는 지속적 배경 소음이라 본질적으로 분류가 어려운 클래스다. 단, Full FT에서 F1이 +0.0510 개선되어, 본체 학습이 도움이 되는 유일한 클래스였다.

---

## 7. 분석 환경

| 항목 | 값 |
| --- | --- |
| 실행 환경 | Google Colab |
| GPU | Tesla T4 (15GB) |
| Python | 3.12 (Colab) |
| 주요 라이브러리 | transformers, datasets, librosa, torch, scikit-learn, matplotlib, seaborn |
| 모델 | MIT/ast-finetuned-audioset-10-10-0.4593 (AudioSet pretrained) |
| 입력 사양 | 16,000Hz mono, 1024 frame × 128 mel bins |
| 한글 폰트 | NanumGothic (Colab 설치) |
| Feature 추출 | AST 본체 pooler_output 768차원 (FE용 캐싱) |
| FE 학습 설정 | head(LayerNorm + Linear 768→10), AdamW(lr=1e-3), 50 epoch |
| Full FT 학습 설정 | 전체 모델, AdamW(lr=1e-5), batch=16, max 5 epoch, early stopping patience=2 |
| 결과 파일 | ast_fe_embeddings.pt, ast_full_ft_results.pt |
| 시각화 | ast_feature_ext_confusion.png, ast_full_ft_confusion.png, ast_training_curve.png, ast_vs_mlp_comparison.png |
| 재현 가능성 | random_seed=42 |

### 분석 결과 요약

| 분석 | accuracy | balanced accuracy | macro F1 | weighted F1 |
| --- | --- | --- | --- | --- |
| MLP 1차-B (비교 기준) | 0.6858 | 0.6988 | 0.7045 | 0.6868 |
| AST Feature Extraction | 0.8829 | 0.8861 | 0.8903 | 0.8824 |
| AST Full Fine-tuning | 0.8901 | 0.8999 | 0.8993 | 0.8890 |

AST는 MLP 대비 약 +0.20 (Accuracy) 개선되었으며, 본체를 동결한 Feature Extraction만으로도 Full Fine-tuning에 근접한 성능을 달성했다.

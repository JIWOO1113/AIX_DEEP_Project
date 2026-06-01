# 다양한 모델을 활용한 도시 환경 소리의 종류 분류 및 성능 평가

Members: 

# I. Proposal (Option A)

# II. Datasets

# III. Methodology

## III-I. 데이터 전처리

### Sample Rate: 22050 Hz
1. 전체 통계
2. 22050 Hz로 판단한 근거.

### Channels: Mono
1. 전체 통계
2. Mono로 판단한 근거(양쪽에서 들리는 소리 차이가 없음)

### Bit Depth: 32-float point
1. 전체 통계
2. float로 판단한 근거(정보 손실이 없다.)

### Duration: 4초, repeat padding
1. 전체 통계
2. repeat padding의 근거

### Noramlization: 
1. MFCC + SVM / RandomForest / XGBoost </br> 
    MFCC나 log-mel 통계 feature를 뽑은 뒤, train fold 기준으로 feature standardization
2. 2D CNN / CRNN용 log-mel spectrogram </br> 
    log-mel spectrogram을 만든 뒤, train set 전체의 mean/std로 정규화

---
## III-II 모델 

### 1. XGboost 

### 2. SVM

### 3. MLP

### 4. 2D CNN

### 5. RCNN

### 6. Pretrained audio model

---

# IV. Evaluation & Analysis

### Train: fold 1-8 / Validation: fold 9 / Test: fold 10 
<br>

### 평가 항목: 
| 통계값 | 설명 |
|---|---|
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

### 1. XGboost 

### 2. SVM

### 3. MLP

### 4. 2D CNN

### 5. RCNN

### 6. Pretrained audio model

### 7. 최종 비교 분석

# V. Related Work

# VI. Conclusion: Discussion

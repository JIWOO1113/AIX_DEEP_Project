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
Model Description
XGBoost는 Gradient Boosting 기반의 앙상블 학습 알고리즘으로, 여러 개의 결정트리(Decision Tree)를 순차적으로 학습하여 예측 성능을 향상시킨다. 과적합 방지 기능과 높은 학습 효율성을 제공하며, 다양한 머신러닝 문제에서 우수한 성능을 보이는 모델이다. 본 프로젝트에서는 다중 클래스 환경음 분류(Multi-class Sound Classification)에 활용하였다.
Hyperparameters
Number of Trees (n_estimators): 300
Max Depth: 5
Learning Rate: 0.05
Subsample: 0.8
Colsample Bytree: 0.8
Objective: multi:softmax
Number of Classes: 10
Evaluation Metric: mlogloss
Random State: 42
Features Used
MFCC (Mel-Frequency Cepstral Coefficients)
40개의 MFCC 계수 추출
각 계수의 평균(mean)과 표준편차(std) 계산
총 Feature 수: 80개
40 MFCC means
+ 40 MFCC standard deviations
= 80-dimensional feature vector

### 2. SVM
Model Description
SVM(Support Vector Machine)은 초평면(hyperplane)을 이용하여 서로 다른 클래스를 분류하는 지도학습 알고리즘이다. 본 프로젝트에서는 비선형 데이터 분류를 위해 RBF(Radial Basis Function) 커널을 사용하였다. SVM은 고차원 특징 공간에서 클래스 간 경계를 효과적으로 학습할 수 있으며, 비교적 적은 데이터에서도 안정적인 성능을 보이는 장점이 있다.
Hyperparameters
Kernel: RBF
C: 10
Gamma: scale
Class Weight: balanced
Random State: 42
Features Used
MFCC (Mel-Frequency Cepstral Coefficients)
40개의 MFCC 계수 추출
각 계수의 평균(mean)과 표준편차(std) 계산
총 Feature 수: 80개
40 MFCC means
+ 40 MFCC standard deviations
= 80-dimensional feature vector

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

# Duration

## 1. 원본 데이터의 길이 분포

### 1-1. FACT — 데이터셋이 어떤 길이 분포를 가지는가

UrbanSound8K 8,732개 파일은 대부분 정확히 4초이지만, 일부는 그보다 짧다.

### 1-2. 분석 기법

`scripts/analyze_duration_basic.py`에서:

- 각 wav 파일을 librosa.load로 읽어 길이(초) 측정
- 전체 분포, 클래스별 분포, Fold별 분포 집계
- 4초 미만 파일은 패딩 대상으로 분류

### 1-3. 결과

#### 전체 분포 (8,732개)

| 길이 구간 | 파일 수 | 비율 | 설명 |
| --- | --- | --- | --- |
| 정확히 4초 | 7,325 | 83.89% | 패딩 불필요 |
| 4초 미만 | 1,399 | 16.02% | 패딩 필요 |
| 4초 초과 | 8 | 0.09% | 자르기 필요 |

UrbanSound8K는 원본 오디오에서 4초 단위로 슬라이스된 데이터셋이다. 따라서 대부분 정확히 4초지만, 원본 길이가 4초가 안 되는 경우 그대로 슬라이스되어 4초 미만이 된다.

#### 클래스별 4초 미만 비율 (높은 순)

| 클래스 | 4초 미만 | 전체 | 비율 |
| --- | --- | --- | --- |
| gun_shot | 358 | 374 | 95.72% |
| car_horn | 226 | 429 | 52.68% |
| dog_bark | 325 | 1,000 | 32.50% |
| jackhammer | 197 | 1,000 | 19.70% |
| drilling | 195 | 1,000 | 19.50% |
| engine_idling | 39 | 1,000 | 3.90% |
| siren | 32 | 929 | 3.44% |
| children_playing | 24 | 1,000 | 2.40% |
| air_conditioner | 3 | 1,000 | 0.30% |
| street_music | 0 | 1,000 | 0.00% |

#### Fold별 4초 미만 비율 (높은 순)

| Fold | 4초 미만 | 전체 | 비율 |
| --- | --- | --- | --- |
| fold10 | 215 | 837 | 25.69% |
| fold6 | 192 | 823 | 23.33% |
| fold8 | 169 | 806 | 20.97% |
| fold2 | 154 | 888 | 17.34% |
| fold9 | 137 | 816 | 16.79% |
| fold1 | 134 | 873 | 15.35% |
| fold5 | 130 | 936 | 13.89% |
| fold3 | 111 | 925 | 12.00% |
| fold4 | 90 | 990 | 9.09% |
| fold7 | 67 | 838 | 7.99% |

### 1-4. 시각화

![길이 분포](https://github.com/JIWOO1113/AIX_DEEP_Project/raw/main/study-notes/Moreumi/urbansound8k-feature-preprocessing/duration_distribution.png)

### 1-5. 결과 해석

- 83.89%가 정확히 4초, 16.02%가 4초 미만 (패딩 필요), 0.09%가 4초 초과 (자르기 필요)
- 클래스별 4초 미만 비율 편차: gun_shot 95.72% ~ street_music 0.00%
- gun_shot의 95.72%는 데이터셋 전체 패턴과 매우 다른 극단값
- Fold별 4초 미만 비율 편차: fold10 25.69% ~ fold7 7.99% (약 3배)

### 1-6. 전처리 단계에서 얻은 정보

- 길이 통일이 필수 (4초 기준)
- 4초 미만 1,399개에 대한 패딩 방식 결정 필요
- gun_shot, car_horn처럼 짧은 파일 비율이 높은 클래스는 패딩 방식에 더 민감할 것으로 예상
- 후속 분석에서 패딩이 MFCC와 분류 정확도에 미치는 영향을 검증

---

## 2. Zero Padding이 MFCC에 미치는 영향

### 2-1. FACT — 4초 미만 파일을 4초로 맞추기 위한 zero padding이 MFCC를 얼마나 변형시키는지 측정한다

현재 전처리 설정은 4초 미만 파일에 zero padding(뒤에 0 채우기)을 적용한다. 이 처리가 원본 MFCC를 얼마나 보존하는지 측정해서 변환 방식 결정의 기초 자료로 삼는다.

zero padding은 신호 끝에 0을 채우는 가장 단순한 방식이다. 구현이 간단하지만 신호의 시간 평균 통계를 변형시킬 수 있다.

### 2-2. 분석 기법

`scripts/analyze_duration_padding.py`에서:

- 4초 미만 파일을 길이 구간별로 분류:
  * R1: 0.0 ~ 0.5초
  * R2: 0.5 ~ 1.0초
  * R3: 1.0 ~ 1.5초
  * R4: 1.5 ~ 2.0초
  * R5: 2.0 ~ 3.0초
  * R6: 3.0 ~ 4.0초
- 각 구간에서 최대 40개씩 샘플링 (클래스 다양성 확보, random_state=42)
- 총 236개 파일 분석
- 측정 지표:
  * 원본 신호 vs zero padding된 신호의 MFCC 평균 벡터 코사인 유사도
  * 원본 신호 vs zero padding된 신호의 MFCC 표준편차 벡터 코사인 유사도
  * 패딩 비율 (전체 4초 중 0이 차지하는 비율)

### 2-3. 결과

#### 길이 구간별 결과

| 구간 | n | 평균 패딩 비율 | MFCC 평균 유사도 | MFCC 표준편차 유사도 |
| --- | --- | --- | --- | --- |
| R1 (~0.5s) | 36 | 91.36% | 0.8494 | 0.8944 |
| R2 (0.5~1s) | 40 | 80.88% | 0.8697 | 0.8961 |
| R3 (1~1.5s) | 40 | 69.12% | 0.8533 | 0.8835 |
| R4 (1.5~2s) | 40 | 56.35% | 0.8382 | 0.8592 |
| R5 (2~3s) | 40 | 39.07% | 0.8903 | 0.8457 |
| R6 (3~4s) | 40 | 15.12% | **0.9704** | 0.8997 |

#### 임계값 0.95 통과율

| 구간 | MFCC 평균 통과 | MFCC 표준편차 통과 |
| --- | --- | --- |
| R1 | 22.2% | 50.0% |
| R2 | 27.5% | 25.0% |
| R3 | 35.0% | 45.0% |
| R4 | 42.5% | 32.5% |
| R5 | 65.0% | 32.5% |
| R6 | **80.0%** | 50.0% |

#### 클래스별 MFCC 평균 유사도 (낮은 순)

| 클래스 | 패딩 비율 | MFCC 평균 유사도 |
| --- | --- | --- |
| gun_shot | 64.68% | 0.7768 |
| jackhammer | 55.34% | 0.8323 |
| air_conditioner | 43.27% | 0.8827 |
| siren | 67.03% | 0.8936 |
| car_horn | 63.56% | 0.8940 |
| dog_bark | 59.51% | 0.9000 |
| engine_idling | 45.49% | 0.9040 |
| drilling | 57.04% | 0.9321 |
| children_playing | 37.97% | 0.9822 |

#### 패딩 비율과 MFCC 유사도의 상관계수

| 상관 | Pearson r |
| --- | --- |
| 패딩 비율 vs MFCC 평균 유사도 | -0.2207 |
| 패딩 비율 vs MFCC 표준편차 유사도 | +0.0330 |

### 2-4. 시각화

![패딩 영향](https://github.com/JIWOO1113/AIX_DEEP_Project/raw/main/study-notes/Moreumi/urbansound8k-feature-preprocessing/duration_padding_impact.png)

### 2-5. 결과 해석

- 길이가 짧을수록 MFCC 평균 유사도가 낮음 (R1 0.849 ~ R6 0.970)
- 다만 R1~R5 구간이 0.83~0.89로 비슷한 대역에 머물고 R6만 0.97로 두드러짐 — 단조감소 패턴이 아님
- 임계값 0.95 통과율은 R6(80%)만 높고, R1~R5는 22%~65%로 낮음
- gun_shot이 0.7768로 클래스 중 최저, 12개 파일에서 MFCC 유사도가 0.5 미만으로 측정됨
- 패딩 비율과 MFCC 유사도의 Pearson 상관계수가 -0.2207로 약함 — "패딩이 많을수록 MFCC가 망가진다"는 가설을 강하게 뒷받침하지 않음

본 측정만으로는 두 가지 의문이 해결되지 않는다:

1. 왜 R1~R5의 코사인 유사도가 단조감소하지 않는가
2. 왜 패딩 비율과 MFCC 유사도의 상관관계가 약한가

코사인 유사도는 벡터의 방향만 보고 크기는 무시한다. MFCC 평균 벡터의 크기가 작은 신호에서는 작은 노이즈에도 방향이 크게 바뀌어 측정이 불안정해질 수 있다. 이를 검증하기 위한 보조 측정을 진행한다.

### 2-6. 보조 측정 — 측정 불안정성 검증

코사인 유사도가 작은 벡터에서 불안정하다는 가설을 검증하기 위해 추가 지표를 측정한다.

#### 추가 지표

`scripts/analyze_duration_padding_aux.py`에서:

- 동일한 236개 파일에 대해 다음 3가지 추가 지표 측정:
  * 유클리드 거리: 벡터 크기까지 반영한 절대 거리
  * 원본 MFCC 평균 벡터의 L2 norm: 벡터 크기
  * 평균 절대 차이(MAE): 가장 직관적인 왜곡량

#### 그룹 비교 (코사인 유사도 < 0.5인 12개 vs 나머지 224개)

| 지표 | 불안정 그룹 (12개) | 안정 그룹 (224개) |
| --- | --- | --- |
| 원본 MFCC 벡터 크기 평균 | 147.7 | 271.3 |
| 유클리드 거리 평균 | - | - |
| 평균 절대 차이 평균 | 29.76 | 20.03 |
| 패딩 비율 평균 | 68.8% | 57.5% |

불안정 그룹의 원본 벡터 크기는 안정 그룹의 54% 수준.

#### 새로운 상관계수

| 상관 | Pearson r |
| --- | --- |
| 패딩 비율 vs 유클리드 거리 | **+0.8246** |
| 패딩 비율 vs MAE | **+0.8956** |
| 원본 벡터 크기 vs 코사인 유사도 | +0.4849 |

### 2-7. 시각화 (보조 측정)

![보조 측정](https://github.com/JIWOO1113/AIX_DEEP_Project/raw/main/study-notes/Moreumi/urbansound8k-feature-preprocessing/duration_padding_aux_impact.png)

### 2-8. 결과 해석 (보조 측정)

- 불안정 그룹(코사인 < 0.5)의 원본 벡터 크기가 안정 그룹의 54% — 측정 불안정 가설 확인
- 원본 벡터 크기와 코사인 유사도의 상관계수 +0.4849 — 벡터가 작을수록 코사인이 낮은 경향
- 유클리드 거리와 패딩 비율의 상관계수 +0.8246, MAE와는 +0.8956 — 패딩 비율과 MFCC 왜곡의 인과관계는 명확히 존재
- 길이 구간별로 보면 코사인은 R1~R5에서 진동하지만, 유클리드 거리와 MAE는 R1에서 R6로 단조감소

코사인 유사도는 작은 벡터에서 측정이 불안정해지므로 본 분석에 적합하지 않은 지표였다. 유클리드 거리와 MAE 기준으로 보면 패딩이 MFCC를 변형시키는 정도는 명확하며, 짧을수록 더 크게 변형된다.

### 2-9. 전처리 단계에서 얻은 정보

- Zero padding은 MFCC를 변형시키며, 그 정도는 패딩 비율에 비례 (유클리드 r=+0.82)
- 코사인 유사도만으로 보면 측정 불안정으로 인해 왜곡이 작은 것처럼 보일 수 있음
- gun_shot 등 짧은 폭발음 신호는 원본 MFCC 벡터 크기가 작아 측정이 특히 불안정
- 다음 단계에서 zero padding을 대체할 수 있는 방식(repeat, mirror)을 비교

---

## 3. 짧은 파일 처리 방식 옵션 비교

### 3-1. FACT — 3가지 패딩 방식의 MFCC 보존도를 측정한다

zero padding 외에도 짧은 파일을 4초로 맞추는 방식이 여러 가지 있다. 본 분석에서는 다음 3가지를 비교한다.

비교할 3가지 방식:

1. zero: 뒤에 0 채움 (현재 baseline)
2. repeat: 신호를 끝까지 반복 (np.tile)
3. mirror: 좌우반전 결합

추가로 짧은 파일을 일부 제외하는 옵션도 함께 검토한다:

- A: 그대로 두기 (전체 236개)
- B-1: R1(0.5초 미만) 제외 (200개)
- B-2: R1+R2(1초 미만) 제외 (160개)

### 3-2. 분석 기법

`scripts/analyze_duration_options.py`에서:

- 동일한 236개 파일 사용 (2단계와 동일, random_state=42)
- 각 파일을 zero/repeat/mirror 3가지 방식으로 4초 패딩
- 측정 지표:
  * 원본 vs 패딩 결과의 MFCC 평균 코사인 유사도
  * 원본 vs 패딩 결과의 MFCC 평균 유클리드 거리 (2단계에서 검증된 보조 지표)

### 3-3. 결과

#### 옵션별 전체 결과

| 옵션 | n | zero 코사인 | repeat 코사인 | mirror 코사인 | zero 유클리드 | repeat 유클리드 | mirror 유클리드 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| A (전체) | 236 | 0.8791 | **0.9996** | **0.9996** | 268.3 | 7.1 | 6.5 |
| B-1 (R1 제외) | 200 | 0.8844 | 0.9997 | 0.9996 | 240.0 | 6.3 | 6.2 |
| B-2 (R1+R2 제외) | 160 | 0.8881 | 0.9997 | 0.9995 | 205.5 | 6.4 | 6.8 |

#### 임계값 0.95 통과율

| 방식 | 통과율 |
| --- | --- |
| zero | 45.8% (108/236) |
| repeat | **100.0% (236/236)** |
| mirror | **100.0% (236/236)** |

#### 길이 구간별 패딩 방식 비교 (코사인)

| 구간 | zero | repeat | mirror |
| --- | --- | --- | --- |
| R1 | 0.8494 | 0.9990 | 0.9996 |
| R2 | 0.8697 | 0.9999 | 0.9999 |
| R3 | 0.8533 | 0.9999 | 0.9999 |
| R4 | 0.8382 | 0.9998 | 0.9998 |
| R5 | 0.8903 | 0.9995 | 0.9996 |
| R6 | 0.9704 | 0.9996 | 0.9988 |

#### 클래스별 최적 방식

| 클래스 | zero | repeat | mirror | best |
| --- | --- | --- | --- | --- |
| gun_shot | 0.7768 | 0.9990 | 0.9994 | mirror |
| jackhammer | 0.8323 | 0.9995 | 0.9987 | repeat |
| air_conditioner | 0.8827 | 1.0000 | 1.0000 | repeat |
| siren | 0.8936 | 0.9998 | 0.9998 | repeat |
| car_horn | 0.8940 | 0.9997 | 0.9998 | mirror |
| dog_bark | 0.9000 | 0.9997 | 0.9998 | mirror |
| engine_idling | 0.9040 | 1.0000 | 1.0000 | repeat |
| drilling | 0.9321 | 0.9997 | 0.9997 | repeat |
| children_playing | 0.9822 | 0.9998 | 0.9998 | repeat |

### 3-4. 시각화

![옵션 비교](https://github.com/JIWOO1113/AIX_DEEP_Project/raw/main/study-notes/Moreumi/urbansound8k-feature-preprocessing/duration_option_comparison.png)

### 3-5. 결과 해석

- repeat과 mirror는 zero 대비 MFCC 보존도가 압도적으로 높음 (코사인 0.879 → 0.9996, 유클리드 268 → 7)
- repeat과 mirror의 통과율은 모두 100% (zero는 45.8%)
- repeat과 mirror의 평균 코사인 차이는 0.000008로 사실상 동일
- 클래스별 best는 repeat 6개, mirror 3개로 두 방식이 거의 균등
- 옵션 B(짧은 파일 제외)는 zero 기준 코사인을 0.005~0.009 올리지만, 패딩 방식을 repeat/mirror로 바꾸면 0.121 상승 — 옵션 B의 효과는 미미
- 길이 구간별로 봐도 repeat/mirror는 R1~R6 어디서도 0.999 이상 — 길이와 무관하게 일관됨

#### repeat과 mirror가 우수한 이유

repeat은 신호를 그대로 반복하므로 시간 평균 통계가 보존된다. mirror도 좌우반전을 결합하지만 신호의 진폭 분포는 동일하므로 시간 평균이 유지된다. MFCC 평균 벡터는 시간축 평균 통계이므로, 두 방식 모두 원본과 거의 같은 값을 가진다.

zero padding은 신호 뒤에 0을 채우므로 시간 평균이 무음 쪽으로 끌려간다. 따라서 원본 MFCC 평균과 달라진다.

### 3-6. 전처리 단계에서 얻은 정보

- MFCC 보존도만 보면 repeat과 mirror가 zero보다 명확히 우수
- repeat과 mirror의 차이는 측정상 무의미 (코사인 차이 0.000008)
- 옵션 B(짧은 파일 제외)는 효과가 미미하며 데이터 손실(36~76개) 비용을 정당화하지 못함
- 다만 MFCC 보존도가 실제 분류 정확도와 같은 방향인지는 별도 검증 필요 (다음 단계)

---

## 4. 사전 점검 — 패딩 방식이 실제 분류 정확도에 미치는 영향

### 4-1. FACT — MFCC 보존도와 실제 분류 정확도는 다를 수 있다

3단계에서 측정한 MFCC 코사인 유사도와 유클리드 거리는 MFCC 평균 벡터(20개 숫자)만 비교한 것이다. 실제 모델은 다음 54개 feature를 입력으로 받는다:

- MFCC 평균 20개 + MFCC 표준편차 20개
- spectral feature 14개 (centroid, bandwidth, rolloff, contrast, zcr, rms)

3단계 결과만으로 패딩 방식을 결정하기 전에, 실제 분류 정확도로 검증한다.

### 4-2. 분석 기법

`scripts/analyze_duration_accuracy.py`에서:

- 전체 8,732개 파일을 3가지 패딩 방식으로 각각 처리
- 각 방식에 대해 54개 feature 추출 후 별도 CSV 저장
- RandomForest Classifier (n_estimators=100, random_state=42)
- UrbanSound8K 표준 10-fold Cross-Validation
- StandardScaler로 feature 표준화
- 측정:
  * 옵션별 전체 정확도 (mean ± std)
  * Fold별 정확도
  * 클래스별 정확도
  * 통계적 유의성 검증 (paired t-test)

### 4-3. 결과

#### 전체 정확도

| 옵션 | accuracy (mean ± std) | min ~ max |
| --- | --- | --- |
| zero | 0.6418 ± 0.0339 | 0.599 ~ 0.707 |
| repeat | **0.6487 ± 0.0350** | 0.609 ~ 0.718 |
| mirror | 0.6471 ± 0.0406 | 0.603 ~ 0.722 |

#### 통계적 유의성 (paired t-test, 10-fold)

| 비교 | t | p-value | 결론 |
| --- | --- | --- | --- |
| repeat vs zero | 2.057 | 0.0698 | 미유의 (경계선) |
| mirror vs zero | 0.918 | 0.3824 | 미유의 |
| repeat vs mirror | 0.489 | 0.6367 | 미유의 |

#### Fold별 정확도

| Fold | zero | repeat | mirror |
| --- | --- | --- | --- |
| 1 | 0.6518 | 0.6495 | 0.6426 |
| 2 | 0.6385 | 0.6453 | 0.6306 |
| 3 | 0.6108 | 0.6086 | 0.6032 |
| 4 | 0.6162 | 0.6212 | 0.6222 |
| 5 | 0.6688 | 0.6944 | **0.7105** |
| 6 | 0.6391 | 0.6379 | 0.6197 |
| 7 | 0.6134 | 0.6241 | 0.6337 |
| 8 | 0.5993 | 0.6216 | 0.6166 |
| 9 | 0.6728 | 0.6667 | 0.6703 |
| 10 | 0.7073 | 0.7180 | **0.7216** |

10개 fold 중 repeat이 zero보다 좋은 fold는 6개, mirror가 zero보다 좋은 fold는 5개.

#### 클래스별 정확도

| 클래스 | zero | repeat | mirror | repeat - zero | mirror - zero |
| --- | --- | --- | --- | --- | --- |
| air_conditioner | 0.458 | 0.412 | 0.436 | -0.046 | -0.022 |
| jackhammer | 0.520 | 0.562 | 0.566 | +0.042 | **+0.046** |
| car_horn | 0.524 | **0.576** | 0.536 | **+0.051** | +0.012 |
| engine_idling | 0.577 | 0.600 | 0.580 | +0.023 | +0.003 |
| drilling | 0.617 | 0.620 | 0.639 | +0.003 | +0.022 |
| siren | 0.695 | 0.700 | 0.700 | +0.004 | +0.004 |
| children_playing | 0.741 | 0.742 | 0.745 | +0.001 | +0.004 |
| dog_bark | 0.743 | 0.760 | 0.754 | +0.017 | +0.011 |
| street_music | 0.753 | 0.746 | 0.744 | -0.007 | -0.009 |
| gun_shot | 0.858 | 0.864 | 0.816 | +0.005 | **-0.043** |

### 4-4. 시각화

![분류 정확도](https://github.com/JIWOO1113/AIX_DEEP_Project/raw/main/study-notes/Moreumi/urbansound8k-feature-preprocessing/duration_option_accuracy.png)

### 4-5. 결과 해석

- 전체 정확도 차이: repeat이 zero보다 +0.69%p, mirror가 zero보다 +0.53%p
- 통계적 유의성: 모든 비교에서 p > 0.05 (미유의), 다만 repeat vs zero는 p=0.0698로 경계선
- Fold별 일관성: repeat 6/10, mirror 5/10에서 zero보다 우수 — 일관되지 않음
- 클래스별 차이: car_horn(+5.1%p, repeat), jackhammer(+4.6%p, mirror)에서 명확한 개선
- mirror의 gun_shot 악화: -4.3%p (가장 큰 변동폭)
- air_conditioner와 street_music이 패딩 변경에도 정확도 변동을 보임

#### MFCC 보존도와 실제 정확도의 괴리 (재확인)

3단계에서 측정한 MFCC 보존도는:

- zero: 코사인 0.879, 유클리드 268
- repeat: 코사인 1.000, 유클리드 7
- mirror: 코사인 1.000, 유클리드 6

repeat과 mirror가 MFCC를 거의 완벽히 보존했음에도 실제 분류 정확도 개선은 1%p 미만이었다. 채널 분석에서는 MFCC 유사도 0.998이 실제 정확도 14.4%p 차이로 나타났는데, 본 분석에서는 반대로 MFCC 보존도 차이가 컸음에도 정확도 차이는 작았다. MFCC 보존도와 분류 정확도의 관계는 양방향으로 다를 수 있다.

#### mirror의 gun_shot 악화

gun_shot은 짧은 폭발음과 잔향으로 구성된다. mirror padding은 좌우반전 신호를 결합하므로 인공적인 거울 구조가 만들어진다. 폭발 → 잔향이라는 시간적 비대칭성이 거울 구조에 의해 사라지면서 분류기에 혼동을 줄 수 있다. zero padding이 의외로 좋은 이유는 "이 시점 이후 신호 없음"이 gun_shot 특성과 일치하기 때문일 가능성이 있다.

#### air_conditioner와 street_music의 간접 영향

air_conditioner는 4초 미만 파일이 1,000개 중 3개(0.3%), street_music은 0개. 즉 이 두 클래스의 데이터 자체는 패딩 방식과 거의 무관하다. 그런데도 정확도가 변동한 이유는 다른 클래스(gun_shot 358개, car_horn 226개 등)의 패딩 처리가 RandomForest의 결정 경계 전체를 바꾸기 때문이다. 한 부분의 전처리 변경이 전체 모델의 예측에 간접 영향을 미친다는 점은 본 단계에서 처음 관찰된 현상이다.

### 4-6. 전처리 단계에서 얻은 정보

- repeat이 평균적으로 가장 좋지만 통계적 유의성은 경계선 (p=0.0698)
- repeat은 car_horn, jackhammer 등 어려운 클래스에서 명확한 개선
- mirror는 gun_shot에서 큰 악화 (-4.3%p)
- zero는 학계 표준이며, 모든 클래스에서 다른 방식과 동등하거나 약간 낮은 수준
- 패딩 방식 결정은 통계적 유의성보다 클래스별 영향과 학계 표준 여부에 더 무게를 두어야 함

---

## 5. 결정 가이드

### 5-1. 패딩 방식 결정 — zero vs repeat

분석 결과 두 선택지 모두 정당화 가능하다. 팀의 우선순위에 따라 결정한다.

#### 선택지 A: zero padding (학계 표준 / 단순성 우선)

**선택 근거**:

1. UrbanSound8K 학계 표준 (대부분의 논문이 zero padding 사용)
2. 다른 연구와 정확도 직접 비교 가능
3. 구현 단순 (numpy.pad의 기본 동작)
4. gun_shot에서 mirror보다 우수, repeat과 동등 (0.858 vs 0.864, 차이 0.6%p)
5. 통계적으로 repeat과 차이가 유의하지 않음 (p=0.0698)

**현재 분류 정확도**: 0.6418

**구현**:

```
import numpy as np
y_padded = np.pad(y, (0, N_SAMPLES - len(y)), mode="constant")
```

#### 선택지 B: repeat padding (정확도 우선)

**선택 근거**:

1. 평균 정확도 +0.69%p (0.6418 → 0.6487)
2. car_horn +5.1%p, jackhammer +4.2%p — 어려운 클래스에서 명확한 개선
3. 10개 클래스 중 8개에서 zero보다 좋거나 같음
4. MFCC 보존도 측면에서 zero 대비 압도적 우수 (코사인 0.879 → 1.000)
5. gun_shot에서도 zero와 동등 (0.858 vs 0.864)

**비용 및 한계**:

1. 비표준 (학계 비교 어려움)
2. 통계적 유의성 경계선 (p=0.0698)
3. air_conditioner -4.6%p (간접 영향, 짧은 파일은 3개뿐)
4. 구현이 약간 복잡 (반복 횟수 계산 필요)

**구현**:

```
import numpy as np
n_repeat = int(np.ceil(N_SAMPLES / len(y)))
y_padded = np.tile(y, n_repeat)[:N_SAMPLES]
```

#### 선택지 비교표

| 항목 | 선택지 A: zero | 선택지 B: repeat |
| --- | --- | --- |
| 전체 정확도 | 0.6418 | **0.6487** |
| 통계적 유의성 | - | p=0.0698 (경계선) |
| car_horn | 0.524 | **0.576** |
| jackhammer | 0.520 | **0.562** |
| gun_shot | 0.858 | 0.864 |
| MFCC 보존도 (코사인) | 0.879 | **1.000** |
| 학계 표준 | **표준** | 비표준 |
| 구현 복잡도 | **단순** | 약간 복잡 |

### 5-2. 짧은 파일 제외(옵션 B)를 채택하지 않은 이유

분석 단계에서 R1(0.5초 미만) 제외와 R1+R2(1초 미만) 제외를 함께 검토했으나 채택하지 않았다.

#### 검토 결과

| 옵션 | n | zero 코사인 | 기준(A) 대비 |
| --- | --- | --- | --- |
| A (전체) | 236 | 0.8791 | - |
| B-1 (R1 제외) | 200 | 0.8844 | +0.005 |
| B-2 (R1+R2 제외) | 160 | 0.8881 | +0.009 |

#### 채택하지 않은 이유

- 파일 36~76개 손실 대비 코사인 0.005~0.009 개선으로 효과 미미
- 패딩 방식을 repeat/mirror로 바꾸면 코사인 0.121 상승 — 같은 데이터 손실 없이 더 큰 효과
- gun_shot의 95.7%가 4초 미만이므로 R1 제외 시 gun_shot 데이터가 크게 손실됨
- 학계 표준은 전체 8,732개 사용

#### mirror padding을 채택하지 않은 이유

| 항목 | 결과 |
| --- | --- |
| 평균 정확도 | +0.53%p (repeat의 +0.69%p보다 낮음) |
| gun_shot | -4.3%p (zero, repeat보다 크게 악화) |
| 통계적 유의성 | p=0.3824 (repeat의 0.0698보다 약함) |

mirror는 평균 개선폭이 repeat보다 작고, gun_shot에서 크게 악화시킨다. zero와 repeat 사이에서 결정하는 것이 합리적이다.

### 5-3. 인지하고 가야 할 사항

**(1) MFCC 보존도와 분류 정확도의 괴리는 양방향**

채널 분석에서는 MFCC 유사도 0.998이 정확도 14.4%p 차이로 나타났다 (유사도가 차이를 숨김). Duration 분석에서는 MFCC 보존도 차이가 컸음에도 정확도 차이는 0.69%p에 그쳤다 (유사도가 차이를 부풀림). 단일 지표로 결정하지 말고 실제 분류로 검증 필수.

**(2) gun_shot은 짧은 파일 비율이 극단적으로 높음**

gun_shot은 1,000개 중 358개(95.7%)가 4초 미만으로, 다른 클래스(평균 16%)와 비교해 극단적이다. 패딩 방식 변경이 가장 큰 영향을 미치는 클래스이며, mirror 사용 시 -4.3%p 악화. 시간적 비대칭성이 중요한 신호에 대해서는 시간 반전을 포함하는 augmentation을 피해야 함.

**(3) car_horn과 jackhammer는 반복적 신호 특성**

car_horn(52.7%), jackhammer(19.7%)는 짧은 파일 비율이 높으면서도 반복적 신호 특성을 가진다. repeat padding에서 각각 +5.1%p, +4.2%p 개선. 본 모델 학습 시 이 두 클래스가 가장 어려운 클래스(정확도 0.5 수준)이므로 추가 검증 필요.

**(4) 간접 영향 — 패딩 방식이 전체 모델에 미치는 영향**

air_conditioner(짧은 파일 0.3%)와 street_music(0.0%)도 패딩 방식 변경에 따라 정확도가 변동했다. 다른 클래스의 패딩 처리가 RandomForest의 결정 경계 전체를 바꾸기 때문. 한 부분의 전처리 변경이 전체에 영향을 미친다는 점을 인지해야 함.

**(5) Fold별 편차 인지**

fold10이 0.71로 가장 높고 fold3, fold8이 0.60~0.62로 낮음. 모든 옵션에서 일관된 패턴. 단일 fold 결과로 옵션 우열을 판단하지 말고 10-fold 전체 평균으로 평가해야 함. 또한 4초 미만 비율도 fold10(25.7%) vs fold7(8.0%)로 약 3배 차이 — fold별 데이터 특성 차이도 함께 고려.

**(6) drilling이 여러 분석에서 일관되게 식별됨**

SR, 비트 깊이, 채널, Duration 분석 모두에서 drilling이 특이 클래스로 식별됨:

| 분석 | 지표 | 값 |
| --- | --- | --- |
| SR | MFCC 유사도 | 0.925 (미달) |
| 비트 깊이 | 저품질 비율 | 2.30% (최고) |
| 채널 | 모노 비율 | 14.10% (최고) |
| Duration | mirror 개선폭 | +2.2%p (repeat의 +0.3%p보다 큼) |

본 모델 학습 시 drilling 정확도를 별도로 모니터링할 가치가 있음.

### 5-4. 향후 모델 학습 단계에서 확인할 항목

- car_horn 정확도가 사전 점검(0.52~0.58)과 본 모델에서 유사한지
- jackhammer 정확도 (사전 점검 0.52~0.57)
- gun_shot이 본 모델에서도 mirror padding에 민감한지 (선택지 B 채택 시 무관)
- Fold3, Fold8의 cross-validation 결과 변동성
- 패딩 방식 변경 시 짧은 파일이 없는 클래스(air_conditioner, street_music)의 정확도 변동
- repeat padding 채택 시 +0.69%p 개선 효과가 본 모델에서도 유지되는지

---

## 6. 분석 환경

| 항목 | 값 |
| --- | --- |
| 사용 라이브러리 | librosa, soundfile, numpy, pandas, matplotlib, scikit-learn, scipy |
| 분석 스크립트 | scripts/analyze_duration_basic.py, scripts/analyze_duration_padding.py, scripts/analyze_duration_padding_aux.py, scripts/analyze_duration_options.py, scripts/analyze_duration_accuracy.py |
| 결과 데이터 | outputs/duration_detail.csv, outputs/duration_by_class.csv, outputs/duration_by_fold.csv, outputs/duration_padding_detail.csv, outputs/duration_padding_by_range.csv, outputs/duration_padding_by_class.csv, outputs/duration_padding_aux_detail.csv, outputs/duration_option_comparison_detail.csv, outputs/duration_option_comparison_summary.csv, outputs/duration_option_by_range.csv, outputs/duration_option_by_class.csv, outputs/duration_option_acc_summary.csv, outputs/duration_option_acc_by_fold.csv, outputs/duration_option_acc_by_class.csv, outputs/features_padding_comparison/features_zero.csv, outputs/features_padding_comparison/features_repeat.csv, outputs/features_padding_comparison/features_mirror.csv |
| 시각화 | outputs/duration_distribution.png, outputs/duration_padding_impact.png, outputs/duration_padding_aux_impact.png, outputs/duration_option_comparison.png, outputs/duration_option_accuracy.png |
| 분석 파라미터 | SR=22050, n_mfcc=20, n_fft=2048, hop_length=512, RandomForest n_estimators=100, 10-fold CV |
| 재현 가능성 | random_state=42 |

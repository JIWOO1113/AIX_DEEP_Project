# Loudness

## 1. 원본 데이터의 음량 분포

### 1-1. FACT — 데이터셋이 어떤 음량 분포를 가지는가

UrbanSound8K 8,732개 파일의 음량은 매우 넓은 범위에 걸쳐 분포한다. 일부 파일은 매우 작은 음량이고, 일부는 이미 클리핑 상태다.

음량 측정의 두 가지 표준 지표:

- **RMS (Root Mean Square)**: 신호 전체의 평균 에너지를 나타낸다. 인지되는 음량과 가장 가깝다.
- **Peak**: 신호의 최대 절대값. 1.0에 가까우면 클리핑(신호가 최대값에 잘림) 위험이 있다.

### 1-2. 분석 기법

`scripts/analyze_loudness_basic.py`에서:

- 기존 `outputs/audio_file_summary.csv` 재활용 (01_inspect.py 결과)
- 전체 분포, 클래스별 분포, Fold별 분포 집계
- Peak를 6개 구간(P1~P6)으로 나누어 분석
- 클리핑(`has_clipping`=True) 파일 식별

### 1-3. 결과

#### 전체 분포 (8,732개)

| 지표 | 평균 | 중앙값 | min | max |
| --- | --- | --- | --- | --- |
| RMS | 0.0768 | 0.0577 | 0.000003 | 0.7090 |
| Peak | 0.4156 | 0.3540 | 0.000031 | 1.0000 |

#### Peak 구간별 분포

| 구간 | 범위 | 파일 수 | 비율 |
| --- | --- | --- | --- |
| P1 | ~0.1 (매우 작음) | 1,242 | 14.22% |
| P2 | 0.1~0.3 | 2,554 | 29.25% |
| P3 | 0.3~0.5 | 1,833 | 21.00% |
| P4 | 0.5~0.7 | 1,401 | 16.04% |
| P5 | 0.7~0.9 | 822 | 9.41% |
| P6 | 0.9~1.0 (클리핑 위험) | 880 | 10.08% |

#### 클리핑 발생

- **전체 클리핑: 567개 (6.49%)**

#### 클래스별 음량 (Peak 평균 내림차순)

| 클래스 | n | RMS 평균 | Peak 평균 | 클리핑 수 | 클리핑 비율 |
| --- | --- | --- | --- | --- | --- |
| gun_shot | 374 | 0.1462 | 0.8943 | 232 | **62.03%** |
| dog_bark | 1000 | 0.0782 | 0.5099 | 108 | 10.80% |
| jackhammer | 1000 | 0.0712 | 0.4972 | 49 | 4.90% |
| drilling | 1000 | 0.0801 | 0.4647 | 35 | 3.50% |
| car_horn | 429 | 0.0859 | 0.4422 | 15 | 3.50% |
| street_music | 1000 | 0.0726 | 0.4226 | 24 | 2.40% |
| engine_idling | 1000 | 0.0939 | 0.3733 | 71 | 7.10% |
| siren | 929 | 0.0666 | 0.3092 | 10 | 1.08% |
| air_conditioner | 1000 | 0.0581 | 0.3019 | 5 | 0.50% |
| children_playing | 1000 | 0.0369 | 0.2755 | 18 | 1.80% |

#### Fold별 음량

| Fold | RMS 평균 | Peak 평균 | 클리핑 비율 |
| --- | --- | --- | --- |
| fold1 | 0.0793 | 0.4180 | 6.18% |
| fold2 | 0.0626 | 0.3669 | 3.15% |
| fold3 | 0.0809 | 0.4313 | 7.46% |
| fold4 | 0.0833 | 0.4421 | 6.46% |
| fold5 | 0.0712 | 0.3987 | 5.34% |
| fold6 | 0.0820 | 0.4492 | 9.48% |
| fold7 | 0.0712 | 0.4076 | 7.04% |
| fold8 | 0.0828 | 0.4337 | 7.94% |
| fold9 | 0.0691 | 0.4070 | 7.23% |
| fold10 | 0.0838 | 0.4053 | 5.50% |

### 1-4. 시각화

![음량 분포](https://github.com/JIWOO1113/AIX_DEEP_Project/raw/main/study-notes/Moreumi/urbansound8k-feature-preprocessing/loudness_distribution.png)

### 1-5. 결과 해석

- 음량 분포 범위가 매우 넓음 (Peak 0.000031 ~ 1.0000, 3자릿수 차이)
- P1 (매우 작음) 14.22%, P6 (클리핑 위험) 10.08% — 양 극단이 모두 의미 있는 비중
- 전체 클리핑 발생 567개 (6.49%) — 무시할 수 없는 수
- gun_shot의 클리핑 비율 62.03%는 다른 클래스(평균 4.4%)와 비교해 극단적
- 클래스별 평균 Peak가 gun_shot 0.894 ~ children_playing 0.275로 약 3배 차이
- Fold별 클리핑 비율도 fold6(9.48%) ~ fold2(3.15%)로 3배 차이 — 녹음 환경 차이 추정

### 1-6. 전처리 단계에서 얻은 정보

- 음량 통일이 필요한 정도의 분포 차이 확인
- gun_shot 클리핑 232개와 그 외 클리핑 335개에 대한 처리 방향 검토 필요
- 클래스 간 음량 차이가 분류 단서로 작동할 가능성 존재
- 정규화 방식(none/peak/rms) 비교를 통해 실제 영향 측정

---

## 2. Peak Normalize가 MFCC에 미치는 영향

### 2-1. FACT — 현재 설정인 peak normalize가 MFCC를 얼마나 변형시키는지 측정한다

현재 전처리 설정은 peak normalize를 적용한다. 이 처리가 원본 MFCC를 얼마나 보존하는지 측정해서 정규화 방식 결정의 기초 자료로 삼는다.

Peak Normalize의 정의:

```
y_normalized = y / max(|y|)
```

신호의 최대 절대값으로 나누어 모든 파일의 peak를 1.0으로 통일하는 가장 단순한 정규화 방식이다. 이미 max=1.0인 클리핑 파일은 변화가 없다.

수학적으로 신호 y를 스칼라 k로 곱하면 log spectrum에 일정한 offset(2·log(k))이 더해지므로, MFCC 평균 벡터의 0번째 계수에 큰 변화가 생긴다.

### 2-2. 분석 기법

`scripts/analyze_loudness_normalize.py`에서:

- Peak 구간(P1~P6) 각 40개씩 샘플링 (클래스 다양성 확보, random_state=42)
- 총 240개 파일 분석
- 측정 지표:
  * 원본 vs Peak normalize MFCC 평균 코사인 유사도
  * 원본 vs Peak normalize MFCC 평균 유클리드 거리 (보조 지표)
  * 증폭 비율 (1 / peak)

### 2-3. 결과

#### Peak 구간별 결과

| 구간 | n | 평균 Peak | 평균 증폭비 | cos_mean | euc_mean |
| --- | --- | --- | --- | --- | --- |
| P1 (~0.1) | 40 | 0.0486 | **20.6×** | **0.635** | 282.8 |
| P2 (0.1~0.3) | 40 | 0.1819 | 5.5× | 0.815 | 164.7 |
| P3 (0.3~0.5) | 40 | 0.3863 | 2.6× | 0.864 | 93.7 |
| P4 (0.5~0.7) | 40 | 0.5871 | 1.7× | 0.963 | 51.7 |
| P5 (0.7~0.9) | 40 | 0.7794 | 1.3× | 0.982 | 28.3 |
| P6 (0.9~1.0) | 40 | 0.9714 | 1.0× | **0.993** | 7.6 |

#### 임계값 0.95 통과율

| 구간 | cos_mean 통과율 |
| --- | --- |
| P1 | 5.0% |
| P2 | 25.0% |
| P3 | 42.5% |
| P4 | 77.5% |
| P5 | 90.0% |
| P6 | 95.0% |

#### 클리핑 vs 정상 파일 비교

| 그룹 | n | cos_mean | euc_mean |
| --- | --- | --- | --- |
| **클리핑** | 22 | **0.9970** | 5.20 |
| 정상 | 218 | 0.8628 | 114.80 |

#### 클래스별 MFCC 보존도 (낮은 순)

| 클래스 | 평균 Peak | cos_mean |
| --- | --- | --- |
| jackhammer | 0.446 | **0.714** |
| drilling | 0.527 | 0.797 |
| engine_idling | 0.472 | 0.827 |
| air_conditioner | 0.354 | 0.864 |
| street_music | 0.515 | 0.912 |
| siren | 0.398 | 0.917 |
| children_playing | 0.388 | 0.919 |
| car_horn | 0.539 | 0.942 |
| dog_bark | 0.489 | 0.944 |
| gun_shot | 0.613 | 0.976 |

#### 증폭 비율과 MFCC 왜곡의 상관계수

| 상관 | Pearson r |
| --- | --- |
| log(증폭비율) vs cos_mean | -0.940 |
| log(증폭비율) vs euc_mean | **+0.997** |
| peak vs euc_mean | -0.912 |

### 2-4. 시각화

![Peak normalize 영향](https://github.com/JIWOO1113/AIX_DEEP_Project/raw/main/study-notes/Moreumi/urbansound8k-feature-preprocessing/loudness_normalize_impact.png)

### 2-5. 결과 해석

- Peak 구간이 작을수록 MFCC 평균 유사도가 단조감소 (P6 0.993 → P1 0.635)
- 임계값 0.95 통과율도 P6 95% → P1 5%로 매우 큰 차이
- 클리핑 파일은 거의 변화 없음 (cos 0.997) — 이미 max=1.0이라 정규화 효과 없음
- 정상 파일은 큰 변형 발생 (cos 0.863)
- log(증폭비율) vs 유클리드 거리의 Pearson 상관계수 +0.997 — 거의 완벽한 비례 관계
- 클래스별로는 jackhammer(0.714)가 가장 크게 영향받음 — 평균 Peak가 작은 클래스일수록 영향 큼

#### 수학적 배경

MFCC는 log spectrum 기반이다. 신호 y에 스칼라 k를 곱하면:

```
log|FFT(k·y)|² = 2·log(k) + log|FFT(y)|²
```

모든 주파수 빈에 동일한 offset(2·log(k))이 더해진다. 이 offset이 DCT(이산코사인변환)를 거치면서 MFCC의 0번째 계수에만 큰 변화를 만든다. 0번째 계수는 신호 전체 에너지를 나타내므로, 음량 변화가 직접 반영된다.

다른 MFCC 계수들은 거의 영향받지 않으므로, MFCC 표준편차(시간 변화 패턴)는 정확히 보존된다. 실제로 240개 모두에서 cos_std = 1.0000으로 측정되었다.

### 2-6. 전처리 단계에서 얻은 정보

- Peak normalize는 MFCC 평균을 명확히 변형 (특히 작은 신호)
- 변형의 크기는 증폭 비율에 비례 (log scale 기준 r=+0.997)
- 클리핑 파일은 정규화 영향이 없으나, 정상 파일은 영향이 큼
- MFCC 표준편차는 정규화에 완전히 무관 (시간 변화 패턴 보존)
- 다음 단계에서 peak 외 다른 정규화 방식(rms)과 정규화 안 함(none)을 비교

---

## 3. 정규화 방식 비교

### 3-1. FACT — 3가지 정규화 방식의 MFCC 보존도를 비교한다

비교할 3가지 방식:

| 방식 | 수식 | 설명 |
| --- | --- | --- |
| **none** | y | 정규화 안 함 (원본 그대로) |
| **peak** | y / max(|y|) | 최대값을 1.0으로 통일 (현재 설정) |
| **rms** | y × (0.1 / current_rms) | RMS를 0.1로 통일 |

RMS Normalize는 평균 에너지(RMS) 기준이라 peak보다 안정적이지만, target_rms 값을 직접 설정해야 하는 자유도가 있다. 본 분석에서는 target = 0.1을 사용했다 (전체 평균 0.077보다 약간 높은, 음향 분석에서 자주 쓰이는 값).

### 3-2. 분석 기법

`scripts/analyze_loudness_methods.py`에서:

- 2단계와 동일한 240개 파일 (random_state=42로 재현)
- 각 파일을 none/peak/rms 3가지 방식으로 처리
- 측정 지표:
  * 원본 vs 각 방식 처리 후의 MFCC 평균 코사인 유사도
  * 원본 vs 각 방식 처리 후의 MFCC 평균 유클리드 거리

### 3-3. 결과

#### 전체 방식별 결과

| 방식 | cos_mean | euc_mean | 0.95 통과율 |
| --- | --- | --- | --- |
| **none** | **1.0000** | **0.00** | **100.0%** |
| peak | 0.8751 | 104.79 | 55.8% |
| rms | 0.9548 | 89.17 | **74.2%** |

#### Peak 구간별 비교 (코사인)

| 구간 | none | peak | rms |
| --- | --- | --- | --- |
| P1 (~0.1) | 1.000 | 0.635 | **0.881** |
| P2 (0.1~0.3) | 1.000 | 0.815 | **0.958** |
| P3 (0.3~0.5) | 1.000 | 0.864 | **0.985** |
| P4 (0.5~0.7) | 1.000 | 0.963 | **0.985** |
| P5 (0.7~0.9) | 1.000 | 0.982 | 0.981 |
| P6 (0.9~1.0) | 1.000 | **0.993** | 0.939 |

#### Peak 구간별 0.95 통과율 비교

| 구간 | peak | rms |
| --- | --- | --- |
| P1 | 5.0% | **20.0%** |
| P2 | 25.0% | **77.5%** |
| P3 | 42.5% | **95.0%** |
| P4 | 77.5% | **92.5%** |
| P5 | 90.0% | 92.5% |
| P6 | **95.0%** | 67.5% |

#### rms vs peak 파일별 비교

| 비교 | 결과 |
| --- | --- |
| rms 코사인 > peak 코사인 | 161/240 (67.1%) |
| rms 유클리드 < peak 유클리드 | 154/240 (64.2%) |
| 평균 cos 차이 (rms - peak) | +0.0797 |
| 평균 euc 차이 (rms - peak) | -15.62 |

#### 클리핑 vs 비클리핑 비교

| 그룹 | none | peak | rms |
| --- | --- | --- | --- |
| 클리핑 (22개) | 1.000 / 0.00 | **0.997 / 5.20** | 0.923 / 70.96 |
| 비클리핑 (218개) | 1.000 / 0.00 | 0.875 / 105 | **0.955 / 89** |

### 3-4. 시각화

![정규화 방식 비교](https://github.com/JIWOO1113/AIX_DEEP_Project/raw/main/study-notes/Moreumi/urbansound8k-feature-preprocessing/loudness_methods_comparison.png)

### 3-5. 결과 해석

- none은 당연히 cos=1.0 (변화 없음, 검증용 baseline)
- 평균적으로 rms(0.955)가 peak(0.875)보다 MFCC 보존도 높음
- 임계값 0.95 통과율: rms 74.2%, peak 55.8% — rms가 18%p 더 높음
- Peak 구간별로 P1~P4에서는 rms가 명확히 우세, P5는 동등, P6는 peak가 우세
- 파일별 승률에서도 rms가 67.1%로 peak보다 우세
- 클리핑 파일에서는 peak가 압도적으로 좋음 (0.997 vs 0.923) — 이미 max=1.0이라 변화 없음
- 비클리핑 파일에서는 rms가 좋음 (0.955 vs 0.875)

#### 왜 P6 구간에서 rms가 더 나쁜가

P6 구간 파일들의 평균 특성:

| 항목 | 값 |
| --- | --- |
| 평균 peak | 0.971 |
| 평균 rms | 0.189 |
| peak 증폭비 | 1.03× (거의 변화 없음) |
| rms 증폭비 (target 0.1) | **0.53× (절반으로 축소)** |

P6 파일들은 이미 RMS가 큰 편이라 target 0.1로 맞추면 오히려 축소된다. 신호 축소도 MFCC를 변형시키므로 보존도가 떨어진다. 즉 RMS normalize는 양방향(증폭/축소)으로 작동할 수 있다.

### 3-6. 전처리 단계에서 얻은 정보

- MFCC 보존도 순위: none > rms > peak
- rms는 peak보다 평균적으로 우수하지만 P6 구간(이미 큰 신호)에서는 약함
- 클리핑 파일과 비클리핑 파일의 정규화 반응이 정반대
- target_rms 값(0.1)이 결과를 결정 — target을 바꾸면 결과도 바뀜
- 다음 단계에서 MFCC 보존도가 실제 분류 정확도로도 이어지는지 검증

---

## 4. 사전 점검 — 정규화 방식이 실제 분류 정확도에 미치는 영향

### 4-1. FACT — MFCC 보존도와 실제 분류 정확도는 같은 방향인지 확인한다

3단계에서 측정한 MFCC 보존도(none > rms > peak)가 실제 RandomForest 분류 정확도에서도 같은 순서로 나타나는지 검증한다. 

채널 분석에서는 MFCC 유사도 0.998이 정확도 14.4%p 차이로 나타났고, Duration 분석에서는 MFCC 보존도 차이가 컸음에도 정확도 차이는 0.69%p에 그쳤다. 두 분석에서 본 것처럼 MFCC와 정확도의 관계는 양방향으로 다를 수 있으므로 별도 검증이 필요하다.

### 4-2. 분석 기법

`scripts/analyze_loudness_accuracy.py`에서:

- 전체 8,732개 파일을 3가지 정규화 방식으로 각각 처리
- 처리 순서: 길이 통일(zero padding 4초) → 정규화 → feature 추출
- 각 방식에 대해 54개 feature 추출 후 별도 CSV 저장
- RandomForest Classifier (n_estimators=100, random_state=42)
- UrbanSound8K 표준 10-fold Cross-Validation
- StandardScaler로 feature 표준화
- 측정:
  * 옵션별 전체 정확도 (mean ± std)
  * Fold별 정확도
  * 클래스별 정확도
  * 통계적 유의성 검증 (paired t-test, paired 10-fold 결과)

### 4-3. 결과

#### 전체 정확도

| 방식 | accuracy (mean ± std) | min ~ max |
| --- | --- | --- |
| **none** | **0.6418 ± 0.0339** | 0.599 ~ 0.707 |
| peak | 0.6314 ± 0.0454 | 0.568 ~ 0.722 |
| rms | 0.6350 ± 0.0341 | 0.597 ~ 0.712 |

#### 통계적 유의성 (paired t-test, 10-fold)

| 비교 | t | p-value | 결론 |
| --- | --- | --- | --- |
| peak vs none | -2.018 | **0.0744** | 미유의 (경계선) |
| rms vs none | -1.599 | 0.1442 | 미유의 |
| rms vs peak | +0.632 | 0.5434 | 미유의 |

#### Fold별 정확도

| Fold | none | peak | rms | peak-none | rms-none |
| --- | --- | --- | --- | --- | --- |
| 1 | 0.6518 | 0.6518 | 0.6598 | 0.000 | +0.008 |
| 2 | 0.6385 | 0.6453 | 0.6182 | +0.007 | -0.020 |
| 3 | 0.6108 | 0.5676 | 0.5968 | -0.043 | -0.014 |
| 4 | 0.6162 | 0.6081 | 0.6263 | -0.008 | +0.010 |
| 5 | 0.6688 | 0.6635 | 0.6496 | -0.005 | -0.019 |
| 6 | 0.6391 | 0.6148 | 0.6221 | -0.024 | -0.017 |
| 7 | 0.6134 | 0.5967 | 0.6074 | -0.017 | -0.006 |
| 8 | 0.5993 | 0.5868 | 0.6079 | -0.013 | +0.009 |
| 9 | 0.6728 | 0.6581 | 0.6495 | -0.015 | -0.023 |
| 10 | 0.7073 | 0.7216 | 0.7121 | +0.014 | +0.005 |

10개 fold 중 none이 가장 좋은 fold: peak 대비 7개, rms 대비 6개.

#### 클래스별 정확도

| 클래스 | none | peak | rms | peak-none | rms-none |
| --- | --- | --- | --- | --- | --- |
| air_conditioner | 0.458 | 0.421 | 0.424 | -0.037 | -0.034 |
| jackhammer | 0.520 | 0.486 | 0.498 | -0.034 | -0.022 |
| car_horn | 0.524 | 0.527 | 0.522 | +0.002 | -0.002 |
| engine_idling | 0.577 | 0.586 | **0.597** | +0.009 | **+0.020** |
| drilling | 0.617 | 0.604 | 0.596 | -0.013 | -0.021 |
| siren | 0.695 | 0.712 | **0.724** | +0.016 | **+0.029** |
| children_playing | 0.741 | 0.717 | 0.704 | -0.024 | **-0.037** |
| dog_bark | 0.743 | 0.732 | 0.732 | -0.011 | -0.011 |
| street_music | 0.753 | 0.758 | **0.773** | +0.005 | **+0.020** |
| gun_shot | 0.858 | 0.853 | 0.858 | -0.005 | 0.000 |

클래스별 best: **none 6개**, rms 3개, peak 1개.

### 4-4. 시각화

![분류 정확도](https://github.com/JIWOO1113/AIX_DEEP_Project/raw/main/study-notes/Moreumi/urbansound8k-feature-preprocessing/loudness_acc.png)

### 4-5. 결과 해석

- 전체 정확도: none(0.6418) > rms(0.6350) > peak(0.6314)
- 3단계 MFCC 보존도 순서와 정확도 순서가 일치
- 통계적 유의성: 모든 비교에서 p > 0.05 (미유의), 다만 peak vs none은 p=0.0744로 경계선
- 클래스별 best는 none이 6개로 압도적
- siren(+2.9%p), street_music(+2.0%p), engine_idling(+2.0%p)에서는 rms가 명확히 우세
- children_playing(-3.7%p), air_conditioner(-3.4%p)에서는 정규화가 큰 손해

#### 음량이 분류 단서로 작동하고 있음

전체적으로 none이 가장 좋다는 결과는 **음량 자체가 분류 단서로 사용되고 있음**을 의미한다. UrbanSound8K에서:

- gun_shot은 평균 peak 0.89로 압도적 → "큰 신호 = gun_shot일 확률 ↑"
- children_playing은 평균 peak 0.27로 가장 작음 → "작은 신호 = children_playing일 확률 ↑"

이런 단서를 정규화로 제거하면 정확도가 떨어진다. 특히 음량이 작은 클래스(children_playing, air_conditioner)에서 -3%p 이상 악화된 것이 이를 뒷받침한다.

#### 일부 클래스에서는 정규화가 도움

siren, street_music, engine_idling은 다양한 환경에서 녹음된 것으로 추정되는 클래스다. 같은 신호여도 녹음 환경에 따라 음량이 다르므로, 정규화가 일관성을 만들어주어 정확도가 개선된다. siren에서 rms +2.9%p 개선이 가장 큰 예시다.

#### MFCC 보존도와 정확도의 관계 — 도메인별로 다름

| 분석 | MFCC 보존도와 정확도 관계 |
| --- | --- |
| 채널 | 보존도 0.998인데 정확도 차이 14.4%p (보존도가 차이 숨김) |
| Duration | 보존도 차이 큰데 정확도 차이 0.69%p (보존도가 차이 부풀림) |
| Loudness | 보존도 순위 = 정확도 순위 일치 |

세 분석을 종합하면, MFCC 보존도와 분류 정확도의 관계는 도메인마다 다르다. 단일 지표로 결정하지 말고 실제 분류로 검증해야 한다는 점이 다시 확인되었다.

### 4-6. 전처리 단계에서 얻은 정보

- none이 평균적으로 가장 좋지만 통계적 유의성은 미유의
- peak vs none은 경계선(p=0.0744), rms vs none은 미유의(p=0.1442)
- rms는 peak와 정확도에서 거의 차이 없음(0.36%p, p=0.5434)
- 음량이 분류 단서로 작동하므로 정규화 시 일부 클래스 손해, 일부 클래스 이득
- 정규화 방식 결정은 학계 표준 여부와 클래스별 영향을 함께 고려해야 함

---

## 5. 결정 가이드

### 5-1. 정규화 방식 결정 — none vs peak vs rms

분석 결과 세 선택지 모두 정당화 가능하다. 팀의 우선순위에 따라 결정한다.

#### 선택지 A: none (정확도 최우선)

**선택 근거**:

1. 평균 정확도가 가장 높음 (0.6418)
2. MFCC 보존 완벽 (cos = 1.000, euc = 0)
3. 10개 클래스 중 6개에서 best
4. 음량 정보 보존 → gun_shot, children_playing 등 음량 특성이 강한 클래스에 유리
5. 구현 단순 (전처리 자체가 없음)

**비용 및 한계**:

1. 음량 통일 효과 없음 → 녹음 환경이 다른 데이터셋과 비교 어려움
2. 음량이 분류 단서로 작동 → 새로운 환경에서 정확도 변동 가능
3. 학계 표준에서 일부 벗어남 (UrbanSound8K 일부 논문은 정규화 사용)

**현재 분류 정확도**: 0.6418

**구현**:

```
# 별도 처리 없음
y_processed = y
```

#### 선택지 B: peak (학계 표준 / 호환성 우선)

**선택 근거**:

1. UrbanSound8K 학계 표준 (대부분의 논문이 peak normalize 사용)
2. 다른 연구와 정확도 직접 비교 가능
3. 구현 단순 (numpy 한 줄)
4. 클리핑 파일 처리에 안정적 (변화 없음)

**비용 및 한계**:

1. 평균 정확도 -1.04%p (0.6418 → 0.6314)
2. 통계적 유의성 경계선 (p=0.0744)
3. children_playing -2.4%p, air_conditioner -3.7%p 등 조용한 클래스에서 손해
4. MFCC 보존도가 세 방식 중 가장 낮음 (cos 0.875)

**구현**:

```
import numpy as np
y_processed = y / np.max(np.abs(y))
```

#### 선택지 C: rms (절충안, 다만 peak와 큰 차이 없음)

**선택 근거**:

1. MFCC 보존도가 peak보다 높음 (0.955 vs 0.875)
2. 평균 정확도가 peak보다 약간 높음 (0.6350 vs 0.6314)
3. siren(+2.9%p), street_music(+2.0%p)에서 명확한 개선

**비용 및 한계**:

1. **정확도가 peak와 통계적으로 차이 없음 (p=0.5434)** — 사실상 동등
2. 평균 정확도가 none보다 -0.68%p 낮음
3. target_rms 값(0.1)이 결과를 좌우하는 또 다른 결정 변수
4. P6 구간(이미 큰 신호)에서는 오히려 보존도 떨어짐
5. peak보다 비표준 (학계 사용 빈도 낮음)

**구현**:

```
import numpy as np
target_rms = 0.1
current_rms = np.sqrt(np.mean(y ** 2))
y_processed = y * (target_rms / current_rms)
```

#### 선택지 비교표

| 항목 | A: none | B: peak | C: rms |
| --- | --- | --- | --- |
| 전체 정확도 | **0.6418** | 0.6314 | 0.6350 |
| 통계적 유의성 (vs none) | - | p=0.0744 (경계선) | p=0.1442 |
| MFCC 보존도 (코사인) | **1.000** | 0.875 | 0.955 |
| children_playing | **0.741** | 0.717 | 0.704 |
| air_conditioner | **0.458** | 0.421 | 0.424 |
| siren | 0.695 | 0.712 | **0.724** |
| street_music | 0.753 | 0.758 | **0.773** |
| gun_shot | **0.858** | 0.853 | **0.858** |
| 학계 표준 | 일부 | **표준** | 비표준 |
| 음량 통일 효과 | ❌ | ✓ | ✓ |
| 음량 정보 보존 | ✓ | ❌ | ❌ |

### 5-2. rms를 별도 선택지로 두지 않은 이유

3단계에서 rms는 peak보다 MFCC 보존도가 명확히 우수했다 (cos 0.955 vs 0.875, +0.08). 그러나 4단계 실제 분류 정확도에서는 두 방식의 차이가 사실상 사라졌다:

| 비교 | 차이 | 통계적 유의성 |
| --- | --- | --- |
| rms vs peak (정확도) | +0.36%p | p=0.5434 (전혀 미유의) |
| rms vs none (정확도) | -0.68%p | p=0.1442 (미유의) |
| peak vs none (정확도) | -1.04%p | p=0.0744 (경계선) |

즉 정확도 관점에서 rms는 peak의 명확한 대안이 되지 못한다. 정확도를 우선한다면 none, 학계 표준을 우선한다면 peak를 선택하는 것이 합리적이다. rms는 두 가지 모두에서 중간 위치에 있으며 명확한 우위 없이 결정 복잡도만 늘린다.

다만 siren, street_music, engine_idling 같은 일부 클래스에서는 rms가 명확히 도움된다. 향후 클래스별 모델을 분리해서 학습할 경우 이 정보를 활용할 수 있다.

### 5-3. 인지하고 가야 할 사항

**(1) 음량은 UrbanSound8K에서 분류 단서로 작동**

평균 Peak 기준 gun_shot(0.89) ~ children_playing(0.27)로 약 3배 차이. 모델은 이 음량 차이를 학습에 활용하고 있으며, 정규화로 음량을 통일하면 children_playing -3.7%p, air_conditioner -3.4%p로 명확한 손해 발생. 본 모델 학습 시 이 점을 인지해야 함.

**(2) 클리핑 파일 567개(6.49%)의 존재**

이미 max=1.0에 닿아 정보가 잘린 파일들. Peak normalize는 이 파일들에 효과 없음(변화 없음). gun_shot 374개 중 232개(62%)가 클리핑이라 gun_shot 분류 자체가 클리핑 패턴 의존 가능성. 데이터셋의 본질적 특성이며 전처리로 해결 불가.

**(3) 매우 조용한 파일 1,242개(14.2%)의 노이즈 증폭 위험**

P1 구간 파일들은 peak normalize 시 평균 20.6배 증폭됨. 일부는 100배 이상. 신호와 함께 환경 노이즈도 함께 증폭되어 SNR 악화 가능. 그러나 4단계 결과에서는 정확도에 큰 영향 없음으로 측정됨.

**(4) MFCC 보존도와 분류 정확도의 관계는 도메인별로 다름**

| 분석 | MFCC 0.95 통과율 차이 | 실제 정확도 차이 |
| --- | --- | --- |
| 채널 | +28%p (좌채널 vs 단순평균) | +14.4%p |
| Duration | +54%p (repeat vs zero) | +0.69%p |
| Loudness | +44%p (none vs peak) | +1.04%p |

세 분석 모두에서 MFCC 보존도와 정확도의 관계가 다르게 나타남. 사전 점검 없는 결정은 신뢰할 수 없음.

**(5) 정규화는 양방향 작용 (증폭과 축소)**

RMS normalize는 작은 신호를 키우는 것뿐 아니라 큰 신호를 줄이기도 함. P6 구간에서 평균 0.53배 축소되어 오히려 MFCC 보존도가 떨어지는 현상 관찰. 정규화를 "신호를 키우는 것"으로만 이해하면 오해.

**(6) target_rms 값이 결과를 좌우**

RMS normalize는 target 값(본 분석 0.1)에 결과가 의존. target을 전체 평균(0.077)이나 중앙값(0.058)으로 바꾸면 다른 결과 발생 가능. RMS normalize 자체의 본질적 자유도이자 약점.

**(7) Fold별 음량 차이 인지**

fold6 평균 peak 0.449, 클리핑 9.48% vs fold2 평균 peak 0.367, 클리핑 3.15% — 약 3배 차이. 녹음 환경/장비 차이로 추정. Cross-validation 결과 변동의 한 원인일 수 있음.

**(8) drilling이 모든 분석에서 일관되게 식별됨**

| 분석 | 지표 |
| --- | --- |
| SR | MFCC 유사도 0.925 (미달) |
| 비트 깊이 | 저품질 비율 2.30% (최고) |
| 채널 | 모노 비율 14.10% (최고) |
| Duration | mirror 개선폭 +2.2%p |
| Loudness | 정규화 모두에서 정확도 하락 (-1.3%p ~ -2.1%p) |

drilling은 전반적으로 전처리 변경에 민감한 클래스. 본 모델 학습 시 별도 모니터링 가치 있음.

### 5-4. 향후 모델 학습 단계에서 확인할 항목

- children_playing, air_conditioner 정확도가 사전 점검 결과와 일치하는지 (정규화 시 -3%p 이상 악화 재현 여부)
- siren, street_music 정확도가 rms 적용 시 개선되는지
- gun_shot의 클리핑 패턴이 분류에 어떻게 작용하는지 (Confusion matrix 확인)
- jackhammer, car_horn처럼 어려운 클래스(정확도 0.5 수준)가 정규화 변경에 어떻게 반응하는지
- Fold3, Fold8의 정확도 변동 (모든 분석에서 일관되게 낮음)
- 본 모델에서 음량 정보 제거(정규화) 시 새로운 환경 데이터에 대한 일반화 성능

---

## 6. 분석 환경

| 항목 | 값 |
| --- | --- |
| 사용 라이브러리 | librosa, soundfile, numpy, pandas, matplotlib, scikit-learn, scipy |
| 분석 스크립트 | scripts/analyze_loudness_basic.py, scripts/analyze_loudness_normalize.py, scripts/analyze_loudness_methods.py, scripts/analyze_loudness_accuracy.py |
| 결과 데이터 | outputs/loudness_detail.csv, outputs/loudness_by_class.csv, outputs/loudness_by_fold.csv, outputs/loudness_normalize_detail.csv, outputs/loudness_normalize_by_range.csv, outputs/loudness_normalize_by_class.csv, outputs/loudness_methods_detail.csv, outputs/loudness_methods_summary.csv, outputs/loudness_methods_by_range.csv, outputs/loudness_methods_by_class.csv, outputs/loudness_acc_summary.csv, outputs/loudness_acc_by_fold.csv, outputs/loudness_acc_by_class.csv, outputs/features_loudness_comparison/features_none.csv, outputs/features_loudness_comparison/features_peak.csv, outputs/features_loudness_comparison/features_rms.csv |
| 시각화 | outputs/loudness_distribution.png, outputs/loudness_normalize_impact.png, outputs/loudness_methods_comparison.png, outputs/loudness_acc.png |
| 분석 파라미터 | SR=22050, n_mfcc=20, n_fft=2048, hop_length=512, target_rms=0.1, RandomForest n_estimators=100, 10-fold CV |
| 재현 가능성 | random_state=42 |

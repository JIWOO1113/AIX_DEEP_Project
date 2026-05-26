# UrbanSound8K 음향 분류 — 전처리 종합 정리

UrbanSound8K 데이터셋(8,732개 wav, 10개 클래스)에 대해 6가지 전처리 항목을 검증한 결과를 정리한 문서입니다.

각 항목별 상세 분석은 별도 마크다운에 있습니다.

| 항목 | 상세 분석 |
|---|---|
| Sample Rate | [Sample_rate.md](./Sample_rate.md) |
| Bit Depth | [Bit_depth.md](./Bit_depth.md) |
| Channels | [Channels.md](./Channels.md) |
| Duration | [Duration.md](./Duration.md) |
| Loudness | [Loudness.md](./Loudness.md) |
| Silence | [Silence.md](./Silence.md) |

---

## 1. 데이터의 특징

UrbanSound8K는 도시 환경음 10개 클래스(air_conditioner, car_horn, children_playing, dog_bark, drilling, engine_idling, gun_shot, jackhammer, siren, street_music)의 8,732개 wav 파일로 구성된 데이터셋입니다. 학계 표준인 10-fold cross-validation 구조를 따릅니다.

### Sample Rate (얼마나 자주 측정할까)

원본 데이터에 **11종의 sample rate**가 혼재되어 있습니다.

| Sample Rate | 파일 수 | 비율 |
|---|---|---|
| 44,100 Hz | 5,370 | 61.5% |
| 48,000 Hz | 2,502 | 28.7% |
| 96,000 Hz | 610 | 7.0% |
| 24,000 Hz | 82 | 0.9% |
| 16,000 Hz | 45 | 0.5% |
| 22,050 Hz | 44 | 0.5% |
| 11,025 Hz | 39 | 0.4% |
| 192,000 Hz | 17 | 0.2% |
| 8,000 Hz | 12 | 0.1% |
| 11,024 Hz | 7 | 0.1% |
| 32,000 Hz | 4 | 0.05% |

→ 가장 많은 44,100Hz도 전체의 62%에 불과. **하나의 SR로 통일이 필수**.

### Bit Depth (얼마나 정밀하게 기록할까)

| 비트 깊이 | 파일 수 | 비율 | 설명 |
|---|---|---|---|
| PCM_16 | 5,758 | 65.94% | 16비트 정수 (CD 표준) |
| PCM_24 | 2,753 | 31.53% | 24비트 정수 (고음질) |
| FLOAT | 169 | 1.94% | 32비트 부동소수점 |
| PCM_U8 | 43 | 0.49% | 8비트 unsigned (저음질) |
| MS_ADPCM | 8 | 0.09% | 압축 포맷 (구형 Windows) |
| IMA_ADPCM | 1 | 0.01% | 압축 포맷 (구형 멀티미디어) |

→ 6종 혼재. PCM_16과 PCM_24가 97.5%로 절대 다수.

### Channels (마이크가 몇 개인가)

- 스테레오(2채널): **7,993개 (91.54%)**
- 모노(1채널): **739개 (8.46%)**

클래스별 모노 비율 편차도 큽니다 (drilling 14.10% ~ gun_shot 2.94%).

### Duration (길이 분포)

- 정확히 4초: 7,325개 (83.89%)
- 4초 미만: 1,399개 (16.02%) — 패딩 필요
- 4초 초과: 8개 (0.09%) — 자르기 필요

**클래스별 4초 미만 비율이 극단적으로 다릅니다.**

| 클래스 | 4초 미만 비율 |
|---|---|
| **gun_shot** | **95.72%** |
| **car_horn** | **52.68%** |
| dog_bark | 32.50% |
| jackhammer | 19.70% |
| drilling | 19.50% |
| engine_idling | 3.90% |
| siren | 3.44% |
| children_playing | 2.40% |
| air_conditioner | 0.30% |
| street_music | 0.00% |

→ gun_shot, car_horn은 **본질적으로 짧은 소리**.

### Loudness (음량 분포)

- Peak 평균 0.4156, 범위 0.000031 ~ 1.0000 (3자릿수 차이)
- RMS 평균 0.0768
- **클리핑 파일 567개 (6.49%)** — 신호가 최대값에 잘린 상태

클래스별 평균 Peak가 약 3배 차이 납니다.

| 클래스 | 평균 Peak | 클리핑 비율 |
|---|---|---|
| gun_shot | 0.894 | **62.03%** |
| dog_bark | 0.510 | 10.80% |
| jackhammer | 0.497 | 4.90% |
| drilling | 0.465 | 3.50% |
| car_horn | 0.442 | 3.50% |
| street_music | 0.423 | 2.40% |
| engine_idling | 0.373 | 7.10% |
| siren | 0.309 | 1.08% |
| air_conditioner | 0.302 | 0.50% |
| children_playing | 0.276 | 1.80% |

→ gun_shot은 본질적으로 큰 소리(클리핑 62%), children_playing은 가장 조용한 환경음.

### Silence (앞/뒤 무음 분포)

- Total Silence **중앙값 0초**, 75% 분위수도 0초
- 98.32%의 파일이 무음 거의 없음 (앞/뒤 합쳐 0.01초 미만)
- 트리밍이 의미 있는 파일(>0.1초)은 단 0.72%

→ UrbanSound8K는 이미 정제된 데이터셋. 무음 제거 불필요.

### 핵심 발견 — 클래스의 본질이 분류 단서로 작동

분석 결과 **음량과 길이 자체가 클래스의 분류 단서**로 작동하고 있습니다.

- gun_shot: 평균 길이 1.65초 + Peak 0.894 + 클리핑 62% → "짧고 큰 폭발음"
- car_horn: 평균 길이 2.46초 + Peak 0.44 → "짧은 경적음"
- children_playing, air_conditioner: Peak 0.27~0.30 → "조용한 환경음"
- street_music: 100% 정확히 4초 → "완전한 4초 음악"

이 패턴이 전처리 결정에 큰 영향을 미쳤습니다.

### drilling — 모든 분석에서 일관되게 식별된 특이 클래스

| 분석 | drilling의 특이점 |
|---|---|
| Sample Rate | MFCC 유사도 0.925 (22050Hz에서 임계값 미달) |
| Bit Depth | 저품질 비율 2.30% (10개 중 최고) |
| Channels | 모노 비율 14.10% (최고) |
| Duration | mirror padding 개선폭 +2.2%p |
| Loudness | 정규화 모두에서 정확도 하락 |

drilling은 광대역 신호 특성으로 전처리 변경에 일관되게 민감합니다.

---

## 2. 사용할 전처리

각 항목별 결정과 근거입니다. 사전 점검(RandomForest 10-fold CV)으로 실제 분류 정확도까지 검증했습니다.

### Sample Rate — 22050Hz 또는 32000Hz (팀 결정 필요)

**나이퀴스트 정리**: 어떤 주파수 X를 정확히 표현하려면 샘플레이트가 최소 2X여야 한다.

분석 결과 두 선택지가 모두 정당화됩니다.

| 항목 | 선택지 A: 22050Hz | 선택지 B: 32000Hz |
|---|---|---|
| 평균 MFCC 유사도 | 0.970 | **0.989** |
| 임계값 통과 클래스 | 8/10 | **10/10** |
| drilling MFCC | 0.925 (미달) | **0.974** |
| car_horn MFCC | 0.941 (미달) | **0.982** |
| 데이터 크기 (44100Hz 대비) | **50%** | 73% |
| 학습 속도 | **2배 빠름** | 1.4배 빠름 |
| 학계 표준 | **표준** | 비표준 |

- **선택지 A (22050Hz)**: UrbanSound8K 학계 표준이며 데이터 효율 우수. drilling/car_horn에서 MFCC 임계값 미달.
- **선택지 B (32000Hz)**: 모든 클래스 통과하나 비표준.

### Bit Depth — librosa 자동 변환에 위임 (확정)

```python
y, sr = librosa.load(path, sr=22050, mono=True)
# librosa가 자동으로 모든 비트 깊이를 float32, [-1.0, 1.0]로 변환
```

**근거**:
- 6종 비트 깊이 모두 정보 손실 없이 float32로 통일됨 (unique 값 개수 100% 일치)
- 8비트도 MFCC 유사도 0.994로 안전 (임계값 0.95 통과)
- 사전 점검: 저품질 파일 52개 제외 효과는 **+0.5%p에 불과**하여 별도 처리 불필요

### Channels — 모노 통일 또는 스테레오 통일 (팀 결정 필요)

| 항목 | 선택지 A: 모노 통일 | 선택지 B: 스테레오 통일 |
|---|---|---|
| 전체 정확도 | 0.8745 | **0.8973** (+2.28%p) |
| feature 수 | **50** | 104 |
| 데이터 크기 | **약 5MB** | 약 10MB |
| 학습 속도 | **2배 빠름** | 1배 |
| 학계 표준 | **표준** | 비표준 |

**변환 방식은 확정**: 모노 선택 시 **librosa의 단순 평균** `(L+R)/2` 사용.

```python
y, sr = librosa.load(path, sr=22050, mono=True)
```

→ 좌채널만 사용 시 -14.4%p 손실 (car_horn -28%p, children_playing -18%p 등 모든 클래스 악화).

### Duration — zero padding 또는 repeat padding (팀 결정 필요)

데이터의 84%가 정확히 4초이므로 **4초로 통일**이 확정. 짧은 16% 파일의 패딩 방식이 결정 사항.

| 항목 | 선택지 A: zero | 선택지 B: repeat |
|---|---|---|
| 전체 정확도 | 0.6418 | **0.6487** (+0.69%p) |
| 통계적 유의성 | - | p=0.0698 (경계선) |
| car_horn | 0.524 | **0.576** (+5.1%p) |
| jackhammer | 0.520 | **0.562** (+4.2%p) |
| gun_shot | 0.858 | 0.864 (동등) |
| MFCC 보존도 | 0.879 | **1.000** |
| 학계 표준 | **표준** | 비표준 |

- **mirror padding은 제외**: gun_shot에서 -4.3%p 악화 (시간적 비대칭성 손상)
- **짧은 파일 제외 옵션도 제외**: 36~76개 손실 대비 코사인 0.005~0.009 개선으로 효과 미미. 특히 gun_shot의 95.7%가 4초 미만이라 데이터 손실 큼.

### Loudness — 정규화 없음(none) 또는 peak normalize (팀 결정 필요)

| 항목 | 선택지 A: none | 선택지 B: peak |
|---|---|---|
| 전체 정확도 | **0.6418** | 0.6314 (-1.04%p) |
| 통계적 유의성 (vs none) | - | p=0.0744 (경계선) |
| children_playing | **0.741** | 0.717 |
| air_conditioner | **0.458** | 0.421 |
| siren | 0.695 | **0.712** |
| street_music | 0.753 | **0.758** |
| 학계 표준 | 일부 | **표준** |

**음량 자체가 분류 단서**로 작동하고 있어, 정규화 시 조용한 클래스(children_playing -2.4%p, air_conditioner -3.7%p)에서 손해 발생.

- **rms는 별도 선택지에서 제외**: peak와 통계적 차이 없음(p=0.5434), target_rms 자유도 문제로 결정 복잡도만 늘림.
- **클리핑 파일은 그대로 유지**: 클리핑 자체가 gun_shot의 본질적 특징이라 제거 불필요.

### Silence Trimming — 적용하지 않음 (확정)

```python
silence_trim = False
```

**근거**:
- 중앙값과 75% 분위수가 모두 0초 (98.32%가 무음 거의 없음)
- 트리밍 의미 있는 파일은 **0.72%**에 불과
- 트리밍 후 다시 패딩 필요 → 추가 비용만 발생
- gun_shot의 "짧은 폭발 + 무음" 패턴이 분류 단서일 가능성

### 최종 전처리 파이프라인

```python
# 권장 baseline (학계 표준 우선)
y, sr = librosa.load(path, sr=22050, mono=True)
# → SR, Bit Depth, Channels 모두 한 줄로 처리

N_SAMPLES = int(22050 * 4.0)
if len(y) < N_SAMPLES:
    y = np.pad(y, (0, N_SAMPLES - len(y)), mode="constant")  # zero padding
else:
    y = y[:N_SAMPLES]

# 정규화 없음 (none)
# Silence trim 없음

# Feature 추출
mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)  # MFCC 평균 20 + 표준편차 20
# + spectral features (centroid, bandwidth, rolloff, contrast, zcr, rms) 평균/표준편차
# 총 54개 feature
```

**현재 baseline 정확도: 0.6418 (RandomForest, 10-fold CV)**

---

## 3. 사용할 모델 (5개 후보)

`02_build_features.py`로 두 가지 형태의 feature를 추출하여 모델별로 활용합니다.

**결과물 1**: ML feature 54개 (MFCC 평균/표준편차 + spectral features)  
→ 사용 모델: RandomForest, SVM, XGBoost

**결과물 2**: Mel spectrogram (128 × 시간, 2D 이미지 형태)  
→ 사용 모델: CNN, CRNN

```
주파수 (높음) ▓▓▒░ ░▒▓▓ ▒░  
              ▓▓▒░░ ░░▒▓▓▒░░
              ▓▒░ ░▒▓▒░
              ▒░ ░▒░  
              ░ ░  
주파수 (낮음) ─────────────→ 시간
```

### 모델 1: RandomForest (Baseline)

- **입력**: ML feature 54개
- **현재 정확도**: 0.6418 (10-fold CV)
- **선정 이유**: 본 프로젝트의 모든 사전 점검에서 사용된 baseline. 전처리 결정의 정당성을 검증한 모델이므로 본 모델 학습에서도 reference로 유지.

### 모델 2: XGBoost (Gradient Boosting)

- **입력**: ML feature 54개
- **장점**: tabular 데이터에서 일반적으로 RandomForest보다 정확도 높음. 클래스 불균형 처리 옵션 제공.
- **선정 이유**: car_horn(429개), gun_shot(374개)의 클래스 불균형 처리에 강점.

### 모델 3: SVM (RBF Kernel)

- **입력**: ML feature 54개 (StandardScaler 적용 필수)
- **장점**: 고차원 feature 공간에서 강력. UrbanSound8K 학계 논문에서 자주 비교군으로 사용.
- **선정 이유**: 학계 표준 비교군. ML 계열의 한계 측정.

### 모델 4: 2D CNN (Mel spectrogram)

- **입력**: Mel spectrogram (128 × 시간)
- **구조**: 3~4층 Conv + BatchNorm + MaxPool + Dense
- **장점**: 음향의 시간-주파수 패턴을 직접 학습. UrbanSound8K SOTA 논문 대부분이 CNN 기반.
- **선정 이유**: ML feature 기반 모델들의 정확도 0.6~0.7 한계를 넘기 위한 핵심 모델.

### 모델 5: CRNN (CNN + RNN)

- **입력**: Mel spectrogram
- **구조**: CNN으로 주파수 패턴 추출 → LSTM/GRU로 시간 패턴 학습
- **장점**: 시간 종속성이 중요한 신호(siren, dog_bark의 반복 패턴, jackhammer의 리듬, gun_shot의 폭발+잔향)에 강점.
- **선정 이유**: 본 데이터 분석에서 반복적/시퀀스 패턴이 중요한 클래스가 다수 존재. CNN으로 부족한 부분을 보완.

### 모델 비교 계획

| 모델 | 입력 | 비교 목적 |
|---|---|---|
| RandomForest | ML feature | Baseline (사전 점검 reference) |
| XGBoost | ML feature | ML 계열 최강 비교 |
| SVM (RBF) | ML feature | 학계 표준 비교군 |
| 2D CNN | Mel spectrogram | SOTA 비교 |
| CRNN | Mel spectrogram | 시간 패턴 클래스 강화 |

모든 모델은 **UrbanSound8K 표준 10-fold cross-validation**으로 평가하며, fold별 정확도 변동성과 클래스별 정확도를 함께 측정합니다.

### 모델 평가 시 별도 모니터링할 항목

본 프로젝트의 6가지 전처리 분석에서 일관되게 식별된 항목:

1. **car_horn**: 현재 정확도 0.78로 최저. 채널 변환/길이 패딩 방식에 가장 민감.
2. **drilling**: 모든 전처리 분석에서 특이 클래스로 식별. 광대역 신호 특성.
3. **gun_shot**: 클리핑 62%, 4초 미만 96%, 평균 길이 1.65초. 짧은 폭발음 분류 패턴이 본 모델에서도 유지되는지.
4. **children_playing, air_conditioner**: 평균 Peak 0.27~0.30의 조용한 클래스. 정규화 적용 시 -3%p 이상 악화 재현 여부.
5. **Fold 편차**: fold3, fold6, fold8이 모든 분석에서 일관되게 낮음. Cross-validation 결과 fold별 변동성 추적.

---

## 4. 핵심 인지 사항

본 프로젝트의 6가지 분석을 종합하여 얻은 4가지 핵심 교훈입니다.

### (1) MFCC 보존도와 실제 분류 정확도의 관계는 도메인마다 다름

| 분석 | MFCC 보존도 차이 | 실제 정확도 차이 | 관계 |
|---|---|---|---|
| Channels | 0.998 (거의 동일) | 14.4%p | **차이 숨김** |
| Duration | 코사인 0.879 → 1.000 (큰 차이) | 0.69%p | **차이 부풀림** |
| Loudness | 0.875 → 1.000 (큰 차이) | 1.04%p | 순위 일치 |

→ 모든 전처리 결정은 이론적 지표(MFCC)와 **실제 분류 정확도(RandomForest 사전 점검)**를 함께 검증해야 합니다.

### (2) 음량과 길이 자체가 분류 단서로 작동

UrbanSound8K에서 음량과 길이는 단순 형식이 아니라 **클래스의 본질**을 담고 있습니다.

- gun_shot: 평균 Peak 0.89, 평균 길이 1.65초 → "짧고 큰 폭발음"
- children_playing: 평균 Peak 0.27 → "조용한 환경음"

이 정보를 정규화로 제거하거나 padding 방식으로 변형하면 분류 정확도가 변동합니다.

### (3) 학계 표준 vs 정확성의 trade-off

대부분의 전처리 결정에서 학계 표준과 정확도 사이의 균형이 필요합니다.

| 항목 | 학계 표준 | 정확도 우선 |
|---|---|---|
| Sample Rate | 22050Hz | 32000Hz (+drilling/car_horn 통과) |
| Channels | 모노 | 스테레오 (+2.28%p) |
| Duration | zero | repeat (+0.69%p) |
| Loudness | peak | none (+1.04%p) |

본 프로젝트는 선택지 A(표준) vs 선택지 B(정확도) 형태로 정리하여 팀이 우선순위에 따라 결정할 수 있도록 했습니다.

### (4) 본 모델 학습 시 별도 모니터링 필요 클래스

- **car_horn**: 가장 어려운 클래스 (정확도 0.78). 변환 방식에 가장 민감.
- **drilling**: 모든 전처리 분석에서 식별. 광대역 신호 특성.
- **gun_shot**: 클리핑 패턴 + 짧은 길이 패턴이 본 모델 분류에 어떻게 작용하는지.

---

## 5. 분석 환경

| 항목 | 값 |
|---|---|
| 사용 라이브러리 | librosa, soundfile, numpy, pandas, matplotlib, scikit-learn, scipy |
| 분석 데이터셋 | UrbanSound8K (8,732개 wav, 10개 클래스) |
| 평가 방법 | RandomForest + UrbanSound8K 표준 10-fold cross-validation |
| 분석 파라미터 | SR=22050, n_mfcc=20, n_fft=2048, hop_length=512 |
| 재현 가능성 | random_state=42 |

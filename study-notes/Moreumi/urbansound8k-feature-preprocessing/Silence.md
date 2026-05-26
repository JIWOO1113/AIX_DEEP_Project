# Silence

## 결론 요약

UrbanSound8K는 이미 정제된 데이터셋으로, 무음(silence) 부분이 거의 없다. **Silence trimming은 불필요**하며 현재 설정(`silence_trim=false`)을 유지한다.

---

## 1. 무음 분포 측정

### 1-1. FACT — 데이터셋이 얼마나 무음을 포함하는지 측정한다

UrbanSound8K 8,732개 파일에 대해 앞/뒤 무음 길이를 측정해서 silence trimming의 필요성을 검증한다.

### 1-2. 분석 기법

`scripts/analyze_silence_basic.py`에서:

- 전체 8,732개 파일에 대해 `librosa.effects.trim(y, top_db=60)` 적용
- 측정 항목:
  * leading_silence: 신호 시작 전 무음 길이 (초)
  * trailing_silence: 신호 끝 후 무음 길이 (초)
  * total_silence: 앞 + 뒤 합계
  * signal_duration: 트리밍 후 남는 신호 길이

top_db=60은 librosa 기본값으로 매우 보수적(거의 들리지 않는 부분만 무음으로 간주).

### 1-3. 결과

#### 전체 분포

| 지표 | 값 |
| --- | --- |
| Total Silence 평균 | 0.0027초 (2.7ms) |
| Total Silence 중앙값 | **0.0000초** |
| Total Silence 75% 분위수 | **0.0000초** |
| Total Silence 최대값 | 1.14초 |

#### Total Silence 구간별 분포

| 구간 | 파일 수 | 비율 |
| --- | --- | --- |
| **S1 (~0.01s)** | **8,584** | **98.32%** |
| S2 (0.01~0.1s) | 85 | 0.97% |
| S3 (0.1~0.5s) | 53 | 0.61% |
| S4 (0.5~1s) | 8 | 0.09% |
| S5 (1~2s) | 2 | 0.02% |
| S6 (2s+) | 0 | 0.00% |

#### 트리밍 영향이 있는 파일

| 기준 | 파일 수 | 비율 |
| --- | --- | --- |
| total_silence > 0.01초 | 148개 | 1.69% |
| total_silence > 0.1초 | 63개 | 0.72% |
| total_silence > 0.5초 | 10개 | 0.11% |

#### 클래스별 무음 (Total Silence 평균 내림차순)

| 클래스 | total_silence 평균 | 무음 비율 |
| --- | --- | --- |
| gun_shot | 0.0278초 | 1.78% |
| dog_bark | 0.0112초 | 0.43% |
| car_horn | 0.0012초 | 0.04% |
| street_music | 0.0005초 | 0.01% |
| children_playing | 0.0005초 | 0.01% |
| siren | 0.0001초 | 0.00% |
| drilling | 0.0001초 | 0.00% |
| air_conditioner | 0.0000초 | 0.00% |
| engine_idling | 0.0000초 | 0.00% |
| jackhammer | 0.0000초 | 0.00% |

### 1-4. 시각화

![무음 분포](https://github.com/JIWOO1113/AIX_DEEP_Project/raw/main/study-notes/Moreumi/urbansound8k-feature-preprocessing/silence_distribution.png)

### 1-5. 결과 해석

- 98.32%의 파일이 무음이 거의 없음 (S1 구간)
- 중앙값과 75% 분위수가 모두 0초 — 절반 이상의 파일이 무음이 정확히 0
- 트리밍이 의미 있는 파일(>0.1초)은 단 0.72%
- gun_shot과 dog_bark에서만 일부 무음 존재 (이산적 이벤트 신호 특성)
- 나머지 5개 클래스(siren, drilling, jackhammer, air_conditioner, engine_idling)는 연속 신호로 무음이 사실상 0

---

## 2. 결정

### 2-1. silence_trim = false (유지)

#### 결정 근거

1. **데이터셋이 이미 정제됨** — UrbanSound8K는 원본 오디오에서 4초 구간을 슬라이스한 데이터셋으로, 슬라이스 과정에서 흥미로운 신호가 있는 구간이 선별됨
2. **트리밍 효과 미미** — 영향받는 파일이 단 1.69%
3. **추가 작업 비용 발생** — 트리밍 후 길이가 짧아져 다시 패딩 필요 (Duration 분석 다시 고려해야 함)
4. **무음이 분류 단서일 가능성** — gun_shot의 짧은 폭발 + 무음 패턴은 분류에 도움될 수 있음

#### 본 분석을 사전 점검까지 진행하지 않은 이유

이전 분석들(채널, Duration, Loudness)은 처리 방식에 따라 결과가 달라질 여지가 있어 사전 점검(분류 정확도 비교)을 진행했다. Silence는 그렇지 않다:

- 98.32% 파일이 트리밍해도 변화 없음
- 트리밍 적용해도 전체 8,732개 중 148개만 영향 — 통계적 유의 차이가 나올 가능성 매우 낮음
- 사전 점검에 1~2시간 소요되지만 결과가 명백히 예측 가능

본 모델 학습 시 의문이 생기면 그때 추가 검증 가능.

### 2-2. 인지하고 가야 할 사항

**(1) gun_shot, dog_bark만 무음 일부 존재**

두 클래스는 이산적 이벤트(폭발음, 짖음) 패턴이라 사이사이 무음이 있음. 본 모델 학습 시 이 두 클래스의 정확도가 다른 분석에서도 특이하게 나오면 무음 패턴이 원인일 가능성 고려.

**(2) 다른 데이터셋에서는 다를 수 있음**

본 결론은 UrbanSound8K 한정. ESC-50, AudioSet 등 다른 환경음 데이터셋은 슬라이스 방식이 다를 수 있으므로 동일한 결론을 적용하기 전에 분포를 확인해야 함.

**(3) top_db 파라미터 의존성**

본 분석은 top_db=60(librosa 기본값) 기준. 더 공격적인 값(top_db=30, 20)에서는 무음 비율이 다르게 측정될 수 있으나, 환경음 분류에서는 보수적 값이 적절함.

---

## 3. 분석 환경

| 항목 | 값 |
| --- | --- |
| 사용 라이브러리 | librosa, numpy, pandas, matplotlib |
| 분석 스크립트 | scripts/analyze_silence_basic.py |
| 결과 데이터 | outputs/silence_detail.csv, outputs/silence_by_class.csv, outputs/silence_by_fold.csv |
| 시각화 | outputs/silence_distribution.png |
| 분석 파라미터 | SR=22050, top_db=60 (librosa 기본값) |
| 재현 가능성 | random_state=42 |

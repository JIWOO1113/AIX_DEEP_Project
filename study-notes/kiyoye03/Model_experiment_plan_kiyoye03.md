# UrbanSound8K 전처리 결과 기반 모델 실험 설계

## 1. 정리 목적

이번 문서는 UrbanSound8K 데이터셋을 실제 모델 실험으로 연결하기 전에, 전처리에서 정한 기준들이 모델 입력 형태와 모델 선택에 어떻게 이어지는지 정리하기 위한 것이다.

회의에서 이야기했듯이 wav 파일이라고 해서 모두 같은 조건을 갖는 것은 아니다. 파일마다 sample rate, channel 수, bit depth, duration 등이 다를 수 있고, 이런 차이가 그대로 남아 있으면 모델이 받는 입력 형태도 일정하지 않다. 따라서 모델을 돌리기 전에 어떤 항목을 통일할지, 통일한 뒤 어떤 feature를 만들지 먼저 정리할 필요가 있다.

여기서는 각 전처리 항목 자체를 깊게 분석하기보다는, 그 전처리가 모델 실험에서 어떤 의미를 가지는지에 초점을 두었다. 아직 모든 선택이 최종 결정된 것은 아니므로, 다음 모델링 단계에서 사용할 baseline과 비교 후보를 정리하는 정도로 보면 된다.

---

## 2. 원본 wav 데이터에서 확인해야 할 요소

모델은 wav 파일을 사람이 듣는 방식으로 이해하는 것이 아니라, 일정한 길이와 숫자 형태로 정리된 입력을 받는다. 따라서 오디오 파일을 모델에 넣기 전에 아래와 같은 요소들을 먼저 확인해야 한다.

| 확인 항목 | 의미 | 모델 입력과의 관계 |
|---|---|---|
| sample rate | 1초 동안 소리를 몇 번 측정했는지 | 시간축 sample 수와 주파수 표현 범위에 영향을 준다 |
| channel 수 | mono / stereo 여부 | 입력 shape을 맞추기 위해 통일이 필요하다 |
| bit depth | 소리를 얼마나 정밀하게 저장했는지 | 실제 학습에서는 보통 float32 형태로 변환되어 사용된다 |
| duration | 오디오 파일의 길이 | 모델 입력 길이를 맞추기 위해 padding 또는 trimming이 필요하다 |
| salience | 소리가 foreground인지 background인지 | 소리의 뚜렷함이 class별 난이도에 영향을 줄 수 있다 |
| clipping | 파형이 최대 진폭에서 잘렸는지 | 제거 대상이라기보다 일부 소리의 특성으로 볼 수도 있다 |

이 중에서 sample rate, channel, duration은 모델 입력의 형태에 직접적인 영향을 준다. bit depth는 원본 저장 방식의 차이이지만, librosa로 읽을 경우 float32 배열로 변환되기 때문에 별도의 변환 코드를 두기보다는 읽는 과정에서 처리되는 부분으로 볼 수 있다.

---

## 3. Baseline 전처리 설정안

처음부터 모든 전처리 옵션을 복잡하게 비교하기보다는, 우선 하나의 baseline을 정해 모델을 돌려보는 것이 필요하다. baseline은 구현이 단순하고, 이후 다른 설정과 비교하기 쉬운 방향으로 잡는 것이 좋다고 보았다.

| 항목 | baseline 설정 | 이유 |
|---|---|---|
| sample rate | 22050Hz | 계산량을 줄일 수 있고, 오디오 분류에서 자주 사용되는 값이다 |
| channel | mono | 입력 구조가 단순하고 기본 feature 추출 코드와 잘 맞는다 |
| bit depth | librosa.load의 float32 변환 사용 | 별도 변환 없이 수치 범위를 통일할 수 있다 |
| duration | 4초로 통일 | UrbanSound8K 데이터 특성과 맞고, 입력 길이를 고정할 수 있다 |
| padding | zero padding을 기본값으로 두고 repeat padding도 비교 | zero는 단순하고 표준적이며, repeat은 짧은 파일에서 비교해볼 만하다 |

기본적인 전처리 코드는 아래와 같은 형태로 생각할 수 있다.

```python
import librosa
import numpy as np

TARGET_SR = 22050
TARGET_DURATION = 4
TARGET_LENGTH = TARGET_SR * TARGET_DURATION

y, sr = librosa.load(path, sr=TARGET_SR, mono=True)

if len(y) < TARGET_LENGTH:
    y = np.pad(y, (0, TARGET_LENGTH - len(y)))
else:
    y = y[:TARGET_LENGTH]
```

위 코드는 가장 단순한 baseline 형태이다. 실제 실험에서는 sample rate나 padding 방식을 바꿔가며 모델 성능 차이를 확인할 수 있다.

---

## 4. 비교해볼 수 있는 전처리 옵션

baseline 하나만 사용하면 전처리 선택이 모델 결과에 어떤 영향을 주는지 알기 어렵다. UrbanSound8K는 class별 소리 특성이 꽤 다르기 때문에, 전체 accuracy만 보면 놓치는 부분이 생길 수 있다. 따라서 baseline을 기준으로 몇 가지 alternative 설정을 비교해보는 것이 좋다.

| 비교 항목 | baseline | alternative | 확인하고 싶은 점 |
|---|---|---|---|
| sample rate | 22050Hz | 32000Hz | drilling, car_horn처럼 고주파나 순음 특성이 있는 class에서 차이가 나는지 |
| channel | mono | stereo feature | 좌우 채널 정보가 분류 성능에 도움이 되는지 |
| padding | zero padding | repeat padding | 짧은 파일이 많은 gun_shot, car_horn에서 성능 차이가 있는지 |
| feature 형태 | MFCC/statistical feature | Mel-spectrogram | 전통 ML 모델과 CNN 계열 모델의 차이 |
| salience 처리 | 전체 사용 | foreground/background 별도 분석 | background sample이 class별 성능을 낮추는지 |

이 옵션들을 한 번에 모두 바꾸면 어떤 요인이 성능 차이를 만든 것인지 해석하기 어렵다. 따라서 우선 baseline 모델을 만들고, 성능이 낮게 나온 class를 중심으로 하나씩 조건을 바꿔보는 방식이 더 적절할 것 같다.

---

## 5. Feature extraction 방향

전처리가 끝난 wav 파일도 그대로 모델에 들어가는 것은 아니다. 모델이 사용할 수 있도록 feature 형태로 변환해야 한다. 이번 프로젝트에서는 크게 두 가지 방향을 생각할 수 있다.

### 5-1. ML feature vector 방식

첫 번째는 wav 파일에서 MFCC와 spectral feature를 뽑아 하나의 숫자 벡터로 만드는 방식이다.

예상 pipeline은 다음과 같다.

```text
wav file
→ resampling / channel 통일 / duration 통일
→ MFCC mean, MFCC std, spectral features 추출
→ feature table 생성
→ RandomForest, SVM, XGBoost, MLP 등에 입력
```

이 방식은 입력 데이터가 비교적 작고, 모델 학습이 빠르다는 장점이 있다. 또한 RandomForest나 SVM 같은 모델을 baseline으로 사용하기 좋다.

다만 시간에 따른 세부적인 변화가 평균과 표준편차로 압축되기 때문에, 소리의 시간적 패턴이 중요한 경우에는 정보가 일부 줄어들 수 있다.

### 5-2. Mel-spectrogram 방식

두 번째는 wav 파일을 Mel-spectrogram으로 바꾸는 방식이다. Mel-spectrogram은 시간과 주파수 정보를 2차원 형태로 표현하기 때문에 이미지와 비슷한 방식으로 다룰 수 있다.

예상 pipeline은 다음과 같다.

```text
wav file
→ resampling / channel 통일 / duration 통일
→ Mel-spectrogram 생성
→ 2D tensor 형태로 변환
→ CNN 또는 CRNN 계열 모델에 입력
```

이 방식은 시간-주파수 패턴을 비교적 잘 유지할 수 있어서 CNN 모델과 잘 맞는다. 대신 feature vector 방식보다 계산량과 메모리 사용량이 커질 수 있다.

따라서 처음에는 ML feature vector 기반 모델로 baseline을 만들고, 이후 Mel-spectrogram 기반 CNN과 비교하는 흐름이 현실적이라고 생각한다.

---

## 6. 모델 후보

다음 단계에서 비교할 모델은 입력 형태에 따라 나눠볼 수 있다.

| 모델 | 입력 형태 | 역할 |
|---|---|---|
| Random Forest | MFCC + spectral feature vector | baseline 모델 |
| SVM | MFCC + spectral feature vector | 전통적인 분류 모델 비교 |
| XGBoost | MFCC + spectral feature vector | tree-based 모델 성능 비교 |
| MLP | feature vector | 간단한 neural network 기준 |
| CNN | Mel-spectrogram | 2D 입력을 활용한 deep learning 모델 |

Random Forest는 구현이 비교적 쉽고 feature importance도 확인할 수 있어서 baseline으로 적합하다. SVM과 XGBoost는 같은 feature vector를 사용하면서 모델 구조에 따른 차이를 비교하기 좋다. MLP는 feature vector를 사용하는 간단한 neural network로 볼 수 있고, CNN은 Mel-spectrogram을 사용할 때 비교 대상으로 둘 수 있다.

시간이 부족하다면 우선 Random Forest, SVM, XGBoost를 먼저 돌리고, 이후 CNN을 추가하는 방식도 가능하다.

---

## 7. 평가 방식

모델 비교에서는 전체 accuracy만 보는 것으로는 부족할 수 있다. UrbanSound8K는 class별 데이터 수와 소리 특성이 다르기 때문에, 어떤 class를 잘 맞추고 어떤 class에서 헷갈리는지 함께 봐야 한다.

기본적으로 사용할 수 있는 지표는 다음과 같다.

| 지표 | 확인 내용 |
|---|---|
| Accuracy | 전체적으로 얼마나 맞췄는지 |
| Precision | 특정 class로 예측한 것 중 실제로 맞은 비율 |
| Recall | 실제 특정 class 중 모델이 찾아낸 비율 |
| F1-score | precision과 recall의 균형 |
| Confusion matrix | 어떤 class끼리 혼동되는지 |

특히 confusion matrix는 이 프로젝트에서 중요할 것 같다. 예를 들어 drilling과 jackhammer는 둘 다 기계음 또는 공사장 소리 계열이라 서로 헷갈릴 가능성이 있고, siren과 car_horn도 도로 환경음이라는 점에서 혼동될 수 있다.

---

## 8. 모델 결과에서 따로 확인할 class

전처리 분석과 데이터 특성을 고려하면, 모델 학습 후 아래 class들은 따로 확인하는 것이 좋다.

| class | 확인 이유 |
|---|---|
| drilling | sample rate, channel, bit depth 분석에서 여러 번 특이점이 나타남 |
| car_horn | sample rate와 channel 처리에 민감할 가능성이 있음 |
| gun_shot | 짧은 파일 비율이 높아 padding 방식의 영향을 받을 수 있음 |
| jackhammer | drilling과 혼동될 가능성이 있음 |
| street_music | 음악적/주파수 패턴이 중요할 수 있고 stereo 정보의 영향을 받을 수 있음 |

이 class들은 전체 accuracy가 괜찮게 나와도 개별적으로 낮은 성능을 보일 수 있다. 따라서 모델별 전체 성능과 함께 class별 성능을 같이 확인해야 한다.

---

## 9. 실험 진행 순서 제안

현재 단계에서는 모든 옵션을 동시에 바꾸기보다 아래 순서로 진행하는 것이 좋을 것 같다.

1. baseline 전처리 설정으로 ML feature vector 생성  
   - sample rate: 22050Hz  
   - channel: mono  
   - duration: 4초  
   - padding: zero padding  
   - feature: MFCC + spectral features  

2. Random Forest로 baseline 성능 확인  
   - 전체 accuracy  
   - class별 accuracy  
   - confusion matrix 확인  

3. 같은 feature로 SVM, XGBoost, MLP 비교  

4. 성능이 낮은 class 확인  
   - car_horn  
   - drilling  
   - gun_shot  
   - jackhammer 등  

5. 필요하면 전처리 옵션을 하나씩 변경  
   - 22050Hz vs 32000Hz  
   - zero padding vs repeat padding  
   - mono feature vs stereo feature  

6. Mel-spectrogram 기반 CNN 모델 비교  

이렇게 진행하면 전처리 선택과 모델 성능 사이의 관계를 조금 더 명확하게 볼 수 있을 것 같다.

---

## 10. 정리

이번 프로젝트에서 전처리는 단순히 wav 파일을 읽기 위한 준비 단계가 아니라, 모델이 어떤 정보를 보게 할지 결정하는 과정에 가깝다. 같은 데이터셋을 사용하더라도 sample rate, channel, duration, padding 방식에 따라 모델이 받는 입력은 달라질 수 있다.

현재 단계에서 가장 먼저 시도해볼 만한 baseline은 22050Hz, mono, 4초 길이 통일, librosa 기반 float32 변환, MFCC/spectral feature 추출이다. 이 설정은 구현이 비교적 간단하고, 빠르게 baseline 결과를 얻기에 적합하다.

그 다음에는 전체 accuracy만 확인하는 것보다 class별 결과를 함께 보는 것이 중요하다. 특히 drilling, car_horn, gun_shot처럼 전처리 방식에 민감할 수 있는 class는 따로 확인해야 한다. 만약 특정 class에서 성능이 낮게 나오면 sample rate, padding, channel 설정을 하나씩 바꿔보면서 원인을 확인할 수 있을 것이다.

최종적으로는 ML feature 기반 모델과 Mel-spectrogram 기반 CNN을 함께 비교하면, 단순 feature 기반 접근과 deep learning 접근이 UrbanSound8K 데이터에서 어떻게 다르게 작동하는지 볼 수 있을 것 같다.

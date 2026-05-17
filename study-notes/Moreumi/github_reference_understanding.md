**1. README.md 파일 <br>**

| 구분         | 내용                                                                |
| ---------- | ----------------------------------------------------------------- |
| repo 목적    | UrbanSound8K를 이용한 sound classification 예제 프로젝트                    |
| 사용 도구      | Librosa, ffmpeg, CNN, Keras, XGBoost, Random Forest               |
| 데이터셋 설명    | UrbanSound8K 데이터셋 구조 설명                                           |
| feature 추출 | librosa로 오디오를 읽고 MFCC, spectral feature, energy 등을 추출할 수 있다고 설명   |
| 모델링 방향     | spectrogram을 이미지처럼 보고 CNN으로 분류하는 방식 설명                            |
| MFCC 설명    | 음성/오디오 인식에서 MFCC가 왜 feature로 쓰이는지 설명                              |
| 참고 링크      | 오디오 feature, pyAudioAnalysis, Urban Sound Classification 관련 자료 링크 |

<br>
오디오 분석을 위해 `librosa` 라이브러리를 사용<br>
feature 추출 방향<br>
: MFCC, spectral features, energy 등 여러 오디오 feature를 추출할 수 있으며, <br>
먼저 MFCC만 사용해 모델을 만들고 정확도를 확인한 뒤 <br>
다른 feature나 여러 feature 조합을 실험한다.<br>

모델링 방식<br>
spectrogram을 이미지처럼 보고 CNN으로 분류하는 방법 사용
README는 비슷한 소리 유형은 비슷한 spectrogram 형태를 가지므로, 
CNN이 spectrogram 이미지의 패턴을 학습해 소리를 분류할 수 있다고 설명한다.<br>
MFCC는 오디오 신호에서 분류에 유용한 특징을 추출하기 위한 대표적인 feature이며, 
배경 소음이나 감정 등 불필요한 정보를 줄이고 소리를 구분하는 데 도움이 되는 특징을 표현하는 방식으로 설명된다.<br>

---

**2. DataCollection.py 파일 <br>
**UrbanSound8K 데이터셋의 오디오 파일을 불러오고, wav 파일의 기본 정보를 확인하거나 feature를 추출하기 위한 보조 코드 파일이다.
이 파일의 핵심 역할은 metadata에 있는 `slice_file_name`과 `fold` 정보를 이용해 실제 오디오 파일 경로를 만들고, 해당 wav 파일의 sampling rate, bit depth, channel 수, duration, class label 등을 확인하는 것이다.
<br>

#### 주요 라이브러리

| 라이브러리 | 역할 |
|---|---|
| `os` | 파일 경로 생성 |
| `struct` | wav 파일의 바이너리 header 정보 해석 |
| `matplotlib.pyplot` | waveform 시각화 |
| `IPython.display` | 노트북에서 오디오 재생 |
| `pandas` | metadata CSV 처리 |
| `numpy` | 배열 연산 |
| `librosa` | 오디오 로드 및 feature 추출 |

#### 주요 함수

| 함수명 | 역할 |
|---|---|
| `path_class(data, filename)` | metadata에서 파일명을 찾아 실제 오디오 경로와 class label을 반환 |
| `wav_plotter(full_path, class_label)` | wav 파일의 sampling rate, bit depth, channel 수, duration 등을 출력하고 waveform을 시각화 |
| `wav_fmt_parser(file_name)` | wav 파일 header를 읽어 channel 수, sampling rate, bit depth를 반환 |
| `sound_to_mat(fullpath)` | librosa로 오디오를 읽고 MFCC feature를 추출하려는 함수이나, 현재는 주석 처리되어 있음 |
| `parser(row)` | Mel Spectrogram feature를 추출하려는 함수이나, 현재는 주석 처리되어 있음 |

#### 코드 흐름 해석

1. `path_class()` 함수는 metadata CSV에서 특정 `slice_file_name`을 찾는다.  
2. 해당 파일이 들어 있는 `fold` 번호를 이용해 실제 파일 경로를 만든다.  
3. 동시에 해당 오디오의 `class` 값을 가져온다.  
4. `wav_plotter()` 함수는 실제 wav 파일을 읽고 기본 오디오 정보를 출력한다.  
5. 출력 정보에는 sampling rate, bit depth, channel 수, duration, sample 수, class label이 포함된다.  
6. 이후 waveform을 그래프로 시각화하고, 노트북 환경에서 오디오를 재생할 수 있도록 한다.  
7. 주석 처리된 함수들은 MFCC나 Mel Spectrogram feature를 추출해 모델 입력 데이터로 변환하려는 목적을 가진다.
8. 단, 현재 코드에는 일부 문제가 있으므로 필요한 함수만 수정해서 사용하는 것을 추천한다.<br>
---

**3. Creating_Dataset _MFCC.ipynb 파일 <br>**
UrbanSound8K의 원본 wav 오디오 파일을 모델 학습에 사용할 수 있는 숫자 feature로 변환하는 코드이다.
각 wav 파일을 모델이 학습할 수 있는 feature vector로 변환하는 역할을 한다. <br>

#### 코드 흐름

1. `UrbanSound8K/metadata/UrbanSound8K.csv` 파일을 읽어 metadata를 불러온다.
2. metadata의 행 수를 확인한다. 전체 데이터 수는 8,732개이다.
3. `(8732, 2)` 형태의 빈 배열 `dataset`을 만든다.
4. 각 행마다 `slice_file_name`을 이용해 실제 wav 파일 경로와 class label을 가져온다.
5. `librosa.load()`로 wav 파일을 읽는다.
6. `librosa.feature.mfcc()`를 이용해 MFCC feature를 추출한다.
7. 시간축 방향으로 평균을 내어 각 오디오 파일을 하나의 feature vector로 요약한다.
8. `dataset[i, 0]`에는 feature를 저장하고, `dataset[i, 1]`에는 class label을 저장한다.
9. 최종 결과를 `.npy` 파일로 저장한다.
10. 저장된 파일을 다시 불러와 shape이 `(8732, 2)`인지 확인한다.
<br>
MFCC는 소리를 더 압축해서 요약한 특징값 <br>
Mel Spectrogram은 오디오를 이미지처럼 바꾼 것<br>

---
4. Creating_Dataset_Mel.ipynb 파일<br>
UrbanSound8K의 원본 wav 오디오 파일을 Mel Spectrogram feature로 변환 <br>
시간에 따라 주파수 성분이 어떻게 변하는지를 나타내는 feature이다. <br>
각 오디오 파일에서 Mel Spectrogram을 추출한 뒤, 시간축 평균을 내어 하나의 feature vector로 요약한다. <br>

#### 코드 흐름

1. `progressbar` 패키지를 설치한다.
2. 필요한 라이브러리를 불러온다.
3. `UrbanSound8K/metadata/UrbanSound8K.csv` 파일을 읽어 metadata를 불러온다.
4. metadata의 전체 행 수를 확인한다. 이 값은 8,732개이다.
5. `(8732, 2)` 형태의 빈 배열 `dataset`을 만든다.
6. 각 행마다 `slice_file_name`을 이용해 실제 WAV 파일 경로와 class label을 가져온다.
7. `librosa.load()`로 WAV 파일을 읽는다.
8. `librosa.feature.melspectrogram()`을 이용해 Mel Spectrogram feature를 추출한다.
9. 추출한 Mel Spectrogram에 대해 시간축 평균을 내어 하나의 feature vector로 만든다.
10. `dataset[i, 0]`에는 feature를 저장하고, `dataset[i, 1]`에는 class label을 저장한다.
11. 최종 결과를 `dataset_melspectrogram.npy` 파일로 저장한다.
12. 저장한 `.npy` 파일을 다시 불러와 shape이나 특정 label, feature shape을 확인한다.
13. 마지막에는 `Real_world_prob.wav` 파일을 예시로 불러와 실제 오디오 정보를 확인하고 재생한다.

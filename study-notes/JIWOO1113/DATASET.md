UrbanSound8K는 도시 환경에서 발생하는 소리를 분류하기 위한 오디오 데이터셋이다.
....

## 1. 전체 디렉토리 구조 

```text
UrbanSound8K/
├── FREESOUNDCREDITS.txt
├── UrbanSound8K_README.txt
├── audio/
│   ├── fold1/
│   ├── fold2/
│   ├── fold3/
│   ├── fold4/
│   ├── fold5/
│   ├── fold6/
│   ├── fold7/
│   ├── fold8/
│   ├── fold9/
│   └── fold10/
└── metadata/
    └── UrbanSound8K.csv
```


## 2. FREESOUNDCREDITS.txt

UrbanSound8K에 포함된 원본 오디오의 출처 정보를 담고 파일


## 3. UrbanSound8K_README.txt

이 데이터셋은 도시 환경음 10개 클래스에 속하는 라벨링된 소리 클립 8732개로 구성되어 있다.
각 클립의 길이는 4초 이하이다.

포함된 클래스는 다음과 같다.
```text
air_conditioner
car_horn
children_playing
dog_bark
drilling
engine_idling
gun_shot
jackhammer
siren
street_music
```
파일들은 자동 소리 분류 실험 결과를 재현하고 비교하기 쉽도록 10개의 fold로 미리 나뉘어 있다.

샘플링 레이트, 비트 깊이, 채널 수는 Freesound에 업로드된 원본 파일의 설정을 그대로 따른다. 따라서 이 값들은 파일마다 다를 수 있다. 
즉, 모든 오디오 파일이 동일한 샘플링 레이트나 동일한 채널 수를 가지는 것은 아니다. 
모델 학습에 사용하려면 전처리 과정에서 샘플링 레이트, 길이, 채널 수 등을 통일하는 것이 일반적이다.
 - sampling rate: 1초동안 소리를 몇 번 측정했는가? (ex. 44100 Hz -> 1초에 44100개의 소리 값을 저장) => 같은 4초짜리 오디오라도 샘플링 레이트에 따라 배열 길이가 달라진다.
 - bit depth: 각 샘플 값을 몇 비트로 표현하는가? (오디오 샘플 하나는 특정 순간의 소리 진폭 -> 비트 깊이는 그 진폭 값을 얼마나 세밀하게 표현할 수 있는지를 결정)
 - number of channels: 소리가 몇 개의 독립된 신호로 저장되어 있는가? (ex. Mono: 채널이 1개만 있는 신호, Stereo: 채널이 2개인 신호(소리의 방향감을 위한 Left, Right등..))

: 딥러닝 모델은 보통 고정된 형태의 텐서를 입력으로 받기 때문에 이 데이터들을 전처리를 통해 수정해야 함.


## 4 audio

10개의 fold가 있으며, 각 fold안에는 .wav 파일이 있다.
.wav 는 오디오 데이터를 담는 파일 형식이다. 

# WAV 파일의 구성

WAV 파일은 보통 RIFF(Resource Interchange File Format) 구조를 따른다.
RIFF 파일은 여러 개의 chunk로 구성되고 각 chunk는 FOURCC(Four-Character Code)라는 4글자 식별자를 가진다.

```text
WAV file
└── RIFF chunk
    ├── chunk ID: "RIFF"
    ├── chunk size
    └── chunk data
        ├── format: "WAVE"
        ├── fmt  subchunk
        │   ├── subchunk ID: "fmt "
        │   ├── subchunk size
        │   └── audio format information
        ├── data subchunk
        │   ├── subchunk ID: "data"
        │   ├── subchunk size
        │   └── audio sample bytes
        └── optional subchunks
            └── metadata, padding, etc.
...        ...
```
- fmt chunk: 이 오디오 데이터를 어떻게 해석해야 하는가?
    - sample rate, channel 수, bit depth, 압축/비압축 여부, byte rate, block align

- data chunk: 실제 소리값을 저장한 청크
    - sample 0, sample 1, sample 2, sample 3, ...
    - L0, R0, L1, R1, L2, R2, ...
    내부 인코딩은 fmt chunk 안에 있는 포맷 필드로 결정됨. 

WAV 파일의 `data` chunk에는 실제 오디오 sample 값들이 byte 형태로 연속 저장되어 있다. 
Mono 오디오는 한 시점에 하나의 sample만 가지므로 sample들이 순서대로 저장된다. 
Stereo 오디오는 left/right channel의 sample이 같은 시점 단위로 교차 저장되며, 일반적으로 `L0, R0, L1, R1, ...` 형태로 해석할 수 있다.

frame은 파일 안에 별도의 이름으로 존재하는 데이터 객체가 아니라, 같은 시점의 모든 channel sample을 묶어 부르는 해석 단위이다. 
stereo 오디오에서 `frame0 = [L0, R0]`, `frame1 = [L1, R1]`이다. 따라서 frame 수는 오디오의 시간축 길이를 나타내며, duration은 `frames / sample_rate`로 계산할 수 있다.

- ByteRate, BlockAlign
    - BlockAlign: 한 frame이 몇 byte인가? (BlockAlign = Channels × BitsPerSample / 8)
    - ByteRate: 1초 동안 재생하려면 몇 byte를 읽어야 하는가? (ByteRate = SampleRate × BlockAlign)


## 5 metadata

| 컬럼 이름 | 의미 | 자료형 |
|---|---|---|
| `slice_file_name` | 오디오 클립의 파일 이름 | string |
| `fsID` | 클립이 추출된 원본 Freesound 녹음 파일의 ID | integer |
| `start` | 원본 녹음에서 클립이 시작되는 시간, 단위는 초 | float |
| `end` | 원본 녹음에서 클립이 끝나는 시간, 단위는 초 | float |
| `salience` | 목표 소리가 전경음인지 배경음인지 나타내는 값. `1`은 foreground, `2`는 background를 의미한다 | integer |
| `fold` | 해당 오디오 파일이 속한 fold 번호 | integer |
| `classID` | 소리 클래스의 숫자 라벨 | integer |
| `class` | 소리 클래스의 문자열 라벨 | string |
- 전경음: 녹음에서 목표 소리가 뚜렷하게 들리는 경우
- 배경음: 목표 소리가 있긴 있지만, 다른 소리들에 묻혀서 덜 두드러지는 경우


```text 100032-3-0-0.wav,100032,0.0,0.317551,1,5,3,dog_bark```
파일 이름: 100032-3-0-0.wav
원본 Freesound ID: 100032
원본 녹음에서 0.0초 ~ 0.317551초 구간을 잘라낸 클립
salience: 1, 즉 목표 소리가 전경에 있음
fold: 5
classID: 3
class: dog_bark


## 6 유용한 라이브러리

[전처리에 쓸 만한 도구]
1. soundfile
audio/fold*/.wav 파일의 실제 포맷 확인에 좋다. 
soundfile은 libsndfile, CFFI, NumPy 기반 오디오 라이브러리이고, read()/write()로 오디오 파일을 직접 읽고 쓸 수 있다.
주로 쓰는 상황 :
- sample rate 확인
- channel 수 확인
- duration 확인
- WAV subtype 확인: PCM_16, PCM_24 등
- waveform을 numpy.ndarray로 읽기

예시1: WAV 포맷 확인
```python
import soundfile as sf

path = "audio/fold5/100032-3-0-0.wav"

info = sf.info(path)

print("samplerate:", info.samplerate)
print("channels:", info.channels)
print("frames:", info.frames)
print("duration:", info.duration)
print("format:", info.format)
print("subtype:", info.subtype)
```

예시2: waveform 읽기
```python
import soundfile as sf

path = "audio/fold5/100032-3-0-0.wav"

x, sr = sf.read(path, dtype="float32")

print(type(x))      # numpy.ndarray
print(x.dtype)     # float32
print(x.shape)     # mono: (num_samples,), stereo: (num_samples, channels)
print(sr)          # sample rate
```

2. librosa
librosa는 오디오 분석과 feature 추출에 특화된 라이브러로 feature extraction에는 mel spectrogram, MFCC 같은 기능이 포함된다.
 - Spectrogram: 소리를 시간별로 잘라서, 각 시간 구간에 어떤 주파수가 얼마나 강한지 나타낸 것
 - Mel scale: 이런 인간 청각 특성을 반영한 주파수 축(100Hz → 200Hz 차이는 크게 느낌, 10100Hz → 10200Hz 차이는 상대적으로 작게 느낌)
 - Mel-spectrogram: spectrogram의 주파수 축을 mel scale로 압축
 - Log-mel spectrogram: 로그를 씌운 것
 - MFCC: Mel-Frequency Cepstral Coefficients, log-mel spectrogram을 더 압축해서 만든 요약 특징

주로 쓰는 상황 :
- 오디오를 mono로 읽기
- sample rate를 자동으로 맞추기
- mel-spectrogram 만들기
- log-mel spectrogram 만들기
- MFCC 만들기
- waveform / spectrogram 시각화

예시: 원본 sampling rate와 channel 수를 보존해서 waveform을 읽기
```python
import librosa

path = "audio/fold5/100032-3-0-0.wav"

y, sr = librosa.load(path, sr=None, mono=False)

print(type(y))      # numpy.ndarray
print(y.dtype)     # float32 계열
print(y.shape)     # mono: (num_samples,), stereo: (channels, num_samples)
print(sr)
```

예시: 학습용으로 통일해서 읽기
```python
import librosa

path = "audio/fold5/100032-3-0-0.wav"

target_sr = 16000

y, sr = librosa.load(
    path,
    sr=target_sr,   # 16000 Hz로 resampling
    mono=True       # mono로 변환
)

print(y.shape)
print(sr)
```

예시: 길이 4초로 통일
```python
import numpy as np

target_sr = 16000
target_duration = 4
target_length = target_sr * target_duration

if len(y) < target_length:
    pad_length = target_length - len(y)
    y = np.pad(y, (0, pad_length))
else:
    y = y[:target_length]

print(y.shape)  # (64000,)
```

예시: log-mel spectrogram 만들기
```python
import librosa
import numpy as np

mel = librosa.feature.melspectrogram(
    y=y,
    sr=target_sr,
    n_fft=1024,
    hop_length=512,
    n_mels=64,
    power=2.0
)

log_mel = librosa.power_to_db(mel, ref=np.max)

print(log_mel.shape)  # (n_mels, time_frames)
```

예시: MFCC 만들기
```python
mfcc = librosa.feature.mfcc(
    y=y,
    sr=target_sr,
    n_mfcc=40
)

print(mfcc.shape)  # (40, time_frames)
```

3. torchaudio.transforms
torchaudio.transforms는 오디오 처리와 feature extraction용 transform들을 제공한다.
주로 쓰는 상황:
- Resample
- MelSpectrogram
- AmplitudeToDB
- FrequencyMasking
- TimeMasking

예시: torchaudio로 전처리
```python
import torch
import torchaudio

path = "audio/fold5/100032-3-0-0.wav"

waveform, sr = torchaudio.load(path)

target_sr = 16000

# resample
if sr != target_sr:
    resampler = torchaudio.transforms.Resample(
        orig_freq=sr,
        new_freq=target_sr
    )
    waveform = resampler(waveform)

# stereo -> mono
if waveform.shape[0] > 1:
    waveform = waveform.mean(dim=0, keepdim=True)

# 길이 4초로 통일
target_length = target_sr * 4

if waveform.shape[1] < target_length:
    pad_length = target_length - waveform.shape[1]
    waveform = torch.nn.functional.pad(waveform, (0, pad_length))
else:
    waveform = waveform[:, :target_length]

print(waveform.shape)  # [1, 64000]
```

4. torch.utils.data.Dataset
모델에 올릴 때 쓸 수 있다.

예시: waveform을 반환하는 Dataset
```python
from pathlib import Path

import pandas as pd
import torch
import torchaudio
from torch.utils.data import Dataset


class UrbanSoundWaveformDataset(Dataset):
    def __init__(self, root, folds, target_sr=16000, duration=4):
        self.root = Path(root)
        self.meta = pd.read_csv(self.root / "metadata" / "UrbanSound8K.csv")
        self.meta = self.meta[self.meta["fold"].isin(folds)].reset_index(drop=True)

        self.target_sr = target_sr
        self.target_length = target_sr * duration

    def __len__(self):
        return len(self.meta)

    def __getitem__(self, idx):
        row = self.meta.iloc[idx]

        path = (
            self.root
            / "audio"
            / f"fold{row['fold']}"
            / row["slice_file_name"]
        )

        waveform, sr = torchaudio.load(path)

        # resample
        if sr != self.target_sr:
            waveform = torchaudio.functional.resample(
                waveform,
                orig_freq=sr,
                new_freq=self.target_sr
            )

        # stereo -> mono
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        # pad / truncate
        if waveform.shape[1] < self.target_length:
            pad_length = self.target_length - waveform.shape[1]
            waveform = torch.nn.functional.pad(waveform, (0, pad_length))
        else:
            waveform = waveform[:, :self.target_length]

        label = int(row["classID"])

        return waveform, label
```
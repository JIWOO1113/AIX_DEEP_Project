1. 데이터셋 개요

   UrbanSound8K는 도시 환경에서 발생하는 소리를 분류하기 위한 오디오 데이터셋이다.<br>
이 데이터셋은 총 8,732개의 WAV 오디오 파일로 구성되어 있으며, 각 오디오는 최대 4초 이하의 짧은 소리 조각이다.  
소리 데이터는 총 10개의 클래스로 라벨링되어 있다.
<br>

2. 다운로드 파일 구조
   
| 파일명 | 설명 |
|---|---|
| `audio` | 실제 WAV 오디오 파일이 들어 있는 폴더 |
| `metadata` | 오디오 파일별 정답 라벨과 부가 정보가 들어 있는 폴더 |
| `.DS_Store` | macOS에서 자동 생성되는 시스템 파일 |
| `FREESOUNDCREDITS` | 원본 음원의 출처와 제작자 정보를 기록한 파일 |
| `UrbanSound8K_README` | 데이터셋 설명서로, 데이터 이해와 보고서 작성에 필요 |
<br>
3. 각 파일 설명<br>
   3-1. audio 폴더(원본 오디오 파일)<br>
        실제 소리 파일이 들어 있다. WAV 형식으로 총 8732개있다.<br>
      
| fold | wav 파일 개수 |
|---|---:|
| fold1 | 873 |
| fold2 | 888 |
| fold3 | 925 |
| fold4 | 990 |
| fold5 | 936 |
| fold6 | 823 |
| fold7 | 838 |
| fold8 | 806 |
| fold9 | 816 |
| fold10 | 837 |
| **총합** | **8,732** |

   3-2. metadata 폴더(원본 디오 파일 설명)
       `metadata` 폴더에는 `UrbanSound8K.csv` 파일이 들어 있다.  
이 파일은 실제 오디오 파일 자체가 아니라, 각 WAV 파일의 파일명, fold 번호, 클래스 라벨, 원본 오디오에서 잘라낸 시작·종료 시점 등을 설명하는 metadata 파일이다.  
즉, `audio` 폴더의 raw data를 모델 학습에 사용할 수 있도록 연결해주는 기준표 역할을 한다.
<br>
   #### metadata 컬럼 설명

| 컬럼명 | 의미 |
|---|---|
| `slice_file_name` | 실제 wav 파일명 |
| `fsID` | 원본 Freesound 오디오 ID |
| `start` | 원본 오디오에서 잘라낸 시작 시점 |
| `end` | 원본 오디오에서 잘라낸 종료 시점 |
| `salience` | 소리가 얼마나 뚜렷한지 나타내는 값 |
| `fold` | 오디오 파일이 들어 있는 fold 번호 |
| `classID` | 소리 클래스 번호 |
| `class` | 소리 클래스 이름 |    
<br>
salience 변수 설명 <br>

| salience | 의미 |
|---:|---|
| 1 | foreground, 중심적으로 들리는 소리 |
| 2 | background, 배경에 가깝게 들리는 소리 |

<br>

#### 클래스별 데이터 개수

| classID | class | 데이터 개수 |
|---:|---|---:|
| 0 | air_conditioner | 1,000 |
| 1 | car_horn | 429 |
| 2 | children_playing | 1,000 |
| 3 | dog_bark | 1,000 |
| 4 | drilling | 1,000 |
| 5 | engine_idling | 1,000 |
| 6 | gun_shot | 374 |
| 7 | jackhammer | 1,000 |
| 8 | siren | 929 |
| 9 | street_music | 1,000 |

   3-3. .DS_Store 폴더
`.DS_Store`는 macOS에서 폴더 설정 정보를 저장하기 위해 자동 생성되는 시스템 파일이다.  <br>

   3-4. FREESOUNDCREDITS 폴더
`FREESOUNDCREDITS` 파일은 UrbanSound8K 데이터셋에 포함된 원본 음원의 출처와 제작자 정보를 기록한 파일이다.  <br>

   3-5. UrbanSound8K_README 폴더<br>
   fold 구조를 기준으로 학습과 평가를 진행하는 것이 타당함.<br>
   데이터 읽을 때 주의해야할 부분 <br>
   이 값들은 원본 Freesound 파일의 특성을 그대로 따르기 때문에 파일마다 다를 수 있다.

따라서 metadata만 보고 sampling rate나 mono/stereo 여부를 판단할 수는 없다.  
이 정보는 실제 WAV 파일을 직접 읽어서 확인해야 한다.
   

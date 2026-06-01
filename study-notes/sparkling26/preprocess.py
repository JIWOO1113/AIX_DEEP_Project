import os
import numpy as np
import pandas as pd
import librosa

DATA_DIR = r"C:\Users\sp9gh\Downloads\UrbanSound8K\UrbanSound8K"

CSV_PATH = os.path.join(DATA_DIR, "metadata", "UrbanSound8K.csv")
AUDIO_DIR = os.path.join(DATA_DIR, "audio")

TARGET_SR = 22050
TARGET_DURATION = 4
TARGET_LENGTH = TARGET_SR * TARGET_DURATION


def preprocess_audio(file_path):
    y, sr = librosa.load(file_path, sr=TARGET_SR, mono=True)
    y = y.astype(np.float32)

    if len(y) > TARGET_LENGTH:
        y = y[:TARGET_LENGTH]

    elif len(y) < TARGET_LENGTH:
        repeat_count = int(np.ceil(TARGET_LENGTH / len(y)))
        y = np.tile(y, repeat_count)[:TARGET_LENGTH]

    return y


def extract_mfcc_features(y, sr=TARGET_SR, n_mfcc=40):
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)

    feature = np.concatenate([
        mfcc.mean(axis=1),
        mfcc.std(axis=1)
    ])

    return feature


df = pd.read_csv(CSV_PATH)

X = []
y = []
folds = []
class_names = []

for idx, row in df.iterrows():
    file_path = os.path.join(
        AUDIO_DIR,
        f"fold{row['fold']}",
        row["slice_file_name"]
    )

    try:
        waveform = preprocess_audio(file_path)
        feature = extract_mfcc_features(waveform)

        X.append(feature)
        y.append(row["classID"])
        folds.append(row["fold"])
        class_names.append(row["class"])

    except Exception as e:
        print("오류 발생:", file_path)
        print(e)

    if (idx + 1) % 500 == 0:
        print(f"{idx + 1}개 처리 완료")


X = np.array(X)
y = np.array(y)
folds = np.array(folds)
class_names = np.array(class_names)

print("X shape:", X.shape)
print("y shape:", y.shape)
print("folds shape:", folds.shape)

np.save("X_mfcc.npy", X)
np.save("y.npy", y)
np.save("folds.npy", folds)
np.save("class_names.npy", class_names)

print("저장 완료!")
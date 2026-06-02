import numpy as np
import librosa
import tensorflow as tf
import gc
import warnings
from datasets import load_dataset
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import ModelCheckpoint
warnings.filterwarnings('ignore')
tf.keras.backend.clear_session()

SAMPLE_RATE = 22050
DURATION = 4.0
N_MELS = 96
IMG_HEIGHT = 96
IMG_WIDTH = 150

NUM_CLASSES = 10
BATCH_SIZE = 16
EPOCHS = 30
LEARNING_RATE = 0.001
print("데이터셋 불러오는 중...")
ds = load_dataset("mteb/urbansound8K", trust_remote_code=True)

def extract_mel_from_audio(audio_dict):
    try:
        y = audio_dict['array']
        sr = audio_dict['sampling_rate']
        if sr != SAMPLE_RATE:
            y = librosa.resample(y, orig_sr=sr, target_sr=SAMPLE_RATE)

        target_len = int(SAMPLE_RATE * DURATION)
        if len(y) < target_len:
            y = np.pad(y, (0, target_len - len(y)), mode='constant')
        else:
            y = y[:target_len]

        mel_spec = librosa.feature.melspectrogram(
            y=y, sr=SAMPLE_RATE, n_fft=2048, hop_length=512,
            n_mels=N_MELS, fmax=8000
        )
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        mel_spec_db = (mel_spec_db - mel_spec_db.min()) / (mel_spec_db.max() - mel_spec_db.min() + 1e-8)

        if mel_spec_db.shape[1] < IMG_WIDTH:
            pad_width = IMG_WIDTH - mel_spec_db.shape[1]
            mel_spec_db = np.pad(mel_spec_db, ((0,0), (0, pad_width)), mode='constant')
        else:
            mel_spec_db = mel_spec_db[:, :IMG_WIDTH]

        return mel_spec_db
    except:
        return None

# 배치 단위 추출 + 저장
def prepare_dataset_batch(batch_size=400):
    features, labels, folds = [], [], []
    total = len(ds['train'])

    for i in range(0, total, batch_size):
        print(f"배치 처리 중 {i//batch_size + 1}/{total//batch_size + 1}")
        batch = ds['train'].select(range(i, min(i + batch_size, total)))

        for item in batch:
            mel = extract_mel_from_audio(item['audio'])
            if mel is not None:
                features.append(mel[..., np.newaxis])
                labels.append(item['classID'])
                folds.append(item['fold'])

        gc.collect()

    X = np.array(features, dtype=np.float32)
    y = np.array(labels)
    folds = np.array(folds)

    # 특성 저장
    np.save('X_mel.npy', X)
    np.save('y.npy', y)
    np.save('folds.npy', folds)
    print(f"특성 추출 완료! 형태: {X.shape}")
    return X, y, folds

X, y, folds = prepare_dataset_batch(batch_size=400)
y_cat = to_categorical(y, num_classes=NUM_CLASSES)
def augment_spectrogram(spec):
    """올바른 SpecAugment 구현"""
    spec = tf.convert_to_tensor(spec, dtype=tf.float32)

    # 시간 마스킹 (Time Masking) - 시간 축 (width) 기준
    if tf.random.uniform([]) > 0.5:
        mask_width = tf.random.uniform([], minval=15, maxval=40, dtype=tf.int32)
        start = tf.random.uniform([], minval=0, maxval=IMG_WIDTH - mask_width, dtype=tf.int32)

        # 마스크 인덱스 생성
        indices = tf.reshape(tf.range(start, start + mask_width), [-1, 1])
        # 업데이트 값은 평균 사용
        mean_val = tf.reduce_mean(spec)
        updates = tf.fill([mask_width, IMG_HEIGHT], mean_val)   # 주의: 형태 (mask_width, height)

        # 전치 후 업데이트하고 다시 전치 (더 안정적)
        spec_t = tf.transpose(spec)  # (width, height)
        spec_t = tf.tensor_scatter_nd_update(spec_t, indices, updates)
        spec = tf.transpose(spec_t)

    # 주파수 마스킹 (Frequency Masking) - 주파수 축 (height) 기준
    if tf.random.uniform([]) > 0.5:
        mask_height = tf.random.uniform([], minval=8, maxval=25, dtype=tf.int32)
        start = tf.random.uniform([], minval=0, maxval=IMG_HEIGHT - mask_height, dtype=tf.int32)

        indices = tf.reshape(tf.range(start, start + mask_height), [-1, 1])
        mean_val = tf.reduce_mean(spec)
        updates = tf.fill([mask_height, IMG_WIDTH], mean_val)   # (mask_height, width)

        spec = tf.tensor_scatter_nd_update(spec, indices, updates)

    # 랜덤 노이즈 추가
    if tf.random.uniform([]) > 0.55:
        noise = tf.random.normal(spec.shape, mean=0.0, stddev=0.055)
        spec = spec + noise

    # [0,1] 범위로 클리핑
    spec = tf.clip_by_value(spec, 0.0, 1.0)

    return spec.numpy()
def build_cnn_model():
    model = Sequential([
        Conv2D(32, (3,3), activation='relu', padding='same', input_shape=(IMG_HEIGHT, IMG_WIDTH, 1)),
        BatchNormalization(),
        MaxPooling2D((2,2)),
        Dropout(0.3),

        Conv2D(64, (3,3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D((2,2)),
        Dropout(0.3),

        Conv2D(128, (3,3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D((2,2)),
        Dropout(0.4),

        Conv2D(256, (3,3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D((2,2)),
        Dropout(0.4),

        Flatten(),
        Dense(512, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        Dense(NUM_CLASSES, activation='softmax')
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model
fold_accuracies = []

for test_fold in range(1, 11):
    print(f"\n{'='*70}")
    print(f"Fold {test_fold} / 10 학습 중")
    print(f"{'='*70}")

    train_mask = folds != test_fold
    test_mask = folds == test_fold

    X_train_raw = X[train_mask]
    y_train_raw = y_cat[train_mask]
    X_test = X[test_mask]
    y_test = y_cat[test_mask]

    X_train, X_val, y_train, y_val = train_test_split(
        X_train_raw, y_train_raw, test_size=0.15, random_state=42,
        stratify=np.argmax(y_train_raw, axis=1)
    )

    # 데이터 증강 (훈련 세트에만 적용)
    print("훈련 세트에 데이터 증강 적용 중...")
    X_train_aug = [augment_spectrogram(spec.squeeze())[..., np.newaxis] for spec in X_train]
    X_train = np.array(X_train_aug, dtype=np.float32)

    model = build_cnn_model()

    callbacks = [
        EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=4, min_lr=1e-6, verbose=1),
        ModelCheckpoint(
            filepath=f'best_model_fold{test_fold}.h5',
            monitor='val_loss',
            save_best_only=True,  
            save_weights_only=False,
            verbose=1
        )
    ]

    model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        verbose=1
    )

    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    fold_accuracies.append(test_acc)
    print(f"Fold {test_fold} 테스트 정확도: {test_acc:.4f}")

    # 메모리 정리
    del model, X_train, X_val, X_train_aug
    gc.collect()

print(f"\n{'='*70}")
print(f"최종 10-Fold 평균 정확도: {np.mean(fold_accuracies):.4f} ± {np.std(fold_accuracies):.4f}")
print(f"{'='*70}")

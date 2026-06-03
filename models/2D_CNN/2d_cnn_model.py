"""
2D CNN 음향 분류 (UrbanSound8K)
================================
log-mel spectrogram을 "이미지처럼" 2차원 그대로 입력받아, 2D CNN으로
10개 환경음 class를 분류한다. (MLP가 시간축을 평균으로 압축한 것과 달리,
여기서는 (주파수 × 시간) 2차원 구조를 그대로 유지한다.)

분석 방식:
  - Train: fold 1~8 / Validation: fold 9 / Test: fold 10
  - Validation accuracy 기준 Early Stopping + best 모델 선정

평가 지표 (팀 합의 12개 항목):
  - 종합: accuracy, balanced accuracy, macro/weighted precision·recall·F1
  - class별: precision, recall, F1
  - confusion matrix

출력 (models/2D_CNN/output/):
  - 2d_cnn_best.pt               : 학습된 best 모델 체크포인트
  - 2d_cnn_metrics.csv           : 종합 지표 8개
  - 2d_cnn_classwise_metrics.csv : class별 precision/recall/F1
  - 2d_cnn_predictions.csv       : 샘플별 y_true/y_pred
그림 (images/):
  - 2d_cnn_confusion.png         : confusion matrix

폴더 구조 (project_root 기준):
  project_root/
  ├── UrbanSound8K/
  │   ├── audio/
  │   └── metadata/
  └── models/
      └── 2D_CNN/
          └── 2d_cnn_model.py    ← 이 파일

실행:
  python models/2D_CNN/2d_cnn_model.py
"""

import time
from pathlib import Path

import numpy as np
import pandas as pd
import librosa
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, f1_score,
    precision_score, recall_score, confusion_matrix,
    precision_recall_fscore_support,
)
import matplotlib.pyplot as plt


# ============================================================
# STEP 0. 경로 설정 (어느 환경에서나 동작하도록 상대 경로 사용)
# ============================================================
# 이 파일은 project_root/models/2D_CNN/2d_cnn_model.py 위치를 가정한다.
# parent.parent.parent = project_root
ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = ROOT / "UrbanSound8K"
AUDIO_DIR = DATA_DIR / "audio"
META_PATH = DATA_DIR / "metadata" / "UrbanSound8K.csv"

# 결과/캐시는 이 파일과 같은 폴더(models/2D_CNN) 아래에 저장
HERE = Path(__file__).resolve().parent
OUT_DIR = HERE / "output"
OUT_DIR.mkdir(parents=True, exist_ok=True)
FEAT_PATH = HERE / "2d_cnn_features.npz"   # 2D feature 캐시 (있으면 재사용)

# 그림은 project_root/images/ 에 저장 (팀 공통, 접두사로 모델 구분)
IMG_DIR = ROOT / "images"
IMG_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# STEP 1. 전처리 / 학습 파라미터 (팀 합의 — MLP와 동일하게 통일)
# ============================================================
SR = 22050
N_SAMPLES = int(SR * 4.0)   # 4초로 고정 → 모든 spectrogram의 시간축 길이가 같아짐
N_MELS = 128                # 주파수축 크기
N_FFT = 2048
HOP_LENGTH = 512
# 4초를 위 설정으로 변환하면 시간축 프레임 수(T)는 약 173으로 항상 일정하다.

NUM_CLASSES = 10
BATCH_SIZE = 64
LEARNING_RATE = 0.001
EPOCHS = 100
EARLY_STOPPING_PATIENCE = 10
SEED = 42

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(SEED)
np.random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)


# ============================================================
# STEP 2. Feature 추출 (없으면 추출, 있으면 재사용)
#   - MLP는 시간축을 평균내 256차원 벡터로 압축했지만,
#     CNN은 (128 × 173) 2차원 spectrogram을 그대로 사용한다.
# ============================================================
def repeat_pad(y, target_len=N_SAMPLES):
    """신호가 짧으면 반복해서 4초로 채움 (MLP와 동일)"""
    if len(y) >= target_len:
        return y[:target_len]
    n_repeats = target_len // len(y) + 1
    return np.tile(y, n_repeats)[:target_len]


def extract_log_mel_2d(y, sr=SR):
    """log-mel spectrogram을 (128 × T) 2차원 그대로 반환 (압축하지 않음)"""
    mel = librosa.feature.melspectrogram(
        y=y, sr=sr, n_mels=N_MELS, n_fft=N_FFT, hop_length=HOP_LENGTH)
    log_mel = librosa.power_to_db(mel, ref=np.max)
    return log_mel.astype(np.float32)   # shape: (128, T)


def build_features():
    """원본 오디오에서 2D log-mel을 추출해 배열로 반환 + npz 캐시 저장.

    반환:
      X    : (N, 128, T) float32  — 2D spectrogram 묶음
      y    : (N,) int64           — classID
      fold : (N,) int             — fold 번호
      class_names : list[str]
    """
    # --- 캐시가 있으면 그대로 사용 ---
    if FEAT_PATH.exists():
        print(f"[Feature] 캐시 사용: {FEAT_PATH}")
        data = np.load(FEAT_PATH, allow_pickle=True)
        return data["X"], data["y"], data["fold"], list(data["class_names"])

    print("[Feature] 캐시 없음 → 원본 오디오에서 추출 시작")
    if not META_PATH.exists():
        raise SystemExit(f"[ERROR] 메타데이터 없음: {META_PATH}\n"
                         f"UrbanSound8K 데이터를 {DATA_DIR} 에 배치했는지 확인하세요.")

    meta = pd.read_csv(META_PATH)
    X_list, y_list, fold_list, fail = [], [], [], 0
    t0 = time.time()
    for idx in range(len(meta)):
        row = meta.iloc[idx]
        wav_path = AUDIO_DIR / f"fold{int(row['fold'])}" / str(row['slice_file_name'])
        try:
            y, _ = librosa.load(str(wav_path), sr=SR, mono=True)
            y = repeat_pad(y)
            spec = extract_log_mel_2d(y)        # (128, T)
            X_list.append(spec)
            y_list.append(int(row['classID']))
            fold_list.append(int(row['fold']))
        except Exception as e:
            fail += 1
            if fail <= 5:
                print(f"  Skip: {row['slice_file_name']} - {e}")
        if (idx + 1) % 1000 == 0:
            print(f"  {idx+1}/{len(meta)} 처리...")

    X = np.stack(X_list)                          # (N, 128, T)
    y = np.array(y_list, dtype=np.int64)
    fold = np.array(fold_list, dtype=np.int64)
    class_names = sorted(meta["class"].unique())

    np.savez_compressed(FEAT_PATH, X=X, y=y, fold=fold, class_names=class_names)
    print(f"[Feature] 추출 완료: 성공 {len(X)}개, 실패 {fail}개, "
          f"{(time.time()-t0)/60:.1f}분 → 캐시 저장 {FEAT_PATH}")
    return X, y, fold, class_names


# ============================================================
# STEP 3. 2D CNN 모델 정의
#   - ConvBlock(Conv2d + BatchNorm + ReLU + MaxPool)을 4번 쌓아
#     spectrogram에서 점점 복잡한 패턴을 뽑는다.
#   - 마지막에 Global Average Pooling으로 1차원으로 요약한 뒤,
#     Linear(분류기)로 10개 class 점수를 낸다. (RNN 없음)
# ============================================================
class ConvBlock(nn.Module):
    """Conv2d → BatchNorm → ReLU → MaxPool 묶음 (CNN 기본 블록)"""
    def __init__(self, in_ch, out_ch, dropout=0.1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout),
            nn.MaxPool2d(kernel_size=2),   # 주파수·시간축을 절반으로 줄임
        )

    def forward(self, x):
        return self.block(x)


class CNNClassifier(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES):
        super().__init__()
        # 입력: (배치, 1채널, 128, ~173)
        self.features = nn.Sequential(
            ConvBlock(1, 32, dropout=0.05),     # (32, 64, ~86)
            ConvBlock(32, 64, dropout=0.10),    # (64, 32, ~43)
            ConvBlock(64, 128, dropout=0.15),   # (128, 16, ~21)
            ConvBlock(128, 128, dropout=0.15),  # (128, 8, ~10)
        )
        # Global Average Pooling: 각 채널을 숫자 1개로 요약 → (배치, 128)
        self.gap = nn.AdaptiveAvgPool2d(1)
        # 분류기(FC): 128 → 10
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.gap(x)
        return self.classifier(x)


# ============================================================
# STEP 4. 정규화 (train 통계로만 — validation/test 정보 누수 방지)
# ============================================================
def normalize(X_train, X_val, X_test):
    """train 전체의 평균/표준편차로 정규화 (MLP와 동일 원리, 2D 버전)"""
    mean = X_train.mean()
    std = X_train.std()
    if std == 0:
        std = 1.0
    return ((X_train - mean) / std,
            (X_val - mean) / std,
            (X_test - mean) / std)


# ============================================================
# STEP 5. 학습 + 평가 (1차-B: train 학습, val로 best 선정, test로 평가)
# ============================================================
def make_loader(X, y, shuffle):
    # (N, 128, T) → (N, 1, 128, T) : CNN은 채널 차원이 필요하다
    X_t = torch.FloatTensor(X).unsqueeze(1)
    y_t = torch.LongTensor(y)
    return DataLoader(TensorDataset(X_t, y_t), batch_size=BATCH_SIZE, shuffle=shuffle)


def train_and_eval(X_train, y_train, X_val, y_val, X_test, y_test):
    train_loader = make_loader(X_train, y_train, shuffle=True)
    val_loader = make_loader(X_val, y_val, shuffle=False)
    test_loader = make_loader(X_test, y_test, shuffle=False)

    model = CNNClassifier(NUM_CLASSES).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    def eval_loader(loader):
        model.eval()
        preds, trues, losses = [], [], []
        with torch.no_grad():
            for X, y in loader:
                X, y = X.to(device), y.to(device)
                logits = model(X)
                losses.append(criterion(logits, y).item())
                preds.extend(logits.argmax(1).cpu().numpy())
                trues.extend(y.cpu().numpy())
        return np.mean(losses), accuracy_score(trues, preds), np.array(trues), np.array(preds)

    best_val_acc, best_state, patience, final_epoch = 0.0, None, 0, 0
    for epoch in range(1, EPOCHS + 1):
        model.train()
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(X), y)
            loss.backward()
            optimizer.step()

        _, val_acc, _, _ = eval_loader(val_loader)
        final_epoch = epoch
        # validation accuracy가 역대 최고면 이 모델을 best로 저장
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience = 0
        else:
            patience += 1
            if patience >= EARLY_STOPPING_PATIENCE:
                print(f"  Early stopping at epoch {epoch}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    test_loss, _, y_true, y_pred = eval_loader(test_loader)
    return y_true, y_pred, best_val_acc, test_loss, final_epoch, model


# ============================================================
# STEP 6. 평가 지표 (종합 8개)
# ============================================================
def compute_metrics(y_true, y_pred):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "macro_precision": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "macro_recall": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "weighted_precision": precision_score(y_true, y_pred, average="weighted", zero_division=0),
        "weighted_recall": recall_score(y_true, y_pred, average="weighted", zero_division=0),
        "weighted_f1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
    }


# ============================================================
# STEP 7. 메인 — 위 STEP들을 순서대로 실행
# ============================================================
def main():
    print("=" * 60)
    print("2D CNN 음향 분류 (Train fold1~8 / Val fold9 / Test fold10)")
    print("=" * 60)
    print(f"Device: {device}\n")

    # 1) feature 준비 (없으면 추출)
    X_all, y_all, fold, class_names = build_features()
    print(f"Feature shape: {X_all.shape}  (N, n_mels, T)\n")

    # 2) fold 분할 (1차-B)
    tr = np.isin(fold, range(1, 9))   # fold 1~8
    va = fold == 9
    te = fold == 10
    Xtr, Xva, Xte = normalize(X_all[tr], X_all[va], X_all[te])
    print(f"Train: {tr.sum()} | Val: {va.sum()} | Test: {te.sum()}\n")

    # 3) 학습 + 평가
    y_true, y_pred, best_val_acc, test_loss, final_epoch, model = train_and_eval(
        Xtr, y_all[tr], Xva, y_all[va], Xte, y_all[te])

    # 4) 12개 지표
    metrics = compute_metrics(y_true, y_pred)
    metrics["best_val_acc"] = best_val_acc
    metrics["test_loss"] = test_loss
    metrics["final_epoch"] = final_epoch

    print("=" * 60)
    print("[전체 평가 지표]")
    print("=" * 60)
    for k in ["accuracy", "balanced_accuracy", "macro_precision", "macro_recall",
              "macro_f1", "weighted_precision", "weighted_recall", "weighted_f1"]:
        print(f"  {k:<20}: {metrics[k]:.4f}")
    print(f"  best_val_acc        : {best_val_acc:.4f}")
    print(f"  final_epoch         : {final_epoch}")

    pd.DataFrame([metrics]).to_csv(OUT_DIR / "2d_cnn_metrics.csv",
                                   index=False, encoding="utf-8-sig")

    # class별 precision/recall/f1
    p, r, f, s = precision_recall_fscore_support(
        y_true, y_pred, labels=range(NUM_CLASSES), zero_division=0)
    cw = pd.DataFrame({"class": class_names, "precision": p, "recall": r,
                       "f1": f, "support": s}).sort_values("f1", ascending=False)
    cw.to_csv(OUT_DIR / "2d_cnn_classwise_metrics.csv", index=False, encoding="utf-8-sig")
    print("\n[클래스별 지표]")
    print(cw.round(4).to_string(index=False))

    # 샘플별 예측값
    pd.DataFrame({"y_true": y_true, "y_pred": y_pred}).to_csv(
        OUT_DIR / "2d_cnn_predictions.csv", index=False, encoding="utf-8-sig")

    # 체크포인트 저장 (학습된 best 모델)
    ckpt_path = OUT_DIR / "2d_cnn_best.pt"
    torch.save({
        "model_state_dict": model.state_dict(),
        "num_classes": NUM_CLASSES,
        "class_names": class_names,
        "seed": SEED,
        "best_val_acc": best_val_acc,
        "metrics": metrics,
    }, ckpt_path)
    print(f"\n체크포인트 저장: {ckpt_path}")

    # confusion matrix 그림
    try:
        plt.rcParams['axes.unicode_minus'] = False
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(10, 8))
        plt.imshow(cm, cmap="Blues")
        plt.colorbar()
        plt.xticks(range(NUM_CLASSES), class_names, rotation=45, ha="right")
        plt.yticks(range(NUM_CLASSES), class_names)
        for i in range(NUM_CLASSES):
            for j in range(NUM_CLASSES):
                plt.text(j, i, cm[i, j], ha="center", va="center",
                         color="white" if cm[i, j] > cm.max()/2 else "black")
        plt.xlabel("Predicted"); plt.ylabel("True")
        plt.title("2D CNN Confusion Matrix (Test fold 10)")
        plt.tight_layout()
        plt.savefig(IMG_DIR / "2d_cnn_confusion.png", dpi=150, bbox_inches="tight")
        plt.close()
    except Exception as e:
        print(f"  (confusion matrix 그림 생략: {e})")

    print(f"\n결과 저장: {OUT_DIR}")
    print("=== 완료 ===")


if __name__ == "__main__":
    main()

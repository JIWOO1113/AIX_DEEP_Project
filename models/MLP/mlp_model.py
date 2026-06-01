"""
MLP 음향 분류 (UrbanSound8K)
============================
사전 추출된 feature 없이도 어느 환경에서나 동작하도록, 원본 오디오에서
feature를 직접 추출(없으면 추출, 있으면 재사용)한 뒤 MLP로 분류한다.

분석 방식:
  - Train: fold 1~8 / Validation: fold 9 / Test: fold 10
  - Validation 기준 Early Stopping 적용

평가 지표 (팀 합의 12개 항목):
  - 종합: accuracy, balanced accuracy, macro/weighted precision·recall·F1
  - class별: precision, recall, F1
  - confusion matrix

출력 (models/MLP/output/):
  - mlp_best.pt              : 학습된 best 모델 체크포인트
  - mlp_metrics.csv          : 종합 지표 8개
  - mlp_classwise_metrics.csv: class별 precision/recall/F1
  - mlp_predictions.csv      : 샘플별 y_true/y_pred
  - mlp_confusion.png        : confusion matrix

폴더 구조 (project_root 기준):
  project_root/
  ├── UrbanSound8K/
  │   ├── audio/
  │   └── metadata/
  └── models/
      └── MLP/
          └── mlp_model.py   ← 이 파일

실행:
  python models/MLP/mlp_model.py
"""

import os
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
# 경로 설정 (어느 환경에서나 동작하도록 상대 경로 사용)
# ============================================================
# 이 파일은 project_root/models/MLP/mlp_model.py 위치를 가정한다.
# parent.parent.parent = project_root
ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = ROOT / "UrbanSound8K"
AUDIO_DIR = DATA_DIR / "audio"
META_PATH = DATA_DIR / "metadata" / "UrbanSound8K.csv"

# 결과/캐시는 이 파일과 같은 폴더(models/MLP) 아래에 저장
HERE = Path(__file__).resolve().parent
OUT_DIR = HERE / "output"
OUT_DIR.mkdir(parents=True, exist_ok=True)
FEAT_PATH = HERE / "mlp_features.csv"   # feature 캐시 (있으면 재사용)

# 그림은 project_root/images/ 에 저장 (팀 공통, 접두사로 모델 구분)
IMG_DIR = ROOT / "images"
IMG_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# 전처리 / 학습 파라미터 (팀 합의)
# ============================================================
SR = 22050
N_SAMPLES = int(SR * 4.0)   # 4초
N_MELS = 128
N_FFT = 2048
HOP_LENGTH = 512

INPUT_DIM = 256             # mean 128 + std 128
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
# 1. Feature 추출 (없으면 추출, 있으면 재사용)
# ============================================================
def repeat_pad(y, target_len=N_SAMPLES):
    """신호가 짧으면 반복해서 4초로 채움"""
    if len(y) >= target_len:
        return y[:target_len]
    n_repeats = target_len // len(y) + 1
    return np.tile(y, n_repeats)[:target_len]


def extract_log_mel(y, sr=SR):
    """log-mel spectrogram → 시간축 평균(128) + 표준편차(128) = 256차원"""
    mel = librosa.feature.melspectrogram(
        y=y, sr=sr, n_mels=N_MELS, n_fft=N_FFT, hop_length=HOP_LENGTH)
    log_mel = librosa.power_to_db(mel, ref=np.max)
    return np.concatenate([log_mel.mean(axis=1), log_mel.std(axis=1)])


def build_features():
    """원본 오디오에서 256차원 feature를 추출해 DataFrame으로 반환 + csv 캐시 저장"""
    if FEAT_PATH.exists():
        print(f"[Feature] 캐시 사용: {FEAT_PATH}")
        return pd.read_csv(FEAT_PATH)

    print(f"[Feature] 캐시 없음 → 원본 오디오에서 추출 시작")
    if not META_PATH.exists():
        raise SystemExit(f"[ERROR] 메타데이터 없음: {META_PATH}\n"
                         f"UrbanSound8K 데이터를 {DATA_DIR} 에 배치했는지 확인하세요.")

    meta = pd.read_csv(META_PATH)
    rows, fail = [], 0
    t0 = time.time()
    for idx in range(len(meta)):
        row = meta.iloc[idx]
        wav_path = AUDIO_DIR / f"fold{int(row['fold'])}" / str(row['slice_file_name'])
        try:
            y, _ = librosa.load(str(wav_path), sr=SR, mono=True)
            y = repeat_pad(y)
            feats = extract_log_mel(y)
            d = {f"feat_{i}": float(feats[i]) for i in range(INPUT_DIM)}
            d.update({"slice_file_name": str(row['slice_file_name']),
                      "fold": int(row['fold']),
                      "classID": int(row['classID']),
                      "class": str(row['class'])})
            rows.append(d)
        except Exception as e:
            fail += 1
            if fail <= 5:
                print(f"  Skip: {row['slice_file_name']} - {e}")
        if (idx + 1) % 1000 == 0:
            print(f"  {idx+1}/{len(meta)} 처리...")

    df = pd.DataFrame(rows)
    feature_cols = [f"feat_{i}" for i in range(INPUT_DIM)]
    df = df[feature_cols + ["slice_file_name", "fold", "classID", "class"]]
    df.to_csv(FEAT_PATH, index=False, encoding="utf-8-sig")
    print(f"[Feature] 추출 완료: 성공 {len(df)}개, 실패 {fail}개, "
          f"{(time.time()-t0)/60:.1f}분 → 캐시 저장 {FEAT_PATH}")
    return df


# ============================================================
# 2. MLP 모델
# ============================================================
class MLPClassifier(nn.Module):
    def __init__(self, input_dim=256, num_classes=10):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        return self.network(x)


def normalize(X_train, X_val, X_test):
    mean, std = X_train.mean(axis=0), X_train.std(axis=0)
    std[std == 0] = 1
    return ((X_train - mean) / std,
            (X_val - mean) / std,
            (X_test - mean) / std)


def train_and_eval(X_train, y_train, X_val, y_val, X_test, y_test):
    """1차-B: train으로 학습, val로 early stopping, test로 평가"""
    train_loader = DataLoader(
        TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train)),
        batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(
        TensorDataset(torch.FloatTensor(X_val), torch.LongTensor(y_val)),
        batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(
        TensorDataset(torch.FloatTensor(X_test), torch.LongTensor(y_test)),
        batch_size=BATCH_SIZE, shuffle=False)

    model = MLPClassifier(INPUT_DIM, NUM_CLASSES).to(device)
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
# 3. 평가 지표 (12개 항목)
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
# 메인
# ============================================================
def main():
    print("=" * 60)
    print("MLP 음향 분류 (Train fold1~8 / Val fold9 / Test fold10)")
    print("=" * 60)
    print(f"Device: {device}\n")

    # 1) feature 준비 (없으면 추출)
    df = build_features()
    feature_cols = [c for c in df.columns if c.startswith("feat_")]
    class_names = sorted(df["class"].unique())
    X_all = df[feature_cols].values.astype(np.float32)
    y_all = df["classID"].values.astype(np.int64)
    fold = df["fold"].values

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

    pd.DataFrame([metrics]).to_csv(OUT_DIR / "mlp_metrics.csv",
                                   index=False, encoding="utf-8-sig")

    # class별 precision/recall/f1
    p, r, f, s = precision_recall_fscore_support(
        y_true, y_pred, labels=range(NUM_CLASSES), zero_division=0)
    cw = pd.DataFrame({"class": class_names, "precision": p, "recall": r,
                       "f1": f, "support": s}).sort_values("f1", ascending=False)
    cw.to_csv(OUT_DIR / "mlp_classwise_metrics.csv", index=False, encoding="utf-8-sig")
    print("\n[클래스별 지표]")
    print(cw.round(4).to_string(index=False))

    # 샘플별 예측값
    pd.DataFrame({"y_true": y_true, "y_pred": y_pred}).to_csv(
        OUT_DIR / "mlp_predictions.csv", index=False, encoding="utf-8-sig")

    # 체크포인트 저장 (학습된 best 모델)
    ckpt_path = OUT_DIR / "mlp_best.pt"
    torch.save({
        "model_state_dict": model.state_dict(),
        "input_dim": INPUT_DIM,
        "num_classes": NUM_CLASSES,
        "class_names": class_names,
        "seed": SEED,
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
        plt.title("MLP Confusion Matrix (Test fold 10)")
        plt.tight_layout()
        plt.savefig(IMG_DIR / "mlp_confusion.png", dpi=150, bbox_inches="tight")
        plt.close()
    except Exception as e:
        print(f"  (confusion matrix 그림 생략: {e})")

    print(f"\n결과 저장: {OUT_DIR}")
    print("=== 완료 ===")


if __name__ == "__main__":
    main()

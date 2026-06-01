"""
AST(Audio Spectrogram Transformer) 음향 분류 (UrbanSound8K)
===========================================================
사전학습된 AST 모델을 UrbanSound8K에 적용한다.
- Feature Extraction: 본체(Transformer) 고정, 분류층만 학습
- Full Fine-tuning: 모델 전체 학습
두 방식을 12개 평가 지표로 비교한다.

분석 방식:
  - Train: fold 1~8 / Validation: fold 9 / Test: fold 10
  - Validation 기준으로 best 모델 선택 (Full FT는 early stopping)

평가 지표 (팀 합의 12개 항목):
  - 종합: accuracy, balanced accuracy, macro/weighted precision·recall·F1
  - class별: precision, recall, F1
  - confusion matrix

출력 (models/AST/output/):
  - ast_fe_best.pt           : Feature Extraction 분류층(head) 체크포인트
  - ast_full_ft_best.pt      : Full Fine-tuning 전체 모델 체크포인트 (약 329MB, git 제외)
  - ast_overall_metrics.csv  : FE vs Full 종합 지표 비교
  - ast_*_classwise_metrics.csv, ast_*_predictions.csv
  - ast_feature_ext_confusion.png, ast_full_ft_confusion.png, ast_training_curve.png

폴더 구조 (project_root 기준):
  project_root/
  ├── UrbanSound8K/
  │   ├── audio/
  │   └── metadata/
  └── models/
      └── AST/
          └── ast_model.py   ← 이 파일

실행 환경:
  - python=3.12
  - torch==2.11.0, torchaudio==2.11.0
  - transformers, librosa, scikit-learn, matplotlib, seaborn
  - GPU 권장 (없으면 CPU로 동작하나 Full Fine-tuning이 매우 느림)

실행:
  python models/AST/ast_model.py
"""

import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import librosa
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import ASTFeatureExtractor, ASTForAudioClassification
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, f1_score,
    precision_score, recall_score, confusion_matrix,
    precision_recall_fscore_support,
)
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")   # 화면 없는 환경에서도 그림 저장 가능

# ============================================================
# 경로 설정 (어느 환경에서나 동작하도록 상대 경로 사용)
# ============================================================
# 이 파일은 project_root/models/AST/ast_model.py 위치를 가정한다.
ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = ROOT / "UrbanSound8K"
AUDIO_DIR = DATA_DIR / "audio"
META_PATH = DATA_DIR / "metadata" / "UrbanSound8K.csv"

HERE = Path(__file__).resolve().parent
OUT_DIR = HERE / "output"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# 그림은 project_root/images/ 에 저장 (팀 공통, 접두사로 모델 구분)
IMG_DIR = ROOT / "images"
IMG_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# 설정
# ============================================================
MODEL_ID = "MIT/ast-finetuned-audioset-10-10-0.4593"
NUM_CLASSES = 10
SEED = 42

# Feature Extraction
FE_EPOCHS = 50
FE_LR = 1e-3

# Full Fine-tuning
FT_EPOCHS = 5
FT_LR = 1e-5
FT_BATCH = 16
FT_PATIENCE = 2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(SEED)
np.random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)


# ============================================================
# 데이터셋 (오디오 → 16kHz resample → AST 입력)
# ============================================================
class UrbanSoundDataset(Dataset):
    def __init__(self, df, feature_extractor, target_sr):
        self.df = df.reset_index(drop=True)
        self.fe = feature_extractor
        self.sr = target_sr

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        path = AUDIO_DIR / f"fold{int(row['fold'])}" / str(row['slice_file_name'])
        wav, _ = librosa.load(str(path), sr=self.sr, mono=True)
        inputs = self.fe(wav, sampling_rate=self.sr, return_tensors="pt")
        return {
            "input_values": inputs["input_values"].squeeze(0),
            "labels": torch.tensor(int(row['classID']), dtype=torch.long),
        }


# ============================================================
# 평가 지표 (12개 항목)
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


def save_classwise(y_true, y_pred, class_names, tag):
    p, r, f, s = precision_recall_fscore_support(
        y_true, y_pred, labels=range(NUM_CLASSES), zero_division=0)
    cw = pd.DataFrame({"class": class_names, "precision": p, "recall": r,
                       "f1": f, "support": s}).sort_values("f1", ascending=False)
    cw.to_csv(OUT_DIR / f"ast_{tag}_classwise_metrics.csv",
              index=False, encoding="utf-8-sig")
    pd.DataFrame({"y_true": y_true, "y_pred": y_pred}).to_csv(
        OUT_DIR / f"ast_{tag}_predictions.csv", index=False, encoding="utf-8-sig")
    return cw


def save_confusion(y_true, y_pred, class_names, tag, title):
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
    plt.xlabel("Predicted"); plt.ylabel("True"); plt.title(title)
    plt.tight_layout()
    plt.savefig(IMG_DIR / f"ast_{tag}_confusion.png", dpi=150, bbox_inches="tight")
    plt.close()


# ============================================================
# Feature Extraction
# ============================================================
@torch.no_grad()
def extract_embeddings(model, loader):
    model.eval()
    embs, labels = [], []
    for batch in loader:
        x = batch["input_values"].to(device)
        out = model.audio_spectrogram_transformer(x)
        embs.append(out.pooler_output.cpu())
        labels.append(batch["labels"])
    return torch.cat(embs), torch.cat(labels)


def run_feature_extraction(model, train_ds, val_ds, test_ds, class_names):
    print("\n" + "=" * 60)
    print("[1] Feature Extraction (본체 고정, 분류층만 학습)")
    print("=" * 60)

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=False, num_workers=2)
    val_loader   = DataLoader(val_ds,   batch_size=32, shuffle=False, num_workers=2)
    test_loader  = DataLoader(test_ds,  batch_size=32, shuffle=False, num_workers=2)

    print("Embedding 추출 중... (본체 1회 통과)")
    train_emb, train_y = extract_embeddings(model, train_loader)
    val_emb, val_y     = extract_embeddings(model, val_loader)
    test_emb, test_y   = extract_embeddings(model, test_loader)

    torch.manual_seed(SEED)
    head = nn.Sequential(nn.LayerNorm(768), nn.Linear(768, NUM_CLASSES)).to(device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=FE_LR)
    criterion = nn.CrossEntropyLoss()

    Xtr, ytr = train_emb.to(device), train_y.to(device)
    Xva, yva = val_emb.to(device), val_y.to(device)
    Xte = test_emb.to(device)

    best_val_acc, best_state = 0.0, None
    for epoch in range(1, FE_EPOCHS + 1):
        head.train()
        optimizer.zero_grad()
        loss = criterion(head(Xtr), ytr)
        loss.backward()
        optimizer.step()
        head.eval()
        with torch.no_grad():
            val_acc = (head(Xva).argmax(1) == yva).float().mean().item()
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.cpu().clone() for k, v in head.state_dict().items()}
        if epoch % 10 == 0 or epoch == 1:
            print(f"  Epoch {epoch:2d} | train_loss {loss.item():.4f} | val_acc {val_acc:.4f}")

    head.load_state_dict(best_state)
    head.eval()
    with torch.no_grad():
        test_pred = head(Xte).argmax(1).cpu().numpy()
    y_true = test_y.numpy()

    metrics = compute_metrics(y_true, test_pred)
    metrics["best_val_acc"] = best_val_acc
    print(f"\n  [FE] accuracy {metrics['accuracy']:.4f} | macro_f1 {metrics['macro_f1']:.4f}")

    # 체크포인트 저장 (학습된 분류층 head)
    torch.save({
        "head_state_dict": head.state_dict(),
        "model_id": MODEL_ID,
        "num_classes": NUM_CLASSES,
        "class_names": class_names,
        "seed": SEED,
        "best_val_acc": best_val_acc,
        "metrics": metrics,
        "note": "Feature Extraction: AST 본체는 사전학습 그대로, 이 head만 학습됨",
    }, OUT_DIR / "ast_fe_best.pt")
    print(f"  체크포인트 저장: {OUT_DIR / 'ast_fe_best.pt'}")

    save_classwise(y_true, test_pred, class_names, "feature_ext")
    save_confusion(y_true, test_pred, class_names, "feature_ext",
                   "AST Feature Extraction - Confusion Matrix (Test fold 10)")
    return metrics


# ============================================================
# Full Fine-tuning
# ============================================================
def run_full_finetuning(train_ds, val_ds, test_ds, id2label, label2id, class_names):
    print("\n" + "=" * 60)
    print("[2] Full Fine-tuning (모델 전체 학습)")
    print("=" * 60)

    torch.manual_seed(SEED)
    model = ASTForAudioClassification.from_pretrained(
        MODEL_ID, num_labels=NUM_CLASSES, ignore_mismatched_sizes=True).to(device)
    model.config.id2label = id2label
    model.config.label2id = label2id
    for p in model.parameters():
        p.requires_grad = True

    train_loader = DataLoader(train_ds, batch_size=FT_BATCH, shuffle=True,  num_workers=2)
    val_loader   = DataLoader(val_ds,   batch_size=FT_BATCH, shuffle=False, num_workers=2)
    test_loader  = DataLoader(test_ds,  batch_size=FT_BATCH, shuffle=False, num_workers=2)

    optimizer = torch.optim.AdamW(model.parameters(), lr=FT_LR)
    criterion = nn.CrossEntropyLoss()

    @torch.no_grad()
    def evaluate(loader):
        model.eval()
        preds, trues, losses = [], [], []
        for batch in loader:
            x = batch["input_values"].to(device)
            y = batch["labels"].to(device)
            logits = model(x).logits
            losses.append(criterion(logits, y).item())
            preds.append(logits.argmax(1).cpu()); trues.append(y.cpu())
        return (np.mean(losses),
                accuracy_score(torch.cat(trues), torch.cat(preds)),
                torch.cat(preds).numpy(), torch.cat(trues).numpy())

    best_val_acc, best_state, no_improve = 0.0, None, 0
    history = {"train_loss": [], "val_loss": [], "val_acc": []}

    for epoch in range(1, FT_EPOCHS + 1):
        model.train()
        t0, ep_losses = time.time(), []
        for batch in train_loader:
            x = batch["input_values"].to(device)
            y = batch["labels"].to(device)
            optimizer.zero_grad()
            loss = criterion(model(x).logits, y)
            loss.backward()
            optimizer.step()
            ep_losses.append(loss.item())

        train_loss = float(np.mean(ep_losses))
        val_loss, val_acc, _, _ = evaluate(val_loader)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        print(f"  Epoch {epoch} | train_loss {train_loss:.4f} | val_loss {val_loss:.4f} "
              f"| val_acc {val_acc:.4f} | {time.time()-t0:.0f}s")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= FT_PATIENCE:
                print(f"  Early stopping at epoch {epoch}")
                break

    model.load_state_dict(best_state)
    _, _, test_pred, y_true = evaluate(test_loader)

    metrics = compute_metrics(y_true, test_pred)
    metrics["best_val_acc"] = best_val_acc
    print(f"\n  [Full FT] accuracy {metrics['accuracy']:.4f} | macro_f1 {metrics['macro_f1']:.4f}")

    save_classwise(y_true, test_pred, class_names, "full_ft")
    save_confusion(y_true, test_pred, class_names, "full_ft",
                   "AST Full Fine-tuning - Confusion Matrix (Test fold 10)")

    # 체크포인트 저장 (학습된 전체 모델)
    ckpt_path = OUT_DIR / "ast_full_ft_best.pt"
    torch.save({
        "model_state_dict": model.state_dict(),
        "model_id": MODEL_ID,
        "num_classes": NUM_CLASSES,
        "class_names": class_names,
        "id2label": id2label,
        "best_val_acc": best_val_acc,
    }, ckpt_path)
    print(f"  체크포인트 저장: {ckpt_path}")

    # 학습 곡선
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    ax1.plot(epochs, history["train_loss"], 'o-', label='Train Loss')
    ax1.plot(epochs, history["val_loss"], 's-', label='Val Loss')
    ax1.set_xlabel('Epoch'); ax1.set_ylabel('Loss'); ax1.set_title('Loss')
    ax1.legend(); ax1.grid(alpha=0.3)
    ax2.plot(epochs, history["val_acc"], 'o-', color='green', label='Val Accuracy')
    ax2.set_xlabel('Epoch'); ax2.set_ylabel('Accuracy'); ax2.set_title('Val Accuracy')
    ax2.legend(); ax2.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(IMG_DIR / "ast_training_curve.png", dpi=150, bbox_inches="tight")
    plt.close()

    return metrics


# ============================================================
# 메인
# ============================================================
def main():
    print("=" * 60)
    print("AST 음향 분류 (Train fold1~8 / Val fold9 / Test fold10)")
    print("=" * 60)
    print(f"Device: {device}")

    if not META_PATH.exists():
        raise SystemExit(f"[ERROR] 데이터 없음: {META_PATH}\n"
                         f"UrbanSound8K 를 {DATA_DIR} 에 배치했는지 확인하세요.")

    meta = pd.read_csv(META_PATH)

    # 모델 + feature extractor
    feature_extractor = ASTFeatureExtractor.from_pretrained(MODEL_ID)
    target_sr = feature_extractor.sampling_rate

    base_model = ASTForAudioClassification.from_pretrained(
        MODEL_ID, num_labels=NUM_CLASSES, ignore_mismatched_sizes=True).to(device)

    # 라벨 매핑
    id2label = (meta[['classID', 'class']].drop_duplicates()
                .sort_values('classID').set_index('classID')['class'].to_dict())
    id2label = {int(k): v for k, v in id2label.items()}
    label2id = {v: k for k, v in id2label.items()}
    base_model.config.id2label = id2label
    base_model.config.label2id = label2id
    class_names = [id2label[i] for i in range(NUM_CLASSES)]

    # fold 분할 (Train 1~8 / Val 9 / Test 10)
    train_df = meta[meta['fold'].isin(range(1, 9))]
    val_df   = meta[meta['fold'] == 9]
    test_df  = meta[meta['fold'] == 10]
    print(f"Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")

    train_ds = UrbanSoundDataset(train_df, feature_extractor, target_sr)
    val_ds   = UrbanSoundDataset(val_df, feature_extractor, target_sr)
    test_ds  = UrbanSoundDataset(test_df, feature_extractor, target_sr)

    # [1] Feature Extraction
    fe_metrics = run_feature_extraction(base_model, train_ds, val_ds, test_ds, class_names)

    # [2] Full Fine-tuning
    ft_metrics = run_full_finetuning(train_ds, val_ds, test_ds, id2label, label2id, class_names)

    # 종합 지표 비교 저장
    summary = pd.DataFrame({
        "지표": list(fe_metrics.keys()),
        "Feature_Extraction": [round(fe_metrics[k], 4) for k in fe_metrics],
        "Full_Fine_tuning":   [round(ft_metrics[k], 4) for k in ft_metrics],
    })
    summary.to_csv(OUT_DIR / "ast_overall_metrics.csv", index=False, encoding="utf-8-sig")

    print("\n" + "=" * 60)
    print("종합 평가 지표 (FE vs Full Fine-tuning)")
    print("=" * 60)
    print(summary.to_string(index=False))
    print(f"\n결과 저장: {OUT_DIR}")
    print("=== 완료 ===")


if __name__ == "__main__":
    main()

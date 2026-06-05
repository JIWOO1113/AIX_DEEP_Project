"""
AST Full Fine-tuning for UrbanSound8K
=====================================
사전학습된 AST 전체 파라미터와 classifier head를 함께 학습한다.

출력:
  - models/AST/ast_full_ft_best.pt
  - models/AST/final/full_ft/overall_metrics.csv
  - models/AST/final/full_ft/classwise_metrics.csv
  - models/AST/final/full_ft/predictions.csv
  - models/AST/final/full_ft/training_curve.png
  - images/ast_full_ft_confusion_matrix.png
"""

import time
from pathlib import Path

import librosa
import matplotlib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    precision_score,
    recall_score,
)
from torch.utils.data import DataLoader, Dataset
from transformers import ASTFeatureExtractor, ASTForAudioClassification

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = ROOT / "UrbanSound8K"
AUDIO_DIR = DATA_DIR / "audio"
META_PATH = DATA_DIR / "metadata" / "UrbanSound8K.csv"

HERE = Path(__file__).resolve().parent
FINAL_DIR = HERE / "final"
IMAGE_DIR = Path(__file__).parent / ".." / ".." / "images"

MODEL_ID = "MIT/ast-finetuned-audioset-10-10-0.4593"
NUM_CLASSES = 10
SEED = 42
FT_EPOCHS = 5
FT_LR = 1e-5
FT_BATCH = 16
FT_PATIENCE = 2
METHOD = "full_ft"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class UrbanSoundDataset(Dataset):
    def __init__(self, df, feature_extractor, target_sr):
        self.df = df.reset_index(drop=True)
        self.fe = feature_extractor
        self.sr = target_sr

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        path = AUDIO_DIR / f"fold{int(row['fold'])}" / str(row["slice_file_name"])
        wav, _ = librosa.load(str(path), sr=self.sr, mono=True)
        inputs = self.fe(wav, sampling_rate=self.sr, return_tensors="pt")
        return {
            "input_values": inputs["input_values"].squeeze(0),
            "labels": torch.tensor(int(row["classID"]), dtype=torch.long),
        }


def seed_everything(seed=SEED):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


def prepare_dirs():
    (FINAL_DIR / METHOD).mkdir(parents=True, exist_ok=True)
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)


def load_metadata():
    if not META_PATH.exists():
        raise SystemExit(
            f"[ERROR] 데이터 없음: {META_PATH}\n"
            f"UrbanSound8K 를 {DATA_DIR} 에 배치했는지 확인하세요."
        )
    return pd.read_csv(META_PATH)


def build_splits(feature_extractor):
    meta = load_metadata()
    id2label = (
        meta[["classID", "class"]]
        .drop_duplicates()
        .sort_values("classID")
        .set_index("classID")["class"]
        .to_dict()
    )
    id2label = {int(k): v for k, v in id2label.items()}
    label2id = {v: k for k, v in id2label.items()}
    class_names = [id2label[i] for i in range(NUM_CLASSES)]

    train_df = meta[meta["fold"].isin(range(1, 9))]
    val_df = meta[meta["fold"] == 9]
    test_df = meta[meta["fold"] == 10]

    target_sr = feature_extractor.sampling_rate
    return {
        "id2label": id2label,
        "label2id": label2id,
        "class_names": class_names,
        "train_ds": UrbanSoundDataset(train_df, feature_extractor, target_sr),
        "val_ds": UrbanSoundDataset(val_df, feature_extractor, target_sr),
        "test_ds": UrbanSoundDataset(test_df, feature_extractor, target_sr),
        "split_sizes": (len(train_df), len(val_df), len(test_df)),
    }


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


def save_outputs(output_dir, metrics, y_true, y_pred, class_names):
    output_dir.mkdir(parents=True, exist_ok=True)
    row = {
        "model": "AST",
        "method": METHOD,
        "selection_metric": "validation_accuracy",
        "best_validation_score": metrics["best_val_acc"],
        "best_params": f"model_id={MODEL_ID}; mode=full_finetuning",
        "test_loss": metrics["test_loss"],
        "final_epoch": metrics["final_epoch"],
        **{k: v for k, v in metrics.items() if k not in {"best_val_acc", "test_loss", "final_epoch"}},
    }
    pd.DataFrame([row]).to_csv(output_dir / "overall_metrics.csv", index=False, encoding="utf-8-sig")

    p, r, f, s = precision_recall_fscore_support(
        y_true, y_pred, labels=range(NUM_CLASSES), zero_division=0
    )
    classwise = pd.DataFrame({
        "model": "AST",
        "method": METHOD,
        "class": class_names,
        "precision": p,
        "recall": r,
        "f1": f,
        "support": s,
    })
    classwise.to_csv(output_dir / "classwise_metrics.csv", index=False, encoding="utf-8-sig")

    predictions = pd.DataFrame({
        "model": "AST",
        "method": METHOD,
        "y_true": y_true,
        "y_pred": y_pred,
        "true_label": [class_names[int(i)] for i in y_true],
        "pred_label": [class_names[int(i)] for i in y_pred],
    })
    predictions.to_csv(output_dir / "predictions.csv", index=False, encoding="utf-8-sig")


def save_confusion_image(y_true, y_pred, class_names):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    plt.imshow(cm, cmap="Blues")
    plt.colorbar()
    plt.xticks(range(NUM_CLASSES), class_names, rotation=45, ha="right")
    plt.yticks(range(NUM_CLASSES), class_names)
    threshold = cm.max() / 2 if cm.size else 0
    for i in range(NUM_CLASSES):
        for j in range(NUM_CLASSES):
            plt.text(
                j,
                i,
                cm[i, j],
                ha="center",
                va="center",
                color="white" if cm[i, j] > threshold else "black",
            )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("AST Full Fine-tuning - Confusion Matrix (Test fold 10)")
    plt.tight_layout()
    image_path = IMAGE_DIR / "ast_full_ft_confusion_matrix.png"
    plt.savefig(image_path, dpi=150, bbox_inches="tight")
    plt.close()
    return image_path


def save_training_curve(history, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    ax1.plot(epochs, history["train_loss"], "o-", label="Train Loss")
    ax1.plot(epochs, history["val_loss"], "s-", label="Val Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Loss")
    ax1.legend()
    ax1.grid(alpha=0.3)
    ax2.plot(epochs, history["val_acc"], "o-", color="green", label="Val Accuracy")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.set_title("Val Accuracy")
    ax2.legend()
    ax2.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()


@torch.no_grad()
def evaluate(model, loader, criterion):
    model.eval()
    preds, trues, losses = [], [], []
    for batch in loader:
        x = batch["input_values"].to(device)
        y = batch["labels"].to(device)
        logits = model(x).logits
        losses.append(criterion(logits, y).item())
        preds.append(logits.argmax(1).cpu())
        trues.append(y.cpu())

    y_pred = torch.cat(preds).numpy()
    y_true = torch.cat(trues).numpy()
    return np.mean(losses), accuracy_score(y_true, y_pred), y_pred, y_true


def main():
    prepare_dirs()
    seed_everything(SEED)

    print("=" * 60)
    print("AST Full Fine-tuning (전체 모델 학습)")
    print("=" * 60)
    print(f"Device: {device}")

    feature_extractor = ASTFeatureExtractor.from_pretrained(MODEL_ID)
    split = build_splits(feature_extractor)
    train_n, val_n, test_n = split["split_sizes"]
    print(f"Train: {train_n} | Val: {val_n} | Test: {test_n}")

    model = ASTForAudioClassification.from_pretrained(
        MODEL_ID,
        num_labels=NUM_CLASSES,
        ignore_mismatched_sizes=True,
    ).to(device)
    model.config.id2label = split["id2label"]
    model.config.label2id = split["label2id"]
    for param in model.parameters():
        param.requires_grad = True

    train_loader = DataLoader(split["train_ds"], batch_size=FT_BATCH, shuffle=True, num_workers=2)
    val_loader = DataLoader(split["val_ds"], batch_size=FT_BATCH, shuffle=False, num_workers=2)
    test_loader = DataLoader(split["test_ds"], batch_size=FT_BATCH, shuffle=False, num_workers=2)

    optimizer = torch.optim.AdamW(model.parameters(), lr=FT_LR)
    criterion = nn.CrossEntropyLoss()

    best_val_acc = -1.0
    best_state = None
    no_improve = 0
    history = {"train_loss": [], "val_loss": [], "val_acc": []}

    for epoch in range(1, FT_EPOCHS + 1):
        model.train()
        start = time.time()
        epoch_losses = []
        for batch in train_loader:
            x = batch["input_values"].to(device)
            y = batch["labels"].to(device)
            optimizer.zero_grad()
            loss = criterion(model(x).logits, y)
            loss.backward()
            optimizer.step()
            epoch_losses.append(loss.item())

        train_loss = float(np.mean(epoch_losses))
        val_loss, val_acc, _, _ = evaluate(model, val_loader, criterion)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        print(
            f"Epoch {epoch} | train_loss={train_loss:.4f} | "
            f"val_loss={val_loss:.4f} | val_acc={val_acc:.4f} | {time.time() - start:.0f}s"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= FT_PATIENCE:
                print(f"Early stopping at epoch {epoch}")
                break

    model.load_state_dict(best_state)
    test_loss, _, test_pred, y_true = evaluate(model, test_loader, criterion)

    metrics = compute_metrics(y_true, test_pred)
    metrics["best_val_acc"] = best_val_acc
    metrics["test_loss"] = test_loss
    metrics["final_epoch"] = len(history["train_loss"])
    print(f"Test accuracy={metrics['accuracy']:.4f} | macro_f1={metrics['macro_f1']:.4f}")

    checkpoint_path = HERE / "ast_full_ft_best.pt"
    torch.save({
        "model_state_dict": model.state_dict(),
        "model_id": MODEL_ID,
        "num_classes": NUM_CLASSES,
        "class_names": split["class_names"],
        "id2label": split["id2label"],
        "best_val_acc": best_val_acc,
        "metrics": metrics,
    }, checkpoint_path)

    output_dir = FINAL_DIR / METHOD
    save_outputs(output_dir, metrics, y_true, test_pred, split["class_names"])
    save_training_curve(history, output_dir / "training_curve.png")
    image_path = save_confusion_image(y_true, test_pred, split["class_names"])

    print(f"체크포인트 저장: {checkpoint_path}")
    print(f"최종 결과 저장: {output_dir}")
    print(f"confusion matrix 이미지 저장: {image_path}")


if __name__ == "__main__":
    main()

"""
AST Feature Extraction for UrbanSound8K
=======================================
사전학습된 AST 본체는 고정하고 classifier head만 학습한다.

출력:
  - models/AST/ast_fe_best.pt
  - models/AST/final/feature_ext/overall_metrics.csv
  - models/AST/final/feature_ext/classwise_metrics.csv
  - models/AST/final/feature_ext/predictions.csv
  - images/ast_feature_ext_confusion_matrix.png
"""

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
FE_EPOCHS = 50
FE_LR = 1e-3
FE_BATCH = 32
METHOD = "feature_ext"

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
        "best_params": f"model_id={MODEL_ID}; mode=feature_extraction",
        "test_loss": np.nan,
        "final_epoch": FE_EPOCHS,
        **{k: v for k, v in metrics.items() if k != "best_val_acc"},
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
    plt.title("AST Feature Extraction - Confusion Matrix (Test fold 10)")
    plt.tight_layout()
    image_path = IMAGE_DIR / "ast_feature_ext_confusion_matrix.png"
    plt.savefig(image_path, dpi=150, bbox_inches="tight")
    plt.close()
    return image_path


@torch.no_grad()
def extract_embeddings(model, loader):
    model.eval()
    embeddings, labels = [], []
    for batch in loader:
        x = batch["input_values"].to(device)
        out = model.audio_spectrogram_transformer(x)
        embeddings.append(out.pooler_output.cpu())
        labels.append(batch["labels"])
    return torch.cat(embeddings), torch.cat(labels)


def main():
    prepare_dirs()
    seed_everything(SEED)

    print("=" * 60)
    print("AST Feature Extraction (head만 학습)")
    print("=" * 60)
    print(f"Device: {device}")

    feature_extractor = ASTFeatureExtractor.from_pretrained(MODEL_ID)
    split = build_splits(feature_extractor)
    train_n, val_n, test_n = split["split_sizes"]
    print(f"Train: {train_n} | Val: {val_n} | Test: {test_n}")

    base_model = ASTForAudioClassification.from_pretrained(
        MODEL_ID,
        num_labels=NUM_CLASSES,
        ignore_mismatched_sizes=True,
    ).to(device)
    base_model.config.id2label = split["id2label"]
    base_model.config.label2id = split["label2id"]

    train_loader = DataLoader(split["train_ds"], batch_size=FE_BATCH, shuffle=False, num_workers=2)
    val_loader = DataLoader(split["val_ds"], batch_size=FE_BATCH, shuffle=False, num_workers=2)
    test_loader = DataLoader(split["test_ds"], batch_size=FE_BATCH, shuffle=False, num_workers=2)

    print("Embedding 추출 중... (AST 본체 1회 통과)")
    train_emb, train_y = extract_embeddings(base_model, train_loader)
    val_emb, val_y = extract_embeddings(base_model, val_loader)
    test_emb, test_y = extract_embeddings(base_model, test_loader)

    torch.manual_seed(SEED)
    head = nn.Sequential(nn.LayerNorm(768), nn.Linear(768, NUM_CLASSES)).to(device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=FE_LR)
    criterion = nn.CrossEntropyLoss()

    x_train, y_train = train_emb.to(device), train_y.to(device)
    x_val, y_val = val_emb.to(device), val_y.to(device)
    x_test = test_emb.to(device)

    best_val_acc = -1.0
    best_state = None
    for epoch in range(1, FE_EPOCHS + 1):
        head.train()
        optimizer.zero_grad()
        loss = criterion(head(x_train), y_train)
        loss.backward()
        optimizer.step()

        head.eval()
        with torch.no_grad():
            val_acc = (head(x_val).argmax(1) == y_val).float().mean().item()

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.cpu().clone() for k, v in head.state_dict().items()}

        if epoch == 1 or epoch % 10 == 0:
            print(f"Epoch {epoch:02d} | train_loss={loss.item():.4f} | val_acc={val_acc:.4f}")

    head.load_state_dict(best_state)
    head.eval()
    with torch.no_grad():
        test_pred = head(x_test).argmax(1).cpu().numpy()
    y_true = test_y.numpy()

    metrics = compute_metrics(y_true, test_pred)
    metrics["best_val_acc"] = best_val_acc
    print(f"Test accuracy={metrics['accuracy']:.4f} | macro_f1={metrics['macro_f1']:.4f}")

    checkpoint_path = HERE / "ast_fe_best.pt"
    torch.save({
        "head_state_dict": head.state_dict(),
        "model_id": MODEL_ID,
        "num_classes": NUM_CLASSES,
        "class_names": split["class_names"],
        "seed": SEED,
        "best_val_acc": best_val_acc,
        "metrics": metrics,
        "note": "Feature Extraction: AST 본체는 사전학습 그대로, 이 head만 학습됨",
    }, checkpoint_path)

    output_dir = FINAL_DIR / METHOD
    save_outputs(output_dir, metrics, y_true, test_pred, split["class_names"])
    image_path = save_confusion_image(y_true, test_pred, split["class_names"])

    print(f"체크포인트 저장: {checkpoint_path}")
    print(f"최종 결과 저장: {output_dir}")
    print(f"confusion matrix 이미지 저장: {image_path}")


if __name__ == "__main__":
    main()

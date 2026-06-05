from __future__ import annotations

import argparse
import json
import math
import os
import random
from dataclasses import asdict, dataclass
from hashlib import sha1
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from torch.utils.data import DataLoader, Dataset


CLASS_NAMES = [
    "air_conditioner",
    "car_horn",
    "children_playing",
    "dog_bark",
    "drilling",
    "engine_idling",
    "gun_shot",
    "jackhammer",
    "siren",
    "street_music",
]


# 전처리 파라미터와 UrbanSound8K fold 분할 기준을 한곳에 모아 둔다.
@dataclass(frozen=True)
class PreprocessConfig:
    sample_rate: int = 22050
    duration: float = 4.0
    padding: str = "repeat"
    n_mels: int = 128
    n_fft: int = 1024
    hop_length: int = 512
    fmin: float = 0.0
    fmax: float = 0.0
    top_db: float = 80.0
    train_folds: Tuple[int, ...] = (1, 2, 3, 4, 5, 6, 7, 8)
    val_fold: int = 9
    test_fold: int = 10

    @property
    def target_length(self) -> int:
        return int(round(self.sample_rate * self.duration))

    @property
    def fmax_value(self) -> float | None:
        return None if self.fmax <= 0 else self.fmax

    @property
    def top_db_value(self) -> float | None:
        return None if self.top_db <= 0 else self.top_db


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def seed_everything(seed: int) -> None:
    # Python, NumPy, PyTorch 난수를 고정해서 실험 재현성을 맞춘다.
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def cache_key(config: PreprocessConfig) -> str:
    # 전처리 설정이 같으면 같은 캐시 폴더를 재사용하도록 짧은 해시를 만든다.
    payload = json.dumps(asdict(config), sort_keys=True)
    return sha1(payload.encode("utf-8")).hexdigest()[:12]


def split_from_fold(fold: int, config: PreprocessConfig) -> str:
    # UrbanSound8K fold 번호를 train/val/test split 이름으로 바꾼다.
    if fold in config.train_folds:
        return "train"
    if fold == config.val_fold:
        return "val"
    if fold == config.test_fold:
        return "test"
    raise ValueError(f"Unsupported fold {fold}. Expected train/val/test folds only.")


def require_librosa():
    try:
        import librosa
    except ImportError as exc:
        raise SystemExit(
            "librosa is required for preprocessing. Install it with: pip install librosa"
        ) from exc
    return librosa


def fix_audio_length(y: np.ndarray, target_length: int, padding: str) -> np.ndarray:
    # 모든 오디오를 같은 길이로 잘라내거나 padding해서 입력 크기를 통일한다.
    y = np.asarray(y, dtype=np.float32)

    if len(y) == target_length:
        return y

    if len(y) > target_length:
        return y[:target_length]

    missing = target_length - len(y)
    if len(y) == 0:
        return np.zeros(target_length, dtype=np.float32)

    if padding == "repeat":
        repeats = math.ceil(target_length / len(y))
        return np.tile(y, repeats)[:target_length].astype(np.float32, copy=False)

    if padding == "zero":
        return np.pad(y, (0, missing), mode="constant").astype(np.float32, copy=False)

    raise ValueError(f"Unknown padding mode: {padding}")


def extract_logmel(audio_path: Path, config: PreprocessConfig) -> np.ndarray:
    # 오디오 파일 하나를 모델 입력으로 사용할 log-mel spectrogram으로 변환한다.
    librosa = require_librosa()
    y, _ = librosa.load(audio_path, sr=config.sample_rate, mono=True)
    y = fix_audio_length(y, config.target_length, config.padding)

    mel = librosa.feature.melspectrogram(
        y=y,
        sr=config.sample_rate,
        n_fft=config.n_fft,
        hop_length=config.hop_length,
        n_mels=config.n_mels,
        fmin=config.fmin,
        fmax=config.fmax_value,
        power=2.0,
    )
    logmel = librosa.power_to_db(
        mel,
        ref=1.0,
        amin=1e-10,
        top_db=config.top_db_value,
    )
    return logmel.astype(np.float32, copy=False)


def read_metadata(data_root: Path) -> pd.DataFrame:
    # UrbanSound8K metadata CSV를 읽고 필요한 컬럼이 있는지 확인한다.
    metadata_path = data_root / "metadata" / "UrbanSound8K.csv"
    if not metadata_path.exists():
        raise FileNotFoundError(
            f"Metadata CSV not found: {metadata_path}. "
            "Pass --data-root pointing to the UrbanSound8K directory."
        )

    metadata = pd.read_csv(metadata_path)
    required = {"slice_file_name", "fold", "classID", "class"}
    missing = required.difference(metadata.columns)
    if missing:
        raise ValueError(f"Metadata is missing required columns: {sorted(missing)}")

    return metadata.sort_values(["fold", "slice_file_name"]).reset_index(drop=True)


def build_manifest(
    data_root: Path,
    cache_dir: Path,
    config: PreprocessConfig,
    force: bool = False,
) -> pd.DataFrame:
    # raw audio를 feature cache로 변환하면서 학습에 쓸 manifest를 만든다.
    metadata = read_metadata(data_root)
    rows: List[Dict[str, object]] = []

    for index, row in metadata.iterrows():
        fold = int(row["fold"])
        split = split_from_fold(fold, config)
        filename = str(row["slice_file_name"])
        audio_path = data_root / "audio" / f"fold{fold}" / filename
        feature_rel_path = Path("features") / f"fold{fold}" / f"{Path(filename).stem}.npy"
        feature_path = cache_dir / feature_rel_path

        rows.append(
            {
                "slice_file_name": filename,
                "fold": fold,
                "classID": int(row["classID"]),
                "class": str(row["class"]),
                "split": split,
                "feature_path": feature_rel_path.as_posix(),
            }
        )

        if feature_path.exists() and not force:
            continue

        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        feature_path.parent.mkdir(parents=True, exist_ok=True)
        logmel = extract_logmel(audio_path, config)
        np.save(feature_path, logmel)

        if (index + 1) % 250 == 0:
            print(f"preprocessed {index + 1}/{len(metadata)} files")

    manifest = pd.DataFrame(rows)
    manifest_path = cache_dir / "manifest.csv"
    manifest.to_csv(manifest_path, index=False)
    return manifest


def compute_train_stats(cache_dir: Path, manifest: pd.DataFrame) -> Dict[str, float]:
    # 데이터 누수를 막기 위해 train split feature만으로 정규화 통계를 계산한다.
    train_df = manifest[manifest["split"] == "train"]
    if train_df.empty:
        raise ValueError("Train split is empty. Check fold settings.")

    total = 0.0
    total_sq = 0.0
    count = 0

    for rel_path in train_df["feature_path"]:
        x = np.load(cache_dir / rel_path)
        total += float(x.sum(dtype=np.float64))
        total_sq += float(np.square(x, dtype=np.float64).sum(dtype=np.float64))
        count += int(x.size)

    mean = total / count
    variance = max(total_sq / count - mean * mean, 1e-12)
    std = math.sqrt(variance)

    stats = {"mean": float(mean), "std": float(std)}
    np.savez(cache_dir / "stats.npz", mean=stats["mean"], std=stats["std"])
    return stats


def ensure_preprocessed(
    data_root: Path,
    cache_root: Path,
    config: PreprocessConfig,
    force: bool = False,
) -> Tuple[Path, pd.DataFrame, Dict[str, float]]:
    # 이미 만든 전처리 결과가 있으면 재사용하고, 없으면 새로 생성한다.
    cache_dir = cache_root / f"logmel_{cache_key(config)}"
    cache_dir.mkdir(parents=True, exist_ok=True)

    config_path = cache_dir / "preprocess_config.json"
    config_path.write_text(json.dumps(asdict(config), indent=2), encoding="utf-8")

    manifest_path = cache_dir / "manifest.csv"
    stats_path = cache_dir / "stats.npz"

    if force or not manifest_path.exists():
        print(f"building log-mel cache: {cache_dir}")
        manifest = build_manifest(data_root, cache_dir, config, force=force)
    else:
        manifest = pd.read_csv(manifest_path)

    if force or not stats_path.exists():
        print("computing train-set normalization statistics")
        stats = compute_train_stats(cache_dir, manifest)
    else:
        loaded = np.load(stats_path)
        stats = {"mean": float(loaded["mean"]), "std": float(loaded["std"])}

    return cache_dir, manifest, stats


class UrbanSoundLogMelDataset(Dataset):
    def __init__(
        self,
        manifest: pd.DataFrame,
        cache_dir: Path,
        mean: float,
        std: float,
    ) -> None:
        self.manifest = manifest.reset_index(drop=True)
        self.cache_dir = cache_dir
        self.mean = float(mean)
        self.std = max(float(std), 1e-8)

    def __len__(self) -> int:
        return len(self.manifest)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # 저장된 log-mel feature를 불러와 정규화한 뒤 채널 차원을 붙인다.
        row = self.manifest.iloc[index]
        x = np.load(self.cache_dir / row["feature_path"]).astype(np.float32, copy=False)
        x = (x - self.mean) / self.std
        x_tensor = torch.from_numpy(x).unsqueeze(0)
        y_tensor = torch.tensor(int(row["classID"]), dtype=torch.long)
        return x_tensor, y_tensor


class ConvBlock(nn.Module):
    # Conv-BatchNorm-ReLU-Dropout을 반복해서 쓰기 위한 CNN 기본 블록이다.
    def __init__(self, in_channels: int, out_channels: int, dropout: float) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class AudioRCNN(nn.Module):
    def __init__(
        self,
        n_mels: int,
        num_classes: int,
        hidden_size: int = 128,
        rnn_layers: int = 2,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        freq_after_pool = n_mels // 8
        if freq_after_pool <= 0:
            raise ValueError("n_mels must be at least 8 for the RCNN pooling stack.")

        # CNN은 주파수 축을 줄이면서 spectrogram의 지역 패턴을 뽑는다.
        self.cnn = nn.Sequential(
            ConvBlock(1, 32, dropout=0.05),
            nn.MaxPool2d(kernel_size=(2, 1)),
            ConvBlock(32, 64, dropout=0.10),
            nn.MaxPool2d(kernel_size=(2, 1)),
            ConvBlock(64, 128, dropout=0.15),
            nn.MaxPool2d(kernel_size=(2, 1)),
            ConvBlock(128, 128, dropout=0.15),
        )
        # GRU는 시간 축 순서를 따라 앞뒤 문맥을 함께 학습한다.
        self.rnn = nn.GRU(
            input_size=128 * freq_after_pool,
            hidden_size=hidden_size,
            num_layers=rnn_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if rnn_layers > 1 else 0.0,
        )
        # 평균 pooling과 max pooling을 합친 표현으로 최종 class 점수를 낸다.
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_size * 4),
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 4, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # CNN 출력의 시간 축을 sequence 차원으로 바꿔 RNN에 넣는다.
        x = self.cnn(x)
        x = x.permute(0, 3, 1, 2).contiguous()
        x = x.flatten(start_dim=2)
        x, _ = self.rnn(x)
        pooled = torch.cat([x.mean(dim=1), x.amax(dim=1)], dim=1)
        return self.classifier(pooled)


def make_loaders(
    cache_dir: Path,
    manifest: pd.DataFrame,
    stats: Dict[str, float],
    batch_size: int,
    num_workers: int,
    device: torch.device,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    # train/val/test Dataset을 같은 정규화 통계로 DataLoader에 묶는다.
    datasets = {}
    for split in ["train", "val", "test"]:
        split_df = manifest[manifest["split"] == split].reset_index(drop=True)
        datasets[split] = UrbanSoundLogMelDataset(
            split_df,
            cache_dir=cache_dir,
            mean=stats["mean"],
            std=stats["std"],
        )

    pin_memory = device.type == "cuda"
    train_loader = DataLoader(
        datasets["train"],
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        datasets["val"],
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        datasets["test"],
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    return train_loader, val_loader, test_loader


def class_weights_from_manifest(manifest: pd.DataFrame, device: torch.device) -> torch.Tensor:
    # class imbalance를 줄이기 위해 train label 빈도의 역수 기반 가중치를 만든다.
    train_labels = manifest.loc[manifest["split"] == "train", "classID"].to_numpy()
    counts = np.bincount(train_labels, minlength=len(CLASS_NAMES)).astype(np.float32)
    weights = counts.sum() / (len(CLASS_NAMES) * np.maximum(counts, 1.0))
    return torch.tensor(weights, dtype=torch.float32, device=device)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    # 전체 성능과 class imbalance를 반영한 macro/weighted 지표를 함께 계산한다.
    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="macro",
        zero_division=0,
    )
    weighted_p, weighted_r, weighted_f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="weighted",
        zero_division=0,
    )
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_precision": float(macro_p),
        "macro_recall": float(macro_r),
        "macro_f1": float(macro_f1),
        "weighted_precision": float(weighted_p),
        "weighted_recall": float(weighted_r),
        "weighted_f1": float(weighted_f1),
    }


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[Dict[str, float], np.ndarray, np.ndarray]:
    # gradient 계산 없이 loss, metric, 예측 결과를 모은다.
    model.eval()
    total_loss = 0.0
    total_items = 0
    y_true: List[int] = []
    y_pred: List[int] = []

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            logits = model(x)
            loss = criterion(logits, y)

            total_loss += float(loss.item()) * y.size(0)
            total_items += int(y.size(0))
            y_true.extend(y.cpu().numpy().tolist())
            y_pred.extend(logits.argmax(dim=1).cpu().numpy().tolist())

    y_true_np = np.asarray(y_true, dtype=np.int64)
    y_pred_np = np.asarray(y_pred, dtype=np.int64)
    metrics = compute_metrics(y_true_np, y_pred_np)
    metrics["loss"] = total_loss / max(total_items, 1)
    return metrics, y_true_np, y_pred_np


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    grad_clip: float,
) -> float:
    # 한 epoch 동안 forward/backward/update를 반복하고 평균 loss를 반환한다.
    model.train()
    total_loss = 0.0
    total_items = 0

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()

        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        optimizer.step()

        total_loss += float(loss.item()) * y.size(0)
        total_items += int(y.size(0))

    return total_loss / max(total_items, 1)


def save_checkpoint(
    path: Path,
    model: nn.Module,
    preprocess_config: PreprocessConfig,
    model_args: Dict[str, object],
    stats: Dict[str, float],
    epoch: int,
    best_metric: float,
) -> None:
    # 재사용 가능한 모델 가중치와 전처리/정규화 정보를 함께 저장한다.
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "preprocess_config": asdict(preprocess_config),
        "model_args": model_args,
        "normalization": stats,
        "class_names": CLASS_NAMES,
        "epoch": epoch,
        "selection_metric": "validation_accuracy",
        "best_metric": best_metric,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, path)


def train_model(
    cache_dir: Path,
    manifest: pd.DataFrame,
    stats: Dict[str, float],
    preprocess_config: PreprocessConfig,
    args: argparse.Namespace,
    device: torch.device,
) -> Path:
    # 학습에 필요한 loader, model, loss, optimizer, scheduler를 준비한다.
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    history_dir = Path(args.cache_dir).parent
    history_dir.mkdir(parents=True, exist_ok=True)

    train_loader, val_loader, _ = make_loaders(
        cache_dir=cache_dir,
        manifest=manifest,
        stats=stats,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=device,
    )

    model_args = {
        "n_mels": preprocess_config.n_mels,
        "num_classes": len(CLASS_NAMES),
        "hidden_size": args.hidden_size,
        "rnn_layers": args.rnn_layers,
        "dropout": args.dropout,
    }
    model = AudioRCNN(**model_args).to(device)

    weights = None if args.no_class_weights else class_weights_from_manifest(manifest, device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=max(1, args.patience // 2),
    )

    history: List[Dict[str, float]] = []
    best_metric = -1.0
    bad_epochs = 0
    best_path = checkpoint_dir / "best_rcnn.pt"

    for epoch in range(1, args.epochs + 1):
        # validation accuracy가 좋아질 때만 best checkpoint를 갱신한다.
        train_loss = train_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            grad_clip=args.grad_clip,
        )
        val_metrics, _, _ = evaluate(model, val_loader, criterion, device)
        scheduler.step(val_metrics["accuracy"])

        row = {"epoch": epoch, "train_loss": train_loss, **val_metrics}
        history.append(row)
        pd.DataFrame(history).to_csv(history_dir / "history.csv", index=False)

        print(
            "epoch "
            f"{epoch:03d}/{args.epochs} "
            f"train_loss={train_loss:.4f} "
            f"val_loss={val_metrics['loss']:.4f} "
            f"val_acc={val_metrics['accuracy']:.4f} "
            f"val_macro_f1={val_metrics['macro_f1']:.4f}"
        )

        if val_metrics["accuracy"] > best_metric + args.min_delta:
            best_metric = val_metrics["accuracy"]
            bad_epochs = 0
            save_checkpoint(
                path=best_path,
                model=model,
                preprocess_config=preprocess_config,
                model_args=model_args,
                stats=stats,
                epoch=epoch,
                best_metric=best_metric,
            )
        else:
            bad_epochs += 1

        if bad_epochs >= args.patience:
            print(f"early stopping at epoch {epoch}")
            break

    return best_path


def load_model(checkpoint_path: Path, device: torch.device) -> Tuple[nn.Module, Dict[str, object]]:
    # checkpoint의 model_args로 같은 구조를 만든 뒤 저장된 가중치를 불러온다.
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = AudioRCNN(**checkpoint["model_args"]).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, checkpoint


def report_to_frame(y_true: np.ndarray, y_pred: np.ndarray) -> pd.DataFrame:
    # sklearn classification_report 결과를 CSV로 저장하기 쉬운 DataFrame으로 바꾼다.
    report = classification_report(
        y_true,
        y_pred,
        labels=list(range(len(CLASS_NAMES))),
        target_names=CLASS_NAMES,
        zero_division=0,
        output_dict=True,
    )
    rows = []
    for label, values in report.items():
        if isinstance(values, dict):
            row = {"label": label, **values}
        else:
            row = {
                "label": label,
                "precision": values,
                "recall": values,
                "f1-score": values,
                "support": len(y_true),
            }
        rows.append(row)
    return pd.DataFrame(rows)


def top_confusion_pairs(cm: np.ndarray, limit: int = 8) -> List[Dict[str, object]]:
    # diagonal을 제외하고 실제 class가 어떤 class로 가장 많이 오분류됐는지 뽑는다.
    pairs: List[Dict[str, object]] = []
    row_totals = cm.sum(axis=1)
    for actual_index, actual_name in enumerate(CLASS_NAMES):
        for pred_index, pred_name in enumerate(CLASS_NAMES):
            if actual_index == pred_index:
                continue
            count = int(cm[actual_index, pred_index])
            if count <= 0:
                continue
            support = int(row_totals[actual_index])
            pairs.append(
                {
                    "actual": actual_name,
                    "predicted": pred_name,
                    "count": count,
                    "actual_support": support,
                    "actual_rate": count / max(support, 1),
                }
            )
    return sorted(pairs, key=lambda item: item["count"], reverse=True)[:limit]


def relative_markdown_path(path: Path, base: Path) -> str:
    try:
        return path.relative_to(base).as_posix()
    except ValueError:
        return path.as_posix()


def save_confusion_matrix_plot(cm: np.ndarray, image_dir: Path) -> Path:
    # 발표/보고서에 바로 넣을 수 있도록 confusion matrix를 PNG로 저장한다.
    os.environ.setdefault("MPLCONFIGDIR", str(Path("/tmp") / "matplotlib"))
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    image_dir.mkdir(parents=True, exist_ok=True)
    image_path = image_dir / "rcnn_confusion_matrix.png"

    fig, ax = plt.subplots(figsize=(11, 9))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    tick_positions = np.arange(len(CLASS_NAMES))
    ax.set_xticks(tick_positions)
    ax.set_yticks(tick_positions)
    ax.set_xticklabels(CLASS_NAMES, rotation=45, ha="right")
    ax.set_yticklabels(CLASS_NAMES)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title("RCNN Confusion Matrix")

    threshold = cm.max() / 2.0 if cm.size else 0.0
    for row_index in range(cm.shape[0]):
        for col_index in range(cm.shape[1]):
            value = int(cm[row_index, col_index])
            color = "white" if value > threshold else "black"
            ax.text(col_index, row_index, value, ha="center", va="center", color=color)

    fig.tight_layout()
    fig.savefig(image_path, dpi=180)
    plt.close(fig)
    return image_path


def save_feature_summary(
    summary_path: Path,
    image_path: Path,
    output_dir: Path,
    checkpoint_path: Path,
    cache_dir: Path,
    preprocess_config: PreprocessConfig,
    stats: Dict[str, float],
    metrics: Dict[str, float],
    report_frame: pd.DataFrame,
    cm: np.ndarray,
) -> None:
    # 모델 입력 feature와 test 결과를 한 문서에서 해석할 수 있게 요약한다.
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    base = summary_path.parent
    image_link = relative_markdown_path(image_path, base)
    output_link = relative_markdown_path(output_dir, base)
    checkpoint_link = relative_markdown_path(checkpoint_path, base)
    cache_link = relative_markdown_path(cache_dir, base)

    metric_order = [
        ("loss", "낮을수록 좋음"),
        ("accuracy", "전체 sample 기준 정답률"),
        ("balanced_accuracy", "class별 recall 평균"),
        ("macro_precision", "class별 precision 단순 평균"),
        ("macro_recall", "class별 recall 단순 평균"),
        ("macro_f1", "class별 F1 단순 평균"),
        ("weighted_precision", "sample 수 가중 precision"),
        ("weighted_recall", "sample 수 가중 recall"),
        ("weighted_f1", "sample 수 가중 F1"),
    ]

    class_rows = report_frame[report_frame["label"].isin(CLASS_NAMES)].copy()
    class_rows = class_rows.sort_values("f1-score")
    confusion_pairs = top_confusion_pairs(cm)

    lines = [
        "# RCNN Feature Result 해석",
        "",
        "## 파일 위치",
        "",
        f"- 코드: `models/rcnn/rcnn.py`",
        f"- 체크포인트: `{checkpoint_link}`",
        f"- test CSV/JSON 결과: `{output_link}`",
        f"- confusion matrix 이미지: `{image_link}`",
        f"- 전처리 feature cache: `{cache_link}`",
        "",
        "## 입력 feature 설정",
        "",
        "| 항목 | 값 | 해석 |",
        "|---|---:|---|",
        f"| sample rate | {preprocess_config.sample_rate} | 모든 wav를 이 Hz로 resampling |",
        f"| duration | {preprocess_config.duration:.2f}s | 모든 입력 길이를 동일하게 맞춤 |",
        f"| padding | {preprocess_config.padding} | 짧은 오디오를 채우는 방식 |",
        f"| n_mels | {preprocess_config.n_mels} | log-mel 주파수 bin 수 |",
        f"| n_fft | {preprocess_config.n_fft} | STFT window 크기 |",
        f"| hop_length | {preprocess_config.hop_length} | 시간 frame 간격 |",
        f"| top_db | {preprocess_config.top_db:.1f} | log 변환 후 dynamic range 제한 |",
        f"| normalization mean | {stats['mean']:.4f} | train split feature 평균 |",
        f"| normalization std | {stats['std']:.4f} | train split feature 표준편차 |",
        "",
        "이 모델은 wav를 직접 넣지 않고 log-mel spectrogram을 입력으로 씁니다. 정규화 통계는 train fold만으로 계산해서 validation/test 정보가 섞이지 않게 했습니다.",
        "",
        "## Test 전체 성능",
        "",
        "| 지표 | 값 | 해석 |",
        "|---|---:|---|",
    ]

    for key, description in metric_order:
        if key in metrics:
            lines.append(f"| {key} | {metrics[key]:.4f} | {description} |")

    lines.extend(
        [
            "",
            "macro F1은 class별 성능을 같은 비중으로 보기 때문에 UrbanSound8K처럼 class 수가 고정된 분류 문제에서 모델 비교용 핵심 지표로 쓰기 좋습니다. weighted F1은 실제 test set 분포를 반영한 체감 성능에 가깝습니다.",
            "",
            "## Class별 성능",
            "",
            "| class | precision | recall | F1 | support | 해석 포인트 |",
            "|---|---:|---:|---:|---:|---|",
        ]
    )

    for _, row in class_rows.iterrows():
        label = str(row["label"])
        precision = float(row["precision"])
        recall = float(row["recall"])
        f1 = float(row["f1-score"])
        support = int(row["support"])
        if precision < recall:
            note = "오탐이 상대적으로 많음"
        elif recall < precision:
            note = "미탐이 상대적으로 많음"
        else:
            note = "precision/recall 균형"
        lines.append(
            f"| {label} | {precision:.4f} | {recall:.4f} | {f1:.4f} | {support} | {note} |"
        )

    lines.extend(
        [
            "",
            "## 많이 헷갈린 class 쌍",
            "",
            "| 실제 class | 예측 class | 건수 | 실제 class 내 비율 |",
            "|---|---|---:|---:|",
        ]
    )

    if confusion_pairs:
        for pair in confusion_pairs:
            lines.append(
                "| "
                f"{pair['actual']} | "
                f"{pair['predicted']} | "
                f"{pair['count']} | "
                f"{pair['actual_rate'] * 100:.1f}% |"
            )
    else:
        lines.append("| - | - | 0 | 0.0% |")

    lines.extend(
        [
            "",
            "confusion matrix는 행이 실제 class, 열이 예측 class입니다. 대각선 값이 클수록 해당 class를 잘 맞힌 것이고, 대각선 밖의 큰 값은 모델이 반복해서 헷갈리는 class 조합입니다.",
            "",
            "## 해석 순서",
            "",
            "1. `macro_f1`로 전체 class 균형 성능을 먼저 확인합니다.",
            "2. class별 F1이 낮은 항목을 보고 어떤 class가 약한지 찾습니다.",
            "3. `images/rcnn_confusion_matrix.png`에서 대각선 밖의 큰 값을 확인합니다.",
            "4. precision이 낮으면 오탐, recall이 낮으면 미탐 관점으로 개선 방향을 잡습니다.",
            "",
        ]
    )

    summary_path.write_text("\n".join(lines), encoding="utf-8")


def save_test_outputs(
    output_dir: Path,
    image_dir: Path,
    summary_path: Path,
    checkpoint_path: Path,
    checkpoint: Dict[str, object],
    cache_dir: Path,
    manifest: pd.DataFrame,
    preprocess_config: PreprocessConfig,
    stats: Dict[str, float],
    metrics: Dict[str, float],
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> None:
    # test metric, sample별 예측, confusion matrix, class별 report와 해석 문서를 남긴다.
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "test_metrics.json").write_text(
        json.dumps(metrics, indent=2),
        encoding="utf-8",
    )
    model_args = checkpoint.get("model_args", {})
    overall_metrics = {
        "model": "RCNN",
        "selection_metric": checkpoint.get("selection_metric", "validation_accuracy"),
        "best_validation_score": checkpoint.get("best_metric"),
        "best_params": json.dumps(model_args, ensure_ascii=False, sort_keys=True),
        "test_loss": metrics.get("loss"),
        "final_epoch": checkpoint.get("epoch"),
        **{k: v for k, v in metrics.items() if k != "loss"},
    }
    pd.DataFrame([overall_metrics]).to_csv(
        output_dir / "overall_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )

    test_manifest = manifest[manifest["split"] == "test"].reset_index(drop=True).copy()
    test_manifest["true_label"] = [CLASS_NAMES[i] for i in y_true]
    test_manifest["pred_classID"] = y_pred
    test_manifest["pred_label"] = [CLASS_NAMES[i] for i in y_pred]
    test_manifest["correct"] = y_true == y_pred
    test_manifest.to_csv(output_dir / "test_predictions.csv", index=False)
    predictions = pd.DataFrame({
        "model": "RCNN",
        "slice_file_name": test_manifest["slice_file_name"],
        "fold": test_manifest["fold"],
        "y_true": y_true,
        "y_pred": y_pred,
        "true_label": test_manifest["true_label"],
        "pred_label": test_manifest["pred_label"],
    })
    predictions.to_csv(output_dir / "predictions.csv", index=False, encoding="utf-8-sig")

    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(CLASS_NAMES))))

    report_frame = report_to_frame(y_true, y_pred)
    report_frame.to_csv(output_dir / "classification_report.csv", index=False)
    p, r, f, s = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=list(range(len(CLASS_NAMES))),
        zero_division=0,
    )
    classwise = pd.DataFrame({
        "model": "RCNN",
        "class": CLASS_NAMES,
        "precision": p,
        "recall": r,
        "f1": f,
        "support": s,
    })
    classwise.to_csv(output_dir / "classwise_metrics.csv", index=False, encoding="utf-8-sig")

    image_path = save_confusion_matrix_plot(cm, image_dir)
    save_feature_summary(
        summary_path=summary_path,
        image_path=image_path,
        output_dir=output_dir,
        checkpoint_path=checkpoint_path,
        cache_dir=cache_dir,
        preprocess_config=preprocess_config,
        stats=stats,
        metrics=metrics,
        report_frame=report_frame,
        cm=cm,
    )


def test_model(
    checkpoint_path: Path,
    cache_dir: Path,
    manifest: pd.DataFrame,
    stats: Dict[str, float],
    args: argparse.Namespace,
    device: torch.device,
) -> Dict[str, float]:
    # 저장된 best checkpoint를 test split에 평가하고 결과 파일을 만든다.
    _, _, test_loader = make_loaders(
        cache_dir=cache_dir,
        manifest=manifest,
        stats=stats,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=device,
    )
    model, checkpoint = load_model(checkpoint_path, device)
    criterion = nn.CrossEntropyLoss()
    metrics, y_true, y_pred = evaluate(model, test_loader, criterion, device)
    save_test_outputs(
        output_dir=Path(args.output_dir),
        image_dir=Path(args.image_dir),
        summary_path=Path(args.summary_md),
        checkpoint_path=checkpoint_path,
        checkpoint=checkpoint,
        cache_dir=cache_dir,
        manifest=manifest,
        preprocess_config=PreprocessConfig(**checkpoint["preprocess_config"]),
        stats=stats,
        metrics=metrics,
        y_true=y_true,
        y_pred=y_pred,
    )

    print(
        "test "
        f"loss={metrics['loss']:.4f} "
        f"acc={metrics['accuracy']:.4f} "
        f"balanced_acc={metrics['balanced_accuracy']:.4f} "
        f"macro_f1={metrics['macro_f1']:.4f}"
    )
    return metrics


def parse_args() -> argparse.Namespace:
    # CLI에서 전처리, 학습, 평가 옵션을 받을 수 있게 argument를 정의한다.
    root = project_root()
    model_dir = Path(__file__).parent
    parser = argparse.ArgumentParser(
        description="RCNN training pipeline for UrbanSound8K log-mel classification."
    )
    parser.add_argument(
        "--mode",
        choices=["preprocess", "train", "test", "all"],
        default="all",
        help="Run preprocessing, training, testing, or the full pipeline.",
    )
    parser.add_argument("--data-root", type=Path, default=root / "UrbanSound8K")
    parser.add_argument("--cache-dir", type=Path, default=model_dir / "intermediate" / "preprocessed")
    parser.add_argument("--output-dir", type=Path, default=model_dir / "final")
    parser.add_argument("--checkpoint-dir", type=Path, default=model_dir)
    parser.add_argument("--image-dir", type=Path, default=model_dir / ".." / ".." / "images")
    parser.add_argument("--summary-md", type=Path, default=model_dir / "final" / "feature_summary.md")
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--force-preprocess", action="store_true")

    parser.add_argument("--sample-rate", type=int, default=22050)
    parser.add_argument("--duration", type=float, default=4.0)
    parser.add_argument("--padding", choices=["repeat", "zero"], default="repeat")
    parser.add_argument("--n-mels", type=int, default=128)
    parser.add_argument("--n-fft", type=int, default=1024)
    parser.add_argument("--hop-length", type=int, default=512)
    parser.add_argument("--fmin", type=float, default=0.0)
    parser.add_argument("--fmax", type=float, default=0.0)
    parser.add_argument("--top-db", type=float, default=80.0)

    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--rnn-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--grad-clip", type=float, default=5.0)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--min-delta", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--no-class-weights", action="store_true")
    return parser.parse_args()


def resolve_device(choice: str) -> torch.device:
    # auto/cpu/cuda 선택값을 실제 PyTorch device로 변환한다.
    if choice == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if choice == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested, but torch.cuda.is_available() is False.")
    return torch.device(choice)


def main() -> None:
    # CLI 설정을 모아 전처리, 학습, 테스트 실행 흐름을 제어한다.
    args = parse_args()
    seed_everything(args.seed)

    preprocess_config = PreprocessConfig(
        sample_rate=args.sample_rate,
        duration=args.duration,
        padding=args.padding,
        n_mels=args.n_mels,
        n_fft=args.n_fft,
        hop_length=args.hop_length,
        fmin=args.fmin,
        fmax=args.fmax,
        top_db=args.top_db,
    )

    data_root = args.data_root.expanduser().resolve()
    cache_root = args.cache_dir.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()
    args.checkpoint_dir = args.checkpoint_dir.expanduser().resolve()
    args.image_dir = args.image_dir.expanduser()
    args.summary_md = args.summary_md.expanduser().resolve()
    device = resolve_device(args.device)
    checkpoint_path = args.checkpoint.expanduser().resolve() if args.checkpoint else None

    if args.mode == "test":
        if checkpoint_path is None:
            checkpoint_path = Path(args.checkpoint_dir) / "best_rcnn.pt"
        if checkpoint_path.exists():
            checkpoint = torch.load(checkpoint_path, map_location="cpu")
            preprocess_config = PreprocessConfig(**checkpoint["preprocess_config"])

    print(f"device: {device}")
    print(f"data_root: {data_root}")

    cache_dir, manifest, stats = ensure_preprocessed(
        data_root=data_root,
        cache_root=cache_root,
        config=preprocess_config,
        force=args.force_preprocess,
    )
    print(f"cache_dir: {cache_dir}")
    print(f"normalization: mean={stats['mean']:.4f}, std={stats['std']:.4f}")

    if args.mode == "preprocess":
        return

    if args.mode in {"train", "all"}:
        checkpoint_path = train_model(
            cache_dir=cache_dir,
            manifest=manifest,
            stats=stats,
            preprocess_config=preprocess_config,
            args=args,
            device=device,
        )

    if args.mode in {"test", "all"}:
        if checkpoint_path is None:
            checkpoint_path = Path(args.checkpoint_dir) / "best_rcnn.pt"
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        test_model(
            checkpoint_path=checkpoint_path,
            cache_dir=cache_dir,
            manifest=manifest,
            stats=stats,
            args=args,
            device=device,
        )


if __name__ == "__main__":
    main()

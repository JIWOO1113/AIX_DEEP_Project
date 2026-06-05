import joblib
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix
)

from xgboost import XGBClassifier

HERE = Path(__file__).resolve().parent
INTERMEDIATE_DIR = HERE / "intermediate"
FINAL_DIR = HERE / "final"
IMAGE_DIR = Path(__file__).parent / ".." / ".." / "images"
INTERMEDIATE_DIR.mkdir(parents=True, exist_ok=True)
FINAL_DIR.mkdir(parents=True, exist_ok=True)
IMAGE_DIR.mkdir(parents=True, exist_ok=True)


def load_intermediate_array(name):
    candidates = [
        INTERMEDIATE_DIR / name,
        HERE / name,
        Path.cwd() / name,
    ]
    for path in candidates:
        if path.exists():
            return np.load(path, allow_pickle=True)
    raise FileNotFoundError(
        f"{name} not found. Run {HERE / 'preprocess.py'} first."
    )


def save_confusion_matrix_plot(cm, class_names, path, title):
    try:
        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(10, 8))
        im = ax.imshow(cm, cmap="Blues")
        fig.colorbar(im, ax=ax)
        ax.set_xticks(range(len(class_names)))
        ax.set_yticks(range(len(class_names)))
        ax.set_xticklabels(class_names, rotation=45, ha="right")
        ax.set_yticklabels(class_names)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title(title)

        threshold = cm.max() / 2 if cm.size else 0
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(
                    j,
                    i,
                    int(cm[i, j]),
                    ha="center",
                    va="center",
                    color="white" if cm[i, j] > threshold else "black",
                )

        fig.tight_layout()
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    except Exception as e:
        print(f"confusion matrix 그림 저장 생략: {e}")


X = load_intermediate_array("X_mfcc.npy")
y = load_intermediate_array("y.npy")
folds = load_intermediate_array("folds.npy")

train_mask = folds <= 8
val_mask = folds == 9
test_mask = folds == 10

X_train, y_train = X[train_mask], y[train_mask]
X_val, y_val = X[val_mask], y[val_mask]
X_test, y_test = X[test_mask], y[test_mask]

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

target_names = [
    "air_conditioner", "car_horn", "children_playing", "dog_bark",
    "drilling", "engine_idling", "gun_shot", "jackhammer",
    "siren", "street_music"
]

param_grid = [
    {"n_estimators": 100, "max_depth": 3, "learning_rate": 0.05},
    {"n_estimators": 100, "max_depth": 5, "learning_rate": 0.05},
    {"n_estimators": 300, "max_depth": 3, "learning_rate": 0.05},
    {"n_estimators": 300, "max_depth": 5, "learning_rate": 0.05},
    {"n_estimators": 300, "max_depth": 5, "learning_rate": 0.1},
]

best_model = None
best_params = None
best_val_acc = 0

print("XGBoost validation tuning 시작")

for params in param_grid:
    model = XGBClassifier(
        n_estimators=params["n_estimators"],
        max_depth=params["max_depth"],
        learning_rate=params["learning_rate"],
        subsample=0.8,
        colsample_bytree=0.8,
        objective="multi:softmax",
        num_class=10,
        eval_metric="mlogloss",
        random_state=42
    )

    model.fit(X_train_scaled, y_train)
    val_pred = model.predict(X_val_scaled)
    val_acc = accuracy_score(y_val, val_pred)

    print(f"Params: {params}, Validation Accuracy: {val_acc:.6f}")

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_model = model
        best_params = params

print("\nBest XGBoost Params:", best_params)
print("Best Validation Accuracy:", best_val_acc)

test_pred = best_model.predict(X_test_scaled)

metrics = {
    "accuracy": accuracy_score(y_test, test_pred),
    "balanced_accuracy": balanced_accuracy_score(y_test, test_pred),
    "macro_precision": precision_score(y_test, test_pred, average="macro", zero_division=0),
    "macro_recall": recall_score(y_test, test_pred, average="macro", zero_division=0),
    "macro_f1": f1_score(y_test, test_pred, average="macro", zero_division=0),
    "weighted_precision": precision_score(y_test, test_pred, average="weighted", zero_division=0),
    "weighted_recall": recall_score(y_test, test_pred, average="weighted", zero_division=0),
    "weighted_f1": f1_score(y_test, test_pred, average="weighted", zero_division=0),
}

results = {
    "model": "XGBoost",
    "selection_metric": "validation_accuracy",
    "best_validation_score": best_val_acc,
    "best_params": str(best_params),
    "test_loss": np.nan,
    "final_epoch": np.nan,
    **metrics,
}

pd.DataFrame([results]).to_csv(
    FINAL_DIR / "overall_metrics.csv",
    index=False,
    encoding="utf-8-sig",
)

report = classification_report(
    y_test,
    test_pred,
    target_names=target_names,
    output_dict=True,
    zero_division=0
)

report_df = pd.DataFrame(report).transpose()
report_df.to_csv(FINAL_DIR / "classification_report.csv", encoding="utf-8-sig")

p, r, f, s = precision_recall_fscore_support(
    y_test,
    test_pred,
    labels=range(len(target_names)),
    zero_division=0,
)
classwise_df = pd.DataFrame({
    "model": "XGBoost",
    "class": target_names,
    "precision": p,
    "recall": r,
    "f1": f,
    "support": s,
})
classwise_df.to_csv(FINAL_DIR / "classwise_metrics.csv", index=False, encoding="utf-8-sig")

predictions_df = pd.DataFrame({
    "model": "XGBoost",
    "y_true": y_test,
    "y_pred": test_pred,
    "true_label": [target_names[int(i)] for i in y_test],
    "pred_label": [target_names[int(i)] for i in test_pred],
})
predictions_df.to_csv(FINAL_DIR / "predictions.csv", index=False, encoding="utf-8-sig")

cm = confusion_matrix(y_test, test_pred, labels=range(len(target_names)))
save_confusion_matrix_plot(
    cm,
    target_names,
    IMAGE_DIR / "xgboost_confusion_matrix.png",
    "XGBoost Confusion Matrix (Test fold 10)",
)

best_model.save_model(str(HERE / "xgb_checkpoint.json"))
joblib.dump(scaler, HERE / "xgb_scaler.pkl")

print("\n저장 완료:")
print(HERE / "xgb_checkpoint.json")
print(HERE / "xgb_scaler.pkl")
print(FINAL_DIR / "overall_metrics.csv")
print(FINAL_DIR / "classwise_metrics.csv")
print(IMAGE_DIR / "xgboost_confusion_matrix.png")

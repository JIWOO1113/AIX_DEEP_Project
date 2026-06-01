import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

from xgboost import XGBClassifier


# =====================
# 데이터 불러오기
# =====================

X = np.load("X_mfcc.npy")
y = np.load("y.npy")
folds = np.load("folds.npy")
class_names = np.load("class_names.npy", allow_pickle=True)

print("X shape:", X.shape)
print("y shape:", y.shape)
print("folds shape:", folds.shape)


# =====================
# Train / Validation / Test 분리
# =====================

train_mask = folds <= 8
val_mask = folds == 9
test_mask = folds == 10

X_train = X[train_mask]
y_train = y[train_mask]

X_val = X[val_mask]
y_val = y[val_mask]

X_test = X[test_mask]
y_test = y[test_mask]

print("Train:", X_train.shape)
print("Validation:", X_val.shape)
print("Test:", X_test.shape)


# =====================
# Train 기준 Standardization
# =====================

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

print("Standardization 완료")


# =====================
# 평가 함수
# =====================

def evaluate_model(model_name, y_true, y_pred):
    results = {
        "Model": model_name,
        "Accuracy": accuracy_score(y_true, y_pred),
        "Balanced Accuracy": balanced_accuracy_score(y_true, y_pred),
        "Macro Precision": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "Macro Recall": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "Macro F1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "Weighted Precision": precision_score(y_true, y_pred, average="weighted", zero_division=0),
        "Weighted Recall": recall_score(y_true, y_pred, average="weighted", zero_division=0),
        "Weighted F1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
    }
    return results


# =====================
# SVM 학습
# =====================

svm_model = SVC(
    kernel="rbf",
    C=10,
    gamma="scale",
    class_weight="balanced",
    random_state=42
)

print("SVM 학습 시작")
svm_model.fit(X_train_scaled, y_train)
print("SVM 학습 완료")

svm_pred = svm_model.predict(X_test_scaled)


# =====================
# XGBoost 학습
# =====================

xgb_model = XGBClassifier(
    n_estimators=300,
    max_depth=5,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="multi:softmax",
    num_class=10,
    eval_metric="mlogloss",
    random_state=42
)

print("XGBoost 학습 시작")
xgb_model.fit(X_train_scaled, y_train)
print("XGBoost 학습 완료")

xgb_pred = xgb_model.predict(X_test_scaled)


# =====================
# 전체 평가 결과표
# =====================

results = [
    evaluate_model("SVM", y_test, svm_pred),
    evaluate_model("XGBoost", y_test, xgb_pred)
]

results_df = pd.DataFrame(results)
print("\n=== Overall Results ===")
print(results_df)

results_df.to_csv("overall_results.csv", index=False, encoding="utf-8-sig")


# =====================
# Per-class 평가 결과
# =====================

target_names = [
    "air_conditioner",
    "car_horn",
    "children_playing",
    "dog_bark",
    "drilling",
    "engine_idling",
    "gun_shot",
    "jackhammer",
    "siren",
    "street_music"
]

svm_report = classification_report(
    y_test,
    svm_pred,
    target_names=target_names,
    output_dict=True,
    zero_division=0
)

xgb_report = classification_report(
    y_test,
    xgb_pred,
    target_names=target_names,
    output_dict=True,
    zero_division=0
)

svm_report_df = pd.DataFrame(svm_report).transpose()
xgb_report_df = pd.DataFrame(xgb_report).transpose()

svm_report_df.to_csv("svm_per_class_results.csv", encoding="utf-8-sig")
xgb_report_df.to_csv("xgb_per_class_results.csv", encoding="utf-8-sig")

print("\n=== SVM Per-class Results ===")
print(svm_report_df)

print("\n=== XGBoost Per-class Results ===")
print(xgb_report_df)


# =====================
# Confusion Matrix 저장
# =====================

svm_cm = confusion_matrix(y_test, svm_pred)
xgb_cm = confusion_matrix(y_test, xgb_pred)

svm_cm_df = pd.DataFrame(
    svm_cm,
    index=target_names,
    columns=target_names
)

xgb_cm_df = pd.DataFrame(
    xgb_cm,
    index=target_names,
    columns=target_names
)

svm_cm_df.to_csv("svm_confusion_matrix.csv", encoding="utf-8-sig")
xgb_cm_df.to_csv("xgb_confusion_matrix.csv", encoding="utf-8-sig")

print("\n저장 완료:")
print("overall_results.csv")
print("svm_per_class_results.csv")
print("xgb_per_class_results.csv")
print("svm_confusion_matrix.csv")
print("xgb_confusion_matrix.csv")
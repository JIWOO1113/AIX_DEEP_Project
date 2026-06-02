import joblib
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


X = np.load("X_mfcc.npy")
y = np.load("y.npy")
folds = np.load("folds.npy")

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
    {"C": 1, "gamma": "scale"},
    {"C": 10, "gamma": "scale"},
    {"C": 100, "gamma": "scale"},
    {"C": 1, "gamma": "auto"},
    {"C": 10, "gamma": "auto"},
    {"C": 100, "gamma": "auto"},
]

best_model = None
best_params = None
best_val_acc = 0

print("SVM validation tuning 시작")

for params in param_grid:
    model = SVC(
        kernel="rbf",
        C=params["C"],
        gamma=params["gamma"],
        class_weight="balanced",
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

print("\nBest SVM Params:", best_params)
print("Best Validation Accuracy:", best_val_acc)

test_pred = best_model.predict(X_test_scaled)

results = {
    "Model": "SVM",
    "Best Params": str(best_params),
    "Validation Accuracy": best_val_acc,
    "Test Accuracy": accuracy_score(y_test, test_pred),
    "Test Balanced Accuracy": balanced_accuracy_score(y_test, test_pred),
    "Test Macro Precision": precision_score(y_test, test_pred, average="macro", zero_division=0),
    "Test Macro Recall": recall_score(y_test, test_pred, average="macro", zero_division=0),
    "Test Macro F1": f1_score(y_test, test_pred, average="macro", zero_division=0),
    "Test Weighted Precision": precision_score(y_test, test_pred, average="weighted", zero_division=0),
    "Test Weighted Recall": recall_score(y_test, test_pred, average="weighted", zero_division=0),
    "Test Weighted F1": f1_score(y_test, test_pred, average="weighted", zero_division=0),
}

results_df = pd.DataFrame([results])
results_df.to_csv("svm_overall_results.csv", index=False, encoding="utf-8-sig")

report = classification_report(
    y_test,
    test_pred,
    target_names=target_names,
    output_dict=True,
    zero_division=0
)

report_df = pd.DataFrame(report).transpose()
report_df.to_csv("svm_per_class_results.csv", encoding="utf-8-sig")

cm = confusion_matrix(y_test, test_pred)
cm_df = pd.DataFrame(cm, index=target_names, columns=target_names)
cm_df.to_csv("svm_confusion_matrix.csv", encoding="utf-8-sig")

joblib.dump(best_model, "svm_checkpoint.pkl")
joblib.dump(scaler, "svm_scaler.pkl")

print("\n저장 완료:")
print("svm_checkpoint.pkl")
print("svm_scaler.pkl")
print("svm_overall_results.csv")
print("svm_per_class_results.csv")
print("svm_confusion_matrix.csv")
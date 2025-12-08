import time
import numpy as np
import pandas as pd
from pre_processing import X_train, X_test, Y_train, Y_test
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression, LinearRegression
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import seaborn as sns

if Y_train.ndim > 1:
    Y_train = np.argmax(Y_train, axis=1)
if Y_test.ndim > 1:
    Y_test = np.argmax(Y_test, axis=1)


def evaluate_model(model, X_train, Y_train, X_test, Y_test, model_name="Model"):
    print(f"\n================== {model_name} ==================")

    start_time = time.time()
    model.fit(X_train, Y_train)
    training_time = time.time() - start_time

    y_pred = model.predict(X_test)

    acc = accuracy_score(Y_test, y_pred)
    prec = precision_score(Y_test, y_pred, average="macro", zero_division=0)
    rec = recall_score(Y_test, y_pred, average="macro", zero_division=0)
    f1 = f1_score(Y_test, y_pred, average="macro", zero_division=0)
    cm = confusion_matrix(Y_test, y_pred)

    print(f"Accuracy       : {acc:.4f}")
    print(f"Precision      : {prec:.4f}")
    print(f"Recall         : {rec:.4f}")
    print(f"F1-score       : {f1:.4f}")
    print(f"Training time  : {training_time:.4f} seconds")
    print("\nConfusion Matrix:")
    print(cm)

    return {
        "Model": model_name,
        "Accuracy": acc,
        "Precision": prec,
        "Recall": rec,
        "F1-score": f1,
        "Training Time": training_time,
        "Confusion Matrix": cm
    }

results = []

models = [
    ("Model 1: MLP Neural Network",
     MLPClassifier(hidden_layer_sizes=(128, 64), learning_rate_init=0.001, max_iter=30)),

    ("Model 2: Logistic Regression",
     LogisticRegression(max_iter=200, solver="lbfgs", multi_class="multinomial")),

    ("Model 3: Linear Regression (OvA)", None)
]

class OVA_LinearRegression:
    def __init__(self):
        self.models = []

    def fit(self, X, y):
        for digit in range(10):
            y_binary = (y == digit).astype(int)
            lr = LinearRegression()
            lr.fit(X, y_binary)
            self.models.append(lr)

    def predict(self, X):
        preds = np.column_stack([m.predict(X) for m in self.models])
        return np.argmax(preds, axis=1)


for name, model in models:
    if model is None:
        model = OVA_LinearRegression()

    results.append(evaluate_model(model, X_train, Y_train, X_test, Y_test, model_name=name))


df_results = pd.DataFrame(results)
print("\n\n===== Comparative Analysis Table =====\n")
print(df_results.drop(columns=["Confusion Matrix"]))


metrics = ["Accuracy", "Precision", "Recall", "F1-score"]
plt.figure(figsize=(10, 6))

for i, metric in enumerate(metrics):
    plt.bar(df_results["Model"], df_results[metric], label=metric)

plt.xticks(rotation=15)
plt.ylabel("Score")
plt.title("Model Performance Comparison")
plt.legend()
plt.show()

plt.figure(figsize=(8, 5))
plt.bar(df_results["Model"], df_results["Training Time"])
plt.xticks(rotation=15)
plt.ylabel("Seconds")
plt.title("Training Time per Model")
plt.show()

for index, row in df_results.iterrows():
    cm = row["Confusion Matrix"]
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f"Confusion Matrix – {row['Model']}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()

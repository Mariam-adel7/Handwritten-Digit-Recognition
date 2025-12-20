import time
import numpy as np
import pandas as pd
from preprocessing import X_train, X_test, y_train, y_test
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression, LinearRegression
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import seaborn as sns


class OVA_LinearRegression:
    def __init__(self):
        self.models = []

    def fit(self, X, y):
        self.models = []
        classes = np.unique(y)
        for c in classes:
            y_bin = (y == c).astype(float)
            lr = LinearRegression()
            lr.fit(X, y_bin)
            self.models.append(lr)

    def predict(self, X):
        preds = np.column_stack([m.predict(X) for m in self.models])
        return np.argmax(preds, axis=1)

def evaluate_model(model, X_train, y_train, X_test, y_test, model_name="Model"):
    print(f"\n================== {model_name} ==================")
    
    start_time = time.time()
    model.fit(X_train, y_train)
    training_time = time.time() - start_time

    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average="macro", zero_division=0)
    rec = recall_score(y_test, y_pred, average="macro", zero_division=0)
    f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)
    cm = confusion_matrix(y_test, y_pred)

    print(f"Accuracy       : {acc*100:.2f} %")
    print(f"Precision      : {prec*100:.2f} %")
    print(f"Recall         : {rec*100:.2f} %")
    print(f"F1-score       : {f1*100:.2f} %")
    print(f"Training time  : {training_time:.4f} seconds")

    return {
        "Model": model_name,
        "Accuracy": acc,
        "Precision": prec,
        "Recall": rec,
        "F1-score": f1,
        "Training Time": training_time,
        "Confusion Matrix": cm
    }


models = [
    ("MLP Neural Network",
     MLPClassifier(hidden_layer_sizes=(128,64), 
    activation='relu',
    solver='adam',
    alpha=0.001,
    max_iter=1000, 
    random_state=42,
    early_stopping=True)),
    
    ("Logistic Regression",
     LogisticRegression(max_iter=1000, solver="lbfgs", random_state=42)),
    
    ("Linear Regression (OvA)",
     OVA_LinearRegression()),

    ("Naive bayes",
     GaussianNB())
]


results = []
for name, model in models:
    results.append(evaluate_model(model, X_train, y_train, X_test, y_test, model_name=name))


df_results = pd.DataFrame(results)
display_df = pd.DataFrame()
display_df["Model"] = df_results["Model"]
display_df["Accuracy"] = df_results["Accuracy"].apply(lambda x: f"{x*100:.2f}%")
display_df["Precision"] = df_results["Precision"].apply(lambda x: f"{x*100:.2f}%")
display_df["Recall"] = df_results["Recall"].apply(lambda x: f"{x*100:.2f}%")
display_df["F1-score"] = df_results["F1-score"].apply(lambda x: f"{x*100:.2f}%")
display_df["Training Time (s)"] = df_results["Training Time"].round(3)

print("\n===== Comparative Analysis Table =====\n")
print(display_df.to_string(index=False, justify="center"))

metrics = ["Accuracy", "Precision", "Recall", "F1-score"]
plt.figure(figsize=(10, 6))
for metric in metrics:
    plt.plot(df_results["Model"], df_results[metric], marker="o", label=metric)

plt.xticks(rotation=15)
plt.ylabel("Score")
plt.title("Model Performance Comparison")
plt.ylim(0, 1)
plt.legend()
plt.grid(True)
plt.show()


plt.figure(figsize=(8, 5))
sns.barplot(x=df_results["Model"], y=df_results["Training Time"], palette="coolwarm")
plt.xticks(rotation=15)
plt.ylabel("Seconds")
plt.title("Training Time per Model")
plt.show()


for index, row in df_results.iterrows():
    cm = row["Confusion Matrix"]
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f"Confusion Matrix – {row['Model']}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()

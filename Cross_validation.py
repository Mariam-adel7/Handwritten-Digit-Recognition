import numpy as np
import pandas as pd
from preprocessing import X_train, X_test, y_train, y_test
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression, LinearRegression
import matplotlib.pyplot as plt
import seaborn as sns

X = np.vstack((X_train, X_test))
y = np.concatenate((y_train, y_test))

class OVA_LinearRegression:
    def __init__(self):
        self.models = []

    def fit(self, X, y):
        self.models = []
        for c in np.unique(y):
            y_bin = (y == c).astype(float)
            lr = LinearRegression()
            lr.fit(X, y_bin)
            self.models.append(lr)

    def predict(self, X):
        preds = np.column_stack([m.predict(X) for m in self.models])
        return np.argmax(preds, axis=1)


models = {
    "MLP Neural Network": MLPClassifier(
    hidden_layer_sizes=(128,64), 
    activation='relu',
    solver='adam',
    alpha=0.001,
    max_iter=1000, 
    random_state=42,
    early_stopping=True
    ),
    "Logistic Regression": LogisticRegression(solver='lbfgs', max_iter=1000, random_state=42),
    "OVA Linear Regression": OVA_LinearRegression()
}


skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
results = []

for name, model in models.items():
    metrics_folds = []
    for train_idx, test_idx in skf.split(X, y):
        model.fit(X[train_idx], y[train_idx])
        y_pred = model.predict(X[test_idx])
        
        acc = accuracy_score(y[test_idx], y_pred)
        prec, rec, f1, _ = precision_recall_fscore_support(
            y[test_idx], y_pred, average='macro', zero_division=0
        )
        metrics_folds.append([acc, prec, rec, f1])
    
    metrics_array = np.array(metrics_folds)
    means = metrics_array.mean(axis=0)
    stds = metrics_array.std(axis=0)

    best_fold_idx = np.argmax(metrics_array[:, 0])
    best_fold_metrics = metrics_array[best_fold_idx]
    print(f"Best Fold: {best_fold_idx+1} for {name} -> Accuracy: {best_fold_metrics[0]*100:.2f}%, Precision: {best_fold_metrics[1]*100:.2f}%, Recall: {best_fold_metrics[2]*100:.2f}%, F1: {best_fold_metrics[3]*100:.2f}%")
    
    results.append({
        "Model": name,
        "Accuracy": means[0], "Acc_Std": stds[0],
        "Precision": means[1], "Prec_Std": stds[1],
        "Recall": means[2], "Rec_Std": stds[2],
        "F1-Score": means[3], "F1_Std": stds[3]
    })


df = pd.DataFrame(results)

display_df = pd.DataFrame()
display_df["Model"] = df["Model"]
display_df["Accuracy (Mean ± Std)"] = df.apply(
    lambda r: f"{r['Accuracy']*100:.2f}% ± {r['Acc_Std']*100:.2f}%", axis=1
)
display_df["Precision"] = df.apply(
    lambda r: f"{r['Precision']*100:.2f}% ± {r['Prec_Std']*100:.2f}%", axis=1
)
display_df["Recall"] = df.apply(
    lambda r: f"{r['Recall']*100:.2f}% ± {r['Rec_Std']*100:.2f}%", axis=1
)
display_df["F1-Score"] = df.apply(
    lambda r: f"{r['F1-Score']*100:.2f}% ± {r['F1_Std']*100:.2f}%", axis=1
)

print("\n" + "="*100)
print(f"{'STATISTICAL COMPARATIVE ANALYSIS REPORT':^100}")
print("="*100)
print(display_df.to_string(index=False, justify='center', col_space=20))
print("="*100)

plt.figure(figsize=(10, 6))
sns.set_style("whitegrid")

sns.barplot(
    x="Model",
    y="Accuracy",
    data=df,
    palette="viridis",
    ci=None  
)

plt.errorbar(
    x=np.arange(len(df)),
    y=df["Accuracy"],
    yerr=df["Acc_Std"],
    fmt='none',
    c='black',
    capsize=5
)

plt.title("Model Accuracy: Mean & Standard Deviation")
plt.ylabel("Accuracy Score")
plt.ylim(0, 1.0)
plt.show()


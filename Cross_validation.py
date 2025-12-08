import numpy as np
from pre_processing import X, Y
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression, LinearRegression
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

X = X.astype(np.float32)
if Y.ndim > 1:
    y = np.argmax(Y, axis=1)
else:
    y = Y.copy()

class LinearRegressionOvA:
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
    "Neural Network (MLP)": MLPClassifier(
        hidden_layer_sizes=(256, 128, 64),
        activation='relu',
        solver='adam',
        learning_rate_init=0.001,
        max_iter=50,
        random_state=42
    ),
    "Logistic Regression": LogisticRegression(
        solver='lbfgs', multi_class='multinomial', max_iter=500, random_state=42
    ),
    "Linear Regression (OvA)": LinearRegressionOvA()
}

k = 5
skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)

results = {}
cv_predictions = {}

for name, model in models.items():
    print("\n===================================")
    print(f"Training Model: {name}")
    print("===================================")

    fold_scores = []
    preds_all = np.zeros_like(y)

    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        preds_all[test_idx] = y_pred
        acc = (y_pred == y_test).mean()
        fold_scores.append(acc)

        print(f"Fold {fold+1} Accuracy: {acc:.4f}")

    results[name] = np.array(fold_scores)
    cv_predictions[name] = preds_all

    print("\nClassification Report (aggregated on all samples):")
    print(classification_report(y, preds_all))

plt.figure(figsize=(7, 5))
avg_acc = [results[m].mean() for m in results]
plt.bar(list(results.keys()), avg_acc)
plt.title("Average Accuracy per Model (5-Fold Cross-Validation)")
plt.ylabel("Accuracy")
plt.ylim(0, 1)
plt.show()

plt.figure(figsize=(7, 5))
plt.boxplot(list(results.values()), labels=list(results.keys()))
plt.title("Performance Distribution Across Folds")
plt.ylabel("Accuracy")
plt.show()

print("\nAll Models Evaluated Successfully!")

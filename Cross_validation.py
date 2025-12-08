import numpy as np
from pre_processing import X, Y
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression, LinearRegression
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


X = X.astype(np.float32) / 255.0
y = Y.copy()

if y.ndim > 1:
    y = np.argmax(y, axis=1)


class LinearRegressionClassifier:
    def __init__(self):
        self.model = LinearRegression()

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        preds = np.round(self.model.predict(X))
        preds = np.clip(preds, 0, 9)
        return preds.astype(int)


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
        solver='lbfgs', multi_class='auto', max_iter=200
    ),
    "Linear Regression": LinearRegressionClassifier()
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

    print("\nClassification Report:")
    print(classification_report(y, preds_all))


plt.figure(figsize=(7, 5))
avg_acc = [results[m].mean() for m in results]
plt.bar(results.keys(), avg_acc)
plt.title("Average Accuracy per Model (5-Fold Cross-Validation)")
plt.ylabel("Accuracy")
plt.ylim(0, 1)
plt.show()


plt.figure(figsize=(7, 5))
plt.boxplot(results.values(), labels=results.keys())
plt.title("Performance Distribution Across Folds")
plt.ylabel("Accuracy")
plt.show()


print("\nAll Models Evaluated Successfully!")

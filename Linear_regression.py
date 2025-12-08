import numpy as np
from pre_processing import X_train, X_test, Y_train, Y_test
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import random

if Y_train.ndim > 1:
    y_train = np.argmax(Y_train, axis=1)
else:
    y_train = Y_train.copy()

if Y_test.ndim > 1:
    y_test = np.argmax(Y_test, axis=1)
else:
    y_test = Y_test.copy()

class OVA_LinearRegression:
    """One-vs-all wrapper around sklearn LinearRegression to act as a classifier."""
    def __init__(self):
        self.models = []

    def fit(self, X, y):
        self.models = []
        classes = np.unique(y)
        for c in classes:
            # target is 1 for class c, 0 otherwise
            y_bin = (y == c).astype(float)
            lr = LinearRegression()
            lr.fit(X, y_bin)
            self.models.append(lr)

    def predict(self, X):
        # Each model predicts a continuous score; pick argmax across classes
        preds = np.column_stack([m.predict(X) for m in self.models])
        return np.argmax(preds, axis=1)


model = OVA_LinearRegression()
print("Training Linear Regression (OvA)...")
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"Linear Regression (OvA) Test Accuracy: {acc*100:.2f}%")

idx = random.randint(0, X_test.shape[0] - 1)
img = X_test[idx].reshape(28, 28)
plt.imshow(img, cmap='gray')
plt.title(f"Predicted: {y_pred[idx]} | True: {y_test[idx]}")
plt.axis('off')
plt.show()

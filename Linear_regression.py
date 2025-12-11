import numpy as np
from pre_processing import X_train, X_test, y_train_labels, y_test_labels
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score
import matplotlib
matplotlib.use('TkAgg') #for backend
import matplotlib.pyplot as plt
import random

print("Training Linear Regression (OvA)...")

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


model = OVA_LinearRegression()

model.fit(X_train, y_train_labels)

y_pred = model.predict(X_test)
acc = accuracy_score(y_test_labels, y_pred)

idx = random.randint(0, X_test.shape[0] - 1)
img = X_test[idx].reshape(28, 28)

print("\n==============================")
print(f"Linear Regression (OvA) Test Accuracy: {acc*100:.2f}%")
print("==============================\n")
print(f"Predicted: {y_pred[idx]} | True: {y_test_labels[idx]}")


plt.imshow(img, cmap='gray')
plt.title(f"Predicted: {y_pred[idx]} | True: {y_test_labels[idx]}")
plt.axis('off')
plt.show()

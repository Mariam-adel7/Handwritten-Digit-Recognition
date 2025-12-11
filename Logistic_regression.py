import numpy as np
from pre_processing import X_train, X_test, y_train_labels, y_test_labels
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import random

print("Training Logistic Regression...")

model = LogisticRegression(
    solver='lbfgs',
    multi_class='multinomial',
    max_iter=500,
    random_state=42
)

model.fit(X_train, y_train_labels)

y_pred = model.predict(X_test)

acc = accuracy_score(y_test_labels, y_pred)

idx = random.randint(0, X_test.shape[0] - 1)
img = X_test[idx].reshape(28, 28)

print("\n==============================")
print(f"Logistic Regression Test Accuracy: {acc*100:.2f}%")
print("==============================\n")

print("Predicted:", y_pred[idx])
print("True:     ", y_test_labels[idx])

plt.imshow(img, cmap='gray')
plt.title(f"Predicted: {y_pred[idx]} | True: {y_test_labels[idx]}")
plt.axis('off')
plt.show()

import numpy as np
from preprocessing import X_train, X_test, y_train, y_test, pca
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import random

print("Training Logistic Regression...")

model = LogisticRegression(
    solver='lbfgs',
    max_iter=1000, 
    random_state=42
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)

print("\n==============================")
print(f"Logistic Regression Test Accuracy: {acc*100:.2f}%")
print("==============================\n")

idx = random.randint(0, X_test.shape[0] - 1)
img = X_test[idx]

img_original = pca.inverse_transform(img).reshape(28, 28)

print(f"Predicted: {y_pred[idx]} | True: {y_test[idx]}\n")

plt.imshow(img_original, cmap='gray')
plt.title(f"Predicted: {y_pred[idx]} | True: {y_test[idx]}")
plt.axis('off')
plt.show()

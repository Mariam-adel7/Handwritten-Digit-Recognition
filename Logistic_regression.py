import numpy as np
from pre_processing import X_train, X_test, Y_train, Y_test
from sklearn.linear_model import LogisticRegression
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

model = LogisticRegression(
    solver='lbfgs',
    multi_class='multinomial',
    max_iter=500,
    random_state=42
)

print("Training Logistic Regression (sklearn)...")
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"Logistic Regression Test Accuracy: {acc*100:.2f}%")

# Visual check
idx = random.randint(0, X_test.shape[0] - 1)
img = X_test[idx].reshape(28, 28)
plt.imshow(img, cmap='gray')
plt.title(f"Predicted: {y_pred[idx]} | True: {y_test[idx]}")
plt.axis('off')
plt.show()

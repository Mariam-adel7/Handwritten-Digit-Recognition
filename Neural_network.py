from pre_processing import X_train, X_test, y_train, y_test
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report
import numpy as np
import random
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

print("Training Neural Network...")

model = MLPClassifier(
    hidden_layer_sizes=(256, 128, 64),
    activation='relu',
    solver='adam',
    learning_rate_init=0.001,
    max_iter=40,
    random_state=42,
    verbose=True
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

acc = accuracy_score(y_test, y_pred)

idx = random.randint(0, X_test.shape[0] - 1)
img = X_test[idx].reshape(28, 28)

print("\n==============================")
print(f"Neural Network Test Accuracy: {acc*100:.2f}%")
print("==============================\n")
print(f"Predicted: {y_pred[idx]} | True: {y_test[idx]}")

plt.imshow(img, cmap='gray')
plt.title(f"Predicted: {y_pred[idx]} | True: {y_test[idx]}")
plt.axis('off')
plt.show()


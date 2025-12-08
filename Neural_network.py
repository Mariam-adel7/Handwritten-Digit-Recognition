=import numpy as np
from pre_processing import X_train, X_test, Y_train, Y_test
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report
import random
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

if Y_train.ndim > 1:
    y_train = np.argmax(Y_train, axis=1)
else:
    y_train = Y_train.copy()

if Y_test.ndim > 1:
    y_test = np.argmax(Y_test, axis=1)
else:
    y_test = Y_test.copy()

model = MLPClassifier(
    hidden_layer_sizes=(256, 128, 64),
    activation='relu',
    solver='adam',
    learning_rate_init=0.001,
    max_iter=40,
    random_state=42,
    verbose=True
)

print("\nTraining Neural Network (sklearn MLP)...")
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)

print("\n==============================")
print(f"Final Test Accuracy: {acc*100:.2f}%")
print("==============================\n")

print("Classification Report:\n")
print(classification_report(y_test, y_pred))

idx = random.randint(0, X_test.shape[0] - 1)
img = X_test[idx].reshape(28, 28)
plt.imshow(img, cmap='gray')
plt.title(f"Predicted: {y_pred[idx]} | True: {y_test[idx]}")
plt.axis('off')
plt.show()

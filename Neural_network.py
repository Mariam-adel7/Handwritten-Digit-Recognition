from pre_processing import X, Y, X_train, X_test, Y_train, Y_test, y_train_labels, y_test_labels,Y_encoded
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report
import random
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


model = MLPClassifier(
    hidden_layer_sizes=(256, 128, 64),
    activation='relu',
    solver='adam',
    learning_rate_init=0.001,
    max_iter=40,
    verbose=True
)

print("\nTraining Neural Network...")
model.fit(X_train, Y_train)

y_pred = model.predict(X_test)
acc = accuracy_score(Y_test, y_pred)

print("\n==============================")
print(f"Final Test Accuracy: {acc*100:.2f}%")
print("==============================\n")

print("Classification Report:\n")
print(classification_report(Y_test, y_pred))


idx = random.randint(0, X_test.shape[0] - 1)
img = X_test[idx].reshape(28, 28)

plt.imshow(img, cmap='gray')
plt.title(f"Predicted: {y_pred[idx]} | True: {Y_test[idx]}")
plt.axis('off')
plt.show()

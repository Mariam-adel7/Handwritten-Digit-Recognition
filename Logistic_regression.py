from pre_processing import X, Y, X_train, X_test, Y_train, Y_test, y_train_labels, y_test_labels,Y_encoded
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


np.random.seed(42)
num_features = X_train.shape[1]
num_classes = 10

W = np.random.randn(num_features, num_classes) * 0.01
b = np.zeros((1, num_classes))

learning_rate = 0.1
epochs = 100

def softmax(z):
    exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)


def cross_entropy(Y_true, Y_pred):
    return -np.mean(np.sum(Y_true * np.log(Y_pred + 1e-9), axis=1))


plt.figure(figsize=(3,3))

for epoch in range(epochs):

    Z = np.dot(X_train, W) + b
    Y_pred = softmax(Z)

    loss = cross_entropy(Y_train, Y_pred)

    dZ = (Y_pred - Y_train) / X_train.shape[0]
    dW = np.dot(X_train.T, dZ)
    db = np.sum(dZ, axis=0, keepdims=True)

    W -= learning_rate * dW
    b -= learning_rate * db

    train_index = np.random.randint(0, X_train.shape[0])
    image = X_train[train_index].reshape(28, 28)
    predicted_label = np.argmax(softmax(np.dot(X_train[train_index].reshape(1, -1), W) + b))

    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {loss:.4f}")
        plt.imshow(image, cmap='gray')
        plt.title(f"Epoch {epoch} | Predicted: {predicted_label}")
        plt.axis('off')
        plt.pause(0.5)
        plt.clf()


Z_test = np.dot(X_test, W) + b
Y_test_pred = softmax(Z_test)
y_pred_labels = np.argmax(Y_test_pred, axis=1)

accuracy = np.mean(y_pred_labels == y_test_labels)
print(f"Test Accuracy: {accuracy * 100:.2f}%")






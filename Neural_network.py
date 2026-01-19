from preprocessing import X_train, X_test, y_train, y_test,pca
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
import random
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

print("Training Neural Network...")

model = MLPClassifier(
    hidden_layer_sizes=(256, 128, 64),
    activation='relu',
    learning_rate='adaptive',
    learning_rate_init=0.001,
    alpha=0.01,
    max_iter=1500,
    batch_size=64,
    random_state=0,
    early_stopping=True
)

model.fit(X_train, y_train)
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)

print(f"\nNeural Network Test Accuracy: {acc*100:.2f}%")

idx = random.randint(0, X_test.shape[0] - 1)
img = X_test[idx]
img_original = pca.inverse_transform(img.reshape(1, -1)).reshape(28, 28)

plt.imshow(img_original, cmap='gray')
plt.title(f"Predicted: {y_pred[idx]} | True: {y_test[idx]}")
plt.axis('off')
plt.show()



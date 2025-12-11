import os
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

data_dir = "images"
image_size = (28, 28)

X = []
Y = []

for digit in range(10):
    folder = os.path.join(data_dir, str(digit))
    for filename in os.listdir(folder):
        img_path = os.path.join(folder, filename)
        try:
            img = Image.open(img_path).convert('L')
            img = img.resize(image_size)
            X.append(np.array(img).flatten() / 255.0)
            Y.append(digit)
        except:
            print(f"Error loading image: {img_path}")

X = np.array(X)
Y = np.array(Y)

ohe = OneHotEncoder()
Y_encoded = ohe.fit_transform(Y.reshape(-1,1)).toarray()

X_train, X_test, y_train, y_test, y_train_labels, y_test_labels = train_test_split(
    X, Y_encoded, Y, test_size=0.2, random_state=50)

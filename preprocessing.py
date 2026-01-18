import os
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(BASE_DIR, "images")

X = []
y = []   

for digit in range(10):
    folder = os.path.join(data_dir, str(digit))
    for filename in os.listdir(folder):
        img_path = os.path.join(folder, filename)
        img = Image.open(img_path).convert('L')
        img = img.resize((28, 28))
        X.append(np.array(img).flatten())
        y.append(digit)


X = np.array(X)
y = np.array(y)
X = X / 255.0

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

pca = PCA(n_components=300)
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)





# used for training the model with dataset

import os
import numpy as np
import cv2
import json
from sklearn.model_selection import train_test_split

DATASET_DIR = "dataset"
IMAGE_SIZE = (128, 128)

X = []
y = []
labels = {}

def prepare_data():
    class_names = os.listdir(DATASET_DIR)
    
    for idx, class_name in enumerate(class_names):
        labels[idx] = class_name
        class_dir = os.path.join(DATASET_DIR, class_name)
        for img_name in os.listdir(class_dir):
            img_path = os.path.join(class_dir, img_name)
            img = cv2.imread(img_path)
            if img is not None:
                img = cv2.resize(img, IMAGE_SIZE)
                X.append(img)
                y.append(idx)

    # Save label mapping
    with open("label_map.json", "w") as f:
        json.dump(labels, f)

    return np.array(X), np.array(y)

if __name__ == "__main__":
    X, y = prepare_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Save as .npy files for model training
    np.save("X_train.npy", X_train)
    np.save("X_test.npy", X_test)
    np.save("y_train.npy", y_train)
    np.save("y_test.npy", y_test)

    print("✅ Dataset prepared and saved!")

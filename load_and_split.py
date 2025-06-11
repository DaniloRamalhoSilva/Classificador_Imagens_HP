import os
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split


def load_images_from_folder(folder_path):
    """Load images from a directory and return a numpy array."""
    images = []
    for img_name in os.listdir(folder_path):
        img_path = os.path.join(folder_path, img_name)
        if not os.path.isfile(img_path):
            continue
        img = tf.keras.utils.load_img(img_path, target_size=(224, 224))
        img_array = tf.keras.utils.img_to_array(img)
        img_array = img_array / 255.0
        images.append(img_array)
    return np.array(images)


HP_DIR = 'dataset/HP_Original'
OTHER_DIR = 'dataset/Outros'

hp_images = load_images_from_folder(HP_DIR)
other_images = load_images_from_folder(OTHER_DIR)

hp_labels = np.zeros(len(hp_images))
other_labels = np.ones(len(other_images))

X = np.concatenate([hp_images, other_images], axis=0)
y = np.concatenate([hp_labels, other_labels], axis=0)

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

print(f'Train images: {X_train.shape}, Train labels: {y_train.shape}')
print(f'Validation images: {X_val.shape}, Validation labels: {y_val.shape}')


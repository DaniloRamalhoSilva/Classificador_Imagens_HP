import os
import numpy as np
import tensorflow as tf
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

from cnn_model import create_cnn_model

DATA_DIR = 'dataset'
IMG_SIZE = (224, 224)
BATCH_SIZE = 16
EPOCHS = 5
VALIDATION_SPLIT = 0.2


def load_dataset(
    data_dir: str = DATA_DIR,
    img_size: tuple = IMG_SIZE,
    batch_size: int = BATCH_SIZE,
    validation_split: float = VALIDATION_SPLIT,
):
    """Load datasets from directories including .webp images."""

    class_dirs = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
    images, labels = [], []

    for label, class_name in enumerate(sorted(class_dirs)):
        class_path = os.path.join(data_dir, class_name)
        for fname in os.listdir(class_path):
            file_path = os.path.join(class_path, fname)
            if not os.path.isfile(file_path):
                continue
            try:
                img = tf.keras.utils.load_img(file_path, target_size=img_size)
            except Exception:
                # Skip unreadable files
                continue
            img_array = tf.keras.utils.img_to_array(img) / 255.0
            images.append(img_array)
            labels.append(label)

    images = np.array(images)
    labels = np.array(labels)

    dataset = tf.data.Dataset.from_tensor_slices((images, labels))
    dataset = dataset.shuffle(len(images), seed=123)

    val_size = int(len(images) * validation_split)
    val_ds = dataset.take(val_size).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    train_ds = dataset.skip(val_size).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return train_ds, val_ds


def train(model, train_ds, val_ds, epochs=EPOCHS):
    """Train the model."""
    history = model.fit(train_ds, epochs=epochs, validation_data=val_ds)
    return history


def evaluate(model, val_ds):
    """Evaluate the model on the validation dataset and print metrics."""
    probabilities = model.predict(val_ds)
    predictions = np.argmax(probabilities, axis=1)
    labels = np.concatenate([y for _, y in val_ds], axis=0)

    acc = accuracy_score(labels, predictions)
    prec = precision_score(labels, predictions)
    rec = recall_score(labels, predictions)
    f1 = f1_score(labels, predictions)

    print(f"Acur\u00e1cia: {acc:.4f}")
    print(f"Precis\u00e3o: {prec:.4f}")
    print(f"Recall: {rec:.4f}")
    print(f"F1-score: {f1:.4f}")

    cm = confusion_matrix(labels, predictions)
    print("Matriz de confus\u00e3o:\n", cm)


def main():
    train_ds, val_ds = load_dataset()
    model = create_cnn_model(num_classes=2)
    train(model, train_ds, val_ds)
    evaluate(model, val_ds)
    model.save('hp_classifier.h5')
    print('Modelo salvo em hp_classifier.h5')


if __name__ == '__main__':
    main()

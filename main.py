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


def load_dataset(data_dir=DATA_DIR, img_size=IMG_SIZE, batch_size=BATCH_SIZE, validation_split=VALIDATION_SPLIT):
    """Load training and validation datasets."""
    train_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=validation_split,
        subset="training",
        seed=123,
        image_size=img_size,
        batch_size=batch_size,
    )
    val_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=validation_split,
        subset="validation",
        seed=123,
        image_size=img_size,
        batch_size=batch_size,
    )

    autotune = tf.data.AUTOTUNE
    train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=autotune)
    val_ds = val_ds.cache().prefetch(buffer_size=autotune)
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

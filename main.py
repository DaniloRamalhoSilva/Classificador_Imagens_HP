import os
import numpy as np
import math
import tensorflow as tf
import matplotlib.pyplot as plt
import pickle

from tensorflow.keras import layers, models
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc, classification_report
from sklearn.metrics import precision_recall_curve, average_precision_score

from cnn_model import create_cnn_model
from convert_webp_to_png import convert_webp_to_png

DATA_DIR = os.path.join(os.path.dirname(__file__), 'dataset')
GRAPH_DIR = os.path.join(os.path.dirname(__file__), 'graficos')
os.makedirs(GRAPH_DIR, exist_ok=True)
IMG_SIZE = (224, 224)
BATCH_SIZE = 16
EPOCHS = 10
VALIDATION_SPLIT = 0.2


def load_dataset(data_dir=DATA_DIR,
                 img_size=IMG_SIZE,
                 batch_size=BATCH_SIZE,
                 validation_split=VALIDATION_SPLIT):
    train_raw = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=validation_split,
        subset="training",
        seed=123,
        image_size=img_size,
        batch_size=batch_size,
    )
    val_raw = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=validation_split,
        subset="validation",
        seed=123,
        image_size=img_size,
        batch_size=batch_size,
    )

    class_names = train_raw.class_names

    autotune = tf.data.AUTOTUNE
    train_ds = train_raw.cache().shuffle(1000).prefetch(buffer_size=autotune)
    val_ds   = val_raw.cache().prefetch(buffer_size=autotune)

    return train_ds, val_ds, class_names

def train(model, train_ds, val_ds, epochs=EPOCHS):
    history = model.fit(train_ds, epochs=epochs, validation_data=val_ds)
    return history

def plot_history(history):
    plt.figure(figsize=(8, 4))
    plt.plot(history.history['accuracy'], label='treino')
    plt.plot(history.history['val_accuracy'], label='validação')
    plt.title('Acurácia por Época')
    plt.xlabel('Época')
    plt.ylabel('Acurácia')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(GRAPH_DIR, 'history_accuracy.png'))
    plt.close()

    plt.figure(figsize=(8, 4))
    plt.plot(history.history['loss'], label='treino')
    plt.plot(history.history['val_loss'], label='validação')
    plt.title('Loss por Época')
    plt.xlabel('Época')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(GRAPH_DIR, 'history_loss.png'))
    plt.close()

def plot_confusion(model, val_ds, class_names):
    y_true = np.concatenate([y.numpy() for x, y in val_ds], axis=0)
    y_prob = model.predict(val_ds)
    y_pred = np.argmax(y_prob, axis=1)

    all_labels = np.arange(len(class_names))
    cm = confusion_matrix(y_true, y_pred, labels=all_labels)

    disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                  display_labels=class_names)
    disp.plot(cmap=plt.cm.Blues, values_format='d')
    plt.title('Matriz de Confusão')
    plt.tight_layout()
    plt.savefig(os.path.join(GRAPH_DIR, 'confusion_matrix.png'))
    plt.close()

def plot_roc(model, val_ds):
    y_true = np.concatenate([y.numpy() for x, y in val_ds], axis=0)
    y_prob = model.predict(val_ds)[:, 1]  

    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(6,6))
    plt.plot(fpr, tpr, label=f'AUC = {roc_auc:.2f}')
    plt.plot([0,1], [0,1], '--', label='Random')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Curva ROC')
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(GRAPH_DIR, 'roc_curve.png'))
    plt.close()

def plot_pr_curve(model, val_ds):
    y_true = np.concatenate([y.numpy() for x, y in val_ds], axis=0)
    y_prob = model.predict(val_ds)[:, 1]

    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    ap = average_precision_score(y_true, y_prob)

    plt.figure(figsize=(6,6))
    plt.plot(recall, precision, label=f'AP = {ap:.2f}')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Curva Precision-Recall')
    plt.legend(loc='upper right')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(GRAPH_DIR, 'pr_curve.png'))
    plt.close()

def plot_classification_report(model, val_ds, class_names):
    y_true = np.concatenate([y.numpy() for x, y in val_ds], axis=0)
    y_pred = np.argmax(model.predict(val_ds), axis=1)

    report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
    precisions = [report[c]['precision'] for c in class_names]
    recalls    = [report[c]['recall']    for c in class_names]
    f1s        = [report[c]['f1-score']  for c in class_names]

    x = np.arange(len(class_names))
    width = 0.25

    plt.figure(figsize=(8,4))
    plt.bar(x - width, precisions, width, label='Precision')
    plt.bar(x, recalls,    width, label='Recall')
    plt.bar(x + width, f1s, width, label='F1-score')
    plt.xticks(x, class_names)
    plt.ylabel('Score')
    plt.title('Métricas por Classe')
    plt.legend()
    plt.grid(True, axis='y')
    plt.tight_layout()
    plt.savefig(os.path.join(GRAPH_DIR, 'classification_report.png'))
    plt.close()


def main():
    convert_webp_to_png(DATA_DIR)
    train_ds, val_ds, class_names = load_dataset()
    model = create_cnn_model(num_classes=2)
    history = train(model, train_ds, val_ds)
    model.save('hp_classifier.h5')
    print('Modelo salvo em hp_classifier.h5')
    plot_history(history)
    plot_confusion(model, val_ds, class_names)
    plot_roc(model, val_ds)
    plot_pr_curve(model, val_ds)
    plot_classification_report(model, val_ds, class_names)


if __name__ == '__main__':
    main()


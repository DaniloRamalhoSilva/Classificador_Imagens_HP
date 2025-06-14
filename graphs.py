"""Funções para geração de gráficos de treinamento e avaliação.

Cada função salva o gráfico correspondente na pasta ``graficos`` localizada no
mesmo diretório deste script. Essas imagens auxiliam na análise do desempenho
do classificador.
"""

import os
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    auc,
    average_precision_score,
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    roc_curve,
)

# Diretório padrão onde os gráficos serão salvos
GRAPH_DIR = os.path.join(os.path.dirname(__file__), "graficos")
os.makedirs(GRAPH_DIR, exist_ok=True)


def plot_history(history) -> None:
    """Salva curvas de acurácia e perda ao longo do treinamento.

    Parameters
    ----------
    history : keras.callbacks.History
        Objeto retornado por ``model.fit`` contendo as métricas registradas
        em cada época.
    """

    # --- Curva de Acurácia ---
    plt.figure(figsize=(8, 4))
    plt.plot(history.history["accuracy"], label="treino")
    plt.plot(history.history["val_accuracy"], label="validação")
    plt.title("Acurácia por Época")
    plt.xlabel("Época")
    plt.ylabel("Acurácia")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(GRAPH_DIR, "history_accuracy.png"))
    plt.close()

    # --- Curva de Loss ---
    plt.figure(figsize=(8, 4))
    plt.plot(history.history["loss"], label="treino")
    plt.plot(history.history["val_loss"], label="validação")
    plt.title("Loss por Época")
    plt.xlabel("Época")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(GRAPH_DIR, "history_loss.png"))
    plt.close()


def plot_confusion(model, val_ds, class_names: Sequence[str]) -> None:
    """Gera e salva a matriz de confusão do modelo.

    Parameters
    ----------
    model : keras.Model
        Modelo treinado utilizado para realizar as previsões.
    val_ds : tf.data.Dataset
        Conjunto de validação contendo imagens e rótulos.
    class_names : Sequence[str]
        Lista com os nomes das classes na ordem dos rótulos.
    """

    # Labels reais e previsões do modelo
    y_true = np.concatenate([y.numpy() for _, y in val_ds], axis=0)
    y_prob = model.predict(val_ds)
    y_pred = np.argmax(y_prob, axis=1)

    # Matriz de confusão considerando todas as classes
    all_labels = np.arange(len(class_names))
    cm = confusion_matrix(y_true, y_pred, labels=all_labels)

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(cmap=plt.cm.Blues, values_format="d")
    plt.title("Matriz de Confusão")
    plt.tight_layout()
    plt.savefig(os.path.join(GRAPH_DIR, "confusion_matrix.png"))
    plt.close()


def plot_roc(model, val_ds) -> None:
    """Plota a curva ROC usando o conjunto de validação.

    A curva Receiver Operating Characteristic (ROC) mostra a relação entre o
    *True Positive Rate* e o *False Positive Rate* para diferentes limiares.
    """

    # Rótulos verdadeiros e probabilidades da classe positiva
    y_true = np.concatenate([y.numpy() for _, y in val_ds], axis=0)
    y_prob = model.predict(val_ds)[:, 1]

    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
    plt.plot([0, 1], [0, 1], "--", label="Random")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Curva ROC")
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(GRAPH_DIR, "roc_curve.png"))
    plt.close()


def plot_pr_curve(model, val_ds) -> None:
    """Salva a curva Precision-Recall baseada no conjunto de validação."""

    y_true = np.concatenate([y.numpy() for _, y in val_ds], axis=0)
    y_prob = model.predict(val_ds)[:, 1]

    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    ap = average_precision_score(y_true, y_prob)

    plt.figure(figsize=(6, 6))
    plt.plot(recall, precision, label=f"AP = {ap:.2f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Curva Precision-Recall")
    plt.legend(loc="upper right")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(GRAPH_DIR, "pr_curve.png"))
    plt.close()


def plot_classification_report(model, val_ds, class_names: Sequence[str]) -> None:
    """Gera um resumo das métricas por classe em formato de gráfico de barras."""

    y_true = np.concatenate([y.numpy() for _, y in val_ds], axis=0)
    y_pred = np.argmax(model.predict(val_ds), axis=1)

    report = classification_report(
        y_true, y_pred, target_names=class_names, output_dict=True
    )
    precisions = [report[c]["precision"] for c in class_names]
    recalls = [report[c]["recall"] for c in class_names]
    f1s = [report[c]["f1-score"] for c in class_names]

    x = np.arange(len(class_names))
    width = 0.25

    plt.figure(figsize=(8, 4))
    plt.bar(x - width, precisions, width, label="Precision")
    plt.bar(x, recalls, width, label="Recall")
    plt.bar(x + width, f1s, width, label="F1-score")
    plt.xticks(x, class_names)
    plt.ylabel("Score")
    plt.title("Métricas por Classe")
    plt.legend()
    plt.grid(True, axis="y")
    plt.tight_layout()
    plt.savefig(os.path.join(GRAPH_DIR, "classification_report.png"))
    plt.close()

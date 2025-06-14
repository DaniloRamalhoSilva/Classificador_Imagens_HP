import io
import os
import tensorflow as tf
from sklearn.metrics import classification_report, roc_curve, auc, precision_recall_curve, average_precision_score

from classificador_imagens_hp import create_cnn_model, load_dataset, train
from graphs import (
    plot_history,
    plot_confusion,
    plot_roc,
    plot_pr_curve,
    plot_classification_report,
    GRAPH_DIR,
)


def generate_sprint1_report(report_path="SPRINT1_REPORT.md"):
    """Treina o modelo, gera gráficos e cria um relatório Markdown."""

    # Carrega os dados
    train_ds, val_ds, class_names = load_dataset()

    # Estatísticas do dataset
    data_dir = os.path.join(os.path.dirname(__file__), "dataset")
    class_counts = {cn: len(os.listdir(os.path.join(data_dir, cn))) for cn in class_names}
    total_images = sum(class_counts.values())

    # Cria e treina o modelo
    model = create_cnn_model(num_classes=len(class_names))
    history = train(model, train_ds, val_ds)

    # Obtém resumo do modelo
    buf = io.StringIO()
    model.summary(print_fn=lambda x: buf.write(x + "\n"))
    model_summary = buf.getvalue()

    # Geração dos gráficos
    plot_history(history)
    plot_confusion(model, val_ds, class_names)
    plot_roc(model, val_ds)
    plot_pr_curve(model, val_ds)
    plot_classification_report(model, val_ds, class_names)

    # Métricas por classe em formato de tabela
    y_true = tf.concat([y for _, y in val_ds], axis=0).numpy()
    y_prob = model.predict(val_ds)
    y_pred = y_prob.argmax(axis=1)
    report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)

    metrics_table = "| Classe | Precision | Recall | F1-score |\n|---|---|---|---|\n"
    for cn in class_names:
        metrics_table += f"| {cn} | {report[cn]['precision']:.2f} | {report[cn]['recall']:.2f} | {report[cn]['f1-score']:.2f} |\n"

    # Calcula AUC e AP
    fpr, tpr, _ = roc_curve(y_true, y_prob[:, 1])
    roc_auc = auc(fpr, tpr)
    precision, recall, _ = precision_recall_curve(y_true, y_prob[:, 1])
    ap = average_precision_score(y_true, y_prob[:, 1])

    # Acurácia final
    final_acc = history.history["val_accuracy"][-1] * 100

    # Estrutura de diretórios
    dir_structure = "".join(f"  - dataset/{cn}\n" for cn in class_names)

    # Montagem do markdown
    md = [
        "## Montagem do Dataset",
        "",
        "- **Critérios de seleção**: foram coletadas imagens considerando preço médio, avaliação, tipo de loja e comentários.",
        "- **Quantitativo de imagens**:",
    ]
    for cn in class_names:
        md.append(f"  - {cn}: {class_counts[cn]}")
    md.append(f"  - total: {total_images}")
    md.extend([
        "- **Estrutura de diretórios** (`/content/dataset`):",
        dir_structure.rstrip(),
        "- **Pré-processamento**:",
        "  - imagens redimensionadas para 224x224",
        "  - normalização automática pelo `image_dataset_from_directory`",
        "  - divisão treino/validação 80/20 usando `validation_split=0.2`",
        "",
        "## Estrutura da Rede Convolucional",
        "",
        "Camadas do modelo:",
        "  - " + " \u2192 ".join(layer.__class__.__name__ for layer in model.layers),
        "",
        "Resumo do modelo:",
        "```",
        model_summary,
        "```",
        "",
        "## Gráficos e Interpretação",
        "",
        "### 3.1 Curvas de treino × validação",
        f"![accuracy]({os.path.join('graficos', 'history_accuracy.png')})",
        f"![loss]({os.path.join('graficos', 'history_loss.png')})",
        "",
        "### 3.2 Matriz de Confusão",
        f"![confusion]({os.path.join('graficos', 'confusion_matrix.png')})",
        "",
        "### 3.3 Curva ROC",
        f"![roc]({os.path.join('graficos', 'roc_curve.png')})",
        f"AUC: {roc_auc:.2f}",
        "",
        "### 3.4 Curva Precision-Recall",
        f"![pr]({os.path.join('graficos', 'pr_curve.png')})",
        f"AP: {ap:.2f}",
        "",
        "### 3.5 Métricas por Classe",
        f"![report]({os.path.join('graficos', 'classification_report.png')})",
        "",
        metrics_table,
        "## Acurácia Final do Modelo",
        f"Último valor de `val_accuracy`: **{final_acc:.0f}%**",
        "",
        "## Próximos Passos Sugeridos",
        "- Aumentar o número de imagens por classe para reduzir overfitting.",
        "- Testar técnicas de data augmentation (giro, zoom) para melhorar generalização.",
        "- Avaliar modelos pré-treinados como MobileNetV2 para comparar desempenho.",
    ])

    with open(report_path, "w", encoding="utf-8") as fp:
        fp.write("\n".join(md))

    print(f"Relatório salvo em {report_path}")


if __name__ == "__main__":
    generate_sprint1_report()

## Montagem do Dataset

- **Critérios de seleção**: foram coletadas imagens considerando preço médio, avaliação, tipo de loja e comentários.
- **Quantitativo de imagens**:
  - HP_original: 30
  - outros: 30
  - total: 60
- **Estrutura de diretórios** (`/content/dataset`):
  - dataset/HP_original
  - dataset/outros
- **Pré-processamento**:
  - imagens redimensionadas para 224x224
  - normalização automática pelo `image_dataset_from_directory`
  - divisão treino/validação 80/20 usando `validation_split=0.2`

## Estrutura da Rede Convolucional

Camadas do modelo:
  - Conv2D → MaxPooling2D → Conv2D → MaxPooling2D → Conv2D → MaxPooling2D → Flatten → Dense → Dense

Resumo do modelo:
```
Model: "sequential"
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┓
┃ Layer (type)                    ┃ Output Shape           ┃       Param # ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━┩
│ conv2d (Conv2D)                 │ (None, 224, 224, 32)   │           896 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ max_pooling2d (MaxPooling2D)    │ (None, 112, 112, 32)   │             0 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ conv2d_1 (Conv2D)               │ (None, 112, 112, 64)   │        18,496 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ max_pooling2d_1 (MaxPooling2D)  │ (None, 56, 56, 64)     │             0 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ conv2d_2 (Conv2D)               │ (None, 56, 56, 128)    │        73,856 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ max_pooling2d_2 (MaxPooling2D)  │ (None, 28, 28, 128)    │             0 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ flatten (Flatten)               │ (None, 100352)         │             0 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dense (Dense)                   │ (None, 64)             │     6,422,592 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dense_1 (Dense)                 │ (None, 2)              │           130 │
└─────────────────────────────────┴────────────────────────┴───────────────┘
 Total params: 19,547,912 (74.57 MB)
 Trainable params: 6,515,970 (24.86 MB)
 Non-trainable params: 0 (0.00 B)
 Optimizer params: 13,031,942 (49.71 MB)


```

## Gráficos e Interpretação

### 3.1 Curvas de treino × validação
![accuracy](graficos/history_accuracy.png)
![loss](graficos/history_loss.png)

### 3.2 Matriz de Confusão
![confusion](graficos/confusion_matrix.png)

### 3.3 Curva ROC
![roc](graficos/roc_curve.png)
AUC: 0.80

### 3.4 Curva Precision-Recall
![pr](graficos/pr_curve.png)
AP: 0.77

### 3.5 Métricas por Classe
![report](graficos/classification_report.png)

| Classe | Precision | Recall | F1-score |
|---|---|---|---|
| HP_original | 1.00 | 0.60 | 0.75 |
| outros | 0.78 | 1.00 | 0.88 |

## Acurácia Final do Modelo
Último valor de `val_accuracy`: **83%**

## Próximos Passos Sugeridos
- Aumentar o número de imagens por classe para reduzir overfitting.
- Testar técnicas de data augmentation (giro, zoom) para melhorar generalização.
- Avaliar modelos pré-treinados como MobileNetV2 para comparar desempenho.

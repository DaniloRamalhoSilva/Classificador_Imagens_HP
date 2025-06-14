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
┏━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━┓
┃ Layer (type)             ┃ Output Shape      ┃    Param # ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━┩
│ conv2d (Conv2D)          │ (None, 224, 224,  │        896 │
│                          │ 32)               │            │
├──────────────────────────┼───────────────────┼────────────┤
│ max_pooling2d            │ (None, 112, 112,  │          0 │
│ (MaxPooling2D)           │ 32)               │            │
├──────────────────────────┼───────────────────┼────────────┤
│ conv2d_1 (Conv2D)        │ (None, 112, 112,  │     18,496 │
│                          │ 64)               │            │
├──────────────────────────┼───────────────────┼────────────┤
│ max_pooling2d_1          │ (None, 56, 56,    │          0 │
│ (MaxPooling2D)           │ 64)               │            │
├──────────────────────────┼───────────────────┼────────────┤
│ conv2d_2 (Conv2D)        │ (None, 56, 56,    │     73,856 │
│                          │ 128)              │            │
├──────────────────────────┼───────────────────┼────────────┤
│ max_pooling2d_2          │ (None, 28, 28,    │          0 │
│ (MaxPooling2D)           │ 128)              │            │
├──────────────────────────┼───────────────────┼────────────┤
│ flatten (Flatten)        │ (None, 100352)    │          0 │
├──────────────────────────┼───────────────────┼────────────┤
│ dense (Dense)            │ (None, 64)        │  6,422,592 │
├──────────────────────────┼───────────────────┼────────────┤
│ dense_1 (Dense)          │ (None, 2)         │        130 │
└──────────────────────────┴───────────────────┴────────────┘
 Total params: 19,547,912 (74.57 MB)
 Trainable params: 6,515,970 (24.86 MB)
 Non-trainable params: 0 (0.00 B)
 Optimizer params: 13,031,942 (49.71 MB)


```

## Gráficos e Interpretação

### 3.1 Curvas de treino × validação
![accuracy](graficos\history_accuracy.png)
![loss](graficos\history_loss.png)

### 3.2 Matriz de Confusão
![confusion](graficos\confusion_matrix.png)

### 3.3 Curva ROC
![roc](graficos\roc_curve.png)
AUC: 0.69

### 3.4 Curva Precision-Recall
![pr](graficos\pr_curve.png)
AP: 0.69

### 3.5 Métricas por Classe
![report](graficos\classification_report.png)

| Classe | Precision | Recall | F1-score |
|---|---|---|---|
| HP_original | 0.60 | 0.60 | 0.60 |
| outros | 0.71 | 0.71 | 0.71 |

## Acurácia Final do Modelo
Último valor de `val_accuracy`: **67%**

## Próximos Passos Sugeridos
- Aumentar o número de imagens por classe para reduzir overfitting.
- Testar técnicas de data augmentation (giro, zoom) para melhorar generalização.
- Avaliar modelos pré-treinados como MobileNetV2 para comparar desempenho.
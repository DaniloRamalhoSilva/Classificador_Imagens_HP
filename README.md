# Classificador de Imagens HP 📦

Bem-vindo ao projeto da **Sprint 1**! Aqui treinamos um modelo simples para identificar cartuchos HP originais e de outras marcas.

## Objetivo da Sprint
O objetivo é montar um pequeno dataset de imagens ✨, criar um classificador em Python e avaliar sua acurácia de forma rápida.

## Estrutura do Dataset 🗌

```
dataset/
├── HP_Original/   # imagens de cartuchos originais
├── Outros/       # imagens de cartuchos falsificados ou de outras marcas
```

## Tecnologias e Bibliotecas 💻
- TensorFlow / Keras
- NumPy
- Matplotlib

## Etapas do Projeto 📝
1. Coleta das imagens
2. Pré-processamento (redimensionamento e normalização)
3. Modelagem com uma CNN simples
4. Avaliação com acurácia e matriz de confusão

## Como Executar no Google Colab 👨‍💻
1. Acesse [Google Colab](https://colab.research.google.com/).
2. Crie um novo notebook e envie a pasta `dataset` com as subpastas `HP_Original` e `Outros`.
3. Copie o código deste repositório ou utilize `main.py` como referência.
4. Execute as células para treinar e avaliar o modelo.

## Resultados Esperados 🎉
Após o treinamento, os gráficos de acurácia, perda e avaliação são gerados pelo módulo `graphs.py` e salvos na pasta `graficos/`. Abra esses arquivos para verificar o desempenho do classificador.

## Possibilidades Futuras ⚡
- Converter o modelo final para o formato `.tflite` e rodá-lo em dispositivos móveis.

- O pipeline agora converte automaticamente qualquer imagem que não esteja em `PNG`.

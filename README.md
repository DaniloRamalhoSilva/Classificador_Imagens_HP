# Classificador_Imagens_HP
Projeto da Sprint 1 para detecção de cartuchos HP falsificados. Criação de dataset de imagens, pré-processamento e treinamento de modelo CNN com Keras/TensorFlow. Classificação binária entre cartuchos originais e falsificados, com avaliação de acurácia e matriz de confusão.

## Treinamento
Execute `train.py` para carregar as imagens do diretório `dataset`, treinar a rede leve baseada em MobileNetV2 e salvar o modelo resultante no formato Keras (`hp_classifier.h5`). Esse arquivo poderá ser convertido posteriormente para o formato TensorFlow Lite (.tflite).

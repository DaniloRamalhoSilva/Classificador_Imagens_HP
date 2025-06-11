SPRINT 1 – Geração do Dataset e Treinamento do Modelo

Objetivo: Criar um dataset de imagens e treinar um modelo de classificação simples em Python, testando a acurácia localmente.

Atividades: Nesta sprint, os alunos deverão buscar imagens na internet, sendo 30 de cartuchos HP originais e 30 de cartuchos falsificados ou de outras marcas. As imagens deverão ser organizadas em duas pastas: dataset/HP_Original e dataset/Outros. Em seguida, os alunos devem utilizar Python, preferencialmente no Google Colab, para carregar e pré-processar as imagens (incluindo redimensionamento e normalização), separar os dados em treino e validação (por exemplo, 80/20), treinar um modelo simples com bibliotecas como Keras, TensorFlow, scikit-learn ou fastai, e avaliar a acurácia do modelo.

Recomendação: É indicado o uso de modelos leves e compatíveis com exportação futura para .tflite, como redes convolucionais simples com Keras Sequential, MobileNetV2 (pré-treinada) ou outras com poucas camadas convolucionais.

Entrega: A entrega deve incluir o dataset organizado com as imagens separadas por classe, o notebook Python com o código comentado e o relatório em PDF. O relatório deve conter a explicação sobre a montagem do dataset, a estrutura da rede utilizada, prints dos gráficos de acurácia ou da matriz de confusão, e a acurácia final do modelo, mesmo que ainda não esteja exportado para .tflite.
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

try:
    import tensorflow as tf
except ImportError as e:
    raise SystemExit('TensorFlow is required to run this script. Install the package and retry.')

DATASET_DIR = 'dataset'
IMG_HEIGHT = 224
IMG_WIDTH = 224
BATCH_SIZE = 16
EPOCHS = 5

# Data loading using Keras utility
train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)

train_generator = train_datagen.flow_from_directory(
    DATASET_DIR,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='binary',
    subset='training'
)

validation_generator = train_datagen.flow_from_directory(
    DATASET_DIR,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='binary',
    subset='validation',
    shuffle=False
)

# Simple CNN model
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(16, (3,3), activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(train_generator, epochs=EPOCHS, validation_data=validation_generator)

# Evaluation block with confusion matrix and metrics
validation_generator.reset()
probabilities = model.predict(validation_generator)
predictions = (probabilities > 0.5).astype(int).ravel()
labels = validation_generator.classes

acc = accuracy_score(labels, predictions)
prec = precision_score(labels, predictions)
rec = recall_score(labels, predictions)
f1 = f1_score(labels, predictions)

print(f"Acur\xE1cia: {acc:.4f}")
print(f"Precis\xE3o: {prec:.4f}")
print(f"Recall: {rec:.4f}")
print(f"F1-score: {f1:.4f}")

cm = confusion_matrix(labels, predictions)
class_names = list(validation_generator.class_indices.keys())

plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.xlabel('Predito')
plt.ylabel('Real')
plt.title('Matriz de Confus\xE3o')
plt.tight_layout()
plt.show()


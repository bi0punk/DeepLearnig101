import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.sequence import pad_sequences
from prettytable import PrettyTable
from sklearn.metrics import confusion_matrix, classification_report

# Separar los datos en conjuntos de entrenamiento y prueba
x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.2, random_state=42)

# Resto del código sigue igual hasta el entrenamiento del modelo...

# Evaluar el modelo en el conjunto de prueba
x_test_encoded = tokenizer.texts_to_sequences(x_test)
x_test_padded = pad_sequences(x_test_encoded, maxlen=max_seq_length)
y_test_encoded = np.array([label_mapping[label] for label in y_test])
y_test_encoded = tf.keras.utils.to_categorical(y_test_encoded, num_classes=num_classes)

# Hacer predicciones en el conjunto de prueba
predictions_test = model.predict(x_test_padded)

# Decodificar las predicciones en el conjunto de prueba
decoded_predictions_test = [inverse_label_mapping[np.argmax(prediction)] for prediction in predictions_test]

# Imprimir la matriz de confusión y el reporte de clasificación en el conjunto de prueba
conf_matrix = confusion_matrix(y_test, decoded_predictions_test)
class_report = classification_report(y_test, decoded_predictions_test)

print("Matriz de Confusión:")
print(conf_matrix)

print("\nReporte de Clasificación:")
print(class_report)

# Resto del código sigue igual para la impresión de la tabla...

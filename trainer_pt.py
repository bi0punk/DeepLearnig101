import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
from prettytable import PrettyTable

# Definir los datos de entrenamiento originales
x_train = np.array(['Hola', 'Bien, ¿y tú?', 'Hola a todos'], dtype=object)
y_train = np.array(['Hola', 'Bien, ¿y tú?', 'Hola a todos'], dtype=object)

# Definir los datos de entrenamiento adicionales
x_train_additional = np.array(['¡Buenos días!', '¿Qué tal?', 'Hola, ¿cómo va todo?', 'Saludos a todos', 'Hola de nuevo',
                               '¡Hola, buenas tardes!', '¿Cómo estuvo tu día?', 'Hola, ¿qué haces?', 'Saludos amigos',
                               '¡Hola, qué gusto verte!'], dtype=object)

                               
y_train_additional = np.array(['Hola', 'Bien, ¿y tú?', 'Hola', 'Hola', 'Hola a todos', 'Hola', 'Bien, ¿y tú?',
                               'Hola', 'Hola a todos', 'Hola'], dtype=object)

# Agregar los datos adicionales a los conjuntos de entrenamiento existentes
x_train = np.concatenate((x_train, x_train_additional))
y_train = np.concatenate((y_train, y_train_additional))

# Convertir las etiquetas a valores numéricos
label_mapping = {'Hola': 0, 'Bien, ¿y tú?': 1, 'Hola a todos': 2}
y_train_encoded = np.array([label_mapping[label] for label in y_train])

# Tokenizar los datos de entrada
tokenizer = tf.keras.preprocessing.text.Tokenizer()
tokenizer.fit_on_texts(x_train)
x_train_encoded = tokenizer.texts_to_sequences(x_train)

# Padding de las secuencias
max_seq_length = max(len(seq) for seq in x_train_encoded)
x_train_padded = pad_sequences(x_train_encoded, maxlen=max_seq_length)

# Convertir las etiquetas en codificación one-hot
num_classes = len(label_mapping)
y_train_encoded = tf.keras.utils.to_categorical(y_train_encoded, num_classes=num_classes)

# Definir el modelo de la red neuronal
model = tf.keras.models.Sequential([
    tf.keras.layers.Embedding(len(tokenizer.word_index) + 1, 32, input_length=max_seq_length),
    tf.keras.layers.GlobalAveragePooling1D(),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(num_classes, activation='softmax')
])

# Compilar el modelo
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Entrenar el modelo
model.fit(x_train_padded, y_train_encoded, epochs=100, verbose=0)

# Guardar el modelo
model.save('modelo_entrenado.h5')

# Hacer predicciones para todos los datos de entrada
x_train_encoded = tokenizer.texts_to_sequences(x_train)
x_train_padded = pad_sequences(x_train_encoded, maxlen=max_seq_length)

predictions = model.predict(x_train_padded)

# Invertir el diccionario label_mapping
inverse_label_mapping = {v: k for k, v in label_mapping.items()}

# Decodificar las predicciones
decoded_predictions = [inverse_label_mapping[np.argmax(prediction)] for prediction in predictions]

# Crear una tabla para imprimir las predicciones
table = PrettyTable(['Entrada', 'Predicción de salida'])
for i in range(len(x_train)):
    table.add_row([x_train[i], decoded_predictions[i]])

# Imprimir la tabla de predicciones
print(table)

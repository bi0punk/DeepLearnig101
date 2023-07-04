import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Definir los datos de entrenamiento
x_train = np.array(['Hola', 'Hola, ¿cómo estás?', 'Buen día', 'Saludos'], dtype=object)
y_train = np.array(['Hola', 'Bien, ¿y tú?', 'Hola', 'Hola'], dtype=object)

# Convertir las etiquetas a valores numéricos
label_mapping = {'Hola': 0, 'Bien, ¿y tú?': 1}
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

# Obtener la entrada del usuario para hacer predicciones
user_input = input("Ingrese una oración: ")
x_test_encoded = tokenizer.texts_to_sequences([user_input])
x_test_padded = pad_sequences(x_test_encoded, maxlen=max_seq_length)

predictions = model.predict(x_test_padded)

# Invertir el diccionario label_mapping
inverse_label_mapping = {v: k for k, v in label_mapping.items()}

# Decodificar las predicciones
decoded_prediction = inverse_label_mapping[np.argmax(predictions)]

# Imprimir la predicción
print(f'Input: {user_input}, Predicted Output: {decoded_prediction}')

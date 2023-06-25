import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Definir los datos de entrenamiento
x_train = np.array(['Hola', 'Hola, ¿cómo estás?', 'Buen día', 'Saludos'], dtype=object)
y_train = np.array(['Hola', 'Bien, ¿y tú?', 'Hola', 'Hola'], dtype=object)

# Convertir las etiquetas a valores numéricos
label_mapping = {'Hola': 0, 'Bien, ¿y tú?': 1, 'Otra clase': 2}  # Agrega todas las clases posibles
y_train_encoded = np.array([label_mapping[label] for label in y_train])

# Tokenizar los datos de entrada
tokenizer = tf.keras.preprocessing.text.Tokenizer()
x_train_encoded = tokenizer.fit_on_texts(x_train)
x_train_encoded = tokenizer.texts_to_sequences(x_train)

# Padding de las secuencias
max_seq_length = len(max(x_train_encoded, key=len))
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

# Hacer predicciones
x_test = np.array(['Hola', 'Buenos días'])
x_test_encoded = tokenizer.texts_to_sequences(x_test)

# Filtrar las palabras que no tienen representación numérica
filtered_x_test = [x_test[i] for i, sequence in enumerate(x_test_encoded) if any(sequence)]
filtered_x_test_encoded = [sequence for sequence in x_test_encoded if any(sequence)]
filtered_x_test_padded = pad_sequences(filtered_x_test_encoded, maxlen=max_seq_length)
predictions = model.predict(filtered_x_test_padded)

# Decodificar las predicciones
decoded_predictions = [list(label_mapping.keys())[label] for label in np.argmax(predictions, axis=1)]

# Imprimir las predicciones
for input_text, prediction in zip(filtered_x_test, decoded_predictions):
    print(f'Input: {input_text}, Predicted Output: {prediction}')




    """ El método fit_on_texts ahora se llama antes de la conversión a secuencias para aprovechar la tokenización y la construcción del vocabulario al mismo tiempo.

    La variable max_seq_length se calcula utilizando len(max(x_train_encoded, key=len)) para obtener la longitud máxima de secuencia en el conjunto de entrenamiento directamente.

    En el bucle de impresión, se utiliza zip para iterar simultáneamente sobre filtered_x_test y decoded_predictions, evitando la necesidad de usar el índice i. """
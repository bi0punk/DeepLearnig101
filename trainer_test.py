import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Datos de entrenamiento
x_train = np.array(['Hola', 'Hola, ¿cómo estás?', 'Buen día'], dtype=object)
y_train = np.array(['Hola Señor', 'Hola', 'Hola'], dtype=object)

# Mapeo de etiquetas
label_mapping = {'Hola': 0, 'Hola Señor': 1, 'Otra clase': 2}  # Agrega todas las clases posibles
y_train_encoded = np.array([label_mapping[label] for label in y_train])

# Tokenización
tokenizer = tf.keras.preprocessing.text.Tokenizer()
tokenizer.fit_on_texts(x_train)
x_train_encoded = tokenizer.texts_to_sequences(x_train)
max_seq_length = len(max(x_train_encoded, key=len))
x_train_padded = pad_sequences(x_train_encoded, maxlen=max_seq_length)

# Definición del modelo
model = tf.keras.models.Sequential([
    tf.keras.layers.Embedding(len(tokenizer.word_index) + 1, 32, input_length=max_seq_length),
    tf.keras.layers.GlobalAveragePooling1D(),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(len(label_mapping), activation='softmax')
])
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(x_train_padded, y_train_encoded, epochs=100, verbose=0)

while True:
    # Datos de prueba
    print("")
    x_test = np.array([input("Ingrese el texto de prueba: ")])
    x_test_encoded = tokenizer.texts_to_sequences(x_test)
    x_test_padded = pad_sequences(x_test_encoded, maxlen=max_seq_length)

    # Predicciones
    predictions = model.predict(x_test_padded)
    decoded_predictions = [list(label_mapping.keys())[np.argmax(pred)] for pred in predictions]

    # Resultados
    for input_text, prediction in zip(x_test, decoded_predictions):
        print(f'Input: {input_text}, Predicted Output: {prediction}')

    choice = input("¿Deseas probar otro dato? (s/n): ")
    if choice.lower() != "s":
        break

import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
import speech_recognition as sr

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

# Especificar el nombre del archivo del modelo entrenado
model_filename = "modelo_entrenado.h5"

# Entrenar el modelo
model.fit(x_train_padded, y_train_encoded, epochs=100, verbose=0)

# Guardar el modelo entrenado
model.save(model_filename)

# Capturar el comando de voz
def capture_audio():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("Di algo...")
        audio = recognizer.listen(source)

    try:
        text = recognizer.recognize_google(audio, language='es')
        return text
    except sr.UnknownValueError:
        print("No se pudo reconocer el comando de voz.")
        return ""

# Cargar el modelo entrenado
model = tf.keras.models.load_model(model_filename)

# Capturar el comando de voz
command = capture_audio()

# Preprocesar el comando de voz
command_encoded = tokenizer.texts_to_sequences([command])
command_padded = pad_sequences(command_encoded, maxlen=max_seq_length)

# Realizar la predicción
prediction = model.predict(command_padded)[0]
predicted_label = label_mapping[np.argmax(prediction)]

# Imprimir la respuesta
print(f"Respuesta: {predicted_label}")

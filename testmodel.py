import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
import speech_recognition as sr

# Definir los datos de entrenamiento
x_train = np.array([
    "Hola",
    "Buenos días",
    "Buenas tardes",
    "Buenas noches",
    "¿Cómo estás?",
    "¿Qué tal?",
    "¿Qué pasa?",
    "¿Cómo va todo?",
    "¿Cómo te va?",
    "¿Qué hay de nuevo?",
    "Saludos",
    "¿Qué tal estás hoy?",
    "¿Cómo ha sido tu día?",
    "¡Hola, qué gusto verte!",
    "¿Cómo te sientes?",
    "¿Qué cuentas?",
    "¿Cómo andas?",
    "¡Hey!",
    "¡Hola, amigo!",
    "¡Buen día!",
    "¿Qué onda?",
    "¿Cómo ha estado todo?",
    "¿Qué tal tu semana?",
    "¿Cómo te ha tratado la vida?",
    "¡Hola, qué alegría verte!",
    "¿Qué tal la familia?",
    "¡Hola, cómo estás de ánimo!",
    "¿Qué hay de nuevo por aquí?",
    "¡Hola, buenos días!",
    "¿Cómo va el trabajo?",
    "¡Hola, cómo te ha ido!",
    "¿Qué tal tu día hasta ahora?",
    "¡Hola, qué tal todo!",
    "¿Cómo van las cosas?",
    "¿Qué ha sido de tu vida?",
    "¡Hola, cómo va eso!",
    "¿Cómo amaneciste hoy?",
    "¡Buenas tardes, amigo!",
    "¿Qué tal tu fin de semana?",
    "¡Hola, qué bueno verte!",
    "¿Cómo está la situación?",
    "¡Hola, cómo has estado!",
    "¿Qué novedades tienes?",
    "¡Hola, buenos días, buenos días!",
    "¿Cómo va tu salud?",
    "¡Hola, cómo te trata la vida!",
    "¿Qué tal en el trabajo?",
    "¡Hola, qué tal todo contigo!",
    "¿Cómo va la familia?",
    "¡Hola, cómo te ha ido últimamente!"], dtype=object)


y_train = np.array(["¡Hola! ¿Cómo puedo ayudarte hoy?",
    "Buenos días. ¿En qué puedo asistirte?",
    "Buenas tardes. ¿En qué puedo colaborar contigo?",
    "Buenas noches. ¿Necesitas algo en particular?",
    "¿Cómo estás? ¡Me alegra verte por aquí!",
    "¡Hola! ¿Qué novedades tienes para compartir?",
    "¿Qué pasa? ¿En qué puedo ser de ayuda?",
    "¿Cómo va todo? ¿Hay algo en lo que pueda colaborar?",
    "¿Cómo te va? Cuéntame, ¿en qué puedo ayudarte hoy?",
    "¿Qué hay de nuevo? Estoy aquí para ayudarte si necesitas algo.",
    "Saludos. ¿Cómo puedo servirte hoy?",
    "¿Qué tal estás hoy? ¿Necesitas alguna asistencia?",
    "¿Cómo ha sido tu día? ¿Hay algo en lo que pueda apoyarte?",
    "¡Hola, qué gusto verte! ¿En qué puedo colaborar contigo hoy?",
    "¿Cómo te sientes? Si necesitas algo, no dudes en decírmelo.",
    "¿Qué cuentas? ¿En qué puedo asistirte?",
    "¿Cómo andas? Si hay algo que necesites, no dudes en decírmelo.",
    "¡Hey! ¿En qué puedo ayudarte hoy?",
    "¡Hola, amigo! ¿En qué puedo colaborar contigo?",
    "¡Buen día! ¿Cómo puedo asistirte hoy?",
    "¿Qué onda? ¿En qué puedo colaborar contigo?",
    "¿Cómo ha estado todo? ¿En qué puedo ayudarte?",
    "¿Qué tal tu semana? ¿Hay algo en lo que pueda apoyarte?",
    "¿Cómo te ha tratado la vida? Si necesitas algo, no dudes en decírmelo.",
    "¡Hola, qué alegría verte! ¿En qué puedo colaborar contigo hoy?",
    "¿Qué tal la familia? ¿Necesitas alguna asistencia?",
    "¡Hola, cómo estás de ánimo! Si necesitas algo, estoy aquí para ayudarte.",
    "¿Qué hay de nuevo por aquí? ¿En qué puedo asistirte?",
    "¡Hola, buenos días! ¿En qué puedo colaborar contigo hoy?",
    "¿Cómo va el trabajo? Si necesitas alguna ayuda laboral, estoy aquí para ti.",
    "¡Hola, cómo te ha ido! ¿En qué puedo colaborar contigo hoy?",
    "¿Qué tal tu día hasta ahora? Si necesitas algo, no dudes en decírmelo.",
    "¡Hola, qué tal todo! ¿En qué puedo ayudarte hoy?",
    "¿Cómo van las cosas? Si necesitas alguna asistencia, estoy aquí para ayudarte.",
    "¿Qué ha sido de tu vida? ¿En qué puedo colaborar contigo?",
    "¡Hola, cómo va eso! ¿En qué puedo asistirte?",
    "¿Cómo amaneciste hoy? Si necesitas algo, no dudes en decírmelo.",
    "¡Buenas tardes, amigo! ¿En qué puedo colaborar contigo?",
    "¿Qué tal tu fin de semana? ¿Hay algo en lo que pueda apoyarte?",
    "¡Hola, qué bueno verte! ¿En qué puedo asistirte hoy?",
    "¿Cómo está la situación? ¿En qué puedo colaborar contigo?",
    "¡Hola, cómo has estado! ¿En qué puedo asistirte hoy?",
    "¿Qué novedades tienes? Si necesitas algo, estoy aquí para ayudarte.",
    "¡Hola, buenos días, buenos días! ¿En qué puedo colaborar contigo hoy?",
    "¿Cómo va tu salud? Si necesitas alguna asistencia médica, no dudes en decírmelo.",
    "¡Hola, cómo te trata la vida! ¿En qué puedo asistirte hoy?",
    "¿Qué tal en el trabajo? Si necesitas alguna ayuda laboral, estoy aquí para ti.",
    "¡Hola, qué tal todo contigo! ¿En qué puedo colaborar hoy?",
    "¿Cómo va la familia? Si necesitas algo relacionado con tu familia, estoy aquí para ayudarte.",
    "¡Hola, cómo te ha ido últimamente! ¿En qué puedo asistirte hoy?"], dtype=object)

# Convertir las etiquetas a valores numéricos
label_mapping = {'Hola': 0, 'Bien, ¿y tú?': 1}
inverse_label_mapping = {v: k for k, v in label_mapping.items()}
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
        print(f"Texto capturado: {text}")
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
predictions = model.predict(command_padded)
predicted_label = inverse_label_mapping[np.argmax(predictions[0])]

# Imprimir la respuesta
print(f"Respuesta: {predicted_label}")

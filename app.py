from testmodel import *


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

print(f"Respuesta: {predicted_label}")

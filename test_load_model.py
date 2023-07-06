import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences

model = tf.keras.models.load_model('modelo_entrenado.h5')

num_inputs = int(input("Ingrese el número de entradas: "))
x_test = []
for i in range(num_inputs):
    input_text = input(f"Ingrese la entrada {i+1}: ")
    x_test.append(input_text)

tokenizer = tf.keras.preprocessing.text.Tokenizer()
tokenizer.fit_on_texts(x_test)
x_test_encoded = tokenizer.texts_to_sequences(x_test)

max_seq_length = max(len(seq) for seq in x_test_encoded)
x_test_padded = pad_sequences(x_test_encoded, maxlen=max_seq_length)

predictions = model.predict(x_test_padded)

label_mapping = {0: 'Hola', 1: 'Bien, ¿y tú?'}
decoded_predictions = [label_mapping[np.argmax(prediction)] for prediction in predictions]

for i in range(len(x_test)):
    print(f'Input: {x_test[i]}, Predicted Output: {decoded_predictions[i]}')

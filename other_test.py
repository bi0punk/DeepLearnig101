from gtts import gTTS
import os

texto = input("Ingrese el texto: ")
tts = gTTS(text=texto, lang='es')
archivo_mp3 = "audio.mp3"
tts.save(archivo_mp3)
os.system(archivo_mp3)

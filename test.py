from gtts import gTTS
import os
from playsound import playsound

text = "This is Google text to speech. Your voice system works!"
tts = gTTS(text=text, lang='en')
tts.save("voice.mp3")
playsound("voice.mp3")

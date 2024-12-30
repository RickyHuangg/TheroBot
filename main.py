
from flask import jsonify
import pickle
from keras.models import load_model
import nltk
nltk.download('punkt')
nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize 
lemmatizer = WordNetLemmatizer()
import json
import pickle
import numpy as np
import random
import pyaudio
from vosk import Model, KaldiRecognizer

FORMAT = pyaudio.paInt16
CHANNEL = 1
RATE = 8192
CHUNK = 1024
output_file = "recorded.wav"

model = Model(r"C:\Users\Rhrua\Downloads\New folder\vosk-model-small-en-us-0.15") # 
recog = KaldiRecognizer(model,16000)


with open('Dataset\intent.json') as file:
  data = json.load(file)

with open('words.pickle', 'rb') as f:
    words = pickle.load(f)

with open('labels.pickle', 'rb') as f:
    labels = pickle.load(f)


model=load_model('chatModelV2.h5')

def process_audio():
    try:
      audio = pyaudio.PyAudio()
      stream = audio.open(rate = 16000 ,channels = CHANNEL, format = FORMAT, frames_per_buffer=CHUNK,input = True ) 
      stream.start_stream()  
      while True:
          data = stream.read(4096)  
          if recog.AcceptWaveform(data):  
              text = recog.Result()
              text = json.loads(text) 
              return text['text']  

    finally:
      stream.stop_stream()
      stream.close()
      audio.terminate()

def get_input (inp):
   inpSpl = word_tokenize(inp)
   inpSpl = [lemmatizer.lemmatize(word.lower()) for word in inpSpl if word.isalnum()]
   return inpSpl

def bag_input(input, words):
   sorted_input = get_input(input)
   store = [0] * len(words)
   for word in sorted_input:
      for w in words:
         if word == w:
            store[words.index(w)] = 1
   return store
      
def predict_ans (input):
   pred = model.predict(np.array([bag_input(input, words)]))
   pred_index = np.argmax(pred)  
   return pred_index

def get_response(predicted_class_index):
    tag = labels[predicted_class_index]
    for intent in data['intents']:  
        if intent['tag'] == tag:
            response = random.choice(intent['responses'])
            break
    return response

from flask import Flask, render_template, request
app = Flask(__name__)
@app.route("/")
def home():
  return render_template("index.html")
@app.route("/get")
def get_bot_response():    
    userText = request.args.get('msg')  
    response_index = predict_ans(userText)  
    response = get_response(response_index)
    return response
@app.route("/tts")
def speech_input():
    text = process_audio()  
    return text 
if __name__ == "__main__":
  app.run()   
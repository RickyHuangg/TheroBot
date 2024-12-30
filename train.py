import nltk
nltk.download('punkt')
nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize 
lemmatizer = WordNetLemmatizer()
import json
import pickle

import numpy as np
from keras.optimizers.schedules import ExponentialDecay
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.optimizers import SGD
import random


with open('Dataset\intent.json') as file:
  data = json.load(file)

words = []
groups = []
labels = []

for intent in data['intents']:
  for pattern in intent['patterns']:
    
    word = word_tokenize(pattern)
    words.extend(word)
    groups.append((word,intent['tag']))

    if intent['tag'] not in labels:
        labels.append(intent['tag'])


words = [lemmatizer.lemmatize(word.lower()) for word in words if word.isalnum() ]
words = sorted(list(set(words)))

labels = sorted(list(set(labels)))

with open("words.pickle", "wb") as f:
    pickle.dump(words, f)

with open("labels.pickle", "wb") as f:
    pickle.dump(labels, f)

training = []
output = [0] * len(labels)

for group in groups:
    bag = []
    bag_check = group[0]
    checked = []
    for word in bag_check:
      checked_word = lemmatizer.lemmatize(word.lower())
      checked.append(checked_word)
    bag_check = checked
   
    for w in words:
      if w in bag_check:
         bag.append(1)
      else:
         bag.append(0)
   
    output_row = list(output)
    output_row[labels.index(group[1])] = 1
   
    training.append([bag,output_row])

random.shuffle(training)
training = np.array(training, dtype=object)

train_x = list(training[:,0])
train_y = list(training[:,1])
   
model = Sequential()
model.add(Dense(256, input_shape=(len(train_x[0]),), activation='relu')) 
model.add(Dropout(0.5))
model.add(Dense(128, activation='relu')) 
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))   
model.add(Dropout(0.5))
model.add(Dense(len(labels), activation='softmax'))


lr_schedule = ExponentialDecay(
    initial_learning_rate=0.01,
    decay_steps=10000,
    decay_rate=0.96,
    staircase=True
)


model.compile(optimizer=SGD(learning_rate=lr_schedule), loss='categorical_crossentropy', metrics=['accuracy'])


model.fit(np.array(train_x), np.array(train_y), epochs=300, batch_size=5, verbose=1)
model.save('chatModelV2.h5')


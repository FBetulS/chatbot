import streamlit as st
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
import json
import random
import pickle
from tensorflow.keras.models import load_model

import os
os.environ['NLTK_DATA'] = os.path.join(os.getcwd(), 'nltk_data')

nltk.download('punkt', download_dir='nltk_data')
nltk.download('wordnet', download_dir='nltk_data')
nltk.download('perluniprops', download_dir='nltk_data')
nltk.download('nonbreaking_prefixes', download_dir='nltk_data')

# NLTK verilerini indir
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')

# Model ve verileri yükle
@st.cache_resource
def load_resources():
    model = load_model('chatbot_model.h5')
    with open('intents.json.txt.txt', 'r', encoding='utf-8') as file:
        intents = json.load(file)
    with open('words_classes.pkl', 'rb') as f:
        words, classes = pickle.load(f)
    return model, intents, words, classes

model, intents, words, classes = load_resources()
lemmatizer = WordNetLemmatizer()

# Chat fonksiyonları
def clean_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    return [lemmatizer.lemmatize(word.lower()) for word in sentence_words]

def bag_of_words(sentence):
    sentence_words = clean_sentence(sentence)
    bag = [0] * len(words)
    for w in sentence_words:
        for i, word in enumerate(words):
            if word == w:
                bag[i] = 1
    return np.array(bag)

def predict_class(sentence):
    bow = bag_of_words(sentence)
    res = model.predict(np.array([bow]))[0]
    results = [[i, r] for i, r in enumerate(res) if r > 0.25]
    results.sort(key=lambda x: x[1], reverse=True)
    return [{'intent': classes[r[0]], 'probability': str(r[1])} for r in results]

def get_response(intents_list):
    tag = intents_list[0]['intent']
    for intent in intents['intents']:
        if intent['tag'] == tag:
            return random.choice(intent['responses'])
    return "I didn't understand that, could you rephrase?"

# Streamlit Arayüzü
st.title('🤖 English Chatbot')
st.caption("Type 'quit' to exit")

if 'history' not in st.session_state:
    st.session_state.history = []

user_input = st.chat_input("Your message...")

if user_input:
    if user_input.lower() == 'quit':
        st.stop()
    
    st.session_state.history.append(('user', user_input))
    
    intents_list = predict_class(user_input)
    response = get_response(intents_list)
    st.session_state.history.append(('bot', response))

for role, message in st.session_state.history:
    with st.chat_message(role):
        st.write(message)
import json
import numpy as np
import pickle as pk
import tensorflow as tf # type: ignore
from nltk.stem import WordNetLemmatizer
import nltk
import string
import random

import pickle as pk
from sklearn.feature_extraction.text import CountVectorizer

# Example vectorizer (you'll replace this with your actual vectorizer)
vectorizer = CountVectorizer()

# Save the vectorizer
#with open('vectorizer.pkl', 'wb') as f:
   # pk.dump(vectorizer, f)
# Example vocabulary list



# Ensure that NLTK resources are available
nltk.download("punkt")
nltk.download("wordnet")

# Load intents data
with open('intents.json','r',encoding='utf-8') as file:
    intents = json.load(file)

# Load model and vectorizer
model = tf.keras.models.load_model('model.h5')  # Load the neural network model
with open('vectorizer.pkl', 'rb') as f:
    vectorizer = pk.load(f)

lemmatizer = WordNetLemmatizer()

# Load vocabulary and classes
with open('vocab.pkl', 'rb') as f:
    words = pk.load(f)
with open('classes.pkl', 'rb') as f:
    classes = pk.load(f)

def preprocess_input(text):
    """ Transform the input text into features using the vectorizer. """
    return vectorizer.transform([text])

def clean_text(text):
    tokens = nltk.word_tokenize(text)
    tokens = (lemmatizer.lemmatize(word.lower()) for word in tokens)
    return tokens

def bag_of_words(text, vocab):
    tokens = clean_text(text)
    bow = [0] * len(vocab)
    for w in tokens:
        for idx, word in enumerate(vocab):
            if word == w:
                bow[idx] = 1
    return np.array(bow)

def pred_class(text, vocab, labels):
    bow = bag_of_words(text, vocab)
    result = model.predict(np.array([bow]))[0]  # Extract probabilities

    thresh = 0.5
    y_pred = [[idx, res] for idx, res in enumerate(result) if res > thresh]
    y_pred.sort(key=lambda x: x[1], reverse=True)
    
    return_list = [labels[r[0]] for r in y_pred]
    return return_list

def get_response(intents_list, intents_json):
    if len(intents_list) == 0:
        return "Sorry! I didn't understand."
    tag = intents_list[0]
    for i in intents_json["intents"]:
        if i["tag"] == tag:
            return random.choice(i["responses"])
    return "Sorry! I didn't understand."
    

import random
import json
import numpy as np
import nltk
import pickle
import urllib.parse  # To create Google search links
from nltk.stem import WordNetLemmatizer
import tensorflow as tf

lemmatizer = WordNetLemmatizer()

# Load intents, words, and classes
intents = json.loads(open('intents.json').read())
words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))
model = tf.keras.models.load_model('chatbot_model.h5')

def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word) for word in sentence_words]
    return sentence_words

def bag_of_words(sentence):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for w in sentence_words:
        for i, word in enumerate(words):
            if word == w:
                bag[i] = 1
    return np.array(bag)

def predict_class(sentence):
    bow = bag_of_words(sentence)
    print(f"Input shape for model: {bow.shape}")  # Debug statement
    res = model.predict(np.array([bow]))[0]
    ERROR_THRESHOLD = 0.6
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({'intent': classes[r[0]], 'probability': str(r[1])})
    return return_list

def get_response(intents_list, intents_json, query):
    if not intents_list:
        # If no intent is recognized, log and offer a search link
        log_unhandled_query(query)
        google_search_link = generate_google_search_link(query)
        return f"Sorry, I couldn't find an answer to your question. You can search for more information here: {google_search_link}"

    tag = intents_list[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if i['tag'] == tag:
            result = random.choice(i['responses'])
            break
    return result

def log_unhandled_query(query):
    """Logs unhandled queries to a file."""
    with open('unhandled_queries.txt', 'a') as file:
        file.write(f"{query}\n")

def generate_google_search_link(query):
    """Generates a Google search URL based on the query."""
    base_url = "https://www.google.com/search?q="
    search_query = urllib.parse.quote(query)
    return base_url + search_query

# Chatbot Interaction
print("GO! Bot is running!")

while True:
    message = input("")
    ints = predict_class(message)
    res = get_response(ints, intents, message)
    print(f"Bot: {res}")

    if message == 'quit':
        break
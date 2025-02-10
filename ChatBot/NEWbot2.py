import sqlite3
from flask import Flask, request, jsonify
import json
import pickle
import tensorflow as tf
import nltk
from nltk.stem import WordNetLemmatizer
import random
import numpy as np

lemmatizer = WordNetLemmatizer()

# Load intents and model
intents = json.loads(open('intents.json').read())
words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))
model = tf.keras.models.load_model('chatbot_model.h5')

app = Flask(__name__)

# Database connection function
def connect_db():
    return sqlite3.connect('chatbot_data.db')

# Clean up and process user input
def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
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
        return "Sorry, I don't understand. Please rephrase or ask another question."

    tag = intents_list[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if i['tag'] == tag:
            return random.choice(i['responses'])

# Log conversation to the database
def log_conversation(user_id, query, response):
    conn = connect_db()
    cursor = conn.cursor()
    cursor.execute("INSERT INTO conversations (user_id, query, response) VALUES (?, ?, ?)", 
                   (user_id, query, response))
    conn.commit()
    conn.close()

@app.route('/get_bot_response', methods=['POST'])
def get_bot_response():
    user_id = request.json.get('user_id', 'anonymous')
    user_input = request.json['message']

    # Predict intent and get response
    intents_list = predict_class(user_input)
    bot_response = get_response(intents_list, intents, user_input)

    # Log the conversation
    log_conversation(user_id, user_input, bot_response)

    return jsonify({'response': bot_response})

@app.route('/add_new_intent', methods=['POST'])
def add_new_intent():
    data = request.json
    tag = data['tag']
    patterns = data['patterns']
    responses = data['responses']

    # Update intents.json
    intents['intents'].append({
        'tag': tag,
        'patterns': patterns,
        'responses': responses,
        'context': ["none"]
    })
    with open('intents.json', 'w') as f:
        json.dump(intents, f, indent=4)

    return jsonify({"message": "Intent added successfully!"})

if __name__ == '__main__':
    app.run(debug=True)

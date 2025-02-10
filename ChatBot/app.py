# from flask import Flask, render_template, request, redirect, url_for, session
# import sqlite3

# app = Flask(__name__)
# app.secret_key = 'your_secret_key'  # Set a secret key for session

# def check_user_credentials(username, password, role):
#     # Connect to the database
#     conn = sqlite3.connect('user_data.db')
#     cursor = conn.cursor()
#     cursor.execute("SELECT * FROM users WHERE username = ? AND password = ? AND role = ?", (username, password, role))
#     user = cursor.fetchone()
#     conn.close()
#     return user

# @app.route('/login', methods=['GET', 'POST'])
# def login():
#     if request.method == 'POST':
#         username = request.form['username']
#         password = request.form['password']
#         role = request.form['role']

#         user = check_user_credentials(username, password, role)
#         if user:
#             session['user'] = user
#             if role == 'student':
#                 return redirect(url_for('student_dashboard'))
#             elif role == 'canteen_incharge':
#                 return redirect(url_for('canteen_dashboard'))
#             elif role == 'mess_incharge':
#                 return redirect(url_for('mess_dashboard'))
#         else:
#             return "Invalid credentials, please try again."

#     return render_template('login.html')

# @app.route('/student_dashboard')
# def student_dashboard():
#     return render_template('student_dashboard.html')

# @app.route('/canteen_dashboard')
# def canteen_dashboard():
#     return render_template('canteen_dashboard.html')

# @app.route('/mess_dashboard')
# def mess_dashboard():
#     return render_template('mess_dashboard.html')

# if __name__ == '__main__':
#     app.run(debug=True)

# from flask import Flask, request, jsonify
# import json

# app = Flask(__name__)

# # Load intents and model
# intents = json.loads(open('intents.json').read())
# words = pickle.load(open('words.pkl', 'rb'))
# classes = pickle.load(open('classes.pkl', 'rb'))
# model = tf.keras.models.load_model('chatbot_model.h5')

# @app.route('/get_bot_response', methods=['POST'])
# def get_bot_response():
#     user_input = request.json['message']
#     ints = predict_class(user_input)
#     res = get_response(ints, intents, user_input)
#     return jsonify({'response': res})

# def predict_class(sentence):
#     # Your model prediction function here (similar to previous implementation)
#     # ...
#     return response

# def get_response(intents_list, intents_json, query):
#     # Your response generation function here
#     # ...
#     return response

# if __name__ == '__main__':
#     app.run(debug=True)

from flask import Flask, render_template, request, redirect, url_for, session, jsonify
import sqlite3
import json
import pickle
import tensorflow as tf
import nltk as np
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
import urllib.parse
import random


lemmatizer = WordNetLemmatizer()

app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Set a secret key for session

# Load intents and model
intents = json.loads(open('intents.json').read())
words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))
model = tf.keras.models.load_model('chatbot_model.h5')

def check_user_credentials(username, password, role):
    # Connect to the database
    conn = sqlite3.connect('user_data.db')
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM users WHERE username = ? AND password = ? AND role = ?", (username, password, role))
    user = cursor.fetchone()
    conn.close()
    return user

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


# Login Routes
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        role = request.form['role']

        user = check_user_credentials(username, password, role)
        if user:
            session['user'] = user
            if role == 'student':
                return redirect(url_for('student_dashboard'))
            elif role == 'canteen_incharge':
                return redirect(url_for('canteen_dashboard'))
            elif role == 'mess_incharge':
                return redirect(url_for('mess_dashboard'))
        else:
            return "Invalid credentials, please try again."

    return render_template('login.html')


@app.route('/student_dashboard')
def student_dashboard():
    return render_template('student_dashboard.html')


@app.route('/canteen_dashboard')
def canteen_dashboard():
    return render_template('canteen_dashboard.html')


@app.route('/mess_dashboard')
def mess_dashboard():
    return render_template('mess_dashboard.html')


# Chatbot Response API
@app.route('/get_bot_response', methods=['POST'])
def get_bot_response():
    user_input = request.json['message']
    ints = predict_class(user_input)
    res = get_response(ints, intents, user_input)
    return jsonify({'response': res})


if __name__ == '__main__':
    app.run(debug=True)

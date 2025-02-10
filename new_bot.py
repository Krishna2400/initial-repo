import random
import json
import pickle
import numpy as np
import nltk

from nltk.stem import WordNetLemmatizer
from keras.models import load_model
from tkinter import *

# Initialization
nltk.download('punkt')
lemmatizer = WordNetLemmatizer()

# Load necessary data
intents = json.loads(open('intents.json').read())
words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))
model = load_model('chatbot_model.h5')

# Context dictionary to track conversation context
context = {}

# Helper Functions
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
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]

    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({'intent': classes[r[0]], 'probability': str(round(r[1] * 100, 2)) + "%"})
    return return_list

def get_response(intents_list, intents_json, user_id):
    if not intents_list:
        return "I'm sorry, I didn't quite understand that. Could you rephrase?"

    tag = intents_list[0]['intent']
    confidence = intents_list[0]['probability']
    
    if tag == 'goodbye':
        if user_id in context:
            del context[user_id]
        return "Goodbye! Have a great day!"
    
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if i['tag'] == tag:
            result = random.choice(i['responses'])
            if 'context' in i:
                context[user_id] = i['context']
            break
    
    return f"{result} (Confidence: {confidence})"

# GUI Implementation
def send_message():
    msg = entry_box.get("1.0", 'end-1c').strip()
    entry_box.delete("0.0", END)
    
    if msg:
        chat_box.config(state=NORMAL)
        chat_box.insert(END, f"You: {msg}\n")
        chat_box.config(foreground="#000000", font=("Verdana", 12))
        
        if msg.lower() in ['quit', 'exit', 'bye']:
            chat_box.insert(END, "Bot: Goodbye! Take care!\n")
            chat_box.config(state=DISABLED)
            return
        
        intents_list = predict_class(msg)
        response = get_response(intents_list, intents, "user")
        chat_box.insert(END, f"Bot: {response}\n")
        chat_box.config(state=DISABLED)
        chat_box.yview(END)

# GUI Setup
root = Tk()
root.title("Chatbot")
root.geometry("500x600")
root.resizable(width=False, height=False)

# Chat Window
chat_box = Text(root, bd=0, bg="White", height="8", width="50",
                font=("Arial", 12))
chat_box.config(state=DISABLED)

# Scrollbar
scrollbar = Scrollbar(root, command=chat_box.yview, cursor="heart")
chat_box['yscrollcommand'] = scrollbar.set

# Send Button
send_button = Button(root, font=("Verdana", 12, 'bold'), text="Send", width="12", height=5,
                     bd=0, bg="#f9a602", activebackground="#3c9d9b", fg='#000000',
                     command=send_message)

# Entry Box
entry_box = Text(root, bd=0, bg="white", width="29", height="5", font=("Arial", 12))

# Layout
scrollbar.place(x=476, y=6, height=486)
chat_box.place(x=6, y=6, height=486, width=470)
entry_box.place(x=6, y=501, height=90, width=365)
send_button.place(x=376, y=501, height=90)

root.mainloop()

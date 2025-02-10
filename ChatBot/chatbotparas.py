import random
import json
import pickle
import numpy as np
import nltk
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
from transformers import GPT2Tokenizer, TFGPT2ForSequenceClassification
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf

# Initialize lemmatizer
lemmatizer = WordNetLemmatizer()

# Load intents JSON file
intents = json.loads(open('intents.json').read())

# Initialize GPT-2 tokenizer from openai-community/gpt2
tokenizer = GPT2Tokenizer.from_pretrained('openai-community/gpt2')
tokenizer.pad_token = tokenizer.eos_token  # GPT-2 doesn't have an official padding token

# Data holders
words = []
classes = []
documents = []
ignore_letters = ['?', '!', '.', ',']

# Data augmentation functions (same as before)
def synonym_replacement(words, n=3):
    # Same as before
    pass

def random_insertion(words, n=2):
    # Same as before
    pass

def random_deletion(words, p=0.3):
    # Same as before
    pass

def random_swap(words, n=2):
    # Same as before
    pass

# Process intents data and augment it
for intent in intents['intents']:
    for pattern in intent['patterns']:
        word_list = nltk.word_tokenize(pattern)
        lemmatized_words = [lemmatizer.lemmatize(word.lower()) for word in word_list]
        documents.append((lemmatized_words, intent['tag']))

        # Augmentation
        augmented_patterns = [
            synonym_replacement(lemmatized_words),
            random_insertion(lemmatized_words),
            random_deletion(lemmatized_words),
            random_swap(lemmatized_words)
        ]
        
        for aug_pattern in augmented_patterns:
            documents.append((aug_pattern, intent['tag']))

        if intent['tag'] not in classes:
            classes.append(intent['tag'])

# Encode the class labels
label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform([doc[1] for doc in documents])

# Tokenize the patterns using GPT-2 tokenizer
input_ids = []
attention_masks = []

for doc in documents:
    pattern_text = ' '.join(doc[0])  # Rebuild sentence from tokenized words
    gpt_input = tokenizer(pattern_text, padding='max_length', max_length=50, truncation=True, return_tensors='tf')
    input_ids.append(gpt_input['input_ids'])
    attention_masks.append(gpt_input['attention_mask'])

# Convert lists to arrays
input_ids = np.array(input_ids).squeeze()
attention_masks = np.array(attention_masks).squeeze()
encoded_labels = np.array(encoded_labels)

# GPT-2 model setup for sequence classification using openai-community/gpt2
model = TFGPT2ForSequenceClassification.from_pretrained('openai-community/gpt2', num_labels=len(classes))

# Compile the model
optimizer = tf.keras.optimizers.Adam(learning_rate=5e-5, epsilon=1e-08)
loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])

# Train the model
history = model.fit(
    [input_ids, attention_masks],
    encoded_labels,
    epochs=3,
    batch_size=16
)

# Save the trained model and label encoder
model.save_pretrained('gpt2_chatbot_model')
with open('label_encoder.pkl', 'wb') as f:
    pickle.dump(label_encoder, f)

print("Model training complete and saved.")

# Optional: Plot training history
import matplotlib.pyplot as plt

# Plot accuracy
plt.plot(history.history['accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.show()

# Plot loss
plt.plot(history.history['loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.show()

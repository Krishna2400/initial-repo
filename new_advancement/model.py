import random
import json
import pickle
import numpy as np
import nltk
from nltk.corpus import wordnet, stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping

# Initialization
# nltk.download('punkt')
# nltk.download('wordnet')
# nltk.download('stopwords')
lemmatizer = WordNetLemmatizer()

# Load intents file
intents = json.loads(open('intents.json').read())

# Augmentation function to generate synonyms
def augment_text(pattern):
    words = nltk.word_tokenize(pattern)
    augmented_patterns = []
    for word in words:
        synonyms = wordnet.synsets(word)
        synonym_words = [syn.lemmas()[0].name() for syn in synonyms if syn.lemmas()]
        if synonym_words:
            augmented_patterns.append(random.choice(synonym_words))
        else:
            augmented_patterns.append(word)
    return " ".join(augmented_patterns)

# Initialize data storage
words = []
classes = []
documents = []
ignore_letters = ['?', '!', '.', ',']
stop_words = set(stopwords.words('english'))

# Process and augment intents
for intent in intents['intents']:
    for pattern in intent['patterns']:
        # Tokenize the pattern and remove stopwords
        word_list = nltk.word_tokenize(pattern)
        word_list = [word for word in word_list if word.lower() not in stop_words]
        words.extend(word_list)
        documents.append((word_list, intent['tag']))
        
        # Augmented patterns
        for _ in range(2):  # Generate 2 augmented patterns for each original pattern
            augmented_pattern = augment_text(pattern)
            augmented_word_list = nltk.word_tokenize(augmented_pattern)
            augmented_word_list = [word for word in augmented_word_list if word.lower() not in stop_words]
            words.extend(augmented_word_list)
            documents.append((augmented_word_list, intent['tag']))
        
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

# Lemmatize and clean up words and classes
words = [lemmatizer.lemmatize(word.lower()) for word in words if word not in ignore_letters]
words = sorted(set(words))
classes = sorted(set(classes))

# Save words and classes
pickle.dump(words, open('words.pkl', 'wb'))
pickle.dump(classes, open('classes.pkl', 'wb'))

# Create training data
training_sentences = []
training_labels = []
output_empty = [0] * len(classes)

for document in documents:
    # Bag of words
    bag = []
    word_patterns = document[0]
    word_patterns = [lemmatizer.lemmatize(word.lower()) for word in word_patterns]
    for word in words:
        bag.append(1) if word in word_patterns else bag.append(0)

    # One-hot encoding for the class
    output_row = list(output_empty)
    output_row[classes.index(document[1])] = 1
    training_sentences.append(bag)
    training_labels.append(output_row)

# Shuffle and convert to NumPy array
training_sentences = np.array(training_sentences)
training_labels = np.array(training_labels)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(training_sentences, training_labels, test_size=0.2)

# Define the model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(256, input_shape=(len(X_train[0]),), activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(len(classes), activation='softmax')
])

# Compile the model
sgd = tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9, nesterov=True, decay=1e-6)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

# Set EarlyStopping callback
early_stopping = EarlyStopping(monitor='accuracy', patience=10, restore_best_weights=True)

# Train the model
hist = model.fit(X_train, y_train, epochs=200, batch_size=5, validation_data=(X_test, y_test), callbacks=[early_stopping], verbose=1)

# Save the trained model
model.save('chatbot_model.h5')

# Save training history
pickle.dump(hist.history, open("training_history.pkl", "wb"))

print("Model training complete")

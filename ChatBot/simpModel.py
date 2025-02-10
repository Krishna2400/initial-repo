import random
import json
import pickle
import numpy as np
import tensorflow as tf
import nltk
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer

nltk.download('punkt')
nltk.download('wordnet')
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
ignoreLetters = ['?', '!', '.', ',']

# Augment and process intents
for intent in intents['intents']:
    for pattern in intent['patterns']:
        # Original pattern
        wordList = nltk.word_tokenize(pattern)
        words.extend(wordList)
        documents.append((wordList, intent['tag']))
        
        # Augmented patterns
        for _ in range(2):  # Generate 2 augmented patterns for each original pattern
            augmented_pattern = augment_text(pattern)
            augmented_wordList = nltk.word_tokenize(augmented_pattern)
            words.extend(augmented_wordList)
            documents.append((augmented_wordList, intent['tag']))
        
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

# Process words and classes
words = [lemmatizer.lemmatize(word.lower()) for word in words if word not in ignoreLetters]
words = sorted(set(words))

classes = sorted(set(classes))

# Save words and classes
pickle.dump(words, open('words.pkl', 'wb'))
pickle.dump(classes, open('classes.pkl', 'wb'))

# Create training data
training = []
outputEmpty = [0] * len(classes)

for document in documents:
    bag = []
    wordPatterns = document[0]
    wordPatterns = [lemmatizer.lemmatize(word.lower()) for word in wordPatterns]
    for word in words:
        bag.append(1) if word in wordPatterns else bag.append(0)

    outputRow = list(outputEmpty)
    outputRow[classes.index(document[1])] = 1
    training.append(bag + outputRow)

# Shuffle and convert to NumPy array
random.shuffle(training)
training = np.array(training)

trainX = training[:, :len(words)]
trainY = training[:, len(words):]

# Define and train the model
model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(128, input_shape=(len(trainX[0]),), activation='relu'))
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Dense(64, activation='relu'))
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Dense(len(trainY[0]), activation='softmax'))

sgd = tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

hist = model.fit(np.array(trainX), np.array(trainY), epochs=200, batch_size=5, verbose=1)
model.save('chatbot_model.h5', hist)

print('Model training complete with data augmentation.')

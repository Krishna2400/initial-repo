import random
import json
import pickle
import numpy as np
import tensorflow as tf
import nltk
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer

# nltk.download('punkt')
# nltk.download('wordnet')
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

# Process and augment intents
for intent in intents['intents']:
    for pattern in intent['patterns']:
        # Tokenize the pattern
        word_list = nltk.word_tokenize(pattern)
        words.extend(word_list)
        documents.append((word_list, intent['tag']))
        
        # Augmented patterns
        for _ in range(2):  # Generate 2 augmented patterns for each original pattern
            augmented_pattern = augment_text(pattern)
            augmented_word_list = nltk.word_tokenize(augmented_pattern)
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
training = []
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
    training.append([bag, output_row])

# Shuffle and convert to NumPy array
random.shuffle(training)
training = np.array(training, dtype=object)

# Create training and testing datasets
train_x = np.array(list(training[:, 0]))
train_y = np.array(list(training[:, 1]))

print(f"Training data: {len(train_x)} samples")

# Define the model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(128, input_shape=(len(train_x[0]),), activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(len(train_y[0]), activation='softmax')
])

# Compile the model
sgd = tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9, nesterov=True, decay=1e-6)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

# Train the model
hist = model.fit(train_x, train_y, epochs=200, batch_size=5, verbose=1)

# Save the trained model
model.save('chatbot_model.h5', hist)

print("Model training complete")

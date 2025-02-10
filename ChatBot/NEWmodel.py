import random
import json
import numpy as np
import tensorflow as tf
import nltk
import pickle
import matplotlib.pyplot as plt
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet

# Uncomment the following lines if you haven't downloaded these resources
# nltk.download('wordnet')
# nltk.download('punkt')

lemmatizer = WordNetLemmatizer()

# Load the dataset
with open('intents.json', 'r') as file:
    dataset = json.load(file)

words = []
classes = []  # Define possible intent tags
documents = []
ignoreLetters = ['?', '!', '.', ',']

# Synonym Replacement for Augmentation
def get_synonyms(word):
    synonyms = set()
    for syn in wordnet.synsets(word):
        for lemma in syn.lemmas():
            synonyms.add(lemma.name())
    return list(synonyms)

# Augmentation Functions
def augment_sentence(sentence):
    words = sentence.split()
    # Random synonym replacement
    aug_sentence = []
    for word in words:
        if random.random() < 0.3:  # Replace with synonym 30% of the time
            synonyms = get_synonyms(word)
            if synonyms:
                aug_sentence.append(random.choice(synonyms))
            else:
                aug_sentence.append(word)
        else:
            aug_sentence.append(word)
    return ' '.join(aug_sentence)

def augment_data(text, times=2):
    augmented_texts = [text]
    for _ in range(times):
        augmented_texts.append(augment_sentence(text))
    return augmented_texts

# Extract patterns and intents from dataset
for intent in dataset['intents']:
    tag = intent['tag']
    if tag not in classes:
        classes.append(tag)
    
    for pattern in intent['patterns']:
        # Tokenize and lemmatize pattern
        wordList = nltk.word_tokenize(pattern)
        words.extend(wordList)
        documents.append((wordList, tag))  # Append pattern with intent tag
        
        # Augment sentences and tokenize those as well
        augmented_texts = augment_data(pattern, times=2)  # Generate 2 augmented versions
        for augmented_text in augmented_texts:
            augmented_wordList = nltk.word_tokenize(augmented_text)
            words.extend(augmented_wordList)
            documents.append((augmented_wordList, tag))  # Append augmented text with intent tag

# Lemmatize and remove duplicates
words = [lemmatizer.lemmatize(word.lower()) for word in words if word not in ignoreLetters]
words = sorted(set(words))

# Sort classes
classes = sorted(set(classes))

# Preparing training data
training = []
outputEmpty = [0] * len(classes)

for document in documents:
    bag = []
    wordPatterns = document[0]
    wordPatterns = [lemmatizer.lemmatize(word.lower()) for word in wordPatterns]
    
    # Create bag of words
    for word in words:
        bag.append(1) if word in wordPatterns else bag.append(0)
    
    # Output is a binary array for classes (tags)
    outputRow = list(outputEmpty)
    outputRow[classes.index(document[1])] = 1
    training.append(bag + outputRow)

random.shuffle(training)
training = np.array(training)

trainX = training[:, :len(words)]
trainY = training[:, len(words):]

# Building the model
model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(128, input_shape=(len(trainX[0]),), activation='relu'))
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Dense(64, activation='relu'))
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Dense(len(trainY[0]), activation='softmax'))

# Compiling the model
sgd = tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

# Training the model
hist = model.fit(np.array(trainX), np.array(trainY), epochs=2000, batch_size=5, verbose=1)

# Save the model
model.save('chatbot_model.h5')
print('Model training complete and saved.')

# Save the words and classes after training
with open('words.pkl', 'wb') as f:
    pickle.dump(words, f)
with open('classes.pkl', 'wb') as f:
    pickle.dump(classes, f)

# Optional: Visualizing training results
# Plot accuracy
plt.plot(hist.history['accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.show()

# Plot loss
plt.plot(hist.history['loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.show()

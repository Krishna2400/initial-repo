# import random
# import json
# import pickle
# import numpy as np
# import tensorflow as tf

# import nltk
# # nltk.download('wordnet')
# # nltk.download('punkt_tab')
# # ntlk.download('punkt')
# from nltk.stem import WordNetLemmatizer
# lemmatizer = WordNetLemmatizer()

# intents = json.loads(open('intents.json').read())

# words = []
# classes = []
# documents = []
# ignoreLetters = ['?', '!', '.', ',']

# for intent in intents['intents']:
#     for pattern in intent['patterns']:
#         wordList = nltk.word_tokenize(pattern)
#         words.extend(wordList)
#         documents.append((wordList, intent['tag']))

#         if intent['tag'] not in classes:
#             classes.append(intent['tag'])

# words = [lemmatizer.lemmatize(word) for word in words if word not in ignoreLetters]
# words = sorted(set(words))

# classes = sorted(set(classes))
# pickle.dump(words, open('words.pkl', 'wb'))
# pickle.dump(classes, open('classes.pkl', 'wb'))

# training = []
# outputEmpty = [0] * len(classes)

# for document in documents:
#     bag = []
#     wordPatterns = document[0]
#     wordPatterns = [lemmatizer.lemmatize(word.lower()) for word in wordPatterns]
#     for word in words:
#         bag.append(1) if word in wordPatterns else bag.append(0)
    
#     outputRow = list(outputEmpty)
#     outputRow[classes.index(document[1])] = 1
#     training.append(bag + outputRow)

# random.shuffle(training)
# training = np.array(training)

# trainX = training[:, :len(words)]
# trainY = training[:, len(words):]

# model = tf.keras.Sequential()

# model.add(tf.keras.layers.Dense(128, input_shape=(len(trainX[0]),), activation = 'relu'))
# model.add(tf.keras.layers.Dropout(0.5))
# model.add(tf.keras.layers.Dense(64, activation = 'relu'))
# model.add(tf.keras.layers.Dropout(0.5))
# model.add(tf.keras.layers.Dense(len(trainY[0]), activation='softmax'))

# sgd = tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9, nesterov=True)
# model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

# hist = model.fit(np.array(trainX), np.array(trainY), epochs=200, batch_size=5, verbose=1)
# model.save('chatbot_model.h5', hist)
# print('Executed')

import random
import json
import pickle
import numpy as np
import tensorflow as tf

import nltk
# nltk.download('wordnet')
# nltk.download('punkt')
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()

# Loading intents JSON file
intents = json.loads(open('intents.json').read())

words = []
classes = []
documents = []
ignoreLetters = ['?', '!', '.', ',']

# Tokenize and lemmatize patterns
for intent in intents['intents']:
    for pattern in intent['patterns']:
        wordList = nltk.word_tokenize(pattern)
        words.extend(wordList)
        documents.append((wordList, intent['tag']))

        if intent['tag'] not in classes:
            classes.append(intent['tag'])

# Lemmatize and remove duplicates
words = [lemmatizer.lemmatize(word) for word in words if word not in ignoreLetters]
words = sorted(set(words))

classes = sorted(set(classes))

# Save words and classes using pickle
pickle.dump(words, open('words.pkl', 'wb'))
pickle.dump(classes, open('classes.pkl', 'wb'))

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
    
    # Output is a binary array
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
hist = model.fit(np.array(trainX), np.array(trainY), epochs=200, batch_size=5, verbose=1)

# Save the model
model.save('chatbot_model.h5', hist)
print('Model training complete and saved.')

# Optional: Visualizing training results
import matplotlib.pyplot as plt

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

# import random
# import json
# import pickle
# import numpy as np
# import tensorflow as tf
# import nltk
# from nltk.stem import WordNetLemmatizer
# from nltk.corpus import wordnet
# from tensorflow.keras.callbacks import ReduceLROnPlateau, LearningRateScheduler

# lemmatizer = WordNetLemmatizer()

# # Check for available GPUs
# physical_devices = tf.config.list_physical_devices('GPU')
# print("Num GPUs Available: ", len(physical_devices))

# # Set memory growth for GPU
# for device in physical_devices:
#     tf.config.experimental.set_memory_growth(device, True)

# # Loading intents JSON file
# intents = json.loads(open('intents.json').read())

# words = []
# classes = []
# documents = []
# ignoreLetters = ['?', '!', '.', ',']

# # Tokenize and lemmatize patterns
# for intent in intents['intents']:
#     for pattern in intent['patterns']:
#         wordList = nltk.word_tokenize(pattern)
#         words.extend(wordList)
#         documents.append((wordList, intent['tag']))

#         if intent['tag'] not in classes:
#             classes.append(intent['tag'])

# # Lemmatize and remove duplicates
# words = [lemmatizer.lemmatize(word) for word in words if word not in ignoreLetters]
# words = sorted(set(words))
# classes = sorted(set(classes))

# # Save words and classes using pickle
# pickle.dump(words, open('words.pkl', 'wb'))
# pickle.dump(classes, open('classes.pkl', 'wb'))

# # Data Augmentation: Function to slightly shuffle words and replace with synonyms
# def augment_sentence(sentence):
#     words = sentence.split()
    
#     # Slight shuffling of words
#     if len(words) > 3:
#         random.shuffle(words)
    
#     # Synonym replacement
#     for i, word in enumerate(words):
#         synonyms = wordnet.synsets(word)
#         if synonyms:
#             # Replace with a random synonym
#             synonym = synonyms[0].lemmas()[0].name()
#             words[i] = synonym
    
#     return ' '.join(words)

# # Augment data
# augmented_documents = []
# for document in documents:
#     pattern = ' '.join(document[0])
#     # Create 2 additional augmented versions of each pattern
#     for _ in range(2):
#         augmented_pattern = augment_sentence(pattern)
#         augmented_documents.append((nltk.word_tokenize(augmented_pattern), document[1]))

# documents.extend(augmented_documents)  # Append augmented data to original dataset

# # Preparing training data
# training = []
# outputEmpty = [0] * len(classes)

# for document in documents:
#     bag = []
#     wordPatterns = document[0]
#     wordPatterns = [lemmatizer.lemmatize(word.lower()) for word in wordPatterns]
    
#     # Create bag of words
#     for word in words:
#         bag.append(1) if word in wordPatterns else bag.append(0)
    
#     # Output is a binary array
#     outputRow = list(outputEmpty)
#     outputRow[classes.index(document[1])] = 1
#     training.append(bag + outputRow)

# random.shuffle(training)
# training = np.array(training)

# trainX = training[:, :len(words)]
# trainY = training[:, len(words):]

# # Reshaping input for Conv1D layer
# trainX = np.expand_dims(trainX, axis=-1)

# # Hyperparameter Tuning
# learning_rate = 0.001  # Experiment with learning rates
# batch_size = 32  # Increase batch size for stability

# # Build a more complex model
# model = tf.keras.Sequential()

# # Adding Conv1D layer
# model.add(tf.keras.layers.Conv1D(64, 3, padding='same', input_shape=(len(trainX[0]), 1), activation='relu'))
# model.add(tf.keras.layers.MaxPooling1D(pool_size=2))

# # Adding Bidirectional LSTM layers
# model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128, return_sequences=True)))
# model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)))

# # Add a fully connected Dense layer with Batch Normalization
# model.add(tf.keras.layers.Dense(128, activation='relu'))
# model.add(tf.keras.layers.BatchNormalization())
# model.add(tf.keras.layers.Dropout(0.5))

# # Add another Dense layer
# model.add(tf.keras.layers.Dense(64, activation='relu'))
# model.add(tf.keras.layers.BatchNormalization())
# model.add(tf.keras.layers.Dropout(0.5))

# # Output layer for classification
# model.add(tf.keras.layers.Dense(len(trainY[0]), activation='softmax'))

# # Compile the model with Adam optimizer
# adam = tf.keras.optimizers.Adam(learning_rate=learning_rate)
# model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])

# # Learning rate scheduler to reduce learning rate dynamically
# def lr_schedule(epoch, lr):
#     if epoch > 50:
#         lr = lr * 0.1
#     return lr

# # Remove EarlyStopping and use ReduceLROnPlateau
# lr_reduce = ReduceLROnPlateau(monitor='loss', factor=0.5, patience=5)
# lr_scheduler = LearningRateScheduler(lr_schedule)

# # Train the model for all epochs
# hist = model.fit(np.array(trainX), np.array(trainY), epochs=200, batch_size=batch_size, verbose=1,
#                  callbacks=[lr_reduce, lr_scheduler])

# # Save the model
# model.save('chatbot_model_augmented.h5', hist)
# print('Model training complete and saved.')

# # Print only augmented data
# print("Augmented Data Only:")

# for i, document in enumerate(augmented_documents):
#     pattern = ' '.join(document[0])
#     tag = document[1]
#     print(f"Augmented Data {i+1}: Pattern = '{pattern}' | Tag = '{tag}'")

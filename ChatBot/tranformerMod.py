import numpy as np
import tensorflow as tf
from transformers import BertTokenizer, TFBertForSequenceClassification
from sklearn.model_selection import train_test_split
from fuzzywuzzy import fuzz
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import json

# Load intents from the intents.json file
with open('intents.json', 'r') as f:
    dataset = json.load(f)

classes = [intent['tag'] for intent in dataset['intents']]
input_texts = [pattern for intent in dataset['intents'] for pattern in intent['patterns']]
labels = [classes.index(intent['tag']) for intent in dataset['intents'] for pattern in intent['patterns']]

# BERT Tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

def encode_sentences(sentences, tokenizer):
    input_ids, attention_masks = [], []
    for sent in sentences:
        encoded_dict = tokenizer.encode_plus(
            sent,
            add_special_tokens=True,
            max_length=64,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='tf'
        )
        input_ids.append(encoded_dict['input_ids'])
        attention_masks.append(encoded_dict['attention_mask'])
    return np.array(input_ids), np.array(attention_masks)

# Encode the input texts
input_ids, attention_masks = encode_sentences(input_texts, tokenizer)
labels = np.array(labels)

# Train-test split
train_inputs, test_inputs, train_labels, test_labels = train_test_split(input_ids.squeeze(), labels, test_size=0.2)
train_masks, test_masks = train_test_split(attention_masks.squeeze(), test_size=0.2)

# Load BERT model
model = TFBertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=len(classes))

# Compile the model
optimizer = tf.keras.optimizers.Adam(learning_rate=2e-5)  # Adjust learning rate if needed
loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])

# Early stopping and learning rate scheduler
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True)
lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=2)

# Train the model
history = model.fit(
    [train_inputs, train_masks], train_labels,
    epochs=20,  # Increase epochs
    batch_size=32,  # Adjust batch size if necessary
    validation_data=([test_inputs, test_masks], test_labels),
    callbacks=[early_stopping, lr_scheduler]
)

# Save the trained model in the TensorFlow SavedModel format
model.save('chatbot_transformer_model.keras')


# Plot training history (optional)
import matplotlib.pyplot as plt

plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend()
plt.show()

plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend()
plt.show()

# Prediction function
# Prediction function
def predict_intent(sentence):
    inputs = tokenizer(sentence, return_tensors="tf", max_length=64, padding='max_length', truncation=True)
    logits = model(inputs)[0]
    probs = tf.nn.softmax(logits, axis=-1)
    
    top_preds = tf.argsort(probs, direction='DESCENDING')[0][:3]
    top_probs = tf.gather(probs[0], top_preds).numpy()  # Use tf.gather to get probabilities

    if top_probs[0] - top_probs[1] < 0.1:  # Handling ambiguous intents
        return top_preds.numpy(), top_probs
    else:
        return top_preds.numpy()[0], top_probs[0]


# Define response function
def get_response(predicted_intents, dataset, query):
    for intent in dataset['intents']:
        if intent['tag'] == predicted_intents[0]:
            return np.random.choice(intent['responses'])  # Randomly choose a response from the intent
    return "Sorry, I couldn't find an answer."

# Example usage
query = "What is today's canteen food?"
predicted_intents, probabilities = predict_intent(query)
response = get_response(predicted_intents, dataset, query)
print(f"Predicted intents: {predicted_intents}, Probabilities: {probabilities}")
print(f"Response: {response}")

# Similar intent function using fuzzy matching
def get_similar_intent(query):
    similar_intents = []
    for intent in dataset['intents']:
        for pattern in intent['patterns']:
            similarity = fuzz.ratio(query.lower(), pattern.lower())
            if similarity > 60:  # A threshold for similarity
                similar_intents.append(intent['tag'])
    return similar_intents

# Unhandled query handling
def get_unhandled_query_suggestion(query):
    similar_intents = get_similar_intent(query)
    if similar_intents:
        return f"Did you mean: {', '.join(similar_intents)}?"
    else:
        return "Sorry, I couldn't find an answer. Could you clarify?"

# Plotting confusion matrix
predictions = model.predict([test_inputs, test_masks])
predicted_classes = tf.argmax(predictions[0], axis=-1).numpy()

cm = confusion_matrix(test_labels, predicted_classes)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()

# Context tracking
context = {}

def set_context(session_id, new_context):
    context[session_id] = new_context

def get_context(session_id):
    return context.get(session_id, '')

def clear_context(session_id):
    context[session_id] = ''

def get_response_with_context(predicted_intents, session_id, query):
    if get_context(session_id):
        previous_context = get_context(session_id)
        if previous_context == 'canteen_menu':
            clear_context(session_id)
            return "Here is the canteen menu based on your request!"

    # Handle normal intent responses
    response = get_response(predicted_intents, dataset, query)
    if predicted_intents[0] == 'canteen_menu':
        set_context(session_id, 'canteen_menu')  # Store context for next turn
    return response

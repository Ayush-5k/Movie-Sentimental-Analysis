import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences 
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from string import punctuation

# Load the dataset
train_df = pd.read_csv('imdb_train.csv')
test_df = pd.read_csv('imdb_test.csv')

# Preprocess the text data
def preprocess_text(text):
    text = text.lower()
    text = ''.join([c for c in text if c not in punctuation])
    return text

train_df['review'] = train_df['review'].apply(preprocess_text)
test_df['review'] = test_df['review'].apply(preprocess_text)

# Tokenize the text
tokenizer = Tokenizer(num_words=10000)
tokenizer.fit_on_texts(train_df['review'])
train_sequences = tokenizer.texts_to_sequences(train_df['review'])
test_sequences = tokenizer.texts_to_sequences(test_df['review'])
train_padded_sequences = pad_sequences(train_sequences, maxlen=100)
test_padded_sequences = pad_sequences(test_sequences, maxlen=100)

# Prepare the labels
train_labels = train_df['sentiment'].values
test_labels = test_df['sentiment'].values

# Split the data
X_train, X_val, y_train, y_val = train_test_split(train_padded_sequences, train_labels, test_size=0.2, random_state=42)

# Build the model
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(input_dim=10000, output_dim=128, input_length=100),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Early stopping callback
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

# Train the model
history = model.fit(X_train, y_train, epochs=10, validation_data=(X_val, y_val), callbacks=[early_stopping])

# Save the model
model.save('sentiment_model.h5')

# Evaluate the model
y_pred = (model.predict(test_padded_sequences) > 0.5).astype("int32")
accuracy = accuracy_score(test_labels, y_pred)
print(f'Test Accuracy: {accuracy}')

conf_matrix = confusion_matrix(test_labels, y_pred)
sns.heatmap(conf_matrix, annot=True, fmt='d')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

print(f'Classification Report:\n{classification_report(test_labels, y_pred)}')

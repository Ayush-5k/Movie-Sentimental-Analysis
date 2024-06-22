import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pandas as pd
from string import punctuation

# Load the trained model
model = load_model('sentiment_model.h5')

# Tokenizer settings
train_df = pd.read_csv('imdb_train.csv')
tokenizer = Tokenizer(num_words=10000)
tokenizer.fit_on_texts(train_df['review'])

# Preprocessing function
def preprocess_text(text):
    text = text.lower()
    text = ''.join([c for c in text if c not in punctuation])
    return text

st.title('Movie Review Sentiment Analysis')

review = st.text_area('Enter a movie review:')
if st.button('Predict'):
    preprocessed_review = preprocess_text(review)
    sequence = tokenizer.texts_to_sequences([preprocessed_review])
    padded_sequence = pad_sequences(sequence, maxlen=100)
    prediction = (model.predict(padded_sequence) > 0.5).astype("int32")[0][0]
    sentiment = 'Positive' if prediction == 1 else 'Negative'
    st.write(f'Sentiment: {sentiment}')

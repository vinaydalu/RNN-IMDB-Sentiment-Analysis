## Import libraries and load the model
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import load_model

# Load imdb dataset word index
word_index = imdb.get_word_index()
reverse_word_index = {value:key for (key,value) in word_index.items()}


# Load the trained model
model = load_model("simple_rnn_imdb.h5")

# Function to decode reviews
def decode_review(encoded_review):
    return ' '.join([reverse_word_index.get(i-3,'?') for i in encoded_review])

# Function to preprocess user input
def preprocess_text(text):
    words = text.lower().split()
    encoded_review = [word_index.get(word, 2) for word in words]  # Use 2 for unknown words
    padded_review = sequence.pad_sequences([encoded_review], maxlen=500)
    return padded_review

# Prediction function

def predict_sentiment(review):
    preprocessed_input = preprocess_text(review)
    prediction = model.predict(preprocessed_input)
    sentiment = "Positive" if prediction[0][0] >= 0.5 else "Negative"
    return sentiment, prediction[0][0]

## Streamlit app
import streamlit as st

st.title("IMDB Movie Review Sentiment Analysis")
st.write("Enter a movie review to predict its sentiment (Positive/Negative).")

# User Input
user_review = st.text_area("Movie Review:")

if st.button("Predict Sentiment"):
    preprocessed_input = preprocess_text(user_review)
    prediction = model.predict(preprocessed_input)
    sentiment, score = predict_sentiment(user_review)
    st.write(f"Sentiment: {sentiment}")
    st.write(f"Prediction Score: {score}")
else:
    st.write("Please enter a review and click 'Predict Sentiment'")

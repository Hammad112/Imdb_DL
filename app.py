import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import load_model

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Set page config
st.set_page_config(page_title="IMDB RNN Sentiment Analyzer", layout="centered")

# Custom CSS styles
st.markdown("""
    <style>
    body {
        background-color: #f2f6fc;
        font-family: 'Segoe UI', sans-serif;
    }
    .main-title {
        font-size: 3em;
        font-weight: bold;
        background: linear-gradient(90deg, #8e2de2, #4a00e0);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        animation: slideIn 1s ease-in-out;
        text-align: center;
    }
    @keyframes slideIn {
        from { opacity: 0; transform: translateY(-30px); }
        to { opacity: 1; transform: translateY(0); }
    }
   
    </style>
""", unsafe_allow_html=True)

# Title
st.markdown('<div class="main-title">💬 IMDB RNN Review Sentiment Analyzer</div>', unsafe_allow_html=True)



# Load word index and model with cache
@st.cache_resource
def load_resources():
    word_index = imdb.get_word_index()
    reversed_word_index = {value: key for key, value in word_index.items()}
    model = load_model("rnn_simple.h5")
    return model, word_index, reversed_word_index

model, word_index, reversed_word_index = load_resources()

# Preprocessing function
def preprocess_text(text):
    words = text.lower().split()
    encoded = [word_index.get(word, 2) + 3 for word in words if word_index.get(word, 2) < 9997]
    if len(encoded) == 0:
        st.warning("⚠️ Your input contains no recognizable words from the model's vocabulary.")
    padded = sequence.pad_sequences([encoded], maxlen=500)
    return padded



# Prediction function
def predict_sentiment(text):
    processed = preprocess_text(text)
    prediction = model.predict(processed)
    sentiment = "✅ Positive" if prediction[0][0] > 0.5 else "❌ Negative"
    return sentiment, float(prediction[0][0])

# Input box
st.subheader("📝 Enter a Movie Review")
user_input = st.text_area("Write your IMDB review below:", height=200)

if st.button("Analyze Sentiment 🎯"):
    with st.spinner("Crunching through words..."):
        sentiment, score = predict_sentiment(user_input)

    st.markdown(f"""
        <div style='padding: 2em; text-align: center; background: #e3f2fd; border-radius: 15px; box-shadow: 0 4px 10px rgba(0,0,0,0.1);'>
            <h2 style='color: #0d47a1;'>🔍 Sentiment Result</h2>
            <p style='font-size: 2em; font-weight: bold; color: #1a237e;'>{sentiment}</p>
            <p style='color: #555;'>Confidence Score: {score:.2f}</p>
        </div>
    """, unsafe_allow_html=True)

# About Section
st.markdown("""
### 📘 About
This Streamlit app uses a pretrained **Recurrent Neural Network (RNN)** on the IMDB dataset to classify movie reviews as positive or negative.
Paste any review above and see the magic of deep learning in action!
""")
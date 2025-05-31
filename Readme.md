# 🎬 IMDB Review Sentiment Analyzer

[![Python](https://img.shields.io/badge/python-3.8%2B-blue)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/streamlit-v1.0-orange)](https://streamlit.io/)
[![TensorFlow](https://img.shields.io/badge/tensorflow-v2.11-yellowgreen)](https://www.tensorflow.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)

---

## 🚀 Project Overview

Welcome to **IMDB Review Sentiment Analyzer**, a sleek and user-friendly Streamlit app powered by a pretrained Recurrent Neural Network (RNN). It classifies movie reviews from the IMDB dataset as **positive** or **negative** — instantly, accurately, and with confidence scores.

Built with **TensorFlow** and **Streamlit**, this app showcases how to deploy NLP deep learning models for real-world applications, wrapped in a modern web UI.

---

## 🔍 Key Features

- 🔮 **Deep Learning with RNN:** Robust sentiment classification with a pretrained LSTM model.
- 🧠 **On-the-fly Preprocessing:** Clean, tokenize, encode, and pad user inputs matching training data.
- 💻 **Streamlit UI:** Responsive, visually appealing interface with smooth animations.
- 📊 **Confidence Scores:** Real-time display of prediction probabilities.
- 💾 **Optimized Caching:** Uses `@st.cache_resource` to minimize latency and maximize performance.
- 🎨 **Modern UX:** Gradient backgrounds, soft shadows, and adaptive layouts for all devices.

---

## 🛠️ Tech Stack

| Layer    | Tools / Libraries                                    |
|----------|-----------------------------------------------------|
| Frontend | `Streamlit`, `HTML/CSS`, Custom CSS animations      |
| Backend  | `TensorFlow`, `Keras`, `NumPy`                      |
| Model    | RNN with Embedding + LSTM layers (Pretrained)       |
| Dataset  | IMDB Movie Review Dataset (Top 10,000 words)        |

---

## 📁 Project Structure

```plaintext
imdb-sentiment-analyzer/
│
├── rnn_simple.h5           # Pretrained RNN model weights
├── app.py                  # Streamlit application script
├── requirements.txt        # Python package dependencies
└── README.md               # This documentation file

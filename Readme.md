markdown
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

- 🔮 **Deep Learning with RNN:** Robust sentiment classification using a pretrained LSTM model.
- 🧠 **On-the-fly Preprocessing:** Clean, tokenize, encode, and pad user inputs using the exact pipeline as training.
- 💻 **Streamlit UI:** Responsive, visually appealing interface with smooth animations and modern design.
- 📊 **Confidence Scores:** Real-time display of prediction probabilities for better transparency.
- 💾 **Optimized Caching:** Uses Streamlit’s `@st.cache_resource` decorator to reduce latency.
- 🎨 **Modern UX:** Gradient backgrounds, soft shadows, and responsive layouts ensure clarity and engagement.

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
└── README.md               # Project documentation (this file)

---

## ⚙️ Installation & Usage

### ✅ Prerequisites

* Python 3.8 or higher
* `pip` or `virtualenv`

### 🔧 Installation Steps

```bash
# Clone the repository
git clone https://github.com/yourusername/imdb-sentiment-analyzer.git
cd imdb-sentiment-analyzer

# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 🚀 Run the App

```bash
streamlit run app.py
```

Open your browser and visit [] to interact with the app.

---

## 🧪 Testing the Model

Try these sample reviews to test the model’s prediction quality:

**Positive Examples:**

* *“An absolutely amazing experience with a heartfelt story.”*
* *“Outstanding acting and direction — a must watch!”*

**Negative Examples:**

* *“Boring and predictable, a total waste of time.”*
* *“The script was weak and performances were flat.”*

---

## ⚠️ Limitations

* Vocabulary is limited to the **top 10,000** most frequent words from the IMDB dataset.
* Misspelled, rare, or slang words may reduce prediction accuracy.
* Currently supports **English** reviews only.

---

## 🌟 Future Enhancements

* Upgrade the model to transformer-based architectures (e.g., BERT, DistilBERT) for improved accuracy.
* Add multilingual support to handle reviews in different languages.
* Enable file upload functionality for batch review analysis (.txt, .csv).
* Implement user feedback collection and model retraining pipeline.
* Add deployment on cloud platforms for scalable access.



✨ *Thank you for checking out this project! Feel free to ⭐ star and contribute.*


# app.py
import streamlit as st
import pickle
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
nltk.download('stopwords')

# Load the model and vectorizer
model = pickle.load(open('fake_news_model.pkl', 'rb'))
vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))

# Preprocessing Function
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

def preprocess_text(content):
    content = re.sub('[^a-zA-Z ]', ' ', content)
    content = content.lower()
    content = content.split()
    content = [stemmer.stem(word) for word in content if word not in stop_words]
    return ' '.join(content)

# Streamlit UI
st.set_page_config(page_title="Fake News Detector", page_icon="🕵️", layout="centered")

st.title("📰 Fake News Detection App")
st.markdown("**Enter a news headline or paragraph to check if it's Real or Fake.**")

# Input box
user_input = st.text_area("Enter News Text", height=200)

# Predict button
if st.button("Check Now"):
    if user_input.strip() == "":
        st.warning("Please enter some text first.")
    else:
        processed = preprocess_text(user_input)
        vectorized = vectorizer.transform([processed])
        prediction = model.predict(vectorized)

        if prediction[0] == 1:
            st.success("✅ This appears to be **Real News**.")
        else:
            st.warning("❌ This appears to be **Fake News**.")
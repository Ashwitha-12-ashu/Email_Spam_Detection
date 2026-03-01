import streamlit as st
import pickle
import re
import nltk
from nltk.corpus import stopwords

# Load stopwords
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Load saved model + vectorizer
model = pickle.load(open("spam_model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

# Text cleaning function
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    words = text.split()
    words = [word for word in words if word not in stop_words]
    return " ".join(words)

# Streamlit UI
st.title("📧 Email/SMS Spam Detection App")
st.write("Enter a message below and I will classify it as **SPAM** or **HAM (NOT SPAM)**.")

user_input = st.text_area("Type your message here...")

if st.button("Predict"):
    cleaned = clean_text(user_input)
    vectorized = vectorizer.transform([cleaned])
    prediction = model.predict(vectorized)[0]

    if prediction == 1:
        st.error("🚨 This message is **SPAM**!")
    else:
        st.success("✅ This message is **NOT SPAM**.")

import streamlit as st
import string
import joblib
import nltk
import re
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# -------------------------
# NLTK Resource Setup
# -------------------------
def download_nltk_resources():
    resources = ['punkt', 'stopwords', 'wordnet', 'omw-1.4']
    for resource in resources:
        try:
            nltk.data.find(f'tokenizers/{resource}' if resource == 'punkt' else f'corpora/{resource}')
        except LookupError:
            nltk.download(resource)

download_nltk_resources()

# -------------------------
# Load Model and Vectorizer
# -------------------------
@st.cache_resource
def load_model():
    model = joblib.load("spam_model1.pkl")             # Trained Naive Bayes model
    vectorizer = joblib.load("tfidf_vectorizer1.pkl")  # TF-IDF Vectorizer
    return model, vectorizer

model, tfidf = load_model()

# -------------------------
# Preprocessing Function
# -------------------------
stopwords_en = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

def clean_text_lemma(text):
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)  # remove punctuation
    tokens = text.split()
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stopwords_en and word.isalpha()]
    return " ".join(tokens)

# -------------------------
# Streamlit UI
# -------------------------
st.set_page_config(page_title="📩 SMS Spam Classifier", page_icon="📲", layout="centered")

st.title("📩 SMS Spam Detection App")
st.markdown("🔍 Enter any SMS message and let the AI model classify it as **Spam** or **Ham**.")

# Example Buttons
st.subheader("✨ Try Example Messages")
col1, col2 = st.columns(2)
with col1:
    if st.button("📨 Example Ham"):
        st.session_state["user_input"] = "Hey, are we still meeting at 7 pm tonight?"
with col2:
    if st.button("🚨 Example Spam"):
        st.session_state["user_input"] = "Congratulations! You've won $1000 cash prize. Click here to claim."

# Input Box
user_input = st.text_area("✍️ Type your SMS message below:", value=st.session_state.get("user_input", ""))

if st.button("🚀 Classify Message"):
    if user_input.strip() == "":
        st.warning("⚠️ Please enter a message before classification.")
    else:
        # Cleaning
        cleaned = clean_text_lemma(user_input)

        # Vectorize
        vect_text = tfidf.transform([cleaned])

        # Predict
        prediction = model.predict(vect_text)[0]
        prob_spam = model.predict_proba(vect_text)[0][1]
        prob_ham = model.predict_proba(vect_text)[0][0]

        # Show Results
        label = "Spam" if prediction == "spam" else "Ham"
        confidence = prob_spam if prediction == "spam" else prob_ham

        if prediction == "spam":
            st.error(f"🚨 **Spam Detected!**")
        else:
            st.success(f"✅ **Ham (Not Spam)**")

        st.markdown("---")
        st.subheader("📊 Prediction Confidence")
        st.metric(label=f"Prediction: {label}", value=f"{confidence*100:.2f}% Confidence")
        st.progress(float(prob_spam))
        st.write(f"Spam Probability: **{prob_spam*100:.2f}%**")
        st.write(f"Ham Probability: **{prob_ham*100:.2f}%**")

        # Optional: Downloadable Result
        result_df = pd.DataFrame({
            "Message": [user_input],
            "Cleaned": [cleaned],
            "Prediction": [prediction],
            "Spam Probability": [prob_spam],
            "Ham Probability": [prob_ham]
        })
        st.download_button("📥 Download Result as CSV", result_df.to_csv(index=False), file_name="spam_result.csv")

# -------------------------
# Sidebar Info
# -------------------------
st.sidebar.title("ℹ️ About this Project")
st.sidebar.markdown("""
- 📌 **Dataset:** [SMS Spam Collection](https://archive.ics.uci.edu/ml/datasets/SMS+Spam+Collection)  
- 📌 **Model:** Multinomial Naive Bayes  
- 📌 **Features:** TF-IDF Vectorizer  
- 🚀 Built with **Streamlit + scikit-learn**
""")

with st.expander("ℹ️ What does 'Ham' mean?"):
    st.write("In spam detection, 'Ham' refers to legitimate, non-spam messages.")






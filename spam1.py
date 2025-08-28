import streamlit as st
import string
import joblib
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk
nltk.download('punkt')


# -------------------------
# Download NLTK data (run once)
# -------------------------
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

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
    text = "".join([char.lower() for char in text if char not in string.punctuation])
    tokens = word_tokenize(text)
    text = " ".join([lemmatizer.lemmatize(word) for word in tokens if word not in stopwords_en])
    return text

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
        if prediction == "spam":
            st.error(f"🚨 **Spam Detected!**")
        else:
            st.success(f"✅ **Ham (Not Spam)**")

        st.markdown("---")
        st.subheader("📊 Prediction Confidence")
        st.progress(float(prob_spam))  # visual confidence bar
        st.write(f"Spam Probability: **{prob_spam*100:.2f}%**")
        st.write(f"Ham Probability: **{prob_ham*100:.2f}%**")

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



import streamlit as st
import joblib
import re
import pandas as pd
from datetime import datetime
import os

# -----------------------------
# Document types dictionary
# -----------------------------
document_types = {
    "Distribution_Notice": ["Distribution Notice", "Distribtn Notice", "Dist Notice", "Distribution", "Distribution Letter", "Dist Letter", "dist letter"],
    "SOA": ["SOA", "Statement of Account", "Statmnt of Acct", "soa"],
    "Advance_Tax": ["statement of income", 'addance tax', 'smt of income', 'advance letter', 'tax advance', 'advanc doc'],
    "Annual_Tax": ['annaully tax report', 'anual tax', 'yearly tax letter', 'annnual tax', 'tax annual', 'annual'],
    "Form_64C": ["Form 64C", "Form64C", "Frm 64C", "form 64", "64c"],
    "Drawdown_Grace_Period_Notice": ["Drawdown Grace Period Notice", "Drawdon Grace Notice", "drqwd grace notice"],
    "Drawdown_Notice": ["Drawdown Notice", "Drawdwn Notice", "draw down ntice", "DDN", "ddn", "drawdown"],
    "TDS": ['TDS', 'FORM16A', 'frm 16a', 'Tax Deducted at Source', "tds's", 'form 16:A', 'form 16'],
    "Newsletter": ['newsletter', "news letter", 'news', 'nws lter', 'NEWSLETTER', 'NEWS letter', 'News Letter']
}
doc_keywords = [item.lower() for sublist in document_types.values() for item in sublist]

# -----------------------------
# Streamlit App Configuration
# -----------------------------
st.set_page_config(page_title="Email Classifier", page_icon="📧")
st.title("📧 Email Classifier — Ontopic vs Offtopic")
st.markdown("Enter the **first email** from a thread. If it refers to a document, the model will classify it.")

# -----------------------------
# Load model and vectorizer
# -----------------------------
@st.cache_resource
def load_model():
    vectorizer = joblib.load("tfidf_vectorizer (2).pkl")
    model = joblib.load("logreg_model (2).pkl")
    return vectorizer, model

vectorizer, clf = load_model()

# -----------------------------
# Email processing helpers
# -----------------------------
def contains_document_keywords(text):
    if not isinstance(text, str):
        return False
    lower_text = text.lower()
    return any(keyword in lower_text for keyword in doc_keywords)

def clean_text(text):
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^a-z0-9\s]", "", text)
    return text.strip()

def log_feedback(email_text, prediction, feedback):
    row = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "email_text": email_text,
        "prediction": prediction,
        "feedback": feedback
    }
    df = pd.DataFrame([row])
    file_exists = os.path.exists("feedback_log.csv")
    df.to_csv("feedback_log.csv", mode="a", index=False, header=not file_exists)

# -----------------------------
# Input & Classification
# -----------------------------
email_input = st.text_area("✉️ Enter the first email in thread:", height=300)

if "last_prediction" not in st.session_state:
    st.session_state.last_prediction = None
    st.session_state.last_email = None

if st.button("🔍 Submit"):
    if not email_input.strip():
        st.warning("Please enter an email.")
    else:
        # Check for document presence
        if not contains_document_keywords(email_input):
            prediction_label = "offtopic"
        else:
            cleaned_email = clean_text(email_input)
            features = vectorizer.transform([cleaned_email])
            prediction_label = clf.predict(features)[0]

        st.session_state.last_prediction = prediction_label
        st.session_state.last_email = email_input.strip()

        st.success(f"🧠 Prediction: **{prediction_label.upper()}**")

# -----------------------------
# Feedback Section
# -----------------------------
if st.session_state.last_prediction:
    st.write("### Was this prediction correct?")
    col1, col2 = st.columns(2)

    with col1:
        if st.button("✅ Approve"):
            log_feedback(st.session_state.last_email, st.session_state.last_prediction, "approved")
            st.success("✅ Feedback recorded as *approved*.")

    with col2:
        if st.button("❌ Disapprove"):
            log_feedback(st.session_state.last_email, st.session_state.last_prediction, "disapproved")
            st.warning("❌ Feedback recorded as *disapproved*.")

# -----------------------------
# Download Feedback CSV
# -----------------------------
if os.path.exists("feedback_log.csv"):
    st.write("---")
    st.subheader("📥 Download Feedback Log")
    with open("feedback_log.csv", "rb") as f:
        st.download_button(
            label="Download CSV",
            data=f,
            file_name="feedback_log.csv",
            mime="text/csv"
        )

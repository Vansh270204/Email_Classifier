import streamlit as st
import joblib
import re
import pandas as pd
from datetime import datetime
import os

# ✅ Set page config first
st.set_page_config(page_title="Email Classifier", page_icon="📧")

# ----------------------
# Load models
# ----------------------
@st.cache_resource
def load_model():
    vectorizer = joblib.load("tfidf_vectorizer.joblib")
    model = joblib.load("email_classifier_model.joblib")
    label_encoder = joblib.load("label_encoder.joblib")
    return vectorizer, model, label_encoder

vectorizer, clf, label_encoder = load_model()

# ----------------------
# Helper functions
# ----------------------
def clean_text(text):
    if not isinstance(text, str):
        return ""
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

# ----------------------
# Streamlit App UI
# ----------------------
st.title("📧 Email Classifier — Ontopic vs Offtopic")
st.markdown("Enter a single email below. The model will classify it, and you can approve or disapprove the result.")

email_input = st.text_area("✉️ Enter an email to classify:", height=300)

# Session state to track predictions
if "last_prediction" not in st.session_state:
    st.session_state.last_prediction = None
    st.session_state.last_email = None

# Submit button
if st.button("🔍 Submit"):
    if not email_input.strip():
        st.warning("Please enter an email.")
    else:
        cleaned_email = clean_text(email_input)
        features = vectorizer.transform([cleaned_email])
        prediction = clf.predict(features)[0]
        label = label_encoder.inverse_transform([prediction])[0]

        st.session_state.last_prediction = label
        st.session_state.last_email = email_input.strip()

        st.success(f"🧠 Prediction: **{label.upper()}**")

# Show feedback buttons only after prediction
if st.session_state.last_prediction:
    st.write("### Was this prediction correct?")
    col1, col2 = st.columns(2)

    with col1:
        if st.button("✅ Approve"):
            log_feedback(
                st.session_state.last_email,
                st.session_state.last_prediction,
                "approved"
            )
            st.success("✅ Feedback recorded as *approved*.")

    with col2:
        if st.button("❌ Disapprove"):
            log_feedback(
                st.session_state.last_email,
                st.session_state.last_prediction,
                "disapproved"
            )
            st.warning("❌ Feedback recorded as *disapproved*.")

# ----------------------
# Download Feedback CSV
# ----------------------
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

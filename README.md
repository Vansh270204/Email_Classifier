# 📧 Email Classifier (Streamlit App)

This is a simple Streamlit-based web app to classify emails as `ontopic` or `offtopic`.

## 🚀 Features
- Classify a single email using a trained ML model
- Approve/Disapprove model predictions
- Save tester feedback to `feedback_log.csv`
- Download feedback directly from browser

## 🛠 Setup Instructions

```bash
# Clone the repo
git clone https://github.com/<your-username>/<repo-name>.git
cd email_classifier

# Set up a virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py

import streamlit as st
import joblib
import os

st.set_page_config(page_title="YouTube Comment Spam Classifier", page_icon="🛡️", layout="centered")

@st.cache_resource
def load_artifacts():
    model_path = "final_spam_classifier_model.pkl"
    vectorizer_path = "tfidf_vectorizer.pkl"

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"model file not found: {model_path}")
    if not os.path.exists(vectorizer_path):
        raise FileNotFoundError(f"vectorizer file not found: {vectorizer_path}")

    model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)
    return model, vectorizer

st.title("🛡️ YouTube Comment Spam Classifier")
st.write("enter a comment below and click **predict**.")

try:
    model, vectorizer = load_artifacts()
except Exception as e:
    st.error("could not load model artifacts.")
    st.exception(e)
    st.stop()

comment = st.text_area("comment", placeholder="e.g., check out my channel and win free iphone...", height=140)

col1, col2 = st.columns(2)

with col1:
    predict_btn = st.button("predict", use_container_width=True)

with col2:
    clear_btn = st.button("clear", use_container_width=True)

if clear_btn:
    st.rerun()

if predict_btn:
    if comment.strip() == "":
        st.warning("please enter a comment.")
    else:
        x_input = vectorizer.transform([comment])
        pred = int(model.predict(x_input)[0])

        if pred == 1:
            st.error("prediction: spam 🚫")
        else:
            st.success("prediction: not spam ✅")

        if hasattr(model, "decision_function"):
            score = float(model.decision_function(x_input)[0])
            st.caption(f"decision score: {score:.4f}")
        elif hasattr(model, "predict_proba"):
            proba = float(model.predict_proba(x_input)[0][1])
            st.caption(f"spam probability: {proba:.4f}")

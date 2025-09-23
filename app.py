# app.py
import streamlit as st
from predictor import predict_fake_news

st.set_page_config(page_title="Fake News Detector", layout="centered")

st.title("📰 Fake News Detector")
st.markdown("Enter a news article below to check if it's **Real** or **Fake**.")

# Input box
user_input = st.text_area("Paste the news text here:", height=200)

# Predict button
if st.button("Detect"):
    if user_input.strip() == "":
        st.warning("Please enter some text.")
    else:
        with st.spinner("Analyzing..."):
            result = predict_fake_news(user_input)
            st.success(f"Prediction: **{result}**")

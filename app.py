import streamlit as st
import joblib

model = joblib.load("model.joblib")
vec = joblib.load("vectorizer.joblib")

st.set_page_config(page_title="Fake News Detector", page_icon="📰")
st.title("📰 Fake News Detector")
st.write("Enter a news headline and article content below to check if it's Fake or Real.")

title = st.text_input("News Headline")
article = st.text_area("News Article Content", height=250)

if st.button("Predict"):
    if (not title or title.strip() == "") and (not article or article.strip() == ""):
        st.error("Please enter at least a headline or article content.")
    else:
        text = (title.strip() + " " + article.strip()).strip()
        if len(text.split()) < 5:
            st.warning("The text is very short — prediction may not be accurate.")
        X = vec.transform([text])
        proba = model.predict_proba(X)[0]
        pred = model.predict(X)[0]
        label = "FAKE" if pred == 1 else "REAL"
        confidence = proba[pred]
        st.markdown(f"### Prediction: **{label}**")
        st.write(f"Confidence: {confidence:.3f}")
        st.progress(int(confidence * 100))

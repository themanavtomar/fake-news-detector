import streamlit as st
import joblib
import numpy as np

model = joblib.load("model.joblib")
vec = joblib.load("vectorizer.joblib")

st.set_page_config(page_title="Fake News Detector — Pro", page_icon="📰", layout="wide")

css = """
<style>
body {background-color: #f7f9fb;}
.header {padding: 10px 0;}
.card {background: white; border-radius: 12px; padding: 18px; box-shadow: 0 4px 20px rgba(0,0,0,0.05);}
.pred-badge {font-weight:700; padding:8px 12px; border-radius:999px;}
.fake {background: #ffe6e6; color:#a10000;}
.real {background: #e6fff0; color:#006a2e;}
.small {font-size:0.9rem; color:#555;}
.example-btn {margin-right:8px;}
.tokens {background:#fafafa; padding:10px; border-radius:8px; font-family:monospace;}
</style>
"""
st.markdown(css, unsafe_allow_html=True)

st.markdown('<div class="header"><h1>📰 Fake News Detector — Pro</h1><div class="small">Improved UX, example presets, confidence meter & explainability</div></div>', unsafe_allow_html=True)

examples = {
    "Select example": ("",""),
    "Real: ISRO GSAT-30 Launch": ("ISRO launches GSAT-30 successfully", "The Indian Space Research Organisation (ISRO) successfully launched its latest communication satellite GSAT-30 today from the Guiana Space Centre in French Guiana. The satellite will provide high-quality telecommunication, television broadcasting, and broadband services across India. Officials said the mission was completed without any technical issues."),
    "Fake: Alien base on moon": ("Secret alien base on the moon discovered", "A secret government document leaked yesterday claims that scientists have discovered a hidden alien base on the moon. According to the report, astronauts found strange metallic structures during the Apollo missions, but NASA covered it up to prevent public panic.")
}

col1, col2 = st.columns([3,1])
with col1:
    title = st.text_input("Headline", "")
with col2:
    choice = st.selectbox("Try example", list(examples.keys()))
    if st.button("Load example"):
        t, a = examples.get(choice, ("",""))
        st.session_state['prefill_title'] = t
        st.session_state['prefill_article'] = a

article = st.text_area("Article content", height=260, value=st.session_state.get('prefill_article', ""))
if 'prefill_title' in st.session_state and title.strip()=="":
    title = st.session_state.get('prefill_title', "")

st.markdown("---")
left, right = st.columns([2,1])

with left:
    if st.button("Predict", key="predict"):
        text = (title.strip() + " " + article.strip()).strip()
        if text=="":
            st.error("Headline ya article dala karo.")
        else:
            if len(text.split()) < 5:
                st.warning("Text bahut chhota hai — prediction unreliable ho sakta hai.")
            X = vec.transform([text])
            proba = model.predict_proba(X)[0]
            pred = int(model.predict(X)[0])
            label = "FAKE" if pred==1 else "REAL"
            confidence = float(proba[pred])
            pct = int(confidence*100)
            conf_text = "Low"
            if confidence >= 0.75:
                conf_text = "High"
            elif confidence >= 0.5:
                conf_text = "Medium"
            st.markdown('<div class="card">', unsafe_allow_html=True)
            if pred==1:
                st.markdown(f'<div style="display:flex; justify-content:space-between; align-items:center"><div><h3>Prediction: <span class="pred-badge fake"> {label} </span></h3></div><div class="small">Confidence: {confidence:.3f} ({pct}%) • {conf_text}</div></div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div style="display:flex; justify-content:space-between; align-items:center"><div><h3>Prediction: <span class="pred-badge real"> {label} </span></h3></div><div class="small">Confidence: {confidence:.3f} ({pct}%) • {conf_text}</div></div>', unsafe_allow_html=True)
            st.progress(pct)
            try:
                feat_names = vec.get_feature_names_out()
                coefs = model.coef_[0]
                X_idx = X.nonzero()[1]
                word_scores = {}
                for idx in X_idx:
                    if idx < len(feat_names):
                        word_scores[feat_names[idx]] = coefs[idx]
                top_pos = sorted(word_scores.items(), key=lambda x: x[1], reverse=True)[:5]
                top_neg = sorted(word_scores.items(), key=lambda x: x[1])[:5]
                st.markdown("**Top contributing words (estimated):**")
                cols = st.columns(2)
                with cols[0]:
                    st.markdown("**Pushes to FAKE**")
                    if top_pos:
                        for w,s in top_pos:
                            st.markdown(f"- `{w}` • {s:.4f}")
                    else:
                        st.write("N/A")
                with cols[1]:
                    st.markdown("**Pushes to REAL**")
                    if top_neg:
                        for w,s in top_neg:
                            st.markdown(f"- `{w}` • {s:.4f}")
                    else:
                        st.write("N/A")
            except Exception:
                st.write("")
            st.markdown('</div>', unsafe_allow_html=True)

with right:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### Quick tips")
    st.markdown("- Clear headline + article improves results")
    st.markdown("- For Hindi text, retrain model on Hindi dataset")
    st.markdown("- Examples available in the dropdown")
    st.markdown("</div>", unsafe_allow_html=True)

st.markdown("<div style='margin-top:18px' class='small'>Built with TF-IDF + Logistic Regression • Model explainability is approximate (word-level weights)</div>", unsafe_allow_html=True)

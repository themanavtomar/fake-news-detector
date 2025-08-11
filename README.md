# 📰 Fake News Detector

A simple yet effective Machine Learning application to detect fake news articles using Natural Language Processing (NLP).

This application uses a Logistic Regression model trained on the "Fake and Real News" dataset from Kaggle. It combines both the news headline and the article content, processes them using a TF-IDF vectorizer, and predicts whether the news is **Fake** or **Real** along with a confidence score.

### 🚀 Features
- Enter **headline** and **article content** separately
- Combines title + content for better accuracy
- Shows prediction label (**FAKE** or **REAL**)
- Displays confidence percentage and progress bar
- Warning for very short text inputs
- Deployable for free on **Streamlit Community Cloud**

### 🛠️ Tech Stack
- **Python 3**
- **Pandas**, **Scikit-learn**
- **Streamlit** for the web interface
- **Joblib** for saving/loading the trained model

### 📂 How it Works
1. Train model using `train_model.py` (TF-IDF + Logistic Regression)
2. Save `model.joblib` and `vectorizer.joblib`
3. Run `app.py` to launch Streamlit app
4. Paste any news headline & content → get prediction

### 📦 Installation & Usage
```bash
# Clone this repository
git clone https://github.com/your-username/fake-news-detector.git
cd fake-news-detector

# Create and activate virtual environment
python -m venv venv
venv\Scripts\activate   # Windows
source venv/bin/activate  # Mac/Linux

# Install dependencies
pip install -r requirements.txt

# Train the model
python train_model.py

# Run the app
streamlit run app.py

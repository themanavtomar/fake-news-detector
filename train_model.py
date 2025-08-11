import pandas as pd
import numpy as np
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib

def clean_text(s):
    if not isinstance(s, str):
        return ""
    s = s.lower()
    s = re.sub(r"http\S+"," ",s)
    s = re.sub(r"[^a-z0-9\s]"," ",s)
    s = re.sub(r"\s+"," ",s).strip()
    return s

df = pd.read_csv("fake_or_real_news.csv")  # download from Kaggle (see README)
if 'text' in df.columns:
    df['content'] = df['title'].fillna("") + " " + df['text'].fillna("")
elif 'content' in df.columns:
    df['content'] = df['content']
elif 'article' in df.columns:
    df['content'] = df['article']
else:
    # try using first text-like column
    text_cols = [c for c in df.columns if df[c].dtype == object]
    df['content'] = df[text_cols[0]].fillna("")

df['content'] = df['content'].apply(clean_text)

if 'label' in df.columns:
    df['label_clean'] = df['label'].astype(str).str.lower()
else:
    if 'class' in df.columns:
        df['label_clean'] = df['class'].astype(str).str.lower()
    else:
        raise Exception("No label column found. Rename your label column to 'label'.")

df = df[df['label_clean'].isin(['fake','real','true','fake news','real news','true'])]
df['y'] = df['label_clean'].apply(lambda x: 1 if ('fake' in x) else 0)

X = df['content'].values
y = df['y'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42, stratify=y)

vec = TfidfVectorizer(max_features=40000, ngram_range=(1,2))
X_train_t = vec.fit_transform(X_train)
X_test_t = vec.transform(X_test)

model = LogisticRegression(max_iter=1000, class_weight='balanced', solver='liblinear')
model.fit(X_train_t, y_train)

pred = model.predict(X_test_t)
print("Accuracy:", accuracy_score(y_test, pred))
print(classification_report(y_test, pred))

joblib.dump(model, "model.joblib")
joblib.dump(vec, "vectorizer.joblib")
print("Saved model.joblib and vectorizer.joblib")
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
import joblib

# 1. Dataset load
df_fake = pd.read_csv("Fake.csv")
df_true = pd.read_csv("True.csv")

df_fake["label"] = 1   # Fake = 1
df_true["label"] = 0   # Real = 0

df = pd.concat([df_fake, df_true], axis=0).sample(frac=1, random_state=42)

# 2. Text column combine
df["text"] = df["title"] + " " + df["text"]

# 3. Train-test split
X_train, X_test, y_train, y_test = train_test_split(df["text"], df["label"], test_size=0.2, random_state=42)

# 4. Vectorizer
vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# 5. Model with balanced class weight
model = LogisticRegression(max_iter=1000, class_weight='balanced')
model.fit(X_train_vec, y_train)

# 6. Evaluation
y_pred = model.predict(X_test_vec)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# 7. Save model and vectorizer
joblib.dump(model, "model.joblib")
joblib.dump(vectorizer, "vectorizer.joblib")
print("✅ Model & vectorizer saved.")
